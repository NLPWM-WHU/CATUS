import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
seed = 0
global_seed = 0
hours = 24*7
torch.manual_seed(seed)
device = 'cuda'


def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


class Attn(nn.Module):
    def __init__(self, emb_loc, loc_max, dropout=0.1):
        super(Attn, self).__init__()
        self.value = nn.Linear(20, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max

    def forward(self, self_attn, self_delta, traj_len, candidates):
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  # squeeze the embed dimension
        [N, L, M] = self_delta.shape
        emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)  # (N, L, M)
        attn_out = torch.sum(attn, -1).view(N, L)  # (N, L)
        return attn_out  # (N, L)


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)

        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)
        #
        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)


        return attn_out  # (N, M, emb)


class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super(Embed, self).__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max

    def forward(self, traj_loc, mat2, vec, traj_len):
        loc_len, maxlen = mat2.shape[-1], mat2.shape[1]
        delta_t = vec.unsqueeze(-1)

        compare = torch.tensor(torch.arange(maxlen)).unsqueeze(0).to(device)
        mask = (traj_len.unsqueeze(-1) > compare).long()
        mask = mask.unsqueeze(-1)
        delta_s = mat2

        # esl, esu, etl, etu = mask, mask, mask, mask
        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1), \
                             (self.su - delta_s).unsqueeze(-1), \
                             (delta_t - self.tl).unsqueeze(-1), \
                             (self.tu - delta_t).unsqueeze(-1)
        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, L, emb)

        return delta

class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, \
        self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self,user,tim, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        tim = (tim-1) % hours + 1  # segment time by 24 hours * 7 days
        time = self.emb_t(tim)  # (N, M) --> (N, M, embed)
        loc = self.emb_l(traj)  # (N, M) --> (N, M, embed)
        user = self.emb_u(user).unsqueeze(1)  # (N, 1) --> (N, 1, embed)
        joint = time + loc + user  # (N, M, embed)

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]  # (N, M, M)
        # mask = torch.zeros_like(delta_s, dtype=torch.long)
        maxlen = traj.shape[-1]
        compare = torch.tensor(torch.arange(maxlen)).unsqueeze(0).to(device)
        mask = (traj_len.unsqueeze(-1)>compare).long()
        mask = mask.unsqueeze(1) * mask.unsqueeze(-1).repeat(1,1,maxlen)
        # for i in range(mask.shape[0]):
        #     mask[i, 0:traj_len[i], 0:traj_len[i]] = 1
        # esl, esu, etl, etu = mask.unsqueeze(-1), mask.unsqueeze(-1), mask.unsqueeze(-1), mask.unsqueeze(-1)
        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1), \
                             (self.su - delta_s).unsqueeze(-1), \
                             (delta_t - self.tl).unsqueeze(-1), \
                             (self.tu - delta_t).unsqueeze(-1)

        space_interval = (esl*vsu+esu*vsl) / (self.su-self.sl)
        time_interval = (etl*vtu+etu*vtl) / (self.tu-self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)

        return joint, delta
