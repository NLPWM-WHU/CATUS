from .layers import *
import torch.nn
from .Initialization import weight_init2
from Pretraining.PretrainMs import OurMethod
from Pretraining.modules import DownstreamEmbed
class STAN(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, dropout, embed_layer):
        super(STAN, self).__init__()
        self.emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        # self.emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        self.emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        self.emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        self.emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        self.emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        self.emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.su, self.sl, self.tu, self.tl = ex
        self.embed_layer = embed_layer
        self.apply(weight_init2)


    def forward(self, **kwargs):
        traj, mat1, mat2, vec, traj_len = kwargs['full_seq'], kwargs['mat1'], \
                                          kwargs['mat2'], kwargs['vec'], kwargs['traj_len']
        user = kwargs['user']
        tim = kwargs['time_seq']
        candidates = kwargs['posneg']
        kwargs['length'] =  kwargs['traj_len']
        ####MultiEmbed
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        tim = (tim - 1) % hours + 1  # segment time by 24 hours * 7 days
        time = self.emb_t(tim)  # (N, M) --> (N, M, embed)
        loc = self.embed_layer(**kwargs)  # (N, M) --> (N, M, embed)
        user = self.emb_u(user).unsqueeze(1)  # (N, 1) --> (N, 1, embed)
        joint = time + loc + user  # (N, M, embed)
        delta_s, delta_t = mat1[:, :, :, 0], mat1[:, :, :, 1]  # (N, M, M)
        maxlen = traj.shape[-1]
        compare = torch.tensor(torch.arange(maxlen)).unsqueeze(0).to(device)
        mask = (traj_len.unsqueeze(-1) > compare).long()
        mask = mask.unsqueeze(1) * mask.unsqueeze(-1).repeat(1, 1, maxlen)
        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1), \
                             (self.su - delta_s).unsqueeze(-1), \
                             (delta_t - self.tl).unsqueeze(-1), \
                             (self.tu - delta_t).unsqueeze(-1)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)


        #### SelfAttn
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1


        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)
        self_attn = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        #### Embed
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
        self_delta = space_interval + time_interval  # (N, M, L, emb)


        #### Attn
        # self_attn (N, M, emb), candidate (N, L, emb), self_delta (N, M, L, emb), len [N]
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  # squeeze the embed dimension
        [N, L, M] = self_delta.shape

        if isinstance(self.embed_layer, OurMethod):
            emb_candidates = self.embed_layer.POIembedding.embed(candidates)
        elif isinstance(self.embed_layer, DownstreamEmbed):
            emb_candidates = self.embed_layer.embed(candidates)

        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)  # (N, L, M)
        # pdb.set_trace()
        output = torch.sum(attn, -1).view(N, L)  # (N, L)
        return output

