import torch
import torch.nn as nn
from Pretraining.modules.Transformer import TransformerLayer
import numpy as np
from .Initialization import weight_init2
from Pretraining.PretrainMs import OurMethod
from Pretraining.modules import DownstreamEmbed
from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix
import scipy.sparse as sp
import pickle

def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return graph

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx  # D^-1 W


class FlashBackp(nn.Module):
    def __init__(self, hidden, sequence_len, embed_layer,
                 location_size, user_size,path_loc_graph,path_usr_graph):
        super().__init__()

        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.hidden = hidden
        self.sequence_len = sequence_len

        with open(path_loc_graph, 'rb') as f:  # 时间POI graph
            transition_graph = pickle.load(f)  # 在cpu上
        transition_graph = coo_matrix(transition_graph)
        with open(path_usr_graph, 'rb') as f:  # 空间POI graph
            interact_graph = pickle.load(f)  # 在cpu上
        interact_graph = csr_matrix(interact_graph)

        self.lambda_t = 0.1
        self.lambda_s = 1000

        self.lambda_loc = 1.0
        self.lambda_user = 1.0
        self.I = identity(transition_graph.shape[0], format='coo') #TODO:
        self.graph = sparse_matrix_to_tensor(
            calculate_random_walk_matrix((transition_graph * self.lambda_loc + self.I).astype(np.float32)))
        self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
            interact_graph))  # (M, N)

        ##
        self.location_size = location_size
        self.user_size = user_size
        self.hidden_size = hidden #TODO: Flashback defualt dim size
        self.encoder = nn.Embedding(location_size, self.hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_size, self.hidden_size)  # user embedding
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size, batch_first=True)
        self.fc1 = nn.Linear(2 * self.hidden_size, location_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, location_size)
        self.drop_layer = nn.Dropout(0.4)

        self.embed_layer = embed_layer
        self.apply(weight_init2)

    def f_t(self, delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
        -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay
    # exp decay  2个functions
    def f_s(self, delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))

    def forward(self, **kwargs):
        Inp_POI = kwargs['full_seq']
        Inp_POI_map = kwargs['full_seq_map']
        HistoryLen = kwargs['length']
        timeD = kwargs['time_delta']
        geoD = kwargs['geo_delta']
        uid = kwargs['user_id']

        user_len, seq_len = Inp_POI.size()
        p_u = self.user_encoder(uid)  # (user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)

        # AX,即GCN
        graph = self.graph.to(Inp_POI.device)
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.location_size))).to(Inp_POI.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(Inp_POI.device)  # (input_size, hidden_size)

        x_emb = encoder_weight[Inp_POI_map]
        # user-poi
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.location_size))).to(Inp_POI.device))
        encoder_weight = loc_emb
        interact_graph = self.interact_graph.to(Inp_POI.device)
        encoder_weight_user = torch.sparse.mm(
            interact_graph, encoder_weight).to(Inp_POI.device)

        user_preference = encoder_weight_user[uid].unsqueeze(1)

        # print(user_preference.size())
        user_loc_similarity = torch.exp(
            -(torch.norm(user_preference - x_emb, p=2, dim=-1))).to(Inp_POI.device)
        # user_loc_similarity = user_loc_similarity.permute(1, 0)

        ###init
        hs = []
        for i in range(user_len):
            mu = 0
            sd = 1 / self.hidden_size
            self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
            hs.append(self.h0)
        h = torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(Inp_POI.device)

        Inp_POI_emb = self.drop_layer(self.embed_layer(**kwargs))
        out, h = self.rnn(x_emb, h)  # (user_len, seq_len, hidden_size)
        a_j = self.f_t(timeD, user_len)  # (user_len, )
        b_j = self.f_s(geoD, user_len)
        w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
        w_j = w_j * user_loc_similarity
        compare = torch.tensor(torch.arange(seq_len)).unsqueeze(0).to(Inp_POI.device)
        mask = (HistoryLen.unsqueeze(-1) > compare).float()
        w_j = w_j * mask
        out_w = (out) * w_j.unsqueeze(-1)
        out_w = torch.sum(out_w, dim = 1)/torch.sum(w_j, dim = 1, keepdim=True)
        out_w2 = (Inp_POI_emb) * w_j.unsqueeze(-1)
        out_w2 = torch.sum(out_w2, dim=1) / torch.sum(w_j, dim=1, keepdim=True)
        out_pu = torch.cat([out_w, p_u, out_w2], dim=1)
        y_linear = self.fc1(out_pu)  # (seq_len, user_len, loc_count)
        return y_linear


