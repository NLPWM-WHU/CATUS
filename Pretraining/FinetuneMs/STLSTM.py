from abc import ABC

import torch
from torch import nn
from .utils import rnn_forward
from .Initialization import weight_init1


class STLSTMM(nn.Module, ABC):
    def __init__(self, embed_layer, num_slots, aux_embed_size, time_thres, dist_thres,
                 input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.time_embed = nn.Parameter(torch.zeros(num_slots + 1, aux_embed_size), requires_grad=True)
        self.dist_embed = nn.Parameter(torch.zeros(num_slots + 1, aux_embed_size), requires_grad=True)

        self.encoder = nn.LSTM(input_size + 2 * aux_embed_size, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(lstm_hidden_size, output_size))
        self.sos = nn.Parameter(torch.zeros(input_size + 2 * aux_embed_size).float(), requires_grad=True)
        self.aux_sos = nn.Parameter(torch.zeros(aux_embed_size * 2).float(), requires_grad=True)

        self.embed_layer = embed_layer
        # self.add_module('embed_layer', self.embed_layer)
        self.apply(weight_init1)
        self.drop_out = torch.nn.Dropout(0.1)



    def forward(self, **kwargs):
        full_seq = kwargs['full_seq']
        batch_size = full_seq.size(0)
        # his_len = valid_len - pre_len
        valid_len = kwargs['length']
        time_delta = kwargs['time_delta'][:, 1:]
        dist = kwargs['dist'][:, 1:]

        time_slot_i = torch.floor(torch.clamp(time_delta, 0, self.time_thres) / self.time_thres * self.num_slots).long()
        dist_slot_i = torch.floor(torch.clamp(dist, 0, self.dist_thres) / self.dist_thres * self.num_slots).long()  # (batch, seq_len-1)
        aux_input = torch.cat([self.aux_sos.reshape(1, 1, -1).repeat(batch_size, 1, 1),
                               torch.cat([self.time_embed[time_slot_i],
                                          self.dist_embed[dist_slot_i]], dim=-1)], dim=1)  # (batch, seq_len, aux_embed_size*2)

        full_embed = self.drop_out(self.embed_layer(**kwargs))  # (batch_size, seq_len, input_size)

        lstm_input = torch.cat([full_embed, aux_input], dim=-1)  # (batch_size, seq_len, input_size + aux_embed_size * 2)
        # lstm_input = full_embed
        lstm_out_pre = rnn_forward(self.encoder, self.sos, lstm_input, valid_len)
        lstm_out_pre = self.out_linear(lstm_out_pre)
        return lstm_out_pre



