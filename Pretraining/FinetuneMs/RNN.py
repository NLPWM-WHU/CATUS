from abc import ABC

import torch
from torch import nn
from .utils import rnn_forward
from .Initialization import weight_init1


class RNNM(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, rnn_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = nn.RNN(input_size, rnn_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.RNN(input_size, rnn_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(rnn_hidden_size, fc_hidden_size),
                                        nn.LeakyReLU(), nn.Linear(fc_hidden_size, output_size))
        self.sos = nn.Parameter(torch.zeros(input_size).float(), requires_grad=True)
        self.embed_layer = embed_layer
        self.apply(weight_init1)

    def forward(self, **kwargs):
        valid_len = kwargs['length']
        full_embed = self.embed_layer(**kwargs)
        rnn_out_pre = rnn_forward(self.encoder, self.sos, full_embed, valid_len)

        rnn_out_pre = self.out_linear(rnn_out_pre)
        return rnn_out_pre