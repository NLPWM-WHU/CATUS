from abc import ABC
import torch
from torch import nn
from .Initialization import weight_init1
from .utils import rnn_forward





class STRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_slots, inter_size):
        super().__init__()
        self.__dict__.update(locals())

        self.time_weights = nn.Parameter(torch.zeros(num_slots+1, input_size, hidden_size), requires_grad=True)
        self.dist_weights = nn.Parameter(torch.zeros(num_slots+1, input_size, hidden_size), requires_grad=True)
        self.hidden_weights = nn.Parameter(torch.zeros(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.hidden_weights.data)
        for i in range(num_slots + 1):
            nn.init.xavier_normal_(self.time_weights.data[i])
            nn.init.xavier_normal_(self.dist_weights.data[i])

    def forward(self, x_context, time_context, dist_context, context_mask, h):
        time_weight = self.time_weights[time_context, :, :]  # (batch, context_size, input_size, inter_size)
        dist_weight = self.dist_weights[dist_context, :, :]  # (batch, context_size, inter_size, hidden_size)
        x_candidate1 = torch.matmul(x_context.unsqueeze(-2), dist_weight).squeeze(-2)
        x_candidate2 = torch.matmul(x_context.unsqueeze(-2), time_weight).squeeze(-2)
        x_candidate = 0.5 * x_candidate1 + 0.5 * x_candidate2
        x_candidate = x_candidate.masked_fill(context_mask.unsqueeze(-1) == False, 0.0).sum(1)  # (batch, hidden_size)
        h_candidate = torch.matmul(h.unsqueeze(-2), self.hidden_weights).squeeze(1)
        return torch.sigmoid(x_candidate + h_candidate)


class STRNN(nn.Module):
    def __init__(self, input_size, hidden_size, inter_size, num_slots):
        super().__init__()
        self.__dict__.update(locals())

        self.strnn_cell = STRNNCell(input_size, hidden_size, num_slots, inter_size)

    def forward(self, x_contexts, time_contexts, dist_contexts, context_masks):
        batch_size = x_contexts.size(0)
        seq_len = x_contexts.size(1)

        hidden_state = torch.zeros(batch_size, self.hidden_size).to(x_contexts.device)
        output = []
        for i in range(seq_len):
            x_content = x_contexts[:, :i+1]  # (batch_size, context_size, input_size)
            time_context = time_contexts[:, i, :i+1]
            dist_context = dist_contexts[:, i, :i+1]
            context_mask = context_masks[:, i, :i+1]  # (batch_size, context_size)
            hidden_state = self.strnn_cell(x_content, time_context, dist_context, context_mask, hidden_state)
            output.append(hidden_state)
        return torch.stack(output, dim=1), hidden_state.unsqueeze(0)


class STRNNM(nn.Module):
    def __init__(self, embed_layer, num_slots, time_window, dist_window,
                 input_size, hidden_size, inter_size, output_size):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = STRNN(input_size, hidden_size, inter_size, num_slots)
        self.decoder = STRNN(input_size, hidden_size, inter_size, num_slots)
        # self.out_linear = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size))
        self.out_linear = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size * 4),
                                        nn.Tanh(), nn.Linear(hidden_size * 4, output_size))
        self.embed_layer = embed_layer
        # self.add_module('embed_layer', self.embed_layer)
        self.apply(weight_init1)

    def forward(self, **kwargs):
        full_seq = kwargs['full_seq']
        batch_size = full_seq.size(0)
        # history_len = valid_len - pre_len
        valid_len = kwargs['length']
        history_len = valid_len
        max_len = history_len.max()

        # Generate input sequence.
        full_embed = self.embed_layer(**kwargs)  # (batch, seq_len, embed_size)
        timestamp = kwargs['timestamp']  # (batch, seq_len)
        lat, lng = kwargs['lat'], kwargs['lng']  # (batch, seq_len)
        cat_input = torch.cat([full_embed, timestamp.unsqueeze(-1),
                               lat.unsqueeze(-1), lng.unsqueeze(-1)], dim=-1)  # (batch, seq_len, input_size + 3)
        sequential_input = cat_input
        seq_len = sequential_input.size(1)

        # Calculate a context mask from a given time window.
        seq_timestamp = sequential_input[:, :, -3]  # (batch, seq_len)
        time_delta = seq_timestamp.unsqueeze(-1) - seq_timestamp.unsqueeze(1)
        context_mask = (time_delta <= self.time_window) * \
                       (time_delta >= 0) * \
                       (valid_len.unsqueeze(-1) > torch.arange(seq_len).to(full_seq.device).unsqueeze(0).repeat(batch_size, 1)).unsqueeze(1)  # (batch, seq_len, seq_len)

        # Calculate distances between locations in the trajectory.
        seq_latlng = sequential_input[:, :, -2:]  # (batch, seq_len, 2)
        # (batch, seq_len, 1, 2) - # (batch, 1, seq_len, 2) -> # (batch, seq_len, seq_len, 2)
        dist = (seq_latlng.unsqueeze(2) - seq_latlng.unsqueeze(1)) ** 2
        dist = torch.sqrt(dist.sum(-1))  # (batch, seq_len, seq_len)

        rnn_out, _ = self.encoder(sequential_input[:, :, :-3],
                                  torch.floor(torch.clamp(time_delta, 0, self.time_window) /
                                              self.time_window * self.num_slots).long(),
                                  torch.floor(torch.clamp(dist, 0, self.dist_window) /
                                              self.dist_window * self.num_slots).long(),
                                  context_mask)

        rnn_out_pre = rnn_out[torch.arange(batch_size), history_len - 1, :]
        rnn_out_pre = self.out_linear(rnn_out_pre)
        return rnn_out_pre