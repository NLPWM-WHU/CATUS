import torch

def rnn_forward(encoder, sos, lstm_input, valid_len):
    batch_size = lstm_input.size(0)
    input_size = lstm_input.size(-1)
    history_len = valid_len
    max_len = history_len.max()

    lstm_input = torch.cat([sos.reshape(1, 1, -1).repeat(batch_size, 1, 1), lstm_input], dim=1)  # (batch, seq_len+1, 1+input_size)
    lstm_out, _ = encoder(lstm_input)
    lstm_out_pre = lstm_out[torch.arange(batch_size), history_len, :]
    # lstm_out_pre = torch.stack([lstm_out[i, s - pre_len:s] for i, s in enumerate(valid_len)])  # (batch, pre_len, lstm_hidden_size)
    return lstm_out_pre