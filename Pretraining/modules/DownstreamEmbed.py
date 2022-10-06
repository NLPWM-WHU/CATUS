import torch.nn as nn


class DownstreamEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_size = embed_size

    def forward(self, **kwargs):
        token = kwargs["full_seq"]
        return self.embed(token)
