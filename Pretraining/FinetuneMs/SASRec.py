import torch
import torch.nn as nn
from Pretraining.modules.Transformer import TransformerLayer
import numpy as np
from .Initialization import weight_init2
from Pretraining.PretrainMs import OurMethod
from Pretraining.modules import DownstreamEmbed

class SASRec(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden, sequence_len, output_size, embed_layer):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()

        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.hidden = hidden
        self.sequence_len = sequence_len
        self.attn_heads = 2
        self.n_layers = 2
        self.len_range = torch.from_numpy(np.arange(self.sequence_len)).to(self.device)
        self.transformer_blocks_poi = nn.ModuleList(
            [TransformerLayer(d_model=hidden, d_ff=hidden, n_heads=self.attn_heads,
                              dropout=0.0, kq_same=False)
             for _ in range(self.n_layers)])

        self.embed_layer = embed_layer
        # self.linear_loss = nn.Linear(hidden, output_size)
        self.apply(weight_init2)


    def forward(self, **kwargs):
        Inp_POI = kwargs['full_seq']
        PosNeg_POI = kwargs['posneg']
        HistoryLen = kwargs['length']

        batch_size, seq_len = Inp_POI.shape
        valid_his = (Inp_POI > 0).long()
        Position = (HistoryLen[:, None] - self.len_range[None, :seq_len]) * valid_his
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)

        Inp_POI_emb = self.embed_layer(**kwargs)
        for i, transformer in enumerate(self.transformer_blocks_poi):
            Inp_POI_emb = transformer.forward(Inp_POI_emb,Inp_POI_emb,Inp_POI_emb, attn_mask)
        Inp_POI_emb = (Inp_POI_emb) * valid_his[:, :, None].float()
        Out_POI_hidden = (Inp_POI_emb * (Position == 1).float()[:, :, None]).sum(1)

        if isinstance(self.embed_layer, OurMethod):
            PosNeg_POI_emb = self.embed_layer.POIembedding.embed(PosNeg_POI)
        elif isinstance(self.embed_layer, DownstreamEmbed):
            PosNeg_POI_emb = self.embed_layer.embed(PosNeg_POI)

        Prediction_POI = (Out_POI_hidden[:, None, :] * (PosNeg_POI_emb)).sum(-1)
        return Prediction_POI


