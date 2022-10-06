
import torch
import torch.nn as nn
from Pretraining.modules.DownstreamEmbed import DownstreamEmbed
from Pretraining.modules.Transformer import TransformerLayer
import numpy as np
class OurMethod(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, category_size, cluster_size, hidden, sequence_len, dropout=0.0, loss_type = 'Pairwise'):
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
        self.loss_type = loss_type
        self.hidden = hidden
        self.sequence_len = sequence_len
        self.attn_heads = 2
        self.n_layers = 2
        self.MAX_Delta_Day = 30
        self.MAX_Delta_Dis = 50
        self.LinearPOI1 = nn.Linear(hidden, hidden)
        self.LinearPOI_TIM_DD = nn.Linear(3 * hidden, hidden)
        self.LinearPOI_TIM = nn.Linear(2 * hidden, hidden)
        self.LinearPOI_DD = nn.Linear(2 * hidden, hidden)
        self.LinearPOI2 = nn.Linear(hidden, hidden)
        self.item_size = vocab_size
        self.category_size = category_size
        self.cluster_size = cluster_size
        # embedding for BERT, sum of positional, segment, token embeddings
        self.POIembedding = DownstreamEmbed(vocab_size=vocab_size, embed_size=hidden)
        self.CATembedding1 = DownstreamEmbed(vocab_size=category_size, embed_size=hidden)
        self.CATembedding2 = DownstreamEmbed(vocab_size=category_size, embed_size=hidden)

        self.DISembedding = DownstreamEmbed(vocab_size=self.MAX_Delta_Dis + 1, embed_size=hidden)
        self.TIMembedding = DownstreamEmbed(vocab_size=48, embed_size=hidden)
        self.CLUembedding = DownstreamEmbed(vocab_size=cluster_size, embed_size=hidden)
        self.POSembedding = DownstreamEmbed(vocab_size=sequence_len + 1, embed_size=hidden)

        self.len_range = torch.from_numpy(np.arange(self.sequence_len)).to(self.device)
        self.transformer_blocks_pretrain = nn.ModuleList(
            [TransformerLayer(d_model=hidden, d_ff=hidden, n_heads=self.attn_heads,
                                    dropout=dropout, kq_same=False)
             for _ in range(self.n_layers)])


        self.dropout = nn.Dropout(p=dropout)

        if self.loss_type == 'Classification':
            self.classification_POI = nn.Linear(hidden, vocab_size)
            self.classification_CAT = nn.Linear(hidden, category_size)

    def finetune_func(self, ** kwargs):
        HistoryLen = kwargs['length']
        Inp_POI = kwargs['full_seq']
        Inp_TIM = kwargs['time_seq']

        batch_size, seq_len = Inp_POI.shape
        valid_his = (Inp_POI > 0).long()
        Position = (HistoryLen[:, None] - self.len_range[None, :seq_len]) * valid_his
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)

        Inp_POI_emb = self.POIembedding.embed(Inp_POI)
        Inp_Pos_emb = self.POSembedding.embed(Position)


        Inp_POI_emb = Inp_POI_emb + Inp_Pos_emb
        for i, transformer in enumerate(self.transformer_blocks_pretrain):
            Inp_POI_emb = transformer.forward(Inp_POI_emb,Inp_POI_emb,Inp_POI_emb, attn_mask)
        return Inp_POI_emb

    def pretrain_func(self, ** kwargs):
        Method = kwargs['Method']
        if Method == 'Cat':
            Inp_CAT = kwargs['Inp_CAT']
            PosNeg_CAT = kwargs['PosNeg_CAT']
            HistoryLen = kwargs['HistoryLen']
            Inp_TIM = kwargs['Inp_TIM']
            Inp_City = kwargs['City']

            Inp_TIM_emb = self.TIMembedding.embed(Inp_TIM)
            batch_size, seq_len = Inp_CAT.shape
            valid_his = (Inp_CAT > 0).long()
            Position = (HistoryLen[:, None] - self.len_range[None, :seq_len]) * valid_his
            Inp_Pos_emb = self.POSembedding.embed(Position)
            Inp_CAT_emb1 = self.CATembedding1.embed(Inp_CAT)
            Inp_CAT_emb2 = self.CATembedding2.embed(Inp_CAT)
            Inp_CAT_emb = torch.where(Inp_City.unsqueeze(1).unsqueeze(1).repeat(1,self.sequence_len, self.hidden)==0,
                                      Inp_CAT_emb1, Inp_CAT_emb2)
            Inp_CAT_emb = Inp_CAT_emb + Inp_Pos_emb + Inp_TIM_emb
            causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
            attn_mask = torch.from_numpy(causality_mask).to(self.device)
            for i, transformer in enumerate(self.transformer_blocks_pretrain):
                Inp_CAT_emb = transformer.forward(Inp_CAT_emb,Inp_CAT_emb,Inp_CAT_emb, attn_mask)
            Inp_CAT_emb = Inp_CAT_emb * valid_his[:, :, None].float()
            Inp_CAT_emb = (Inp_CAT_emb * (Position == 1).float()[:, :, None]).sum(1)
            Out_CAT_hidden = Inp_CAT_emb


            PosNeg_CAT_emb1 = self.CATembedding1.embed(PosNeg_CAT)
            PosNeg_CAT_emb2 = self.CATembedding2.embed(PosNeg_CAT)

            PosNeg_CAT_emb = torch.where(Inp_City.unsqueeze(1).unsqueeze(1).repeat(1, 2, self.hidden)==0,
                                         PosNeg_CAT_emb1, PosNeg_CAT_emb2)

            Prediction_CAT = ((Out_CAT_hidden[:, None, :]) * PosNeg_CAT_emb).sum(-1)
            return Prediction_CAT

        elif Method == 'POI':
            Inp_POI = kwargs['Inp_POI']
            PosNeg_POI = kwargs['PosNeg_POI']
            HistoryLen = kwargs['HistoryLen']
            Inp_TIM = kwargs['Inp_TIM']
            batch_size, seq_len = Inp_POI.shape
            Inp_POI_emb = self.POIembedding.embed(Inp_POI)
            Inp_TIM_emb = self.TIMembedding.embed(Inp_TIM)
            valid_his = (Inp_POI > 0).long()
            Position = (HistoryLen[:, None] - self.len_range[None, :seq_len]) * valid_his
            Inp_Pos_emb = self.POSembedding.embed(Position)
            Inp_POI_emb = Inp_POI_emb + Inp_Pos_emb + Inp_TIM_emb
            causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
            attn_mask = torch.from_numpy(causality_mask).to(self.device)
            for i, transformer in enumerate(self.transformer_blocks_pretrain):
                Inp_POI_emb = transformer.forward(Inp_POI_emb,Inp_POI_emb,Inp_POI_emb, attn_mask)
            Inp_POI_emb = Inp_POI_emb * valid_his[:, :, None].float()
            Inp_POI_emb = (Inp_POI_emb * (Position == 1).float()[:, :, None]).sum(1)
            Out_POI_hidden = Inp_POI_emb

            PosNeg_POI_emb = self.POIembedding.embed(PosNeg_POI)
            Prediction_POI = ((Out_POI_hidden[:, None, :]) * PosNeg_POI_emb).sum(-1)

            return Prediction_POI
        elif Method == 'POICAT':
            Inp_POI = kwargs['Inp_POI']
            Inp_CAT = kwargs['Inp_CAT']
            PosNeg_POI = kwargs['PosNeg_POI']
            HistoryLen = kwargs['HistoryLen']
            Inp_TIM = kwargs['Inp_TIM']
            batch_size, seq_len = Inp_POI.shape
            Inp_POI_emb = self.POIembedding.embed(Inp_POI)
            Inp_CAT_emb1 = self.CATembedding1.embed(Inp_CAT)
            Inp_CAT_emb2 = self.CATembedding2.embed(Inp_CAT)


            Inp_TIM_emb = self.TIMembedding.embed(Inp_TIM)
            valid_his = (Inp_POI > 0).long()
            Position = (HistoryLen[:, None] - self.len_range[None, :seq_len]) * valid_his
            Inp_Pos_emb = self.POSembedding.embed(Position)
            Inp_POI_emb1 = Inp_POI_emb + Inp_Pos_emb + Inp_TIM_emb
            Inp_POI_emb2 = Inp_POI_emb + Inp_Pos_emb + Inp_TIM_emb
            Inp_CAT_emb1 = Inp_CAT_emb1 + Inp_Pos_emb + Inp_TIM_emb
            Inp_CAT_emb2 = Inp_CAT_emb2 + Inp_Pos_emb + Inp_TIM_emb

            causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
            attn_mask = torch.from_numpy(causality_mask).to(self.device)
            for i, transformer in enumerate(self.transformer_blocks_pretrain):
                Inp_POI_emb1 = transformer.forward(Inp_CAT_emb1,Inp_CAT_emb1,Inp_POI_emb1, attn_mask)
            Inp_POI_emb1 = Inp_POI_emb1 * valid_his[:, :, None].float()
            Inp_POI_emb1 = (Inp_POI_emb1 * (Position == 1).float()[:, :, None]).sum(1)
            Out_POI_hidden1 = Inp_POI_emb1

            for i, transformer in enumerate(self.transformer_blocks_pretrain):
                Inp_POI_emb2 = transformer.forward(Inp_CAT_emb2,Inp_CAT_emb2,Inp_POI_emb2, attn_mask)
            Inp_POI_emb2 = Inp_POI_emb2 * valid_his[:, :, None].float()
            Inp_POI_emb2 = (Inp_POI_emb2 * (Position == 1).float()[:, :, None]).sum(1)
            Out_POI_hidden2 = Inp_POI_emb2

            PosNeg_POI_emb = self.POIembedding.embed(PosNeg_POI)

            Prediction_POI1 = ((Out_POI_hidden1[:, None, :]) * PosNeg_POI_emb).sum(-1)
            Prediction_POI2 = ((Out_POI_hidden2[:, None, :]) * PosNeg_POI_emb).sum(-1)
            return Prediction_POI1, Prediction_POI2

    def forward(self, **kwargs):
        downstream = kwargs['downstream']
        if downstream:
            return self.finetune_func(**kwargs)
        else:
            return self.pretrain_func(**kwargs)
