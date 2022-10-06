import torch
import torch.nn as nn
from Pretraining.PretrainMs import OurMethod
import torch.nn.functional as F
import numpy as np

class LossModel(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, model: OurMethod, loss_type):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.model = model
        self.loss_type = loss_type
        if self.loss_type == 'Pairwise':
            self.loss = self.loss_pairwise
        if self.loss_type == 'Classification':
            self.loss = nn.CrossEntropyLoss()


    def forward(self, **kwargs):
        Prediction = self.model(**kwargs)
        return Prediction


    def loss_pairwise(self, Prediction):
        Pos_Pred, Neg_Pred = Prediction[:, 0], Prediction[:, 1:]
        Neg_Softmax = (Neg_Pred - Neg_Pred.max()).softmax(dim=1)
        Neg_Pred = (Neg_Pred * Neg_Softmax).sum(dim=1)
        Loss = F.softplus(-(Pos_Pred - Neg_Pred))
        # ↑ For numerical stability, we use 'softplus(-x)' instead of '-log_sigmoid(x)'
        return Loss




