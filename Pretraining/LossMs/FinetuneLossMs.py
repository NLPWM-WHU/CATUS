import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseFinetuneLossM(nn.Module):
    def __init__(self, model, loss_type, data_type):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.data_type = data_type
        if self.data_type == 'Common':
            self.data_keys = ["Inp_POI", "PosNeg_POI", "HistoryLen", "Inp_TIM", "Inp_TIMSTP", "Inp_CAT"]
            self.correspond_parameter_keys = ["full_seq", "posneg", "length", "time_seq", "timestamp", "category"]
        elif self.data_type == 'Delta':
            self.data_keys = ["Inp_POI", "PosNeg_POI", "HistoryLen", "Inp_TIMSTP", "Inp_TIM", "Inp_DeltaT", "Inp_DeltaD", "Inp_Lat", "Inp_Lon", "Inp_CAT"]
            self.correspond_parameter_keys = ["full_seq", "posneg", "length", "timestamp", "time_seq", "time_delta", "dist","lat", "lng", "category"]
        elif self.data_type == 'Deltav2':
            self.data_keys = ["Inp_POI", "PosNeg_POI", "HistoryLen", "Inp_TIM_Interval", "Inp_DIS_Interval","User_ID" ,"Inp_POI_map", "Inp_TIM", "Inp_TIMSTP"]
            self.correspond_parameter_keys = ["full_seq", "posneg", "length", "time_delta", "geo_delta","user_id", "full_seq_map", "time_seq", "timestamp"]
        elif self.data_type == 'Delta_Matrix':
            self.data_keys = ["Inp_User", "Inp_TIM", "Inp_POI", "PosNeg_POI", "HistoryLen", "Inp_MAT1", "Inp_MAT2t", "Inp_MAT2s", "Inp_TIMSTP", "Inp_CAT"]
            self.correspond_parameter_keys = ["user", "time_seq", "full_seq", "posneg", "traj_len", 'mat1', 'vec', 'mat2', "timestamp", "category"]
        self.Long_data = ["Inp_POI", "HistoryLen", "PosNeg_POI", "Inp_TIM", "Inp_User", "Inp_CAT","User_ID", "Inp_POI_map"]
        self.Float_data = ["Inp_TIMSTP", "Inp_DeltaT", "Inp_DeltaD", "Inp_Lat", "Inp_Lon", "Inp_MAT1", "Inp_MAT2t", "Inp_MAT2s","Inp_TIM_Interval", "Inp_DIS_Interval" ]

        if self.loss_type == 'Pairwise':
            self.loss_func = self.loss_pairwise
        elif self.loss_type == 'Classification':
            self.loss_func = nn.CrossEntropyLoss()


    def forward(self, data_tuple, test = False):
        data_input = dict()
        ###input format
        for i, data_key in enumerate(self.data_keys):
            if data_key in self.Long_data:
                data_input[self.correspond_parameter_keys[i]] = data_tuple[data_key].long()
            if data_key in self.Float_data:
                data_input[self.correspond_parameter_keys[i]] = data_tuple[data_key].float()
        data_input['downstream'] = True

        if test:
            return self.test_forward(data_input)
        else:
            return self.train_forward(data_input)

    def test_forward(self, data_tuple):
        pass

    def train_forward(self, data_tuple):
        pass

    def loss_pairwise(self, Prediction, _):
        Pos_Pred, Neg_Pred = Prediction[:, 0], Prediction[:, 1:]
        Neg_Softmax = (Neg_Pred - Neg_Pred.max()).softmax(dim=1)
        Neg_Pred = (Neg_Pred * Neg_Softmax).sum(dim=1)
        Loss = F.softplus(-(Pos_Pred - Neg_Pred)).mean()
        return Loss


class OurMethodFinetuneLossM(BaseFinetuneLossM):
    def __init__(self, model, loss_type, data_type, lambda2):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__(model, loss_type, data_type)
        self.lambda2 = lambda2

    def forward(self, data_tuple, test = False):
        data_input = dict()
        ###input format
        for i, data_key in enumerate(self.data_keys):
            if data_key in self.Long_data:
                data_input[self.correspond_parameter_keys[i]] = data_tuple[data_key].long()
            if data_key in self.Float_data:
                data_input[self.correspond_parameter_keys[i]] = data_tuple[data_key].float()
        data_input['downstream'] = True
        if test:
            return self.test_forward(data_input)
        else:
            Loss_POI1 = self.train_forward(data_input)
            data_input["full_seq"] = data_tuple["Inp_POI_Neg"].long()
            # data_input["full_seq_map"] = data_tuple["Inp_POI_Neg_map"].long() # for GraphFlashBack method
            Loss_POI2 = self.train_forward(data_input)
            return [(self.lambda2) * Loss_POI1, (1 - self.lambda2) * Loss_POI2]


    def test_forward(self, data_input):
        if self.loss_type == "Pairwise":
            (prediction) = self.model(**data_input).cpu().data.numpy()
        elif self.loss_type == "Classification":
            (prediction) = self.model(**data_input).cpu().data.numpy()
            prediction = np.take_along_axis(prediction, data_input["posneg"].cpu().numpy(), axis=1)
        return prediction


    def train_forward(self, data_input):
        (Predictions) = self.model(**data_input)
        Loss_POI = self.loss_func(Predictions, data_input["posneg"])
        return Loss_POI

class NoneFinetuneLossM(BaseFinetuneLossM):
    def __init__(self, model, loss_type, data_type, lambda2):
        super().__init__(model, loss_type, data_type)

    def forward(self, data_tuple, test = False):
        data_input = dict()
        ###input format
        for i, data_key in enumerate(self.data_keys):
            if data_key in self.Long_data:
                data_input[self.correspond_parameter_keys[i]] = data_tuple[data_key].long()
            if data_key in self.Float_data:
                data_input[self.correspond_parameter_keys[i]] = data_tuple[data_key].float()
        data_input['downstream'] = True
        if test:
            return self.test_forward(data_input)
        else:
            Loss_POI = self.train_forward(data_input)
            return [Loss_POI]


    def test_forward(self, data_input):
        if self.loss_type == "Pairwise":
            (prediction) = self.model(**data_input).cpu().data.numpy()
        elif self.loss_type == "Classification":
            (prediction) = self.model(**data_input).cpu().data.numpy()
            prediction = np.take_along_axis(prediction, data_input["posneg"].cpu().numpy(), axis=1)
        return prediction


    def train_forward(self, data_input):
        (Predictions) = self.model(**data_input)
        Loss_POI = self.loss_func(Predictions, data_input["posneg"])
        return Loss_POI



