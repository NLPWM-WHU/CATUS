import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import List, Dict, NoReturn, Any
import numpy as np
import os
from Pretraining.LossMs.PretrainLossM import LossModel
from Pretraining.PretrainMs import OurMethod
from Pretraining.trainer.optimizer.adamw import AdamW

class ModelTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, pretrainm: OurMethod, loss_type, lambda1,
                 train_dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.pretrainm = pretrainm
        self.loss_type = loss_type
        self.model = LossModel(self.pretrainm, loss_type).to(self.device)

        # Setting the train and test data loader
        self.test_data = train_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.masked_criterion = nn.NLLLoss(ignore_index=0)
        self.segment_shuffle = nn.NLLLoss(ignore_index=0)
        self.next_criterion = nn.NLLLoss()
        self.lambda1 = lambda1

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        logging.info("Total Parameters:%d"%sum([p.nelement() for p in self.model.parameters()]))

    def test(self, epoch):
        result = self.iteration_test(epoch, self.test_data)
        return result

    def iteration_test(self, epoch, data_loader):
        str_code = "test"
        self.model.eval()
        data_loader.dataset.current_epoch = epoch
        pois = list()
        predictions = list()
        predictions_score = list()
        # data_len = len(data_loader)
        for i, data in enumerate(data_loader):
            if i%10==0:
                print(i)
            data = {key: value.to(self.device).long() for key, value in data.items()}
            Target_POI_emb = self.model.model.POIembedding.embed(data["Inp_POI"])
            Candidate_POI_emb = self.model.model.POIembedding.embed.weight.data
            prediction_score = (Target_POI_emb[:, None, :] * (Candidate_POI_emb[None, :, :])).sum(-1)
            prediction_score = np.array(prediction_score.cpu().data.numpy())
            prediction = (-prediction_score).argsort(axis=1)[:,:20]
            pois.extend(data["Inp_POI"].cpu().data.numpy())
            predictions.extend(prediction)
            predictions_score.extend(np.take_along_axis(prediction_score, prediction, axis=1))
        pois = np.array(pois)
        predictions = np.array(predictions)
        predictions_score = np.array(predictions_score)
        return (pois, predictions, predictions_score)

    def save(self, epoch, file_path="output/", model_name = "PretrainModel_CAT_CAT_POICATnearest_POI_POI.pkl"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        path, _ = os.path.split(file_path)
        if not os.path.exists(path):
            os.mkdir(path)
        # output_path = file_path + "_encdec.ep%d" % epoch
        output_path = file_path + model_name + ".ep%d" % epoch
        # torch.save(self.bert.cpu(), output_path)
        if torch.__version__ == '1.1.0':
            torch.save(self.model.model.cpu(), output_path)
        else:
            torch.save(self.model.model.cpu(), output_path, _use_new_zipfile_serialization=False)
        self.model.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        logging.info("EP:%d Model Saved on:" % epoch + output_path)
        return output_path

    def format_metric(self, result_dict):
        assert type(result_dict) == dict
        format_str = []
        for name in np.sort(list(result_dict.keys())):
            m = result_dict[name]
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('{}:{:<.4f}'.format(name, m))
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('{}:{}'.format(name, m))
        return ','.join(format_str)