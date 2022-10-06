import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import List, Dict, NoReturn, Any
import numpy as np
# from ..model import BERTLM, BERT
# from .optimizer.optim_schedule import ScheduledOptim
# from .optimizer.adamw import AdamW
import os
from Pretraining.LossMs.PretrainLossM import LossModel

from Pretraining.PretrainMs import OurMethod
from Pretraining.trainer.optimizer.adamw import AdamW
import torch.nn.functional as F
class ModelTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, pretrainm: OurMethod, loss_type, lambda1,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
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
        self.train_data = train_dataloader
        self.test_data = test_dataloader

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
    def train(self, epoch):
        result = self.iteration_train(epoch, self.train_data)
        return result
    def test(self, epoch):
        self.iteration_test(epoch, self.test_data)

    def iteration_train(self, epoch, data_loader):
        str_code = 'train'
        self.model.train()
        data_loader.dataset.current_epoch = epoch
        Avg_Loss = 0.0
        ##train category sample
        for i, data in enumerate(data_loader):
            data = {key: value.to(self.device).long() for key, value in data.items()}
            Input_dict = {'Inp_CAT':data["Inp_CAT"],'City':data["City"],'Method':'Cat','PosNeg_CAT':data["PosNeg_CAT"],
                                                'Inp_TIM':data["Inp_TIM"],'HistoryLen':data["HistoryLen"],'downstream':False}
            Prediction_CAT = self.model.forward(**Input_dict)

            if self.loss_type == 'Classification':
                Loss_CAT = self.model.loss(Prediction_CAT, data['PosNeg_CAT'][:,0])
            elif self.loss_type == 'Pairwise':
                Loss_CAT = self.model.loss(Prediction_CAT)
            Loss_CAT = Loss_CAT.mean()
            Loss = Loss_CAT
            self.optim.zero_grad()
            Loss.backward()
            self.optim.step()
            Avg_Loss += Loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": "[%d/%d]" % (i, len(data_loader)),
                "Avg_Loss": Avg_Loss / (i + 1),
                "Loss_CAT": Loss_CAT.item(),
            }
            if i % self.log_freq == 0:
                print(post_fix)
                logging.info(post_fix)

        ##train POI sample
        Avg_Loss = 0.0
        Sum_cate_replace = 0.0
        for i, data in enumerate(data_loader):
            data = {key: value.to(self.device) for key, value in data.items()}

            Input_dict = {'Inp_POI': data["Inp_POI"], 'Method': 'POI', 'PosNeg_POI': data["PosNeg_POI"],
                          'Inp_TIM': data["Inp_TIM"], 'HistoryLen': data["HistoryLen"], 'downstream': False}
            Prediction_POI1 = self.model.forward(**Input_dict)
            Loss_POI1 = self.model.loss(Prediction_POI1)

            Input_dict = {'Inp_POI': data["Inp_POI_Neg_c1"], 'Method': 'POI', 'PosNeg_POI': data["PosNeg_POI"],
                          'Inp_TIM': data["Inp_TIM"], 'HistoryLen': data["HistoryLen"], 'downstream': False}
            Prediction_POI2 = self.model.forward(**Input_dict)
            Loss_POI2 = self.model.loss(Prediction_POI2)

            Input_dict = {'Inp_POI': data["Inp_POI"],'Inp_CAT': data["Inp_CAT"], 'Method': 'POICAT', 'PosNeg_POI': data["PosNeg_POI"],
                          'Inp_TIM': data["Inp_TIM"], 'HistoryLen': data["HistoryLen"], 'downstream': False}
            Prediction_POI31, Prediction_POI32 = self.model.forward(**Input_dict)
            Loss_POI31 = self.model.loss(Prediction_POI31)
            Loss_POI32 = self.model.loss(Prediction_POI32)

            Loss_POI3 = torch.where(Loss_POI31 > Loss_POI32, Loss_POI31, Loss_POI32)

            Input_dict = {'Inp_POI': data["Inp_POI_Neg_c1"],'Inp_CAT': data["Inp_CAT"], 'Method': 'POICAT', 'PosNeg_POI': data["PosNeg_POI"],
                          'Inp_TIM': data["Inp_TIM"], 'HistoryLen': data["HistoryLen"], 'downstream': False}
            Prediction_POI41, Prediction_POI42 = self.model.forward(**Input_dict)
            Loss_POI41 = self.model.loss(Prediction_POI41)
            Loss_POI42 = self.model.loss(Prediction_POI42)
            Loss_POI4 = torch.where(Loss_POI41 > Loss_POI42, Loss_POI41, Loss_POI42)

            Loss_POI3_mean = Loss_POI3.mean()
            Loss_POI4_mean = Loss_POI4.mean()
            Loss_POI1 = torch.where(Loss_POI1 > Loss_POI3_mean, Loss_POI1, Loss_POI3_mean)
            Loss_POI2 = torch.where(Loss_POI2 > Loss_POI4_mean, Loss_POI2, Loss_POI4_mean)

            cross_cate = torch.where(Loss_POI1 > Loss_POI3_mean, torch.zeros_like(Loss_POI1),
                                     torch.ones_like(Loss_POI1)).sum()

            Loss_POI1 = (Loss_POI1).mean()
            Loss_POI2 = (Loss_POI2).mean()
            Loss = self.lambda1 * (Loss_POI1) + (1 - self.lambda1) * (Loss_POI2)
            self.optim.zero_grad()
            Loss.backward()
            self.optim.step()
            Avg_Loss += Loss.item()
            Sum_cate_replace += cross_cate.item()
            post_fix = {
                "epoch": epoch,
                "iter": "[%d/%d]" % (i, len(data_loader)),
                "Avg_Loss": Avg_Loss / (i + 1),
                "Sum_cate_replace":Sum_cate_replace,
                "Loss_POI1": Loss_POI1.mean().item(),
                "Loss_POI2": Loss_POI2.mean().item(),
                "Loss_POI3": Loss_POI3.mean().item(),
                "Loss_POI4": Loss_POI4.mean().item(),
            }
            if i % self.log_freq == 0:
                print(post_fix)
                logging.info(post_fix)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), Avg_Loss / len(data_loader))
        logging.info("EP%d_%s, avg_loss=%f" % (epoch, str_code, Avg_Loss / len(data_loader)))
        return Avg_Loss / len(data_loader)

    def iteration_test(self, epoch, data_loader):
        str_code = "test"
        self.model.eval()
        data_loader.dataset.current_epoch = epoch
        predictions = list()
        for i, data in enumerate(data_loader):
            data = {key: value.to(self.device).long() for key, value in data.items()}
            Prediction_POI = self.model.forward(Inp_POI=data["Inp_POI"],Method='POI',PosNeg_POI=data["PosNeg_POI"],
                                                Inp_TIM=data["Inp_TIM"],HistoryLen=data["HistoryLen"], downstream=False)
            if self.loss_type == "Pairwise":
                prediction = Prediction_POI
            elif self.loss_type == "Classification":
                prediction = np.take_along_axis(Prediction_POI, data["PosNeg_POI"].cpu().numpy(), axis=1)
            predictions.extend(prediction.cpu().data.numpy())
        predictions = np.array(predictions)
        evaluations = dict()
        sort_idx = (-predictions).argsort(axis=1)
        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        topk = [5, 10]
        metrics = [m.strip().upper() for m in eval('["NDCG","HR"]')]
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        print("test=({}) ".format(self.format_metric(evaluations)))
        logging.info("test=({}) ".format(self.format_metric(evaluations)))

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