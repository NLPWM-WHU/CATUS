import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import List, Dict, NoReturn, Any
import numpy as np
import os
from Pretraining.LossMs import OurMethodFinetuneLossM, NoneFinetuneLossM
from Pretraining.trainer.optimizer.adamw import AdamW
from torch.optim import SparseAdam
import time
class ModelTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, pretrainm, pretrain_model_type, loss_type, data_type,lambda2,
                 dataloader: dict,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, log_freq: int = 10):
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
        self.loss_type = loss_type
        self.data_type = data_type
        self.pretrainm = pretrainm
        self.lambda2 = lambda2
        FinetuneLossModel = eval('{0}FinetuneLossM'.format(pretrain_model_type))
        self.model = FinetuneLossModel(pretrainm, loss_type, data_type, self.lambda2).to(self.device)
        # Setting the train and test data loader
        self.train_data = dataloader['train']
        self.test_data = dataloader['test']
        if self.data_type != "Delta_Matrix":
            self.dev_data = dataloader['dev']
        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim = Adam(self.model.parameters(), lr=lr)
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        logging.info("Total Parameters:%d"%sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        result = self.iteration_train(epoch, self.train_data)
        return result
    def test(self, epoch):
        if self.data_type != "Delta_Matrix":
            self.iteration_test(epoch, self.dev_data, 'dev')
        self.iteration_test(epoch, self.test_data, 'test')

    def iteration_train(self, epoch, data_loader):
        str_code = 'train'
        self.model.train()
        data_loader.dataset.current_epoch = epoch
        Avg_Loss = 0.0
        for i, data in enumerate(data_loader):
            Loss_patch = self.batch_train(data, epoch)
            Loss = 0.0
            for loss in Loss_patch:
                Loss += loss
            self.optim.zero_grad()
            Loss.backward()
            self.optim.step()
            Avg_Loss += Loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": "[%d/%d]" % (i, len(data_loader)),
                "Avg_Loss": Avg_Loss / (i + 1),
                "Loss_POI": [Loss_POI.item() for Loss_POI in Loss_patch],
            }
            if i % self.log_freq == 0:
                print(post_fix)
                logging.info(post_fix)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), Avg_Loss / len(data_loader))
        logging.info("EP%d_%s, avg_loss=%f" % (epoch, str_code, Avg_Loss / len(data_loader)))
        return Avg_Loss / len(data_loader)

    def batch_train(self, data, epoch):
        data = {key: value.to(self.device) for key, value in data.items()}
        data['epoch'] = epoch
        Loss_patch = self.model.forward(data)
        return Loss_patch


    def iteration_test(self, epoch, data_loader, str_code):
        self.model.eval()
        data_loader.dataset.current_epoch = epoch
        predictions = list()
        for i, data in enumerate(data_loader):
            if self.data_type == "Delta_Matrix":
                if i % 100 == 0:
                    print("batch", i)
            data = {key: value.to(self.device) for key, value in data.items()}
            prediction = self.model.forward(data, test = True)
            predictions.extend(prediction)
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
        if str_code == 'test':
            print("test=({}) ".format(self.format_metric(evaluations)))
            logging.info("test=({}) ".format(self.format_metric(evaluations)))
        else:
            print("dev=({}) ".format(self.format_metric(evaluations)))
            logging.info("dev=({}) ".format(self.format_metric(evaluations)))


    def save(self, epoch, file_path="output/fine_tuned.model", model_name = 'STAN'):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        path, _ = os.path.split(file_path)
        if not os.path.exists(path):
            os.mkdir(path)
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
