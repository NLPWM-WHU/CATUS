import argparse
from Pretraining.DataReader import BERTDataset
import random
import numpy as np
from torch.utils.data import DataLoader
import torch
from  Pretraining.PretrainMs import OurMethod
from Pretraining.trainer.pretrain import ModelTrainer
from vocab import WordVocab
import os
import logging

def init_weights(m):
    if 'Linear' in str(type(m)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
    if 'Embedding' in str(type(m)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)


def train():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--datasets", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-n", "--model_name", required=True, type=str,default='PretrainModel_CAT_CAT_POICATnearest_POI_POI.pkl',
                        help="output/bert.model")
    parser.add_argument("-lt", "--loss_type", required=True, type=str,
                        default='PretrainModel_CAT_CAT_POICATnearest_POI_POI.pkl',
                        help="output/XXXX.model")
    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("-r", "--seed", type=int, default=2019, help="the random seed")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--Drop", type=int, default=0, help="whether use drop loss")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=500, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=str, default='1', help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=False, help="Loading on memory: true or false")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--lambda1", type=float, default=0.5, help="category lambda1")


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    args.train_dataset = "data/COM_{}/corpus.COM_{}".format(args.datasets, args.datasets)
    args.test_dataset = "data/COM_{}/test_corpus.COM_{}".format(args.datasets, args.datasets)
    args.vocab_path = "data/COM_{}/vocab.COM_{}".format(args.datasets, args.datasets)
    args.output_path = "output/COM_{}/".format(args.datasets)

    # Create Log dir and file
    args_str = "b{}_r{}_s{}_lr{}".format(args.batch_size, args.seed, args.seq_len, args.lr)
    logdir = os.path.join('log/pretrain/', 'COM_' + args.datasets)
    logfile = os.path.join('log/pretrain/', 'COM_' + args.datasets,
                           args_str + args.model_name + 'lambda1' + str(args.lambda1))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logging.info(args)
    # Logging Parameter
    print(args)

    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    logging.info("Vocab Size: %d"%len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, args.loss_type, sequence_len=args.seq_len, train = True)
    category_size = train_dataset.category_nums
    cluster_size = train_dataset.cluster_nums
    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, args.loss_type, sequence_len=args.seq_len, train = False)

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building Pra-training model")
    pretrainmodel = OurMethod(len(vocab), category_size, cluster_size, sequence_len=args.seq_len, hidden=args.hidden, dropout=args.dropout,
                              loss_type=args.loss_type)
    pretrainmodel.apply(init_weights)
    trainer = ModelTrainer(pretrainmodel, loss_type = args.loss_type,lambda1=args.lambda1, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                          )
    print("Training Start")
    early_stop_results = []
    early_stop_step = 5
    trainer.test(0)
    for epoch in range(args.epochs):
        result = trainer.train(epoch)
        early_stop_results.append(result)
        trainer.save(epoch, args.output_path, args.model_name)
        print(early_stop_results[-early_stop_step:])
        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == '__main__':
    train()