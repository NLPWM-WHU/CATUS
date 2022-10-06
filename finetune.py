import argparse
import random
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from Pretraining.PretrainMs import OurMethod
from Pretraining.FinetuneMs import SASRec, STRNNM, STLSTMM, RNNM, STAN, FlashBack, FlashBackp
from Pretraining.modules import DownstreamEmbed
from Pretraining.DataReader import FinetuneDatasettype1, FinetuneDatasettype2, FinetuneDatasettype3, FinetuneDatasettype4
from vocab import WordVocab
import os
import logging
from Pretraining.DataReader import DataReadernopretrain
from Pretraining.DataReader import DataReaderpretrain
from Pretraining.trainer.finetune import ModelTrainer
from Pretraining.LossMs import FinetuneLossMs, NoneFinetuneLossM




def non_increasing(lst: list) -> bool:
    current_result = lst[-1]
    previous_result = lst[:-1]
    return all(x >= current_result for x in previous_result)


def non_increasing(lst: list) -> bool:
    return all(x >= y for x, y in zip(lst, lst[1:]))


def finetunemodelparameter(args):
    finetuned_model = args.finetuned_model
    if finetuned_model == 'SASRec':
        parameters = {'sequence_len':args.seq_len, 'hidden':args.hidden, 'output_size':args.item_size}
    if finetuned_model == 'Ablation':
        parameters = {'sequence_len':args.seq_len, 'hidden':args.hidden, 'output_size':args.item_size}
    if finetuned_model == 'STLSTMM':
        parameters = {'num_slots': 10, 'aux_embed_size': 16, 'time_thres':10800, 'dist_thres':0.1,
                      'input_size':args.hidden,'lstm_hidden_size':args.hidden * 4, 'fc_hidden_size':args.hidden * 4,
                      'num_layers':1,'seq2seq':False,'output_size':args.item_size}
    if finetuned_model == 'RNNM':
        parameters = {'input_size': args.hidden, 'rnn_hidden_size': args.hidden * 4, 'fc_hidden_size': args.hidden * 4,
                      'num_layers': 1, 'seq2seq': False, 'output_size': args.item_size}

    if finetuned_model == 'STAN':
        hours = 24 * 7
        parameters = {'t_dim': hours + 1, 'l_dim': args.item_size,
                      'embed_dim': args.hidden, 'u_dim': args.user_size,
                      'ex': args.ex, 'dropout': 0}
        # pre_model = RnnLocPredictor(embed_layer, input_size=embed_size, rnn_hidden_size=hidden_size, fc_hidden_size=hidden_size,
        #                                         output_size=num_loc, num_layers=1, seq2seq=pre_model_seq2seq)
    if finetuned_model == 'STRNNM':
        st_time_window = 7200
        st_dist_window = 0.1
        st_inter_size = args.hidden
        st_num_slots = 10
        parameters = {'num_slots':st_num_slots,'time_window':st_time_window, 'dist_window':st_dist_window,
                                              'input_size':args.hidden, 'hidden_size':args.hidden * 4,
                                              'inter_size':st_inter_size,'output_size':args.item_size}
    if finetuned_model == 'FlashBack' or finetuned_model == 'FlashBackp':
        path_loc_graph = os.path.join(args.path, args.dataset, args.loc_graph)
        path_usr_graph = os.path.join(args.path, args.dataset, args.usr_graph)
        parameters = {'sequence_len':args.seq_len, 'hidden':args.hidden,
                      'location_size':args.item_size, 'user_size':args.user_size,
                      'path_loc_graph':path_loc_graph, 'path_usr_graph':path_usr_graph}

    return parameters

def use_pretrain(args):
    fintune_model_name = eval('{0}'.format(args.finetuned_model))
    pretrain_model_name = eval('{0}'.format(args.pretrained_model_type))

    print("Loading Dataset", args.dataset)
    Corpus = DataReaderpretrain(args)

    print("Loading Pra-training model")
    pretrainmodel = torch.load(
        os.path.join('output/', 'COM_' + '_'.join(args.ndatasets), args.pretrained_model_path))
    print("Loading Dataset", args.dataset)
    data_dict = dict()

    print("Create DatasetLoader")
    print("data type", args.data_type)
    print("loss type", args.loss_type)
    for phase in ['train', 'dev', 'test']:
        if args.data_type == "Common":
            data_dict[phase] = FinetuneDatasettype1(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
        elif args.data_type == 'Delta':
            data_dict[phase] = FinetuneDatasettype2(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
        elif args.data_type == "Deltav2":
            data_dict[phase] = FinetuneDatasettype4(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
        elif args.data_type == "Delta_Matrix":
            data_dict[phase] = FinetuneDatasettype3(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
            if phase == 'train':
                args.ex = data_dict[phase].max_dis, data_dict[phase].min_dis, data_dict[phase].max_tim, data_dict[phase].min_tim
                print(args.ex)

    dataloader_dict = dict()
    if args.data_type == "Delta_Matrix":
        for phase in ['train', 'test']:
            if phase == 'test':
                batch_size = 16
            else:
                batch_size = args.batch_size
            dataloader_dict[phase] = DataLoader(data_dict[phase], batch_size=batch_size, shuffle=True,
                                                num_workers=args.num_workers, collate_fn=data_dict[phase].collate_batch)
    else:
        for phase in ['train', 'dev', 'test']:
            if phase != 'train':
                batch_size = 128
            else:
                batch_size = args.batch_size
            dataloader_dict[phase] = DataLoader(data_dict[phase], batch_size=batch_size, shuffle=True,
                                                num_workers=args.num_workers, collate_fn=data_dict[phase].collate_batch)

    print("Load PretrainModel Parameters")
    args.vocab_path = "data/COM_{}/vocab.COM_{}".format('_'.join(args.ndatasets), '_'.join(args.ndatasets))
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    item_size = Corpus.items_num
    print(item_size)
    args.item_size = item_size
    cate_size = Corpus.category_size
    args.user_size = Corpus.n_users

    # embed_layer = pretrainmodel
    parameters = finetunemodelparameter(args)
    finetunemodel = fintune_model_name(**parameters, embed_layer=None)
    finetunemodel.embed_layer = pretrainmodel

    trainer = ModelTrainer(finetunemodel, args.pretrained_model_type, args.loss_type, args.data_type,lambda2=args.lambda2,
                           dataloader=dataloader_dict,
                           lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                           with_cuda=args.gpu, log_freq=args.log_freq,
                           )
    return trainer

def no_pretrain(args):
    fintune_model_name = eval('{0}'.format(args.finetuned_model))
    print("Loading Dataset", args.dataset)
    Corpus = DataReadernopretrain(args)

    print("Loading Dataset", args.dataset)
    data_dict = dict()

    print("Create DatasetLoader")
    print("data type", args.data_type)
    print("loss type", args.loss_type)
    for phase in ['train', 'dev', 'test']:
        if args.data_type == "Common":
            data_dict[phase] = FinetuneDatasettype1(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
        elif args.data_type == "Delta":
            data_dict[phase] = FinetuneDatasettype2(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
        elif args.data_type == "Deltav2":
            data_dict[phase] = FinetuneDatasettype4(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
        elif args.data_type == "Delta_Matrix":
            data_dict[phase] = FinetuneDatasettype3(Corpus, args.seq_len, phase=phase, loss_type=args.loss_type)
            if phase == 'train':
                args.ex = data_dict[phase].max_dis, data_dict[phase].min_dis, data_dict[phase].max_tim, data_dict[phase].min_tim
                print(args.ex)
    dataloader_dict = dict()
    if args.data_type == "Delta_Matrix":
        for phase in ['train', 'test']:
            if phase == 'test':
                batch_size = 16
            else:
                batch_size = args.batch_size
            dataloader_dict[phase] = DataLoader(data_dict[phase], batch_size=batch_size, shuffle=True,
                                                num_workers=args.num_workers, collate_fn=data_dict[phase].collate_batch)
    else:
        for phase in ['train', 'dev', 'test']:
            if phase != 'train':
                batch_size = 128
            else:
                batch_size = args.batch_size
            dataloader_dict[phase] = DataLoader(data_dict[phase], batch_size=batch_size, shuffle=True,
                                                num_workers=args.num_workers, collate_fn=data_dict[phase].collate_batch)
    item_size = Corpus.items_num
    print(item_size)
    args.item_size = item_size
    args.user_size = Corpus.n_users

    embed_layer = DownstreamEmbed(item_size, embed_size=args.hidden)
    parameters = finetunemodelparameter(args)
    finetunemodel = fintune_model_name(**parameters, embed_layer=embed_layer)


    trainer = ModelTrainer(finetunemodel, args.pretrained_model_type, args.loss_type, args.data_type, lambda2=args.lambda2,
                           dataloader=dataloader_dict,
                           lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                           with_cuda=args.gpu, log_freq=args.log_freq,
                           )
    return trainer

def train():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--ndatasets", required=True, nargs='+', help="pre-train datasets")
    parser.add_argument("-f", "--dataset", required=True, type=str, help="fine-tune dataset")
    parser.add_argument("-u", "--use_pretrain", type=int, default=None, help="whether use pretrain or not")
    parser.add_argument("-mpp", "--pretrained_model_path", type=str, default='None', help="whether use pretrain or not")
    parser.add_argument("-mpt", "--pretrained_model_type", type=str, default='None', help="the type of pretrained model")
    parser.add_argument("-mft", "--finetuned_model", type=str, default='SASRec', help="whether use pretrain or not")
    parser.add_argument("-dtt", "--data_type", type=str, default='Common', help="which kind of loss function")
    parser.add_argument("-lst", "--loss_type", type=str, default='Pairwise', help="which kind of loss function")

    parser.add_argument('--path', type=str, default='./data/', help='Input data dir.')
    parser.add_argument('--sep', type=str, default='\t', help='sep of csv file.')
    parser.add_argument("-lg",'--loc_graph', type=str, help='Input graph dir.')
    parser.add_argument("-ug",'--usr_graph', type=str, help='Input graph dir.')

    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="dropout rate")


    parser.add_argument("-r", "--seed", type=int, default=2019, help="the random seed")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=500, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--gpu", type=str, default='1', help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=False, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--lambda2", type=float, default=0.1, help="geo lambda2")



    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create Log dir and file
    args_str = "b{}_r{}_p{}_s{}_mf{}_data{}_lambda2{}".format(args.batch_size, args.seed,
                                                    int(args.use_pretrain), args.seq_len, args.finetuned_model, args.dataset, args.lambda2)
    logdir = os.path.join('log/finetune/', 'COM_' + '_'.join(args.ndatasets))
    logfile = os.path.join('log/finetune/', 'COM_' + '_'.join(args.ndatasets), args_str + args.pretrained_model_path)
    output_path = os.path.join('output_finetune/', 'COM_' + '_'.join(args.ndatasets)) + '/'
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

    if args.use_pretrain:
        trainer = use_pretrain(args)
    else:
        trainer = no_pretrain(args)

    print("Training Start")
    logging.info("Training Start")
    early_stop_results = []
    early_stop_step = 5

    trainer.test(0)
    for epoch in range(args.epochs):
        result = trainer.train(epoch)
        early_stop_results.append(result)
        if args.finetuned_model == 'STAN':
            if (epoch + 1) % 5 == 0:
                trainer.test(epoch)
        else:
            trainer.test(epoch)
        print(early_stop_results[-early_stop_step:])

if __name__ == '__main__':
    train()