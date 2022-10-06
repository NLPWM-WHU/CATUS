from torch.utils.data import Dataset
import tqdm
import torch
import random
import time
import numpy as np
from math import ceil
import pickle
from Pretraining import utils
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import networkx as nx
class GenerateDataset(Dataset):
    def __init__(self, corpus_path, vocab, loss_type, sequence_len, encoding="utf-8", train = True):
        self.history_len = sequence_len
        self.loss_type = loss_type
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_epoch = 0
        self.train_flag = train
        self.MAX_Delta_Day = 30
        self.MAX_Delta_Dis = 50
        self.datas = []
        self.sequences_POI = []
        self.sequences_TIM = []
        self.ALL_POIs = []
        self.times = []
        time_path = corpus_path.replace('corpus', 'time')
        ### load key information
        category_path = corpus_path.replace('corpus', 'category')
        with open(category_path, "rb") as f:
            self.category_dict = pickle.load(f)
        city_path = corpus_path.replace('corpus', 'city')
        with open(city_path, "rb") as f:
            self.city_dict = pickle.load(f)

        coordinate_path = corpus_path.replace('corpus', 'coordinate')
        with open(coordinate_path, "rb") as f:
            self.coordinate_dict = pickle.load(f)
        self.category_lib = dict()
        for key in self.category_dict:
            value = self.category_dict[key]
            if value not in self.category_lib:
                self.category_lib[value] = []
            self.category_lib[value].append(key)

        cluster_path = corpus_path.replace('corpus', 'cluster')
        with open(cluster_path, "rb") as f:
            self.cluster_dict = pickle.load(f)
            for poi in self.cluster_dict:
                self.cluster_dict[poi] += 1


        self.category_nums = max(set(self.category_dict.values())) + 1
        self.ALL_CATs = list(set(self.category_dict.values()))
        self.ALL_POIs = list(self.category_dict.keys())
        self.POI_trans_frequence = dict()
        self.datas = sorted(self.ALL_POIs)
        print("xixi")
        with open('POI_nearest_reg_w0.0.pkl', 'rb') as read:
            pois, poi_near_dict,poi_near_score_dict = pickle.load(read)
        POI_city = dict()
        POI_city_score = dict()
        poi_lens = np.size(pois, axis=0)
        for idx in range(poi_lens):
            POI = pois[idx]
            near_pois = poi_near_dict[idx]
            near_pois_score = poi_near_score_dict[idx]
            current_city = self.city_dict[POI]
            for idxx in range(2):
                near_poi = near_pois[idxx]
                near_poi_score = near_pois_score[idxx]
                if self.city_dict[near_poi]!=current_city:
                    if POI not in POI_city:
                        POI_city[POI] = []
                        POI_city_score[POI] = []
                    POI_city[POI].append(near_poi)
                    POI_city_score[POI].append(near_poi_score)
        print("xixi")



    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self._get_feed_dict(item)


    def _get_feed_dict(self, item):
        sample = self.random_sample(item)

        samp_POI = sample

        output = {"Inp_POI": samp_POI,
                  "HistoryLen": 1,
                  "item": item}
        return {key: torch.tensor(value).long() for key, value in output.items()}

    def random_sample(self, index):
        sample = self.get_corpus_line(index)
        return sample

    def get_corpus_line(self, item):
        return self.datas[item]

