# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from vocab import WordVocab
from typing import NoReturn
from math import ceil
from Pretraining import utils
import torch

###read combine dataset, so that it can map to a subdataset
class DataReaderpretrain(object):
    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.history_max = args.seq_len
        self.ndataset = args.ndatasets
        self.items_all = []
        self._read_data()
        self._append_his_info()

    def _read_data(self) -> NoReturn:
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        self.load_pretrain_data()
        logging.info('Counting dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time']] for df in self.data_df.values()])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        self.items_all = self.all_df['item_id'].unique()
        self.items_all.sort(axis=0)
        self.items_num = len(self.items_all) + 5
        self.subdata_item_map = dict(zip(self.items_all, range(5, len(self.items_all) + 5))) ##map 5-
        ##for safe
        for key in ['dev', 'test']:
            neg_items = np.array(self.data_df[key]['neg_items'].tolist())
        logging.info(
            '"# user": {}, "# item": {}, "# entry": {}'.format(self.n_users, self.n_items, len(self.all_df)))
        self.data_df['train'].sort_values(by=['user_id', 'time', 'item_id'], inplace=True)


    def load_pretrain_data(self):
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(
                os.path.join(self.prefix, 'COM_' + '_'.join(self.ndataset), self.dataset, key + '_map.csv'),
                sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])
        vocab = WordVocab.load_vocab(
            os.path.join(self.prefix, 'COM_' + '_'.join(self.ndataset), 'vocab.' + 'COM_' + '_'.join(self.ndataset)))
        self.vocab_size = len(vocab)
        self.local_global_dict = pickle.load(
            open(os.path.join('data/', 'COM_' + '_'.join(self.ndataset), self.dataset, 'MAP'), 'rb'))
        self.category_dict = pickle.load(
            open(os.path.join('data/', 'COM_' + '_'.join(self.ndataset), 'category.COM_' + '_'.join(self.ndataset)),
                 'rb'))
        self.category_size = max(set(self.category_dict.values())) + 1
        self.cluster_dict = pickle.load(
            open(os.path.join('data/', 'COM_' + '_'.join(self.ndataset), 'cluster.COM_' + '_'.join(self.ndataset)),
                 'rb'))
        self.cluster_size = max(set(self.cluster_dict.values())) + 1
        ###create nearest POI
        self.coordinate_dict = pickle.load(
            open(os.path.join('data/', 'COM_' + '_'.join(self.ndataset), 'coordinate.COM_' + '_'.join(self.ndataset)),
                 'rb'))
        POI_nearest_path = os.path.join('data/', 'COM_' + '_'.join(self.ndataset),
                                        'POI_nearest.COM_' + '_'.join(self.ndataset))
        if not os.path.exists(POI_nearest_path):
            self.POI_nearest = self.create_nearest(self.coordinate_dict, POI_nearest_path)
        else:
            self.POI_nearest = pickle.load(open(POI_nearest_path, 'rb'))
            if isinstance(self.POI_nearest,tuple):
                self.POI_nearest = self.POI_nearest[0]

        print("neg len")
        print(len(self.POI_nearest[0]))
        self.category_similarity = np.zeros((self.category_size, self.category_size))


    def _append_his_info(self) -> NoReturn:
        """
        Add history info to data_df: item_his, time_his, his_length
        ! Need data_df to be sorted by time in ascending order
        :return:
        """
        logging.info('Appending history info...')
        user_his_dict = dict()  # store the already seen sequence of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            i_history, t_history = [], []
            for uid, iid, t in zip(df['user_id'], df['item_id'], df['time']):
                if uid not in user_his_dict:
                    user_his_dict[uid] = []
                sub_i_history, sub_t_history = [], []
                for x in user_his_dict[uid]:
                    sub_i_history.append(x[0])
                    sub_t_history.append(x[1])
                i_history.append(sub_i_history)
                t_history.append(sub_t_history)
                user_his_dict[uid].append((iid, t))
            df['item_his'] = i_history
            df['time_his'] = t_history
            if self.history_max > 0:
                df['item_his'] = df['item_his'].apply(lambda x: x[-self.history_max:])
                df['time_his'] = df['time_his'].apply(lambda x: x[-self.history_max:])
            df['his_length'] = df['item_his'].apply(lambda x: len(x))
        self.user_clicked_set = dict()
        for uid in user_his_dict:
            self.user_clicked_set[uid] = set([x[0] for x in user_his_dict[uid]])

    def create_nearest(self, coordinate_dict, POI_nearest_path):
        poi_num = max(coordinate_dict.keys()) + 1
        coor_matrix = np.zeros((poi_num, 2))
        for poi in coordinate_dict:
            coor_matrix[poi] = coordinate_dict[poi]
        with open(POI_nearest_path, 'wb') as output:
            results, results_distance = self.calculate_distance(coor_matrix, coor_matrix, 256)
            POI_nearest = dict(zip(range(poi_num), results))
            POI_nearest_distance = dict(zip(range(poi_num), results_distance))
            pickle.dump((POI_nearest, POI_nearest_distance), output)
        return POI_nearest

    def calculate_distance(self, cluster_center_matrix, coor_matrix, Batch_size):
        cluster_num = cluster_center_matrix.shape[0]  ###N个
        cluster_center_matrix_batch = np.expand_dims(cluster_center_matrix, 0)  ###1 * N * 2
        coor_num = coor_matrix.shape[0]  ###N个
        batch_num = ceil(coor_num / Batch_size)
        coor_list = list(range(coor_num))  ###N个
        new_clusters = np.zeros((coor_num, 5))  ###N个
        new_clusters_distance = np.zeros((coor_num, 5))
        for it in range(batch_num):
            idx = coor_list[it * Batch_size: min(it * Batch_size + Batch_size, coor_num)]
            current_Batch_size = len(idx)
            current_coors = coor_matrix[idx]  ###Batch size * 2
            batch_coors = np.expand_dims(current_coors, 1)  ###Batch size*1*2
            batch_coors = batch_coors.repeat(cluster_num, axis=1)  ###Batch size*N*2
            cluster_center_matrix_batch_current = cluster_center_matrix_batch.repeat(current_Batch_size,
                                                                                     axis=0)  ###Batch size*N*2
            distance_matrix = self.get_distance(batch_coors[:,:,0],batch_coors[:,:,1],cluster_center_matrix_batch_current[:,:,0],
                                                cluster_center_matrix_batch_current[:,:,1])
            cluster_center = np.argsort(distance_matrix, axis=-1)[:, 1:6]
            cluster_distance = np.take_along_axis(distance_matrix, cluster_center, axis=1)
            new_clusters_distance[idx] = cluster_distance
            new_clusters[idx] = cluster_center
        return new_clusters, new_clusters_distance

    def get_distance(self, lat1, lng1, lat2, lng2):
        R = 6373.0

        lat1 = np.radians(lat1)
        lon1 = np.radians(lng1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lng2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = (R * c)
        return distance

    def data_analysis(self, df):
        u_records = dict()
        i_records = dict()
        for uid, iid, t in zip(df['user_id'], df['item_id'], df['time']):
            if uid not in u_records:
                u_records[uid] = []
            u_records[uid].append([iid])
            if iid not in i_records:
                i_records[iid] = 0
            i_records[iid] += 1

    def create_category_similarity(self, pretrainmodel):
        category_emb = pretrainmodel.CATembedding.token.weight.data.cpu().numpy()
        category_norm = np.linalg.norm(category_emb, axis=1, keepdims=True)
        category_emb = category_emb / category_norm
        category_sim = np.matmul(category_emb, category_emb.T)
        self.category_similarity = category_sim

    def filterout_nearest_POI(self):
        POI_numbers = 0
        for POI in self.POI_nearest:
            if POI not in self.category_dict:
                continue
            POI_cate = self.category_dict[POI]
            neighbor_POIs = self.POI_nearest[POI]
            neighbor_POIs_distance = self.POI_nearest_distance[POI]
            filter_neighbor_POIs = []
            for i, neighbor_POI in enumerate(neighbor_POIs):
                if neighbor_POIs_distance[i]<0.5:
                    filter_neighbor_POIs.append(neighbor_POI)
                elif neighbor_POIs_distance[i]>2:
                    break
                else:
                    neighbor_POI_cate = self.category_dict[neighbor_POI]
                    if self.category_similarity[POI_cate][neighbor_POI_cate] >0.5:
                    # if neighbor_POI_cate in self.category_similarity[POI_cate]:
                        filter_neighbor_POIs.append(neighbor_POI)
            self.POI_nearest[POI] = filter_neighbor_POIs
            if len(filter_neighbor_POIs) == 0:
                POI_numbers += 1
        print(POI_numbers/len(self.POI_nearest))
