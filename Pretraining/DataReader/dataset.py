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
from collections import defaultdict
from collections import Counter
class BERTDataset(Dataset):
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
        if self.train_flag:
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

            ###create category transition pattern
            self.category_trans = [dict(), dict()]
            self.category_trans_all = [dict(), dict()]
            self.triple_cate_trans = [dict(), dict()]
            self.triple_cate_trans_count = [dict(), dict()]
            self.category_trans_city = dict()
            self.timecate_trans = dict()
            self.time_trans_hour = dict()
            self.time_trans_wday = dict()
            self.time_trans_month = dict()
            self.time_lib = dict()

        else:
            category_path = corpus_path.replace('test_corpus', 'category')
            with open(category_path, "rb") as f:
                self.category_dict = pickle.load(f)
            coordinate_path = corpus_path.replace('test_corpus', 'coordinate')
            with open(coordinate_path, "rb") as f:
                self.coordinate_dict = pickle.load(f)
            city_path = corpus_path.replace('test_corpus', 'city')
            with open(city_path, "rb") as f:
                self.city_dict = pickle.load(f)
            cluster_path = corpus_path.replace('test_corpus', 'cluster')
            with open(cluster_path, "rb") as f:
                self.cluster_dict = pickle.load(f)


        self.category_nums = max(set(self.category_dict.values())) + 1
        self.ALL_CATs = list(set(self.category_dict.values()))
        self.ALL_POIs = list(self.category_dict.keys())
        self.POI_trans_frequence = dict()
        ###city corresponding
        city_num = len(set(list(self.city_dict.values())))
        self.ALL_POIs_city = dict()
        self.category_lib_city = dict()
        for city in range(city_num):
            self.ALL_POIs_city[city] = []
            self.category_lib_city[city] = dict()
        for POI in self.ALL_POIs:
            self.ALL_POIs_city[self.city_dict[POI]].append(POI)
            cate = self.category_dict[POI]
            if cate not in self.category_lib_city[self.city_dict[POI]]:
                self.category_lib_city[self.city_dict[POI]][cate] = []
            self.category_lib_city[self.city_dict[POI]][cate].append(POI)


        self.ALL_CLUs = list(set(self.cluster_dict.values()))
        self.cluster_nums = max(set(self.cluster_dict.values())) + 1
        with open(corpus_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_POI.append([int(POI) for POI in sentences[0].split()])
        with open(time_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_TIM.append([float(TIM) for TIM in sentences[0].split()])

        ####process sequences, different between train and test
        if self.train_flag:
            self.POI_nearest = dict()
            self.POI_user_time_nearest = defaultdict(dict)
            self.POI_time_checkin = dict()
            for POI in self.ALL_POIs:
                self.POI_time_checkin[POI] = [0] * 24

            for i, sequence in enumerate(self.sequences_POI):
                seqlen = len(sequence)
                for ind in range(seqlen - 1):
                    begin_ind = max(ind-self.history_len + 1, 0)
                    prev_POIs = sequence[begin_ind:ind + 1]
                    next_POI = sequence[ind + 1]
                    prev_TIM = [timestamp for timestamp in self.sequences_TIM[i][begin_ind:ind + 1]]
                    # last_TIM = time.localtime(self.sequences_TIM[i][ind]).tm_hour
                    next_TIM = self.sequences_TIM[i][ind + 1]
                    self.datas.append([[prev_POIs, next_POI], [prev_TIM, next_TIM],
                                  [[self.category_dict[POI] for POI in prev_POIs], self.category_dict[next_POI]],
                                       [[self.cluster_dict[POI] for POI in prev_POIs], self.cluster_dict[next_POI]]])
                    ##twice
                    key = "{},{}".format(self.category_dict[prev_POIs[-1]], self.category_dict[next_POI])
                    value = "{},{}".format(prev_POIs[-1], next_POI)
                    cityid = self.city_dict[prev_POIs[-1]]
                    if key not in self.category_trans[cityid]:
                        self.category_trans[cityid][key] = set()
                        self.category_trans_all[cityid][key] = set()
                    self.category_trans[cityid][key].add(value)
                    self.category_trans_all[cityid][key].add(value)


            for cityid in range(2):
                for key in list(self.category_trans[cityid].keys()):
                    self.category_trans[cityid][key] = list(self.category_trans[cityid][key])
                    if len(self.category_trans[cityid][key]) < 5:
                        self.category_trans[cityid].pop(key)

            self.common_trans = self.category_trans[0].keys()&self.category_trans[1].keys()
            self.individual_trans = [dict(), dict()]
            self.individual_trans[0] = self.category_trans[1].keys() - self.common_trans
            self.individual_trans[1] = self.category_trans[0].keys() - self.common_trans


            ###create nearest POI
            POI_nearest_path = corpus_path.replace('corpus', 'POI_nearest')
            if not os.path.exists(POI_nearest_path):
                self.POI_nearest = self.create_nearest(self.coordinate_dict, POI_nearest_path)
            else:
                self.POI_nearest = pickle.load(open(POI_nearest_path, 'rb'))
            POI_time_nearest_path = corpus_path.replace('corpus', 'POI_time_nearest')
            if not os.path.exists(POI_time_nearest_path):
                self.POI_time_nearest = self.create_time_nearest(self.POI_time_checkin, POI_time_nearest_path)
            else:
                self.POI_time_nearest = pickle.load(open(POI_time_nearest_path, 'rb'))

            if self.loss_type == 'Classification':
                self.buffer = 1
                self.buffer_dict = dict()
                self._prepare()
            elif self.loss_type == 'Pairwise':
                self.buffer = 0


        else:
            self.category_trans_test = [dict(), dict()]
            # with open("Yelp_Filter_POI.pkl", 'rb') as read:
            #     self.filter_POI = pickle.load(read)
            # for i, sequence in enumerate(self.sequences_POI):
            #     for POI in sequence:
            #         if POI in self.filter_POI:
            #             self.filter_POI.remove(POI)
            # with open("Yelp_Filter_POI_1.pkl", 'wb') as out:
            #     pickle.dump(self.filter_POI,out)
            #
            # print("xist")
            for i, sequence in enumerate(self.sequences_POI):
                seqlen = len(sequence)
                if seqlen<2:
                    continue
                ind = seqlen - 2
                begin_ind = max(ind - self.history_len + 1, 0)
                prev_POIs = sequence[begin_ind:ind + 1]
                next_POI = sequence[ind + 1]
                prev_TIM = [timestamp for timestamp in self.sequences_TIM[i][begin_ind:ind + 1]]
                next_TIM = self.sequences_TIM[i][ind + 1]
                self.datas.append([[prev_POIs, next_POI], [prev_TIM, next_TIM],
                              [[self.category_dict[POI] for POI in prev_POIs], self.category_dict[next_POI]],
                                   [[self.cluster_dict[POI] for POI in prev_POIs], self.cluster_dict[next_POI]]])
            self.buffer = 1
            self.buffer_dict = dict()
            self._prepare()



    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.buffer_dict[item] if self.buffer else self._get_feed_dict(item)

    def _prepare(self):
        if self.buffer:
            for i in tqdm(range(len(self)), leave=False, ncols=100, mininterval=1,
                          desc=('Prepare for test')):
                self.buffer_dict[i] = self._get_feed_dict(i)

    def _get_feed_dict(self, item):
        sample = self.random_sample(item)
        samp_POI, samp_TIM, samp_CAT, samp_CLU = sample[0], sample[1], sample[2], sample[3]
        Inp_Sample_POI = np.array(samp_POI[0])
        Pos_Sample_POI = np.array([samp_POI[1]])
        Inp_Sample_TIM_stamp = np.array(samp_TIM[0])
        Inp_Sample_TIM = np.array([self.calculate_timid(timestamp) for timestamp in samp_TIM[0]])
        Pos_Sample_TIM = np.array([self.calculate_timid(samp_TIM[1])])

        Inp_Sample_CAT = np.array(samp_CAT[0])
        Pos_Sample_CAT = np.array([samp_CAT[1]])

        City = np.array(self.city_dict[samp_POI[1]])
        if self.train_flag:
            Inp_Sample_POI_Neg_c1 = self.create_neg_samples_c1(Inp_Sample_POI)
            Neg_Sample_POI, Neg_Sample_CAT, Neg_Sample_CLU = self.Sample_Neg_Next_POI(samp_POI)
            Inp_Sample_POI_Neg_c2 = self.create_neg_samples_c2(Inp_Sample_POI)

        else:
            Neg_Sample_POI, Neg_Sample_CAT, Neg_Sample_CLU = self.All_Neg_Next_POI(samp_POI)


        ###attention!
        HistoryLen = sum([1 for i in Inp_Sample_POI if i > 0])
        Padding = np.array([self.vocab.pad_index for _ in range(self.history_len - len(Inp_Sample_POI))])
        Inp_Sample_POI = np.concatenate((Inp_Sample_POI, Padding), axis=0)
        Inp_Sample_CAT = np.concatenate((Inp_Sample_CAT, Padding), axis=0)
        Inp_Sample_TIM = np.concatenate((Inp_Sample_TIM, Padding), axis=0)
        Inp_Sample_TIM_stamp  = np.concatenate((Inp_Sample_TIM_stamp, Padding), axis=0)
        if self.train_flag:
            Inp_Sample_POI_Neg_c2 = np.concatenate((Inp_Sample_POI_Neg_c2, Padding), axis=0)
            Inp_Sample_POI_Neg_c1 = np.concatenate((Inp_Sample_POI_Neg_c1, Padding), axis=0)
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_CAT": Inp_Sample_CAT,
                      "Inp_TIM": Inp_Sample_TIM,
                      "Inp_TIM_stamp": Inp_Sample_TIM_stamp,
                      "Inp_POI_Neg_c1": Inp_Sample_POI_Neg_c1,
                      "Inp_POI_Neg_c2": Inp_Sample_POI_Neg_c2,
                      "City": City,
                      # "PosNeg_POI_Neg_c2": np.concatenate((Pos_Sample_POI_Neg_c2, Neg_Sample_POI_Neg_c2), axis=0),
                      "PosNeg_POI": np.concatenate((Pos_Sample_POI, Neg_Sample_POI), axis=0),
                      "PosNeg_CAT": np.concatenate((Pos_Sample_CAT, Neg_Sample_CAT), axis=0),
                      "Pos_TIM": Pos_Sample_TIM,
                      "HistoryLen": HistoryLen,
                      "item": item}
        else:
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_CAT": Inp_Sample_CAT,
                      "Inp_TIM": Inp_Sample_TIM,
                      "Inp_TIM_stamp": Inp_Sample_TIM_stamp,
                      "PosNeg_POI": np.concatenate((Pos_Sample_POI, Neg_Sample_POI), axis=0),
                      "PosNeg_CAT": np.concatenate((Pos_Sample_CAT, Neg_Sample_CAT), axis=0),
                      "Pos_TIM": Pos_Sample_TIM,
                      "HistoryLen": HistoryLen,
                      "item": item}
        return {key: torch.tensor(value).long() for key, value in output.items()}

    def calculate_timid(self, timestamp):
        timetuple = time.localtime(timestamp)
        wid = 0 if timetuple.tm_wday < 5 else 1
        timeid = wid * 24 + timetuple.tm_hour
        return timeid

    def create_nearest(self, coordinate_dict, POI_nearest_path):
        poi_num = max(coordinate_dict.keys()) + 1
        coor_matrix = np.zeros((poi_num, 2))
        for poi in coordinate_dict:
            coor_matrix[poi] = coordinate_dict[poi]
        with open(POI_nearest_path, 'wb') as output:
            results = self.calculate_distance(coor_matrix, coor_matrix, 256)
            POI_nearest = dict(zip(range(poi_num), results))
            pickle.dump(POI_nearest, output)
        return POI_nearest

    def create_time_nearest(self, POI_time_checkin, POI_nearest_path):
        poi_num = max(POI_time_checkin.keys()) + 1
        time_matrix = np.zeros((poi_num, 24))
        for poi in POI_time_checkin:
            dis_min =min(POI_time_checkin[poi])
            dis_max = max(POI_time_checkin[poi])
            time_matrix[poi] = (np.array(POI_time_checkin[poi]) - dis_min)/(dis_max - dis_min + 1e-8)
        with open(POI_nearest_path, 'wb') as output:
            results = self.calculate_distance(time_matrix, time_matrix, 256, euclid = True)
            POI_nearest = dict(zip(range(poi_num), results))
            pickle.dump(POI_nearest, output)
        return POI_nearest

    def create_neg_samples_pair(self, samp_POI):
        seqlen = len(samp_POI)
        samp_POI_neg = []
        cityid = 1 - self.city_dict[samp_POI[0]]
        for i in range(seqlen):
            neg_cat = self.category_dict[samp_POI[i]]
            if i == 0:
                neg_POI = self.generate_from_category(cityid, neg_cat, samp_POI[i])
                samp_POI_neg.append(neg_POI)
            else:
                neg_cat_last = self.category_dict[samp_POI[i - 1]]
                key = "{},{}".format(neg_cat_last, neg_cat)
                neg_POI = self.generate_neg_POI(key, cityid, neg_cat, samp_POI[i])
                samp_POI_neg.append(neg_POI)
        return np.array(samp_POI_neg)


    def create_neg_samples_c1(self, samp_POI):
        P = 0.3
        seqlen = len(samp_POI)
        sample_num = int(np.ceil(seqlen * P))
        random_index = random.sample(list(range(seqlen)), sample_num)
        samp_POI_neg = []
        for i in range(seqlen):
            if i not in random_index or i == 0:
                samp_POI_neg.append(samp_POI[i])
            else:
                neg_cat_last = self.category_dict[samp_POI[i - 1]]
                neg_cat = self.category_dict[samp_POI[i]]
                key = "{},{}".format(neg_cat_last, neg_cat)
                cityid = self.city_dict[samp_POI[i]]
                neg_POI = self.generate_neg_POI(key, cityid, neg_cat, samp_POI[i])
                samp_POI_neg.append(neg_POI)
        return np.array(samp_POI_neg)

    def create_neg_samples_c2(self, samp_POI):
        P = 0.3
        seqlen = len(samp_POI)
        sample_num = int(np.ceil(seqlen * P))
        random_index = list(range(seqlen))[-sample_num:]
        samp_POI_neg = []
        if seqlen == 1:
            samp_POI_neg = [POI for POI in samp_POI]
        else:
            for i in range(seqlen):
                if i not in random_index:
                    samp_POI_neg.append(samp_POI[i])
                else:
                    neg_POI = self.vocab.mask_index
                    samp_POI_neg.append(neg_POI)
        return np.array(samp_POI_neg)

    def create_neg_samples_c3(self, samp_POI, pos_POI1, pos_POI2):
        P = 1.0
        seqlen = len(samp_POI)
        sample_num = int(np.ceil(seqlen * P))
        random_index = random.sample(list(range(seqlen)), sample_num)
        samp_POI_neg = []
        for i in range(seqlen):
            if i not in random_index or i == 0:
                samp_POI_neg.append(samp_POI[i])
            else:
                neg_POI = self.generate_neg_POI2(samp_POI[i])
                samp_POI_neg.append(neg_POI)

        neg_pos_POI1 = self.generate_neg_POI2(pos_POI1)
        neg_pos_POI2 = self.generate_neg_POI2(pos_POI2)
        return np.array(samp_POI_neg), np.array([neg_pos_POI1]), np.array([neg_pos_POI2])


    def generate_neg_POI(self, key, cityid, neg_cat, pos_POI):
        if key not in self.category_trans[cityid]:
            neg_POI = self.generate_from_category(cityid, neg_cat, pos_POI)
        else:
            samp_trans = random.choice(self.category_trans[cityid][key])
            neg_POI = int(samp_trans.split(',')[1])
        return neg_POI

    def generate_neg_individual_POI(self, key, cityid, neg_cat, pos_POI):
        if key not in self.category_trans[cityid] and key in self.category_trans[1-cityid]:
            samp_trans = random.choice(self.category_trans[1-cityid][key])
            neg_POI = int(samp_trans.split(',')[1])
        else:
            neg_POI = self.generate_from_category(cityid, neg_cat, pos_POI)
        return neg_POI

    def generate_neg_POI2(self, pos_POI):
        if pos_POI not in self.POI_cross_city_near:
            neg_POI = pos_POI
        else:
            # neg_POI = random.choice(self.POI_cross_city_near[pos_POI])
            neg_POI = self.POI_cross_city_near[pos_POI][0]
        return neg_POI

    def generate_from_category(self, cityid, neg_cat, pos_POI):
        if neg_cat not in self.category_lib_city[cityid]:
            neg_POI = pos_POI
        else:
            neg_POI = random.choice(self.category_lib_city[cityid][neg_cat])
        return neg_POI

    def random_all_POIs(self, pos_POI):
        neg_POI = random.choice(self.ALL_POIs)
        while neg_POI == pos_POI:
            neg_POI = random.choice(self.ALL_POIs)
        return neg_POI
    def create_neg_samples_version3(self, samp_POI):
        P = 0.3
        samp_POI_neg = []
        seqlen = len(samp_POI)
        sample_num = int(np.ceil(seqlen * P))
        random_index = random.sample(list(range(seqlen)), sample_num)
        for i in range(seqlen):
            if i in random_index:
                samp_POI_neg.append(np.random.choice(self.POI_nearest[samp_POI[i]]))
            else:
                samp_POI_neg.append(samp_POI[i])
        return np.array(samp_POI_neg)

    def City_label(self, POI):
        if self.city_dict[POI] == 0:
            return np.array([1, 0])
        else:
            return np.array([0, 1])

    def Sample_Neg_Next_POI(self, samp_POI):

        Neg_Sample = random.choice(self.ALL_POIs)
        while Neg_Sample == samp_POI[1]:
            Neg_Sample = random.choice(self.ALL_POIs)

        # Neg_Sample_TIM = random.choice(list(range(24 * 7 + 1)))
        # while Neg_Sample_TIM == samp_TIM[1]:
        #     Neg_Sample_TIM = random.choice(list(range(24 * 7 + 1)))

        Neg_Sample_CAT = random.choice(self.ALL_CATs)
        while Neg_Sample_CAT == self.category_dict[samp_POI[1]]:
            Neg_Sample_CAT = random.choice(self.ALL_CATs)

        Neg_Sample_CLU = random.choice(self.ALL_CLUs)
        while Neg_Sample_CLU == self.cluster_dict[samp_POI[1]]:
            Neg_Sample_CLU = random.choice(self.ALL_CLUs)

        return np.array([Neg_Sample]), np.array([Neg_Sample_CAT]), np.array([Neg_Sample_CLU])

    def Sample_Neg_Next_CAT(self, samp_CAT):
        Neg_Sample = random.choice(self.ALL_CATs)
        while Neg_Sample in samp_CAT:
            Neg_Sample = random.choice(self.ALL_CATs)
        return [Neg_Sample]

    def Sample_Neg_Next_CLU(self, samp_CLU):
        Neg_Sample = random.choice(self.ALL_CLUs)
        while Neg_Sample in samp_CLU:
            Neg_Sample = random.choice(self.ALL_CLUs)
        return [Neg_Sample]


    def Sample_Neg_Delta_DIS(self, samp_Delta_DIS):
        Neg_Sample = random.choice(list(range(self.MAX_Delta_Dis + 1)))
        while Neg_Sample == samp_Delta_DIS:
            Neg_Sample = random.choice(list(range(self.MAX_Delta_Dis + 1)))
        return np.array([Neg_Sample])

    def Sample_Neg_TIM(self, samp_TIM):
        Neg_Sample = random.choice(list(range(24 * 7 + 1)))
        while Neg_Sample == samp_TIM[1]:
            Neg_Sample = random.choice(list(range(24 * 7 + 1)))
        return [Neg_Sample]

    def Sample_Neg_Delta_TIM(self, samp_Delta_TIM):
        Neg_Sample = random.choice(list(range(self.MAX_Delta_Day + 1)))
        while Neg_Sample == samp_Delta_TIM:
            Neg_Sample = random.choice(list(range(self.MAX_Delta_Day + 1)))
        return [Neg_Sample]


    def All_Neg_Next_POI(self, samp_POI):
        Pos_POI = samp_POI[1]
        # Neg_All = [x for x in self.ALL_POIs if x != Pos_POI]
        Neg_All_CAT = []
        Neg_All_CLU = []
        Neg_All = list(np.random.choice(self.ALL_POIs, size=(99)))
        for i, _ in enumerate(Neg_All):
            while Neg_All[i] == Pos_POI:
                Neg_All[i] = np.random.choice(self.ALL_POIs)
            Neg_All_CAT.append(self.category_dict[Neg_All[i]])
            Neg_All_CLU.append(self.cluster_dict[Neg_All[i]])
        return np.array(Neg_All), np.array(Neg_All_CAT), np.array(Neg_All_CLU)


    def All_Neg_Next_CAT(self, samp_CAT):
        Pos_CAT = samp_CAT[1]
        Neg_All = [x for x in self.ALL_CATs if x != Pos_CAT]
        return Neg_All

    def All_Neg_Delta_DIS(self, samp_Delta_DIS):
        Neg_All = [x for x in list(range(self.MAX_Delta_Dis + 1)) if x != samp_Delta_DIS]
        return np.array(Neg_All)
    def All_Neg_TIM(self, samp_TIM):
        Pos_TIM = samp_TIM[1]
        Neg_All = [x for x in list(range(24 * 7 + 1)) if x != Pos_TIM]
        return Neg_All

    def All_Neg_Delta_TIM(self, samp_Delta_TIM):
        Neg_All = [x for x in list(range(self.MAX_Delta_Day + 1)) if x != samp_Delta_TIM]
        return Neg_All

    def Calculate_Cat_Accuracy(self, samp_CAT, Pos_CAT):
        samp_CAT = samp_CAT.cpu().numpy()
        Pos_CAT = Pos_CAT.cpu().numpy()
        Hit = 0
        for idx in range(len(samp_CAT)):
            if Pos_CAT[idx][0] == samp_CAT[idx]:
                Hit += 1
        return Hit

    def Sample_Neg_Next_POI_Cat(self, Inp_POI, samp_CAT):
        Inp_POI = Inp_POI.cpu().numpy()
        samp_Pos = np.zeros((np.size(Inp_POI, axis=0), 1), dtype=np.int64)
        samp_CAT = samp_CAT.cpu().numpy()
        for idx in range(len(samp_CAT)):
            if samp_CAT[idx] in self.category_lib:
                Target_POI = Inp_POI[idx][0]
                Target_Lat = self.coordinate_dict[Target_POI][0]
                Target_Lon = self.coordinate_dict[Target_POI][1]
                Pos_Sample = self.Category_coor(samp_CAT[idx], Target_Lat, Target_Lon)

                # Pos_Sample = random.choice(self.category_lib[samp_CAT[idx]])
            else:
                Pos_Sample = Inp_POI[idx][0]
            samp_Pos[idx] = Pos_Sample
        Inp_POI = np.concatenate((Inp_POI, samp_Pos), axis = -1)
        return torch.tensor(Inp_POI)

    def Category_coor(self, cate, Target_Lat, Target_Lon):
        current_cluster = self.category_lib[cate]
        Lat_cluster = np.zeros((len(current_cluster)))
        Lon_cluster = np.zeros((len(current_cluster)))
        for i, POI in enumerate(current_cluster):
            Lat_cluster[i] = self.coordinate_dict[POI][0]
            Lon_cluster[i] = self.coordinate_dict[POI][1]
        distances = self.get_distance(Target_Lat, Target_Lon, Lat_cluster, Lon_cluster)
        closet_distances = np.argmax(distances)
        return current_cluster[closet_distances]


    def random_sample(self, index):
        sample = self.get_corpus_line(index)
        return sample

    def get_corpus_line(self, item):
        return self.datas[item]

    def get_random_line(self):
        return self.datas[random.randrange(len(self.datas))][1]

    def get_distance(self, lat1, lng1, lat2, lng2):
        # coords_1 = (lat1, lng1)
        # coords_2 = (lat2, lng2)
        #
        # return geopy.distance.vincenty(coords_1, coords_2).km

        # approximate radius of earth in km
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
        # distance = (R * c).astype(np.int)
        # return int(distance/4)
        return distance

    def euclid_distance(self, x, y):
        d1 = np.sqrt(np.sum(np.square(x - y), axis=-1))
        return d1

    def calculate_distance(self, cluster_center_matrix, coor_matrix, Batch_size, euclid = False):
        cluster_num = cluster_center_matrix.shape[0]  ###N个
        cluster_center_matrix_batch = np.expand_dims(cluster_center_matrix, 0)  ###1 * N * 2
        coor_num = coor_matrix.shape[0]  ###N个
        batch_num = ceil(coor_num / Batch_size)
        coor_list = list(range(coor_num))  ###N个
        new_clusters = np.zeros((coor_num, 20))  ###N个
        for it in range(batch_num):
            # print(it)
            idx = coor_list[it * Batch_size: min(it * Batch_size + Batch_size, coor_num)]
            current_Batch_size = len(idx)
            current_coors = coor_matrix[idx]  ###Batch size * 2
            batch_coors = np.expand_dims(current_coors, 1)  ###Batch size*1*2
            batch_coors = batch_coors.repeat(cluster_num, axis=1)  ###Batch size*N*2
            cluster_center_matrix_batch_current = cluster_center_matrix_batch.repeat(current_Batch_size,
                                                                                     axis=0)  ###Batch size*N*2
            if euclid:
                distance_matrix = self.euclid_distance(batch_coors, cluster_center_matrix_batch_current)
            else:
                distance_matrix = self.get_distance(batch_coors[:,:,0],batch_coors[:,:,1],cluster_center_matrix_batch_current[:,:,0],
                                                    cluster_center_matrix_batch_current[:,:,1])
            cluster_center = np.argsort(distance_matrix, axis=-1)[:, 1:21]
            new_clusters[idx] = cluster_center
        return new_clusters



class FinetuneDataset(Dataset):
    def __init__(self, corpus, max_history, phase: str, loss_type: str):
        self.corpus = corpus  # reader object reference
        self.phase = phase
        self.max_history = max_history
        self.data = utils.df_to_dict(corpus.data_df[phase])
        self.loss_type = loss_type
        # ↑ DataFrame is not compatible with multi-thread operations
        self.buffer_dict = dict()
        self.buffer = None


    def _prepare(self):
        idx_select = np.array(self.data['his_length']) > 0  # history length must be non-zero
        for key in self.data:
            self.data[key] = np.array(self.data[key])[idx_select]

        if self.buffer:
            for i in tqdm(range(len(self)), leave=False, ncols=100, mininterval=1,
                          desc=('Prepare ' + self.phase)):
                self.buffer_dict[i] = self._get_feed_dict(i)

    def calculate_timid(self, timestamp):
        timetuple = time.localtime(timestamp)
        wid = 0 if timetuple.tm_wday < 5 else 1
        timeid = wid * 24 + timetuple.tm_hour
        return timeid

    def __len__(self):
        if type(self.data) == dict:
            for key in self.data:
                return len(self.data[key])
        return len(self.data)

    def collate_batch(self, feed_dicts):
        feed_dict = dict()
        for key in feed_dicts[0]:
            stack_val = np.array([d[key] for d in feed_dicts])
            feed_dict[key] = torch.from_numpy(stack_val)
        return feed_dict

    def __getitem__(self, item):
        return self.buffer_dict[item] if self.buffer else self._get_feed_dict(item)

    def _get_feed_dict(self, item):
        data_tuple = self.random_sample(item)
        output = self.get_item_samples(data_tuple, item)
        return output

    def get_item_samples(self, data_tuple, item):
        pass

    def create_CAT_Sample(self, Inp_Sample_POI):
        Inp_Sample_CAT = []
        for POI in Inp_Sample_POI:
            Inp_Sample_CAT.append(self.corpus.category_dict[POI])
        return  Inp_Sample_CAT

    def create_CLU_Sample(self, Inp_Sample_POI):
        Inp_Sample_CLU = []
        for POI in Inp_Sample_POI:
            Inp_Sample_CLU.append(self.corpus.cluster_dict[POI])
        return  Inp_Sample_CLU

    def Sample_Neg_Next_POI(self, target_item):
        Neg_Sample = random.choice(self.corpus.items_all)
        while Neg_Sample == target_item:
            Neg_Sample = random.choice(self.corpus.items_all)
        return [Neg_Sample]

    def get_distance(self, lat1, lng1, lat2, lng2):
        # coords_1 = (lat1, lng1)
        # coords_2 = (lat2, lng2)
        #
        # return geopy.distance.vincenty(coords_1, coords_2).km

        # approximate radius of earth in km
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
        # distance = (R * c).astype(np.int)
        # return int(distance/4)
        return distance

    def create_neg_samples_version3(self, samp_POI):
        P = 0.3
        samp_POI_neg = []
        seqlen = len(samp_POI)
        sample_num = int(np.ceil(seqlen * P))
        random_index = random.sample(list(range(seqlen)), sample_num)
        for i in range(seqlen):
            if i in random_index and len(self.corpus.POI_nearest[samp_POI[i]])!=0:
                samp_POI_neg.append(int(np.random.choice(self.corpus.POI_nearest[samp_POI[i]])))
            else:
                samp_POI_neg.append(samp_POI[i])
        return samp_POI_neg

    def All_Neg_Next_POI(self, samp_POI):
        Pos_POI = samp_POI
        Neg_All = [x for x in self.corpus.items_all if x!=Pos_POI]
        # Neg_All_CAT = [self.corpus.category_dict[Neg_Sample] for Neg_Sample in Neg_All]
        # Neg_All_CLU = [self.corpus.cluster_dict[Neg_Sample] for Neg_Sample in Neg_All]
        # return Neg_All, Neg_All_CAT, Neg_All_CLU
        return Neg_All

    def random_sample(self, index):
        target_item = self.data['item_id'][index]
        user_id = self.data['user_id'][index]
        history_items = np.array(self.data['item_his'][index])
        history_times = np.array(self.data['time_his'][index])
        length = self.data['his_length'][index]
        return target_item, user_id, history_items, length, history_times

    def data_analysis(self):
        category_dict = {}
        for i in range(len(self)):
            item = self.data['item_id'][i]
            cate = self.corpus.category_dict[item]
            if cate not in category_dict:
                category_dict[cate] = 0
            else:
                category_dict[cate] += 1




class FinetuneDatasettype1(FinetuneDataset):
    def __init__(self, corpus, max_history, phase: str, loss_type: str):
        super().__init__(corpus, max_history, phase, loss_type)
        if self.loss_type == "Pairwise":
            # self.buffer = self.phase != 'train'
            self.buffer = 0
        elif self.loss_type == "Classification":
            self.buffer = 1
        self._prepare()

    def get_item_samples(self, data_tuple, item):
        target_item, user_id, history_items, length, history_times = data_tuple
        Inp_Sample_POI = list(history_items)
        Inp_Sample_TIM_stamp = list(history_times)
        Inp_Sample_TIM = [self.calculate_timid(timestamp) for timestamp in list(history_times)]
        Inp_Sample_CAT = [self.corpus.category_dict[POI] for POI in Inp_Sample_POI]
        # Inp_Sample_CAT = self.create_CAT_Sample(Inp_Sample_POI)
        # Inp_Sample_CLU = self.create_CLU_Sample(Inp_Sample_POI)
        Pos_Sample_POI = target_item
        Padding_float = [0.0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        Padding = [0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        Inp_Sample_TIM_stamp.extend(Padding_float)
        Inp_Sample_TIM.extend(Padding)
        Inp_Sample_CAT.extend(Padding)
        # Pos_Sample_CAT = self.corpus.category_dict[Pos_Sample_POI]
        # Pos_Sample_CLU = self.corpus.cluster_dict[Pos_Sample_POI]
        if self.phase == 'train':
            ### w.r.t user
            # Neg_Sample_POI, Neg_Sample_CAT, Neg_Sample_CLU = self.Sample_Neg_Next_POI(target_item)
            Neg_Sample_POI = self.Sample_Neg_Next_POI(target_item)
            Inp_Sample_POI_Neg = self.create_neg_samples_version3(Inp_Sample_POI)
            HistoryLen = len(Inp_Sample_POI)
            Inp_Sample_POI.extend(Padding)
            if self.loss_type == "Pairwise":
                Pos_Sample_POI = [Pos_Sample_POI] + Neg_Sample_POI
                # Pos_Sample_POI = Pos_Sample_POI
            elif self.loss_type == "Classification":
                Pos_Sample_POI = Pos_Sample_POI
            Inp_Sample_POI_Neg.extend(Padding)
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_TIM": Inp_Sample_TIM,
                      "Inp_CAT":Inp_Sample_CAT,
                      "Inp_TIMSTP":Inp_Sample_TIM_stamp,
                      "Inp_POI_Neg": Inp_Sample_POI_Neg,
                      "PosNeg_POI": Pos_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        else:
            ### w.r.t current item id
            # Neg_Sample_POI, Neg_Sample_CAT, Neg_Sample_CLU = self.All_Neg_Next_POI(target_item)
            Neg_Sample_POI = self.All_Neg_Next_POI(target_item)
            HistoryLen = len(Inp_Sample_POI)
            Inp_Sample_POI.extend(Padding)
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_CAT": Inp_Sample_CAT,
                      "Inp_TIM": Inp_Sample_TIM,
                      "Inp_TIMSTP": Inp_Sample_TIM_stamp,
                      "PosNeg_POI": [Pos_Sample_POI] + Neg_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        return output





class FinetuneDatasettype2(FinetuneDataset):
    def __init__(self, corpus, max_history, phase: str, loss_type: str):
        super().__init__(corpus, max_history, phase, loss_type)
        if self.loss_type == "Pairwise":
            self.buffer = self.phase != 'train'
        elif self.loss_type == "Classification":
            self.buffer = 1
        self._prepare()

    def get_item_samples(self, data_tuple, item):
        target_item, user_id, history_items, length, history_times = data_tuple
        Inp_Sample_POI = list(history_items)
        Inp_Sample_TIM_stamp = list(history_times)
        Inp_Sample_TIM = [self.calculate_timid(timestamp) for timestamp in list(history_times)]
        Inp_Sample_CAT = [self.corpus.category_dict[POI] for POI in Inp_Sample_POI]
        time_delta = self.delta_time(history_times)
        lat_sequence, lon_sequence, delta_dist = self.delta_dist(history_items)

        if self.loss_type == "Pairwise":
            Pos_Sample_POI = target_item
        elif self.loss_type == "Classification":
            Pos_Sample_POI = self.corpus.subdata_item_map[target_item]

        Padding = [0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        Padding_float = [0.0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        HistoryLen = len(Inp_Sample_POI)
        Inp_Sample_TIM.extend(Padding)
        Inp_Sample_CAT.extend(Padding)
        Inp_Sample_TIM_stamp.extend(Padding_float)
        lat_sequence.extend(Padding_float)
        lon_sequence.extend(Padding_float)
        delta_dist.extend(Padding_float)
        time_delta.extend(Padding_float)

        if self.phase == 'train':
            Neg_Sample_POI = self.Sample_Neg_Next_POI(target_item)
            Inp_Sample_POI_Neg = self.create_neg_samples_version3(Inp_Sample_POI)
            Inp_Sample_POI.extend(Padding)
            Inp_Sample_POI_Neg.extend(Padding)
            if self.loss_type == "Pairwise":
                Pos_Sample_POI = [Pos_Sample_POI] + Neg_Sample_POI
            elif self.loss_type == "Classification":
                Pos_Sample_POI = Pos_Sample_POI
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_CAT": Inp_Sample_CAT,
                      "Inp_TIMSTP": Inp_Sample_TIM_stamp,
                      "Inp_TIM":Inp_Sample_TIM,
                      "Inp_Lat": lat_sequence,
                      "Inp_Lon": lon_sequence,
                      "Inp_DeltaD": delta_dist,
                      "Inp_DeltaT": time_delta,
                      "PosNeg_POI": Pos_Sample_POI,
                      "Inp_POI_Neg": Inp_Sample_POI_Neg,
                      # "PosNeg_POI": [Pos_Sample_POI] + Neg_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        else:
            Inp_Sample_POI.extend(Padding)
            Neg_Sample_POI = self.All_Neg_Next_POI(target_item)
            if self.loss_type == "Pairwise":
                Neg_Sample_POI = Neg_Sample_POI
            elif self.loss_type == "Classification":
                Neg_Sample_POI = [self.corpus.subdata_item_map[target_item] for target_item in Neg_Sample_POI]
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_CAT": Inp_Sample_CAT,
                      "Inp_TIMSTP": Inp_Sample_TIM_stamp,
                      "Inp_TIM": Inp_Sample_TIM,
                      "Inp_Lat": lat_sequence,
                      "Inp_Lon": lon_sequence,
                      "Inp_DeltaD": delta_dist,
                      "Inp_DeltaT": time_delta,
                      # "PosNeg_POI": Pos_Sample_POI,
                      "PosNeg_POI": [Pos_Sample_POI] + Neg_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        return output



    def delta_time(self, sequence_time):
        time_delta = [0]
        a = np.array(sequence_time[1:])
        b = np.array(sequence_time[:-1])
        c = list(a - b)
        time_delta.extend(c)
        return time_delta

    def delta_dist(self, sequence_POI):
        # lat_sequence = [self.corpus.coordinate_dict[sequence_POI[0]][0]]
        # lon_sequence = [self.corpus.coordinate_dict[sequence_POI[0]][1]]
        lat_sequence = [self.corpus.coordinate_dict[POI][0] for POI in sequence_POI]
        lon_sequence = [self.corpus.coordinate_dict[POI][1] for POI in sequence_POI]
        a1 = np.expand_dims(np.array(lat_sequence[1:]), axis=-1)
        b1 = np.expand_dims(np.array(lon_sequence[1:]), axis=-1)
        a2 = np.expand_dims(np.array(lat_sequence[:-1]), axis=-1)
        b2 = np.expand_dims(np.array(lon_sequence[:-1]), axis=-1)
        a = np.concatenate((a1, b1), axis=-1)
        b = np.concatenate((a2, b2), axis=-1)
        c = list(np.sqrt(((a-b) ** 2).sum(-1)))
        delta_dist = [0]
        delta_dist.extend(c)
        return lat_sequence, lon_sequence, delta_dist

###mainly for the fucking rubbish STAN
class FinetuneDatasettype3(FinetuneDataset):
    def __init__(self, corpus, max_history, phase: str, loss_type: str):
        super().__init__(corpus, max_history, phase, loss_type)
        if self.loss_type == "Pairwise":
            # self.buffer = self.phase != 'train'
            self.buffer = 0
        elif self.loss_type == "Classification":
            self.buffer = 0
        self._prepare()
        if self.phase == 'train':
            # self.max_dis = 62.1543374282927
            # self.max_tim = 8210
            # self.min_dis = 0
            # self.min_tim = 0
            self.max_dis = 0
            self.max_tim = 0
            self.min_dis = 9999999
            self.min_tim = 9999999
            self.calculate_ex()





    def calculate_ex(self):
        for i in tqdm(range(len(self)), leave=False, ncols=100, mininterval=1,
                      desc=('Prepare ' + self.phase)):
            data_tuple = self.random_sample(i)
            self.get_ex_samples(data_tuple, i)


    def get_ex_samples(self,  data_tuple, item):
        target_item, user_id, history_items, length, history_times, target_time = data_tuple
        Inp_Sample_POI = list(history_items)
        Inp_Sample_TIM_stamp = list(history_times)
        Inp_Sample_TIM_hour = np.array([int(timestamp / (60 * 60)) for timestamp in Inp_Sample_TIM_stamp])
        STAN_mat1t = np.abs(np.repeat(np.expand_dims(Inp_Sample_TIM_hour, -1), length, axis=-1) - Inp_Sample_TIM_hour)
        min_tim = STAN_mat1t.min()
        max_tim = STAN_mat1t.max()

        Lat_mat = np.array([self.corpus.coordinate_dict[POI][0] for POI in Inp_Sample_POI])
        Lon_mat = np.array([self.corpus.coordinate_dict[POI][1] for POI in Inp_Sample_POI])
        STAN_mat1d = self.distance_mat(Lat_mat, Lon_mat, length)
        min_dis = STAN_mat1d.min()
        max_dis = STAN_mat1d.max()

        if min_tim<self.min_tim:
            self.min_tim = min_tim
        if min_dis<self.min_dis:
            self.min_dis = min_dis
        if max_tim>self.max_tim:
            self.max_tim = max_tim
        if max_dis>self.max_dis:
            self.max_dis = max_dis


    def get_item_samples(self, data_tuple, item):
        target_item, user_id, history_items, length, history_times, target_time = data_tuple
        Inp_Sample_POI = list(history_items)
        Inp_Sample_TIM_stamp = list(history_times)
        Inp_Sample_TIM_hour = np.array([int(timestamp/(60 * 60)) for timestamp in Inp_Sample_TIM_stamp])
        STAN_mat1t = np.abs(np.repeat(np.expand_dims(Inp_Sample_TIM_hour, -1), length, axis=-1) - Inp_Sample_TIM_hour)
        Lat_mat = np.array([self.corpus.coordinate_dict[POI][0] for POI in Inp_Sample_POI])
        Lon_mat = np.array([self.corpus.coordinate_dict[POI][1] for POI in Inp_Sample_POI])
        STAN_mat1d = self.distance_mat(Lat_mat, Lon_mat, length)
        STAN_mat1 = np.concatenate((np.expand_dims(STAN_mat1d, -1), np.expand_dims(STAN_mat1t, -1)), axis=-1).astype(int)
        STAN_mat1_full = np.zeros((self.max_history,self.max_history, 2), dtype=np.int)
        STAN_mat1_full[:length, :length] = STAN_mat1
        SATN_mat2t = list(int(target_time/(60 * 60)) - Inp_Sample_TIM_hour)
        Inp_Sample_CAT = [self.corpus.category_dict[POI] for POI in Inp_Sample_POI]

        Pos_Sample_POI = target_item
        Padding = [0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        Padding_float = [0.0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        Inp_Sample_TIM_stamp.extend(Padding_float)
        SATN_mat2t.extend(Padding)
        HistoryLen = len(Inp_Sample_POI)
        Inp_Sample_TIM_hour = list(Inp_Sample_TIM_hour)
        Inp_Sample_TIM_hour.extend(Padding)
        Inp_Sample_CAT.extend(Padding)
        if self.phase == 'train':
            Neg_Sample_POI = self.Sample_Neg_Next_POI(target_item)
            Pos_Sample_POI = [Pos_Sample_POI] + Neg_Sample_POI
            Candidate_Lat = np.array([self.corpus.coordinate_dict[POI][0] for POI in Pos_Sample_POI])
            Candidate_Lon = np.array([self.corpus.coordinate_dict[POI][1] for POI in Pos_Sample_POI])
            Candidate_dist = self.get_distance(np.expand_dims(Lat_mat, -1), np.expand_dims(Lon_mat, -1),
                                               np.expand_dims(Candidate_Lat, 0), np.expand_dims(Candidate_Lon, 0))
            STAN_mat2_full = np.zeros((self.max_history, 2), dtype=np.float)
            STAN_mat2_full[:length, :] = Candidate_dist


            Inp_Sample_POI_Neg = self.create_neg_samples_version3(Inp_Sample_POI)

            Inp_Sample_POI.extend(Padding)
            Inp_Sample_POI_Neg.extend(Padding)

            ### STAN cannot be classification loss function
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_CAT": Inp_Sample_CAT,
                      "Inp_TIM": Inp_Sample_TIM_hour,
                      "Inp_TIMSTP": Inp_Sample_TIM_stamp,
                      "Inp_User": user_id,
                      "Inp_MAT2t": SATN_mat2t,
                      "Inp_MAT1": STAN_mat1_full,
                      "Inp_MAT2s":STAN_mat2_full,
                      "PosNeg_POI": Pos_Sample_POI,
                      "Inp_POI_Neg": Inp_Sample_POI_Neg,
                      # "PosNeg_POI": [Pos_Sample_POI] + Neg_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        else:
            Inp_Sample_POI.extend(Padding)
            Neg_Sample_POI = self.All_Neg_Next_POI(target_item)
            Pos_Sample_POI = [Pos_Sample_POI] + Neg_Sample_POI
            Candidate_Lat = np.array([self.corpus.coordinate_dict[POI][0] for POI in Pos_Sample_POI])
            Candidate_Lon = np.array([self.corpus.coordinate_dict[POI][1] for POI in Pos_Sample_POI])
            Candidate_dist = self.get_distance(np.expand_dims(Lat_mat, -1), np.expand_dims(Lon_mat, -1),
                                               np.expand_dims(Candidate_Lat, 0), np.expand_dims(Candidate_Lon, 0))
            STAN_mat2_full = np.zeros((self.max_history, len(Pos_Sample_POI)), dtype=np.int)
            STAN_mat2_full[:length, :] = Candidate_dist
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_CAT": Inp_Sample_CAT,
                      "Inp_TIM": Inp_Sample_TIM_hour,
                      "Inp_TIMSTP": Inp_Sample_TIM_stamp,
                      "Inp_User": user_id,
                      "Inp_MAT2t": SATN_mat2t,
                      "Inp_MAT1": STAN_mat1_full,
                      "Inp_MAT2s":STAN_mat2_full,
                      "PosNeg_POI": Pos_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        return output

    def distance_mat(self, Lat, Lon, length):
        Lat_mat1 = np.repeat(np.expand_dims(Lat, -1), length, axis=-1)
        Lon_mat1 = np.repeat(np.expand_dims(Lon, -1), length, axis=-1)
        Lat_mat2 = np.repeat(np.expand_dims(Lat, 0), length, axis=0)
        Lon_mat2 = np.repeat(np.expand_dims(Lon, 0), length, axis=0)
        STAN_mat1d = self.get_distance(Lat_mat1, Lon_mat1, Lat_mat2, Lon_mat2)
        return STAN_mat1d


    def delta_time(self, sequence_time):
        time_delta = [0]
        a = np.array(sequence_time[1:])
        b = np.array(sequence_time[:-1])
        c = list(a - b)
        time_delta.extend(c)
        return time_delta

    def delta_dist(self, sequence_POI):
        # lat_sequence = [self.corpus.coordinate_dict[sequence_POI[0]][0]]
        # lon_sequence = [self.corpus.coordinate_dict[sequence_POI[0]][1]]
        lat_sequence = [self.corpus.coordinate_dict[POI][0] for POI in sequence_POI]
        lon_sequence = [self.corpus.coordinate_dict[POI][1] for POI in sequence_POI]
        a1 = np.expand_dims(np.array(lat_sequence[1:]), axis=-1)
        b1 = np.expand_dims(np.array(lon_sequence[1:]), axis=-1)
        a2 = np.expand_dims(np.array(lat_sequence[:-1]), axis=-1)
        b2 = np.expand_dims(np.array(lon_sequence[:-1]), axis=-1)
        a = np.concatenate((a1, b1), axis=-1)
        b = np.concatenate((a2, b2), axis=-1)
        c = list(np.sqrt(((a-b) ** 2).sum(-1)))
        delta_dist = [0]
        delta_dist.extend(c)
        return lat_sequence, lon_sequence, delta_dist

    def random_sample(self, index):
        target_item = self.data['item_id'][index]
        target_time = self.data['time'][index]
        user_id = self.data['user_id'][index]
        history_items = np.array(self.data['item_his'][index])
        history_times = np.array(self.data['time_his'][index])
        length = self.data['his_length'][index]
        return target_item, user_id, history_items, length, history_times, target_time

class FinetuneDatasettype4(FinetuneDataset):
    def __init__(self, corpus, max_history, phase: str, loss_type: str):
        super().__init__(corpus, max_history, phase, loss_type)
        if self.loss_type == "Pairwise":
            self.buffer = self.phase != 'train'
        elif self.loss_type == "Classification":
            self.buffer = 1
        self._prepare()

    def get_item_samples(self, data_tuple, item):
        target_item, user_id, history_items, length, history_times = data_tuple
        Inp_Sample_POI = list(history_items)
        Inp_Sample_POI_map = [self.corpus.subdata_item_map[target_item] for target_item in Inp_Sample_POI]
        Inp_Sample_TIM_stamp = list(history_times)
        Inp_Sample_TIM_interval = [Inp_Sample_TIM_stamp[-1] - timestamp for timestamp in Inp_Sample_TIM_stamp]
        Inp_Sample_TIM = [self.calculate_timid(timestamp) for timestamp in list(history_times)]
        # Inp_Sample_CAT = [self.corpus.category_dict[POI] for POI in Inp_Sample_POI]
        # time_delta = self.delta_time(history_times)
        Inp_Sample_DIS_interval = self.delta_dist(history_items)

        if self.loss_type == "Pairwise":
            Pos_Sample_POI = target_item
        elif self.loss_type == "Classification":
            Pos_Sample_POI = self.corpus.subdata_item_map[target_item]

        Padding = [0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        Padding_float = [0.0 for _ in range(self.max_history - len(Inp_Sample_POI))]
        HistoryLen = len(Inp_Sample_POI)
        Inp_Sample_TIM_interval.extend(Padding_float)
        Inp_Sample_TIM.extend(Padding)
        Inp_Sample_TIM_stamp.extend(Padding_float)
        Inp_Sample_DIS_interval.extend(Padding_float)
        # delta_dist.extend(Padding_float)
        # time_delta.extend(Padding_float)

        if self.phase == 'train':
            Neg_Sample_POI = self.Sample_Neg_Next_POI(target_item)
            Inp_Sample_POI_Neg = self.create_neg_samples_version3(Inp_Sample_POI)
            Inp_Sample_POI_Neg_map = [self.corpus.subdata_item_map[target_item] for target_item in Inp_Sample_POI_Neg]
            Inp_Sample_POI.extend(Padding)
            Inp_Sample_POI_Neg.extend(Padding)
            Inp_Sample_POI_map.extend(Padding)
            Inp_Sample_POI_Neg_map.extend(Padding)
            if self.loss_type == "Pairwise":
                Pos_Sample_POI = [Pos_Sample_POI] + Neg_Sample_POI
            elif self.loss_type == "Classification":
                Pos_Sample_POI = Pos_Sample_POI
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_POI_map": Inp_Sample_POI_map,
                      "Inp_TIM": Inp_Sample_TIM,
                      "Inp_TIMSTP": Inp_Sample_TIM_stamp,
                      "Inp_TIM_Interval": Inp_Sample_TIM_interval,
                      "Inp_DIS_Interval": Inp_Sample_DIS_interval,
                      "PosNeg_POI": Pos_Sample_POI,
                      "Inp_POI_Neg": Inp_Sample_POI_Neg,
                      "Inp_POI_Neg_map": Inp_Sample_POI_Neg_map,
                      "User_ID": user_id,
                      # "PosNeg_POI": [Pos_Sample_POI] + Neg_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        else:
            Inp_Sample_POI.extend(Padding)
            Inp_Sample_POI_map.extend(Padding)
            Neg_Sample_POI = self.All_Neg_Next_POI(target_item)
            if self.loss_type == "Pairwise":
                Neg_Sample_POI = Neg_Sample_POI
            elif self.loss_type == "Classification":
                Neg_Sample_POI = [self.corpus.subdata_item_map[target_item] for target_item in Neg_Sample_POI]
            output = {"Inp_POI": Inp_Sample_POI,
                      "Inp_POI_map": Inp_Sample_POI_map,
                      "Inp_TIM": Inp_Sample_TIM,
                      "Inp_TIM_Interval": Inp_Sample_TIM_interval,
                      "Inp_DIS_Interval": Inp_Sample_DIS_interval,
                      "Inp_TIMSTP": Inp_Sample_TIM_stamp,
                      # "PosNeg_POI": Pos_Sample_POI,
                      "User_ID": user_id,
                      "PosNeg_POI": [Pos_Sample_POI] + Neg_Sample_POI,
                      "HistoryLen": HistoryLen,
                      "item": item}
        return output

    def FlashBack_timeid(self, timestamp):
        timetuple = time.localtime(timestamp)
        wid = timetuple.tm_wday
        timeid = wid * 24 + timetuple.tm_hour
        return timeid


    def delta_time(self, sequence_time):
        time_delta = [0]
        a = np.array(sequence_time[1:])
        b = np.array(sequence_time[:-1])
        c = list(a - b)
        time_delta.extend(c)
        return time_delta

    def delta_dist(self, sequence_POI):
        # lat_sequence = [self.corpus.coordinate_dict[sequence_POI[0]][0]]
        # lon_sequence = [self.corpus.coordinate_dict[sequence_POI[0]][1]]
        lat_sequence = [self.corpus.coordinate_dict[POI][0] for POI in sequence_POI]
        lon_sequence = [self.corpus.coordinate_dict[POI][1] for POI in sequence_POI]
        a1 = np.expand_dims(np.array(lat_sequence), axis=-1)
        b1 = np.expand_dims(np.array(lon_sequence), axis=-1)
        a = np.concatenate((a1, b1), axis=-1)
        b = np.array([lat_sequence[-1], lon_sequence[-1]])
        c = np.linalg.norm(b-a, axis=-1)
        return list(c)


class CTLEDataset(Dataset):
    def __init__(self, corpus_path, vocab, sequence_len, encoding="utf-8", train = True):
        self.history_len = sequence_len
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_epoch = 0
        self.train_flag = train
        self.MAX_Delta_Day = 30
        self.MAX_Delta_Dis = 50
        self.max_sequence_len = 0
        self.datas = []
        self.sequences_POI = []
        self.sequences_TIM = []
        self.ALL_POIs = []
        self.times = []
        time_path = corpus_path.replace('corpus', 'time')
        if self.train_flag:
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


        else:
            category_path = corpus_path.replace('test_corpus', 'category')
            with open(category_path, "rb") as f:
                self.category_dict = pickle.load(f)
            coordinate_path = corpus_path.replace('test_corpus', 'coordinate')
            with open(coordinate_path, "rb") as f:
                self.coordinate_dict = pickle.load(f)
            city_path = corpus_path.replace('test_corpus', 'city')
            with open(city_path, "rb") as f:
                self.city_dict = pickle.load(f)



        self.category_nums = max(set(self.category_dict.values())) + 1
        self.ALL_CATs = list(set(self.category_dict.values()))
        self.ALL_POIs = list(self.category_dict.keys())
        with open(corpus_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_POI.append([int(POI) for POI in sentences[0].split()])
        with open(time_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_TIM.append([float(TIM) for TIM in sentences[0].split()])

    def gen_sequence(self, include_delta=False):
        datas = []
        ####process sequences, different between train and test
        for i, sequence in enumerate(self.sequences_POI):
            user_index = i
            sequence_time = self.sequences_TIM[i]
            sequence_weekdays = [time.localtime(timestamp).tm_wday for timestamp in sequence_time]

            seqlen = len(sequence)
            if seqlen <= self.history_len:
                if include_delta:
                    time_delta = self.delta_time(sequence_time)
                    lat_sequence, lon_sequence, delta_dist = self.delta_dist(sequence)
                    datas.append([user_index, sequence, sequence_weekdays, sequence_time, len(sequence),
                                       time_delta, delta_dist, lat_sequence, lon_sequence])
                else:
                    datas.append([user_index, sequence, sequence_weekdays, sequence_time, len(sequence)])

            else:
                for ind in range(seqlen - self.history_len + 1):
                    begin_ind = ind
                    end_ind = ind + self.history_len
                    if include_delta:
                        time_delta = self.delta_time(sequence_time[begin_ind:end_ind])
                        lat_sequence, lon_sequence, delta_dist = self.delta_dist(sequence[begin_ind:end_ind])
                        datas.append([user_index, sequence[begin_ind:end_ind], sequence_weekdays[begin_ind:end_ind],
                                           sequence_time[begin_ind:end_ind], len(sequence[begin_ind:end_ind]),
                                           time_delta, delta_dist, lat_sequence, lon_sequence])
                    else:
                        datas.append([user_index, sequence[begin_ind:end_ind], sequence_weekdays[begin_ind:end_ind],
                                           sequence_time[begin_ind:end_ind], len(sequence[begin_ind:end_ind])])
        return datas

    def delta_time(self, sequence_time):
        time_delta = [0]
        for i in range(len(sequence_time) - 1):
            time_delta.append(sequence_time[i + 1] - sequence_time[i])
        return time_delta

    def delta_dist(self, sequence_POI):
        lat_sequence = [self.coordinate_dict[sequence_POI[0]][0]]
        lon_sequence = [self.coordinate_dict[sequence_POI[0]][1]]
        delta_dist = [0]
        for i in range(len(sequence_POI) - 1):
            lat_sequence.append(self.coordinate_dict[sequence_POI[i+1]][0])
            lon_sequence.append(self.coordinate_dict[sequence_POI[i+1]][1])
            coor_delta = np.array([self.coordinate_dict[sequence_POI[i+1]][0], self.coordinate_dict[sequence_POI[i+1]][1]]) - \
                         np.array([self.coordinate_dict[sequence_POI[i]][0], self.coordinate_dict[sequence_POI[i]][1]])
            delta_dist.append(np.sqrt((coor_delta ** 2).sum(-1)))
        return lat_sequence, lon_sequence, delta_dist

class CCLEDataset(Dataset):
    def __init__(self, corpus_path, vocab, sequence_len, encoding="utf-8", train = True):
        self.history_len = sequence_len
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_epoch = 0
        self.train_flag = train
        self.MAX_Delta_Day = 30
        self.MAX_Delta_Dis = 50
        self.max_sequence_len = 0
        self.datas = []
        self.sequences_POI = []
        self.sequences_TIM = []
        self.ALL_POIs = []
        self.times = []
        time_path = corpus_path.replace('corpus', 'time')
        if self.train_flag:
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


        else:
            category_path = corpus_path.replace('test_corpus', 'category')
            with open(category_path, "rb") as f:
                self.category_dict = pickle.load(f)
            coordinate_path = corpus_path.replace('test_corpus', 'coordinate')
            with open(coordinate_path, "rb") as f:
                self.coordinate_dict = pickle.load(f)
            city_path = corpus_path.replace('test_corpus', 'city')
            with open(city_path, "rb") as f:
                self.city_dict = pickle.load(f)



        self.category_nums = max(set(self.category_dict.values())) + 1
        self.ALL_CATs = list(set(self.category_dict.values()))
        self.ALL_POIs = list(self.category_dict.keys())
        with open(corpus_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_POI.append([int(POI) for POI in sentences[0].split()])
        with open(time_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_TIM.append([float(TIM) for TIM in sentences[0].split()])

    def gen_sequence(self, include_delta=False):
        datas = []
        ####process sequences, different between train and test
        for i, sequence in enumerate(self.sequences_POI):
            user_index = i
            sequence_time = self.sequences_TIM[i]
            sequence_weekdays = [time.localtime(timestamp).tm_wday for timestamp in sequence_time]

            seqlen = len(sequence)
            if seqlen <= self.history_len:
                if include_delta:
                    time_delta = self.delta_time(sequence_time)
                    lat_sequence, lon_sequence, delta_dist = self.delta_dist(sequence)
                    datas.append([user_index, sequence, sequence_weekdays, sequence_time, len(sequence),
                                       time_delta, delta_dist, lat_sequence, lon_sequence])
                else:
                    datas.append([user_index, sequence, sequence_weekdays, sequence_time, [self.category_dict[poi] for poi in sequence], len(sequence)])

            else:
                for ind in range(seqlen - self.history_len + 1):
                    begin_ind = ind
                    end_ind = ind + self.history_len
                    if include_delta:
                        time_delta = self.delta_time(sequence_time[begin_ind:end_ind])
                        lat_sequence, lon_sequence, delta_dist = self.delta_dist(sequence[begin_ind:end_ind])
                        datas.append([user_index, sequence[begin_ind:end_ind], sequence_weekdays[begin_ind:end_ind],
                                           sequence_time[begin_ind:end_ind], len(sequence[begin_ind:end_ind]),
                                           time_delta, delta_dist, lat_sequence, lon_sequence])
                    else:
                        datas.append([user_index, sequence[begin_ind:end_ind], sequence_weekdays[begin_ind:end_ind], sequence_time[begin_ind:end_ind],
                                      [self.category_dict[poi] for poi in sequence[begin_ind:end_ind]], len(sequence[begin_ind:end_ind])])
        return datas

    def delta_time(self, sequence_time):
        time_delta = [0]
        for i in range(len(sequence_time) - 1):
            time_delta.append(sequence_time[i + 1] - sequence_time[i])
        return time_delta

    def delta_dist(self, sequence_POI):
        lat_sequence = [self.coordinate_dict[sequence_POI[0]][0]]
        lon_sequence = [self.coordinate_dict[sequence_POI[0]][1]]
        delta_dist = [0]
        for i in range(len(sequence_POI) - 1):
            lat_sequence.append(self.coordinate_dict[sequence_POI[i+1]][0])
            lon_sequence.append(self.coordinate_dict[sequence_POI[i+1]][1])
            coor_delta = np.array([self.coordinate_dict[sequence_POI[i+1]][0], self.coordinate_dict[sequence_POI[i+1]][1]]) - \
                         np.array([self.coordinate_dict[sequence_POI[i]][0], self.coordinate_dict[sequence_POI[i]][1]])
            delta_dist.append(np.sqrt((coor_delta ** 2).sum(-1)))
        return lat_sequence, lon_sequence, delta_dist

class W2VData:
    def __init__(self, sentences, indi_context):
        self.indi_context = indi_context
        self.word_freq = self.gen_token_freq(sentences)  # (num_vocab, 2)

    def gen_token_freq(self, sentences):
        freq = Counter()
        for sentence in sentences:
            freq.update(sentence)
        freq = np.array(sorted(freq.items()))
        return freq

class HuffmanNode:
    """
    A node in the Huffman tree.
    """
    def __init__(self, id, frequency):
        """
        :param id: index of word (leaf nodes) or inner nodes.
        :param frequency: frequency of word.
        """
        self.id = id
        self.frequency = frequency

        self.left = None
        self.right = None
        self.father = None
        self.huffman_code = []
        self.path = []  # (path from root node to leaf node)

    def __str__(self):
        return 'HuffmanNode#{},freq{}'.format(self.id, self.frequency)

class HuffmanTree:
    """
    Huffman Tree class used for Hierarchical Softmax calculation.
    """
    def __init__(self, freq_array):
        """
        :param freq_array: numpy array containing all words' frequencies, format {id: frequency}.
        """
        self.num_words = freq_array.shape[0]
        self.id2code = {}
        self.id2path = {}
        self.id2pos = {}
        self.id2neg = {}
        self.root = None  # Root node of this tree.
        self.num_inner_nodes = 0  # Records the number of inner nodes of this tree.

        unmerged_node_list = [HuffmanNode(id, frequency) for id, frequency in freq_array]
        self.tree = {node.id: node for node in unmerged_node_list}
        self.id_offset = max(self.tree.keys())  # Records the starting-off ID of this tree.
        # Because the ID of leaf nodes will not be needed during calculation,
        # you can minus this value to all inner nodes' IDs to save some space in output embeddings.

        self._offset = self.id_offset
        self._build_tree(unmerged_node_list)
        self._gen_path()
        self._get_all_pos_neg()

    def _merge_node(self, node1: HuffmanNode, node2: HuffmanNode):
        """
        Merge two nodes into one, adding their frequencies.
        """
        sum_freq = node1.frequency + node2.frequency
        self._offset += 1
        mid_node_id = self._offset
        father_node = HuffmanNode(mid_node_id, sum_freq)
        if node1.frequency >= node2.frequency:
            father_node.left, father_node.right = node1, node2
        else:
            father_node.left, father_node.right = node2, node1
        self.tree[mid_node_id] = father_node
        self.num_inner_nodes += 1
        return father_node

    def _build_tree(self, node_list):
        while len(node_list) > 1:
            i1, i2 = 0, 1
            if node_list[i2].frequency < node_list[i1].frequency:
                i1, i2 = i2, i1
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        i1, i2 = i2, i1
            father_node = self._merge_node(node_list[i1], node_list[i2])
            assert not i1 == i2
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            else:
                node_list.pop(i1)
                node_list.pop(i2)
            node_list.insert(0, father_node)
        self.root = node_list[0]

    def _gen_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            while node.left or node.right:
                code = node.huffman_code
                path = node.path
                node.left.huffman_code = code + [1]
                node.right.huffman_code = code + [0]
                node.left.path = path + [node.id]
                node.right.path = path + [node.id]
                stack.append(node.right)
                node = node.left
            id = node.id
            code = node.huffman_code
            path = node.path
            self.tree[id].huffman_code, self.tree[id].path = code, path
            self.id2code[id], self.id2path[id] = code, path

    def _get_all_pos_neg(self):
        for id in self.id2code.keys():
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.tree[id].huffman_code):
                if code == 1:
                    pos_id.append(self.tree[id].path[i] - self.id_offset)  # This will make the generated inner node IDs starting from 1.
                else:
                    neg_id.append(self.tree[id].path[i] - self.id_offset)
            self.id2pos[id] = pos_id
            self.id2neg[id] = neg_id
import math
class TALEDataset(W2VData):
    def __init__(self, corpus_path, vocab, sequence_len, slice_len, influ_len, indi_context, encoding="utf-8"):
        """
        :param sentences: sequences of location visiting records.
        :param timestamps: sequences of location visited timestamp (second), corresponding to sentences.
        :param slice_len: length of one time slice, in minute.
        :param influ_len: length of influence span, in minute.
        :param indi_context:
        """
        self.history_len = sequence_len
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_epoch = 0
        self.MAX_Delta_Day = 30
        self.MAX_Delta_Dis = 50
        self.max_sequence_len = 0
        self.datas = []
        self.sequences_POI = []
        self.sequences_TIM = []
        self.ALL_POIs = []
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

        self.category_nums = max(set(self.category_dict.values())) + 1
        self.ALL_CATs = list(set(self.category_dict.values()))
        self.ALL_POIs = list(self.category_dict.keys())
        with open(corpus_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_POI.append([int(POI) for POI in sentences[0].split()])
        with open(time_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_TIM.append([float(TIM) for TIM in sentences[0].split()])

        sentences = self.sequences_POI
        timestamps = self.sequences_TIM



        super().__init__(sentences, indi_context)
        self.sentences = self.sequences_POI
        self.timestamps = self.sequences_TIM

        self.visit2slice = {}  # Map a binary tuple (poi_index, timestamp) to slice indices.
        self.visit2prop = {}  # Map a binary tuple (poi_index, timestamp) to proportion corresponding to time slices.
        slice2poi = {}  # Record the included POIs for every time slice.

        for sentence, timestamp in zip(sentences, timestamps):
            for poi_index, visited_second in zip(sentence, timestamp):
                slice, prop = self.gen_all_slots(visited_second / 60, slice_len, influ_len)
                for s in slice:
                    slice2poi[s] = slice2poi.get(s, []) + [poi_index]
                int_minute = math.floor(visited_second % (24 * 60 * 60) / 60)
                self.visit2slice[(poi_index, int_minute)] = slice
                self.visit2prop[(poi_index, int_minute)] = prop

        self.slice2tree = {}  # Save all root node of temporal trees.
        self.slice2offset = {}  # Record node ID offset of temporal trees.
        _total_offset = 0
        idx = 0
        for slice_index, poi_list in slice2poi.items():
            # Generate one Huffman Tree for every temporal slot.
            poi_freq = np.array(sorted(Counter(poi_list).items()))
            huffman_tree = HuffmanTree(poi_freq)
            self.slice2tree[slice_index] = huffman_tree
            self.slice2offset[slice_index] = _total_offset
            _total_offset += huffman_tree.num_inner_nodes
        self.num_inner_nodes = _total_offset + 1

    def gen_path_pairs(self, window_size):
        path_pairs = []
        for sentence, timestamp in zip(self.sentences, self.timestamps):
            for i in range(len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i+window_size]
                visit = (target, math.floor(timestamp[i+window_size] % (24 * 60 * 60) / 60))
                slice = self.visit2slice[visit]
                prop = self.visit2prop[visit]
                huffman_pos = [(np.array(self.slice2tree[s].id2pos[target]) + self.slice2offset[s]).tolist() for s in slice]
                huffman_neg = [(np.array(self.slice2tree[s].id2neg[target]) + self.slice2offset[s]).tolist() for s in slice]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                if self.indi_context:
                    path_pairs += [[[c], huffman_pos, huffman_neg, slice, prop] for c in context]
                else:
                    path_pairs.append([context, huffman_pos, huffman_neg, slice, prop])
        return path_pairs

    def gen_all_slots(self, minute, time_slice_length, influence_span_length):
        """
        :param minute: UTC timestamp in minute.
        :param time_slice_length: length of one slot in seconds.
        :param influence_span_length: length of influence span in seconds.
        """

        def _cal_slice(x):
            return int((x % (24 * 60)) / time_slice_length)

        if influence_span_length == 0:
            slices, props = [_cal_slice(minute)], [1.0]

        else:
            minute_floors = list({minute - influence_span_length / 2, minute + influence_span_length / 2} |
                                 set(range((int(
                                     (minute - influence_span_length / 2) / time_slice_length) + 1) * time_slice_length,
                                           int(minute + influence_span_length / 2), time_slice_length)))
            minute_floors.sort()

            slices = [_cal_slice(time_minute) for time_minute in minute_floors[:-1]]
            props = [(minute_floors[index + 1] - minute_floors[index]) / influence_span_length
                     for index in range(len(minute_floors) - 1)]
        return slices, props

class CALEDataset(W2VData):
    def __init__(self, corpus_path, vocab, sequence_len, slice_len, influ_len, indi_context, encoding="utf-8"):
        """
        :param sentences: sequences of location visiting records.
        :param timestamps: sequences of location visited timestamp (second), corresponding to sentences.
        :param slice_len: length of one time slice, in minute.
        :param influ_len: length of influence span, in minute.
        :param indi_context:
        """
        self.history_len = sequence_len
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_epoch = 0
        self.MAX_Delta_Day = 30
        self.MAX_Delta_Dis = 50
        self.max_sequence_len = 0
        self.datas = []
        self.sequences_POI = []
        self.sequences_TIM = []
        self.ALL_POIs = []
        self.sequences_cat = []
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

        self.category_nums = max(set(self.category_dict.values())) + 1
        self.ALL_CATs = list(set(self.category_dict.values()))
        self.ALL_POIs = list(self.category_dict.keys())

        with open(corpus_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_POI.append([int(POI) for POI in sentences[0].split()])
                self.sequences_cat.append([self.category_dict[int(POI)] for POI in sentences[0].split()])
        with open(time_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_TIM.append([float(TIM) for TIM in sentences[0].split()])

        sentences = self.sequences_POI
        timestamps = self.sequences_TIM
        categorys = self.sequences_cat


        super().__init__(sentences, indi_context)
        self.sentences = self.sequences_POI
        self.timestamps = self.sequences_TIM
        self.categorys = self.sequences_cat
        self.visit2slice = {}  # Map a binary tuple (poi_index, timestamp) to slice indices.
        self.visit2prop = {}  # Map a binary tuple (poi_index, timestamp) to proportion corresponding to time slices.
        slice2poi = {}  # Record the included POIs for every time slice.

        for sentence, category_sentence in zip(sentences, categorys):
            for poi_index, visited_category in zip(sentence, category_sentence):
                slice, prop = self.gen_all_slots(visited_category, slice_len, 0)
                for s in slice:
                    slice2poi[s] = slice2poi.get(s, []) + [poi_index]
                int_minute = visited_category
                self.visit2slice[(poi_index, int_minute)] = slice
                self.visit2prop[(poi_index, int_minute)] = prop

        self.slice2tree = {}  # Save all root node of temporal trees.
        self.slice2offset = {}  # Record node ID offset of temporal trees.
        _total_offset = 0
        idx = 0
        for slice_index, poi_list in slice2poi.items():
            # Generate one Huffman Tree for every temporal slot.
            poi_freq = np.array(sorted(Counter(poi_list).items()))
            huffman_tree = HuffmanTree(poi_freq)
            self.slice2tree[slice_index] = huffman_tree
            self.slice2offset[slice_index] = _total_offset
            _total_offset += huffman_tree.num_inner_nodes
        self.num_inner_nodes = _total_offset + 1

    def gen_path_pairs(self, window_size):
        path_pairs = []
        for sentence, category_sentence in zip(self.sentences, self.categorys):
            for i in range(len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i+window_size]
                visit = (target, category_sentence[i+window_size])
                slice = self.visit2slice[visit]
                prop = self.visit2prop[visit]
                huffman_pos = [(np.array(self.slice2tree[s].id2pos[target]) + self.slice2offset[s]).tolist() for s in slice]
                huffman_neg = [(np.array(self.slice2tree[s].id2neg[target]) + self.slice2offset[s]).tolist() for s in slice]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                if self.indi_context:
                    path_pairs += [[[c], huffman_pos, huffman_neg, slice, prop] for c in context]
                else:
                    path_pairs.append([context, huffman_pos, huffman_neg, slice, prop])
        return path_pairs

    def gen_all_slots(self, minute, time_slice_length, influence_span_length):
        """
        :param minute: UTC timestamp in minute.
        :param time_slice_length: length of one slot in seconds.
        :param influence_span_length: length of influence span in seconds.
        """

        def _cal_slice(x):
            return int((x % (24 * 60)) / time_slice_length)

        if influence_span_length == 0:
            slices, props = [minute], [1.0]

        else:
            minute_floors = list({minute - influence_span_length / 2, minute + influence_span_length / 2} |
                                 set(range((int(
                                     (minute - influence_span_length / 2) / time_slice_length) + 1) * time_slice_length,
                                           int(minute + influence_span_length / 2), time_slice_length)))
            minute_floors.sort()

            slices = [_cal_slice(time_minute) for time_minute in minute_floors[:-1]]
            props = [(minute_floors[index + 1] - minute_floors[index]) / influence_span_length
                     for index in range(len(minute_floors) - 1)]
        return slices, props

class CLUEDataset(Dataset):
    def __init__(self, corpus_path, vocab, sequence_len, encoding="utf-8", train = True):
        self.history_len = sequence_len
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
        with open(corpus_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                sentences = line.split('\t')
                self.sequences_POI.append([int(POI) for POI in sentences[0].split()])

        for i, sequence in enumerate(self.sequences_POI):
            seqlen = len(sequence)

            for ind in range(seqlen):
                begin_ind = max(ind-self.history_len + 1, 0)
                seq_POIs = sequence[begin_ind:ind + 1]
                self.datas.append(seq_POIs)
        self.buffer = 0


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self._get_feed_dict(item)

    def _get_feed_dict(self, item):
        sample = self.random_sample(item)

        samp_POI = sample
        Inp_Sample_POI = np.array(samp_POI)
        ###attention!
        HistoryLen = sum([1 for i in Inp_Sample_POI if i > 0])
        Padding = np.array([self.vocab.pad_index for _ in range(self.history_len - len(Inp_Sample_POI))])
        Inp_Sample_POI = np.concatenate((Inp_Sample_POI, Padding), axis=0)
        output = {"Inp_POI": Inp_Sample_POI,
                  "HistoryLen": HistoryLen,
                  "item": item}
        return {key: torch.tensor(value).long() for key, value in output.items()}


    def random_sample(self, index):
        sample = self.get_corpus_line(index)
        return sample

    def get_corpus_line(self, item):
        return self.datas[item]



