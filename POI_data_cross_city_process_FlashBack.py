import os
import gzip
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

from vocab import WordVocab
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt




def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def generate_train_test_checkin(train_file, test_file, train_df_dict, test_df_dict, user2id, item2id):
    if os.path.exists(train_file):
        print('train.txt and test.txt has existed!!!')
        return
    usermin = min(user2id.values())
    usermax = max(user2id.values())
    # 1,2,3,...,8276
    users_count = usermax + 1
    itemmin = min(item2id.values())
    itemmax = max(item2id.values())
    # 0,1,2,3,..,21356
    # 0+8277,1+8277,2+8277,3+8277,..,21356+8277
    with open(train_file, 'w+') as f_train, open(test_file, 'w+') as f_test:
        for user_id in train_df_dict.keys():
            train_locs = [visit[0] for visit in
                       train_df_dict[user_id]]  # poi2id字典中的org_id，在entity2id中对应id是org_id + users_count
            train_locs.insert(0, user_id)
            for train_elem in train_locs:
                f_train.write(str(train_elem) + ' ')
            f_train.write('\n')

        for user_id in test_df_dict.keys():
            test_locs = [visit[0] for visit in
                          test_df_dict[user_id]]  # poi2id字典中的org_id，在entity2id中对应id是org_id + users_count
            test_locs.insert(0, user_id)
            for test_elem in test_locs:
                f_test.write(str(test_elem) + ' ')
            f_test.write('\n')
    print('Successfully generate train/test checkins!')


def generate_entity_file(entity2id_file, user2id, item2id):  # 构造entity2id文件
    if os.path.exists(entity2id_file):
        print('entity2id.txt has existed!!!')
        return
    usermin = min(user2id.values())
    usermax = max(user2id.values())
    # 1,2,3,...,8276
    users_count = usermax + 1
    itemmin = min(item2id.values())
    itemmax = max(item2id.values())
    # 0,1,2,3,..,21356
    # 0+8277,1+8277,2+8277,3+8277,..,21356+8277
    with open(entity2id_file, 'w+') as f:
        # users_count = len(users)
        for i in range(users_count):
            f.write(str(i) + ' ')
            f.write(str(i) + ' ')
            f.write('\n')
        for value in item2id.values():
            poi_id = value + users_count
            f.write(str(poi_id) + ' ')
            f.write(str(poi_id) + ' ')
            f.write('\n')
    print('Successfully generate entity2id.txt!')

# 可能会重复添加triplet，所以要进行去重操作，得到最终train triplets
def filter_train_triplet(read_file, write_file):
    filter_set = set()
    print('Filter repeated triplets......')
    count = 0
    with open(read_file, 'r') as f_read, open(write_file, 'w+') as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip('\n')
            if f_read_line not in filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + '\n')
    print('Original triplets: ', count)
    print('Final triplets: ', len(filter_set))
    return filter_set


# 去重且保证test triplets与train triplets不同
def filter_test_triplet(read_file, write_file, train_filter_set):
    filter_set = set()
    print('Filter repeated triplets......')
    count = 0
    with open(read_file, 'r') as f_read, open(write_file, 'w+') as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip('\n')
            if f_read_line not in filter_set and f_read_line not in train_filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + '\n')
    print('Original triplets: ', count)
    print('Final triplets: ', len(filter_set))


def get_df(path):
    i = 0
    df = {}
    for d in open(path, 'r'):
        df[i] = d.split(',')
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def generate_train_test_triplets(train_df_dict, train_triplets_file, user2id, item2id, poi2gps):  # 构造train/test 三元组
    f_train_triplets = open(train_triplets_file, 'w+')
    print('Construct interact relation and temporal relation......')
    usermin = min(user2id.values())
    usermax = max(user2id.values())
    #1,2,3,...,8276
    users_count = usermax + 1
    itemmin = min(item2id.values())
    itemmax = max(item2id.values())
    #GowallaLoc1
    # 1,2,3,...,8276
    #0,1,2,3,..,21356
    #0+8277,1+8277,2+8277,3+8277,..,21356+8277
    # GowallaLoc2
    # 1,2,3,...,8599
    # 0,1,2,3,..,35007
    # 0+8600,1+8600,2+8600,3+8600,..,35007+8600
    # Loc1TKY
    # 1,2,3,...,2293
    # 0,1,2,3,..,28628
    # 0+2294,1+2294,2+2294,3+2294,..,28628+2294
    # Loc2NYC
    # 1,2,3,...,1083
    # 0,1,2,3,..,18580
    # 0+1084,1+1084,2+1084,3+1084,..,18580+1084
    print("shit")
    with tqdm(total=len(train_df_dict)) as bar:
        for user_id in train_df_dict.keys():
            poi_ids = [visit[0] for visit in train_df_dict[user_id]]  # poi2id字典中的org_id，在entity2id中对应id是org_id + users_count
            # 构建interact关系
            for poi_id in poi_ids:
                poi_id = str(int(poi_id) + users_count)
                f_train_triplets.write(str(user_id) + '\t')
                f_train_triplets.write(str(poi_id) + '\t')
                f_train_triplets.write('0' + '\n')  # 0代表interact relation
            # 构建temporal关系  相邻poi相连
            # print('Construct temporal relation......')
            for i in range(len(poi_ids) - 1):
                poi_prev = str(int(poi_ids[i]) + users_count)
                poi_next = str(int(poi_ids[i + 1]) + users_count)
                if poi_prev != poi_next:
                    f_train_triplets.write(poi_prev + '\t')
                    f_train_triplets.write(poi_next + '\t')
                    f_train_triplets.write('1' + '\n')  # 1代表temporal relation
            bar.update(1)
    # 构建spatial关系  两个poi的距离小于距离阈值lambda_d，就相连
    print('Construct spatial relation......')
    pois_list = []
    for poi, coord in poi2gps.items():  # 生成元组列表, 即[(poi_1, coord_1), ...]
        pois_list.append((poi, coord[0]))
    # 方案2
    lambda_d = 3  # 距离阈值为3千米, 再取top k, 即双重限制
    all_lat = np.array([poi[1][0] for poi in pois_list])
    all_lon = np.array([poi[1][1] for poi in pois_list])
    all_pois = [poi[0] for poi in pois_list]
    pois_list_np = np.array(all_pois)
    pois_list_next = (pois_list_np + users_count)
    with tqdm(total=len(pois_list)) as bar:
        for i in range(len(pois_list)):
            loci_list = []
            # SK: improved version with batch calculate
            poi_prev, coord_prev = pois_list[i]
            poi_prev = poi_prev + users_count  # poi在entity中所对应的实体集
            lat_prev, lon_prev = coord_prev
            dist_ = haversine_np(lat_prev, lon_prev, all_lat, all_lon)
            dist_[i] = 100000000
            dist_limit = dist_[dist_ <= lambda_d]
            pois_limit = pois_list_next[dist_ <= lambda_d]
            loci_list = list(zip(pois_limit, dist_limit))
            sort_list = sorted(loci_list, key=lambda x: x[1])  # 从小到大排序,距离越小,排名越靠前
            length = min(len(sort_list), 50)
            select_pois = sort_list[:length]  # 一般情况下, sort_list的长度肯定不止50, 取top 50  这是第二重限制
            for poi_entity, _ in select_pois:
                f_train_triplets.write(str(poi_prev) + '\t')
                f_train_triplets.write(str(poi_entity) + '\t')
                f_train_triplets.write('2' + '\n')  # 2代表spatial relation
                # spatial relation是对称的
                f_train_triplets.write(str(poi_entity) + '\t')
                f_train_triplets.write(str(poi_prev) + '\t')
                f_train_triplets.write('2' + '\n')
            bar.update(1)


def create_graphs(train_df, dev_df, test_df, user2id, item2id, item_meta_df, SOURCE_PATH):
    train_df_dict = {k: list(zip(g["item_id"].tolist(), g["time"].tolist())) for k, g in train_df.groupby("user_id")}
    left_df = pd.concat([dev_df, test_df]).sort_index()
    left_df_dict = {k: list(zip(g["item_id"].tolist(), g["time"].tolist())) for k, g in left_df.groupby("user_id")}
    meta_df_dict = {k: list(zip(g["latitude"].tolist(), g["longitude"].tolist())) for k, g in item_meta_df.groupby("item_id")}
    train_file = os.path.join(SOURCE_PATH, 'train.txt')
    test_file = os.path.join(SOURCE_PATH, 'test.txt')
    entity2id_file = os.path.join(SOURCE_PATH, 'entity2id.txt')
    train_triplets = os.path.join(SOURCE_PATH, 'train_triplets.txt')
    test_triplets = os.path.join(SOURCE_PATH, 'test_triplets.txt')
    final_train_triplets = os.path.join(SOURCE_PATH, 'final_train_triplets.txt')
    final_test_triplets = os.path.join(SOURCE_PATH, 'final_test_triplets.txt')
    print('Generate train/test checkins......')
    generate_train_test_checkin(train_file, test_file, train_df_dict, left_df_dict, user2id, item2id)  # 划分train/test check-ins
    print('Generate entity2id......')
    generate_entity_file(entity2id_file, user2id, item2id)
    print('Construct train triplets......')
    generate_train_test_triplets(train_df_dict, train_triplets, user2id, item2id, meta_df_dict)  # 构造train/test 三元组
    print('Construct test triplets......')
    generate_train_test_triplets(left_df_dict, test_triplets, user2id, item2id, meta_df_dict)
    train_filter_triplets = filter_train_triplet(train_triplets, final_train_triplets)  # train三元组去重
    filter_test_triplet(test_triplets, final_test_triplets, train_filter_triplets)  # test三元组去重


def generate_data(DATANAME = None, RAW_PATH = None, SOURCE_PATH = None, cityid = None, Remain_sequence_test = 1):
    DATASET = DATANAME
    # RAW_PATH = os.path.join('./', DATASET)
    DATA_FILE = '{}_visit.txt'.format(DATASET)
    META_FILE = '{}_meta.txt'.format(DATASET)
    # DATA_FILE = 'ratings_{}.csv'.format(DATASET)

    # data_df = pd.read_csv(os.path.join(RAW_PATH, DATA_FILE), names=['reviewerID', 'asin', 'rating', 'unixReviewTime'])
    data_df = pd.read_csv(os.path.join(SOURCE_PATH, DATA_FILE), names=['userid', 'poiid', 'time'])
    data_df.head()

    meta_df = pd.read_csv(os.path.join(SOURCE_PATH, META_FILE), names=['poiid', 'latitude', 'longitude', 'category'])
    meta_df.head()


    # Filter items

    useful_meta_df = meta_df[meta_df['poiid'].isin(data_df['poiid'])]

    all_items = set(useful_meta_df['poiid'].values.tolist())
    data_df = data_df[data_df['poiid'].isin(all_items)]




    n_users = data_df['userid'].value_counts().size
    n_items = data_df['poiid'].value_counts().size
    n_clicks = len(data_df)
    min_time = data_df['time'].min()
    max_time = data_df['time'].max()



    time_format = '%Y-%m-%d'

    print('# Users:', n_users)
    print('# Items:', n_items)
    print('# Interactions:', n_clicks)
    print('Time Span: {}/{}'.format(
        datetime.utcfromtimestamp(min_time).strftime(time_format),
        datetime.utcfromtimestamp(max_time).strftime(time_format))
    )



    np.random.seed(2019)
    NEG_ITEMS = 99


    out_df = data_df.rename(columns={'poiid': 'item_id', 'userid': 'user_id', 'time': 'time'})
    out_df = out_df[['user_id', 'item_id', 'time']]
    out_df = out_df.drop_duplicates(['user_id', 'item_id', 'time'])
    out_df.sort_values(by=['time', 'user_id', 'item_id'], inplace=True)
    out_df.head()


    # reindex (start from 1)

    uids = sorted(out_df['user_id'].unique())
    user2id = dict(zip(uids, range(1, len(uids) + 1)))
    id2user = dict(zip(user2id.values(), user2id.keys()))

    iids = sorted(out_df['item_id'].unique())
    item2id = dict(zip(iids, range(5, len(iids) + 5)))
    STdict = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4}
    item2id.update(STdict)
    id2item = dict(zip(item2id.values(), item2id.keys()))

    out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
    out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])
    out_df = out_df.reset_index(drop=True)
    out_df.head()

    # leave one out spliting

    clicked_item_set = dict()
    for user_id, seq_df in out_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())


    def generate_dev_test(data_df):
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            neg_items = np.random.randint(5, len(iids) + 5, (len(result_df), NEG_ITEMS))
            for i, uid in enumerate(result_df['user_id'].values):
                user_clicked = clicked_item_set[uid]
                for j in range(len(neg_items[i])):
                    while neg_items[i][j] in user_clicked:
                        neg_items[i][j] = np.random.randint(5, len(iids) + 5)
            result_df['neg_items'] = neg_items.tolist()
            result_dfs.append(result_df)
        return result_dfs, data_df



    leave_df = out_df.groupby('user_id').head(Remain_sequence_test)
    data_df = out_df.drop(leave_df.index)

    [test_df, dev_df], data_df = generate_dev_test(data_df)
    train_df = pd.concat([leave_df, data_df]).sort_index()

    len(train_df), len(dev_df), len(test_df)




    train_df_source = train_df.copy(deep=True)
    test_df_source = test_df.copy(deep=True)
    dev_df_source = dev_df.copy(deep=True)

    train_df_source['user_id'] = train_df_source['user_id'].apply(lambda x: id2user[x])
    train_df_source['item_id'] = train_df_source['item_id'].apply(lambda x: id2item[x])
    test_df_source['user_id'] = test_df_source['user_id'].apply(lambda x: id2user[x])
    test_df_source['item_id'] = test_df_source['item_id'].apply(lambda x: id2item[x])
    dev_df_source['user_id'] = dev_df_source['user_id'].apply(lambda x: id2user[x])
    dev_df_source['item_id'] = dev_df_source['item_id'].apply(lambda x: id2item[x])



    # save results
    if not os.path.exists(os.path.join(SOURCE_PATH, 'train.csv')):
        train_df.to_csv(os.path.join(SOURCE_PATH, 'train.csv'), sep='\t', index=False)
    if not os.path.exists(os.path.join(SOURCE_PATH, 'dev.csv')):
        dev_df.to_csv(os.path.join(SOURCE_PATH, 'dev.csv'), sep='\t', index=False)
    if not os.path.exists(os.path.join(SOURCE_PATH, 'test.csv')):
        test_df.to_csv(os.path.join(SOURCE_PATH, 'test.csv'), sep='\t', index=False)

    # l2 category


    meta_df_source = useful_meta_df.copy(deep=True)

    item_meta_df = useful_meta_df.copy(deep=True)
    item_meta_df = item_meta_df.rename(columns={'poiid': 'item_id'})
    item_meta_df['item_id'] = item_meta_df['item_id'].apply(lambda x: item2id[x])

    item_meta_df.head()
    item_meta_df = item_meta_df.drop_duplicates(['item_id', 'latitude', 'longitude', 'category'])
    create_graphs(train_df, dev_df, test_df, user2id, item2id, item_meta_df, SOURCE_PATH)

    ###source meta data
    item_meta_df_source = useful_meta_df.copy(deep=True)
    item_meta_df_source = item_meta_df_source.rename(columns={'poiid': 'item_id'})
    item_meta_df_source['item_id'] = meta_df_source['poiid'].copy(deep=True)
    item_meta_df_source.insert(loc=0, column='City', value = cityid)

    ###save category
    cates = sorted(item_meta_df['category'].dropna().unique())
    cates_dict = dict(zip(cates, range(5, len(cates) + 5)))
    item_meta_df['category'] = item_meta_df['category'].apply(lambda x: cates_dict[x])
    file_category = open(os.path.join(SOURCE_PATH, 'category.pkl'), 'wb')
    category_dict = dict(zip(item_meta_df['item_id'], item_meta_df['category']))
    pickle.dump(category_dict, file_category)
    coordinate_dict = dict(zip(item_meta_df['item_id'], zip(item_meta_df['latitude'], item_meta_df['longitude'])))
    file_coordinate = open(os.path.join(SOURCE_PATH, 'coordinate.pkl'), 'wb')
    pickle.dump(coordinate_dict, file_coordinate)


    # save results
    if not os.path.exists(os.path.join(SOURCE_PATH, 'item_meta.csv')):
        item_meta_df.to_csv(os.path.join(SOURCE_PATH, 'item_meta.csv'), sep='\t', index=False)

    return user2id, item2id, train_df_source, test_df_source, dev_df_source, item_meta_df_source


def eval_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda x: eval(str(x)))  # some list-value columns
    return df

def generate_segment(current_sentence_time, interval_time):
    current_sentence_segment = []
    begin1 = 1
    current_time = current_sentence_time[0]
    for idx, time in enumerate(current_sentence_time):
        if time - current_time < interval_time:
            current_sentence_segment.append(begin1)
        else:
            begin1 += 1
            current_time = time
            current_sentence_segment.append(begin1)
    return current_sentence_segment

def generate_segment_hier(current_sentence_time, last_sentence_segment, interval_time):
    current_sentence_segment = []
    begin1 = 1
    current_time = current_sentence_time[0]
    current_segment = last_sentence_segment[0]
    begin_index = 0
    indicate_index = 0
    for idx, time in enumerate(current_sentence_time):
        if last_sentence_segment[idx] == current_segment:
            continue
            # current_sentence_segment.append(begin1)
        else:
            ## find segment interval
            if current_sentence_time[idx - 1] - current_time < interval_time:
                current_sentence_segment.extend([begin1]*(idx- indicate_index))
                current_segment = last_sentence_segment[idx]
                indicate_index = idx
            else:
                begin1 += 1
                current_sentence_segment.extend([begin1] * (idx - indicate_index))
                current_segment = last_sentence_segment[idx]
                current_time = time
                indicate_index = idx
    if current_sentence_time[len(current_sentence_time) - 1] - current_time < interval_time:
        current_sentence_segment.extend([begin1] * (len(current_sentence_time) - indicate_index))
    else:
        begin1 += 1
        current_sentence_segment.extend([begin1] * (len(current_sentence_time) - indicate_index))
    return current_sentence_segment


def df_to_dict(df: pd.DataFrame) -> dict:
    res = df.to_dict('list')
    for key in res:
        res[key] = np.array(res[key])
    return res


def df_len(data):
    if type(data) == dict:
        for key in data:
            return len(data[key])
    return len(data)

if __name__ == '__main__':
    MAXLEN = 20
    Remain_sequence_test = 1
    # DATASETS = ['Digital_Music']
    # DATASETS = ['Grocery_and_Gourmet_Food']
    DATASETS = ['GowallaLoc1','GowallaLoc2']
    # DATASETS = ['Beauty']
    user2ids = []
    all_item2ids = dict()
    train_dfs = []
    test_dfs = []
    dev_dfs = []
    meta_dfs = []
    DATASETS_maps_dict = dict()
    DATASET_COM = '_'.join(DATASETS)
    RAW_PATH = os.path.join('./data/', 'COM_' + DATASET_COM)
    SOURCE_PATH = './data/'

    if not os.path.exists(RAW_PATH):
        os.mkdir(RAW_PATH)
    # create sub file
    for DATASET in DATASETS:
        if not os.path.exists(os.path.join(RAW_PATH, DATASET)):
            os.mkdir(os.path.join(RAW_PATH, DATASET))

    for i, DATASET in enumerate(DATASETS):
        user2id, item2id, train_df, test_df, dev_df, meta_df = \
            generate_data(DATASET, os.path.join(RAW_PATH, DATASET),  os.path.join(SOURCE_PATH, DATASET), cityid=i, Remain_sequence_test = Remain_sequence_test)
        DATASETS_maps_dict[DATASET] = [user2id, item2id]
        # print(len(train_df))
        # train_df = train_df.drop_duplicates(['user_id', 'item_id', 'time'])
        # train_df.sort_values(by=['time', 'user_id', 'item_id'], inplace=True)
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        dev_dfs.append(dev_df)
        meta_dfs.append(meta_df)
        all_item2ids.update(item2id)
    # import sys
    # sys.exit()
    # with open(os.path.join(RAW_PATH, 'DICTs'), 'wb') as writer:
    #     pickle.dump(DATASETS_maps_dict, writer)

    com_train_df = pd.concat(train_dfs, axis=0, ignore_index = True)
    com_train_df = com_train_df.drop_duplicates(['user_id', 'item_id', 'time'])
    com_train_df.sort_values(by=['time', 'user_id', 'item_id'], inplace=True)
    com_dev_df = pd.concat(dev_dfs, axis=0, ignore_index=True)
    com_dev_df = com_dev_df.drop_duplicates(['user_id', 'item_id', 'time'])
    com_dev_df.sort_values(by=['time', 'user_id', 'item_id'], inplace=True)
    com_test_df = pd.concat(test_dfs, axis=0, ignore_index=True)
    com_test_df = com_test_df.drop_duplicates(['user_id', 'item_id', 'time'])
    com_test_df.sort_values(by=['time', 'user_id', 'item_id'], inplace=True)


    # print(len(com_train_df))

    # output word list
    file_words = open(os.path.join(RAW_PATH, 'words.COM_' + DATASET_COM), 'w')
    for item in all_item2ids.keys():
        if item in ['<pad>','<unk>', '<eos>', '<sos>', '<mask>']:
            continue
        outstr = item + '\n'
        file_words.write(outstr)
    file_words.close()

    with open(os.path.join(RAW_PATH, 'words.COM_' + DATASET_COM), "r", encoding="utf-8") as f:
        globalvocab = WordVocab(f, max_size=None, min_freq=0)
    f.close()
    print("VOCAB SIZE:", len(globalvocab))
    output_path = os.path.join(RAW_PATH, 'vocab.COM_' + DATASET_COM)
    globalvocab.save_vocab(output_path)
    # build project map
    global_vocab_dict = globalvocab.stoi

    ###都用
    com_meta_df = pd.concat(meta_dfs, axis=0, ignore_index=True)
    cates = sorted(com_meta_df['category'].dropna().unique())
    cates_dict = dict(zip(cates, range(5, len(cates) + 5)))
    com_meta_df['category'] = com_meta_df['category'].apply(lambda x: cates_dict[x])
    com_meta_df['item_id'] = com_meta_df['item_id'].apply(lambda x: global_vocab_dict[x])
    file_category = open(os.path.join(RAW_PATH, 'category.COM_' + DATASET_COM), 'wb')
    com_meta_df_dict = dict(zip(com_meta_df['item_id'], com_meta_df['category']))
    pickle.dump(com_meta_df_dict, file_category)

    # xixi = zip(com_meta_df['item_id'], com_meta_df['latitude'])
    com_meta_df_dict = dict(zip(com_meta_df['item_id'], zip(com_meta_df['latitude'], com_meta_df['longitude'])))
    file_coordinate = open(os.path.join(RAW_PATH, 'coordinate.COM_' + DATASET_COM), 'wb')
    pickle.dump(com_meta_df_dict, file_coordinate)

    com_meta_df_dict = dict(zip(com_meta_df['item_id'], com_meta_df['City']))
    file_city = open(os.path.join(RAW_PATH, 'city.COM_' + DATASET_COM), 'wb')
    pickle.dump(com_meta_df_dict, file_city)

    ###直接输出ID编码，不要字符串了
    file_corpus = open(os.path.join(RAW_PATH, 'corpus.COM_' + DATASET_COM), 'w')
    file_time = open(os.path.join(RAW_PATH, 'time.COM_' + DATASET_COM), 'w')
    for user_id, seq_df in com_train_df.groupby('user_id'):
        sentence = [str(global_vocab_dict[item]) for item in seq_df['item_id'].values.tolist()]
        sentence_time = seq_df['time'].values.tolist()
        sentence_len = len(sentence)
        outstr = ' '.join(sentence) + '\n'
        outstr_time = ' '.join([str(time) for time in sentence_time]) + '\n'
        file_corpus.write(outstr)
        file_time.write(outstr_time)
    file_corpus.close()
    file_time.close()




    ###输出test的ID编码
    data_dfs = [com_train_df, com_dev_df, com_test_df]
    user_his_dict = dict()
    for df in data_dfs:
        i_history, t_history = [], []
        for uid, iid, t in zip(df['user_id'], df['item_id'], df['time']):
            if uid not in user_his_dict:
                user_his_dict[uid] = []
            i_history.append([x[0] for x in user_his_dict[uid]])
            t_history.append([x[1] for x in user_his_dict[uid]])
            user_his_dict[uid].append((iid, t))
        df['item_his'] = i_history
        df['time_his'] = t_history
        df['item_his'] = df['item_his'].apply(lambda x: x[-MAXLEN:])
        df['time_his'] = df['time_his'].apply(lambda x: x[-MAXLEN:])
        df['his_length'] = df['item_his'].apply(lambda x: len(x))
    file_corpus = open(os.path.join(RAW_PATH, 'test_corpus.COM_' + DATASET_COM), 'w')
    file_time = open(os.path.join(RAW_PATH, 'test_time.COM_' + DATASET_COM), 'w')
    test_data = df_to_dict(com_test_df)
    lens = df_len(test_data)
    for index in range(lens):
        sentence = [str(global_vocab_dict[item]) for item in test_data['item_his'][index]]
        sentence.append(str(global_vocab_dict[test_data['item_id'][index]]))
        sentence_time = list(test_data['time_his'][index])
        sentence_time.append(test_data['time'][index])
        sentence_len = len(sentence)
        outstr = ' '.join(sentence) + '\n'
        outstr_time = ' '.join([str(time) for time in sentence_time]) + '\n'
        file_corpus.write(outstr)
        file_time.write(outstr_time)
    file_corpus.close()
    file_time.close()




    for DATASET in DATASETS_maps_dict:
        CURRENT_PATH =  os.path.join(RAW_PATH, DATASET, 'MAP')
        CURRENT_MAP = dict()
        current_item2id = DATASETS_maps_dict[DATASET][1]
        for item in current_item2id:
            CURRENT_MAP[current_item2id[item]] = global_vocab_dict[item]
        with open(CURRENT_PATH, 'wb') as output:
            pickle.dump(CURRENT_MAP, output)


        TRAIN = pd.read_csv(os.path.join('./data/', DATASET, 'train.csv'), sep='\t')
        eval_list_columns(TRAIN)
        TRAIN['item_id'] = TRAIN['item_id'].apply(lambda x: CURRENT_MAP[x])
        TRAIN.to_csv(os.path.join(RAW_PATH, DATASET, 'train_map.csv'), sep='\t', index=False)

        DEV = pd.read_csv(os.path.join('./data/', DATASET, 'dev.csv'), sep='\t')
        eval_list_columns(DEV)
        DEV['item_id'] = DEV['item_id'].apply(lambda x: CURRENT_MAP[x])
        DEV['neg_items'] = DEV['neg_items'].apply(lambda x: [CURRENT_MAP[y] for y in x])
        DEV.to_csv(os.path.join(RAW_PATH, DATASET, 'dev_map.csv'), sep='\t', index=False)

        TEST = pd.read_csv(os.path.join('./data/', DATASET, 'test.csv'), sep='\t')
        eval_list_columns(TEST)
        TEST['item_id'] = TEST['item_id'].apply(lambda x: CURRENT_MAP[x])
        TEST['neg_items'] = TEST['neg_items'].apply(lambda x: [CURRENT_MAP[y] for y in x])
        TEST.to_csv(os.path.join(RAW_PATH, DATASET, 'test_map.csv'), sep='\t', index=False)

        META = pd.read_csv(os.path.join('./data/', DATASET, 'item_meta.csv'), sep='\t')
        eval_list_columns(META)
        META['item_id'] = META['item_id'].apply(lambda x: CURRENT_MAP[x])
        # META['r_complement'] = META['r_complement'].apply(lambda x: [CURRENT_MAP[y] for y in x])
        # META['r_substitute'] = META['r_substitute'].apply(lambda x: [CURRENT_MAP[y] for y in x])
        META.to_csv(os.path.join(RAW_PATH, DATASET, 'item_meta_map.csv'), sep='\t', index=False)

