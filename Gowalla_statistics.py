from collections import Counter
from datetime import datetime
import time
#####all dataset
POI_dict = dict()
User_dict = dict()
Coor_dict = dict()
with open('data/Gowalla/poi_feature.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        inf = line.strip().split(',')
        POI_id = int(inf[0])
        Coor_Cat = [int(float(inf[1])), int(float(inf[2])), int(inf[3])]
        if POI_id not in POI_dict:
            POI_dict[POI_id] = Coor_Cat

with open('data/Gowalla/checkins.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        inf = line.strip().split(',')
        User_id = int(inf[0])
        POI_id = int(inf[1])
        Time_str = inf[2]
        Timestamp = time.mktime(datetime.strptime(Time_str, "%Y-%m-%dT%H:%M:%SZ").timetuple())
        if User_id not in User_dict:
            User_dict[User_id] = []
        User_dict[User_id].append([POI_id, Timestamp])
        ##统计访问该区域的record数目
        Coor_str = '{},{}'.format(POI_dict[POI_id][0], POI_dict[POI_id][1])
        if Coor_str not in Coor_dict:
            Coor_dict[Coor_str] = 0
        Coor_dict[Coor_str] += 1
        ###
####从小到大排序
for user in User_dict:
    User_dict[user] = sorted(User_dict[user], key=lambda x: x[-1])

def generate_subdata(data_name, loca_center):
    with open('data/{}/{}_visit.txt'.format(data_name, data_name), 'w') as o:
        Location_POIs_indata = []
        for i, user in enumerate(User_dict):
            if i%10000 == 0:
                print(i)
            records = User_dict[user]
            for record in records:
                POI_id = record[0]
                Timestamp = record[1]
                if POI_dict[POI_id][0]<=loca_center[0] + 1 and POI_dict[POI_id][0]>=loca_center[0] - 1 and \
                        POI_dict[POI_id][1] <= loca_center[1] + 1 and POI_dict[POI_id][1] >= loca_center[1] - 1:
                # if POI_id in Location_POIs:
                    Location_POIs_indata.append(POI_id)
                    record_outstr = "{},{},{}\n".format('{}{}{}'.format(data_name, 'u', user),
                                                        '{}{}{}'.format(data_name, 'p', POI_id), Timestamp)
                    o.write(record_outstr)
    with open('data/{}/{}_meta.txt'.format(data_name, data_name), 'w') as o:
        for POI_id in Location_POIs_indata:
            record_outstr = "{},{},{},{}\n".format('{}{}{}'.format(data_name, 'p', POI_id),
                                                   POI_dict[POI_id][0], POI_dict[POI_id][1],POI_dict[POI_id][2])
            o.write(record_outstr)

counter = Counter(Coor_dict)
print(counter.most_common(10))
LOCATION1 = [30,-97]
LOCATION2 = [37,-122]
data_names = ['GowallaLoc1', 'GowallaLoc2']
loca_centers = [LOCATION1, LOCATION2]
for data_name,loca_center  in zip(data_names, loca_centers):
    generate_subdata(data_name, loca_center)


