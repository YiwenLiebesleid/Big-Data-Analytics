import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os


def slices(x, index1,index2,index3=0):
    if index3 == 0:
        return x[:,:, index1:index2]
    else:
        return x[:,0,index1:index2]

Seq_len = 50
Data_dim= 4
epoch_num = 3
batch_num = 100
batch_size = 128

def read_data():
    data = pd.read_csv(os.path.join('/kaggle/input/driverdata', 'dataset.csv'))
    data = data.drop('Unnamed: 0',axis=1)
    return data

def feature_extract(df,long,lat):
    top_long = []
    top_lat = []
    tempdf = df
    t_long = np.around(tempdf.long.astype(float),decimals=4).value_counts()
    t_lat = np.around(tempdf.lat.astype(float),decimals=4).value_counts()
    t_sta = list(tempdf.status)
    flag = t_sta[0]
    cnt = flag
    for i in range(3):    #return top 3
        if len(t_long) == 0:
            x = top_long[-1]+long
        else:
            x = t_long.argmax()
            t_long = t_long.drop(x)
        top_long.append(x-long)
        if len(t_lat) == 0:
            y = top_lat[-1]+lat
        else:
            y = t_lat.argmax()
            t_lat = t_lat.drop(y)
        top_lat.append(y-lat)
    for id in range(0,len(t_sta)-1):  #get passenger numbers in a day
        if flag == 0 and t_sta[id] == 1:
            cnt += 1
        flag = t_sta[id]
    return [cnt/100] + top_long + top_lat

def make_data(data):
#     data = pd.concat([data, pd.DataFrame(columns=['sintime','costime'])],sort=False)
    data = pd.concat([data, pd.DataFrame(columns=['sintime'])],sort=False)
    data.sintime = np.sin(data.time / 86400 * np.pi)
#     data.costime = np.cos(data.time / 86400 * 2 * np.pi)
    data = data.drop('time',axis=1)
    long_mean = np.mean(data.long)
    lat_mean = np.mean(data.lat)
#     data.long = (data.long - long_mean) / (np.var(data.long) + 1e-5) ** 0.5
#     data.lat = (data.lat - lat_mean) / (np.var(data.lat) + 1e-5) ** 0.5
    data_dict = dict()
    del_cnt = 0
    for d in tqdm(range(500)):
        flag = 0
        for i in range(5):
            dt = data[(data.plate == d) & (data.date == i)]
            if len(dt) <= 1000:
                del_cnt += 1
                flag = 1
                break
        if flag == 0:
            for i in range(5):
                dt = data[(data.plate == d) & (data.date == i)]
                feature = feature_extract(dt,long_mean,lat_mean)
                feature = [feature] * len(dt)
                temp = np.array(dt.drop('plate',axis=1).drop('date',axis=1))
                temp = np.concatenate([temp, feature], 1)
                for j in range(11):
                    temp[:, j] = (temp[:, j] - np.mean(temp[:, j])) / (np.var(temp[:, j]) + 1e-5) ** 0.5
                data_dict.update({(d-del_cnt,i):temp})
    return data_dict

def padding(data):
    portion = 0.9
    X = np.zeros((Seq_len,Data_dim+7))
    length = data.shape[0]
    kernel_size = int(length*portion)//Seq_len + 1
    new_len = int(length*portion)//kernel_size
    start = random.randint(0,length - new_len*kernel_size)
    data = data[start: start + new_len*kernel_size,:]
    data = data.reshape(-1,kernel_size,Data_dim+7)
    data = np.mean(data,1) #mean_pooling to length about Seq_len
    X[:new_len, :] = data
    return X
