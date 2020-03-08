import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from keras.utils.np_utils import to_categorical

def feature_extract(df):
    top_long = []
    top_lat = []
    tempdf = df
    t_long = np.around(tempdf.longitude.astype(float),decimals=4).value_counts()
    t_lat = np.around(tempdf.latitude.astype(float),decimals=4).value_counts()
    t_sta = list(tempdf.status)
    flag = t_sta[0]
    cnt = flag
    for i in range(3):    #return top 3
        x = t_long.argmax()
        top_long.append(x-114)
        t_long = t_long.drop(x)
        y = t_lat.argmax()
        top_lat.append(y-22)
        t_lat = t_lat.drop(y)
    for id in range(0,len(t_sta)-1):  #get passenger numbers in a day
        if flag == 0 and t_sta[id] == 1:
            cnt += 1
        flag = t_sta[id]
    return [cnt/100] + top_long + top_lat

def test_split(driver,step,ave):
    whole_data = []
    t = 0
    x = 0
    times = len(driver) // step
    res = len(driver) % step
    while t <= times:
        if t == times:
            if res == 0:
                break
            temp = list(driver[-step:])
            newappend = []
            for subt in range(step // ave):
                sub = temp[subt * ave:(subt + 1) * ave]
                sub = [np.average(sub, 0)]
                newappend += sub
            whole_data.append(newappend)
            break
        else:
            x = t * step
            temp = list(driver[x:x + step])
            newappend = []
            for subt in range(step // ave):
                sub = temp[subt * ave:(subt + 1) * ave]
                sub = [np.average(sub, 0)]
                newappend += sub
            whole_data.append(newappend)
        t += 1
        while t < times - 1:
            for increment in [-step // 4, 0, step // 4]:
                x = t * step + step // 2 + increment
                temp = list(driver[x:x + step])
                newappend = []
                for subt in range(step // ave):
                    sub = temp[subt * ave:(subt + 1) * ave]
                    sub = [np.average(sub, 0)]
                    newappend += sub
                whole_data.append(newappend)
            t += 1
    return whole_data

def train_test_split(rate,step,ave):
    dataset = pd.read_csv("5drivers_dataset.csv")
    whole_data = []
    whole_label = []
    test_data = []
    test_label = []
    for dd in tqdm(range(2,150)):
        for i in range(5):
            temp_dataset = dataset[(dataset.date == dd) & (dataset.plate == i)].drop('date', axis=1).drop('index',
                                                                                                          axis=1).drop(
                'plate', axis=1)
            driver = np.array(temp_dataset)
            driver[:, 0] -= 114  # normalize longitude
            driver[:, 1] -= 22  # normalize latitude

            feature = feature_extract(temp_dataset)
            feature = [feature] * len(driver)

            driver = np.concatenate([driver, np.sin(driver[:, 2:3] / 86400 * np.pi - np.pi / 2)], 1)
            driver[:, 2] = np.cos(driver[:, 2] / 86400 * 2 * np.pi)
            #             driver[:,4] = driver[:,4]*1000
            driver = np.concatenate([driver, feature], 1)

            for j in range(6):
                driver[:, j] = (driver[:, j] - np.mean(driver[:, j])) / (np.var(driver[:, j]) + 1e-5) ** 0.5

            driver = driver.tolist()
            t = 0
            x = 0
            times = len(driver) // step
            res = len(driver) % step

            mod = int(1 / rate)
            while t <= times:
                if t == times:
                    if res == 0:
                        break
                    temp = list(driver[-step:])
                else:
                    x = t * step
                    temp = list(driver[x:x + step])
                newappend = []
                for subt in range(step // ave):
                    sub = temp[subt * ave:(subt + 1) * ave]
                    sub = [np.average(sub, 0)]
                    newappend += sub
                label = to_categorical(np.array([[i] for cnt in range(step // ave)]), num_classes=5)
                if dd % mod != 0:
                    whole_data.append(newappend)
                    whole_label.append(label)
                else:
                    test_data.append(newappend)
                    test_label.append(label)
                t += 1
            t = 0
            while t < times - 1:
                for increment in [-step // 4, 0, step // 4]:
                    x = t * step + step // 2 + increment
                    temp = list(driver[x:x + step])
                    newappend = []
                    for subt in range(step // ave):
                        sub = temp[subt * ave:(subt + 1) * ave]
                        sub = [np.average(sub, 0)]
                        newappend += sub
                    label = to_categorical(np.array([[i] for cnt in range(step // ave)]), num_classes=5)
                    if dd % mod != 0:
                        whole_data.append(newappend)
                        whole_label.append(label)
                    else:
                        test_data.append(newappend)
                        test_label.append(label)
                t += 1
    datazip = list(zip(whole_data,whole_label))
    random.shuffle(datazip)
    whole_data, whole_label = zip(*datazip)
    train_len = int(len(whole_data) * (1 - rate))
    whole_data = np.array(whole_data)
    whole_label = np.array(whole_label)
    train_data = whole_data[0:train_len]
    test_data = whole_data[train_len:]
    train_label = whole_label[0:train_len]
    test_label = whole_label[train_len:]

    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
    train_test_split(0.1,100,10)
