import pandas as pd
import numpy as np
import pickle
from pandas.core.frame import DataFrame
import datagen as dg

def process_data(traj):
    traj = np.array(traj)
    traj = traj[0]
    l = traj[:,2]

    new_l = []
    for item in l:
        tm = item.split(" ")[1]  # change format of time
        hr, mn, sec = tm.split(':')[0], tm.split(':')[1], tm.split(':')[2]
        newsec = int(hr) * 60 * 60 + int(mn) * 60 + int(sec)
        new_l.append(newsec)
    traj[:,2] = new_l

    velocity = []
    v = 0
    velocity.append(v)
    pos_long = list(traj[:,0])
    pos_lat = list(traj[:,1])
    tm = list(traj[:,2])
    for step in range(len(traj) - 1):
        if tm[step + 1] <= tm[step]:
            vel = 0
        else:
            vel = (abs(pos_long[step + 1] - pos_long[step]) + abs(pos_lat[step + 1] - pos_lat[step])) / (
                        tm[step + 1] - tm[step])
        velocity.append(vel)
    traj = np.insert(traj, 4, values=[velocity], axis=1)

    df = DataFrame(traj,index=None,columns=['longitude','latitude','time','status','velocity'])
    feature = dg.feature_extract(df)
    feature = [feature] * len(traj)

    traj[:,0] -= 114
    traj[:,1] -= 22
    traj = np.concatenate([traj,np.sin((traj[:,2:3]/86400*np.pi-np.pi/2).tolist())],1)
    traj[:, 2] = np.cos(list(traj[:, 2] / 86400 * 2 * np.pi))
    # traj[:, 4] = traj[:, 4] * 1000
    traj = np.concatenate([traj, feature], 1)

    for i in range(5):
        traj[:, i] = (traj[:, i] - np.mean(traj[:, i])) / (np.var(traj[:, i]) + 1e-5) ** 0.5

    Feature = dg.test_split(traj,1000,20)# needs modification if length shorter than 1000
    return np.array(Feature)

def run(data, model):
    pred = model.predict(data)# needs modification
    # print(pred)
    # pred_label = []
    # for item in pred:
    #     pred_label.append(np.array(item).argmax())
    # print(pred_label)
    pred_prob = np.sum(pred,0)
    # print(pred_prob)
    maxlabel = np.argmax(pred_prob)
    # maxlabel = -1
    # max_cnt = -1
    # for i in range(5):
    #     if pred_label.count(i) > max_cnt:
    #         maxlabel = i
    #         max_cnt = pred_label.count(i)
    return maxlabel

# if __name__ == "__main__":
#     traj = pickle.load(open('test41.pkl', 'rb'))
#     feature = process_data([traj])
#     model = pickle.load(open('driver_model_35.pkl','rb'))
#     pred = run(feature,model)
#     print(pred)
