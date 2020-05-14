import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
import datagen as dg

class Evaluation:
    def __init__(self):
        self.data, self.label = self.read_test_data()
        self.prd = []
        self.probs = []

    def read_test_data(self):
        f = open('validate_set.pkl', 'rb')
        data = pickle.load(f)
        f.close()
        f = open('validate_label.pkl', 'rb')
        label = pickle.load(f)
        f.close()
        return data, label

    def load_model(self, model_name='test.pkl'):
        model = pickle.load(open(model_name, 'rb'))
        self.model = model

    def process_data(self, data0, data1):
        ls = [data0, data1]
        dr0 = np.array(ls[0])[:, 0:-1].astype(np.float32)
        dr1 = np.array(ls[1])[:, 0:-1].astype(np.float32)

        dt = []
        for dr in [dr0, dr1]:
            dr = np.concatenate([dr, np.sin(dr[:, 2:3] / 86400 * np.pi)], 1)
            #             dr = np.concatenate([dr, np.cos(dr[:, 2:3] / 86400 * np.pi * 2)], 1)
            dr = np.delete(dr, [2], axis=1)

            df = DataFrame(dr, index=None, columns=['long', 'lat', 'status', 'sintime'])
            feature = dg.feature_extract(df, np.mean(df.long), np.mean(df.lat))
            feature = [feature] * len(dr)
            dr = np.concatenate([dr, feature], 1)
            for j in range(11):
                dr[:, j] = (dr[:, j] - np.mean(dr[:, j])) / (np.var(dr[:, j]) + 1e-5) ** 0.5

            length = dr.shape[0]
            kernel_size = length // dg.Seq_len + 1
            new_len = length // kernel_size * kernel_size
            new_data = np.zeros((dg.Seq_len * kernel_size, dg.Data_dim + 7))
            new_data[:new_len, :] = dr[:new_len, :]
            new_data = new_data.reshape(-1, kernel_size, dg.Data_dim + 7)
            new_data = np.mean(new_data, 1)  # mean_pooling to length about Seq_len
            dt.append([new_data])
        return dt

    def run(self, dt, threshold=0.5):
        prob = self.model.predict(dt)
        if prob > threshold:
            return 1, prob
        else:
            return 0, prob

    def pred(self, threshold=0.5):
        score = 0
        for d, label_sample in zip(self.data, self.label):
            data = self.process_data(d[0], d[1])
            prd, prob = self.run(data, threshold)
            self.prd.append(prd)
            self.probs.append(prob)
            if prd == label_sample:
                score += 1
        return score / len(self.label)

def load_model():
    evaluate = Evaluation()
    evaluate.load_model(model_name='test.pkl')
    return evaluate.model

def process_data(traj_1, traj_2):
    evaluate = Evaluation()
    return evaluate.process_data(traj_1,traj_2)

def run(data, model):
    evaluate = Evaluation()
    evaluate.model = model
    return evaluate.run(data,0.56)

if __name__ == '__main__':
    evaluate = Evaluation()
    evaluate.load_model('test.pkl')
    score = evaluate.pred(threshold=0.56)
    print(score)