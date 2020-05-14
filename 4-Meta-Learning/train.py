import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import random
import matplotlib.pyplot as plt
from keras.layers import TimeDistributed, Input, LSTM, Dropout, Activation, Dense, AveragePooling1D,Conv1D,Flatten,Add,Subtract,Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.layers import concatenate
import datagen as dg

class FewShot:
    def __init__(self, lr=0.001, dropout=0.5, activation='relu', batch_size=128, batch_num=100):
        # x1 = Lambda(slices, arguments={"index1": 0,"index2":6,"index3":0}, name="trajectory")(inputs)
        # x2 = Lambda(slices, arguments={"index1": 6,"index2":13,"index3":1}, name="features")(inputs)
        # x = TimeDistributed(Dense(128))(x1)
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.driver_num = 0
        self.loss = []
        self.acc = []

        # model define
        inputs = Input(shape=(dg.Seq_len, dg.Data_dim + 7))
        x1 = Lambda(dg.slices, arguments={"index1": 0, "index2": 4, "index3": 0}, name="trajectory")(inputs)
        x2 = Lambda(dg.slices, arguments={"index1": 4, "index2": 11, "index3": 1}, name="features")(inputs)
        x = TimeDistributed(Dense(128))(x1)
        embedding = TimeDistributed(Dropout(0.5))(x)
        x = LSTM(units=128, return_sequences=True)(embedding)
        x = Dropout(dropout)(x)
        x = LSTM(units=128, return_sequences=True)(x)
        x = Dropout(dropout)(x)
        lstm_emb = LSTM(units=128, return_sequences=False)(x)

        x2 = Dense(units=64, activation='relu')(x2)
        lstm_emb = concatenate([lstm_emb, x2])

        lstm_model = Model(inputs=inputs, outputs=lstm_emb)

        inputs_1 = Input(shape=(dg.Seq_len, dg.Data_dim + 7))
        inputs_2 = Input(shape=(dg.Seq_len, dg.Data_dim + 7))

        emb_1 = lstm_model(inputs_1)
        emb_2 = lstm_model(inputs_2)
        # concatenated = concatenate([emb_1, emb_2])
        x = Subtract()([emb_1, emb_2])
        x = Multiply()([x, x])
        x = Dense(units=32, activation=activation)(x)
        out = Dense(units=1, activation='sigmoid')(x)
        self.model = Model([inputs_1, inputs_2], out)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    def new_batch(self, data_dict):
        pairs = []
        labels = []
        batch_data = [[], []]
        for i in range(self.batch_size):
            pair = []
            drs = [0, 0]
            days = [0, 0]
            if i < self.batch_size // 2:  # 0
                while drs[0] == drs[1]:
                    drs = np.random.randint(0, self.driver_num - 1, 2, np.int)
                days = np.random.randint(0, 4, 2, np.int)
                labels.append([0])
            else:  # 1
                drs = [random.randint(0, self.driver_num - 1)] * 2
                while days[0] == days[1]:
                    days = np.random.randint(0, 4, 2, np.int)
                labels.append([1])
            pairs.append([(drs[0], days[0]), (drs[1], days[1])])
        datazip = list(zip(pairs, labels))
        random.shuffle(datazip)
        pairs, labels = zip(*datazip)
        for i in range(self.batch_size):
            batch_data[0].append(dg.padding(data_dict[(pairs[i][0][0], pairs[i][0][1])]))
            batch_data[1].append(dg.padding(data_dict[(pairs[i][1][0], pairs[i][1][1])]))
        batch_data[0] = np.array(batch_data[0])
        batch_data[1] = np.array(batch_data[1])
        labels = np.concatenate(labels)
        return batch_data, labels, pairs

    def train(self, data_dict, epochs):
        self.driver_num = len(data_dict) // 5
        for epoch in tqdm(range(epochs)):
            print("epoch " + str(epoch + 1) + "/" + str(epochs))
            for batch_i in range(self.batch_num):
                x_batch, y_batch, _ = self.new_batch(data_dict)
                loss, accuracy = self.model.train_on_batch(x_batch, y_batch)
                if batch_i % 10 == 0:
                    print("batch " + str(batch_i) + ":", "loss: " + str(loss), "acc: " + str(accuracy))
                if batch_i % 20 == 0:
                    self.loss.append(loss)
                    self.acc.append(accuracy)

    def summary(self):
        print(self.model.summary())

    def save(self, name):
        pickle.dump(self.model, open(name, 'wb'))

    def loss_history(self):
        fig = plt.figure(figsize=(10, 10))
        subp = fig.add_subplot(111)
        subp.set_title('Loss')
        subp.plot(range(0, len(self.loss), 1), self.loss)
        fig.show()
        fig = plt.figure(figsize=(10, 10))
        subp = fig.add_subplot(111)
        subp.set_title('Accuracy')
        subp.plot(range(0, len(self.acc), 1), self.acc)
        fig.show()

if __name__ == '__main__':
    data = dg.read_data()
    data_dict = dg.make_data(data)

    fewshot = FewShot(lr=0.001,dropout=0.8,activation='relu',batch_size=256,batch_num=100)
    fewshot.summary()
    fewshot.train(data_dict,epochs=15)
    fewshot.loss_history()
    fewshot.save('test.pkl')