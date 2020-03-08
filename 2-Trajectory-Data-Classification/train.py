import numpy as np
import pandas as pd
import pickle
from keras.layers import TimeDistributed, Input, LSTM, Dropout, Activation, Dense,Flatten,concatenate
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import KFold
from pandas.core.frame import DataFrame
from keras.optimizers import Adam
import datagen as dg

def slices(x, index1,index2,index3=0):
    if index3 == 0:
        return x[:,:, index1:index2]
    else:
        return x[:,0,index1:index2]

BATCHSIZE = 256

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def lstm(rate, step, ave):
    X, Y, X_test, Y_test = dg.train_test_split(rate, step, ave)
    Y = Y[:,0,:]
    Y_test = Y_test[:,0,:]
    # kFold = KFold(n_splits=5)
    # kFold.get_n_splits(X)
    # for train, test in kFold.split(X):
    # X_train, Y_train, X_test, Y_test = X[train], Y[train], X[test], Y[test]

    X_train, Y_train = X, Y
    inputs = Input(shape=(step // ave, 13))
    x1 = Lambda(slices, arguments={"index1": 0, "index2": 6, "index3": 0}, name="trajectory")(inputs)
    x2 = Lambda(slices, arguments={"index1": 6, "index2": 13, "index3": 1}, name="features")(inputs)
    x = TimeDistributed(Dense(128))(x1)
    embedding = TimeDistributed(Dropout(0.5))(x)
    x = LSTM(units=128, return_sequences=True)(embedding)
    x = Dropout(0.5)(x)
    x = LSTM(units=128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(units=128, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x2 = Dense(units=32, activation='relu')(x2)
    x = concatenate([x, x2])
    x = Dense(units=50, activation='relu')(x)
    predictions = Dense(units=5, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    # print(model.summary())
    # print(X_train.shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    losshistory = LossHistory()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=50, batch_size=BATCHSIZE, validation_data=(X_test, Y_test), verbose=2,callbacks=[early_stopping,losshistory])

    # model = Sequential()
    #
    # model.add(TimeDistributed(Dense(128), input_shape=(step // ave, 6)))
    # model.add(TimeDistributed(Dropout(0.2)))
    # model.add(LSTM(units=128, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=128, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=16, return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=50, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=5, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    # model.fit(X, Y, epochs=40, batch_size=BATCHSIZE, validation_data=(X_test, Y_test), verbose = 2,callbacks=[early_stopping,losshistory])
    #
    # print(losshistory.losses)
    df = DataFrame(losshistory.losses)
    df.to_csv("history.csv")
    pickle.dump(model, open('driver_model_35.pkl', 'wb'))

if __name__ == "__main__":
    lstm(0.1,1000,20)
