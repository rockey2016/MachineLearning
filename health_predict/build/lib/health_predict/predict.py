# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:52:53 2018

@author: sxx
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import time
from keras.optimizers import Adam

timesteps_in = 15
timesteps_out = 5
dim_in = 62
current_time=time.strftime('%Y-%m-%d-%H-%M',time.localtime())
adam_lr = 0.00699 #默认值为0.001
adam_beta_1 = 0.8 #默认值是0.9
adam_beta_2 = 0.5 #默认值是0.999
epoches = 446
batch_size = 164
dr=0.3 #dropout_rate
units = {'lstm_units':{
        'units1': 128,
        'units2': 64,
        'units3': 32,
        'units4': 16},
        'dense_units':{
        'units5': 8,
        'units6': 5}
        }

def load_data(file):
    xy = np.loadtxt(file, delimiter=',',skiprows=1)
    x = xy[:,8:xy.shape[1]]
    y = xy[:, 0]
    return x,y

def transform_data(x,y,is_predict=True):
    if is_predict is True:
        with open('./data/scaler_176.hdf5', 'rb') as f:
            scaler = pickle.load(f)
            x_ = scaler.fit_transform(x)
            y_ = y/100
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_ = scaler.fit_transform(x)
        y_ = y/100
        with open('./data/scaler_176.hdf5', 'wb') as f:
            pickle.dump(scaler,f)
    return x_, y_

def format_data(x,y):
    dataX = []
    dataY = []
    for i in range(0, len(y) - timesteps_in - timesteps_out):
        _x = x[i:i + timesteps_in]
        _y = y[i + timesteps_in:i + timesteps_in + timesteps_out]
        #print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)   
    return np.array(dataX), np.array(dataY)


def divide_data(x,y):
    train_size = int(len(y) * 0.7)
    valid_size = int(len(y) * 0.9) - train_size 
    trainX, validX, testX = x[0:train_size], x[train_size:train_size+valid_size], x[train_size+valid_size:len(x)]
    trainY, validY, testY = y[0:train_size], y[train_size:train_size+valid_size], y[train_size+valid_size:len(y)]
    return trainX,trainY,validX,validY,testX,testY


#记录损失函数的历史数据
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#保存训练最佳模型。
checkpoint = ModelCheckpoint(filepath='./models/health_checkpoint_{0}.hdf5'.format(current_time), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')


class LstmModel():
    def __init__(self, model=None, epoches=epoches, batch_size=batch_size, lr=adam_lr, units=units, dr=dr,
                 create_time=current_time, history = LossHistory(),checkpoint=checkpoint,earlystop=earlystop):
        self._model = model
        self._epoches = epoches
        self._batch_size = batch_size
        self._history = history
        self._checkpoint = checkpoint
        self._earlystop = earlystop
        self._lr = lr
        self._units = units
        self._dr = dr
        self._create_time = create_time
        print ("learn_rate: ",lr, "-->", self._lr)
        print ("units: ",self._units)
        
    def create_model(self):
        self._model = Sequential()
        self._model.add(LSTM(self._units['lstm_units']['units1'], input_shape=(timesteps_in, dim_in), return_sequences=True))
        self._model.add(Dropout(self._dr))
        self._model.add(LSTM(self._units['lstm_units']['units2'], return_sequences=True))
        self._model.add(Dropout(self._dr))
        self._model.add(LSTM(self._units['lstm_units']['units3'], return_sequences=True))
        self._model.add(Dropout(self._dr))
        self._model.add(LSTM(self._units['lstm_units']['units4'], return_sequences=False))
        self._model.add(Dropout(self._dr))
        self._model.add(Dense(self._units['dense_units']['units5'],activation = 'relu'))
        #self._model.add(Dropout(0.3))
        self._model.add(Dense(self._units['dense_units']['units6']))
        #self._model.add(Activation('linear'))
        #self._model.add(Activation('sigmoid'))
        self._model.add(Activation('tanh'))
        '''
        self._model = Sequential([
        LSTM(128, input_shape=(timesteps_in, dim_in), return_sequences=True),
        Dropout(self._dr),
        LSTM(64, return_sequences=True),
        Dropout(self._dr),
        LSTM(32, return_sequences=True),
        Dropout(self._dr),
        LSTM(16, return_sequences=False),
        Dropout(self._dr),
        Dense(8,activation = 'relu'),
        Dense(5,activation = 'tanh'),
        ])'''
        
        self._model.compile(loss='mean_squared_error', optimizer=Adam(lr=self._lr))
        self._model.summary()
        return self._model
    
    def train_model(self, train_x, train_y, valid_x, valid_y, model_save_path):
        self._model.fit(train_x, train_y, epochs=self._epoches, verbose=1, batch_size=self._batch_size, validation_data=(valid_x, valid_y), callbacks=[self._history,self._checkpoint,self._earlystop], shuffle=False)
        self._model.save(model_save_path)
        plot_model(self._model, to_file='./models/model_{0}.png'.format(self._create_time))
        
        losses = self._history.losses
        return losses

def show_plot(testY,testPredict):
    plt.figure(figsize=(24,48))
    plt.subplot(511)
    plt.plot(testY[0:200,1],c='r',label="train")
    plt.plot(testPredict[0:200,1],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend() 
    
    plt.subplot(512)
    plt.plot(testY[0:200,2],c='r',label="train")
    plt.plot(testPredict[0:200,2],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend()
    
    plt.subplot(513)
    plt.plot(testY[0:200,3],c='r',label="train")
    plt.plot(testPredict[0:200,3],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend()
    
    plt.subplot(514)
    plt.plot(testY[0:200,4],c='r',label="train")
    plt.plot(testPredict[0:200,4],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend()
    
    plt.subplot(515)
    plt.plot(testY[0:200,0],c='r',label="train")
    plt.plot(testPredict[0:200,0],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend() 
    plt.savefig('reslut_{0}.png'.format(current_time))
    plt.show()
    
if __name__ == '__main__':
    csvfile = './data/health_176.csv'
    x,y = load_data(csvfile)
    x_,y_ = transform_data(x,y,False)
    datax,datay = format_data(x_,y_)
    trainX,trainY,validX,validY,testX,testY = divide_data(datax,datay)
    
    lstmModel = LstmModel(epoches=446, batch_size=164,lr=0.00699)
    health_model = lstmModel.create_model()
    save_path = './models/health_{0}.h5'.format(current_time)
    losses = lstmModel.train_model(trainX,trainY,validX,validY,save_path)
    testPredict = health_model.predict(testX)
    
    print ("predict health loss",len(losses))
    #save_path = './models/health_0615.h5'
    #loadModel = load_model(save_path)
    #testPredict = loadModel.predict(testX)
    print (testY,"-->",testPredict)
    show_plot(testY,testPredict)
    