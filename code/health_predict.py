
# coding: utf-8

# In[101]:


import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import sys
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
import time
from keras.callbacks import Callback, ModelCheckpoint


# In[35]:


timesteps = seq_length = 10
data_dim = 50


# In[123]:


xy = np.loadtxt('./data/health_161.csv', delimiter=',',skiprows=1)


# In[124]:


x = xy[:,8:xy.shape[1]]
y = xy[:, 0]


# In[125]:


scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)
y = y/100


# In[129]:


dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)


# In[130]:


train_size = int(len(dataY) * 0.6)
valid_size = int(len(dataY) * 0.8) - train_size 
test_size = len(dataY) - valid_size
trainX, validX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:train_size+valid_size]), np.array(
    dataX[train_size+valid_size:len(dataX)])
trainY, validY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:train_size+valid_size]),np.array(
    dataY[train_size+valid_size:len(dataY)])


# model = Sequential()
# model.add(LSTM(1, input_shape=(seq_length, data_dim), return_sequences=False))
# # model.add(Dense(1))
# model.add(Activation("linear"))
# model.compile(loss='mean_squared_error', optimizer='adam')
# 
# model.summary()
# #plot_model(model, to_file=os.path.basename('model') + '.png', show_shapes=True)
# #plot_model(model, to_file='model.png', show_shapes=True)
# print(trainX.shape, trainY.shape)

# In[131]:


model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(8, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(2, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(8,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[132]:


#记录损失函数的历史数据
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
history = LossHistory()

#如果验证损失下降， 那么在每个训练轮之后保存模型。
checkpointer = ModelCheckpoint(filepath='./models/health_check.hdf5', verbose=1, save_best_only=True)


# In[133]:


model.fit(trainX, trainY, epochs=200, verbose=1, batch_size=32, 
          validation_data=(validX, validY), callbacks=[history, checkpointer])


# In[134]:


#model.save('./models/health_'+ time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + '.h5')
model.save('./models/health.h5')
#plot_model(model, to_file='./models/model.png', show_shapes=True, show_layer_names=True)


# In[135]:


losses = history.losses
print ("history: ",np.mean(losses))
#SVG(plot_model(model, to_file='./models/model.png', show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))


# In[136]:


loadModel = load_model('./models/health.h5')
#loss = model.evaluate(x=testX, y=testY, batch_size=2)


# In[137]:


testPredict = loadModel.predict(testX)


# In[155]:


for i in range(0,len(testY)):
    print (testPredict[i],"-->",testY[i])


# In[156]:


plt.figure(figsize=(24,6))
plt.plot(testY[0:200],c='r',label="train")
plt.plot(testPredict[0:200],c='g',label="predict")
plt.xlabel("time aix")
plt.ylabel("value aix")
plt.legend() 
plt.show()
plt.savefig('reslut.jpg')

