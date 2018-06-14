
# coding: utf-8

# In[91]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import sys
from keras.utils.vis_utils import plot_model


# In[4]:


timesteps = seq_length = 7
data_dim = 5


# In[83]:


xy = np.loadtxt('./data-02-stock_daily.csv', delimiter=',')
type(xy)


# In[101]:


xy.shape


# In[84]:


xy = xy[::-1]


# In[85]:


scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)


# In[86]:


x = xy
y = xy[:, [-1]]


# In[72]:


y


# In[87]:


dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)


# In[111]:


len(dataX)


# In[88]:


train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])


# In[80]:


testY


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

# In[94]:


model = Sequential()
model.add(LSTM(5, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[95]:


model.fit(trainX, trainY, epochs=200, batch_size=2)


# In[96]:


loss = model.evaluate(x=testX, y=testY, batch_size=2)


# In[102]:


print (loss)


# In[98]:


testPredict = model.predict(testX)


# In[99]:


print (testPredict)


# In[100]:


plt.plot(testY,c='r',label="train")
plt.plot(testPredict,c='g',label="predict")
plt.xlabel("time aix")
plt.ylabel("value aix")
plt.legend() 
plt.show()


# In[51]:


print (model.layers)

