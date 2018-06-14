
# coding: utf-8

# In[34]:


import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import sys
from keras.utils.vis_utils import plot_model
import time


# In[35]:


timesteps = seq_length = 10
data_dim = 50


# In[36]:


xy = np.loadtxt('./data/health_161.csv', delimiter=',',skiprows=1)
type(xy)


# In[38]:


xy.shape[1]


# In[84]:


#xy = xy[::-1]


# In[39]:


scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)


# In[40]:


x = xy[:,8:xy.shape[1]]
y = xy[:, 0]


# In[33]:


#xy[1,1:len(xy[1,:])]
#type(x)
x.shape
#y[0]


# In[41]:


dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)


# In[43]:


len(dataX)


# In[44]:


train_size = int(len(dataY) * 0.6)
valid_size = int(len(dataY) * 0.8) - train_size 
test_size = len(dataY) - valid_size
trainX, validX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:train_size+valid_size]), np.array(
    dataX[train_size+valid_size:len(dataX)])
trainY, validY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:train_size+valid_size]),np.array(
    dataY[train_size+valid_size:len(dataY)])


# In[32]:


validX


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

# In[45]:


model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(8, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(2, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(8,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[49]:


model.fit(trainX, trainY, epochs=100, verbose=2, batch_size=32, validation_data=(validX, validY))


# In[50]:


#model.save('./models/health_'+ time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + '.h5')
model.save('./models/health.h5')
#plot_model(model, to_file='./models/model.png', show_shapes=True, show_layer_names=True)


# In[52]:


loadModel = load_model('./models/health.h5')
#loss = model.evaluate(x=testX, y=testY, batch_size=2)


# In[65]:


a_testX = testX[0].reshape(1,10,50)
a_testY = testY[0]


# In[63]:


a_testPredict = loadModel.predict(a_testX)


# In[67]:


print (a_testPredict)
print (a_testY)


# In[68]:


testPredict = loadModel.predict(testX)


# In[54]:


print (testPredict)


# In[55]:


plt.figure(figsize=(12,6))
plt.plot(testY,c='r',label="train")
plt.plot(testPredict,c='g',label="predict")
plt.xlabel("time aix")
plt.ylabel("value aix")
plt.legend() 
plt.show()
plt.savefig('reslut.jpg')


# In[21]:


print (model.layers)


# In[50]:


import time
print ("local time is "+time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

