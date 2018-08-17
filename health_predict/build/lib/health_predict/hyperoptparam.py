# -*- coding: utf-8 -*-
"""
Created on Mon July 16 10:08:53 2018

@author: sxx
"""
import time
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import traceback 

from health_predict.predict import load_data as load_data
from health_predict.predict import transform_data as transform_data
from health_predict.predict import format_data as format_data
from health_predict.predict import divide_data as divide_data
from health_predict.predict import LstmModel as LstmModel

current_time=time.strftime('%Y-%m-%d-%H-%M',time.localtime())

def experiments(params):
    print ("epoch:",params['epoch']," batch_size: ",params['batch'])
    csvfile = './data/health_176.csv'
    x,y = load_data(csvfile)
    x_,y_ = transform_data(x,y,False)
    datax,datay = format_data(x_,y_)
    trainX,trainY,validX,validY,testX,testY = divide_data(datax,datay)
    
    try:
        ##调试模型epoch,batch_size,learnrate
        #model = LstmModel(epoches=params['epoch'], batch_size=params['batch'],lr=params['lr'])

        ##测试模型units,dropout_rate
        model = LstmModel(units=params['units'],dr=params['dr'])
        health_model = model.create_model()
        save_path = './models/health_{0}.h5'.format(current_time)
        model.train_model(trainX,trainY,validX,validY,save_path)
        testPredict = health_model.predict(testX)
        print (testY,"-->",testPredict)
    except Exception as e:
        print ("something happen:",repr(e))
        print ('-' * 20)
        with open("./data/err.log","a") as logf:
            traceback.print_exc(file=logf)
        return {'loss': 999999, 'status': STATUS_OK}
    
    mse = np.mean(np.square(testPredict - testY))
    print ("mse:",mse)
    print ('-' * 20)

    return {'loss': mse, 'status': STATUS_OK}

if __name__ == '__main__':
    space = {
        'units': {'lstm_units':{
        'units1': hp.choice('units1', range(120, 256)),
        'units2': hp.choice('units2', range(60, 128)),
        'units3': hp.choice('units3', range(30, 64)),
        'units4': hp.choice('units4', range(15, 32))},
        'dense_units':{
        'units5': hp.choice('units5', range(6, 16)),
        'units6': 5}
        },
        'dr': hp.uniform('dr',0.1,0.6),
        'batch': hp.randint('batch',256),
        'epoch': hp.randint('epoch',500),
        'lr': hp.uniform('lr',0.0001,0.01),
        'activation': hp.choice('activation',['relu',
                                                'sigmoid',
                                                'tanh',
                                                'linear'])
        }
    
    trials = Trials()
    best = fmin(experiments, space, algo=tpe.suggest, max_evals=50, trials=trials)
    print ('best: ',best)
    print ("hyperoptparam end")