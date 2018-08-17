# -*- coding: utf-8 -*-
from health_predict import predict as pdt

def test_data_preprocess():
    csvfile = './data/health_161.csv'
    x,y = pdt.load_data(csvfile)
    x_,y_ = pdt.transform_data(x,y,False)
    datax,datay = pdt.format_data(x_,y_)
    trainX,trainY,validX,validY,testX,testY = pdt.divide_data(datax,datay)
    print ("trainX",trainX.shape)
    print ("validX",validX.shape)
    print ("testX",testX.shape)

if __name__ == '__main__':
    test_data_preprocess()