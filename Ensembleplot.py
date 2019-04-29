# -*- coding: utf-8 -*-
"""
Created on 2019-04-20 13:16:42
@author: <dzg>

@software: spyder
"""
import LR_global_path as gpath
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import readdata
import LR_linreg
from matplotlib.font_manager import FontProperties 
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12) 


# 随机森林回归
def RFR(xtr,ytr,xte):
    rfr=RandomForestRegressor()
    rfr.fit(xtr,ytr)
    rfr_y_predict=rfr.predict(xte)
    return rfr_y_predict
    
# 极端树回归
def ETR(xtr,ytr,xte):
    etr=ExtraTreesRegressor()
    etr.fit(xtr,ytr)
    etr_y_predict=etr.predict(xte)
    return etr_y_predict
    

def GBR(xtr,ytr,xte): 
#    params = {'n_estimators': 500, 'max_depth': 4, 
#              'learning_rate': 0.01, 'loss': 'ls'}
    gbr=GradientBoostingRegressor()
    gbr.fit(xtr,ytr)
    gbr_y_predict=gbr.predict(xte)
    return gbr_y_predict



def plotpic(ytelist,preRFR,preETR,preGBR):
    pltdata = ytelist[-50:-1]
    plt.figure(figsize=(15,10))
    plt.plot(np.arange(len(pltdata)),pltdata,'rs--',label='true value')
    plt.plot(np.arange(len(pltdata)),preRFR[-50:-1],'c^',label='preRFR')
#    plt.plot(np.arange(len(pltdata)),preETR[-50:-1],'yx',label='preETR')
    plt.plot(np.arange(len(pltdata)),preETR[-50:-1],'yx--',label='SVM')
    plt.plot(np.arange(len(pltdata)),preGBR[-50:-1],'ko',label='preGBR')
#   plt.title('MSE: %f'%metrics.mean_squared_error(data[3],predata))
    plt.legend()
    plt.title(u'集成学习预测对比',fontproperties=font_set)
    plt.savefig(gpath.SavePic+'Ensemble.jpg')
    plt.show()

if __name__ == '__main__':
    
    data = readdata.getdata(gpath.DataPath,gpath.FileName)
    predata = LR_linreg.data_pre(data)
    splitdata = LR_linreg.split_data(predata)
    xtr, ytr, xte, yte = splitdata[0], splitdata[2], splitdata[1], splitdata[3]
    
    ytrlist = [i for item in ytr.values for i in item]
    ytelist = [i for item in yte.values for i in item]
    svmdata = LR_linreg.SVMmodel(splitdata)
    
    a = RFR(xtr,ytrlist,xte)
    b = ETR(xtr,ytrlist,xte)
    c = GBR(xtr,ytrlist,xte)
    print('RandomForestRegressor MSE:\n',metrics.mean_squared_error(yte,a))
    print('ExtraTreesRegressor MSE:\n',metrics.mean_squared_error(yte,b))
    print('GradientBoostingRegressor MSE:\n',metrics.mean_squared_error(yte,c))
    print('SVMlinear MSE:\n',metrics.mean_squared_error(yte,svmdata))
#    plotpic(ytelist,a,b,c)  
    plotpic(ytelist,a,svmdata,c)  




#a = randombr(data[0],yt,data[1])
#b = randomfr(data[0],yt,data[1])
#c = randomtr(data[0],yt,data[1])
#print('ytest:',type(yte),len(yte),yte)
#print('ypred6br:',type(b),len(b),b)
#print(metrics.mean_squared_error(data[3],a))
#print(metrics.mean_squared_error(data[3],b))
#print(metrics.mean_squared_error(data[3],c))
#plotpic(yte,a,b,c)
##plotpic(yte,b)
##plotpic(yte,c)
#
#
