# -*- coding: utf-8 -*-
"""
Created on 2019-04-11 22:43:07
@author: <dzg>

@software: spyder
"""
import numpy as np
import LR_global_path as gpath
import readdata
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor

import sys
import os


from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import Ridge,RidgeCV

from sklearn.model_selection import cross_val_predict
from sklearn import svm
import seaborn
import matplotlib.pyplot as plt
import tensorboard
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from matplotlib.font_manager import FontProperties 
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12) 


# 正则归一化初始数据
def data_pre(dataV0):
#    isinsys(workfolder)
#    import readdata
#    data = readdata.getdata(folder,file) 
    #data = data.dropna(axis=0)
    X = dataV0[['AT','V','AP','RH']]
    y = dataV0[['PE']]     
    scaler = preprocessing.StandardScaler().fit(X)  # 标准化
    X_pre = scaler.transform(X)
    print("Data pre deal finish!")
    return X_pre,y

    
# 随机划分训练 测试集
def split_data(dataV1):
#    data = data_pre()
    features = dataV1[0]
    targets =dataV1[1]
    X_train, X_test, y_train, y_test = train_test_split(features,targets,random_state=1,test_size=0.33)
    print("Data split finish!\n")
    print("训练集：\n Xtrain：{m},  y_train: {n}".format(m=X_train.shape,n=len(y_train)))
    print("测试集：\n Xtest：{m},  y_test: {n}".format(m=X_test.shape,n=len(y_test)))
    return X_train, X_test, y_train, y_test


# 普通线性回归 LR方法
def lrModel(spldata):
    X_train, y_train = spldata[0], spldata[2]

    linreg = linear_model.LinearRegression()
    linreg.fit(X_train,y_train)
    print("普通 liner regression 参数:\n",linreg.coef_)
    print("普通 liner regression 截距:\n",linreg.intercept_)

#    X_test_trans = scaler.transform(X_test)
#    X_test_trans = X_test
    X_test, y_test = spldata[1], spldata[3]
    y_pre = linreg.predict(X_test)
    print(type(y_pre))
    print("普通 liner regression 预测值:\n",y_pre)
    print("普通 LR MSE 损失值:\n",metrics.mean_squared_error(y_test,y_pre))
    
    return y_pre
  
    
# 带 L2 正则线性回归  即 Ridge 岭回归
def lrridge(spldata):
    xtrain, ytrain = spldata[0], spldata[2]
    alpha_list = list(np.arange(0.01,4,0.03)) # 正则参数列表，选出最佳的alpha
    lr_model = RidgeCV(alphas=alpha_list)
    lr_ridgecv = lr_model.fit(xtrain,ytrain)
    print("LR Ridge 训练完成!\n")
    print("最佳的正则参数为：\n",lr_ridgecv.alpha_)
    # 选 参数 最好的这个 alpha 值 做岭回归
    lr_ridgemodel = Ridge(alpha=lr_ridgecv.alpha_)
    lr_ridge = lr_ridgemodel.fit(xtrain,ytrain)
    print("LR Ridge 参数为:\n",lr_ridge.coef_)
    print("LR Ridge 截距为:\n",lr_ridge.intercept_)
    
    xtest, ytest = spldata[1], spldata[3]
    y_pre_ridge = lr_ridgemodel.predict(xtest)
    print("linear regression ridge 预测：\n",y_pre_ridge)
    print('lr Ridge MSE:',metrics.mean_squared_error(ytest,y_pre_ridge))

#    predict = cross_val_predict(lr_model, data[0], data[2], cv=5)
#    print('交叉验证 MSE:\n',metrics.mean_squared_error(data[2],predict))
    
    return y_pre_ridge


#  线性 SNM 回归
def SVMmodel(spldata):
    svmlr=svm.SVR(kernel='linear')
#    print(len(data[2]))
#    pdata=data[2]
#    print(pdata[0:206])
    xtr, ytr = spldata[0], spldata[2]   #训练 测试集
    xte, yte = spldata[1], spldata[3]
    ytrlist = [i for item in ytr.values for i in item]
    svmlr.fit(xtr,ytrlist) 
    svmpre=svmlr.predict(xte)
    print("model have finished..")
    print('线性 SVM 预测的 MSE 损失值:\n',metrics.mean_squared_error(yte,svmpre))
    result = svmpre[:,np.newaxis]
    print("线性 SVM predict result:\n",result)
    
    return result


def drawpic(yte,lrpre,lrrpre,svmpre):
    ytelist = [i for item in yte.values for i in item]
    pltdata = ytelist[-50:-1]
    plt.figure(figsize=(15,10))
    plt.plot(np.arange(len(pltdata)),pltdata,'rs--',label='true value')
    plt.plot(np.arange(len(pltdata)),lrpre[-50:-1],'g^',label='LR')
    plt.plot(np.arange(len(pltdata)),lrrpre[-50:-1],'yo',label='LR Ridge')
    plt.plot(np.arange(len(pltdata)),svmpre[-50:-1],'cx',label='SVM')
#   plt.title('MSE: %f'%metrics.mean_squared_error(data[3],predata))
    plt.legend()
    plt.title(u'普通线性回归预测对比',fontproperties=font_set)
    plt.savefig(gpath.SavePic+'simplelinear.jpg')
    plt.show()


if __name__ == '__main__':
    data = readdata.getdata(gpath.DataPath,gpath.FileName)
#    isinsys(workfolder)
    predata = data_pre(data)
    splitdata = split_data(predata)
    lrdata = lrModel(splitdata) 
    lrrdata = lrridge(splitdata)
    svmdata = SVMmodel(splitdata)
    drawpic(splitdata[3],lrdata,lrrdata,svmdata)
#print("交叉验证：")
#predict = cross_val_predict(linreg, X, y, cv=5)
#print('MSE:',metrics.mean_squared_error(y,predict))
#print("RMSE:",np.sqrt(metrics.mean_squared_error(y,predict)))


'''
几种常用评估函数：
explained_variance_score：
  解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
  的方差变化，值越小则说明效果越差。
mean_absolute_error：
    平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
    ，其其值越小说明拟合效果越好。
mean_squared_error：均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
    平方和的均值，其值越小说明拟合效果越好。
r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
    变量的方差变化，值越小则说明效果越差。
数学定义：https://blog.csdn.net/Softdiamonds/article/details/80061191
'''
#model_metrics_list=['explained_variance_score',
#                    'mean_absolute_error', 
#                    'mean_squared_error', 
#                    'r2_score']
#
#model_pre_score = []
#
#for item in model_metrics_list:
#    i = eval(item)
#    score = i(y_test,y_pre)
#    model_pre_score.append(score) 
##model_list = list(map(str,model_metrics_list))
#for j in range(4):
#    print('计算 {m} 值为:\n{n}\n'.format(m=model_metrics_list[j],n=model_pre_score[j]))
  


#df['maker'] = df.car_name.map(lambda x: x.split()[0])
#df.origin = df.origin.map({1: 'America', 2: 'Europe', 3: 'Asia'})
#df=df.applymap(lambda x: np.nan if x == '?' else x).dropna()
#df[''] = df.horsepower.astype(float)
