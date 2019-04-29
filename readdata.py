# -*- coding: utf-8 -*-
"""
Created on 2019-04-11 16:31:14
@author: <dzg>

@software: spyder
"""
import LR_global_path as gpath
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


# 读数据
def getdata(folderpath,filename):
    print('数据加载中..........\n')
    folderpath = os.chdir(os.path.normpath(folderpath)) # 切换工作路径 方便读数据
    gotdata = pd.read_excel(filename)
    print('读取到数据：{m}\n部分内容如下：\n{n}'.format(m=gotdata.shape,n=gotdata.head()))
    return gotdata

# 特征偏态
def kur_status(kur):
    if kur==0:
        kurs = "正态分布"
    elif kur>0:
        kurs = "尖峰分布"
    else:
        kurs = "平峰分布"
    return kurs
        
# 特征峰态
def ske_status(ske):
    if abs(ske)>1:
        skes = "高度偏态"
    elif 0.5<abs(ske)<1:
        skes = "中度偏态"
    else:
        skes = "低度偏态"
    return skes        

# 绘制特征散点图
def drawpairplot(data):
    print('开始绘制属性散点图.....\n')
    sbn.set()
    sbn.pairplot(data, diag_kind = 'kde')
    plt.suptitle('Pairplots of CCCP')
    plt.savefig(gpath.SavePic+'pairplots.jpg')
    plt.show()
    print("绘图结束.......\n")

# 皮尔森相关系数热力图
def drawpier(X):
    print('开始计算绘制属性皮尔森热力图.....\n')
    pier = X.corr()
    print("属性间pierson系数为： \n",pier)
    sbn.heatmap(pier, linewidths=0.2, vmax=1, vmin=-1, 
                linecolor='w',annot=True,square=True)
    plt.suptitle('Pierson of CCCP columns')
    plt.savefig(gpath.SavePic+'pierson.jpg')
    plt.show()
    print('绘图结束.......\n')
    
# 数据特征    
def caldata(data):
    print('开始统计特征状态....\n')
    for i in data.columns:
        kur = data[i].kurt()
        k_status = kur_status(kur)  # 每列属性峰态系数
        ske = data[i].skew()
        s_status = ske_status(ske)  # 每个属性偏态系数
        print('特征 {} 的峰态系数：{:.5f} 属于:{}'.format(i,kur,k_status))
        print('特征 {} 的偏态系数：{:.5f} 属于:{}'.format(i,ske,s_status))
        print('\n')
    print('特征状态统计完成....\n')

# 主函数
if __name__ == '__main__':  
    data = getdata(gpath.DataPath,gpath.FileName) 
    drawpairplot(data.iloc[0:200])
    caldata(data)    
    drawpier(data)
