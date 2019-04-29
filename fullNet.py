# -*- coding: utf-8 -*-
"""
Created on 2019-04-16 17:47:07
@author: <dzg>

@software: spyder
"""
import LR_global_path as gpath
import numpy as np
import tensorflow as tf
import pandas as pd
import readdata
import LR_linreg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties 
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12) 


lrate = 0.01 # 学习率
epoch = 2000


# 给回归模型的参数矩阵的 偏置 加一列 1 ， 方便写成矩阵乘法运算
def conect(data):
    data = pd.DataFrame(data)
    ones = pd.DataFrame({'ones':np.ones(len(data))})
    df = pd.concat([ones,data],axis=1)
    df.rename(columns={'ones':'a', 0:'AT', 1:'V', 2:'AP', 3:'RH'}, inplace = True)
    #df.rename(columns={'0':'AT', '1':'V', '2':'AP', '3':'RH'}, inplace = True)
    return df


# 绘制loss 变化曲线
def drawloss(xx): 

    sns.set(context="notebook", style="whitegrid", palette="dark")
    ax = sns.lineplot(x='epoch', y='loss', data=pd.DataFrame({'loss': xx[1], 'epoch': np.arange(epoch)}))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.title(u'Loss 变化情况',fontproperties=font_set)
    plt.savefig(gpath.SavePic+'loss.jpg')
    plt.show()


# 全连接神经网络结构
def network(xdata,ydata):
    
    with tf.name_scope('input'): # 抽象节点 方便在tensorboard中观察
        X = tf.placeholder(tf.float32,xdata.shape,name='X')
        y = tf.placeholder(tf.float32,ydata.shape,name='y')
        
    with tf.name_scope('hypothesis'):   # 权重变量 W，形状[5,1] 行数为 x 输入特征
        W = tf.get_variable("weights",(xdata.shape[1], 1), initializer=tf.constant_initializer())
#        W = tf.Variable(tf.truncated_normal((xdata.shape[1], 1),stddev=0.1)) 
        y_pred = tf.matmul(X, W, name='y_pred')     # 预测值 y_pred  形状[x记录条数,1]

    with tf.name_scope('loss'):  # 损失函数操作 loss
        #loss_op = 1 / (2 * len(xdata)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)
        #tf.matmul(a,b,transpose_a=True) 矩阵a的转置乘矩阵b
        loss_op = tf.losses.mean_squared_error(y_pred, y)
    
    with tf.name_scope('train'):    # 随机梯度下降优化器
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lrate).minimize(loss_op)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())         # 初始化
    # 创建FileWriter实例，传入当前sess加载的数据流图
        writer = tf.summary.FileWriter(r'D:\dzg_project\UCIpowerPre\CCPP\summary\lr2', sess.graph)
        loss_data = []
        for i in range(1, epoch + 1):     # 开始训练
            _, loss, w, y_p = sess.run([train_op, loss_op, W, y_pred], feed_dict={X: xdata, y: ydata})
            loss_data.append(float(loss))  #记录每轮loss
            if i % 100 == 0:
                logvar = "Now Epoch:%d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4gx3 + %.4gx4 + %.4g"
                print(logvar % (i, loss, w[1], w[2], w[3], w[3],w[0]))
            if i == epoch: 
#                print(type(y_p))
                print("完成全连接神经网络的运算！\n")  # 打印最小loss的预测输出
    writer.close()
    return  y_p, loss_data
        
if __name__ == '__main__':
    data = readdata.getdata(gpath.DataPath,gpath.FileName)
    X_data, y_data = LR_linreg.data_pre(data)
#    print(X_data,type(X_data))
    X_data = conect(X_data)
    tf.reset_default_graph()
    xx = network(X_data,y_data)
    drawloss(xx)
#    print(type(X_data))
#    print(X_data.head())
    

    
        

