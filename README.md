# 0 MachineLearning_Regression
该数据集是收集于联合循环发电厂的9568个数据点, 共包含5个特征: 每小时平均环境变量温度（AT），环境压力（AP），相对湿度（RH），排气真空（V）和净每小时电能输出（PE）, 其中电能输出PE是我们要预测的变量。
数据：[介绍地址](http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)	      [下载地址](http://archive.ics.uci.edu/ml/machine-learning-databases/00294/)

文件说明：

- LR_global_path.py    全局文件路径

- readdata.py  读取数据，简单的特征统计显示

- LR_linreg.py 普通线性回归

- Ensembleplot.py 集成回归

- fullNet.py 全连接神经网络

普通的回归：
 - 线性回归（LinearRegression）
 - 岭回归（RidgeCV、Ridge）
 - 支持向量机线性回归（svm.SVR(kernel='linear')）

集成回归学习方法：

- 随机森林回归（RandomForestRegressor）
- 极端树回归（ExtraTreesRegressor）
- 梯度提升回归（GradientBoostingRegressor）

全连接神经网络

数据特征：

    开始统计特征状态....
    特征 AT 的峰态系数：-1.03755 属于:平峰分布
    特征 AT 的偏态系数：-0.13639 属于:低度偏态
    
    特征 V 的峰态系数：-1.44434 属于:平峰分布
    特征 V 的偏态系数：0.19852 属于:低度偏态
    
    特征 AP 的峰态系数：0.09424 属于:尖峰分布
    特征 AP 的偏态系数：0.26544 属于:低度偏态
    
    特征 RH 的峰态系数：-0.44453 属于:平峰分布
    特征 RH 的偏态系数：-0.43184 属于:低度偏态
    
    特征 PE 的峰态系数：-1.04852 属于:平峰分布
    特征 PE 的偏态系数：0.30651 属于:低度偏态
    特征状态统计完成....

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429154940471.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6Z19jaGF0,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429155049718.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6Z19jaGF0,size_16,color_FFFFFF,t_70)

简单线性回归：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429154333892.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6Z19jaGF0,size_16,color_FFFFFF,t_70)

集成回归：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429154357764.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6Z19jaGF0,size_16,color_FFFFFF,t_70)

全连接神经网络：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429154452889.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6Z19jaGF0,size_16,color_FFFFFF,t_70)

tensorbord的图示，以抽象节点显示，实际的数据流图和过程可以再tensorboard进一步展开观察。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190429173234629.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6Z19jaGF0,size_16,color_FFFFFF,t_70)