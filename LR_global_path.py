# -*- coding: utf-8 -*-
"""
Created on 2019-04-14 11:08:45
@author: <dzg>

@software: spyder
"""
import sys
import os

WorkFolder = 'D:/dzg_project/UCIpowerPre/'      # 文件夹存放路径
DataPath = 'D:/dzg_project/UCIpowerPre/CCPP'    # 数据文件存放路径
FileName = 'Folds5x2_pp.xlsx'                   #数据文件的名称
SavePic = 'D:/dzg_project/UCIpowerPre/CCPP/images/'     #图片保存路径

 # 判断工作路径在 sys中sys.path
def isinsys(folder):        
    folderpath = os.path.normpath(folder)
    if folderpath in sys.path:
        os.chdir(WorkFolder)
    else:
        sys.path.append(folderpath)
        os.chdir(WorkFolder)
        
isinsys(WorkFolder)    
