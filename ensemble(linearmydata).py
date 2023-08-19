# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:33:44 2023

@author: Administrator
"""


        
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:49:48 2023

@author: Administrator
"""


"""
Created on Mon Dec 19 20:06:21 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 23:28:14 2022

@author: Administrator
"""

import os
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt #引用画图包
import numpy as np #引用数值计算扩展包
import pandas as pd #引用数据分析包
from sklearn.model_selection import train_test_split #引用sklearn包的数据分割库
from sklearn.ensemble import RandomForestRegressor #引用随机森林算法库
from sklearn.metrics import r2_score #引用评价指标函数
import scipy.io as scio #引用文件输入包
import emg_features as ef
from scipy.signal import butter, lfilter, freqz
import scipy
# from deepforest import CascadeForestRegressor
from sklearn.model_selection import KFold
import csv
import scipy.stats as stats
import xlrd
from xlutils import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from math import sqrt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#--------------------------------------
import pandas as pd
#regress method
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import numpy as np


def get_file(file_path: str, suffix: str, res_file_path: list) -> list:
    """获取路径下的指定文件类型后缀的文件

    Args:
        file_path: 文件夹的路径
        suffix: 要提取的文件类型的后缀
        res_file_path: 保存返回结果的列表

    Returns: 文件路径

    """

    for file in os.listdir(file_path):

        if os.path.isdir(os.path.join(file_path, file)):
            get_file(os.path.join(file_path, file), suffix, res_file_path)
        else:
            res_file_path.append(os.path.join(file_path, file))

    # endswith：表示以suffix结尾。可根据需要自行修改；如：startswith：表示以suffix开头，__contains__：包含suffix字符串
    return res_file_path if suffix == '' or suffix is None else list(filter(lambda x: x.endswith(suffix), res_file_path))

def butter_bandpass(lowcut, highcut, fs, order=5):

    nyq = 0.5 * fs

    low = lowcut / nyq

    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    y = lfilter(b, a, data)

    return y


def Implement_Notch_Filter(fs, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                      analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def pearson_correlation(object1, object2):
    values = range(len(object1))

    # Summation over all attributes for both objects
    sum_object1 = sum([float(object1[i]) for i in values])
    sum_object2 = sum([float(object2[i]) for i in values])

    # Sum the squares
    square_sum1 = sum([pow(object1[i],2) for i in values])
    square_sum2 = sum([pow(object2[i],2) for i in values])

    # Add up the products
    product = sum([object1[i]*object2[i] for i in values])

    #Calculate Pearson Correlation score
    numerator = product - (sum_object1*sum_object2/len(object1))
    denominator = ((square_sum1 - pow(sum_object1,2)/len(object1)) * (square_sum2 -
    	pow(sum_object2,2)/len(object1))) ** 0.5

    # Can"t have division by 0
    if denominator == 0:
        return 0

    result = numerator/denominator
    return result


if __name__ == '__main__':

    parent = os.path.dirname(os.path.realpath("__file__"))
    parent1 = os.path.dirname(parent)
    parent2 = os.path.dirname(parent1)
    parent3 = os.path.dirname(parent2)
    parent4 = os.path.dirname(parent3)
    print(parent)
    excel_path=parent+'\\ensemble_stacking(linearmydata).xls'#文件路径
    #excel_path=unicode('D:\\测试.xls','utf-8')#识别中文路径
    rbook = xlrd.open_workbook(excel_path,formatting_info=True)#打开文件
    wbook = copy.copy(rbook)#复制文件并保留格式
    w_sheet = wbook.get_sheet(0)#索引sheet表

    col=0
    row=0

    # MyFolder = r'C:/Users/Administrator/Desktop/transport door/data/upper/20210822hand/lpy/' ##待移动Excel的文件夹
    # UploadFolder = r'C:/Users/Administrator/Desktop/transport door/data/processed_data/lpyemg/' ##Folder2目标结果文件夹

    order=pd.read_excel(parent+'/data_filename - 1.xls')
    order_f1=order['f1']
    order_f2=order['f2']
    for file_i in range(len(order)):
        angle= pd.read_excel(parent+'/vicon/'+order_f1[file_i],header=4,usecols=["a1","a2"])
        emg_signal=pd.read_csv(parent+'/emg/'+str(order_f2[file_i])+'.csv',header=2,usecols=["Noraxon Ultium 2-Noraxon Ultium 2.拇长展肌 (uV)","Noraxon Ultium 2-Noraxon Ultium 2.拇长屈肌 (uV)","Noraxon Ultium 2-Noraxon Ultium 2.桡侧腕屈肌 (uV)","Noraxon Ultium 2-Noraxon Ultium 2.指浅屈肌 (uV)","Noraxon Ultium 2-Noraxon Ultium 2.指深屈肌 (uV)","Noraxon Ultium 2-Noraxon Ultium 2.指伸肌 (uV)","Noraxon Ultium 2-Noraxon Ultium 2.示指伸肌 (uV)","Noraxon Ultium 2-Noraxon Ultium 2.桡侧腕伸长肌 (uV)"])
        x=emg_signal.values
        y1=angle["a1"]
        y2=angle["a2"]


        len_database=len(x)
        len_data=(len_database-40)//20
  
    
    
        for i in range(8):
            x[:,i] = Implement_Notch_Filter(2000,20,50,100,3,'butter',x[:,i])
            x[:,i] = Implement_Notch_Filter(2000,20,100,100,3,'butter',x[:,i])
            x[:,i] = Implement_Notch_Filter(2000,20,150,100,3,'butter',x[:,i])
            x[:,i] = Implement_Notch_Filter(2000,20,200,100,3,'butter',x[:,i])
            x[:,i] = Implement_Notch_Filter(2000,20,250,100,3,'butter',x[:,i])
            x[:,i] = Implement_Notch_Filter(2000,20,300,100,3,'butter',x[:,i])
            x[:,i] = Implement_Notch_Filter(2000,20,350,100,3,'butter',x[:,i])
            x[:,i] = Implement_Notch_Filter(2000,20,400,100,3,'butter',x[:,i])
    
    
        # ---------------------------------------------------------------------------------------------
    
        dx_mean= np.zeros([len_data,8], dtype = float)
        dx_i=np.zeros([len_data,8], dtype = float)
        dx_zero=np.zeros([len_data,8], dtype = float)
        dx_slope=np.zeros([len_data,8], dtype = float)
        dx_wl=np.zeros([len_data,8], dtype = float)
        dx_emg_mav=np.zeros([len_data,8], dtype = float)
        dx_log=np.zeros([len_data,8], dtype = float)
    
        dx=np.zeros([len_data,56], dtype = float)
        X=np.zeros([len_data,113], dtype = float)
    
        dy=np.zeros([len_data,1], dtype = float)
    
        for i in range(len_data):
            dy[i]=y2[i*1:i*1+2].mean()
    
            for j in range(8):
    
                dx_mean[i,j]=ef.emg_rms(x[i*20:i*20+40,j])
                dx_zero[i,j]=ef.emg_zc(x[i*20:i*20+40,j])
                dx_i[i,j]=ef.emg_iemg(x[i*20:i*20+40,j])
                dx_wl[i,j]=ef.emg_wl(x[i*20:i*20+40,j])
                dx_log[i,j]=ef.emg_log(x[i*20:i*20+40,j])
                dx_emg_mav[i,j]=ef.emg_mav(x[i*20:i*20+40,j])
                dx_slope[i,j]=ef.emg_ssc(x[i*20:i*20+40,j])
    
    
        # run_one(op_up=True)
        # -----------------------------------------------------------------------------
        
        y_pred_mean_xgr=np.zeros([len_data,1], dtype = float)
        y_pred_mean_LGBR=np.zeros([len_data,1], dtype = float)
        y_pred_mean_cat=np.zeros([len_data,1], dtype = float)
        y_pred_mean_gp = np.zeros([len_data, 1], dtype=float)
        dx = np.append(dx_mean, dx_i, axis=1)
        dx=np.append(dx,dx_zero,axis=1)
        dx=np.append(dx,dx_log,axis=1)
        dx=np.append(dx,dx_wl,axis=1)
        dx=np.append(dx,dx_slope,axis=1)
        dx=np.append(dx,dx_emg_mav,axis=1)

        scaler = MinMaxScaler()
        scaler.fit(dx)
        dx1 = scaler.transform(dx)

        X=np.append(dy[0:len_data-1],dx[0:len_data-1],axis=1)
        X=np.append(X,dx[1:len_data],axis=1)

        scaler_x = MinMaxScaler()
        X1 = scaler_x.fit_transform(X)

        Y=dy[1:len_data]
        scaler_y = MinMaxScaler()
        Y1 = scaler_y.fit_transform(Y)
    
        
     
        rmse_mean_xgr=np.zeros([5,1], dtype = float)
        p_mean_xgr=np.zeros([5,1], dtype = float)
        rmse_mean_LGBR=np.zeros([5,1], dtype = float)
        p_mean_LGBR=np.zeros([5,1], dtype = float)
        rmse_mean_cat=np.zeros([5,1], dtype = float)
        p_mean_cat=np.zeros([5,1], dtype = float)
        rmse_mean_gp = np.zeros([5, 1], dtype=float)
        p_mean_gp = np.zeros([5, 1], dtype=float)

        KF=KFold(n_splits=5) #建立5折交叉验证方法 查一下KFold函数的参数
        num_r2=0
        for train_index,test_index in KF.split(X):
    
            X_train,X_test=X[train_index],X[test_index]
            X1_train, X1_test = X1[train_index], X1[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            Y1_train, Y1_test = Y1[train_index], Y1[test_index]

    
       
            X_t_xgr=np.zeros([1,113], dtype = float)
            X_t_LGBR=np.zeros([1,113], dtype = float)
            X_t_cat=np.zeros([1,113], dtype = float)
            X_t_gp = np.zeros([1, 113], dtype=float)

            model_mean_xgr =XGBRegressor(n_estimators=100) # 基模型
            model_mean_LGBR = lgb.LGBMRegressor(n_estimators=100) # 基模型
            model_mean_cat = CatBoostRegressor(n_estimators=100) # 基模型
            model_mean_gp = GaussianProcessRegressor()  # 基模型

            model_mean_xgr.fit(X_train, Y_train)
            model_mean_LGBR.fit(X_train, Y_train)
            model_mean_cat.fit(X_train, Y_train)
            model_mean_gp.fit(X1_train, Y1_train)

            y_pred_mean_xgr[test_index[0]]=Y_test[0]
            y_pred_mean_LGBR[test_index[0]]=Y_test[0]
            y_pred_mean_cat[test_index[0]]=Y_test[0]
            y_pred_mean_gp[test_index[0]] = Y1_test[0]
            
            for i in test_index:
    
                X_t_xgr[0,0]=y_pred_mean_xgr[test_index[i-test_index[0]]] #每一时刻的特征向量的第一部分为上一时刻解码出的关节角度
                X_t_LGBR[0,0]=y_pred_mean_LGBR[test_index[i-test_index[0]]]
                X_t_cat[0,0]=y_pred_mean_cat[test_index[i-test_index[0]]]
                X_t_gp[0, 0] = y_pred_mean_gp[test_index[i - test_index[0]]]
                
                for j in range(56):
    
                      X_t_xgr[0,j+1]=dx[i,j] #将相应的肌电特征组合输入到特征向量中
                      X_t_xgr[0,j+57]=dx[i+1,j]
                      X_t_LGBR[0,j+1]=dx[i,j]
                      X_t_LGBR[0,j+57]=dx[i+1,j]
                      X_t_cat[0,j+1]=dx[i,j]
                      X_t_cat[0,j+57]=dx[i+1,j]
                      X_t_gp[0, j + 1] = dx1[i, j]
                      X_t_gp[0, j + 57] = dx1[i + 1, j]
                      
                y_pred_mean_xgr[i+1,0] = model_mean_xgr.predict(X_t_xgr[:]) #每次单独解码出一个关节角度
                y_pred_mean_LGBR[i+1,0] = model_mean_LGBR.predict(X_t_LGBR[:])
                y_pred_mean_cat[i+1,0] = model_mean_cat.predict(X_t_cat[:])
                y_pred_mean_gp[i + 1, 0] = model_mean_gp.predict(X_t_gp[:])
            # rmse_mean_xgr[num_r2]=mean_squared_error(Y_test, y_pred_mean_xgr[test_index])
            p_mean_xgr[num_r2]=pearson_correlation(Y_test, y_pred_mean_xgr[test_index]) #最终对曲线的相关系数进行评价
            # rmse_mean_LGBR[num_r2]=mean_squared_error(Y_test, y_pred_mean_LGBR[test_index])
            p_mean_LGBR[num_r2]=pearson_correlation(Y_test, y_pred_mean_LGBR[test_index])
            # rmse_mean_cat[num_r2]=mean_squared_error(Y_test, y_pred_mean_cat[test_index])
            p_mean_cat[num_r2]=pearson_correlation(Y_test, y_pred_mean_cat[test_index])
            
            # rmse_mean_gp[num_r2]=mean_squared_error(Y_test, y_pred_mean_gp[test_index])
            p_mean_gp[num_r2] = pearson_correlation(Y_test, y_pred_mean_gp[test_index])
            num_r2=num_r2+1
            
        y_pred_mean_gp1=scaler_y.inverse_transform(y_pred_mean_gp)    
        rmse_xgr=sqrt(mean_squared_error(dy, y_pred_mean_xgr))
        pxgr=p_mean_xgr.mean()
        r2xgr=r2_score(dy, y_pred_mean_xgr)
        rmse_LGBR=sqrt(mean_squared_error(dy, y_pred_mean_LGBR))
        pLGBR=p_mean_LGBR.mean()
        r2LGBR=r2_score(dy, y_pred_mean_LGBR)
        rmse_cat=sqrt(mean_squared_error(dy, y_pred_mean_cat))
        pcat=p_mean_cat.mean()
        r2cat=r2_score(dy, y_pred_mean_cat)
        rmse_gp = sqrt(mean_squared_error(dy, y_pred_mean_gp1))
        pgp = p_mean_gp.mean()
        r2gp = r2_score(dy, y_pred_mean_gp1)
            
    #xgb+lgb-----------------------------------------------------------------------------------   
        X= np.zeros([len_data,2], dtype = float)
        X=np.append(y_pred_mean_xgr[0:len_data],y_pred_mean_LGBR[0:len_data],axis=1)
    
    
        Y=dy[0:len_data]
        y_pred_xgb_lgb=np.zeros([len_data,1], dtype = float)
        r2_xgb_lgb=np.zeros([10,1], dtype = float)
        p_xgb_lgb=np.zeros([10,1], dtype = float)
        KF=KFold(n_splits=10) #建立10折交叉验证方法 查一下KFold函数的参数
        num_r2=0
        for train_index,test_index in KF.split(X):
    
            X_train,X_test=X[train_index],X[test_index]
            Y_train,Y_test=Y[train_index],Y[test_index]
            
            
            model = LinearRegression()
            # 训练模型
            model.fit(X_train,Y_train)
    
            y_pred_xgb_lgb[test_index]=(model.predict(X_test)).reshape(-1,1)
    
    
            # regressor = Sequential()
            # #first hidden layer
            # regressor.add(Dense(13, input_shape=(2,)))
            # #second hidden layer
            # # regressor.add(Dense(16, activation='relu',kernel_initializer='glorot_uniform'))
            # # regressor.add(Dense(16, activation='relu',kernel_initializer='glorot_uniform'))
            # # regressor.add(Dense(16, activation='relu',kernel_initializer='glorot_uniform'))
            # regressor.add(Dense(1))
    
            # regressor.compile(optimizer= 'adam',loss= 'mean_squared_error')
    
            # regressor.fit(X_train,Y_train, epochs=100)# batch_size= 10,
    
            # scoreval = regressor.evaluate(X_train,Y_train)# ,batch_size= 10
    
            # y_pred_xgb_lgb[test_index]= regressor.predict(X_test)# ,batch_size= 10
    
    
    
            r2_xgb_lgb[num_r2]=r2_score(Y_test, y_pred_xgb_lgb[test_index])
            p_xgb_lgb[num_r2]=pearson_correlation(Y_test, y_pred_xgb_lgb[test_index])
            num_r2=num_r2+1
        r2xgb_lgb=r2_xgb_lgb.mean()
        pxgb_lgb=p_xgb_lgb.mean()
        rmse_xgb_lgb = sqrt(mean_squared_error(dy, y_pred_xgb_lgb))
    
    #xgb+cat-----------------------------------------------------------------------------------   
        X= np.zeros([len_data,2], dtype = float)
        X=np.append(y_pred_mean_xgr[0:len_data],y_pred_mean_cat[0:len_data],axis=1)
    
    
        Y=dy[0:len_data]
        y_pred_xgb_cat=np.zeros([len_data,1], dtype = float)
        r2_xgb_cat=np.zeros([10,1], dtype = float)
        p_xgb_cat=np.zeros([10,1], dtype = float)
        KF=KFold(n_splits=10) #建立10折交叉验证方法 查一下KFold函数的参数
        num_r2=0
        for train_index,test_index in KF.split(X):
    
            X_train,X_test=X[train_index],X[test_index]
            Y_train,Y_test=Y[train_index],Y[test_index]
            
            model = LinearRegression()
            # 训练模型
            model.fit(X_train,Y_train)
    
            y_pred_xgb_cat[test_index]=(model.predict(X_test)).reshape(-1,1)
    
    
    
            r2_xgb_cat[num_r2]=r2_score(Y_test, y_pred_xgb_cat[test_index])
            p_xgb_cat[num_r2]=pearson_correlation(Y_test, y_pred_xgb_cat[test_index])
            num_r2=num_r2+1
        r2xgb_cat=r2_xgb_cat.mean()
        pxgb_cat=p_xgb_cat.mean()
        rmse_xgb_cat = sqrt(mean_squared_error(dy, y_pred_xgb_cat))
    #lgb+cat-----------------------------------------------------------------------------------   
        X= np.zeros([len_data,2], dtype = float)
        X=np.append(y_pred_mean_LGBR[0:len_data],y_pred_mean_cat[0:len_data],axis=1)
    
    
        Y=dy[0:len_data]
        y_pred_lgb_cat=np.zeros([len_data,1], dtype = float)
        r2_lgb_cat=np.zeros([10,1], dtype = float)
        p_lgb_cat=np.zeros([10,1], dtype = float)
        KF=KFold(n_splits=10) #建立10折交叉验证方法 查一下KFold函数的参数
        num_r2=0
        for train_index,test_index in KF.split(X):
    
            X_train,X_test=X[train_index],X[test_index]
            Y_train,Y_test=Y[train_index],Y[test_index]
    
            model = LinearRegression()
            # 训练模型
            model.fit(X_train,Y_train)
    
            y_pred_lgb_cat[test_index]=(model.predict(X_test)).reshape(-1,1)
    
    
    
            r2_lgb_cat[num_r2]=r2_score(Y_test, y_pred_lgb_cat[test_index])
            p_lgb_cat[num_r2]=pearson_correlation(Y_test, y_pred_lgb_cat[test_index])
            num_r2=num_r2+1
        r2lgb_cat=r2_lgb_cat.mean()
        plgb_cat=p_lgb_cat.mean()
        rmse_lgb_cat = sqrt(mean_squared_error(dy, y_pred_lgb_cat))
        
    #xgb+lgb+cat-----------------------------------------------------------------------------------   
        X= np.zeros([len_data,3], dtype = float)
        X=np.append(y_pred_mean_xgr[0:len_data],y_pred_mean_LGBR[0:len_data],axis=1)
        X=np.append(X,y_pred_mean_cat[0:len_data],axis=1)
    
        Y=dy[0:len_data]
        y_pred_xgb_lgb_cat=np.zeros([len_data,1], dtype = float)
        r2_xgb_lgb_cat=np.zeros([10,1], dtype = float)
        p_xgb_lgb_cat=np.zeros([10,1], dtype = float)
        KF=KFold(n_splits=10) #建立10折交叉验证方法 查一下KFold函数的参数
        num_r2=0
        for train_index,test_index in KF.split(X):
    
            X_train,X_test=X[train_index],X[test_index]
            Y_train,Y_test=Y[train_index],Y[test_index]
    
            
            model = LinearRegression()
            # 训练模型
            model.fit(X_train,Y_train)
    
            y_pred_xgb_lgb_cat[test_index]=(model.predict(X_test)).reshape(-1,1)
    
    
    
    
            r2_xgb_lgb_cat[num_r2]=r2_score(Y_test, y_pred_xgb_lgb_cat[test_index])
            p_xgb_lgb_cat[num_r2]=pearson_correlation(Y_test, y_pred_xgb_lgb_cat[test_index])
            num_r2=num_r2+1
        r2xgb_lgb_cat=r2_xgb_lgb_cat.mean()
        pxgb_lgb_cat=p_xgb_lgb_cat.mean()
        rmse_xgb_lgb_cat = sqrt(mean_squared_error(dy, y_pred_xgb_lgb_cat))

        # ------------------------------------------------------------------------------


        plt.plot(y_pred_mean_xgr, 'g-', label=str("y_pred_xgr"))
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()
        
        plt.plot(y_pred_mean_LGBR, 'g-', label="y_pred_LGBR")
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()
        
        plt.plot(y_pred_mean_cat, 'g-', label="y_pred_cat")
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()

        plt.plot(y_pred_mean_gp1, 'g-', label=str("y_pred_gp"))
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()
        
        plt.plot(y_pred_xgb_lgb, 'g-', label="y_pred_xgb+lgb")
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()
    
        plt.plot(y_pred_xgb_cat, 'g-', label="y_pred_xgb+cat")
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()
    
        plt.plot(y_pred_lgb_cat, 'g-', label="y_pred_lgb+cat")
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()
        
        plt.plot(y_pred_xgb_lgb_cat, 'g-', label="y_pred_xgb+lgb+cat")
        plt.plot(Y, 'c-', label="real")
        plt.legend()
        plt.show()
        
        w_sheet.write(row,col, order_f1[file_i])
        row=row+1
        w_sheet.write(row,col, rmse_xgr)
        row=row+1
        w_sheet.write(row,col, pxgr)
        row=row+1
        # w_sheet.write(row,col, r2xgr)
        # row=row+1
        w_sheet.write(row,col, rmse_LGBR)
        row=row+1
        w_sheet.write(row,col, pLGBR)
        row=row+1
        # w_sheet.write(row,col, r2LGBR)
        # row=row+1
        w_sheet.write(row,col, rmse_cat)
        row=row+1
        w_sheet.write(row,col, pcat)
        row=row+1
        # w_sheet.write(row,col, r2cat)
        # row=row+1
        w_sheet.write(row, col, rmse_gp)
        row = row + 1
        w_sheet.write(row, col, pgp)
        row = row + 1
        # w_sheet.write(row, col, r2gp)
        # row = row + 1
        w_sheet.write(row,col, rmse_xgb_lgb)
        row=row+1
        w_sheet.write(row,col, pxgb_lgb)
        row=row+1
        # w_sheet.write(row,col, r2xgb_lgb)
        # row=row+1
        w_sheet.write(row,col, rmse_xgb_cat)
        row=row+1
        w_sheet.write(row,col, pxgb_cat)
        row=row+1
        # w_sheet.write(row,col, r2xgb_cat)
        # row=row+1
        
        w_sheet.write(row,col, rmse_lgb_cat)
        row=row+1
        w_sheet.write(row,col, plgb_cat)
        row=row+1
        # w_sheet.write(row,col, r2lgb_cat)
        # row=row+1
        w_sheet.write(row,col, rmse_xgb_lgb_cat)
        row=row+1
        w_sheet.write(row,col, pxgb_lgb_cat)
        row=row+1
        # w_sheet.write(row,col, r2xgb_lgb_cat)
        # row=row+1
        
    
        
        wbook.save(excel_path)#保存文件
        col = col + 1
        row = 0
        wbook.save(excel_path)  # 保存文件
    
    
        # data_dy = pd.DataFrame(dy)
        # data_y_pred_mean_xgr = pd.DataFrame(y_pred_mean_xgr)
        # data_y_pred_mean_LGBR = pd.DataFrame(y_pred_mean_LGBR)
        # data_y_pred_mean_cat = pd.DataFrame(y_pred_mean_cat)
        # data_y_pred_xgb_lgb = pd.DataFrame(y_pred_xgb_lgb)
        # data_y_pred_xgb_cat = pd.DataFrame(y_pred_xgb_cat)
        # data_y_pred_lgb_cat = pd.DataFrame(y_pred_lgb_cat)
        # data_y_pred_xgb_lgb_cat = pd.DataFrame(y_pred_xgb_lgb_cat)
      
        # writer = pd.ExcelWriter(parent+'_y2.xlsx')  # 写入Excel文件
        # data_dy.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data_y_pred_mean_xgr.to_excel(writer, 'page_2', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data_y_pred_mean_LGBR.to_excel(writer, 'page_3', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data_y_pred_mean_cat.to_excel(writer, 'page_4', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data_y_pred_xgb_lgb.to_excel(writer, 'page_5', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data_y_pred_xgb_cat.to_excel(writer, 'page_6', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data_y_pred_lgb_cat.to_excel(writer, 'page_7', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # data_y_pred_xgb_lgb_cat.to_excel(writer, 'page_8', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # writer.save()
    
        # writer.close()
    
import pyttsx3
engine = pyttsx3.init()  # 创建engine并初始化
engine.say("结束")
engine.runAndWait()  # 等待语音播报完毕

     