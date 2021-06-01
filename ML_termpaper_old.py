# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:31:10 2021

@author: chanho
"""

# get ready to start
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import os
os.chdir('G:/내 드라이브/대학원/2021-1 재무기계학습/term paper')
import time
import datetime
import calendar
import pandas as pd
import numpy as np
import pickle, gzip, os
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import torch
torch.manual_seed(42)            
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

def main():
    pd.set_option('display.max_columns', None)    
    # load data
    df = getData(19650701,20180630)
    # summary statistics
    sumStat(df)    
    # make sample dataset
    test_fold, Xtrain, ytrain, Xtest, ytest = makeDf(df)        
    # predict default using machine learning
    tdf = doML(test_fold, Xtrain, ytrain, Xtest, ytest)    
    tdf.to_excel('Table6Result.xlsx', index=False)
    print('Table 6 (predicting 2009)')
    print(tdf.round(1))    
    print('')


def doML(Xtrain, ytrain, Xval, yval, Xtest, ytest):
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xval = scaler.transform(Xval)
    Xtest = scaler.transform(Xtest)
    ytrain = np.asarray(ytrain)
    yval = np.asarray(yval)
    ytest = np.asarray(ytest)
    Xtrainval = np.append(Xtrain, Xval, axis = 0)
    ytrainval = np.append(ytrain, yval, axis = 0)
    ps = PredefinedSplit(test_fold=np.append(np.repeat(-1, Xtrain.shape[0]), np.ones(Xval.shape[0])))

    # OLS 
    start = time.time()    
    OLS = LinearRegression()
    OLS.fit(Xtrainval, ytrainval)
    OLS.predict(Xtest)
    # OLS3 

    # PCR    
    start = time.time()    
    param_grid_PCR = {'pca__n_components':np.arange(1,Xtrain.shape[1])} 
    PCR  = Pipeline([('pca' , PCA()), ('lr' , LinearRegression())])
    grid_PCR = GridSearchCV(estimator=PCR, param_grid=param_grid_PCR, cv=ps, verbose=2)
    grid_PCR.fit(Xtrainval, ytrainval)
    grid_PCR.predict(Xtest)
    print("time :", time.time() - start)

    # PLS 
    start = time.time()    
    param_grid_PLS = {'n_components':np.arange(1,Xtrain.shape[1])} 
    PLS = PLSRegression()
    grid_PLS = GridSearchCV(estimator=PLS, param_grid=param_grid_PLS, cv=ps, verbose=2)
    grid_PLS.fit(Xtrainval, ytrainval)
    grid_PLS.predict(Xtest)
    print("time :", time.time() - start)
    
    # ENet
    start = time.time()    
    param_grid_ENet = {'alpha': np.linspace(start=0.01,stop=0.2,num=20),
                      'l1_ratio' : np.array([.1, .5, .7, .9, .95, .99, 1])}
    ENet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)   # if max_iter is small, it may not converge
    grid_ENet = GridSearchCV(estimator=ENet, param_grid=param_grid_ENet, cv=ps, verbose=2)
    grid_ENet.fit(Xtrainval, ytrainval)
    grid_ENet.predict(Xtest)
    print("time :", time.time() - start)

    # GLM 
    
    # RF 
    start = time.time()    
    param_grid_RF = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [10],
        'min_samples_split': [10],
        'n_estimators': [100, 200, 300, 1000]
    }
    RF = RandomForestRegressor()
    grid_RF = GridSearchCV(estimator=RF, param_grid=param_grid_RF, cv=ps, verbose=2)
    grid_RF.fit(Xtrainval, ytrainval)
    grid_RF.predict(Xtest) 
    print("time :", time.time() - start)

    # GBRT 
    start = time.time()    
    param_grid_GBRT ={'n_estimators':[100, 200, 300, 1000], 
            'learning_rate': [0.001],
            'max_depth':[80, 90, 100, 110], 
            'min_samples_leaf':[10], 
            'max_features':[1.0] } 
    GBRT = GradientBoostingRegressor()
    grid_GBRT = GridSearchCV(estimator=GBRT, param_grid=param_grid_GBRT, cv=ps, verbose=2)
    grid_GBRT.fit(Xtrainval, ytrainval)
    grid_GBRT.predict(Xtest) 
    print("time :", time.time() - start)

    
    # NNs
    XtrainNN = torch.FloatTensor(Xtrain)
    ytrainNN = torch.FloatTensor(ytrain.squeeze())
    XvalNN = torch.FloatTensor(Xval)
    yvalNN = torch.FloatTensor(yval.squeeze())
    XtestNN = torch.FloatTensor(Xtest)
    ytestNN = torch.FloatTensor(ytest.squeeze())
    print(XtrainNN.shape,ytrainNN.shape)
    print(XvalNN.shape,yvalNN.shape)
    print(XtestNN.shape,ytestNN.shape)
    epochs = 10000

    # NN1 
    class NN1(nn.Module):
        def __init__(self, n_features, NN1_layer1_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN1_layer1_Range)
            self.fc2 = nn.Linear(NN1_layer1_Range, 1)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    start = time.time()    
    NN1_layer1_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    resultsCV = pd.DataFrame(columns=['layer1_Range'])
    for i in range(NN1_layer1_Range.size):            
        num_layer1 = NN1_layer1_Range[i]
        model = NN1(XtrainNN.shape[1], num_layer1)
        # loss function that measures difference between two binary vectors
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            ytrainNN_hat = model(XtrainNN)
            loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
            if epochs % 100 == 0:
                print(f'NN1: {i} Epoch: {epoch} Loss: {loss}')
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
        yvalNN_hat = model(XvalNN)
        val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
        resultsCV = resultsCV.append({'layer1_Range':num_layer1, 
                                      'CV_Score':val_loss.detach().numpy()},
                                     ignore_index=True
                                    )
    resultsCV_NN1 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN1)
    print("time :", time.time() - start)

    # NN2 
    class NN2(nn.Module):
        def __init__(self, n_features, NN2_layer1_Range, NN2_layer2_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN2_layer1_Range)
            self.fc2 = nn.Linear(NN2_layer1_Range, NN2_layer2_Range)
            self.fc3 = nn.Linear(NN2_layer2_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)        
    start = time.time()    
    NN2_layer1_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    NN2_layer2_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range'])
    for i in range(NN2_layer1_Range.size):
        for j in range(NN2_layer2_Range.size):            
            num_layer1 = NN2_layer1_Range[i]
            num_layer2 = NN2_layer2_Range[j]    
            model = NN2(XtrainNN.shape[1], num_layer1, num_layer2)
            # loss function that measures difference between two binary vectors
            criterion = nn.MSELoss() 
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(epochs):
                ytrainNN_hat = model(XtrainNN)
                loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                if epochs % 100 == 0:
                    print(f'NN2: {i,j} Epoch: {epoch} Loss: {loss}')
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            yvalNN_hat = model(XvalNN)
            val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
            resultsCV = resultsCV.append({'layer1_Range':num_layer1, 
                                          'layer2_Range':num_layer2, 
                                          'CV_Score':val_loss.detach().numpy()},
                                         ignore_index=True
                                        )
    resultsCV_NN2 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN2)
    print("time :", time.time() - start)

    # NN3 
    class NN3(nn.Module):
        def __init__(self, n_features, NN3_layer1_Range, NN3_layer2_Range, NN3_layer3_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN3_layer1_Range)
            self.fc2 = nn.Linear(NN3_layer1_Range, NN3_layer2_Range)
            self.fc3 = nn.Linear(NN3_layer2_Range, NN3_layer3_Range)
            self.fc4 = nn.Linear(NN3_layer3_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)        
    start = time.time()    
    NN3_layer1_Range = np.array([2, 4, 8, 16, 32, 64])
    NN3_layer2_Range = np.array([2, 4, 8, 16, 32, 64])
    NN3_layer3_Range = np.array([2, 4, 8, 16, 32, 64])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range', 'layer3_Range'])
    for i in range(NN3_layer1_Range.size):
        for j in range(NN3_layer2_Range.size):            
            for k in range(NN3_layer3_Range.size):            
                num_layer1 = NN3_layer1_Range[i]
                num_layer2 = NN3_layer2_Range[j]    
                num_layer3 = NN3_layer3_Range[k]    
                model = NN3(XtrainNN.shape[1], num_layer1, num_layer2, num_layer3)
                # loss function that measures difference between two binary vectors
                criterion = nn.MSELoss() 
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                for epoch in range(epochs):
                    ytrainNN_hat = model(XtrainNN)
                    loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                    if epochs % 100 == 0:
                        print(f'NN3: {i,j,k} Epoch: {epoch} Loss: {loss}')
                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                yvalNN_hat = model(XvalNN)
                val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
                resultsCV = resultsCV.append({'layer1_Range':num_layer1, 
                                              'layer2_Range':num_layer2, 
                                              'layer3_Range':num_layer3, 
                                              'CV_Score':val_loss.detach().numpy()},
                                             ignore_index=True
                                            )
    resultsCV_NN3 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN3)
    print("time :", time.time() - start)

    # NN4 
    class NN4(nn.Module):
        def __init__(self, n_features, NN4_layer1_Range, NN4_layer2_Range, NN4_layer3_Range, NN4_layer4_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN4_layer1_Range)
            self.fc2 = nn.Linear(NN4_layer1_Range, NN4_layer2_Range)
            self.fc3 = nn.Linear(NN4_layer2_Range, NN4_layer3_Range)
            self.fc4 = nn.Linear(NN4_layer3_Range, NN4_layer4_Range)
            self.fc5 = nn.Linear(NN4_layer4_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return self.fc5(x)        
    start = time.time()    
    NN4_layer1_Range = np.array([2, 4, 8, 16, 32])
    NN4_layer2_Range = np.array([2, 4, 8, 16, 32])
    NN4_layer3_Range = np.array([2, 4, 8, 16, 32])
    NN4_layer4_Range = np.array([2, 4, 8, 16, 32])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range', 'layer3_Range', 'layer4_Range'])
    for i in range(NN4_layer1_Range.size):
        for j in range(NN4_layer2_Range.size):            
            for k in range(NN4_layer3_Range.size):            
                for l in range(NN4_layer4_Range.size):            
                    num_layer1 = NN4_layer1_Range[i]
                    num_layer2 = NN4_layer2_Range[j]    
                    num_layer3 = NN4_layer3_Range[k]    
                    num_layer4 = NN4_layer4_Range[l]    
                    model = NN4(XtrainNN.shape[1], num_layer1, num_layer2, num_layer3, num_layer4)
                    # loss function that measures difference between two binary vectors
                    criterion = nn.MSELoss() 
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    for epoch in range(epochs):
                        ytrainNN_hat = model(XtrainNN)
                        loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                        if epochs % 100 == 0:
                            print(f'NN4: {i,j,k,l} Epoch: {epoch} Loss: {loss}')
                        # backpropagation
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    yvalNN_hat = model(XvalNN)
                    val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
                    resultsCV = resultsCV.append({'layer1_Range':num_layer1, 
                                                  'layer2_Range':num_layer2, 
                                                  'layer3_Range':num_layer3, 
                                                  'layer4_Range':num_layer4, 
                                                  'CV_Score':val_loss.detach().numpy()},
                                                 ignore_index=True
                                                )
    resultsCV_NN4 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN4)
    print("time :", time.time() - start)
    
    # NN5
    class NN5(nn.Module):
        def __init__(self, n_features, NN5_layer1_Range, NN5_layer2_Range, NN5_layer3_Range, NN5_layer4_Range, NN5_layer5_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN5_layer1_Range)
            self.fc2 = nn.Linear(NN5_layer1_Range, NN5_layer2_Range)
            self.fc3 = nn.Linear(NN5_layer2_Range, NN5_layer3_Range)
            self.fc4 = nn.Linear(NN5_layer3_Range, NN5_layer4_Range)
            self.fc5 = nn.Linear(NN5_layer4_Range, NN5_layer5_Range)
            self.fc6 = nn.Linear(NN5_layer5_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            return self.fc6(x)        
    start = time.time()    
    NN5_layer1_Range = np.array([2, 4, 8, 16, 32])
    NN5_layer2_Range = np.array([2, 4, 8, 16, 32])
    NN5_layer3_Range = np.array([2, 4, 8, 16, 32])
    NN5_layer4_Range = np.array([2, 4, 8, 16, 32])
    NN5_layer5_Range = np.array([2, 4, 8, 16, 32])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range', 'layer3_Range', 'layer4_Range', 'layer5_Range'])
    for i in range(NN5_layer1_Range.size):
        for j in range(NN5_layer2_Range.size):            
            for k in range(NN5_layer3_Range.size):            
                for l in range(NN5_layer4_Range.size):            
                    for m in range(NN5_layer5_Range.size):            
                        num_layer1 = NN5_layer1_Range[i]
                        num_layer2 = NN5_layer2_Range[j]    
                        num_layer3 = NN5_layer3_Range[k]    
                        num_layer4 = NN5_layer4_Range[l]    
                        num_layer5 = NN5_layer5_Range[m]    
                        model = NN5(XtrainNN.shape[1], num_layer1, num_layer2, num_layer3, num_layer4, num_layer5)
                        # loss function that measures difference between two binary vectors
                        criterion = nn.MSELoss() 
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        for epoch in range(epochs):
                            ytrainNN_hat = model(XtrainNN)
                            loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                            if epochs % 100 == 0:
                                print(f'NN5: {i,j,k,l,m} Epoch: {epoch} Loss: {loss}')
                            # backpropagation
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        yvalNN_hat = model(XvalNN)
                        val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
                        resultsCV = resultsCV.append({'layer1_Range':num_layer1, 
                                                      'layer2_Range':num_layer2, 
                                                      'layer3_Range':num_layer3, 
                                                      'layer4_Range':num_layer4, 
                                                      'layer5_Range':num_layer5, 
                                                      'CV_Score':val_loss.detach().numpy()},
                                                     ignore_index=True
                                                    )
    resultsCV_NN5 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN5)
    print("time :", time.time() - start)

def makeDf(test_num, df, xlist):
    xlist = ['mom1m','mom6m','bm']
    test_num = 0 # 부터 22까지?
    i = 18 + test_num
    train_begin = 196507
    train_end = 196601 + i*100 -1
    val_begin = 196601 + i*100
    val_end = 196601 + (i+12)*100 - 1
    test_begin = 196601 + (i+12)*100
    test_end = 196601 + (i+13)*100 - 1
    # training set (196507-198312)
    dfT_temp = df[(df.ym>=train_begin) & (df.ym<=train_end)].copy()
    dfT = dfT_temp.reset_index(drop=True) 
    # validation set (198401-199512)
    dfV_temp = df[(df.ym>=val_begin) & (df.ym<=val_end)].copy()
    dfV = dfV_temp.reset_index(drop=True) 
    # test set for prediction (199601 - 199612)
    dfP_temp = df[(df.ym>=test_begin) & (df.ym<=test_end)].copy()
    dfP = dfP_temp.reset_index(drop=True) 
    Xtrain = dfT[xlist]
    ytrain = dfT.ret
    Xval = dfV[xlist]
    yval = dfV.ret
    Xtest = dfP[xlist]
    ytest = dfP.ret    
    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def getData():
    # merged data of stock from CRSP and Xiu's homepage
    stock = pd.read_csv('stock.csv') # 54 seconds
    stock_columns = stock.columns
    stock.rename(columns={'PERMNO':'permno', 'SICCD':'siccd', 'TICKER':'ticker', 'COMNAM':'comnam',
                       'PERMCO':'permco', 'HSICCD':'hsiccd', 'PRC':'prc', 'VOL':'vol', 'RET':'ret', 'SPREAD':'spread',
                       'CUSIP':'cusip'}, inplace=True)
    stock = stock.dropna(subset = ['prc'])
    stock['ym'] = pd.to_datetime(round(stock.date/100), format='%Y%m')
    # proxies and macro variables
    proxies = pd.read_excel('Investor_Sentiment_Data_20190327_POST.xlsx', sheet_name = 'DATA')
    proxies['ym'] = pd.to_datetime(proxies.yearmo, format='%Y%m')
    proxies = proxies.drop(['yearmo'], axis = 1)
    macros = pd.read_excel('PredictorData2020.xlsx', sheet_name = 'Monthly')
    macros['ym'] = pd.to_datetime(macros.yyyymm, format = '%Y%m')
    macros = macros.drop(['yyyymm'], axis = 1)
    # merge stock, proxies, macro dataset
    df_temp = pd.merge(proxies, macros, how = 'left', on=['ym']).reset_index(drop=True)    
    start = time.time()    
    df_temp = pd.merge(stock, df_temp, how = 'left', on = ['ym']).reset_index(drop=True)
    df = df_temp.dropna(subset = ['cefd'])
    print("time :", time.time() - start)
    df['y'] = df['ym'].dt.year
    df['m'] = df['ym'].dt.month
    df['ym'] = df['y']*100 + df['m']
    begin_date = 196507
    end_date = 201806 # 이후는 결측치가 많음 
    # predictive variables
    # for shifting    df['ROE(-1)'] = df.sort_values(by=['cusip','year']).groupby('cusip')['ROE'].shift(1)    
    df = df[(df.ym>=begin_date) & (df.ym<=end_date)]
    df_columns = df.columns.difference(['permno','date','siccd','ticker','comnam','permco','hsiccd','cusip','sic2','y','m','ym', 'price'])    
    df = df.replace('-', np.nan)
    df = df.replace('C', np.nan)
    for i in df_columns:
        df[i] = pd.to_numeric(df[i])
        df[i].fillna(df.groupby('ym')[i].transform('median'), inplace = True)
    return df


