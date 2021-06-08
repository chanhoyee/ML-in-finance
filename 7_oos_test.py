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
import gc
import matplotlib.pyplot as plt
import statsmodels.api as sm


df_saved = pd.read_csv('df_new_char_proxies_macros_all.csv')

df = df_saved
df = df[df.ym>199600]





models_all = ['ytest_hat_OLS_char_proxies',
'ytest_hat_PCR_char_proxies',
'ytest_hat_PLS_char_proxies',
'ytest_hat_ENet_char_proxies',
'ytest_hat_OLS_char_macros',
'ytest_hat_PCR_char_macros',
'ytest_hat_PLS_char_macros',
'ytest_hat_ENet_char_macros',
'ytest_hat_OLS_char_proxies_macros',
'ytest_hat_PCR_char_proxies_macros',
'ytest_hat_PLS_char_proxies_macros',
'ytest_hat_ENet_char_proxies_macros',
'ytest_hat_GBRT_char_proxies',
'ytest_hat_RF_char_proxies',
'ytest_hat_NN1_char_proxies',
'ytest_hat_NN2_char_proxies',
'ytest_hat_NN3_char_proxies',
'ytest_hat_GBRT_char_macros',
'ytest_hat_RF_char_macros',
'ytest_hat_NN1_char_macros',
'ytest_hat_NN2_char_macros',
'ytest_hat_NN3_char_macros',
'ytest_hat_GBRT_char_proxies_macros',
'ytest_hat_RF_char_proxies_macros',
'ytest_hat_NN1_char_proxies_macros',
'ytest_hat_NN2_char_proxies_macros',
'ytest_hat_NN3_char_proxies_macros']

models_replications = ['ytest_hat_OLS_char_macros',
'ytest_hat_PCR_char_macros',
'ytest_hat_PLS_char_macros',
'ytest_hat_ENet_char_macros',
'ytest_hat_GBRT_char_macros',
'ytest_hat_RF_char_macros',
'ytest_hat_NN1_char_macros',
'ytest_hat_NN2_char_macros',
'ytest_hat_NN3_char_macros']

models_comparison = [['ytest_hat_OLS_char_proxies',
'ytest_hat_OLS_char_macros',
'ytest_hat_OLS_char_proxies_macros'
],['ytest_hat_PCR_char_proxies',
'ytest_hat_PCR_char_macros',
'ytest_hat_PCR_char_proxies_macros'
],['ytest_hat_PLS_char_proxies',
'ytest_hat_PLS_char_macros',
'ytest_hat_PLS_char_proxies_macros'
],['ytest_hat_ENet_char_proxies',
'ytest_hat_ENet_char_macros',
'ytest_hat_ENet_char_proxies_macros'
],['ytest_hat_GBRT_char_proxies',
'ytest_hat_GBRT_char_macros',
'ytest_hat_GBRT_char_proxies_macros'
]['ytest_hat_RF_char_proxies',
'ytest_hat_RF_char_macros',
'ytest_hat_RF_char_proxies_macros'
]['ytest_hat_NN1_char_proxies',
'ytest_hat_NN1_char_macros',
'ytest_hat_NN1_char_proxies_macros'
],['ytest_hat_NN2_char_proxies',
'ytest_hat_NN2_char_macros',
'ytest_hat_NN2_char_proxies_macros'
],['ytest_hat_NN3_char_proxies',
'ytest_hat_NN3_char_macros',
'ytest_hat_NN3_char_proxies_macros'
]]


   
   




#################################
# oos R^2
#################################



column_names = ['OLS', 'PCR', 'PLS', 'ENet', 'GBRT', 'RF', 'NN1', 'NN2', 'NN3']
row_names = ['proxies', 'macros', 'both']
table1_oosR2 = pd.DataFrame(columns = column_names, index = row_names)
for i in range(len(models_comparison)): # OLS, PCR, PLS, ENet, GBRT, RF, NN1, NN2, NN3
    for j in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method = models_comparison[i][j]
        y_hat = df[models_method]
        oos_R_2 = 1-np.nansum(np.square(y_true-y_hat))/np.nansum(np.square(y_true-y_hat+y_hat))
        table1_oosR2.iloc[j,i] = oos_R_2
table1_oosR2.to_excel('table1_oosR2.xlsx')

#################################
# DM test
#################################
column_names = ['OLS', 'PCR', 'PLS', 'ENet', 'GBRT', 'RF', 'NN1', 'NN2', 'NN3']
row_names = ['OLS', 'PCR', 'PLS', 'ENet', 'GBRT', 'RF', 'NN1', 'NN2', 'NN3']
table2_0_DM_replication = pd.DataFrame(columns = column_names, index = row_names)
for i1 in range(len(models_comparison)): # OLS, PCR, PLS, ENet, GBRT, RF, NN1, NN2, NN3
    for i2 in range(len(models_comparison)): # OLS, PCR, PLS, ENet, GBRT, RF, NN1, NN2, NN3
        j = 1 # macros only
        y_true = df.excess_ret
        models_method1 = models_comparison[i1][j]
        models_method2 = models_comparison[i2][j]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_0_DM_replication.iloc[i1,i2] = DM_stat[0]
table2_0_DM_replication.to_excel('table2_0_DM_replication.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_1_DM_OLS = pd.DataFrame(columns = column_names, index = row_names)
i = 0 # OLS
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_1_DM_OLS.iloc[j1,j2] = DM_stat[0]
table2_1_DM_OLS.to_excel('table2_1_DM_OLS.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_2_DM_PCR = pd.DataFrame(columns = column_names, index = row_names)
i = 1 # PCR
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_2_DM_PCR.iloc[j1,j2] = DM_stat[0]
table2_2_DM_PCR.to_excel('table2_2_DM_PCR.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_3_DM_PLS = pd.DataFrame(columns = column_names, index = row_names)
i = 2 # PLS
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_3_DM_PLS.iloc[j1,j2] = DM_stat[0]
table2_3_DM_PLS.to_excel('table2_3_DM_PLS.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_4_DM_ENet = pd.DataFrame(columns = column_names, index = row_names)
i = 3 # ENet
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_4_DM_ENet.iloc[j1,j2] = DM_stat[0]
table2_4_DM_ENet.to_excel('table2_4_DM_ENet.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_5_DM_GBRT = pd.DataFrame(columns = column_names, index = row_names)
i = 4 # GBRT
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_5_DM_GBRT.iloc[j1,j2] = DM_stat[0]
table2_5_DM_GBRT.to_excel('table2_5_DM_GBRT.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_6_DM_RF = pd.DataFrame(columns = column_names, index = row_names)
i = 5 # RF
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_6_DM_RF.iloc[j1,j2] = DM_stat[0]
table2_6_DM_RF.to_excel('table2_6_DM_RF.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_7_DM_NN1 = pd.DataFrame(columns = column_names, index = row_names)
i = 6 # NN1
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_7_DM_NN1.iloc[j1,j2] = DM_stat[0]
table2_7_DM_NN1.to_excel('table2_7_DM_NN1.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_8_DM_NN2 = pd.DataFrame(columns = column_names, index = row_names)
i = 7 # NN2
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_8_DM_NN2.iloc[j1,j2] = DM_stat[0]
table2_8_DM_NN2.to_excel('table2_8_DM_NN2.xlsx')

column_names = ['proxies', 'macros', 'both']
row_names = ['proxies', 'macros', 'both']
table2_8_DM_NN3 = pd.DataFrame(columns = column_names, index = row_names)
i = 8 # NN3
for j1 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
    for j2 in range(len(models_comparison[i])): # proxies, macros, proxies_macros
        y_true = df.excess_ret
        models_method1 = models_comparison[i][j1]
        models_method2 = models_comparison[i][j2]        
        y_hat_1 = df[models_method1]
        y_hat_2 = df[models_method2]        
        df['e_hat_sq_diff'] = np.square(y_true - y_hat_1) - np.square(y_true - y_hat_2)  # squared error difference
        temp = df.groupby('ym')['e_hat_sq_diff'].mean()
        temp = sm.add_constant(temp)
        DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues
        table2_8_DM_NN3.iloc[j1,j2] = DM_stat[0]
table2_8_DM_NN3.to_excel('table2_8_DM_NN3.xlsx')

