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


df = pd.read_csv('df_new_char_proxies_macros_small.csv')





#################################
# plotting weight pf, equal pf
#################################
df['vw'] = df.mvel1/(df.groupby('ym')['mvel1'].transform('sum'))
df['vw_excess_ret'] = df.excess_ret*df.mv
df['ew'] = 1/df.groupby('ym')['mvel1'].transform('count')
df['ew_excess_ret'] = df.excess_ret*df.ew



method = 'ytest_hat_PCR_char_proxies'

vw_pf = df.sort_values(by=['ym',method]).groupby('ym')['vw_excess_ret'].sum()
ew_pf = df.sort_values(by=['ym',method]).groupby('ym')['ew_excess_ret'].sum()
plt.plot(vw_pf)
plt.plot(ew_pf)
plt.show()


#################################
# oos R^2
#################################

y_true = df.excess_ret
method = 'ytest_hat_PCR_char_proxies'
y_hat = df[method]

oos_R_2 = 1-np.nansum(np.square(y_true-y_hat))/np.nansum(np.square(y_true-y_hat+y_hat))



#################################
# DM test
#################################

method_1 = 'ytest_hat_OLS_char_proxies'
method_2 = 'ytest_hat_PCR_char_proxies'

y_true = df.excess_ret
y_hat_1 = df[method_1]
y_hat_2 = df[method_2]

df['e_hat_sq_diff'] = np.square(df.excess_ret - df[method_1]) - np.square(df.excess_ret - df[method_2])  # squared error difference
temp = sm.add_constant(df['e_hat_sq_diff'])
DM_stat = sm.OLS(temp.e_hat_sq_diff, temp.const, missing = 'drop').fit().get_robustcov_results(cov_type='HAC',maxlags=1).tvalues



