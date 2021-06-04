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
import gc

def get_data():
    stock = pd.read_csv('stock.csv')#.sample(n=1000000, random_state=1)
    gc.disable()
    print('getting stock data done')
    stock = stock.replace('-', np.nan)
    stock = stock.replace('B', np.nan)
    stock = stock.replace('C', np.nan)
    stock_columns = stock.columns
    stock.rename(columns={'PERMNO':'permno', 'SICCD':'siccd', 'TICKER':'ticker', 'COMNAM':'comnam',
                       'PERMCO':'permco', 'HSICCD':'hsiccd', 'PRC':'prc', 'VOL':'vol', 'RET':'ret', 'SPREAD':'spread',
                       'CUSIP':'cusip'}, inplace=True)
    stock = stock.dropna(subset = ['prc'])
    stock['ym'] = pd.to_datetime(round(stock.date/100), format='%Y%m')
    # proxies and macro variables
    proxies = pd.read_excel('Investor_Sentiment_Data_20190327_POST.xlsx', sheet_name = 'DATA')
    proxies = proxies.dropna(axis = 0)
    H = np.array(proxies[['indpro_g', 'consdur_g', 'consnon_g', 'consserv_g', 'employ_g', 'cpi_g', 'recess']])
    H = H.dot(np.linalg.inv(H.T.dot(H))).dot(H.T) 
    orth = np.eye(H.shape[0]) - H
    X = np.array(proxies[['cefd', 'nipo', 'ripo', 'pdnd', 's']])
    proxies[['cefd', 'nipo', 'ripo', 'pdnd', 's']] = orth.dot(X)
    del H, X, orth
    proxies['ym'] = pd.to_datetime(proxies.yearmo, format='%Y%m')
    proxies = proxies.drop(['yearmo'], axis = 1)
    macros = pd.read_excel('PredictorData2020.xlsx', sheet_name = 'Monthly')
    macros['ym'] = pd.to_datetime(macros.yyyymm, format = '%Y%m')
    macros = macros.drop(['yyyymm'], axis = 1)
    # merge stock, proxies, macro dataset
    df_temp = pd.merge(proxies, macros, how = 'left', on=['ym']).reset_index(drop=True)    
    df_temp = pd.merge(stock, df_temp, how = 'left', on = ['ym']).reset_index(drop=True)
    df = df_temp.dropna(subset = ['cefd', 'sic2'])
    del df_temp, stock, proxies, macros
    df['y'] = df['ym'].dt.year
    df['m'] = df['ym'].dt.month
    df['ym'] = df['y']*100 + df['m']
    begin_date = 196507
    end_date = 201806 # 이후는 결측치가 많음 
    gc.disable()
    # predictive variables
    # for shifting    df['ROE(-1)'] = df.sort_values(by=['cusip','year']).groupby('cusip')['ROE'].shift(1)    
    df = df[(df.ym>=begin_date) & (df.ym<=end_date)]
    df = df.replace('-', np.nan)
    df = df.replace('B', np.nan)
    df = df.replace('C', np.nan)
    columns_char_identity = ['permno','date','y','m','ym','siccd','sic2','ticker','comnam','permco','hsiccd','cusip']
    columns_char_float = ['mvel1','beta','betasq','chmom','dolvol','idiovol','indmom','mom1m','mom6m','mom12m','mom36m','pricedelay','turn','absacc','acc','age','agr','bm','bm_ia','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chpmia','convind','currat','depr','divi','divo','dy','egr','ep','gma','grcapx','grltnoa','herf','hire','invest','lev','lgr','mve_ia','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchquick','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga','pchsaleinv','pctacc','ps','quick','rd','rd_mve','rd_sale','realestate','roic','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','tang','tb','aeavol','cash','chtx','cinvest','ear','nincr','roaq','roavol','roeq','rsup','stdacc','stdcf','ms','baspread','ill','maxret','retvol','std_dolvol','std_turn','zerotrade',]
    for i in columns_char_float:
        df[i] = pd.to_numeric(df[i]).astype('float32')
        df[i].fillna(df.groupby('ym')[i].transform('median'), inplace = True)
    for i in columns_char_float:
        df[i] = pd.to_numeric(df[i]).astype('float32')
        df[i].fillna(df[i].median(), inplace = True)
    df['ret'] = df.ret.astype('float32')
    df = pd.concat([df, pd.get_dummies(df['sic2'], prefix = 'sic2')], axis = 1)
    columns_char_industry = list(pd.get_dummies(df['sic2'], prefix = 'sic2').columns)
    df[columns_char_industry] = df[columns_char_industry].astype('float32')
    columns_proxies = ['cefd', 'nipo', 'ripo', 'pdnd', 's']
    df['dp'] = np.log(df['D12'] / df['Index'])  # d/p
    df['ep'] = np.log(df['E12'] / df['Index'])  # e/p
    df['bm'] = df['b/m']                   # B/M
    df['ntis'] = df['ntis']                  # net equity expansion
    df['tbl'] = df['tbl']                   # Treasury-bill rate
    df['tms'] = df['lty'] - df['tbl']    # term-spread
    df['dfy'] = df['BAA'] - df['AAA']    # default-spread
    df['svar'] = df['svar']                  # stock variance
    columns_macros = ['dp','ep','bm','ntis','tbl','tms','dfy','svar']
    df[columns_macros] = df[columns_macros].astype('float32')
    df['excess_ret'] = df.ret - df.tbl
    df = df.dropna(subset = ['excess_ret'])
    columns_inter_char_proxies = []
    gc.disable()
    i = df.shape[1]
    for var_proxy in columns_proxies:
        for var_char_float in columns_char_float:
            i += 1
            columns_inter_char_proxies.append(var_char_float + '_' +var_proxy)
            df[var_char_float + '_' +var_proxy] = df[var_char_float]*df[var_proxy]
            print('column ', i, ' produced')
    columns_inter_char_macros = []
    gc.disable()
    i = df.shape[1]
    for var_macro in columns_macros:
        for var_char_float in columns_char_float:
            i += 1
            columns_inter_char_macros.append(var_char_float + '_' +var_macro)
            df[var_char_float + '_' +var_macro] = df[var_char_float]*df[var_macro]
            print('column ', i, ' produced')            
    gc.disable()
    return df, columns_char_industry, columns_inter_char_proxies, columns_inter_char_macros

