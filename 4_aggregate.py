# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:49:53 2021

@author: chanho
"""

def main():
    pd.set_option('display.max_columns', None)    
    columns_char_identity = ['permno','date','y','m','ym','siccd','sic2','ticker','comnam','permco','hsiccd','cusip']
    columns_char_float = ['mvel1','beta','betasq','chmom','dolvol','idiovol','indmom','mom1m','mom6m','mom12m','mom36m','pricedelay','turn','absacc','acc','age','agr','bm','bm_ia','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chpmia','convind','currat','depr','divi','divo','dy','egr','ep','gma','grcapx','grltnoa','herf','hire','invest','lev','lgr','mve_ia','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchquick','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga','pchsaleinv','pctacc','ps','quick','rd','rd_mve','rd_sale','realestate','roic','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','tang','tb','aeavol','cash','chtx','cinvest','ear','nincr','roaq','roavol','roeq','rsup','stdacc','stdcf','ms','baspread','ill','maxret','retvol','std_dolvol','std_turn','zerotrade',]
    columns_char_industry = ['sic2_1.0', 'sic2_2.0', 'sic2_7.0', 'sic2_8.0', 'sic2_9.0', 'sic2_10.0', 'sic2_12.0', 'sic2_13.0', 'sic2_14.0', 'sic2_15.0', 'sic2_16.0', 'sic2_17.0', 'sic2_20.0', 'sic2_21.0', 'sic2_22.0', 'sic2_23.0', 'sic2_24.0', 'sic2_25.0', 'sic2_26.0', 'sic2_27.0', 'sic2_28.0', 'sic2_29.0', 'sic2_30.0', 'sic2_31.0', 'sic2_32.0', 'sic2_33.0', 'sic2_34.0', 'sic2_35.0', 'sic2_36.0', 'sic2_37.0', 'sic2_38.0', 'sic2_39.0', 'sic2_40.0', 'sic2_41.0', 'sic2_42.0', 'sic2_44.0', 'sic2_45.0', 'sic2_46.0', 'sic2_47.0', 'sic2_48.0', 'sic2_49.0', 'sic2_50.0', 'sic2_51.0', 'sic2_52.0', 'sic2_53.0', 'sic2_54.0', 'sic2_55.0', 'sic2_56.0', 'sic2_57.0', 'sic2_58.0', 'sic2_59.0', 'sic2_60.0', 'sic2_61.0', 'sic2_62.0', 'sic2_63.0', 'sic2_64.0', 'sic2_65.0', 'sic2_67.0', 'sic2_70.0', 'sic2_72.0', 'sic2_73.0', 'sic2_75.0', 'sic2_76.0', 'sic2_78.0', 'sic2_79.0', 'sic2_80.0', 'sic2_81.0', 'sic2_82.0', 'sic2_83.0', 'sic2_84.0', 'sic2_86.0', 'sic2_87.0', 'sic2_89.0', 'sic2_99.0']
    columns_proxies = ['cefd', 'nipo', 'ripo', 'pdnd', 's']
    columns_macros = ['dp','ep','bm','ntis','tbl','tms','dfy','svar']
    # load data
    df = get_data()
    # summary statistics
    # sumStat(df)    
    # make sample dataset
    
    
   
    
    xlist = columns_proxies
    for test_num in np.arange(23):
        Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test = make_df(test_num, df, xlist)

        scaler = StandardScaler()
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xval = scaler.transform(Xval)
        Xtest = scaler.transform(Xtest)
        ytrain = np.asarray(ytrain).reshape(-1,1)
        yval = np.asarray(yval).reshape(-1,1)
        ytest = np.asarray(ytest).reshape(-1,1)
        Xtrainval = np.append(Xtrain, Xval, axis = 0)
        ytrainval = np.append(ytrain, yval, axis = 0).reshape(-1,1)
        ps = PredefinedSplit(test_fold=np.append(np.repeat(-1, Xtrain.shape[0]), np.ones(Xval.shape[0])))
        
        ytest_hat_OLS, ytest_hat_PCR, ytest_hat_PLS, ytest_hat_ENet = linear_ML(Xtrainval, ytrainval, Xtest, ytest, ps)
        ytest_hat_NN1, ytest_hat_NN2, ytest_hat_NN3, ytest_hat_NN4, ytest_hat_NN5 = nonlinear_ML(Xtrain, ytrain, Xval, yval, Xtrainval, ytrainval, Xtest, ytest, ps)
        df['ytest_hat_OLS_columns_proxies'] = pd.DataFrame(ytest_hat_OLS, index = index_test)
        df['ytest_hat_PCR_columns_proxies'] = pd.DataFrame(ytest_hat_PCR, index = index_test)
        df['ytest_hat_PLS_columns_proxies'] = pd.DataFrame(ytest_hat_PLS, index = index_test)
        df['ytest_hat_ENet_columns_proxies'] = pd.DataFrame(ytest_hat_ENet, index = index_test)
        df['ytest_hat_NN1_columns_proxies'] = pd.DataFrame(ytest_hat_NN1, index = index_test)
        df['ytest_hat_NN2_columns_proxies'] = pd.DataFrame(ytest_hat_NN2, index = index_test)
        df['ytest_hat_NN3_columns_proxies'] = pd.DataFrame(ytest_hat_NN3, index = index_test)
        df['ytest_hat_NN4_columns_proxies'] = pd.DataFrame(ytest_hat_NN4, index = index_test)
        df['ytest_hat_NN5_columns_proxies'] = pd.DataFrame(ytest_hat_NN5, index = index_test)

    xlist = columns_macros
    for test_num in np.arange(23):
        Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test = make_df(test_num, df, xlist)

        scaler = StandardScaler()
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xval = scaler.transform(Xval)
        Xtest = scaler.transform(Xtest)
        ytrain = np.asarray(ytrain).reshape(-1,1)
        yval = np.asarray(yval).reshape(-1,1)
        ytest = np.asarray(ytest).reshape(-1,1)
        Xtrainval = np.append(Xtrain, Xval, axis = 0)
        ytrainval = np.append(ytrain, yval, axis = 0).reshape(-1,1)
        ps = PredefinedSplit(test_fold=np.append(np.repeat(-1, Xtrain.shape[0]), np.ones(Xval.shape[0])))
        
        ytest_hat_OLS, ytest_hat_PCR, ytest_hat_PLS, ytest_hat_ENet = linear_ML(Xtrainval, ytrainval, Xtest, ytest)
        ytest_hat_NN1, ytest_hat_NN2, ytest_hat_NN3, ytest_hat_NN4, ytest_hat_NN5 = nonlinear_ML(Xtrain, ytrain, Xval, yval, Xtrainval, ytrainval, Xtest, ytest)
        df['ytest_hat_OLS_columns_macros'] = pd.DataFrame(ytest_hat_OLS, index = index_test)
        df['ytest_hat_PCR_columns_macros'] = pd.DataFrame(ytest_hat_PCR, index = index_test)
        df['ytest_hat_PLS_columns_macros'] = pd.DataFrame(ytest_hat_PLS, index = index_test)
        df['ytest_hat_ENet_columns_macros'] = pd.DataFrame(ytest_hat_ENet, index = index_test)
        df['ytest_hat_NN1_columns_macros'] = pd.DataFrame(ytest_hat_NN1, index = index_test)
        df['ytest_hat_NN2_columns_macros'] = pd.DataFrame(ytest_hat_NN2, index = index_test)
        df['ytest_hat_NN3_columns_macros'] = pd.DataFrame(ytest_hat_NN3, index = index_test)
        df['ytest_hat_NN4_columns_macros'] = pd.DataFrame(ytest_hat_NN4, index = index_test)
        df['ytest_hat_NN5_columns_macros'] = pd.DataFrame(ytest_hat_NN5, index = index_test)

    xlist = columns_proxies + columns_macros
    for test_num in np.arange(23):
        Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test = make_df(test_num, df, xlist)

        scaler = StandardScaler()
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xval = scaler.transform(Xval)
        Xtest = scaler.transform(Xtest)
        ytrain = np.asarray(ytrain).reshape(-1,1)
        yval = np.asarray(yval).reshape(-1,1)
        ytest = np.asarray(ytest).reshape(-1,1)
        Xtrainval = np.append(Xtrain, Xval, axis = 0)
        ytrainval = np.append(ytrain, yval, axis = 0).reshape(-1,1)
        ps = PredefinedSplit(test_fold=np.append(np.repeat(-1, Xtrain.shape[0]), np.ones(Xval.shape[0])))
        
        ytest_hat_OLS, ytest_hat_PCR, ytest_hat_PLS, ytest_hat_ENet = linear_ML(Xtrainval, ytrainval, Xtest, ytest)
        ytest_hat_NN1, ytest_hat_NN2, ytest_hat_NN3, ytest_hat_NN4, ytest_hat_NN5 = nonlinear_ML(Xtrain, ytrain, Xval, yval, Xtrainval, ytrainval, Xtest, ytest)
        df['ytest_hat_OLS_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_OLS, index = index_test)
        df['ytest_hat_PCR_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_PCR, index = index_test)
        df['ytest_hat_PLS_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_PLS, index = index_test)
        df['ytest_hat_ENet_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_ENet, index = index_test)
        df['ytest_hat_NN1_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_NN1, index = index_test)
        df['ytest_hat_NN2_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_NN2, index = index_test)
        df['ytest_hat_NN3_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_NN3, index = index_test)
        df['ytest_hat_NN4_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_NN4, index = index_test)
        df['ytest_hat_NN5_columns_proxiesmacros'] = pd.DataFrame(ytest_hat_NN5, index = index_test)
        
        df.to_csv('df_new_subsamples.csv', index=False)

        
    return df
    
    
df = main()    
df.to_csv('df_new_allsamples.csv', index=False)
