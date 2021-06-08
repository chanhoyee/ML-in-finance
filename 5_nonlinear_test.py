



def nonlinear_test(df, xlist, xlist_name):
    df['ytest_hat_RF' + '_' + xlist_name] = np.nan
    df['ytest_hat_GBRT' + '_' + xlist_name] = np.nan
    df['ytest_hat_NN1' + '_' + xlist_name] = np.nan
    df['ytest_hat_NN2' + '_' + xlist_name] = np.nan
    df['ytest_hat_NN3' + '_' + xlist_name] = np.nan
    for test_num in np.arange(20):
        gc.disable()
        Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test = make_df(test_num, df, xlist)
        scaler = StandardScaler()
        scaler.fit(Xtrain)
        Xtrain = Xtrain
        Xtrain = scaler.transform(Xtrain)
        Xval = scaler.transform(Xval)
        Xtest = scaler.transform(Xtest)
        ytrain = np.asarray(ytrain)
        yval = np.asarray(yval)
        ytest = np.asarray(ytest)
        Xtrainval = np.append(Xtrain, Xval, axis = 0)
        ytrainval = np.append(ytrain, yval, axis = 0)
        ps = PredefinedSplit(test_fold=np.append(np.repeat(-1, Xtrain.shape[0]), np.ones(Xval.shape[0])))  
        ytest_hat_RF ytest_hat_GBRT ytest_hat_NN1, ytest_hat_NN2 = nonlinear_ML(Xtrain, ytrain, Xval, yval, Xtrainval, ytrainval, Xtest, ytest, ps)
        df.update(pd.DataFrame({'ytest_hat_RF' + '_' + xlist_name: ytest_hat_NN1}, index = index_test))
        df.update(pd.DataFrame({'ytest_hat_GBRT' + '_' + xlist_name: ytest_hat_NN1}, index = index_test))
        df.update(pd.DataFrame({'ytest_hat_NN1' + '_' + xlist_name: ytest_hat_NN1}, index = index_test))
        df.update(pd.DataFrame({'ytest_hat_NN2' + '_' + xlist_name: ytest_hat_NN2}, index = index_test))
        df.update(pd.DataFrame({'ytest_hat_NN3' + '_' + xlist_name: ytest_hat_NN3}, index = index_test))
    return df
