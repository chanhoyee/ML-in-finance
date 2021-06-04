
def linear_test(df, xlist, xlist_name):
    df['ytest_hat_OLS' + '_' + xlist_name] = np.nan
    df['ytest_hat_PCR' + '_' + xlist_name] = np.nan
    df['ytest_hat_PLS' + '_' + xlist_name] = np.nan
    df['ytest_hat_ENet' + '_' + xlist_name] = np.nan
    for test_num in np.arange(0,20):
        gc.disable()
        Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test = make_df(test_num, df, xlist)
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
        ytest_hat_OLS, ytest_hat_PCR, ytest_hat_PLS, ytest_hat_ENet = linear_ML(Xtrainval, ytrainval, Xtest, ytest, ps)
        df.update(pd.DataFrame({'ytest_hat_OLS' + '_' + xlist_name: ytest_hat_OLS}, index = index_test))
        df.update(pd.DataFrame({'ytest_hat_PCR' + '_' + xlist_name: ytest_hat_PCR}, index = index_test))
        df.update(pd.DataFrame({'ytest_hat_PLS' + '_' + xlist_name: ytest_hat_PLS[0]}, index = index_test))
        df.update(pd.DataFrame({'ytest_hat_ENet' + '_' + xlist_name: ytest_hat_ENet}, index = index_test))

    return df

