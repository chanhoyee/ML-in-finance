        
def make_df(test_num, df, xlist):
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
    index_test = dfP_temp.index
    dfP = dfP_temp.reset_index(drop=True) 
    Xtrain = dfT[xlist]
    ytrain = dfT.excess_ret
    Xval = dfV[xlist]
    yval = dfV.excess_ret
    Xtest = dfP[xlist]
    ytest = dfP.excess_ret
    return Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test

