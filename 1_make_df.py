        
def make_df(test_num, df, xlist):
    i = 18 + test_num
    train_begin = 196507
    train_end = 196601 + i*100 -1
    val_begin = 196601 + i*100
    val_end = 196601 + (i+12)*100 - 1
    test_begin = 196601 + (i+12)*100
    test_end = 196601 + (i+13)*100 - 1
    # training set (196507-198312)
    dfT = df[(df.ym>=train_begin) & (df.ym<=train_end)].copy()
    #dfT = dfT.sample(100000, random_state=1)
    # validation set (198401-199512)
    dfV = df[(df.ym>=val_begin) & (df.ym<=val_end)].copy()
    # test set for prediction (199601 - 199612)
    dfP = df[(df.ym>=test_begin) & (df.ym<=test_end)].copy()
    index_test = dfP.index
    Xtrain = dfT[xlist].astype('float32')
    ytrain = dfT.excess_ret.astype('float32')
    Xval = dfV[xlist].astype('float32')
    yval = dfV.excess_ret.astype('float32')
    Xtest = dfP[xlist].astype('float32')
    ytest = dfP.excess_ret.astype('float32')
    return Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test


