

def linear_ML(Xtrainval, ytrainval, Xtest, ytest, ps):
    # OLS 
    OLS = LinearRegression()
    gc.disable()
    OLS.fit(Xtrainval, ytrainval)
    gc.disable()
    ytest_hat_OLS = OLS.predict(Xtest)
    gc.disable()
    
    # OLS3 
    
    # PCR    
    param_grid_PCR = {'pca__n_components':np.arange(1,min(Xtrainval.shape[1], 30))} 
    gc.disable()
    PCR  = Pipeline([('pca' , PCA()), ('lr' , LinearRegression())])
    gc.disable()
    grid_PCR = GridSearchCV(estimator=PCR, param_grid=param_grid_PCR, cv=ps, verbose=2)
    gc.disable()
    grid_PCR.fit(Xtrainval, ytrainval)
    gc.disable()
    ytest_hat_PCR = grid_PCR.predict(Xtest)
    gc.disable()
    
    # PLS 
    param_grid_PLS = {'n_components':np.arange(1,min(Xtrainval.shape[1], 30))} 
    gc.disable()
    PLS = PLSRegression()
    gc.disable()
    grid_PLS = GridSearchCV(estimator=PLS, param_grid=param_grid_PLS, cv=ps, verbose=2)
    gc.disable()
    grid_PLS.fit(Xtrainval, ytrainval)
    gc.disable()
    ytest_hat_PLS = grid_PLS.predict(Xtest)
    gc.disable()
    
    # ENet
    param_grid_ENet = {'alpha': np.linspace(start=0.01,stop=0.2,num=20),
                      'l1_ratio' : 0.5}
    gc.disable()
    ENet = ElasticNet()   # if max_iter is small, it may not converge
    gc.disable()
    grid_ENet = GridSearchCV(estimator=ENet, param_grid=param_grid_ENet, cv=ps, verbose=2)
    gc.disable()
    grid_ENet.fit(Xtrainval, ytrainval)
    gc.disable()
    ytest_hat_ENet = grid_ENet.predict(Xtest)
    gc.disable()
    return ytest_hat_OLS, ytest_hat_PCR, ytest_hat_PLS, ytest_hat_ENet



