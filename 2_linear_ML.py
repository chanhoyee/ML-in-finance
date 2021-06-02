

def linear_ML(Xtrainval, ytrainval, Xtest, ytest, ps):
    # OLS 
    start = time.time()    
    OLS = LinearRegression()
    OLS.fit(Xtrainval, ytrainval)
    ytest_hat_OLS = OLS.predict(Xtest)
    
    # OLS3 
    
    # PCR    
    param_grid_PCR = {'pca__n_components':np.arange(1,Xtrainval.shape[1])} 
    PCR  = Pipeline([('pca' , PCA()), ('lr' , LinearRegression())])
    grid_PCR = GridSearchCV(estimator=PCR, param_grid=param_grid_PCR, cv=ps, verbose=2)
    grid_PCR.fit(Xtrainval, ytrainval)
    ytest_hat_PCR = grid_PCR.predict(Xtest)
    
    # PLS 
    param_grid_PLS = {'n_components':np.arange(1,Xtrainval.shape[1])} 
    PLS = PLSRegression()
    grid_PLS = GridSearchCV(estimator=PLS, param_grid=param_grid_PLS, cv=ps, verbose=2)
    grid_PLS.fit(Xtrainval, ytrainval)
    ytest_hat_PLS = grid_PLS.predict(Xtest)
    
    # ENet
    param_grid_ENet = {'alpha': np.linspace(start=0.01,stop=0.2,num=20),
                      'l1_ratio' : np.array([.1, .5, .7, .9, .95, .99, 1])}
    ENet = ElasticNet()   # if max_iter is small, it may not converge
    grid_ENet = GridSearchCV(estimator=ENet, param_grid=param_grid_ENet, cv=ps, verbose=2)
    grid_ENet.fit(Xtrainval, ytrainval)
    ytest_hat_ENet = grid_ENet.predict(Xtest)

    return ytest_hat_OLS, ytest_hat_PCR, ytest_hat_PLS, ytest_hat_ENet

