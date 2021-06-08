def nonlinear_ML(Xtrain, ytrain, Xval, yval, Xtrainval, ytrainval, Xtest, ytest, ps):

    
    # RF 
    param_grid_RF = {
        'bootstrap': [True],
        'max_depth': [1, 2, 3, 4, 5, 6],
        'min_samples_split': [3, 5, 10, 20, 30, 50,80, 100],
        'n_estimators': [300]
    }
    RF = RandomForestRegressor()
    grid_RF = GridSearchCV(estimator=RF, param_grid=param_grid_RF, cv=ps, verbose=2)
    grid_RF.fit(Xtrainval, ytrainval)
    ytest_hat_RF = grid_RF.predict(Xtest) 

    
    # GBRT 
    param_grid_GBRT ={'n_estimators':[300], 
            'learning_rate': [0.01],
            'max_depth':[1, 2], 
            'max_features': np.arange(1, 1001) } 
    GBRT = GradientBoostingRegressor()
    grid_GBRT = GridSearchCV(estimator=GBRT, param_grid=param_grid_GBRT, cv=ps, verbose=2)
    grid_GBRT.fit(Xtrainval, ytrainval)
    ytest_hat_GBRT = grid_GBRT.predict(Xtest) 
    
    
    # NNs
    torch.cuda.empty_cache()
    gc.disable()
    epochs = 50000
    lr_ = 0.005
    batch_size_ = 10000
    XtrainNN = torch.FloatTensor(Xtrain)
    ytrainNN = torch.FloatTensor(ytrain.squeeze())
    XytrainNN = TensorDataset(XtrainNN, ytrainNN)
    XytrainNN = DataLoader(XytrainNN, batch_size = batch_size_, shuffle = True)
    XvalNN = torch.FloatTensor(Xval).to('cuda')
    yvalNN = torch.FloatTensor(yval.squeeze()).to('cuda')
    XtestNN = torch.FloatTensor(Xtest)
    ytestNN = torch.FloatTensor(ytest.squeeze())
    print(XtrainNN.shape,ytrainNN.shape)
    print(XvalNN.shape,yvalNN.shape)
    print(XtestNN.shape,ytestNN.shape)
    gc.disable()
    # NN1 
    class NN1(nn.Module):
        def __init__(self, n_features, NN1_layer1_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN1_layer1_Range)
            self.fc2 = nn.Linear(NN1_layer1_Range, 1)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    NN1_layer1_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    resultsCV = pd.DataFrame(columns=['layer1_Range'])
    for i in range(NN1_layer1_Range.size):            
        num_layer1 = NN1_layer1_Range[i]
        model = NN1(XtrainNN.shape[1], num_layer1).to('cuda')
        # loss function that measures difference between two binary vectors
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
        for epoch in range(epochs):
            loss_batch = []
            oos_loss_batch = []
            for batch_idx, batch_set in enumerate(XytrainNN):
                x_train, y_train = batch_set
                x_train = x_train.to('cuda')
                y_train = y_train.to('cuda')
                gc.disable()
                ytrainNN_hat = model(x_train)
                loss = criterion(torch.squeeze(ytrainNN_hat),y_train)
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                loss_batch.append(loss.cpu().detach().numpy())
                yvalNN_hat = model(XvalNN)
                oos_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
                oos_loss_batch.append(oos_loss.cpu().detach().numpy())     
                print(f'NN1: {i} Epoch: {epoch} Loss: {loss} oosLoss: {oos_loss}')

            oosloss_batch_mean = np.mean(oos_loss_batch)
            if epoch == 0:
                oosloss_batch_mean_past = oosloss_batch_mean
            elif oosloss_batch_mean_past - oosloss_batch_mean > 0:
                    oosloss_batch_mean_past = oosloss_batch_mean
            elif oosloss_batch_mean_past - oosloss_batch_mean <= 0:
                    break
        yvalNN_hat = model(XvalNN)
        val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN).cpu().detach().numpy()
        resultsCV = resultsCV.append({'layer1_Range':num_layer1, 
                                      'CV_Score':val_loss},
                                     ignore_index=True
                                    )
        if i==0:
            temp_val_loss = val_loss
            model_temp = model
        elif temp_val_loss >= val_loss:
                model_temp = model
                temp_val_loss = val_loss
    
    model_temp = model_temp.cpu()
    resultsCV_NN1 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN1)
    ytest_hat_NN1 = model_temp(XtestNN).detach().numpy()
     
    # NN2 
    class NN2(nn.Module):
        def __init__(self, n_features, NN2_layer1_Range, NN2_layer2_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN2_layer1_Range)
            self.fc2 = nn.Linear(NN2_layer1_Range, NN2_layer2_Range)
            self.fc3 = nn.Linear(NN2_layer2_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)        
    NN2_layer1_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    NN2_layer2_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range'])
    for i in range(NN2_layer1_Range.size):
        for j in range(NN2_layer2_Range.size):            
            num_layer1 = NN2_layer1_Range[i]
            num_layer2 = NN2_layer2_Range[j]    
            model = NN2(XtrainNN.shape[1], num_layer1, num_layer2).to('cuda')
            # loss function that measures difference between two binary vectors
            criterion = nn.MSELoss() 
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
            for epoch in range(epochs):
                loss_batch = []
                oos_loss_batch = []
                for batch_idx, batch_set in enumerate(XytrainNN):
                    x_train, y_train = batch_set
                    x_train = x_train.to('cuda')
                    y_train = y_train.to('cuda')
                    gc.disable()
                    ytrainNN_hat = model(x_train)
                    loss = criterion(torch.squeeze(ytrainNN_hat),y_train)
                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    loss_batch.append(loss.cpu().detach().numpy()) 
                    yvalNN_hat = model(XvalNN)
                    oos_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
                    oos_loss_batch.append(oos_loss.cpu().detach().numpy())     
                    print(f'NN2: {i, j} Epoch: {epoch} Loss: {loss} oosLoss: {oos_loss}')
                oosloss_batch_mean = np.mean(oos_loss_batch)
                if epoch == 0:
                    oosloss_batch_mean_past = oosloss_batch_mean
                elif oosloss_batch_mean_past - oosloss_batch_mean > 0:
                        oosloss_batch_mean_past = oosloss_batch_mean
                elif oosloss_batch_mean_past - oosloss_batch_mean <= 0:
                        break
                yvalNN_hat = model(XvalNN)
                val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN).cpu().detach().numpy()
                resultsCV = resultsCV.append({'layer1_Range':num_layer1,
                                          'layer2_Range':num_layer2,
                                          'CV_Score':val_loss},
                                         ignore_index=True
                                        )
            if i==0&j==0:
                temp_val_loss = val_loss
                model_temp = model
            elif temp_val_loss >= val_loss:
                    model_temp = model
                    temp_val_loss = val_loss
    model_temp = model_temp.cpu()
    resultsCV_NN2 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN2)
    ytest_hat_NN2 = model_temp(XtestNN).detach().numpy()








    

    # NN3
    class NN3(nn.Module):
        def __init__(self, n_features, NN3_layer1_Range, NN3_layer2_Range, NN3_layer3_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN3_layer1_Range)
            self.fc2 = nn.Linear(NN3_layer1_Range, NN3_layer2_Range)
            self.fc3 = nn.Linear(NN3_layer2_Range, NN3_layer3_Range)
            self.fc4 = nn.Linear(NN3_layer3_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)        
    NN3_layer1_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    NN3_layer2_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    NN3_layer3_Range = np.array([2, 4, 8, 16, 32, 64, 128])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range'])
    for i in range(NN3_layer1_Range.size):
        for j in range(NN3_layer2_Range.size):            
            for k in range(NN3_layer3_Range.size):            
                num_layer1 = NN3_layer1_Range[i]
                num_layer2 = NN3_layer2_Range[j]    
                num_layer3 = NN3_layer3_Range[k]    
                model = NN3(XtrainNN.shape[1], num_layer1, num_layer2, num_layer3).to('cuda')
                # loss function that measures difference between two binary vectors
                criterion = nn.MSELoss() 
                optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
                for epoch in range(epochs):
                    loss_batch = []
                    oos_loss_batch = []
                    for batch_idx, batch_set in enumerate(XytrainNN):
                        x_train, y_train = batch_set
                        x_train = x_train.to('cuda')
                        y_train = y_train.to('cuda')
                        gc.disable()
                        ytrainNN_hat = model(x_train)
                        loss = criterion(torch.squeeze(ytrainNN_hat),y_train)
                        # backpropagation
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step() 
                        loss_batch.append(loss.cpu().detach().numpy())    
                        yvalNN_hat = model(XvalNN)
                        oos_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN)
                        oos_loss_batch.append(oos_loss.cpu().detach().numpy())     
                        print(f'NN3: {i, j, k} Epoch: {epoch} Loss: {loss} oosLoss: {oos_loss}')
                    oosloss_batch_mean = np.mean(oos_loss_batch)
                    if epoch == 0:
                        oosloss_batch_mean_past = oosloss_batch_mean
                    elif oosloss_batch_mean_past - oosloss_batch_mean > 0:
                            oosloss_batch_mean_past = oosloss_batch_mean
                    elif oosloss_batch_mean_past - oosloss_batch_mean <= 0:
                            break
                    yvalNN_hat = model(XvalNN)
                    val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN).cpu().detach().numpy()
                    resultsCV = resultsCV.append({'layer1_Range':num_layer1,
                                              'layer2_Range':num_layer2,
                                              'layer3_Range':num_layer3,
                                              'CV_Score':val_loss},
                                             ignore_index=True
                                            )
                if i==0&j==0:
                    temp_val_loss = val_loss
                    model_temp = model
                elif temp_val_loss >= val_loss:
                        model_temp = model
                        temp_val_loss = val_loss
    model_temp = model_temp.cpu()
    resultsCV_NN3 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN3)
    ytest_hat_NN3 = model_temp(XtestNN).detach().numpy()
    
    torch.cuda.empty_cache()
    gc.disable()
    return ytest_hat_RF ytest_hat_GBRT ytest_hat_NN1, ytest_hat_NN2, ytest_hat_NN3
