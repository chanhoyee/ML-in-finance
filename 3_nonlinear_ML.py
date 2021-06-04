def nonlinear_ML(Xtrain, ytrain, Xval, yval, Xtrainval, ytrainval, Xtest, ytest, ps):

    '''
    # RF 
    param_grid_RF = {
        'bootstrap': [True],
        'max_depth': [3],
        'max_features': [2],
        'min_samples_leaf': [2],
        'min_samples_split': [2],
        'n_estimators': [2]
    }
    RF = RandomForestRegressor()
    grid_RF = GridSearchCV(estimator=RF, param_grid=param_grid_RF, cv=ps, verbose=2)
    grid_RF.fit(Xtrainval, ytrainval)
    ytest_hat_RF = grid_RF.predict(Xtest) 

    
    # GBRT 
    param_grid_GBRT ={'n_estimators':[2], 
            'learning_rate': [0.001],
            'max_depth':[3], 
            'min_samples_leaf':[2], 
            'max_features':[1.0] } 
    GBRT = GradientBoostingRegressor()
    grid_GBRT = GridSearchCV(estimator=GBRT, param_grid=param_grid_GBRT, cv=ps, verbose=2)
    grid_GBRT.fit(Xtrainval, ytrainval)
    ytest_hat_GBRT = grid_GBRT.predict(Xtest) 
    '''
    # NNs
    XtrainNN = torch.FloatTensor(Xtrain).to('cuda')
    ytrainNN = torch.FloatTensor(ytrain.squeeze()).to('cuda')
    XvalNN = torch.FloatTensor(Xval).to('cuda')
    yvalNN = torch.FloatTensor(yval.squeeze()).to('cuda')
    XtestNN = torch.FloatTensor(Xtest).to('cuda')
    ytestNN = torch.FloatTensor(ytest.squeeze()).to('cuda')
    print(XtrainNN.shape,ytrainNN.shape)
    print(XvalNN.shape,yvalNN.shape)
    print(XtestNN.shape,ytestNN.shape)
    epochs = 50000
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
    NN1_layer1_Range = np.array([2, 4, 8, 16, 32, 64])
    resultsCV = pd.DataFrame(columns=['layer1_Range'])
    for i in range(NN1_layer1_Range.size):            
        num_layer1 = NN1_layer1_Range[i]
        model = NN1(XtrainNN.shape[1], num_layer1).to('cuda')
        # loss function that measures difference between two binary vectors
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            gc.disable()
            ytrainNN_hat = model(XtrainNN)
            loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
            print(f'NN1: {i} Epoch: {epoch} Loss: {loss}')
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            if epoch == 0:
                loss_past = loss
            elif abs(loss_past-loss)>1e-9:
                    loss_past = loss
            elif abs(loss_past-loss)<=1e-9:
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
    resultsCV_NN1 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN1)
    ytest_hat_NN1 = model_temp(XtestNN).cpu().detach().numpy()
     
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
    NN2_layer1_Range = np.array([2, 4, 8, 16, 32, 64])
    NN2_layer2_Range = np.array([2, 4, 8, 16, 32, 64])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range'])
    for i in range(NN2_layer1_Range.size):
        for j in range(NN2_layer2_Range.size):            
            num_layer1 = NN2_layer1_Range[i]
            num_layer2 = NN2_layer2_Range[j]    
            model = NN2(XtrainNN.shape[1], num_layer1, num_layer2).to('cuda')
            # loss function that measures difference between two binary vectors
            criterion = nn.MSELoss() 
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(epochs):
                gc.disable()
                ytrainNN_hat = model(XtrainNN)
                loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                print(f'NN2: {i,j} Epoch: {epoch} Loss: {loss}')
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch == 0:
                    loss_past = loss
                elif abs(loss_past-loss)>1e-9:
                        loss_past = loss
                elif abs(loss_past-loss)<=1e-9:
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
    resultsCV_NN2 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN2)
    ytest_hat_NN2 = model_temp(XtestNN).cpu().detach().numpy()
    
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
    NN3_layer1_Range = np.array([2, 4, 8, 16, 32, 64])
    NN3_layer2_Range = np.array([2, 4, 8, 16, 32, 64])
    NN3_layer3_Range = np.array([2, 4, 8, 16, 32, 64])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range', 'layer3_Range'])
    for i in range(NN3_layer1_Range.size):
        for j in range(NN3_layer2_Range.size):            
            for k in range(NN3_layer3_Range.size):            
                num_layer1 = NN3_layer1_Range[i]
                num_layer2 = NN3_layer2_Range[j]    
                num_layer3 = NN3_layer3_Range[k]    
                model = NN3(XtrainNN.shape[1], num_layer1, num_layer2, num_layer3).to('cuda')
                # loss function that measures difference between two binary vectors
                criterion = nn.MSELoss() 
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.995)
                for epoch in range(epochs):
                    gc.disable()
                    ytrainNN_hat = model(XtrainNN)
                    loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                    print(f'NN3: {i,j,k} Epoch: {epoch} Loss: {loss}')
                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch == 0:
                        loss_past = loss
                    elif abs(loss_past-loss)>1e-9:
                            loss_past = loss
                    elif abs(loss_past-loss)<=1e-9:
                            break
                yvalNN_hat = model(XvalNN)
                val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN).cpu().detach().numpy()
                resultsCV = resultsCV.append({'layer1_Range':num_layer1,
                                              'layer2_Range':num_layer2,
                                              'layer3_Range':num_layer3,
                                              'CV_Score':val_loss},
                                             ignore_index=True
                                            )
                if i==0&j==0&k==0:
                    temp_val_loss = val_loss
                    model_temp = model
                elif temp_val_loss >= val_loss:
                        model_temp = model
                        temp_val_loss = val_loss
    resultsCV_NN3 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN3)
    ytest_hat_NN3 = model_temp(XtestNN).cpu().detach().numpy()
    
    
    '''
    # NN4 
    class NN4(nn.Module):
        def __init__(self, n_features, NN4_layer1_Range, NN4_layer2_Range, NN4_layer3_Range, NN4_layer4_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN4_layer1_Range)
            self.fc2 = nn.Linear(NN4_layer1_Range, NN4_layer2_Range)
            self.fc3 = nn.Linear(NN4_layer2_Range, NN4_layer3_Range)
            self.fc4 = nn.Linear(NN4_layer3_Range, NN4_layer4_Range)
            self.fc5 = nn.Linear(NN4_layer4_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return self.fc5(x)        
    NN4_layer1_Range = np.array([2, 4, 8, 16])
    NN4_layer2_Range = np.array([2, 4, 8, 16])
    NN4_layer3_Range = np.array([2, 4, 8, 16])
    NN4_layer4_Range = np.array([2, 4, 8, 16])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range', 'layer3_Range', 'layer4_Range'])
    for i in range(NN4_layer1_Range.size):
        for j in range(NN4_layer2_Range.size):            
            for k in range(NN4_layer3_Range.size):            
                for l in range(NN4_layer4_Range.size):            
                    num_layer1 = NN4_layer1_Range[i]
                    num_layer2 = NN4_layer2_Range[j]    
                    num_layer3 = NN4_layer3_Range[k]    
                    num_layer4 = NN4_layer4_Range[l]    
                    model = NN4(XtrainNN.shape[1], num_layer1, num_layer2, num_layer3, num_layer4).to('cuda')
                    # loss function that measures difference between two binary vectors
                    criterion = nn.MSELoss() 
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    for epoch in range(epochs):
                        gc.disable()
                        ytrainNN_hat = model(XtrainNN)
                        loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                        print(f'NN4: {i,j,k,l} Epoch: {epoch} Loss: {loss}')
                        # backpropagation
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if epoch == 0:
                            loss_past = loss
                        elif abs(loss_past-loss)>1e-9:
                                loss_past = loss
                        elif abs(loss_past-loss)<=1e-9:
                                break
                    yvalNN_hat = model(XvalNN)
                    val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN).cpu().detach().numpy()
                    resultsCV = resultsCV.append({'layer1_Range':num_layer1,
                                                  'layer2_Range':num_layer2,
                                                  'layer3_Range':num_layer3,
                                                  'layer4_Range':num_layer4,
                                                  'CV_Score':val_loss},
                                                 ignore_index=True
                                                    )
                    if i==0&j==0&k==0&l==0:
                        temp_val_loss = val_loss
                        model_temp = model
                    elif temp_val_loss >= val_loss:
                            model_temp = model
                            temp_val_loss = val_loss
    resultsCV_NN4 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN4)
    ytest_hat_NN4 = model_temp(XtestNN).cpu().detach().numpy()
        
    # NN5
    class NN5(nn.Module):
        def __init__(self, n_features, NN5_layer1_Range, NN5_layer2_Range, NN5_layer3_Range, NN5_layer4_Range, NN5_layer5_Range):
            super().__init__()
            self.fc1 = nn.Linear(n_features, NN5_layer1_Range)
            self.fc2 = nn.Linear(NN5_layer1_Range, NN5_layer2_Range)
            self.fc3 = nn.Linear(NN5_layer2_Range, NN5_layer3_Range)
            self.fc4 = nn.Linear(NN5_layer3_Range, NN5_layer4_Range)
            self.fc5 = nn.Linear(NN5_layer4_Range, NN5_layer5_Range)
            self.fc6 = nn.Linear(NN5_layer5_Range, 1)        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            return self.fc6(x)        
    NN5_layer1_Range = np.array([2, 4, 8, 16, 32])
    NN5_layer2_Range = np.array([2, 4, 8, 16])
    NN5_layer3_Range = np.array([2, 4, 8, 16])
    NN5_layer4_Range = np.array([2, 4, 8, 16])
    NN5_layer5_Range = np.array([2, 4, 8, 16])
    resultsCV = pd.DataFrame(columns=['layer1_Range', 'layer2_Range', 'layer3_Range', 'layer4_Range', 'layer5_Range'])
    for i in range(NN5_layer1_Range.size):
        for j in range(NN5_layer2_Range.size):            
            for k in range(NN5_layer3_Range.size):            
                for l in range(NN5_layer4_Range.size):            
                    for m in range(NN5_layer5_Range.size):            
                        num_layer1 = NN5_layer1_Range[i]
                        num_layer2 = NN5_layer2_Range[j]    
                        num_layer3 = NN5_layer3_Range[k]    
                        num_layer4 = NN5_layer4_Range[l]    
                        num_layer5 = NN5_layer5_Range[m]    
                        model = NN5(XtrainNN.shape[1], num_layer1, num_layer2, num_layer3, num_layer4, num_layer5).to('cuda')
                        # loss function that measures difference between two binary vectors
                        criterion = nn.MSELoss() 
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        for epoch in range(epochs):
                            gc.disable()
                            ytrainNN_hat = model(XtrainNN)
                            loss = criterion(torch.squeeze(ytrainNN_hat),ytrainNN)
                            print(f'NN5: {i,j,k,l,m} Epoch: {epoch} Loss: {loss}')
                            # backpropagation
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            if epoch == 0:
                                loss_past = loss
                            elif abs(loss_past-loss)>1e-9:
                                    loss_past = loss
                            elif abs(loss_past-loss)<=1e-9:
                                    break
                        yvalNN_hat = model(XvalNN)
                        val_loss = criterion(torch.squeeze(yvalNN_hat),yvalNN).cpu().detach().numpy()
                        resultsCV = resultsCV.append({'layer1_Range':num_layer1,
                                                      'layer2_Range':num_layer2,
                                                      'layer3_Range':num_layer3,
                                                      'layer4_Range':num_layer4,
                                                      'layer5_Range':num_layer5,
                                                      'CV_Score':val_loss},
                                                     ignore_index=True
                                                        )
                        if i==0&j==0&k==0&l==0&m==0:
                            temp_val_loss = val_loss
                            model_temp = model
                        elif temp_val_loss >= val_loss:
                                model_temp = model
                                temp_val_loss = val_loss
    resultsCV_NN5 = resultsCV.sort_values('CV_Score',ascending=True)
    print(resultsCV_NN5)
    ytest_hat_NN5 = model_temp(XtestNN).cpu().detach().numpy()
    '''
    torch.cuda.empty_cache()
    gc.disable()
    return ytest_hat_NN1, ytest_hat_NN2, ytest_hat_NN3#, ytest_hat_NN4, ytest_hat_NN5
