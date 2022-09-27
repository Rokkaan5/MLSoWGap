# -*- coding: utf-8 -*-
"""
Created on 5/18/2022

@author: Rokka Kobayashi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

# %% Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# %% function to print number of missing values (nan's) in each column of a given df
def print_missing(df):
    for i in df.columns:
        print('Missing',i,':',df[i].isnull().sum())

# %% Produce (omni) dataframe with interpolated input
'''
Purpose of this function is to create a dataframe with the gaps of the
features (input parameters) interpolated (defaulted to linear interpolation). 

Fucntion provides the choice to drop the rows with missing target values 
(desired output (plasma parameters in this case)).
Should be set to True if want to use the dataframe to train the model.
'''
def interpolate_input(df,features,targets,interpolation_method = 'linear',has_time_col=True,
                      drop_target_nan=True,includes_TH=False,th_len=15,include_target_th = False):
    
    #Creating X and y df from Time History data th_df (if applicable)-----------------------------------------------
    if includes_TH:
        X_col = []
        y_col =[]
        
        for f in features:
            for i in range(0,th_len+1):
                X_col.append(f+'_m{}'.format(i))
        if include_target_th:
            for tt in targets:
                for j in range(1,th_len+1):
                    X_col.append(tt+'_m{}'.format(j))
        for tt in targets:
            y_col.append(tt+'_m0')
            
        features = X_col
        targets =y_col
    #---------------------------------------------------------------------------------------------------------------    
    
    
    #separate df into features only & targets only dataframes========================================================
    
    feat_df = df[features].astype('float32')
    targ_df = df[targets].astype('float32')
    if has_time_col:
        if includes_TH:
            time = df['time']
        else:
            time = df['Epoch']      #may want to change this so it doesn't only for dfs with column = 'Epoch'
    
    #interpolate features-only dataframe==============================================================================
    interpolated_feat_df = feat_df.interpolate(interpolation_method)    #if method not specified when function 
                                                                        # is passed, defaults to linear interpolation
    #combine df's together again=====================================================================================
    if has_time_col:
        concat_df = pd.concat([time,interpolated_feat_df,targ_df],axis=1)
    else:
        concat_df = pd.concat([interpolated_feat_df,targ_df],axis=1)
        
    print("New df with interpolated input created")
    # TO drop or NOT to drop... the nan of target parameters---------------------------------------------------------
    if drop_target_nan:
        new_df = concat_df.dropna()
        print("Target Nan's were dropped (no missing values in df;can be used for model training & testing)")
    else:
        new_df = concat_df
        print("Target NaN's were NOT dropped (dataframe still contains missing values)")
    
        
    return new_df

# %% Function to produce the Training and Testing sets
def train_test_dataframes(df,features,targets,split_type='random',test_size=0.2,scale=True,
                          includes_TH=False,th_len=15,include_target_th = False):
    
    #Creating X and y df from Time History data th_df (if applicable)-----------------------------------------------
    if includes_TH:
        X_col = []
        y_col =[]
        
        for f in features:
            for i in range(0,th_len+1):
                X_col.append(f+'_m{}'.format(i))
        if include_target_th:
            for tt in targets:
                for j in range(1,th_len+1):
                    X_col.append(tt+'_m{}'.format(j))
        for tt in targets:
            y_col.append(tt+'_m0')
    else:
        X_col = features
        y_col = targets
    #--------------------------------------------------------------------------------------------------------------- 
    
    X = df[X_col].astype('float32')
    y = df[y_col].astype('float32')
    
    # Training-Test Split=====================================================
    #(Random) Train-Test Split-----------------------------------------------
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= test_size,random_state=123)
    split = 'Random split'
        
    # (Sequential) Train-Test Split------------------------------------------
    if split_type == 'sequential':
    
        train_size = 1-test_size
        
        X_train = X[:int(train_size*len(X))]   #values up to % indicated by train_size
        X_test = X[int(train_size*len(X)):]    #remaining sequential values from X
        y_train = y[:int(train_size*len(y))]
        y_test = y[int(train_size*len(y)):]
        
        split = 'Sequential split'
    # Print split type--------------------------------------------------------
    print('Train-Test split was:', split)
    
    #Scaling==================================================================
    if scale==True:
        scaler=StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print('X_train & X_test have been scaled')
    else:
        print('X_train & X_test are not scaled')
        
    print('Dataframes are complete')  
    
    return X_train,X_test,y_train,y_test

# %% Function to create the prediction df
def pred_df(model,X_test,y_train):
    predict = model.predict(X_test)
    print('prediction for X_test is complete')
    # predict = 2D array, where 1st element of EACH array inside predict is
    # the prediction for 'Vx', the 2nd element is 'Vy', etc. 
    
    #predict(2D array) => dictionary => dataframe-----------------------------
    data = []
    for i in range(predict.shape[1]):
        column =[]
        for j in range(predict.shape[0]):
            column.append(predict[j][i])
        data.append(column)
    
    dictionary = dict((y_train.columns[d],data[d]) for d in range(len(data)))    
    prediction_df = pd.DataFrame(dictionary)
    print('prediction_df is complete')
    
    return prediction_df

# %% The Model
def vpt_ml_model(X_train,y_train,X_test,y_test,
                 ml_model_type=RandomForestRegressor(n_estimators=10,random_state=123),):
        
    #Model Fitting(Training & Testing)========================================  
    print('Selected Model:',ml_model_type)
    model = ml_model_type
    model.fit(X_train,y_train)
    print('model has been fitted')
    
    prediction_df = pred_df(model,X_test,y_train)
    
    return prediction_df,model

# %% Model Performance (prediction vs. actual graph)
def ml_model_performance(target,prediction,target_name,ml_model_name,bins=[10,10],n_levels=10):
    print('Model {} Target Prediction Score:'.format(target_name),r2_score(target[target_name],prediction[target_name]))
    
    #Mean-Squared and Root-Mean-Squared Errors
    mse1 = mse(target[target_name],prediction[target_name],squared=True)
    rmse1 = mse(target[target_name],prediction[target_name],squared=False)
    
    print('Mean-squared error =',mse1)
    print('Root-mean-squared error =', rmse1)
    
    #Graph
    from density_scatter import density_scatter
    
    fig , ax = plt.subplots(1,figsize=(8,5))
    
    density_scatter(prediction[target_name],target[target_name],line_z=1.4e12,ax=ax,sort=False,bins=bins,
                    nlevels=n_levels,plot_cbar=True)
    
    min_diff = target[target_name].min()-min(prediction[target_name])
    max_diff = target[target_name].max()-max(prediction[target_name])
    
    ax.set_xlim([int(target[target_name].min()-min_diff),int(target[target_name].max()+abs(max_diff))])
    ax.set_ylim([int(target[target_name].min()-min_diff),int(target[target_name].max()+abs(max_diff))])
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Actual')
    ax.set_title(ml_model_name + '; Target = {}'.format(target_name),fontsize=15)
    plt.legend()
    plt.show()
    
# %% Automated Model Analysis for all targets
def collective_performance_analysis(target,prediction,ml_model_name):
    print('{} Model Score:'.format(ml_model_name),r2_score(target,prediction))
    
    for select_target in target.columns:
        ml_model_performance(target,prediction,select_target,ml_model_name)
        
# %% Histogram of predicted values
def pred_histo(prediction_df,show_all=True, figsize=(15,8),parameter='Vx'):
    
    if show_all:                            #displays histogram for all parameters in one call
        for parameter in prediction_df.columns:
            plt.figure(figsize=figsize)
            sns.histplot(data=prediction_df,x=parameter)
            plt.show()
    else:                                   #displays single histogram for specified parameter
        plt.figure(figsize=figsize)
        sns.histplot(data=prediction_df,x=parameter)
        plt.show()
    