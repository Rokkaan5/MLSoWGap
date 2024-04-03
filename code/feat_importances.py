# -*- coding: utf-8 -*-
"""
Created on Wed May 18 22:27:01 2022

@author: rokka
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% Function to organize the feature importances of a timehistory model
def th_feat_imports(rf_model,features,th_len=15):
    X_col=[]
    feat_imports = []
    
    Q=0

    for f in range(len(features)):        #for each feature
        X_col.append([])                        #add a new list to list X_col corresponding to the feature
        feat_imports.append([])                 # " " " " to list feat_imports " " " "
        
        for i in range(th_len+1):                       #for each time history minute
            X_col[f].append(features[f]+'_m{}'.format(i))               #fill the new list with each TH min of feature
            feat_imports[f].append(rf_model.feature_importances_[Q+i])  #" " " " with each feature importance " "
            
        Q+=th_len+1
    
    return X_col,feat_imports             #return both lists of lists



# %% function for using sns line plot to visualize feature importance of time-history model
def th_fi_line_graph(rf_model,features,th_len=15,figsize=(15,8)):
    
    X_col,feat_imports = th_feat_imports(rf_model,features,th_len)
        
    #create dataframe with the columns as each feature and the index as each minute of the time history-----------------
    d= dict((features[f],feat_imports[f]) for f in range(len(features))) 
    df = pd.DataFrame(d)
    
    #Plot the line graph------------------------------------------------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.lineplot(data=df)
    plt.xlabel('time history minute')
    plt.ylabel('Feature Importance')
    plt.title(str(rf_model)+' with TH',fontsize=15)
    
    return df

# %% function for using bar plot to visualize feat importance of each parameter
def fi_bar_plot(rf_model,features,figsize=(15,8),includes_th=False,th_len=15):
    
    #for model WITH time history========================================================================================
    if includes_th:
        #organize list of feature importances to match with order of features (columns of X)----------------------------
        X_col,feat_imports = th_feat_imports(rf_model,features,th_len)
        
        #create a dataframe with columns of each time history minute and each row labels the feature parameter----------
        dfs=[]
        for f in range(len(features)):
            di= dict(('m_{}'.format(j),feat_imports[f][j]) for j in range(th_len+1))
            dfs.append(pd.DataFrame(di,index=[features[f]]))
        
        final_df = pd.concat(dfs)
        #Plot the stacked bargraph using the dataframe------------------------------------------------------------------
        final_df.plot(kind='bar',stacked=True,figsize=figsize)
        
        plt.title(str(rf_model)+' with TH',fontsize=15)
        plt.xlabel('Features')
        plt.ylabel('Model Feature Importance')
        plt.show()
        return final_df    #returns the df used for the plot
    
    #for model without time history=====================================================================================
    else:
        #dictionary of the feature importances (not used for the plot but in case the information is useful)------------
        fi_di = dict((features[f],rf_model.feature_importances_[f]) for f in range(len(features)))
        
        #barplot of feature importances for model w/o TH----------------------------------------------------------------
        plt.figure(figsize=figsize)
        plt.bar(features,rf_model.feature_importances_,width=0.5)
        
        plt.title(str(rf_model)+'; (no TH)',fontsize=15)
        plt.xlabel('Features')
        plt.ylabel('Model Feature Importance')
        plt.show()
        
        return fi_di      #returns dictionary of feature importances associated with features