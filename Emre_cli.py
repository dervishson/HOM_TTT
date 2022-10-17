# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:53:58 2022

@author: TOSHIBA
"""
import pandas as pd
import argparse
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.metrics import r2_score as R
from catboost import CatBoostRegressor as CBR

def data_process(file_name):
    data=pd.read_excel(args.file_name)  #dataset.xlsx
    x=data.sample(frac=1).reset_index(drop=True)
    HOM_mean,TTT_mean=x.mean()[:2]
    HOM_std,TTT_std=x.std()[:2]
    x["HOM"]=(x["HOM"]-HOM_mean)/HOM_std
    x["TTT"]=(x["TTT"]-TTT_mean)/TTT_std

    ###Train-test split and only second index of j changes when output changes
    X_train,X_test=x.iloc[:450,:2],x.iloc[450:,:2]
    y_train,y_test=x.iloc[:450,2:],x.iloc[450:,2:]
    X_train,X_test=X_train.values,X_test.values
    y_train,y_test=y_train.values,y_test.values


    ###Output mean and standard deviations
    y_tr_means,y_ts_means=y_train.mean(axis=0),y_test.mean(axis=0)
    y_tr_stds,y_ts_stds=y_train.std(axis=0),y_test.std(axis=0)

    #####Normalization of output
    y_train,y_test=(y_train-y_tr_means)/y_tr_stds,(y_test-y_ts_means)/y_ts_stds
    
    return X_train,X_test,y_train,y_test,y_tr_means,y_tr_stds,y_ts_means,y_ts_stds

def work_on(which_model):     
    X_train,X_test,y_train,y_test,y_tr_means,y_tr_stds,y_ts_means,y_ts_stds=data_process(args.file_name)
    if args.which_model=="SVR":
        #define model
        model=SVR(C=10,epsilon=0.1,gamma=0.01,kernel="rbf")
        mor=MultiOutputRegressor(model)
        fitted=mor.fit(X_train,y_train)
        preds=fitted.predict(X_test)
        
    elif args.which_model=="KRR":
        #define model
        model=KernelRidge(alpha=0.1,degree=1,kernel="rbf")
        mor=MultiOutputRegressor(model)

        #Fit and predict
        fitted=mor.fit(X_train,y_train)
        preds=fitted.predict(X_test)
    elif args.which_model=="KNNR":
        #define model
        model=KNNR(metric="euclidean",n_neighbors=1,weights="uniform")
        mor=MultiOutputRegressor(model)

        #Fit and predict
        fitted=mor.fit(X_train,y_train)
        preds=fitted.predict(X_test)
    elif args.which_model=="GBR":
        #define model
        model=GBR(learning_rate=0.1,max_depth=5,n_estimators=500)
        mor=MultiOutputRegressor(model)
        fitted=mor.fit(X_train,y_train)
        preds=fitted.predict(X_test)
    elif args.which_model=="ABR":
        #define model
        model=ABR(learning_rate=0.3,n_estimators=200)
        mor=MultiOutputRegressor(model)
        fitted=mor.fit(X_train,y_train)
        preds=fitted.predict(X_test)

    elif args.which_model=="MLPR":
        #define model
        model=MLPR(hidden_layer_sizes=100,max_iter=1500)
        mor=MultiOutputRegressor(model)
        fitted=mor.fit(X_train,y_train)
        preds=fitted.predict(X_test)
    elif args.which_model=="CBR":
        #define model
        model=CBR(learning_rate=0.09,max_depth=6,n_estimators=500)
        mor=MultiOutputRegressor(model)
        fitted=mor.fit(X_train,y_train)
        preds=fitted.predict(X_test)        
    #Computations
    y_test,y_hat=(y_test*y_ts_stds)+y_ts_means,(preds*y_ts_stds)+y_ts_means
    Max_E=np.around(np.max(np.abs(y_test-y_hat),axis=0),5)
    MAE_vals=np.mean(np.abs(y_test-y_hat),axis=0) 
    MAE_vals=np.around(MAE_vals,5)
    MSE_vals=np.mean(np.power((y_test-y_hat),2),axis=0)
    MSE_vals=np.around(MSE_vals,5)
    R_scores=[]
    for i in range(7):
        R_scores.append(R(y_test[:,i],y_hat[:,i]))
        a=np.mean(R_scores)
    R_scores=np.around(R_scores,4).tolist()
    list_1=["Load_level","Throughput","CDR","RLF","SE","HO PP","HO Prob."]
    list_2=MAE_vals.tolist()
    list_3=MSE_vals.tolist()
    list_4=Max_E.tolist()
    list_5=R_scores
    df=pd.DataFrame(list(zip(list_1,list_2,list_3,list_4,list_5)))
    df.columns=["Outputs","MAE","MSE","Max_Errors","R Scores"]
    print(df)
    print("MAE means:",df["MAE"].mean())
    print("MSE means",df["MSE"].mean()) 
    #print("Mean absolute values:",MAE_vals)
    #print("Mean square error values:",MSE_vals)
    

def main(args):
    np.random.seed(13)
    #X_train,X_test,y_train,y_test,y_tr_means,y_tr_stds,y_ts_means,y_ts_stds=data_process(args.file_name)
    work_on(args.which_model)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Result shows')
    parser.add_argument("--file-name", type=str,
                        help="Name of the data file")
    parser.add_argument("--which-model",type=str,
                        help="Which model will be worked")
    
    args = parser.parse_args()

    main(args)
