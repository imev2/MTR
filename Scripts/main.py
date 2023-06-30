# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023

@author: rafae
"""

from Data import Data,Log_transformer,Standard_tranformer
import pickle as pk
import os
import numpy as np
import pandas as pd
from Tools import RF
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
#generate data ST1
seed = 1235711
fold = os.getcwd()
data = Data()

# data.load(fold + "/data/ST1_base")
# print(data._get_data(0))
# data.save("Scripts/data/ST1_base_log")
# df,y = data.get_poll_cells(seed=seed)
# data.start("C:/repos/MTR/data/ST1_transformed","C:/repos/MTR/data/ST1_base")
# #split test group
# data.split_data_test(fold+"/data/ST1_base_train_val", fold+"/data/ST1_base_test",perc_train = 0.9,seed=seed+1)
# #split train val group
#data.load("C:/repos/MTR/data/ST1/ST1_train_val")
#data.split_data_test("C:/repos/MTR/data/ST1/ST1_train", "C:/repos/MTR/data/ST1/ST1_val",perc_train = 0.8,seed=seed+1)

# log transformation
# data.load(fold + "/data/ST1_base_train")
# logt = Log_transformer()
# logt.fit_transform("Scripts/data/ST1_base_log")
# data.load("Scripts/data/ST1_base_log")
# print(data._get_data(0))
# data.load("C:/repos/MTR/data/ST1/ST1_test")
# data.save("C:/repos/MTR/data/ST1/ST1_test_log")
# logt.fit_transform("C:/repos/MTR/data/ST1/ST1_test_log")

#log transformation - train_val
# data.load(fold + "/data/ST1_base_train_val")
# data.save(fold + "/data/ST1_base_train_val_log")
# logt = Log_transformer()
# logt.fit_transform(fold + "/data/ST1_base_train_val_log")

# log transformation - test
# data.load(fold + "/data/ST1_base_test")
# data.save(fold + "/data/ST1_base_test_log")
# logt = Log_transformer()
# logt.fit_transform(fold + "/data/ST1_base_test_log")

# STANDARD FITS
#standard fit - batch
# scaler = Standard_tranformer(by_batch=True,seed=seed+2,num_cells=1000)
# scaler.fit(fold + "/data/ST1_base_train_val")
# scaler.save(fold + "/data/ST1_base_train_val_scaler_batch")
# scaler.fit(fold + "/data/ST1_base_train_val_log")
# scaler.save(fold + "/data/ST1_base_train_val_scaler_batch_log")

#standard fit - no batch
# scaler = Standard_tranformer(by_batch=False,seed=seed+2,num_cells=1000)
# scaler.fit(fold + "/data/ST1_base_train_val")
# scaler.save(fold + "/data/ST1_base_train_val_scaler")
# scaler.fit(fold + "/data/ST1_base_train_val_log")
# scaler.save(fold + "/data/ST1_base_train_val_scaler_log")

# STANDARD TRANFORMS

# standard transform - batch - no log
# data.load(fold + "/data/ST1_base_train_val")
# data.save(fold + "/data/ST1_base_train_val_batch")
# data.load(fold + "/data/ST1_base_test")
# data.save(fold + "/data/ST1_base_test_batch")
# scaler = Standard_tranformer()
# scaler.load(fold + "/data/ST1_base_train_val_scaler_batch")
# scaler.transform(fold + "/data/ST1_base_train_val_batch")
# scaler.transform(fold + "/data/ST1_base_test_batch")

# standard transform - batch - log
# data.load(fold + "/data/ST1_base_train_val_log")
# data.save(fold + "/data/ST1_base_train_val_log_batch")
# data.load(fold + "/data/ST1_base_test_log")
# data.save(fold + "/data/ST1_base_test_log_batch")
# scaler = Standard_tranformer()
# scaler.load(fold + "/data/ST1_base_train_val_scaler_batch_log")
# scaler.transform(fold + "/data/ST1_base_train_val_log_batch")
# scaler.transform(fold + "/data/ST1_base_test_log_batch")

# standard transform - no batch - no log
# data.load(fold + "/data/ST1_base_train_val")
# data.save(fold + "/data/ST1_base_train_val_scaled")
# data.load(fold + "/data/ST1_base_test")
# data.save(fold + "/data/ST1_base_test_scaled")
# scaler = Standard_tranformer()
# scaler.load(fold + "/data/ST1_base_train_val_scaler")
# scaler.transform(fold + "/data/ST1_base_train_val_scaled")
# scaler.transform(fold + "/data/ST1_base_test_scaled")

# standard transform - no batch - log
# data.load(fold + "/data/ST1_base_train_val_log")
# data.save(fold + "/data/ST1_base_train_val_log_scaled")
# data.load(fold + "/data/ST1_base_test_log")
# data.save(fold + "/data/ST1_base_test_log_scaled")
# scaler = Standard_tranformer()
# scaler.load(fold + "/data/ST1_base_train_val_scaler_log")
# scaler.transform(fold + "/data/ST1_base_train_val_log_scaled")
# scaler.transform(fold + "/data/ST1_base_test_log_scaled")

#augment
#data.load("C:/repos/MTR/data/ST1_train")
#data.save("C:/repos/MTR/data/ST1_train_augment")
#data.load("C:/repos/MTR/data/ST1_train_augment")
#data.augmentation(10,seed=seed+3)

#data.load("C:/repos/MTR/data/ST1_train_log")
#data.save("C:/repos/MTR/data/ST1_train_augment_log")
#data.load("C:/repos/MTR/data/ST1_train_augment_log")
#data.augmentation(10,seed=seed+3)

# CELL POOL - TODO, SAVE CSV

# batch - no log
# train
# data.load(fold + "/data/ST1_base_train_val_batch")
# df_batch_train, df_y_batch_train = data.get_poll_cells(fold=fold, filename="/data/ST1_base_train_val_batch_pool", save=True)
# test 
# data.load(fold + "/data/ST1_base_test_batch")
# df_batch_test, df_y_batch_test = data.get_poll_cells(fold=fold, filename="/data/ST1_base_test_batch_pool", save=True)

# batch - log
# train
# data.load(fold + "/data/ST1_base_train_val_log_batch")
# df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST1_base_train_val_log_batch_pool", save=True)
# test
# data.load(fold + "/data/ST1_base_test_log_batch")
# df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST1_base_test_log_batch_pool", save=True)

# no batch - no log
# train
# data.load(fold + "/data/ST1_base_train_val_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="/data/pooled/ST1_base_train_val_scaled_pool", save=True)
# # test
# data.load(fold + "/data/ST1_base_test_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="/data/pooled/ST1_base_test_scaled_pool", save=True)

# # no batch - log
# # train
# data.load(fold + "/data/ST1_base_train_val_log_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="/data/pooled/ST1_base_train_val_log_scaled_pool", save=True)
# # test
# data.load(fold + "/data/ST1_base_test_log_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="/data/pooled/ST1_base_test_log_scaled_pool", save=True)


# FEATURE SELECTION
 
# batch - no log

def FeatureSelectionRF(fold, train_path, test_path, out_path, verbose=2):
    # Read in sampled cell file, extract df and df_y. 
    # Perform RF feature selection and save output. 
    print('Reading training file')
    file = pd.read_csv(fold+train_path) 
    df_train = file.iloc[:, :-2].to_numpy()
    df_y_train = file.iloc[:, -1].to_numpy().reshape(-1,)
    
    print('Fitting using training data')
    rf = RF(random_state=0 ,n_jobs = 15)
    rf.fit(df_train, df_y_train, verbose=verbose)
    
    print('Reading test file')
    file = pd.read_csv(fold+test_path) 
    df_test = file.iloc[:, :-2].to_numpy()
    df_y_test = file.iloc[:, -1].to_numpy().reshape(-1,)
    
    print('Predicting')
    y_pred = rf.predict(df_test)
    y_ppred = rf.predict_proba(df_test)[:,1]
    
    print('Saving results')
    mod = {}
    mod["accuracy"] = accuracy_score(df_y_test,y_pred)
    mod["b_accuracy"] = balanced_accuracy_score(df_y_test,y_pred)
    mod["ROC"] = roc_auc_score(df_y_test,y_ppred)
    # TODO: Edit with painel!
    mod["importance"] = pd.DataFrame({"importance":rf.rf.feature_importances_}) 
    mod["y_t"] = df_y_test
    mod["y_t_pred"] = y_pred
    mod["y_t_ppred"] = y_ppred
    mod["par"] = rf.par
    # TODO: Fix so that compatible! Just testing here
    # rf.fit(x,y)
    # mod["x"] = x
    # mod["y"] = y
    # mod["y_pred"] = rf.predict_proba(x)[:,1]
    # TODO: Edit with painel!
    # mod["painel"] = self.painel
    file = open(fold+out_path,"wb")
    pk.dump(mod, file)
    file.close()
    
    return mod

MOD = FeatureSelectionRF(fold=fold, 
                         train_path="/data/pooled/ST1_base_train_val_batch_pool",
                         test_path="/data/pooled/ST1_base_train_val_batch_pool", 
                         out_path="/data/MOD/batch.dat")

# RAFAEL'S CODE BELOW!

#data.save("C:/repos/MTR/data/ST1__train_standard")
#data.load("C:/repos/MTR/data/ST1_train_val")
#data.save("C:/repos/MTR/data/umap_no_stdar")
#data.save("C:/repos/MTR/data/umap_stdar")
#data.load("C:/repos/MTR/data/umap_stdar")
#data.standard_by_batch(1000)
#data.load("C:/repos/MTR/data/umap_stdar")
#data.augmentation(1.2,seed+1)
#data.load("C:/repos/MTR/data/umap_no_stdar")
#data.augmentation(1.2,seed+1)
# data.load("C:/repos/MTR/data/umap_no_stdar")
# df = data.umap_space()
# data.writefile("C:/repos/MTR/umap_no_stdar.dat", df, 1)
#data.load("C:/repos/MTR/data/umap_stdar")
#df = data.umap_space()
#data.writefile("C:/repos/MTR/umap_stdar.dat", df, 1)
#data.load("C:/repos/MTR/data/ST1__train_standard")
#data.save("C:/repos/MTR/data/ST1__train_argument")
#data.load("C:/repos/MTR/data/ST1__train_argument")
#
#data.load("C:/repos/MTR/data/ST1__train_standard")
#df = data.umap_space()
#data.save("C:/repos/MTR/data/test1")
#data.load("C:/repos/MTR/data/test1/train")
#train,test = data.get_dataload("C:/repos/MTR/data/test1/train", "C:/repos/MTR/data/test1/test",perc_train= 0.9,numcells=1000,seed=0)
#df1 = data._get_data(1000)[0]
#mod = {"mixsample_standard":df,"df1":df1}

#file = open("C:/repos/MTR/data/umap.dat","wb")
#pk.dump(df,file)
#file.close()

# file = open("C:/repos/MTR/data/test_test1.dat","wb")
# pk.dump(test,file)
# file.close()
#split
#data.split_data_test("C:/repos/MTR/data/ST1_train_val", "C:/repos/MTR/data/ST1_test",perc_train=0.8,seed = seed)

#feature selection
#data.load("C:/repos/MTR/data/ST1_train_val")
#mod = data._feature_inportance(num_cells=1000,cv = 1,n_jobs = 15,seed = seed+1)
#file = open("C:/repos/MTR/data/randomforest_no_standart.dat","wb")
#pk.dump(mod, file)
#file.close()

#data.save("C:/repos/MTR/data/teste")

#data.load("C:/repos/MTR/data/teste")
#data.standard_by_batch(1000)
#data.load("C:/repos/MTR/data/teste")
#mod = data._feature_inportance(num_cells=1000,cv = 1,n_jobs = 15,seed = seed+2)
#file = open("C:/repos/MTR/data/randomforest_with_standart.dat","wb")
#pk.dump(mod, file)
#file.close()