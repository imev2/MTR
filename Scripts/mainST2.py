# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023
ST2 Processing
@author: Nina
"""

from DataNOUMAP import Data,Log_transformer,Standard_tranformer
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

### TRANFORMED --> BASE DATA ###
# data.start(fold+"/ST2_10/ST2_transformed","ST2_10/ST2_base", panel="ST2")

# ### BASE --> TRAIN/VAL/TEST ###
# ## load base data ##
# data.load(fold+"/data/ST2/ST2_base")

# ## split test/train ##
# #split test group
# data.split_data_test(fold+"/data/ST2/ST2_base_train_val", fold+"/data/ST2/ST2_base_test",perc_train = 0.9,seed=seed+1)

# NOT PERFORMED ## split train/val ##
# data.load(fold+"/data/ST2/ST2_train_val")
# data.split_data_test(fold+"/data/ST2/ST2_train", fold+"/data/ST2/ST2_val",perc_train = 0.8,seed=seed+1)


# ### LOG TRANSFORM ALL DATA ###

# ## log transformation - train_val ##
# data.load(fold+"/data/ST2/ST2_base_train_val")
# data.save(fold+"/data/ST2/ST2_base_train_val_log")
# logt = Log_transformer()
# logt.fit_transform(fold+"/data/ST2/ST2_base_train_val_log")

# ## log transformation - test ##
# data.load(fold+"/data/ST2/ST2_base_test")
# data.save(fold+"/data/ST2/ST2_base_test_log")
# logt = Log_transformer()
# logt.fit_transform(fold+"/data/ST2/ST2_base_test_log")


# ### STANDARD FITS ###
# ## standard fit - batch ##
# scaler = Standard_tranformer(by_batch=True,seed=seed+2,num_cells=10000)
# scaler.fit(fold +"/data/ST2/ST2_base_train_val")
# scaler.save(fold +"/data/ST2/ST2_base_train_val_scaler_batch")
# scaler.fit(fold +"/data/ST2/ST2_base_train_val_log")
# scaler.save(fold +"/data/ST2/ST2_base_train_val_scaler_batch_log")

# ## standard fit - no batch ##
# scaler = Standard_tranformer(by_batch=False,seed=seed+2,num_cells=10000)
# scaler.fit(fold +"/data/ST2/ST2_base_train_val")
# scaler.save(fold +"/data/ST2/ST2_base_train_val_scaler")
# scaler.fit(fold +"/data/ST2/ST2_base_train_val_log")
# scaler.save(fold +"/data/ST2/ST2_base_train_val_scaler_log")

# ### STANDARD SCALING ###

# ## standard transform - batch - no log ##
# data.load(fold +"/data/ST2/ST2_base_train_val")
# data.save(fold +"/data/ST2/ST2_base_train_val_batch")
# data.load(fold +"/data/ST2/ST2_base_test")
# data.save(fold +"/data/ST2/ST2_base_test_batch")
# scaler = Standard_tranformer()
# scaler.load(fold +"/data/ST2/ST2_base_train_val_scaler_batch")
# scaler.transform(fold +"/data/ST2/ST2_base_train_val_batch")
# scaler.transform(fold +"/data/ST2/ST2_base_test_batch")

# ## standard transform - batch - log ##
# data.load(fold +"/data/ST2/ST2_base_train_val_log")
# data.save(fold +"/data/ST2/ST2_base_train_val_log_batch")
# data.load(fold +"/data/ST2/ST2_base_test_log")
# data.save(fold +"/data/ST2/ST2_base_test_log_batch")
# scaler = Standard_tranformer()
# scaler.load(fold +"/data/ST2/ST2_base_train_val_scaler_batch_log")
# scaler.transform(fold +"/data/ST2/ST2_base_train_val_log_batch")
# scaler.transform(fold +"/data/ST2/ST2_base_test_log_batch")

# ## standard transform - no batch - no log ##
# data.load(fold +"/data/ST2/ST2_base_train_val")
# data.save(fold +"/data/ST2/ST2_base_train_val_scaled")
# data.load(fold +"/data/ST2/ST2_base_test")
# data.save(fold +"/data/ST2/ST2_base_test_scaled")
# scaler = Standard_tranformer()
# scaler.load(fold +"/data/ST2/ST2_base_train_val_scaler")
# scaler.transform(fold +"/data/ST2/ST2_base_train_val_scaled")
# scaler.transform(fold +"/data/ST2/ST2_base_test_scaled")

# ## standard transform - no batch - log ##
# data.load(fold +"/data/ST2/ST2_base_train_val_log")
# data.save(fold +"/data/ST2/ST2_base_train_val_log_scaled")
# data.load(fold +"/data/ST2/ST2_base_test_log")
# data.save(fold +"/data/ST2/ST2_base_test_log_scaled")
# scaler = Standard_tranformer()
# scaler.load(fold +"/data/ST2/ST2_base_train_val_scaler_log")
# scaler.transform(fold +"/data/ST2/ST2_base_train_val_log_scaled")
# scaler.transform(fold +"/data/ST2/ST2_base_test_log_scaled")

# =============================================================================
# #augment
# #data.load("C:/repos/MTR/data/ST1_train")
# #data.save("C:/repos/MTR/data/ST1_train_augment")
# #data.load("C:/repos/MTR/data/ST1_train_augment")
# #data.augmentation(10,seed=seed+3)
# 
# #data.load("C:/repos/MTR/data/ST1_train_log")
# #data.save("C:/repos/MTR/data/ST1_train_augment_log")
# #data.load("C:/repos/MTR/data/ST1_train_augment_log")
# #data.augmentation(10,seed=seed+3)
# 
# =============================================================================


### POOLING ###
## batch - no log ##
# train
data.load(fold + "data/ST2/ST2_base_train_val_batch")
df_batch_train, df_y_batch_train = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_train_val_batch_pool", num_cells=10000,save=True)
# test 
data.load(fold + "data/ST2/ST2_base_test_batch")
df_batch_test, df_y_batch_test = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_batch_pool_unbalanced", balanciate=False, num_cells=10000,save=True)

## batch - log ##
# train
data.load(fold + "data/ST2/ST2_base_train_val_log_batch")
df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_train_val_log_batch_pool", save=True)
# test
data.load(fold + "data/ST2/ST2_base_test_log_batch")
df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_log_batch_pool_unbalanced", balanciate=False, num_cells=10000,save=True)

## SCALED - no log ##
# train
data.load(fold + "data/ST2/ST2_base_train_val_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2__base_train_val_scaled_pool", save=True)
# # test
data.load(fold + "data/ST2/ST2_base_test_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_scaled_pool_unbalanced", balanciate=False, num_cells=10000,save=True)

## BATCH SCALED - log ##
# train
data.load(fold + "data/ST2/ST2_base_train_val_log_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_train_val_log_scaled_pool", save=True)
# test
data.load(fold + "data/ST2/ST2_base_test_log_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_log_scaled_pool_unbalanced", balanciate=False,num_cells=10000, save=True)
