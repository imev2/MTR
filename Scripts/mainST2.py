# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023
ST2 Processing
@author: Nina
"""

from Data import Data,Log_transformer,Standard_tranformer,Oversample
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

# # ### BASE --> TRAIN/VAL/TEST ###
# ## load base data ##
data.load(fold+"/data/ST2/ST2_base")

## split test/train ##
#split test group
data.split_data_test(fold+"/data/ST2/ST2_base_train_val", fold+"/data/ST2/ST2_base_test",perc_train = 0.9,seed=seed+1)

# # NOT PERFORMED ## split train/val ##
data.load(fold+"/data/ST2/ST2_base_train_val")
data.split_data_test(fold+"/data/ST2/ST2_base_train", fold+"/data/ST2/ST2_base_val",perc_train = 0.8,seed=seed+2)


# # ### LOG TRANSFORM ALL DATA ###

# ## log transformation - train_val ##
data.load(fold+"/data/ST2/ST2_base_train")
data.save(fold+"/data/ST2/ST2_train_log")
logt = Log_transformer()
logt.fit_transform(fold+"/data/ST2/ST2_train_log")

# # ## log transformation - val ##
data.load(fold+"/data/ST2/ST2_base_val")
data.save(fold+"/data/ST2/ST2_val_log")
logt = Log_transformer()
logt.fit_transform(fold+"/data/ST2/ST2_val_log")


# # ## log transformation - test ##
data.load(fold+"/data/ST2/ST2_base_test")
data.save(fold+"/data/ST2/ST2_test_log")
logt = Log_transformer()
logt.fit_transform(fold+"/data/ST2/ST2_test_log")


# # ### STANDARD FITS ###

## standard fit ##
scaler = Standard_tranformer(seed=seed+4,num_cells=1000)
scaler.fit(fold +"/data/ST2/ST2_base_train")
scaler.save(fold +"/data/ST2/ST2_scaler")
scaler.fit(fold +"/data/ST2/ST2_train_log")
scaler.save(fold +"/data/ST2/ST2_scaler_log")

## standard transform - ##
data.load(fold +"/data/ST2/ST2_base_train")
data.save(fold +"/data/ST2/ST2_train_scale")
data.load(fold +"/data/ST2/ST2_base_test")
data.save(fold +"/data/ST2/ST2_test_scale")
data.load(fold +"/data/ST2/ST2_base_val")
data.save(fold +"/data/ST2/ST2_val_scale")
scaler = Standard_tranformer()
scaler.load(fold +"/data/ST2/ST2_scaler")
scaler.transform(fold +"/data/ST2/ST2_train_scale")
scaler.transform(fold +"/data/ST2/ST2_val_scale")
scaler.transform(fold +"/data/ST2/ST2_test_scale")

data.load(fold +"/data/ST2/ST2_train_log")
data.save(fold +"/data/ST2/ST2_train_log_scale")
data.load(fold +"/data/ST2/ST2_val_log")
data.save(fold +"/data/ST2/ST2_val_log_scale")
data.load(fold +"/data/ST2/ST2_test_log")
data.save(fold +"/data/ST2/ST2_test_log_scale")
scaler = Standard_tranformer()
scaler.load(fold +"/data/ST2/ST2_scaler_log")
scaler.transform(fold +"/data/ST2/ST2_train_log_scale")
scaler.transform(fold +"/data/ST2/ST2_val_log_scale")
scaler.transform(fold +"/data/ST2/ST2_test_log_scale")

### sample 10000 cells
data.load(fold +"/data/ST2/ST2_train_scale")
data.sample_all_cells(numcells=10000, seed=seed+5)
data.load(fold +"/data/ST2/ST2_val_scale")
data.sample_all_cells(numcells=10000, seed=seed+6)
data.load(fold +"/data/ST2/ST2_test_scale")
data.sample_all_cells(numcells=10000, seed=seed+7)


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
# data.load(fold + "data/ST2/ST2_base_train_val_batch")
# df_batch_train, df_y_batch_train = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_train_val_batch_pool", num_cells=10000,save=True)
# # test 
# data.load(fold + "data/ST2/ST2_base_test_batch")
# df_batch_test, df_y_batch_test = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_batch_pool_unbalanced", balanciate=False, num_cells=10000,save=True)

# ## batch - log ##
# # train
# data.load(fold + "data/ST2/ST2_base_train_val_log_batch")
# df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_train_val_log_batch_pool", save=True)
# # test
# data.load(fold + "data/ST2/ST2_base_test_log_batch")
# df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_log_batch_pool_unbalanced", balanciate=False, num_cells=10000,save=True)

# ## SCALED - no log ##
# # train
# data.load(fold + "data/ST2/ST2_base_train_val_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2__base_train_val_scaled_pool", save=True)
# # # test
# data.load(fold + "data/ST2/ST2_base_test_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_scaled_pool_unbalanced", balanciate=False, num_cells=10000,save=True)

# ## BATCH SCALED - log ##
# # train
# data.load(fold + "data/ST2/ST2_base_train_val_log_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_train_val_log_scaled_pool", save=True)
# # test
# data.load(fold + "data/ST2/ST2_base_test_log_scaled")
# df, df_y = data.get_poll_cells(fold=fold, filename="data/ST2/ST2_base_test_log_scaled_pool_unbalanced", balanciate=False,num_cells=10000, save=True)
