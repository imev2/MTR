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
#generate data ST2
seed = 1235711
fold = os.getcwd()
data = Data(seed=seed)

# ### TRANFORMED --> BASE DATA ###
# # data.start(fold+"/ST2_10/ST2_transformed","ST2_10/ST2_base", panel="ST2")

# # # ### BASE --> TRAIN/VAL/TEST ###
# # ## load base data ##
# data.load(fold+"/data/ST2/ST2_base")

# ## split test/train ##
# #split test group
# data.split_data_test(fold+"/data/ST2/ST2_base_train_val", fold+"/data/ST2/ST2_base_test",perc_train = 0.9)

# # # NOT PERFORMED ## split train/val ##
# data.load(fold+"/data/ST2/ST2_base_train_val")
# data.split_data_test(fold+"/data/ST2/ST2_base_train", fold+"/data/ST2/ST2_base_val",perc_train = 0.8)


# # ### LOG TRANSFORM ALL DATA ###

# # ## log transformation - train_val ##
# # data.load(fold+"/data/ST2/ST2_base_train")
# # data.save(fold+"/data/ST2/ST2_logscale/ST2_train_log")
# logt = Log_transformer()
# logt.fit_transform(fold+"/data/ST2/ST2_logscale/ST2_train_log")

# # # # ## log transformation - val ##
# # data.load(fold+"/data/ST2/ST2_base_val")
# # data.save(fold+"/data/ST2/ST2_logscale/ST2_val_log")
# logt = Log_transformer()
# logt.fit_transform(fold+"/data/ST2/ST2_logscale/ST2_val_log")


# # # # ## log transformation - test ##
# # data.load(fold+"/data/ST2/ST2_base_test")
# # data.save(fold+"/data/ST2/ST2_logscale/ST2_test_log")
# logt = Log_transformer()
# logt.fit_transform(fold+"/data/ST2/ST2_logscale/ST2_test_log")


# # ### STANDARD FITS ###

# ## standard fit ##
# scaler = Standard_tranformer(num_cells=1000, by_batch=False)
# scaler.fit(fold +"/data/ST2/ST2_base_train")
# scaler.save(fold +"/data/ST2/ST2_scale/ST2_scaler")
# scaler.fit(fold +"/data/ST2/ST2_logscale/ST2_train_log")
# scaler.save(fold +"/data/ST2/ST2_logscale/ST2_scaler_log")

## standard transform - ##
# data.load(fold +"/data/ST2/ST2_base_train")
# data.save(fold +"/data/ST2/ST2_scale/ST2_train_scale")
# data.load(fold +"/data/ST2/ST2_base_test")
# data.save(fold +"/data/ST2/ST2_scale/ST2_test_scale")
# data.load(fold +"/data/ST2/ST2_base_val")
# data.save(fold +"/data/ST2/ST2_scale/ST2_val_scale")
scaler = Standard_tranformer()
scaler.load(fold +"/data/ST2/ST2_scale/ST2_scaler")
scaler.transform(fold +"/data/ST2/ST2_scale/ST2_train_scale")
scaler.transform(fold +"/data/ST2/ST2_scale/ST2_val_scale")
scaler.transform(fold +"/data/ST2/ST2_scale/ST2_test_scale")

# data.load(fold +"/data/ST2/ST2_logscale/ST2_train_log")
# data.save(fold +"/data/ST2/ST2_logscale/ST2_train_logscale")
# data.load(fold +"/data/ST2/ST2_logscale/ST2_val_log")
# data.save(fold +"/data/ST2/ST2_logscale/ST2_val_logscale")
# data.load(fold +"/data/ST2/ST2_logscale/ST2_test_log")
# data.save(fold +"/data/ST2/ST2_logscale/ST2_test_logscale")
scaler = Standard_tranformer()
scaler.load(fold +"/data/ST2/ST2_logscale/ST2_scaler_log")
scaler.transform(fold +"/data/ST2/ST2_logscale/ST2_train_logscale")
scaler.transform(fold +"/data/ST2/ST2_logscale/ST2_val_logscale")
scaler.transform(fold +"/data/ST2/ST2_logscale/ST2_test_logscale")

# ### sample 10000 cells
# data.load(fold +"/data/ST2/ST2_cell/ST2_train_scale")
# data.sample_all_cells(numcells=10000, seed=seed+5)
# data.load(fold +"/data/ST2//ST2_cell/ST2_val_scale")
# data.sample_all_cells(numcells=10000, seed=seed+6)
# data.load(fold +"/data/ST2//ST2_cell/ST2_test_scale")
# data.sample_all_cells(numcells=10000, seed=seed+7)


# =============================================================================
#augment
#data.load("C:/repos/MTR/data/ST2_train")
#data.save("C:/repos/MTR/data/ST2_train_augment")
#data.load("C:/repos/MTR/data/ST2_train_augment")
#data.augmentation(10,seed=seed+3)

#data.load("C:/repos/MTR/data/ST2_train_log")
#data.save("C:/repos/MTR/data/ST2_train_augment_log")
#data.load("C:/repos/MTR/data/ST2_train_augment_log")
#data.augmentation(10,seed=seed+3)

# =============================================================================


## POOLING ###
## SCALED - no log ##
# train
data.load(fold + "/data/ST2/ST2_scale/ST2_train_scale")
df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST2/pooled/ST2_train_scale_pool", save=True, num_cells=1000)
# val
data.load(fold + "/data/ST2/ST2_scale/ST2_val_scale")
df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST2/pooled/ST2_val_scale_pool", save=True, num_cells=1000)
# # test
data.load(fold + "/data/ST2/ST2_scale/ST2_test_scale")
df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST2/pooled/ST2_test_scale_pool_unbalanced", balanciate=False, num_cells=1000,save=True)

## LOG SCALED ##
# train
data.load(fold + "/data/ST2/ST2_logscale/ST2_train_logscale")
df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST2/pooled/ST2_train_logscale_pool", save=True, num_cells=1000)
# val
data.load(fold + "/data/ST2/ST2_logscale/ST2_val_logscale")
df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST2/pooled/ST2_val_logscale_pool", save=True, num_cells=1000)
# # test
data.load(fold + "/data/ST2/ST2_logscale/ST2_test_logscale")
df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST2/pooled/ST2_test_logscale_pool_unbalanced", balanciate=False, num_cells=1000,save=True)
