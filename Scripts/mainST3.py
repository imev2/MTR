# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023
ST2 Processing
@author: Nina
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

data.load(fold+"/ST3_base")

#split base into train/val and test 
data.split_data_test(fold+"/ST3/ST3_base_train_val", fold+"/ST3/ST3_base_test",perc_train = 0.9,seed=seed+1)

#log transformation - train_val
data.load(fold + "/ST3/ST3_base_train_val")
data.save(fold + "/ST3/ST3_base_train_val_log")
logt = Log_transformer()
logt.fit_transform(fold + "/ST3/ST3_base_train_val_log")

# log transformation - test
data.load(fold + "/ST3/ST3_base_test")
data.save(fold + "/ST3/ST3_base_test_log")
logt = Log_transformer()
logt.fit_transform(fold + "/ST3/ST3_base_test_log")

# STANDARD FITS
#standard fit - batch
scaler = Standard_tranformer(by_batch=True,seed=seed+2,num_cells=1000)
scaler.fit(fold + "/ST3/ST3_base_train_val")
scaler.save(fold + "/ST3/ST3_base_train_val_scaler_batch")
scaler.fit(fold + "/ST3/ST3_base_train_val_log")
scaler.save(fold + "/ST3/ST3_base_train_val_scaler_batch_log")

#standard fit - no batch
scaler = Standard_tranformer(by_batch=False,seed=seed+2,num_cells=1000)
scaler.fit(fold + "/ST3/ST3_base_train_val")
scaler.save(fold + "/ST3/ST3_base_train_val_scaler")
scaler.fit(fold + "/ST3/ST3_base_train_val_log")
scaler.save(fold + "/ST3/ST3_base_train_val_scaler_log")

# STANDARD TRANFORMS

# standard transform - batch - no log
data.load(fold + "/ST3/ST3_base_train_val")
data.save(fold + "/ST3/ST3_base_train_val_batch")
data.load(fold + "/ST3/ST3_base_test")
data.save(fold + "/ST3/ST3_base_test_batch")
scaler = Standard_tranformer()
scaler.load(fold + "/ST3/ST3_base_train_val_scaler_batch")
scaler.transform(fold + "/ST3/ST3_base_train_val_batch")
scaler.transform(fold + "/ST3/ST3_base_test_batch")

# standard transform - batch - log
data.load(fold + "/ST3/ST3_base_train_val_log")
data.save(fold + "/ST3/ST3_base_train_val_log_batch")
data.load(fold + "/ST3/ST3_base_test_log")
data.save(fold + "/ST3/ST3_base_test_log_batch")
scaler = Standard_tranformer()
scaler.load(fold + "/ST3/ST3_base_train_val_scaler_batch_log")
scaler.transform(fold + "/ST3/ST3_base_train_val_log_batch")
scaler.transform(fold + "/ST3/ST3_base_test_log_batch")

# standard transform - no batch - no log
data.load(fold + "/ST3/ST3_base_train_val")
data.save(fold + "/ST3/ST3_base_train_val_scaled")
data.load(fold + "/ST3/ST3_base_test")
data.save(fold + "/ST3/ST3_base_test_scaled")
scaler = Standard_tranformer()
scaler.load(fold + "/ST3/ST3_base_train_val_scaler")
scaler.transform(fold + "/ST3/ST3_base_train_val_scaled")
scaler.transform(fold + "/ST3/ST3_base_test_scaled")

# standard transform - no batch - log
data.load(fold + "/ST3/ST3_base_train_val_log")
data.save(fold + "/ST3/ST3_base_train_val_log_scaled")
data.load(fold + "/ST3/ST3_base_test_log")
data.save(fold + "/ST3/ST3_base_test_log_scaled")
scaler = Standard_tranformer()
scaler.load(fold + "/ST3/ST3_base_train_val_scaler_log")
scaler.transform(fold + "/ST3/ST3_base_train_val_log_scaled")
scaler.transform(fold + "/ST3/ST3_base_test_log_scaled")

# CELL POOL - TODO, SAVE CSV
# batch - no log
# train
data.load(fold + "/ST3/ST3_base_train_val_batch")
df_batch_train, df_y_batch_train = data.get_poll_cells(fold=fold, filename="/ST3/ST3_base_train_val_batch_pool", save=True)
# test 
data.load(fold + "/ST3/ST3_base_test_batch")
df_batch_test, df_y_batch_test = data.get_poll_cells(fold=fold, filename="/ST3/ST3_base_test_batch_pool_unbalanced", balanciate=False, save=True)

# batch - log
# train
data.load(fold + "/ST3/ST3_base_train_val_log_batch")
df, df_y = data.get_poll_cells(fold=fold, filename="/ST3/ST3_base_train_val_log_batch_pool", save=True)
# test
data.load(fold + "/ST3/ST3_base_test_log_batch")
df, df_y = data.get_poll_cells(fold=fold, filename="/ST3/ST3_base_test_log_batch_pool_unbalanced", balanciate=False, save=True)

# no batch - no log
# train
data.load(fold + "/ST3/ST3_base_train_val_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="/ST3/ST3__base_train_val_scaled_pool", save=True)
# # test
data.load(fold + "/ST3/ST3_base_test_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="/ST3/ST3_base_test_scaled_pool_unbalanced", balanciate=False, save=True)

# # no batch - log
# # train
data.load(fold + "/ST3/ST3_base_train_val_log_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="/ST3/ST3_base_train_val_log_scaled_pool", save=True)
# # test
data.load(fold + "/ST3/ST3_base_test_log_scaled")
df, df_y = data.get_poll_cells(fold=fold, filename="/ST3/ST3_base_test_log_scaled_pool_unbalanced", balanciate=False, save=True)
