# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:00 2023

@author: nina working on RF
"""

from Data import Data,Log_transformer,Standard_tranformer
import pickle as pk
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
#generate data ST1
seed = 1235711
fold = os.getcwd()
fold
data = Data()
data.load(fold + "/data/ST1/ST1_base_train_val_batch")
data.split_data_test(fold_train=fold + "/data/ST1/ST1_base_train_batch", fold_test=fold + "/data/ST1/ST1_base_val_batch",perc_train=0.8, seed=seed)
data.load(fold + "/data/ST1/ST1_base_train_batch")
data.augmentation(factor=10, seed=seed)


data.load(fold + "/data/ST1/ST1_base_train_batch")
data.save(fold + "/data/ST1/ST1_cell_train_batch")
data.load(fold + "/data/ST1/ST1_cell_train_batch")
data.sample_all_cells(numcells=1000,seed=seed)

data.load(fold + "/data/ST1/ST1_base_val_batch")
data.save(fold + "/data/ST1/ST1_cell_val_batch")
data.load(fold + "/data/ST1/ST1_cell_val_batch")
data.sample_all_cells(numcells=1000,seed=seed)

data.load(fold + "/data/ST1/ST1_base_test_batch")
data.save(fold + "/data/ST1/ST1_cell_test_batch")
data.load(fold + "/data/ST1/ST1_cell_test_batch")
data.sample_all_cells(numcells=1000,seed=seed)

#save train and valalidation dataset
dataset = data.get_dataload(fold_train=fold + "/data/ST1/ST1_cell_train_batch", fold_test=fold + "/data/ST1/ST1_cell_val_batch")
file = open(fold +"/data/ST1/dataset_cell_cnn.dat","wb")
pk.dump(dataset,file)
file.close()