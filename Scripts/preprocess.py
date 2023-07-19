from Data import Data,Log_transformer,Standard_tranformer
import pickle as pk
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
from joblib import Parallel, delayed

 
seed = 1235711
fold = os.getcwd()

#generate data ST1

data = Data()

#generate data ST1
data.start(fold+"/data/ST1/ST1_transformed",fold+"/data/ST1/ST1_base", panel="ST1")
#split test group
#data.split_data_test(fold+"/data/ST1/ST1_base_train_val", fold+"/data/ST1/ST1_base_test",perc_train = 0.9,seed=seed+1)
#split train val group
#data.load("C:/repos/MTR/data/ST1/ST1_train_val")
#data.split_data_test("C:/repos/MTR/data/ST1/ST1_train", "C:/repos/MTR/data/ST1/ST1_val",perc_train = 0.8,seed=seed+1)

data = Data()
#generate data ST2
data.start(fold+"/data/ST2/ST2_transformed",fold+"/data/ST2/ST2_base", panel="ST2")
#split test group
#data.split_data_test(fold+"/data/ST2/ST2_base_train_val", fold+"/data/ST2/ST2_base_test",perc_train = 0.9,seed=seed+1)
#split train val group
#data.load("C:/repos/MTR/data/ST2/ST2_train_val")
#data.split_data_test("C:/repos/MTR/data/ST2/ST2_train", "C:/repos/MTR/data/ST2/ST2_val",perc_train = 0.8,seed=seed+1)



data = Data()
#generate data ST3
data.start(fold+"/data/ST3/ST3_transformed",fold+"/data/ST3/ST3_base", panel="ST3")
#split test group
#data.split_data_test(fold+"/data/ST3/ST3_base_train_val", fold+"/data/ST3/ST3_base_test",perc_train = 0.9,seed=seed+1)
#split train val group
#data.load("C:/repos/MTR/data/ST3/ST3_train_val")
#data.split_data_test("C:/repos/MTR/data/ST3/ST3_train", "C:/repos/MTR/data/ST3/ST3_val",perc_train = 0.8,seed=seed+1)