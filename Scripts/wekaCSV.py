#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:55:11 2023

@author: listonlab
"""
from Data import Data,Log_transformer,Standard_tranformer,Oversample
import pickle as pk
import os
import numpy as np
import pandas as pd
from Tools import RF
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
#generate data ST3
seed = 1235711
fold = os.getcwd()
data = Data()
data.load(fold + "/data/ST2_base")
df, df_y = data.get_poll_cells(balanciate=False, save=False, num_cells=1000)
print("Saving pooled cells.")
df_log = np.log1p(df)
df_log_pd = pd.DataFrame(df_log)
# df_y = df_y.reshape(-1, 1)
df_y_pd = pd.DataFrame(df_y)
df_y_pd = df_y_pd.replace(0, "control")
df_y_pd = df_y_pd.replace(1, "positive")
df_csv = pd.concat([df_log_pd, df_y_pd], ignore_index=True, axis=1)
df_csv.to_csv(fold+"/data/ST2/pooled_final/ST2_log_base_1000.csv")
print("Saved pooled cell file.")