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

# Step 1: Read the meta.txt file to get the marker names

meta_file = os.path.join(fold, "data/ST3/ST3_base/meta.txt")

with open(meta_file, "r") as f:
    marker_names = f.read().splitlines()[2]
    
marker_names = marker_names.split()

marker_names.append("outcome")

data = Data()
data.load(fold + "/data/ST3/ST3_base")
df, df_y = data.get_poll_cells(balanciate=False, save=False, num_cells=1000)
print("Saving pooled cells.")
df_log = np.log1p(df)
df_log_pd = pd.DataFrame(df_log)
# df_y = df_y.reshape(-1, 1)
df_y_pd = pd.DataFrame(df_y)
df_y_pd = df_y_pd.replace(0, "control")
df_y_pd = df_y_pd.replace(1, "positive")
df_csv = pd.concat([df_log_pd, df_y_pd], ignore_index=True,  axis=1)
df_csv.columns = marker_names

# Step 2: Assign the marker names as column names to df_log_pd DataFrame
df_csv.to_csv(fold+"/data/ST3/pooled/ST3_log_base_1000_markers.csv", index=False)
print("Saved pooled cell file.")


