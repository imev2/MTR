#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:37:00 2023

@author: listonlab
"""

    # def get_poll_cells(self,fold=None, filename=None, balanciate=True,num_cells=1000,seed = 0, save=False):
    #     df = self._sample_data(0,num_cells,seed=seed)
    #     df_y = np.repeat(self.pheno[0], num_cells)
    #     if self.pheno ==1:
    #         print("yes")
    #     tam = len(self.id)
    #     print("generate sample")
    #     for i in range(1,tam):
    #         print(str(i)+ " of "+str(tam))
    #         df = np.concatenate((df, self._sample_data(i,num_cells,seed=seed+i)), axis=0)
    #         df_y = np.concatenate((df_y,np.repeat(self.pheno[i], num_cells)))
    #     if balanciate:
    #         df,df_y = self._oversample(df, df_y,seed+11)
    #     if save:
    #         print("Saving pooled cells.")
    #         df_y = df_y.reshape(-1, 1)
    #         df_combined = np.hstack((df, df_y))
    #         df = pd.DataFrame(df_combined)
    #         df.to_csv(fold+filename)
    #         print("Saved pooled cell file.")
            
        # return(df,df_y)
    
from Data import Data,Log_transformer,Standard_tranformer
import pickle as pk
import os
import numpy as np
import pandas as pd
from Tools import RF, LR, SVM
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
from joblib import Parallel, delayed

#generate data ST1
seed = 1235711
fold = os.getcwd()
data = Data()

data.load(fold + "/data/ST3/ST3_base")

df, df_y = data.get_poll_cells(fold=fold, filename="/data/ST3/pooled/ST3_base_LMM_1000.csv", balanciate = False, save=True, num_cells=1000)
# df, df_y = data.get_poll_cells(fold=fold, filename="/data/pooled/ST1_base_LMM_10_balanced.csv", save=True, num_cells=10)

