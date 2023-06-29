# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023

@author: rafae
"""

from Data import Data,Log_transformer,Standard_tranformer
import pickle as pk
import os
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
# QUESTION --> do we need to do this for test - think no

data.load(fold + "/data/ST1_base_train_val_batch")
df, df_y = data.get_poll_cells()
# df["df_y"] = df_y
# df.to.csv(fold + "/data/ST1_base_train_val_batch_pool")

# batch - log
data = Data()
data.load(fold + "/data/ST1_base_train_val_log_batch")
df, df_y = data.get_poll_cells()
# df["df_y"] = df_y
# df.to.csv(fold + "/data/ST1_base_train_val_log_batch_pool")

# no batch - no log
data = Data()
data.load(fold + "/data/ST1_base_train_val_scaled")
df, df_y = data.get_poll_cells()
# df["df_y"] = df_y
# df.to.csv(fold + "/data/ST1_base_train_val_scaled_pool")

# # no batch - log
data = Data()
data.load(fold + "/data/ST1_base_train_val_log_scaled")
df, df_y = data.get_poll_cells()
# df["df_y"] = df_y
# df.to.csv(fold + "/data/ST1_base_train_val_log_scaled_pool")
    

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
