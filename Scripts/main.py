# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023

@author: rafae
"""

from Data import Data
import pickle as pk

seed = 1235711
data = Data()

#data.start_transform("C:/repos/MTR/data/ST1_transformed","C:/repos/MTR/data/ST1_base")
#data.load("C:/repos/MTR/data/ST1_train_val")
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
data.load("C:/repos/MTR/data/umap_stdar")
df = data.umap_space()
data.writefile("C:/repos/MTR/umap_stdar.dat", df, 1)
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
