# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023

@author: rafae
"""

from Data import Data

data = Data()

#3data.start_transform("C:/repos/MTR/data/ST1_transformed")

#data.load("C:/repos/MTR/data/st1.dat")
data.load_c("C:/repos/MTR/data/st1_c.dat")
#data.save_c("C:/repos/MTR/data/st1_c.dat")
#train,val = data.get_train_validation_dataset()
#d=train.__getitem__(0)
#data.sample_all(100)


#data.augmentation_by_batch(1.2)