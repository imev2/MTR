# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023

@author: rafae
"""

from Data import Data

data = Data()

#data.start_transform("C:/repos/MTR/data/ST1_transformed","C:/repos/MTR/data/ST1_base")

data.load("C:/repos/MTR/data/ST1_test")
#data.split_data_test("C:/repos/MTR/data/ST1_train", "C:/repos/MTR/data/ST1_test")
#data.save("C:/repos/MTR/data/teste")
#df = data._sample_data(0, 200)
print("10")
#data.load_c("C:/repos/MTR/data/st1_c.dat")
#data.save_c("C:/repos/MTR/data/st1_c.dat")
#train,val = data.get_train_validation_dataset()
#d=train.__getitem__(0)
#data.sample_all(100)


#data.augmentation_by_batch(1.2)