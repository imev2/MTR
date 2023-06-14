# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:07:46 2023

@author: rafae
"""

from Data import Data

data = Data()


#data.start_transform("C:/repos/MTR/data/ST1_transformed")
#data.save("C:/repos/MTR/data/st1.dat")

data.load("C:/repos/MTR/data/st1.dat")
#data.sample_all(100)


data.augmentation_by_batch(1.2)