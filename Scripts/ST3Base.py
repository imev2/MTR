#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:46:27 2023

@author: listonlab
"""

import pickle as pk
import os
import numpy as np
import pandas as pd
from Data import Data
#generate data ST1
seed = 1235711
fold = os.getcwd()
fold

data = Data()
data.start(folder_in=fold+"/ST3_transformed",folder_out=fold+"/ST3_base", panel="ST3")