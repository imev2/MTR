# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:55:52 2023

@author: rafae
"""

from Data_umap import Data,Density_tranformer
import pickle as pk
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools
import plotly.express as px
import plotly.io as pio
import torchvision.transforms as T
pio.renderers.default='browser'
from sklearn import metrics
#generate data ST1
seed = 1235711
fold = os.getcwd()
n_jobs=15
torch.manual_seed(seed+1)
torch.set_default_dtype(torch.float64)

### load result


mod = torch.load(fold+"/data/Results/ST3/end_2.dat")

### generate data
print("load Data")
data = Data()

train_data, val_data, test_data = data.get_dataload(fold +"/data/ST3/ST3_2D_train_scale",fold +"/data/ST3/ST3_2D_val_scale",fold +"/data/ST3/ST3_2D_test_scale")
data.load(fold+"/data/ST3/ST3_2D_train_scale")
tam = len(data.pheno)
pos = sum(data.pheno)
pos_weight = (tam-pos)/pos

data.load(fold +"/data/ST3/ST3_2D_test_scale")

class Model_Density_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.flatten
        #self.fc1 = torch.nn.Linear(in_features=3, out_features=3)
        #self.fc2 = torch.nn.Linear(in_features=3, out_features=1)
        self.cov1 = torch.nn.Conv2d(in_channels=30, out_channels=2, kernel_size=(1,1))
        self.cov2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        self.cov3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,1))
        self.avPoll=torch.nn.MaxPool2d(kernel_size=(50, 50),stride =1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer=None
    def forward(self, x):
        x1,x2=x.split((30),1)    
        x1=self.cov1(x1)
        x1 = self.relu(x1)
        x = torch.cat((x1,x2),1)
        x = self.cov3(x)
        x = self.relu(x)
        #x = self.avPoll(x)
        #x = self.flatten(x,start_dim=1)
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = self.sigmoid(x)
        return x

class Model_Density_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.flatten
        self.fc1 = torch.nn.Linear(in_features=3, out_features=3)
        self.fc2 = torch.nn.Linear(in_features=3, out_features=1)
        #self.cov1 = torch.nn.Conv2d(in_channels=30, out_channels=2, kernel_size=(1,1))
        #self.cov2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        #self.cov3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,1))
        #self.avPoll=torch.nn.MaxPool2d(kernel_size=(50, 50),stride =1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer=None
    def forward(self, x):
        #x1,x2=x.split((30),1)    
        #x1=self.cov1(x1)
        #x1 = self.relu(x1)
        #x = torch.cat((x1,x2),1)
        #x = self.cov3(x)
        #x = self.relu(x)
        #x = self.avPoll(x)
        #x = self.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
model = Model_Density_1()
#model.load_state_dict(mod["list_models"][2263])

##############################################################
##### get test performancy
# sample = []
# y_hat = []
# y_pred = []
# y_true = []

# i=0
# for  bx, by in test_data:
#     print(i)
#     bx = torch.unsqueeze(bx,0)
#     y_ = model(bx).detach().numpy()[0][0]
#     y_hat.append(y_)
#     y_pred.append(0 if y_ < 0.5 else 1)
#     y_true.append(int(by.detach().numpy()))
#     sample.append(data.id[i])
#     i+=1

# df = pd.DataFrame({"sample":sample,"y_true":y_true,"y_hat":y_hat,"y_pred":y_pred})
############################################################
# performacy matrix per sample
#### model
data.load(fold +"/data/ST3/ST3_2D_train_scale")
density = []
mat = []
sample = []


i=0
j=0
dicionary = mod["list_models"][2263].copy()
del dicionary["fc1.weight"]
del dicionary["fc1.bias"]
del dicionary["fc2.weight"]
del dicionary["fc2.bias"]

dic2 =  mod["list_models"][2263]
del dic2["cov1.weight"]
del dic2["cov1.bias"]
del dic2["cov2.weight"]
del dic2["cov2.bias"]
del dic2["cov3.weight"]
del dic2["cov3.bias"]
model2 = Model_Density_2()

model.load_state_dict(dicionary)
model2.load_state_dict(dic2)
################## sample
i=0
for  bx, by in train_data:
    print(i)
    density.append(bx[0])
    bx = torch.unsqueeze(bx,0)
    x = model(bx)#.detach().numpy()
    aux =np.ones([50,50])
    for l in range(50):
        for c in range(50):
            aux2 = model2(x[:,:,l,c]).detach().numpy()
            aux[l,c] = aux2[0][0]
    mat.append(aux)
    
    sample.append(data.id[i])
    i+=1
#################################################