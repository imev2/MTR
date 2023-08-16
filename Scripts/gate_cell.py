# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:55:52 2023

@author: rafae
"""

from Data import Data,Density_tranformer
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

### generate data
print("load Data")
data = Data()

file = open(fold+"/data/ST3/Model_Cell_2_ST3_0108_1000_bs64_tested.dat","rb")
mod = torch.load(file)
file.close()

class Model_Cell_2(torch.nn.Module):
    def __init__(self,num_markers):
        super().__init__()
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,num_markers))
        self.cov2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,1))
        self.relu = torch.nn.ReLU()
        self.optimizer=None
    def forward(self, x):
        #x1,x2=x.split((30,31),3)
        x = self.cov1(x)
        
        x = self.relu(x)
        x = self.cov2(x)
        x = self.relu(x)
        return x

num_markers = 30
model = Model_Cell_2(num_markers)
d = mod["model"]
del d['fc1.weight']
del d['fc1.bias']
model.load_state_dict(mod["model"])
train = Data()

###########################################################################
###############save csv for tree
train.load(fold +"/data/ST3/UMAP/ST3_2D_train_scale")

# ###############
# i = 1
# train.id[i] ###AD003P
# bx,by = next(iter(train_loader))
# x,_=bx.split((30,31),3)
# y_inport = model(x)
# x = x[0][0]
# y_inport = y_inport[0][0]

# df = torch.cat((x,y_inport),1)
# painel = train.painel[:30]
# painel = painel+["y_impor"]
# df = pd.DataFrame(df.detach().numpy(),columns=painel)
# df.loc[df["y_impor"]>1,"y_impor"] = 1
# df.loc[df["y_impor"]<1,"y_impor"] = 0
# df.to_csv(fold +"/data/ST3/AD003P.csv",index=False)
# #######################################
# i = 2
# train.id[i] ###AD005P
# bx,by = next(iter(train_loader))
# x,_=bx.split((30,31),3)
# y_inport = model(x)
# x = x[0][0]
# y_inport = y_inport[0][0]

# df = torch.cat((x,y_inport),1)
# painel = train.painel[:30]
# painel = painel+["y_impor"]
# df = pd.DataFrame(df.detach().numpy(),columns=painel)
# df.loc[df["y_impor"]>1,"y_impor"] = 1
# df.loc[df["y_impor"]<1,"y_impor"] = 0
# df.to_csv(fold +"/data/ST3/AD005P.csv",index=False)

# ######################################
# i = 0
# train.id[i] ###AD003C
# bx,by = next(iter(train_loader))
# x,dim=bx.split((30,2),3)
# y_inport = model(x)
# x = x[0][0]
# y_inport = y_inport[0][0]
# dim = dim[0][0]

# df = torch.cat((x,dim),1)
# df = torch.cat((df,y_inport),1)
# painel = train.painel
# painel = painel+["y_impor"]
# df = pd.DataFrame(df.detach().numpy(),columns=painel)
# df.loc[df["y_impor"]>1,"y_impor"] = 1
# df.loc[df["y_impor"]<1,"y_impor"] = 0
# df.to_csv(fold +"/data/ST3/pos_AD003C.csv",index=False)
#################################################################
###### apply ummap space
# train.load(fold +"/data/ST3/UMAP/ST3_2D_train_scale")
# train_data, val_data, test_data = data.get_dataload(train.data,fold +"/data/ST3/UMAP/ST3_2D_val_scale",fold +"/data/ST3/UMAP/ST3_2D_test_scale")
# train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
#############################################################################
i = 7
train.id[i] ###AD003C
bx, by= train._get_data(i)
bx = torch.as_tensor(bx)
bx = torch.unsqueeze(torch.unsqueeze(bx,0),0)
x,dim=bx.split((30,2),3)
y_inport = model(x)
x = x[0][0]
y_inport = y_inport[0][0]
dim = dim[0][0]

df = torch.cat((x,dim),1)
df = torch.cat((df,y_inport),1)
painel = train.painel
painel = painel+["y_impor"]
df = pd.DataFrame(df.detach().numpy(),columns=painel)
df.loc[df["y_impor"]>1,"y_impor"] = 1
df.loc[df["y_impor"]<1,"y_impor"] = 0
df.to_csv(fold +"/data/ST3/nina_AD010p.csv",index=False)
############################################################
# 