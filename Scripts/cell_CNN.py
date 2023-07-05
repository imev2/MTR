# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:00 2023

@author: nina working on RF
"""

from Data import Data,Log_transformer,Standard_tranformer
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
#generate data ST1
seed = 1235711
fold = os.getcwd()
fold

### generate data

data = Data()
# data.load(fold + "/data/ST1/ST1_base_train_val_batch")
# data.split_data_test(fold_train=fold + "/data/ST1/ST1_base_train_batch", fold_test=fold + "/data/ST1/ST1_base_val_batch",perc_train=0.8, seed=seed)
# data.load(fold + "/data/ST1/ST1_base_train_batch")
# data.augmentation(factor=10, seed=seed)


# data.load(fold + "/data/ST1/ST1_base_train_batch")
# data.save(fold + "/data/ST1/ST1_cell_train_batch")
# data.load(fold + "/data/ST1/ST1_cell_train_batch")
# data.sample_all_cells(numcells=1000,seed=seed)

# data.load(fold + "/data/ST1/ST1_base_val_batch")
# data.save(fold + "/data/ST1/ST1_cell_val_batch")
# data.load(fold + "/data/ST1/ST1_cell_val_batch")
# data.sample_all_cells(numcells=1000,seed=seed)

# data.load(fold + "/data/ST1/ST1_base_test_batch")
# data.save(fold + "/data/ST1/ST1_cell_test_batch")
# data.load(fold + "/data/ST1/ST1_cell_test_batch")
# data.sample_all_cells(numcells=1000,seed=seed)

# #save train and valalidation dataset
# dataset = data.get_dataload(fold_train=fold + "/data/ST1/ST1_cell_train_batch", fold_test=fold + "/data/ST1/ST1_cell_val_batch")
# file = open(fold +"/data/ST1/dataset_cell_cnn.dat","wb")
# pk.dump(dataset,file)
# file.close()

### definnnig hyperparamiters
lr= 0.00001
bach_size = 8
device = "cpu"
num_epochs = 20
torch.set_default_dtype(torch.float64)

### load and contruct dataset
file = open(fold +"/data/ST1/dataset_cell_cnn.dat","rb")
train_data, val_data  = pk.load(file)
file.close()
imput_shape = train_data.__getitem__(0)[0].size()
imput_size = 1
for v in imput_shape:
    imput_size*=v

### definning model
class Model(torch.nn.Module):
    def __init__(self,imput_size):
        super().__init__()
        self.flatten = torch.flatten
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1)
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,28))
        self.maxPoll=torch.nn.MaxPool2d(kernel_size=(1000,1),stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.optimizer=None
    def forward(self, x):
        x = self.relu(self.cov1(x))
        x = self.maxPoll(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x

model = Model(imput_size)
### loss function

loss_f = torch.nn.BCELoss()
#### construct Neural_network

class Neural:
    def __init__(self,train_dataset,test_dataset,model,loss_f,device,sumary_lab=False,bach_size=16,lr=0.0001):
        train_loader = DataLoader(dataset=train_dataset, batch_size=bach_size, shuffle=True)
        val_loader = DataLoader(dataset=test_dataset, batch_size=bach_size, shuffle=False)
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.bach_size = bach_size
        self.val_loader = val_loader
        self.model = model
        self.loss_f = loss_f
        self.device = device
        self.sumary_lab = sumary_lab
        self.model.to(device)
        self.train_load = None
        self.test_load = None
    def trainning(self,num_epochs):
        #sumarywrite def
        for epoch in range(num_epochs):
            print(epoch)
            sloss = 0
            si=0
            self.model.train()
            for batch_x,batch_y in self.train_loader:
                for i in range(len(batch_x)):
                    si+=1
                    x=batch_x[i]
                    y=batch_y[i]
                    x.to(self.device)
                    y.to(self.device)
                    y_pred = self.model(x)
                    loss = self.loss_f(y_pred, y.view((1)))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    sloss+=loss.detach()
            sloss = sloss/si
            print("train: "+str(sloss))
            #sumarywrite train
            #validation
            self.model.eval()
            sloss = 0
            si=0
            for batch_x,batch_y in self.val_loader:
                for i in range(len(batch_x)):
                    si+=1
                    x=batch_x[i]
                    y=batch_y[i]
                    x.to(self.device)
                    y.to(self.device)
                    y_pred = self.model(x)
                    loss = self.loss_f(y_pred, y.view((1)))
                    sloss+=loss.detach()
            sloss = sloss/si
            print("val: "+str(sloss))
            #sumarywrite val
       
net =   Neural(train_data,val_data,model,loss_f,device,sumary_lab=False,bach_size=bach_size,lr=lr)                  
net.trainning(100)               
       
    

