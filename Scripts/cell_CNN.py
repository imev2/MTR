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
from torch.utils.tensorboard import SummaryWriter
import itertools
#generate data ST1
seed = 1235711
fold = os.getcwd()
fold

### generate data

# data = Data()
# data.load(fold + "/data/ST1/ST1_base_train_val_batch")
# data.split_data_test(fold_train=fold + "/data/ST1/ST1_base_train_batch", fold_test=fold + "/data/ST1/ST1_base_val_batch",perc_train=0.8, seed=seed)
# data.load(fold + "/data/ST1/ST1_base_train_batch")
# data.augmentation(factor=50, seed=seed)


# data.load(fold + "/data/ST1/ST1_base_train_batch")
# # data.save(fold + "/data/ST1/ST1_cell_train_batch")
# # data.load(fold + "/data/ST1/ST1_cell_train_batch")
# data.sample_all_cells(numcells=1000,seed=seed)

# data.load(fold + "/data/ST1/ST1_base_val_batch")
# data.save(fold + "/data/ST1/ST1_cell_val_batch")
# data.load(fold + "/data/ST1/ST1_cell_val_batch")
# data.sample_all_cells(numcells=1000,seed=seed)

# data.load(fold + "/data/ST1/ST1_base_test_batch")
# data.save(fold + "/data/ST1/ST1_cell_test_batch")
# data.load(fold + "/data/ST1/ST1_cell_test_batch")
# data.sample_all_cells(numcells=1000,seed=seed)

#save train and valalidation dataset
# dataset = data.get_dataload(fold_train=fold + "/data/ST1/ST1_base_train_batch", fold_test=fold + "/data/ST1/ST1_cell_val_batch")
# file = open(fold +"/data/ST1/dataset_cell_cnn.dat","wb")
# pk.dump(dataset,file)
# file.close()

### load and contruct dataset
file = open(fold +"/data/ST1/dataset_cell_cnn.dat","rb")
train_data, val_data  = pk.load(file)
file.close()
train_loader = DataLoader(dataset=train_data, batch_size=bach_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=bach_size, shuffle=False)
imput_shape = train_data.__getitem__(0)[0].size()
imput_size = 1
for v in imput_shape:
    imput_size*=v

#batch_x,batch_y=next(iter(train_loader))

### definning model
class Model_CVRobust(torch.nn.Module):
    def __init__(self,imput_size):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,28))
        self.cov2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1,1))
        # self.fc1 = torch.nn.Linear(in_features=1, out_features=1)
        self.avPoll=torch.nn.AvgPool2d(kernel_size=(1000,1),stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.do = torch.nn.Dropout1d(p=0.1)
        self.optimizer=None
    def forward(self, x):
        x = self.do(self.relu(self.cov1(x)))
        x = self.do(self.relu(self.cov2(x)))
        x = self.avPoll(x)
        x = self.flatten(x)
        x = self.sigmoid(x)
        return x
    
class Model_CV1(torch.nn.Module):
    def __init__(self,imput_size):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        self.fc1 = torch.nn.Linear(in_features=imput_size, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.do = torch.nn.Dropout1d(p=0.1)
        self.optimizer=None
    def forward(self, x):
        x = self.do(self.relu(self.cov1(x)))
        x = self.maxPoll(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x
    
class Model_CV2(torch.nn.Module):
    def __init__(self,imput_size):
        super().__init__()
        self.flatten = torch.flatten
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1)
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,28))
        self.cov2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,1))
        self.maxPoll=torch.nn.MaxPool2d(kernel_size=(1000,1),stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.optimizer=None
    def forward(self, x):
        x = self.relu(self.cov1(x))
        x = self.relu(self.cov2(x))
        x = self.maxPoll(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x
    

### loss function

loss_f = torch.nn.BCELoss()
#### construct Neural_network

class Neural:
    def __init__(self,train_dataset,train_dataset,model,optimizer,loss_f,device,sumary_lab=False,bach_size=16):
        
        self.train_loader = train_loader
        #self.bach_size = bach_size
        self.val_loader = val_loader
        self.model = model
        self.opt = optimizer
        self.loss_f = loss_f
        self.device = device
        self.sumary_lab = sumary_lab
        self.model.to(device)
        self.writer= SummaryWriter("runs/"+self.sumary_lab)
    def trainning(self,num_epochs,file_out,test_dataset=None):
        #sumarywrite def
        self.mode.train()
        for epoch in range(num_epochs):
            if(epoch%20==0):
                self._save(fold+"/runs/"+self.sumary_lab +".dat", epoch, num_epochs)
            print(epoch)
            tloss = 0
            si=0
            for batch_x,batch_y in self.train_loader:
<<<<<<< HEAD
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
                    tloss+=loss.detach()
            tloss = tloss/si
            print("train: "+str(tloss))
            #sumarywrite train
            #validation
            self.model.eval()
            sloss = 0
            si=0
            with torch.no_grad():
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
            self.writer.add_scalars(main_tag=self.sumary_lab, tag_scalar_dict={"train":tloss,"validation":sloss},global_step=epoch)
        self.save_res(file_out,test_dataset)
        self._writer_close()
        
    def _save(self,file,epoch,num_epochs):
        mod = {"model":self.model.state_dict(),"epoch":epoch,"opt":self.optimizer.state_dict(),"train_loader":self.train_loader,
                "val_loader":self.val_loader,"num_epochs":num_epochs,"loss_f":self.loss_f,"sumary_lab":self.sumary_lab,
                "device":self.device}
        torch.save(mod, file)
    def _writer_close(self):
        self.writer.close()
    def save_res(self,file,test_dataset=None):
        self.model.eval()
        with torch.no_grad():
            ytrain_true=[]
            ytrain_pred=[]
            for bx,by in self.train_loader:
                for i in range(len(by)):
                    bx[i].to(self.device)
                    ytrain_true.append(by[i]) 
                    ytrain_pred.append(self.model(bx[i]))
            yval_true=[]
            yval_pred=[]
            for bx,by in self.val_loader:
                for i in range(len(by)):
                    bx[i].to(self.device)
                    yval_true.append(by[i]) 
                    yval_pred.append(self.model(bx[i]))
            if(test_dataset!=None):
                tes_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
                ytest_true=[]
                ytest_pred=[]
                for bx,by in tes_loader:
                    for i in range(len(by)):
                        bx[i].to(self.device)
                        ytest_true.append(by[i]) 
                        ytest_pred.append(self.model(bx[i]))
                ytrain_true=[]
                ytrain_pred=[]
                
                mod = {"model":self.model.state_dict(),"train_loader":self.train_loader,
                        "val_loader":self.val_loader,"test_loader":tes_loader,"sumary_lab":self.sumary_lab,
                        "ytrain_true":ytrain_true,"ytrain_pred":ytrain_pred, 
                        "yval_true":yval_true,"yval_pred":yval_pred, 
                        "ytest_true":ytest_true,"ytest_pred":ytest_pred}
            
            else:
                mod = {"model":self.model.state_dict(),"train_loader":self.train_loader,
                        "val_loader":self.val_loader,"sumary_lab":self.sumary_lab,
                        "ytrain_true":ytrain_true,"ytrain_pred":ytrain_pred, 
                        "yval_true":yval_true,"yval_pred":yval_pred}
            file = open(file,"wb")
            pk.dump(mod, file)
            file.close()
            #sumarywrite val

        


#####################################################################################################

# Define the hyperparameter values to explore
bach_size=16
lr = 0.0001
device = "cpu"

# hyperparameters = {
#     'epochs': [50, 100, 500],
#     'lr': [0.0001, 0.00001, 0.000001]
# }

# Create a grid of all possible combinations as dictionaries
# grid = [dict(zip(hyperparameters.keys(), combination)) for combination in itertools.product(*hyperparameters.values())]

# res = {"Parameters":[], "Score":[]}
model = Model_CVRobust(imput_size)
net = Neural(train_data,val_data,model,loss_f,device,sumary_lab="modelCVRobust_bs16",bach_size=bach_size,lr=lr)                  
net.trainning(num_epochs=200, file_out=fold+"/data/cellCnn/modelRobust", test_dataset=None)  


model = Model_CV1(imput_size)
net = Neural(train_data,val_data,model,loss_f,device,sumary_lab="modelCV1_bs16",bach_size=bach_size,lr=lr)                  
net.trainning(num_epochs=200, test_dataset=None, file_out=fold+"/data/cellCnn/modelCV1")               
       
    

