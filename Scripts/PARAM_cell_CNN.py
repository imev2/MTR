#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:31:28 2023

@author: listonlab
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
from sklearn.metrics import confusion_matrix
import itertools
#generate data ST1
seed = 1235711
fold = os.getcwd()
fold


### generate data

data = Data()

### load and contruct dataset ###
file = open(fold +"/ST1/dataset_cell_cnn.dat","rb")
train_data, val_data  = pk.load(file)
file.close()

train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=128, shuffle=False)

# Input shape is determined by the number of cells sampled from each sample and the number of markers (30 = ST2)
imput_shape = train_data.__getitem__(0)[0].size()
imput_size = 1
for v in imput_shape:
    imput_size*=v
    
### defining models ###

class Model_CVRobust(torch.nn.Module):
    def __init__(self, input_size, num_markers, num_channels, use_max_pooling, use_three_dense, use_two_dense, use_one_dense):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(1, num_markers))
        self.cov2 = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 1))
        if use_three_dense:
            self.fc3 = torch.nn.Linear(in_features=num_channels, out_features=128)
            self.fc2 = torch.nn.Linear(in_features=128, out_features=32)
            self.fc2 = torch.nn.Linear(in_features=32, out_features=4)
            self.fc = torch.nn.Linear(in_features=4, out_features=1)
        elif use_two_dense:
            self.fc2 = torch.nn.Linear(in_features=num_channels, out_features=64)
            self.fc1 = torch.nn.Linear(in_features=64, out_features=4)
            self.fc = torch.nn.Linear(in_features=4, out_features=1)
        elif use_one_dense:
            self.fc1 = torch.nn.Linear(in_features=num_channels, out_features=8)
            self.fc = torch.nn.Linear(in_features=8, out_features=1)
        else:
            self.fc = torch.nn.Linear(in_features=num_channels, out_features=1)
        if use_max_pooling:
            self.pooling = torch.nn.MaxPool2d(kernel_size=(1000, 1), stride=1)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=(1000, 1), stride=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()     
        self.optimizer = None
        self.three_dense=use_three_dense
        self.two_dense=use_two_dense
        self.one_dense=use_one_dense
    def forward(self, x):
        x = self.relu(self.cov1(x))
        x = self.relu(self.cov2(x))
        x = self.pooling(x)
        x = self.flatten(x)
        if self.three_dense:
            x = self.relu(self.fc3(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        elif self.two_dense: 
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        elif self.one_dense: 
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        else: 
            x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
class Model_CV1(torch.nn.Module):
    def __init__(self, input_size, num_markers, num_channels, use_max_pooling, use_three_dense, use_two_dense, use_one_dense):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(1, num_markers))
        self.cov2 = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 1))
        if use_three_dense:
            self.fc3 = torch.nn.Linear(in_features=num_channels, out_features=128)
            self.fc2 = torch.nn.Linear(in_features=128, out_features=32)
            self.fc2 = torch.nn.Linear(in_features=32, out_features=4)
            self.fc = torch.nn.Linear(in_features=4, out_features=1)
        elif use_two_dense:
            self.fc2 = torch.nn.Linear(in_features=num_channels, out_features=64)
            self.fc1 = torch.nn.Linear(in_features=64, out_features=4)
            self.fc = torch.nn.Linear(in_features=4, out_features=1)
        elif use_one_dense:
            self.fc1 = torch.nn.Linear(in_features=num_channels, out_features=8)
            self.fc = torch.nn.Linear(in_features=8, out_features=1)
        else:
            self.fc = torch.nn.Linear(in_features=num_channels, out_features=1)
        if use_max_pooling:
            self.pooling = torch.nn.MaxPool2d(kernel_size=(1000, 1), stride=1)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=(1000, 1), stride=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()     
        self.optimizer = None
        self.three_dense=use_three_dense
        self.two_dense=use_two_dense
        self.one_dense=use_one_dense
    def forward(self, x):
        x = self.relu(self.cov1(x))
        x = self.relu(self.cov2(x))
        x = self.pooling(x)
        x = self.flatten(x)
        if self.three_dense:
            x = self.relu(self.fc3(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        elif self.two_dense: 
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        elif self.one_dense: 
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        else: 
            x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
class Model_Linear(torch.nn.Module):
    def __init__(self, input_size, use_three_dense, use_two_dense, use_one_dense):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        if use_three_dense:
            self.fc3 = torch.nn.Linear(in_features=input_size, out_features=128)
            self.fc2 = torch.nn.Linear(in_features=128, out_features=32)
            self.fc2 = torch.nn.Linear(in_features=32, out_features=4)
            self.fc = torch.nn.Linear(in_features=4, out_features=1)
        elif use_two_dense:
            self.fc2 = torch.nn.Linear(in_features=input_size, out_features=64)
            self.fc1 = torch.nn.Linear(in_features=64, out_features=4)
            self.fc = torch.nn.Linear(in_features=4, out_features=1)
        elif use_one_dense:
            self.fc1 = torch.nn.Linear(in_features=input_size, out_features=8)
            self.fc = torch.nn.Linear(in_features=8, out_features=1)
        else:
            self.fc = torch.nn.Linear(in_features=input_size, out_features=1)
       
        
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()     
        self.optimizer = None
        self.three_dense=use_three_dense
        self.two_dense=use_two_dense
        self.one_dense=use_one_dense
    def forward(self, x):
        x = self.flatten(x)
        if self.three_dense:
            x = self.relu(self.fc3(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        elif self.two_dense: 
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        elif self.one_dense: 
            x = self.relu(self.fc1(x))
            x = self.fc(x)
        else: 
            x = self.fc(x)
        x = self.sigmoid(x)
        return x

### construct neural network ### 
class Neural:
    def __init__(self,train_dataset,val_dataset,model,optimizer,loss_f, device,
                 sumary_lab=False,bach_size=16):
        self.train_loader = train_loader
        self.bach_size = bach_size
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.device = device
        self.sumary_lab = sumary_lab
        self.model.to(device)
        self.writer= SummaryWriter(fold+"/runs/ST1/"+self.sumary_lab)
        
    def trainning(self, num_epochs, file_out, test_dataset=None):
        self.model.train()
        
        for epoch in range(num_epochs):
            if epoch % 20 == 0:
                self._save(fold + "/ST1/models/" + self.sumary_lab + ".dat", epoch, num_epochs)
            print(epoch)
            
            ### TRAINING ###
            # tp, tn, fp, fn = 0, 0, 0, 0, 0
            tloss, si = 0, 0 
            n_correct, n_incorrect, si = 0, 0, 0
            
            for batch_x, batch_y in self.train_loader:
                for i in range(len(batch_x)):
                    si += 1
                    
                    x = batch_x[i].to(self.device)
                    y = batch_y[i].to(self.device)
                    y_pred = self.model(x)
                    
                    ### Add loss ###
                    loss = self.loss_f(y_pred, y.view((1)))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tloss += loss.detach()
                    
                    ### Add normalized accuracy ###
                    if y_pred < 0.5 and y == 0 or y_pred > 0.5 and y == 1:
                        n_correct += 1
                    else:
                        n_incorrect +=1
            
            ### Average training loss and score for all batches ###
            tloss = tloss/si
            tscore = (n_correct)/(n_correct+n_incorrect)
            
            print("------------------")
            print("Training loss:", tloss)
            print("Accuracy:", tscore)
            # print("Balanced Accuracy:", balanced_accuracy)
            print("------------------")
    
    
            ###VALIDATION###
            self.model.eval()
            
            ### Add accuracy ###
            sloss, sscore, n_correct, n_incorrect, si = 0, 0, 0, 0, 0
            with torch.no_grad():
                for batch_x,batch_y in self.val_loader:
                    for i in range(len(batch_x)):
                        ### Add opt.step? ###
                        si+=1
                        x=batch_x[i]
                        y=batch_y[i]
                        x.to(self.device)
                        y.to(self.device)
                        y_pred = self.model(x)
                        
                        ### Add loss ###
                        loss = self.loss_f(y_pred, y.view((1)))
                        sloss+=loss.detach()
                        
                        ### Add normalized accuracy ###
                        if y_pred < 0.5 and y == 0 or y_pred > 0.5 and y == 1:
                            n_correct += 1
                        else:
                            n_incorrect +=1
                        ### Add F score ##
                        
                ### Average validation loss and score for all batches ###
                # print(si)
                sloss = sloss/si
                sscore = (n_correct)/(n_correct+n_incorrect)
                # print("------------------")
                print("val loss: ", str(sloss), "val accuracy: "+str(sscore)) #, " fscore: ", str(sfscore))
                print("------------------")
                
            self.writer.add_scalars(main_tag=self.sumary_lab, 
                                    tag_scalar_dict={"Loss/train":tloss,
                                                      "Loss/validation":sloss, 
                                                      "Accuracy/train":tscore, 
                                                      "Accuracy/validation":sscore},
                                    global_step=epoch)
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
                tes_loader = DataLoader(dataset=test_dataset, batch_size=self.bach_size, shuffle=False)
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
            
            
            
### Define the hyperparameter values to explore ###
batch_x,batch_y=next(iter(train_loader))
batch_size=128
device = "cpu"
loss_f = torch.nn.BCELoss()

lr = 0.0001
model = Model_CVRobust(input_size=imput_size, num_markers=28, 
                                            num_channels=3, 
                                            use_max_pooling=False, 
                                            use_three_dense=False, 
                                            use_two_dense=False, 
                                            use_one_dense=False)
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,
             device=device,sumary_lab="bs128/CVLinear_1607_17_30",
             bach_size=batch_size)                  
net.trainning(num_epochs=1, file_out=fold+"/Results/ST1/CVRobust/1607_17_30", test_dataset=None) 

file = open("/Users/listonlab/MPHIL/MTR/Scripts/Results/ST1/Linear/Linear_1206_18_10","rb")
model, train_loader, val_loader, sumary_lab, ytrain_true, ytrain_pred, yval_true, yval_pred = pk.load(file)
file.close()

lr = 0.0001
model = Model_Linear(use_three_dense=False, use_two_dense=False, use_one_dense=False)
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,
             device=device,sumary_lab="bs128/CVLinear_1607_17_30",
             bach_size=batch_size)                  
net.trainning(num_epochs=1, file_out=fold+"/Results/ST1/Linear/1607_17_30", test_dataset=None) 

file = open(fold +"/Results/ST1/dataset_cell_cnn.dat","rb")
train_data, val_data  = pk.load(file)
file.close()    