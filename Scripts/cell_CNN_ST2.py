# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:00 2023

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
import itertools
#generate data ST1
seed = 1235711
fold = os.getcwd()
fold

### generate data

data = Data()

### load and contruct dataset ###
file = open(fold +"/data/ST1/dataset_cell_cnn.dat","rb")
train_data, val_data  = pk.load(file)
file.close()
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16, shuffle=False)

# Input shape is determined by the number of cells sampled from each sample and the number of markers (30 = ST2)
imput_shape = train_data.__getitem__(0)[0].size()
imput_size = 1
for v in imput_shape:
    imput_size*=v



### defining model ###
class Model_CVRobust(torch.nn.Module):
    def __init__(self,imput_size, num_markers):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,num_markers))
        self.cov2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1,1))
        # self.fc1 = torch.nn.Linear(in_features=3, out_features=1)
        self.avPoll=torch.nn.AvgPool2d(kernel_size=(1000,1),stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.do = torch.nn.Dropout1d(p=0.2)
        self.bn = torch.nn.BatchNorm1d(1000)
        self.optimizer=None
    def forward(self, x):
        # print(x.shape)
        x = self.do(self.relu(self.cov1(x)))
        # print(x.shape)
        x = self.bn(x)
        # print(x.shape)
        x = self.do(self.relu(self.cov2(x)))
        # print(x.shape)
        # x = self.bn(x)
        # print(x.shape)
        x = self.avPoll(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        x = self.sigmoid(x)
        return x
    
class Model_CVRobust_Dense(torch.nn.Module):
    def __init__(self,imput_size, num_markers):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,num_markers))
        self.cov2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,1))
        # self.fc1 = torch.nn.Linear(in_features=1, out_features=1)
        self.avPoll=torch.nn.AvgPool2d(kernel_size=(1000,1),stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=10)
        self.fc2 = torch.nn.Linear(in_features=10, out_features=1)
        self.do = torch.nn.Dropout1d(p=0.2)
        self.bn = torch.nn.BatchNorm1d(1000)
        self.optimizer=None
    def forward(self, x):
        x = self.do(self.relu(self.cov1(x)))
        x = self.bn(x)
        # print(x.shape)
        x = self.do(self.relu(self.cov2(x)))
        # print(x.shape)
        x = self.bn(x)
        # print(x.shape)
        x = self.avPoll(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x
    

class Model_Linear(torch.nn.Module):
    def __init__(self,imput_size, num_markers):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.flatten = torch.flatten
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,num_markers))
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.avPoll=torch.nn.AvgPool2d(kernel_size=(1000, 1),stride =1)
        self.do = torch.nn.Dropout1d(p=0.2)
        self.optimizer=None
    def forward(self, x):
        x = self.do(self.relu(self.cov1(x)))
        x = self.avPoll(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x
    
### construct Neural_network ### 

class Neural:
    def __init__(self,train_dataset,val_dataset,model,optimizer,loss_f, device,sumary_lab=False,bach_size=16):
        self.train_loader = train_loader
        self.bach_size = bach_size
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.device = device
        self.sumary_lab = sumary_lab
        self.model.to(device)
        self.writer= SummaryWriter(fold+"/runs/"+self.sumary_lab)
        
    def trainning(self,num_epochs,file_out,test_dataset=None):
        self.model.train()
        for epoch in range(num_epochs):
            if(epoch%20==0):
                self._save(fold+"/data/ST1/models/"+self.sumary_lab +".dat", epoch, num_epochs)
            print(epoch)
            
            ###TRAINING###
            tloss, tscore, n_correct, n_incorrect, si = 0, 0, 0, 0, 0
            for batch_x,batch_y in self.train_loader:
                for i in range(len(batch_x)):
                    si+=1
                    x=batch_x[i]
                    y=batch_y[i]
                    x.to(self.device)
                    y.to(self.device)
                    y_pred = self.model(x) 
                    
                    ### Add loss ###
                    loss = self.loss_f(y_pred, y.view((1)))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tloss+=loss.detach()
                        
                    ### Add normalized accuracy ###
                    if y_pred < 0.5 and y == 0 or y_pred > 0.5 and y == 1:
                        n_correct += 1
                    else:
                        n_incorrect +=1
                    
                    ### Add F score ###

                    
            ### Average validation loss and score for all batches ###
            # print(si)
            tloss = tloss/si
            tscore = (n_correct)/(n_correct+n_incorrect)
            # sfscore = sfscore/si
            # sscore = sscore/si
            print("------------------")
            print("training loss: ", str(tloss), "training accuracy: "+str(tscore)) #, " fscore: ", str(sfscore))
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
        
        
    # def F_score(self, output, label, threshold=0.5, beta=1):
    #     prob = output > threshold
    #     label = label > threshold
    
    #     TP = (prob & label).sum(1).float()
    #     # TN = ((~prob) & (~label)).sum(1).float()
    #     FP = (prob & (~label)).sum(1).float()
    #     FN = ((~prob) & label).sum(1).float()
    
    #     precision = torch.mean(TP / (TP + FP + 1e-12))
    #     recall = torch.mean(TP / (TP + FN + 1e-12))
    #     F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    #     return F2.mean(0)
    
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

# #####################################################################################################

### Define the hyperparameter values to explore ###
batch_x,batch_y=next(iter(train_loader))
batch_size=16
lr = 0.0000001

device = "cpu"
loss_f = torch.nn.BCELoss()


model = Model_CVRobust(imput_size, num_markers=28)
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelCVRobust_bs16_do02lr05",bach_size=batch_size)                  
net.trainning(num_epochs=100, file_out=fold+"/data/ST1/models/scoresModelCVRobust_do02lr05", test_dataset=None)  


# model = Model_Linear(imput_size, num_markers=30)
# optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelLinear_bs16",bach_size=batch_size)                  
# net.trainning(num_epochs=500, test_dataset=None, file_out=fold+"/ST2/cellCnn/scoresmodelLinear")               
       
model = Model_CVRobust_Dense(imput_size, num_markers=28)
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelCV_dense10_do02lr05",bach_size=batch_size)                  
net.trainning(num_epochs=100, file_out=fold+"/data/ST1/models/scoresModelCV_dense10_do02lr05", test_dataset=None)  
    

