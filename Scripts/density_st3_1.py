# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:00 2023

@author: listonlab
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

### generate data
print("load Data")
data = Data()

## load and contruct dataset ### umap previus apply

#data.load(fold +"/data/ST3/ST3_train_1")
#data.augmentation(20,n_jobs=n_jobs)

# trans = Density_tranformer(n_jobs=n_jobs)
# trans.fit(file_space=fold +"/data/ST3/umap_fit_2D_train_scale.csv", file_split=fold +"/data/ST3/d_split.dat", num_partition=50)
# print("test")
# data.load(fold +"/data/ST3/ST3_2D_test_scale")
# trans.transform(data)
# print("validation")
# data.load(fold +"/data/ST3/ST3_2D_val_scale")
# trans.transform(data)
# print("train")
# data.load(fold +"/data/ST3/ST3_2D_train_scale")
# trans.transform(data)

train_data, val_data, test_data = data.get_dataload(fold +"/data/ST3/ST3_2D_train_scale",fold +"/data/ST3/ST3_2D_val_scale",fold +"/data/ST3/ST3_2D_test_scale")
data.load(fold+"/data/ST3/ST3_2D_train_scale")
tam = len(data.pheno)
pos = sum(data.pheno)
pos_weight = (tam-pos)/pos

# #train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# #val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)

##Input shape is determined by the number of cells sampled from each sample and the number of markers (30 = ST2)
imput_shape = train_data.__getitem__(0)[0].size()
imput_size = 1
for v in imput_shape:
    imput_size*=v

# # # #####################################################################################################
# #model

class Model_Density_1(torch.nn.Module):
    def __init__(self,imput_shape):
        super().__init__()
        self.flatten = torch.flatten
        self.fc1 = torch.nn.Linear(in_features=3, out_features=3)
        self.fc2 = torch.nn.Linear(in_features=3, out_features=1)
        self.cov1 = torch.nn.Conv2d(in_channels=imput_shape[0]-1, out_channels=2, kernel_size=(1,1))
        self.cov2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        self.cov3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,1))
        self.avPoll=torch.nn.MaxPool2d(kernel_size=(50, 50),stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        #self.do1 = torch.nn.Dropout(p=0.5)
        #self.do2 = torch.nn.Dropout2d(p=0.3)
        self.optimizer=None
    def forward(self, x):
        # if self.training:
        #     x = T.RandomRotation(degrees=5)(x)
        #     x=T.CenterCrop(size=(np.random.randint(48, 50),np.random.randint(48, 50)))(x)
        #     x = T.Resize(size=(50,50),antialias=True)(x)
        x1,x2=x.split((30),1)    
        #x1 = self.do1(x1)
        #x1 = self.do2(self.cov1(x1))
        x1=self.cov1(x1)
        x1 = self.relu(x1)
        x = torch.cat((x1,x2),1)
        x = self.cov3(x)
        #x = self.do2(self.cov3(x))
        x = self.relu(x)
        x = self.avPoll(x)
        x = self.flatten(x,start_dim=1)
        #print(x.shape)
        x = self.fc1(x)
        #x = self.do1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# # #####################################################################################################
class Neural:
    def __init__(self,train_data,val_data,model,optimizer,loss_f, device,sumary_lab=False,bach_size=16, fixed_train_size=False,test_data=False):
        
        self.train_loader = DataLoader(dataset=train_data, batch_size=bach_size, shuffle=True)
        self.bach_size = bach_size
        self.val_loader = DataLoader(dataset=val_data, batch_size=bach_size, shuffle=True)
        self.model = model
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.device = device
        self.sumary_lab = sumary_lab
        self.model.to(device)
        self.fixed_train_size=fixed_train_size
        if test_data:
            self.test_loader=DataLoader(dataset=test_data, batch_size=bach_size, shuffle=True)
        else:
            self.test_loader = False
        
        if(sumary_lab!=False):
            self.writer= SummaryWriter(fold+"/runs/"+self.sumary_lab)
        else:
            self.writer=None
        
    def trainning(self,num_epochs,file_out):
        self.li_auc = []
        self.li_model = []
        for epoch in range(num_epochs):
            ###TRAINING###
            tloss = []
            vloss = []
            t_y = []
            v_y = []
            t_yp = []
            v_yp = []
            if self.fixed_train_size==False:
                for batch_x,batch_y in self.train_loader:
                    self.model.train()
                    batch_x.to(self.device)
                    batch_y.to(self.device)
                    y_pred = self.model(batch_x) 
                    ### Add loss ###
                    loss = self.loss_f(torch.flatten(y_pred), batch_y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tloss.append(loss.detach().item())
                    t_y = t_y + batch_y.detach().tolist()
                    t_yp = t_yp + torch.flatten(y_pred.detach()).tolist()
            else:
                tam = 1
                for batch_x,batch_y in self.train_loader:
                    if tam>self.fixed_train_size:
                        break
                    self.model.train()
                    batch_x.to(self.device)
                    batch_y.to(self.device)
                    y_pred = self.model(batch_x) 
                    tam+=self.bach_size
                    #print(tam)
                    ### Add loss ###
                    loss = self.loss_f(torch.flatten(y_pred), batch_y)
                    #loss.retain_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tloss.append(loss.detach().item())
                    t_y = t_y + batch_y.detach().tolist()
                    t_yp = t_yp + torch.flatten(torch.sigmoid(y_pred.detach())).tolist()
                
                
            fpr, tpr, thresholds = metrics.roc_curve(t_y,t_yp, pos_label=1)
            b_acuracy = roc_auc_score(t_y,t_yp)        
            ### Average validation loss and score for all batches ###
            # print(si)
            tloss = np.mean(np.array(tloss))
            # sfscore = sfscore/si
            # sscore = sscore/si
            print("------------------")
            print("training loss: ", str(tloss), "training accuracy: "+str(b_acuracy)) #, " fscore: ", str(sfscore))
            # print("------------------")

            ###VALIDATION###
            self.model.eval()
            ### Add accuracy ###  
            with torch.no_grad():
                for batch_x,batch_y in self.val_loader:
                    self.model.train()
                    batch_x.to(self.device)
                    batch_y.to(self.device)
                    y_pred = self.model(batch_x) 
                    ### Add loss ###
                    loss = self.loss_f(torch.flatten(y_pred), batch_y)
                    vloss.append(loss.detach().item())
                    v_y = v_y + batch_y.detach().tolist()
                    v_yp = v_yp + torch.flatten(torch.sigmoid(y_pred.detach())).tolist()
                        
                ### Average validation loss and score for all batches ###
                # print(si)
                fpr, tpr, thresholds = metrics.roc_curve(v_y,v_yp, pos_label=1)
                vb_acuracy = roc_auc_score(v_y,v_yp)
                vloss = np.mean(np.array(vloss))
                    # print("------------------")
                print("val loss: ", str(vloss), "val accuracy: "+str(vb_acuracy)) #, " fscore: ", str(sfscore))
                print("------------------")
                if(epoch%20==0):
                    self._save(fold+"/data/ST3/ST3_models/"+self.sumary_lab +".dat", epoch, num_epochs)
                print(epoch)
                self.li_auc.append(vb_acuracy)
                self.li_model.append(self.model.state_dict().copy())
            if(self.sumary_lab!=False):
                self.writer.add_scalars(main_tag=self.sumary_lab, 
                                        tag_scalar_dict={"Loss/train":tloss,
                                                          "Loss/validation":vloss, 
                                                          "B_Accuracy/train":b_acuracy, 
                                                          "b_Accuracy/validation":vb_acuracy},
                                        global_step=epoch)
        self.save_res(file_out)
        if(self.sumary_lab!=False):
            self._writer_close()
        
    def _save(self,file,epoch,num_epochs):
        mod = {"model":self.model.state_dict(),"epoch":epoch,"opt":self.optimizer.state_dict(),"train_loader":self.train_loader,
                "val_loader":self.val_loader,"num_epochs":num_epochs,"loss_f":self.loss_f,"sumary_lab":self.sumary_lab,
                "device":self.device}
        torch.save(mod, file)
        
    def _writer_close(self):
        self.writer.close()
        
    def save_res(self,file):
        self.model.eval()
        with torch.no_grad():
            # ytrain_true=[]
            # ytrain_pred=[]
            # tam=0
            # for bx,by in self.train_loader:
            #     bx.to(self.device)
            #     ytrain_true.append(by) 
            #     ytrain_pred.append(self.model(bx))
            yval_true=[]
            yval_pred=[]
            for bx,by in self.val_loader:
                bx.to(self.device)
                yval_true += list(by)
                yval_pred +=  list(torch.sigmoid(self.model(bx)))
            yval_true=[yval_true[i].item() for i in range(len(yval_true))]
            yval_pred=[yval_pred[i].item() for i in range(len(yval_pred))]
            if  self.test_loader:
                ytest_true=[]
                ytest_pred=[]
                for bx,by in self.test_loader:
                    bx.to(self.device)
                    ytest_true += list(by)
                    ytest_pred +=  list(self.model(bx))
                ytest_true=[ytest_true[i].item() for i in range(len(ytest_true))]
                ytest_pred=[ytest_pred[i].item() for i in range(len(ytest_pred))]                
                mod = {"model":self.model.state_dict(),"val_loader":self.val_loader,"test_loader":self.test_loader,"sumary_lab":self.sumary_lab,
                        "yval_true":yval_true,"yval_pred":yval_pred, 
                        "ytest_true":ytest_true,"ytest_pred":ytest_pred,"list_models":self.li_model,"list_auc":self.li_auc}
            
            else:
                mod = {"model":self.model.state_dict(),
                        "val_loader":self.val_loader,"sumary_lab":self.sumary_lab,
                        "yval_true":yval_true,"yval_pred":yval_pred}
            file = open(file,"wb")
            torch.save(mod, file)
            file.close()

# # #####################################################################################################
# RUN WITH DATAUMAP BELOW
batch_size=70
lr = 0.005
device = "cpu"
torch.set_num_threads(16)
loss_f = torch.nn.BCEWithLogitsLoss(reduction="mean",pos_weight=torch.as_tensor(pos_weight))
model = Model_Density_1(imput_shape)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

net = Neural(train_data=train_data,val_data=val_data,test_data=test_data,model=model,loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="end_2",bach_size=batch_size)                  
net.trainning(num_epochs=50000, file_out=fold+"/data/Results/ST3/end_2.dat")

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
with open("C:/repos/MTR/Scripts/data/Results/ST3/end_2.dat", 'rb') as f:
    a = torch.load(f)
    f.close()
print(a.keys())
y_true = a["ytest_true"]
y_prob = a["ytest_pred"]
y_prob_sig = torch.sigmoid(torch.as_tensor(y_prob))
auroc = roc_auc_score(y_true=y_true, y_score=y_prob_sig, average='weighted')
