# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:00 2023

@author: listonlab
"""

from Data_umap import Data,Standard_tranformer,Umap_tranformer,Density_tranformer
import pickle as pk
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
import torchvision.transforms as T

from sklearn import metrics
import torch.optim as optim
torch.set_default_dtype(torch.float64)


#generate data ST1
seed = 12352
fold = os.getcwd()
n_jobs = 15
torch.manual_seed(seed+1)
data = Data(seed=seed)

#agumented
# print("expanda train")
# data.load(fold +"/data/ST2/ST2_train_1")
# data.augmentation(50,numcells=False)

#density fit
# den = Density_tranformer(n_jobs=n_jobs)
# den.fit(file_space= fold +"/data/ST2/space_umap_points.csv",
#           file_split=fold +"/data/ST2/split.dat",num_partition=50)

# # density transtorm
# data.load(fold +"/data/ST2/ST2_test_1")
# den.transform(data)
# data.load(fold +"/data/ST2/ST2_val_1")
# den.transform(data)
# data.load(fold +"/data/ST2/ST2_train_1")
# den.transform(data)

### defining model ###
class Model_Density_1(torch.nn.Module):
    def __init__(self,shape):
        super().__init__()
        self.flatten = torch.flatten
        ker = (int(shape[0]/3),int(shape[0]/3))
        self.cov1 = torch.nn.Conv2d(in_channels=shape[0], out_channels=shape[0], kernel_size=(1,1))
        self.cov2 = torch.nn.Conv2d(in_channels=shape[0], out_channels=3, kernel_size=(10,10))
        #self.cov3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2))
        #self.cov4 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3,3))
        self.max_poll1=torch.nn.MaxPool2d(kernel_size=(4, 4),stride =3)
        self.max_poll2=torch.nn.MaxPool2d(kernel_size=(3,3),stride =3)
        self.av_poll=torch.nn.AvgPool2d(kernel_size=ker,stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.do = torch.nn.Dropout(p=0.3)
        self.do2 = torch.nn.Dropout(p=0.1)
        self.fc1 = torch.nn.Linear(in_features=12, out_features=6)
        self.fc2 = torch.nn.Linear(in_features=6, out_features=1)
        self.optimizer=None
        self.rot = T.RandomRotation(degrees=5)
        self.resize = T.Resize(size=(shape[1],shape[1]),antialias=True)
    def forward(self, x):
        if self.training:
            x = self.rot(x)
            #x = T.CenterCrop(size=np.random.randint(shape[1]-3,shape[1],2))(x)
            x=self.resize(x)
        #x = self.do2(self.cov1(x))
        x = self.cov1(x)
        x = self.do2(x)
        x = self.relu(x)
        x = self.max_poll1(x)
        x = self.cov2(x)
        x = self.relu(x)
        x = self.max_poll2(x)
        # x = self.cov3(x)
        # x = self.relu(x)
        # x = self.max_poll2(x)
        x = self.flatten(x,start_dim=1)
        
        #print(x.shape)

        x = self.do(self.fc1(x))
        x = self.relu(x)
        x = self.relu(self.fc2(x))
        return x

# ### construct Neural_network ### 

class Neural:
    def __init__(self,train_data,val_data,model,optimizer,loss_f, device,sumary_lab=False,bach_size=16, fixed_train_size=False):
        
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
        
        if(sumary_lab!=False):
            self.writer= SummaryWriter(fold+"/runs/"+self.sumary_lab)
        else:
            self.writer=None
        
    def trainning(self,num_epochs,file_out,test_dataset=None):
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
                    loss.retain_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tloss.append(loss.detach().item())
                    t_y = t_y + batch_y.detach().tolist()
                    t_yp = t_yp + torch.flatten(y_pred.detach()).tolist()
                
                
            fpr, tpr, thresholds = metrics.roc_curve(t_y,t_yp, pos_label=1)
            b_acuracy = metrics.auc(fpr, tpr)        
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
                    v_yp = v_yp + torch.flatten(y_pred.detach()).tolist()
                        
                ### Average validation loss and score for all batches ###
                # print(si)
                fpr, tpr, thresholds = metrics.roc_curve(v_y,v_yp, pos_label=1)
                vb_acuracy = metrics.auc(fpr, tpr)
                vloss = np.mean(np.array(vloss))
                    # print("------------------")
                print("val loss: ", str(vloss), "val accuracy: "+str(vb_acuracy)) #, " fscore: ", str(sfscore))
                print("------------------")
                if(epoch%20==0):
                    self._save(fold+"/data/ST2/ST2_models/"+self.sumary_lab +".dat", epoch, num_epochs)
                print(epoch)
                self.li_auc.append(vb_acuracy)
                self.li_model.append(self.model.state_dict())
            if(self.sumary_lab!=False):
                self.writer.add_scalars(main_tag=self.sumary_lab, 
                                        tag_scalar_dict={"Loss/train":tloss,
                                                          "Loss/validation":vloss, 
                                                          "B_Accuracy/train":b_acuracy, 
                                                          "b_Accuracy/validation":vb_acuracy},
                                        global_step=epoch)
        self.save_res(file_out,test_dataset)
        if(self.sumary_lab!=False):
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
                yval_pred +=  list(self.model(bx))
            yval_true=[yval_true[i].item() for i in range(len(yval_true))]
            yval_pred=[yval_pred[i].item() for i in range(len(yval_pred))]
            if(test_dataset!=None):
                tes_loader = DataLoader(dataset=test_dataset, batch_size=self.bach_size, shuffle=False)
                ytest_true=[]
                ytest_pred=[]
                for bx,by in tes_loader:
                    bx.to(self.device)
                    ytest_true += list(by)
                    ytest_pred +=  list(self.model(bx))
                ytest_true=[ytest_true[i].item() for i in range(len(ytest_true))]
                ytest_pred=[ytest_pred[i].item() for i in range(len(ytest_pred))]                
                mod = {"model":self.model.state_dict(),"val_loader":self.val_loader,"test_loader":tes_loader,"sumary_lab":self.sumary_lab,
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
# ### load and contruct dataset ###
train_data, val_data, test_data = data.get_dataload(fold_train=fold +"/data/ST2/ST2_train_1",fold_val=fold +"/data/ST2/ST2_val_1",fold_test=fold +"/data/ST2/ST2_test_1")
# train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)
data.load(fold +"/data/ST2/ST2_train_1")
tam = len(data.pheno)
pos = sum(data.pheno)
pos_weight = torch.as_tensor((tam-pos)/pos) 
shape = train_data.__getitem__(0)[0].size()

# ### Define the hyperparameter values to explore ###
batch_size=12
lr = 0.0000005


device = "cpu"
torch.set_num_threads(16)
loss_f = torch.nn.BCEWithLogitsLoss(reduction="mean",pos_weight=pos_weight)
print("run model")

model = Model_Density_1(shape=shape)
optimizer = optim.Adam(model.parameters(), lr=lr)

net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="test7",bach_size=batch_size)                  
net.trainning(num_epochs=1000, file_out=fold+"/data/Results/ST2/test7.dat", test_dataset=test_data)

# RUN WITH DATAUMAP BELOW
# ### load and contruct dataset ###
train_data, val_data, test_data = data.get_dataload(fold_train=fold +"/data/ST3/ST3_train_1",fold_val=fold +"/data/ST3/ST3_val_1",fold_test=fold +"/data/ST3/ST3_test_1")
# train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)
data.load(fold +"/data/ST3/ST3_train_1")
tam = len(data.pheno)
pos = sum(data.pheno)
pos_weight = torch.as_tensor((tam-pos)/pos) 
shape = train_data.__getitem__(0)[0].size()

# ### Define the hyperparameter values to explore ###
batch_size=12
lr = 0.0000005

device = "cpu"
torch.set_num_threads(16)
loss_f = torch.nn.BCEWithLogitsLoss(reduction="mean",pos_weight=pos_weight)
print("run model")

model = Model_Density_1(shape=shape)
optimizer = optim.Adam(model.parameters(), lr=lr)

net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="test7",bach_size=batch_size)                  
net.trainning(num_epochs=1000, file_out=fold+"/data/Results/ST3/test7.dat", test_dataset=test_data)

# model = Model_Density_2(shape=shape)
# optimizer = optim.Adam(model.parameters(), lr=lr)

# net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="test2",bach_size=batch_size,fixed_train_size=fixed_train_size)                  
# net.trainning(num_epochs=1000, file_out=fold+"/data/Results/ST2/test2.dat", test_dataset=test_data)


# model = Model_Density_4(shape=shape)
# optimizer = optim.Adam(model.parameters(), lr=lr)

# net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="test4",bach_size=batch_size,fixed_train_size=fixed_train_size)                  
# net.trainning(num_epochs=1000, file_out=fold+"/data/Results/ST2/test4.dat", test_dataset=test_data)

# model = Model_Density_5(shape=shape)
# optimizer = optim.Adam(model.parameters(), lr=lr)

# net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="test5",bach_size=batch_size,fixed_train_size=fixed_train_size)                  
# net.trainning(num_epochs=1000, file_out=fold+"/data/Results/ST2/test5.dat", test_dataset=test_data)

