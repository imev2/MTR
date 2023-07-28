# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:00 2023

@author: listonlab
"""

from Data import Data,Standard_tranformer
import pickle as pk
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics
import torch.optim as optim
torch.set_default_dtype(torch.float64)


#generate data ST1
seed = 12357
fold = os.getcwd()
fold
torch.manual_seed(seed+1)
data = Data(seed=seed)
# data.load(fold+"/data/ST2/ST2_base")

# # ## split test/train ##
# #split test group
# data.split_data_test(fold+"/data/ST2/ST2_base_train_val", fold+"/data/ST2/ST2_base_test",perc_train = 0.9)
# data.load(fold+"/data/ST2/ST2_base_train_val")
# data.split_data_test(fold+"/data/ST2/ST2_base_train", fold+"/data/ST2/ST2_base_val",perc_train = 0.8)
# # # # # ### STANDARD FITS ###

# # # ## standard fit ##
# scaler = Standard_tranformer(num_cells=1000)
# scaler.fit(fold +"/data/ST2/ST2_base_train")
# scaler.save(fold +"/data/ST2/ST2_scaler.dat")


# # # ## standard transform - ##
# scaler.transform(fold +"/data/ST2/ST2_base_train")
# scaler.transform(fold +"/data/ST2/ST2_base_val")
# scaler.transform(fold +"/data/ST2/ST2_base_test")

# # ##agumented
# data.load(fold +"/data/ST2/ST2_base_train")
# data.save(fold +"/data/ST2/ST2_train_1")
# # data.save(fold +"/data/ST2/ST2_train_2")
# data.load(fold +"/data/ST2/ST2_base_test")
# data.save(fold +"/data/ST2/ST2_test_1")
# # data.save(fold +"/data/ST2/ST2_test_2")
# data.load(fold +"/data/ST2/ST2_base_val")
# data.save(fold +"/data/ST2/ST2_val_1")


# data.load(fold +"/data/ST2/ST2_train_1")
# data.augmentation(100,numcells=10000)

### sample cells
# data.load(fold +"/data/ST2/ST2_val_1")
# data.sample_all_cells(numcells=10000)
# data.load(fold +"/data/ST2/ST2_test_1")
# data.sample_all_cells(numcells=10000)


### load and contruct dataset ###
train_data, val_data, test_data = data.get_dataload(fold_train=fold +"/data/ST2/ST2_train_1",fold_val=fold +"/data/ST2/ST2_val_1",fold_test=fold +"/data/ST2/ST2_test_1")

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)

#Input shape is determined by the number of cells sampled from each sample and the number of markers (30 = ST2)
imput_shape = train_data.__getitem__(0)[0].size()
imput_size = 1
for v in imput_shape:
    imput_size*=v



### defining model ###
class Model_Cell_1(torch.nn.Module):
    def __init__(self,imput_size, num_markers):
        super().__init__()
        self.flatten = torch.flatten
        self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,num_markers))
        self.avPoll=torch.nn.AvgPool2d(kernel_size=(10000, 1),stride =1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.do1 = torch.nn.Dropout(p=0.2)
        self.do2 = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1)
        self.optimizer=None
    def forward(self, x):
        x = self.do1(x)
        x = self.do2(self.cov1(x))
        x = self.relu(x)
        x = self.avPoll(x)
        x = self.flatten(x,start_dim=1)
        #print(x.shape)
        x = self.fc1(x)
        return x
    
    
### construct Neural_network ### 

class Neural:
    def __init__(self,train_dataset,val_dataset,model,optimizer,loss_f, device,sumary_lab=False,bach_size=16, fixed_train_size=False):
        
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

# # #####################################################################################################f

### Define the hyperparameter values to explore ###
batch_size=4
lr = 0.00001
fixed_train_size = 50

device = "cpu"
torch.set_num_threads(8)
loss_f = torch.nn.BCEWithLogitsLoss(reduction="mean")
print("run model")

model = Model_Cell_1(imput_size, num_markers=30)
optimizer = optim.Adam(model.parameters(), lr=lr)

net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="Model_Cell_1_ST2",bach_size=batch_size,fixed_train_size=fixed_train_size)                  
net.trainning(num_epochs=200, file_out=fold+"/data/Results/ST2/Model_Cell_1_ST2.dat", test_dataset=test_data)

# # model = Model_CVRobust(imput_size, num_markers=30)
# # optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# # net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelCVRobust_bs16_do02lr05",bach_size=batch_size)                  
# # net.trainning(num_epochs=100, file_out=fold+"/ST2/scoresModelCVRobust_do02lr05", test_dataset=None)  


# # # model = Model_Linear(imput_size, num_markers=30)
# # # optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# # # # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# # # net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelLinear_bs16",bach_size=batch_size)                  
# # # net.trainning(num_epochs=500, test_dataset=None, file_out=fold+"/ST2/cellCnn/scoresmodelLinear")               
       
# # model = Model_CVRobust_Dense(imput_size, num_markers=30)
# # optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# # net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelCV_dense10_do02lr05",bach_size=batch_size)                  
# # net.trainning(num_epochs=100, file_out=fold+"/ST2/scoresModelCV_dense10_do02lr05", test_dataset=None)  

# # model = Model_CV2(imput_size, num_markers=28)
# # optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# # net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelCV2_1307_9_45s",bach_size=batch_size)                  
# # net.trainning(num_epochs=1, file_out=fold+"/Results/ST1/CV2/1307_8_38", test_dataset=None)  


  

# # model = Model_Linear(imput_size, num_markers=28)
# # optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# # net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelLinear_1307_8_5channels",bach_size=batch_size)                  
# # net.trainning(num_epochs=1, test_dataset=None, file_out=fold+"/Results/ST1/Linear/Linear_1206_18_10")               
       
# # model = Model_CVRobust_Dense(imput_size, num_markers=28)
# # optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# # net = Neural(train_data,val_data,model=model, loss_f=loss_f,optimizer=optimizer,device=device,sumary_lab="modelCV_dense_1307_8_5channels",bach_size=batch_size)                  
# # net.trainning(num_epochs=1, file_out=fold+"/Results/ST1/Dense/Dense_1307_8_38", test_dataset=None)  






# # # ### definning model
# # # class Model_CVRobust(torch.nn.Module):
# # #     def __init__(self,imput_size):
# # #         super().__init__()
# # #         torch.set_default_dtype(torch.float64)
# # #         self.flatten = torch.flatten
# # #         self.cov1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,28))
# # #         self.cov2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1,1))
# # #         # self.fc1 = torch.nn.Linear(in_features=1, out_features=1)
# # #         self.avPoll=torch.nn.AvgPool2d(kernel_size=(1000,1),stride =1)
# # #         self.sigmoid = torch.nn.Sigmoid()
# # #         self.relu = torch.nn.ReLU()
# # #         self.do = torch.nn.Dropout1d(p=0.1)
# # #         self.optimizer=None
# # #     def forward(self, x):
# # #         x = self.do(self.relu(self.cov1(x)))
# # #         x = self.do(self.relu(self.cov2(x)))
# # #         x = self.avPoll(x)
# # #         x = self.flatten(x)
# # #         x = self.sigmoid(x)
# # #         return x
    
# # # class Model_CV1(torch.nn.Module):
# # #     def __init__(self,imput_size):
# # #         super().__init__()
# # #         torch.set_default_dtype(torch.float64)
# # #         self.flatten = torch.flatten
# # #         self.fc1 = torch.nn.Linear(in_features=imput_size, out_features=1)
# # #         self.sigmoid = torch.nn.Sigmoid()
# # #         self.relu = torch.nn.ReLU()
# # #         self.do = torch.nn.Dropout1d(p=0.1)
# # #         self.optimizer=None
# # #     def forward(self, x):
# # #         x = self.do(self.relu(self.cov1(x)))
# # #         x = self.maxPoll(x)
# # #         x = self.flatten(x)
# # #         x = self.sigmoid(self.fc1(x))
# # #         return x
    

    

