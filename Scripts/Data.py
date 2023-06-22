# -*- coding: utf-8 -*-

## Script to perform random data augmentation

# 0. IMPORT STATEMENTS 

import pandas as pd
from sklearn.preprocessing import StandardScaler
#from joblib import Parallel, delayed
#import plotly.express as px
#import umap
import pickle as pk
import os
#from os import listdir
#from os.path import isfile, join
import random
import numpy as np
import struct
import ctypes
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Tools import RF
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score

import shutil
#import subprocess




class Data:
    class AdDataset(Dataset):
        def __init__(self,data):
            
             self.data = data
    
        def __len__(self):
            ## Size of whole data set
            return len(self.data.id)
    
        def __getitem__(self, idx):
            data,y = self.data._get_data(idx)
            data = torch.from_numpy(data)
            #y =  torch.from_numpy(np.array(y)) # Get the class label for the corresponding file WATCH OUT FOR FLOAT --> MAY CAUSE ERRORS BECAUSE DATA NOT IN SAME DTYPE AS CLASS_LABEL
            #dimensions = data.shape  # Get the dimensions of the data  
            return data, y

    def __init__(self,seed = 123571113):
        self.id = None
        self.pheno = None
        self.data = None 
        self.batch = None
        self.painel = None
        self.dim = None
        self.sizes = None
        np.random.seed(seed)
        random.seed(seed)
    

    def save(self,folder):
        data = Data()
        data.data = folder + "/"
        data.id = self.id
        data.pheno = self.pheno
        data.batch = self.batch
        data.painel = self.painel
        data.dim = self.dim
        data.size = self.size
        data._save_meta()
        tam = len(data.id)
        for i in range(tam):
            src = self.data + self.id[i] + ".dat"
            dest = data.data + self.id[i] + ".dat"
            shutil.copyfile(src,dest)
         

            
    def load(self,folder):
        self.data = folder+ "/"
        meta = self._get_meta()
        self.id = meta["id"]
        self.batch = meta["batch"]
        self.dim = meta["dim"]
        self.size = meta["size"]
        self.painel = meta["painel"]
        
        self.pheno = []
        tam = len(self.id)
        for i in range(tam):
            print(str(i) + " of " + str(tam))
            self.pheno.append(self._get_pheno(i))
            
        
        
        
    def start_transform(self,folder_in,folder_out):
        batch_f = os.listdir(folder_in)
        print(batch_f)
        self.id = []
        self.pheno = []
        self.data = folder_out + "/"
        self.batch = []
        self.painel = []
        self.map_batch = {}
        self.dim = 0
        self.sizes = None
        data = []
        index=0
        lowcell = []
        for b in batch_f:
            files = os.listdir(folder_in + "/"+b)
            print(b)
            count = 1
            for f in files:
                print(str(count) + " of " + str(len(files)))
                count+=1
                s = f.replace(" ", "").replace("-", "").replace("_", "")
                x = re.findall(r"AD(\d\d\d)([CP])",s )[0]
                idd = "AD"+x[0]+x[1]
                if idd in self.id:
                    print("the ID "+idd + " is duplicate")
                    continue
                self.id.append(idd)
                self.batch.append(b)
                self.map_batch[idd] = b
                if x[1]=="C":
                    self.pheno.append(0)
                else:
                    self.pheno.append(1)
                file = folder_in + "/" + b + "/" + f
                df = pd.read_csv(file)
                mapa = {'KI67':'Ki67',"CTLA4":"CTLA-4","FOXP3":"FoxP3","CD11c":"FoxP3","ICOS(CD278)":"ICOS","CD80":"ICOS","CD123":"CD28","CD141":"Ki67","CD86":"CD95","IgM":"CD127","IL7Ra(CD127)":"CD127",
                        "CD19":"CD31","HLADR":"HLA-DR","CD94":"CCR2","CCR2(CD192)":"CCR2","BAFF-R":"CXCR5","CD24":"CD25","CD27":"PD-1","CD21":"CXCR3","Comp-BYG584-A":"RORgT","CD10":"CTLA-4","CTLA-4(CD152)":"CTLA-4",
                        "IgD":"CCR7", "CCR7(CD197)":"CCR7","CD57":"CD45RA","CD38":"CCR4","CCR4(CD194)":"CCR4","CD40":"-","CD16":"-","RORgt":"RORgT","CD56":"-","CCR7(CD197":"CCR7","IL1Ra(CD127)":"CD127","IL7RA":"CD127",
                        "PD1":"PD-1"}
                df.rename(columns = mapa, inplace = True)
                my_cols = list(df.columns)    
                for c in df.columns:
                    if c.startswith("-"):
                        my_cols.remove(c)
                    elif c.startswith("Comp-"):
                        my_cols.remove(c)
                my_cols.remove("Time")
                if "livedead" in my_cols:
                    my_cols.remove("livedead")
                if "Livedead" in my_cols:
                    my_cols.remove("Livedead")
                
                df = df[my_cols]
                
                if len(self.painel)<1:
                    self.painel = my_cols
                else:
                    for i in range(len(my_cols)):
                        if my_cols[i] not in  self.painel:
                            print("painel "+ self.painel[i] + " \t col " + my_cols[i])
                
                df = df[self.painel]
                if len(df) < 1000:
                    lowcell.append("Less than 1000 cells\t"+b + "\t"+ f + "\tnum of cells sample: " + str(len(df)))
                self._save_data(index, df.to_numpy())
                index+=1
        print("save meta")
        self._save_meta()
        for f in lowcell:
            print(f)
    
    def _save_meta(self):
        if not os.path.exists(self.data):
            os.makedirs(self.data)
        with open(self.data + "meta.txt","w") as f:
            tam = len(self.id)
            #dim
            f.write(str(self.dim)+"\n")
            #size
            if self.dim != 0:
                s = str(self.size[0])
                for i in range(len(self.size)):
                    s+=" "+str(self.size[i])
                f.write(s+"\n")
            #id
            s = self.id[0]
            for i in range(1,tam):
                s+=" " + self.id[i]
            f.write(s+"\n")
            #painel
            s = self.painel[0]
            for i in range(1,len(self.painel)):
                s+=" " + self.painel[i]
            f.write(s+"\n")
            #batch
            s = self.batch[0]
            for i in range(1,tam):
                s+=" " + self.batch[i]
            f.write(s+"\n")
    
    def _get_meta(self):
        meta = {}
        file = self.data+"meta.txt"
        with open(file,"r") as f:
            dim = int(f.readline())
            size = None
            if dim != 0:
                s_size = f.readline()[:-1]
                s_size = s_size.split(" ")
                size = []
                for s in s_size:
                    size.append(int(s))
                
            idd = f.readline()[:-1]
            painel = f.readline()[:-1]
            batch =  f.readline()[:-1]
            idd = idd.split(" ")
            painel = painel.split(" ")
            batch = batch.split(" ")
            meta["id"] = idd
            meta["painel"] = painel
            meta["batch"] = batch
            meta["dim"] = dim
            meta["size"] = size
            return meta
            
    def _save_data(self,index,data):
        file = self.data + self.id[index] + ".dat"
        with open(file,"wb") as f:
            if self.dim ==0:
                nlin = len(data)
                ncol = len(data[0])
                d = struct.pack("i i i", self.pheno[index],nlin,ncol)
                f.write(d)
                df = data.flatten().astype(np.float32).tobytes()
                f.write(df)

    def _get_data(self, index):
        file = self.data + self.id[index] + ".dat"
        with open(file,"rb") as f:
            if self.dim ==0:
                d= f.read(12)
                y,nlin,ncol =struct.unpack("i i i", d)
                sz = nlin*ncol
                d= f.read(sz*struct.calcsize("f"))
                data = struct.unpack(str(sz)+"f", d)
                data = np.array(data)
                data = data.reshape((nlin, ncol))
                return data,y 
            
    def _get_pheno(self,index):
        file = self.data + self.id[index] + ".dat"
        with open(file,"rb") as f:
            d = f.read(4)
            return struct.unpack("i", d)[0]
        
    

            
    def _sample_data(self,index ,num_lin,seed=0):
        np.random.seed(seed)
        random.seed(seed)
        df,y = self._get_data(index)
        if num_lin > len(df):
            print("sample " + str(index) + " num cells large " + str(len(df)))
            return
        else:
            ilin = np.arange(len(df))
            np.random.shuffle(ilin)
            df = df[ilin[:num_lin],:]
            return df
    
    
                
        
            
    def split_data_test(self,fold_train,fold_test,perc_train = 0.7,seed=123571113):
        np.random.seed(seed)
        random.seed(seed)
        pos = [i for i in range(len(self.id)) if self.pheno[i]==1]
        neg = [i for i in range(len(self.id)) if self.pheno[i]==0]
        np.random.shuffle(pos)
        np.random.shuffle(neg)
        aux_pos = int(perc_train*len(pos))
        aux_neg = int(perc_train*len(neg))
        train = pos[:aux_pos] + neg[:aux_neg]
        train = np.sort(train)
        test = pos[aux_pos:] + neg[aux_neg:]
        test = np.sort(test)
        
        data_train = Data()
        data_test = Data()
        
        data_train.data = fold_train+"/"
        data_test.data = fold_test+"/"
        
        data_train.dim = self.dim
        data_test.dim = self.dim
        
        data_train.size = self.size
        data_test.size = self.size
        
        data_train.id = [self.id[i] for i in train]
        data_test.id = [self.id[i] for i in test]
        
        data_train.pheno = [self.pheno[i] for i in train]
        data_test.pheno = [self.pheno[i] for i in test]

        data_train.batch = [self.batch[i] for i in train]
        data_test.batch = [self.batch[i] for i in test]
        
        data_train.painel= self.painel
        data_test.painel =self.painel
        
        data_train._save_meta()
        data_test._save_meta()
        tam = len(data_train.id)
        for i in range(tam):
            src = self.data + data_train.id[i] + ".dat"
            dest = data_train.data + data_train.id[i] + ".dat"
            shutil.copyfile(src,dest)
        
        tam = len(data_test.id)
        for i in range(tam):
            src = self.data + data_test.id[i] + ".dat"
            dest = data_test.data + data_test.id[i] + ".dat"
            shutil.copyfile(src,dest)

        
    def augmentation(self,factor,seed = 0):
        np.random.seed(seed)
        random.seed(seed)
        #balanciate
        print("balanciate")
        pos = [i for i in range(len(self.id)) if self.pheno[i]==1]
        neg = [i for i in range(len(self.id)) if self.pheno[i]==0]
        num_neg = len(neg)
        num_pos = len(pos)
        i=1
        tot = np.abs(num_neg-num_pos)
        while(num_neg!=num_pos):
            
            print(str(i) + " of " + str(tot))
            i = i+1
            samp1 = None
            samp2 = None
            if num_neg < num_pos:
               samp1 = neg[random.randint(0, len(neg)-1)]
               samp2 = neg[random.randint(0, len(neg)-1)]
               num_neg+=1
            else:
               samp1 = pos[random.randint(0, len(pos)-1)]
               samp2 = pos[random.randint(0, len(pos)-1)]
               num_pos+=1
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            self.id.append(idd)
            df1 = self._get_data(samp1)[0]
            df2 = self._get_data(samp2)[0]
            n_lin = np.min((len(df1),len(df2)))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            self.pheno.append(self.pheno[samp1])
            batch = self.batch[samp1]+ "_"+self.batch[samp2]
            self.batch.append(batch)
            self._save_data(len(self.id)-1, df)
        #aumentation
        print("aumentation")
        fac = int(num_pos*factor)
        while(num_pos< fac):
            print(str(num_pos) + " of " + str(fac))
            #neg
            samp1 = neg[random.randint(0, len(neg)-1)]
            samp2 = neg[random.randint(0, len(neg)-1)]
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            self.id.append(idd)
            df1 = self._get_data(samp1)[0]
            df2 = self._get_data(samp2)[0]
            n_lin = np.min((len(df1),len(df2)))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            self.pheno.append(self.pheno[samp1])
            batch = self.batch[samp1]+"_"+self.batch[samp2]
            self.batch.append(batch)
            self._save_data(len(self.id)-1, df)
            #pos
            samp1 = pos[random.randint(0, len(pos)-1)]
            samp2 = pos[random.randint(0, len(pos)-1)]
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            self.id.append(idd)
            df1 = self._get_data(samp1)[0]
            df2 = self._get_data(samp2)[0]
            n_lin = np.min((len(df1),len(df2)))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            self.pheno.append(self.pheno[samp1])
            batch = self.batch[samp1]+"_"+self.batch[samp2]
            self.batch.append(batch)
            self._save_data(len(self.id)-1, df)
            num_pos+=1
        self._save_meta()
            
    def augmentation_by_batch(self,factor,seed = 0):
        np.random.seed(seed)
        random.seed(seed)
        #balanciate
        print("balanciate")
        pos = [i for i in range(len(self.id)) if self.pheno[i]==1]
        neg = [i for i in range(len(self.id)) if self.pheno[i]==0]
        num_neg = len(neg)
        num_pos = len(pos)
        diff = abs(num_neg-num_pos)
        
        mapN = {}
        mapP = {}
        u_batch = set(self.batch)
        for b in u_batch:
            files_p =[pos[i] for i in range(len(pos)) if self.batch[pos[i]]==b]
            files_n = [neg[i] for i in range(len(neg)) if self.batch[neg[i]]==b]
            for i in files_p:
                mapP[i] = files_p
            for i in files_n:
                mapN[i] = files_n
        i=1
        while(num_neg!=num_pos):
            print(str(i), " out of ", str(diff))
            i+=1
            if num_neg < num_pos:
               samp1 = neg[random.randint(0, len(neg))]
               samp2 = mapN[samp1][random.randint(0, len(mapN[samp1])-1)]
               num_neg+=1
            else:
               samp1 = pos[random.randint(0, len(neg))]
               samp2 = mapP[samp1][random.randint(0, len(mapP[samp1])-1)]
               num_pos+=1
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            self.id.append(idd)
            df1 = self.data[samp1]
            df2 = self.data[samp2]
            n_lin = np.min((len(df1),len(df2)))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            self.pheno.append(self.pheno[samp1])
            batch = self.batch[samp1]
            self.batch.append(batch)
            self._save_data(len(self.id)-1, df)
        #aumentation
        print("aumentation")
        fac = int(num_pos*factor)
        while(num_pos< fac):
            print(str(num_pos), " out of ", str(fac))
            #neg
            samp1 = neg[random.randint(0, len(neg))]
            samp2 = mapN[samp1][random.randint(0, len(mapN[samp1])-1)]
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            self.id.append(idd)
            df1 = self.data[samp1]
            df2 = self.data[samp2]
            n_lin = np.min((len(df1),len(df2)))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            self.pheno.append(self.pheno[samp1])
            self.data.append(df) 
            batch = self.batch[samp1]+self.batch[samp2]
            self.batch.append(batch)
            self._save_data(len(self.id)-1, df)
            
            #pos
            samp1 = pos[random.randint(0, len(neg))]
            samp2 = mapP[samp1][random.randint(0, len(mapP[samp1])-1)]
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            self.id.append(idd)
            df1 = self.data[samp1]
            df2 = self.data[samp2]
            n_lin = np.min((len(df1),len(df2)))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            self.pheno.append(self.pheno[samp1])
            self.data.append(df) 
            batch = self.batch[samp1]
            self.batch.append(batch)
            self._save_data(len(self.id)-1, df)
            num_pos+=1
        self._save_meta()
             

    
    def _feature_inportance(self,num_cells=1000,cv = 5,n_jobs = 15,seed = 0):
        np.random.seed(seed)
        random.seed(seed)
        neg = pd.DataFrame()
        pos = pd.DataFrame()
        print("sample data")
        tam = len(self.id)
        for i in range(tam):
            print(str(i) + " of " + str(tam))
            if self.pheno[i]==0:
                neg = pd.concat([neg, pd.DataFrame(self._sample_data(i, num_cells))], axis=0)
            else:
                pos = pd.concat([pos, pd.DataFrame(self._sample_data(i, num_cells))], axis=0)
        neg["y"] = 0
        pos["y"] = 1
        x = pd.concat([neg, pos], axis=0)
        y = x["y"].copy()
        x.drop('y', axis=1, inplace=True)
        
        
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10,stratify=y,shuffle=True,random_state=seed)
        print("train model")
        rf = RF(random_state=seed ,n_jobs = n_jobs)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        y_ppred = rf.predict_proba(x_test)[:,1]
        mod = {}
        mod["acuracy"] = accuracy_score(y_test,y_pred)
        mod["b_acuracy"] = balanced_accuracy_score(y_test,y_pred)
        mod["ROC"] = roc_auc_score(y_test,y_ppred)
        mod["importance"] = pd.DataFrame({"mark":self.painel,"importance":rf.rf.feature_importances_})
        mod["y_t"] = y_test
        mod["y_t_pred"] = y_pred
        mod["y_t_ppred"] = y_ppred
        
        mod["par"] = rf.par
        rf.fit(x,y)
        mod["x"] = x
        mod["y"] = y
        mod["y_pred"] = rf.predict_proba(x)[:,1]
        mod["painel"] = self.painel
        return mod
            
    def standard_by_batch(self,num_cells=1000):
        u_batch = np.unique(self.batch)
        for b in u_batch:
            print(b)
            idd = []
            for i in range(len(self.id)):
                if b==self.batch[i]:
                    idd.append(i)
            df = self._sample_data(idd[0], num_cells)
            tam = len(idd)
            print("generate sample")
            for i in range(1,tam):
                print(str(i)+ " of "+str(tam))
                df = np.concatenate((df, self._sample_data(idd[i], num_cells)), axis=0)
            scaler = StandardScaler()
            scaler.fit(df)
            print("standard sample")
            for i in range(tam):
                print(str(i)+ " of "+str(tam))
                df = self._get_data(idd[i])[0]
                df = np.array(scaler.transform(df))
                self._save_data(idd[i], df)
                
                 
    def sample_all_cells(self,numcells,seed):
        
        tam = len(self.id)
        print("sample cells")
        for i in range(tam):
            print(str(i)+" of "+str(tam))
            seed +=1
            df =self._sample_data(i,numcells,seed=seed)
            self._save_data(i, df)
    
    def get_dataload(self,fold_train,fold_test,perc_train= 0.7,numcells=1000,factor=10,seed=0):
        self.split_data_test(fold_train,fold_test,perc_train = 0.7,seed=123571113)
        train = Data()
        train.load(fold_train)
        train.augmentation(factor,seed+1)
        train.load(fold_train)
        train.sample_all_cells(numcells,seed=seed+2)
        
        test = Data()
        test.load(fold_test)
        test.sample_all_cells(numcells,seed=seed+2)
        
        
        return self.AdDataset(train),self.AdDataset(test)
        
            
            
        



