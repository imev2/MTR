# -*- coding: utf-8 -*-

## Script to perform random data augmentation

# 0. IMPORT STATEMENTS 

import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
#import plotly.express as px
import umap
import pickle as pk
import os
#from os import listdir
#from os.path import isfile, join
import random
import numpy as np
import struct
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
# from Tools import RF
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score

import shutil
import subprocess
fold = os.getcwd()
class Data:
    class AdDataset(Dataset):
        def __init__(self,data,shape,transformer=None):
             self.data = data
             self.shape = shape
             self.transformer = transformer
             
        def __len__(self):
            ## Size of whole data set
            return len(self.data.id)
    
        def __getitem__(self, idx):
            data,y = self.data._get_data(idx,self.shape)
            data = torch.as_tensor(data,dtype=torch.float64).view(self.shape)
            y=torch.as_tensor(y,dtype=torch.float64)
            if self.transformer:
                data =self.transformer(data)
            #y =  torch.from_numpy(np.array(y)) # Get the class label for the corresponding file WATCH OUT FOR FLOAT --> MAY CAUSE ERRORS BECAUSE DATA NOT IN SAME DTYPE AS CLASS_LABEL
            #dimensions = data.shape  # Get the dimensions of the data  
            return data, y
        def set_transformer(self,transformer):
            self.transformer = transformer

    def __init__(self,seed = 0):
        self.id = None
        self.pheno = None
        self.data = None 
        self.batch = None
        self.painel = None
        self.dim = None
        self.size = None
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed

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
            #print(str(i) + " of " + str(tam))
            self.pheno.append(self._get_pheno(i))
            
        
        
        
    def start(self,folder_in,folder_out, panel):
        batch_f = os.listdir(folder_in)
        print(batch_f)
        self.id = []
        self.pheno = []
        self.data = folder_out + "/"
        self.batch = []
        self.painel = []
        self.map_batch = {}
        self.dim = 0
        self.size = None
        #data = []
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
                #print(re.findall(r"AD(\d\d\d)([CP])",s ))
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
   
                
                # LABELS
                # incorrect:correct
                if (panel=="ST1"):
                    mapa = {'KI67':'Ki67',"CTLA4":"CTLA-4","FOXP3":"FoxP3","CD11c":"FoxP3","ICOS(CD278)":"ICOS","CD80":"ICOS","CD123":"CD28","CD141":"Ki67","CD86":"CD95","IgM":"CD127","IL7Ra(CD127)":"CD127",
                            "CD19":"CD31","HLADR":"HLA-DR","CD94":"CCR2","CCR2(CD192)":"CCR2","BAFF-R":"CXCR5","CD24":"CD25","CD27":"PD-1","CD21":"CXCR3","Comp-BYG584-A":"RORgT","CD10":"CTLA-4","CTLA-4(CD152)":"CTLA-4",
                            "IgD":"CCR7", "CCR7(CD197)":"CCR7","CD57":"CD45RA","CD38":"CCR4","CCR4(CD194)":"CCR4","CD40":"-","CD16":"-","RORgt":"RORgT","CD56":"-","CCR7(CD197":"CCR7","IL1Ra(CD127)":"CD127","IL7RA":"CD127",
                            "PD1":"PD-1"}
                elif (panel=="ST2"):
                    mapa = {}
                    
                elif (panel=="ST3"):
                    mapa = {"PD1":"PD-1", "CD27":"PD-1",
                            "CD57":"TNFa", 
                            "CD21":"CD40L", 
                            "BAFFR":"IL-10", "IL10":"IL-10",
                            "CD24":"CD25",
                            "CD94":"4-1BB", "4-IBB":"4-1BB", 
                            "IL7RA":"IL-4", "IgM":"IL-4", 
                            "CD86":"IFNg", 
                            "Ki67":"Tbet", "CD141":"Tbet", 
                            "CD56":"CD45RA", 
                            "CD80":"IL-2", "IL2":"IL-2", 
                            "FOX-P3":"FoxP3", "FOXP3":"FoxP3","CD11c":"FoxP3",
                            "CD40":"IL-17a", "IL-17":"IL-17a", 
                            "HLADR":"HLA-DR", 
                            "CD21":"CD40L",
                            "Gata3":"GATA-3","CD16":"GATA-3", "GATA3":"GATA-3",
                            "CD27":"PD-1",
                            "CD38":"IL-6",
                            "CTLA-4":"CTLA4", "CD10":"CTLA4", "CTLA-4":"CTLA4",
                            "RORgt":"RORgT","Comp-BYG584-A":"RORgT", 
                            "IgD":"CCR7",
                            "CD123":"-"}

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
                if len(df) < 10000:
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
            #shape
            if self.dim != 0:
                if self.size!=None:
                    s = str(self.size[0])
                    for i in range(1,len(self.size)):
                        s+=" "+str(self.size[i])
                    f.write(s+"\n")
                else:
                    f.write("-1\n")
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
                s_size = f.readline()
                if s_size=="-1":
                    size = None
                else:
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
            

            
    def _save_data(self,index,data,idd=False,pheno=1):
        
        if idd!=False:
            file = self.data + idd + ".dat"
        else:
            pheno = self.pheno[index]
            file = self.data + self.id[index] + ".dat"
        with open(file,"wb") as f:
            if self.dim ==0:
                nlin = len(data)
                ncol = len(data[0])
                d = struct.pack("i i i",pheno ,nlin,ncol)
                f.write(d)
                df = data.flatten().astype(np.float64).tobytes()
                f.write(df)
            elif self.size[0] <0:
                nlin = len(data)
                ncol = len(data[0])
                d = struct.pack("i i i",pheno ,nlin,ncol)
                f.write(d)
                df = data.flatten().astype(np.float64).tobytes()
                f.write(df)

    def _get_data(self, index,shape = False):
        file = self.data + self.id[index] + ".dat"
        with open(file,"rb") as f:
            if shape ==False:
                d= f.read(12)
                y,nlin,ncol =struct.unpack("i i i", d)
                sz = nlin*ncol
                d= f.read(sz*struct.calcsize("d"))
                data = struct.unpack(str(sz)+"d", d)
                data = np.array(data)
                data = data.reshape((nlin, ncol))
                return data,y
            else:
                d= f.read(12)
                y,nlin,ncol =struct.unpack("i i i", d)
                aux = 1
                for a in shape:
                    aux= aux*a
                d= f.read(aux*struct.calcsize("d"))
                data = struct.unpack(str(aux)+"d", d)
                data = np.array(data)
                data = data.reshape(shape)
                return data,y
            
    def readfile(self, file):
        with open(file,"rb") as f:
                d= f.read(12)
                y,nlin,ncol =struct.unpack("i i i", d)
                sz = nlin*ncol
                d= f.read(sz*struct.calcsize("d"))
                data = struct.unpack(str(sz)+"d", d)
                data = np.array(data)
                data = data.reshape((nlin, ncol))
                return data,y
    def writefile(self,file,data,y):
        with open(file,"wb") as f:
            nlin = len(data)
            ncol = len(data[0])
            d = struct.pack("i i i", y,nlin,ncol)
            f.write(d)
            df = data.flatten().astype(np.float64).tobytes()
            f.write(df)    
    def _get_pheno(self,index):
        file = self.data + self.id[index] + ".dat"
        with open(file,"rb") as f:
            d = f.read(4)
            return struct.unpack("i", d)[0]
        
    def _sample_data(self,index ,num_lin):
        df,y = self._get_data(index)
        if num_lin > len(df):
            print("sample " + str(index) + " num cells large " + str(len(df)))
            return
        else:
            ilin = np.arange(len(df))
            np.random.shuffle(ilin)
            df = df[ilin[:num_lin],:]
            return df
                
    def get_poll_cells(self,fold=None, filename=None, balanciate=True,num_cells=1000, save=False):
        df = self._sample_data(0,num_cells)
        df_y = np.repeat(self.pheno[0], num_cells)
        tam = len(self.id)
        print("generate sample")
        for i in range(1,tam):
            print(str(i)+ " of "+str(tam))
            df = np.concatenate((df, self._sample_data(i,num_cells)), axis=0)
            df_y = np.concatenate((df_y,np.repeat(self.pheno[i], num_cells)))
        if balanciate:
            df,df_y = self._oversample(df, df_y)  
        if save:
            print("Saving pooled cells.")
            df_y = df_y.reshape(-1, 1)
            df_combined = np.hstack((df, df_y))
            df = pd.DataFrame(df_combined)
            df.to_csv(fold+filename)
            print("Saved pooled cell file.")  
        return(df,df_y)
        
            
    def split_data_test(self,fold_train,fold_test,perc_train = 0.7):
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

        
    def augmentation(self,factor,n_jobs=15,numcells=False):
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
            if numcells!=False:
                vec =  np.arange(len(df))
                np.random.shuffle(vec)
                df = df[vec[:numcells]]
            self._save_data(len(self.id)-1, df)
        #aumentation
        print("aumentation")
        fac = int(len(self.id)*factor)
        def aument(pos,neg,i,numcells,self):
            idd_n = []
            batch_n = []
            pheno_n = []
            samp1 = neg[random.randint(0, len(neg)-1)]
            samp2 = neg[random.randint(0, len(neg)-1)]
            pheno_n.append(self.pheno[samp1])
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            idd_n.append(idd)
            df1 = self._get_data(samp1)[0]
            df2 = self._get_data(samp2)[0]
            n_lin = np.min((len(df1),len(df2)))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            batch = self.batch[samp1]+"_"+self.batch[samp2]
            batch_n.append(batch)
            if numcells!=False:
                vec =  np.arange(len(df))
                np.random.shuffle(vec)
                df = df[vec[:numcells]]
            self._save_data(i, df,idd_n[0],0)
            #pos
            samp1 = pos[random.randint(0, len(pos)-1)]
            samp2 = pos[random.randint(0, len(pos)-1)]
            pheno_n.append(self.pheno[samp1])
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            idd_n.append(idd)
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
            batch_n.append(batch)
            if numcells!=False:
                vec =  np.arange(len(df))
                np.random.shuffle(vec)
                df = df[vec[:numcells]]
            self._save_data(i+1, df,idd_n[1],1)
            return{"id":idd_n,"batch":batch_n,"pheno":pheno_n}
        
        print("start")
        res = Parallel(n_jobs=n_jobs,verbose=10)(delayed(aument)(pos,neg,i,numcells,self) for i in range(len(self.id),fac,2))
        #res = aument(pos,neg,len(self.id)+10,self)
        print("stop")    
        for d in res:
            self.id.append(d["id"][0])
            self.id.append(d["id"][1])
            self.batch.append((d["batch"][0]))
            self.batch.append((d["batch"][1]))
            self.pheno.append(d["pheno"][0])
            self.pheno.append(d["pheno"][1])
        if numcells!=False:
            for s in neg:
                df = self._sample_data(s ,numcells)
                self._save_data(s, df,pheno=self.pheno[s])
            for s in pos:
                df = self._sample_data(s ,numcells)
                self._save_data(s, df,pheno=self.pheno[s])
        self._save_meta()
            
    def augmentation_by_batch(self,factor):
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
               samp1 = neg[random.randint(0, len(neg)-1)]
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
            samp1 = pos[random.randint(0, len(neg)-1)]
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
             

    
    # def _feature_inportance(self,num_cells=1000,cv = 5,n_jobs = 15):
    #     neg = pd.DataFrame()
    #     pos = pd.DataFrame()
    #     print("sample data")
    #     tam = len(self.id)
    #     for i in range(tam):
    #         print(str(i) + " of " + str(tam))
    #         if self.pheno[i]==0:
    #             neg = pd.concat([neg, pd.DataFrame(self._sample_data(i, num_cells))], axis=0)
    #         else:
    #             pos = pd.concat([pos, pd.DataFrame(self._sample_data(i, num_cells))], axis=0)
    #     neg["y"] = 0
    #     pos["y"] = 1
    #     x = pd.concat([neg, pos], axis=0)
    #     y = x["y"].copy()
    #     x.drop('y', axis=1, inplace=True)
        
        
    #     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10,stratify=y,shuffle=True,random_state=self.seed)
    #     print("train model")
    #     self.seed +=1
    #     rf = RF(random_state=self.seed ,n_jobs = n_jobs)
    #     rf.fit(x_train, y_train)
    #     y_pred = rf.predict(x_test)
    #     y_ppred = rf.predict_proba(x_test)[:,1]
    #     mod = {}
    #     mod["acuracy"] = accuracy_score(y_test,y_pred)
    #     mod["b_acuracy"] = balanced_accuracy_score(y_test,y_pred)
    #     mod["ROC"] = roc_auc_score(y_test,y_ppred)
    #     mod["importance"] = pd.DataFrame({"mark":self.painel,"importance":rf.rf.feature_importances_})
    #     mod["y_t"] = y_test
    #     mod["y_t_pred"] = y_pred
    #     mod["y_t_ppred"] = y_ppred
    #     mod["par"] = rf.par
    #     rf.fit(x,y)
    #     mod["x"] = x
    #     mod["y"] = y
    #     mod["y_pred"] = rf.predict_proba(x)[:,1]
    #     mod["painel"] = self.painel
    #     return mod
    
    def _oversample(self,df,y):
        neg = [i for i in range(len(y)) if y[i]==0]
        pos = [i for i in range(len(y)) if y[i]==1]
        s_neg = len(neg)
        s_pos = len(pos)
        idd = []
        y_n = []
        while s_neg !=s_pos:
            if s_neg>s_pos:
                idd.append(pos[random.randint(0, len(pos)-1)])
                y_n.append(1)
                s_pos+=1
            else:
                idd.append(neg[random.randint(0, len(neg)-1)])
                y_n.append(0)
                s_neg+=1
        df1 = np.take(df, idd, axis=0)
        df = np.concatenate((df, df1), axis=0)
        y_n = list(y)+y_n
        y_n = np.array(y_n)
        return df,y_n
                 
    def sample_all_cells(self,numcells):
        
        tam = len(self.id)
        print("sample cells")
        for i in range(tam):
            print(str(i)+" of "+str(tam))
            df =self._sample_data(i,numcells)
            self._save_data(i, df)
    
    def get_dataload(self,fold_train,fold_val,fold_test):
        train = Data()
        train.load(fold_train)
        test = Data()
        test.load(fold_test)
        val = Data()
        val.load(fold_val)
        if train.size!=None:
            if train.size[0]>0:
                return self.AdDataset(train,train.size),self.AdDataset(val,train.size),self.AdDataset(test,train.size)
            else:
                aux1 =train._get_data(0)
                shape = list(aux1[0].shape)
                return self.AdDataset(train,shape),self.AdDataset(val,shape),self.AdDataset(test,shape)
        shape = list(aux1[0].shape)
        return self.AdDataset(train,shape),self.AdDataset(val,shape),self.AdDataset(test,shape)
                  
    def umap_space(self,num_cells=1000):
         df = self._sample_data(0, num_cells)
         tam = len(self.id)
         print("generate sample")
         for i in range(1,tam):
             print(str(i)+ " of "+str(tam))
             df = np.concatenate((df, self._sample_data(i, num_cells)), axis=0)
         return df.copy()
        

class Standard_tranformer:
    def __init__(self,by_batch=False,num_cells=1000):
        self.by_batch=by_batch
        self.batch = None
        self.mean = None
        self.sd = None
        self.num_cells=num_cells
    def fit(self,fold):
        data = Data()
        data.load(fold)
        if self.by_batch:
            self.batch = []
            self.sd = []
            self.mean = []
            u_batch = np.unique(data.batch)
            for b in u_batch:
                self.batch.append(b)
                print(b)
                idd = []
                yd = []
                for i in range(len(data.id)):
                    if b==data.batch[i]:
                        idd.append(i)
                        yd.append(i)
                df = data._sample_data(idd[0], self.num_cells)
                df_y = np.repeat(data.pheno[yd[0]], self.num_cells)
                tam = len(idd)
                print("generate sample")
                for i in range(1,tam):
                    print(str(i)+ " of "+str(tam))
                    df = np.concatenate((df, data._sample_data(idd[i], self.num_cells)), axis=0)
                    df_y = np.concatenate((df_y,np.repeat(data.pheno[yd[i]], self.num_cells)))
                df = data._oversample(df, df_y)[0]
                scaler = StandardScaler()
                scaler.fit(df)
                self.sd.append(scaler.scale_)
                self.mean.append(scaler.mean_)
        else:
            tam = len(data.id)
            df = data._sample_data(0, self.num_cells)
            print("generate sample")
            for i in range(1,tam):
                print(str(i)+ " of "+str(tam))
                df = np.concatenate((df, data._sample_data(i, self.num_cells)), axis=0)
            scaler = StandardScaler()
            scaler.fit(df)
            self.sd = scaler.scale_
            self.mean = scaler.mean_
            
    def transform(self,folder):
        data = Data()
        data.load(folder)
        if self.by_batch:
            scaler = StandardScaler()
            for b in range(len(self.batch)):
                print(self.batch[b])
                tam = len(data.id)
                for i in range(tam):
                    
                    if self.batch[b]==data.batch[i]:
                        print(str(i))
                        scaler.mean_=self.mean[b]
                        scaler.scale_ = self.sd[b]
                        df = data._get_data(i)[0]                
                        df = np.array(scaler.transform(df))
                        data._save_data(i, df)
        else:
            scaler = StandardScaler()
            tam = len(data.id)
            for i in range(tam):
                scaler.mean_=self.mean
                scaler.scale_ = self.sd
                print(str(i)+ " of "+str(tam))
                df = data._get_data(i)[0]
                df = np.array(scaler.transform(df))
                data._save_data(i, df)
    
    def save(self,file):
        f = open(file,"wb")
        pk.dump(self,f)
        f.close()
    
    def load(self,file):
        f = open(file,"rb")
        a = pk.load(f)
        f.close()
        self.mean = a.mean
        self.sd = a.sd
        self.num_cells = a.num_cells
        self.batch = a.batch
        self.by_batch=a.by_batch
        
    
class Log_transformer():
    def fit_transform(self,folder):
        data = Data()
        data.load(folder)
        tam = len(data.id)
        for i in range(tam):
            print(str(i)+" of "+str(tam))
            df = data._get_data(i)[0]
            df = np.log1p(df)
            data._save_data(i, df)
            
class Oversample():
    def fit_transform(self,folder):
        data = Data()
        data.load(folder)
        y = data.pheno
        neg = [i for i in range(len(y)) if y[i]==0]
        pos = [i for i in range(len(y)) if y[i]==1]
        s_neg = len(neg)
        s_pos = len(pos)
        while s_neg !=s_pos:
            if s_neg>s_pos:
                idd = pos[random.randint(0, len(pos)-1)]
                nidd = data.id[idd]+"_"
                data.id.append(nidd)
                data.pheno.append(data.pheno[idd])
                data.batch.append(data.batch[idd])
                src = data.data + data.id[idd] + ".dat"
                dest = data.data + nidd + ".dat"
                shutil.copyfile(src,dest)
                s_pos+=1
            else:
                idd = neg[random.randint(0, len(neg)-1)]
                nidd = data.id[idd]+"_"
                data.id.append(nidd)
                data.pheno.append(data.pheno[idd])
                data.batch.append(data.batch[idd])
                src = data.data + data.id[idd] + ".dat"
                dest = data.data + nidd + ".dat"
                shutil.copyfile(src,dest)
                s_neg+=1
        data._save_meta()

class Umap_tranformer:
    def __init__(self,dimentions=2):
        self.dimention=dimentions
        self.x = None
        
    def fit(self,data,num_cells=1000):
        x,y = data.get_poll_cells(num_cells=num_cells)
        
        self.space = umap.UMAP(n_components=self.dimention,densmap=False, low_memory=False,min_dist = 0.0)
        self.space.fit(x)
        x_ = self.space.transform(x)
        x_ = np.concatenate((x, x_),axis=1)
        self.x = x_.copy()
    
    def save_umap_points(self,file):
        n_col = len(self.x[0])
        n_lin = len(self.x)
        f = open(file,"w")
        f.write(str(n_lin)+" "+str(n_col)+" "+str(self.dimention)+"\n")
        for l in range(n_lin):
            f.write(str(self.x[l][0]))
            for c in range(1,n_col):
                f.write(" "+str(self.x[l][c]))
            f.write("\n")
        f.close()
        
    
    def transform(self,data,umap_space,n_jobs=15):
        def cal_st2(i,data,umap_space):
            x,y = data._get_data(i)
            space = Umap_tranformer()
            space.load(umap_space)
            space = space.space
            x_ = space.transform(x)
            x_ = np.concatenate((x, x_),axis=1)
            data._save_data(i,x_)
        tam = len(data.id)
        v = list(range(tam))
        Parallel(n_jobs=n_jobs,verbose=10)(delayed(cal_st2)(p,data,umap_space) for p in v)
        #cal_st2(self.space,0,data)
        data.painel=data.painel+["dim"+str(i+1) for i in range(self.dimention)]
        data.dim = self.dimention
        data.size = [-1]
        data._save_meta()    
        
    
    def save(self,file):
        f = open(file,"wb")
        pk.dump(self,f,pk.HIGHEST_PROTOCOL)
        f.close()
    
    def load(self,file):
        f = open(file,"rb")
        a = pk.load(f)
        f.close()
        self.dimention=a.dimention
        self.space = a.space
        self.x = a.x
        
class Cell_Umap_tranformer:
    def __init__(self,n_jobs = 15):
        self.num_partition=None
        self.n_jobs=n_jobs
        self.file_split = None
        
    def fit(self,file_space,file_split,num_partition):
        self.num_partition=num_partition
        self.file_split = file_split
        fold = os.getcwd()
        #p   file_quimera   num_partition  file_split
        print("start")
        subprocess.Popen([fold+"/Flow_c.exe","p",file_space,str(self.num_partition),file_split]).wait()
        print("end")
        
        
    def transform(self,data):
        v = []
        for i in range(len(data.id)):
            v.append((fold+"/Flow_c.exe",data.data + data.id[i] + ".dat",self.file_split))
        def multi(v):
            program,file, file_split = v
            subprocess.Popen([program,"c",file,file_split]).wait()
            
        Parallel(n_jobs=self.n_jobs,verbose=10)(delayed(multi)(a) for a in v)             
        data.size = [-2,len(data.painel),self.num_partition] 
        data._save_meta()
    
    def save(self,file):
        f = open(file,"wb")
        pk.dump(self,f,pk.HIGHEST_PROTOCOL)
        f.close()
    
    def load(self,file):
        f = open(file,"rb")
        a = pk.load(f)
        f.close()
        self.dimention=a.dimention
        self.space = a.space
        self.x = a.x
        
        
class Density_tranformer:
    def __init__(self,n_jobs = 15):
        self.num_partition=None
        self.n_jobs=n_jobs
        self.file_split = None
        
    def fit(self,file_space,file_split,num_partition):
        self.num_partition=num_partition
        self.file_split = file_split
        fold = os.getcwd()
        #p   file_quimera   num_partition  file_split
        print("start")
        subprocess.Popen([fold.replace("\\","/")+"/Flow_c.exe","p",file_space.replace("\\","/"),str(self.num_partition),file_split.replace("\\","/")]).wait()
        print("end")
        
        
    def transform(self,data):
        v = []
        for i in range(len(data.id)):
            v.append((fold+"/Flow_c.exe",data.data + data.id[i] + ".dat",self.file_split))
        def multi(v):
            program,file, file_split = v
            subprocess.Popen([program.replace("\\","/"),"s",file.replace("\\","/"),file_split.replace("\\","/")]).wait()
            
        #Parallel(n_jobs=self.n_jobs,verbose=10)(delayed(multi)(a) for a in v)
        multi(v[0])
        data.size = [len(data.painel)-data.dim+1]
        for d in range(data.dim):
            data.size += [self.num_partition]
        data._save_meta()
    
    def save(self,file):
        f = open(file,"wb")
        pk.dump(self,f,pk.HIGHEST_PROTOCOL)
        f.close()
    
    def load(self,file):
        f = open(file,"rb")
        a = pk.load(f)
        f.close()
        self.dimention=a.dimention
        self.space = a.space
        self.x = a.x 