# -*- coding: utf-8 -*-

## Script to perform random data augmentation

# 0. IMPORT STATEMENTS 

import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
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
import re
#import subprocess



class Data:

    def __init__(self,seed = 123571113):
        self.id = None
        self.pheno = None
        self.data = None 
        self.batch = None
        self.painel = None
        self.map_batch = None
        self.dim = None
        self.sizes = None
        np.random.seed(seed)
        random.seed(seed)
        
    def save_c(self,file):
        with open(file,"wb") as f:
            #id
            s =self.id[0]
            tam = len(self.id)
            for i in range(1,tam):
                s = s+"," + self.id[i]
            tam = len(s)
            d =struct.pack("i", tam)
            f.write(d)
            mask = str(tam)+"s"
            d = struct.pack(mask, bytes(s, 'utf-8'))
            f.write(d)
            #painel
            s =self.painel[0]
            ncol = len(self.painel)
            for i in range(1,ncol):
                s = s+"," + self.painel[i]
            tam = len(s)
            d =struct.pack("i", tam)
            f.write(d)
            mask = str(tam)+"s"
            d = struct.pack(mask, bytes(s, 'utf-8'))
            f.write(d)
            #batch
            s =self.batch[0]
            ncol = len(self.batch)
            for i in range(1,ncol):
                s = s+"," + self.batch[i]
            tam = len(s)
            d =struct.pack("i", tam)
            f.write(d)
            mask = str(tam)+"s"
            d = struct.pack(mask, bytes(s, 'utf-8'))
            f.write(d)
            #dim
            d =struct.pack("i", self.dim)
            f.write(d)
            if self.dim == 0:
                #data
                tam = len(self.id)
                for i in range(tam):
                    print("save " + str(i) + " of " + str(tam))
                    #nlin, ncol
                    nlin = len(self.data[i])
                    ncol = len(self.data[i][0])
                    d = struct.pack("i i", nlin,ncol)
                    f.write(d)
                    #value
                    for l in range(nlin):
                        for c in range(ncol):
                            f.write(struct.pack("f", self.data[i][l][c]))
                 
            
            
    def load_c(self,file):
        with open(file,"rb") as f:
            #id
            d = f.read(struct.calcsize("i"))
            tam2 = struct.unpack("i", d)[0]
            s = f.read(tam2)
            s = struct.unpack(str(tam2)+"s", s)[0]
            self.id = tuple(s.decode().split(","))
            #painel
            d = f.read(struct.calcsize("i"))
            tam2 = struct.unpack("i", d)[0]
            s = f.read(tam2)
            s = struct.unpack(str(tam2)+"s", s)[0]
            self.painel = tuple(s.decode().split(","))
            #batch
            d = f.read(struct.calcsize("i"))
            tam2 = struct.unpack("i", d)[0]
            s = f.read(tam2)
            s = struct.unpack(str(tam2)+"s", s)[0]
            self.batch = tuple(s.decode().split(","))
            #map
            self.map_batch = {}
            for i in range(len(self.id)):
                self.map_batch[self.id[i]] = self.batch[i]
            #dim
            d = f.read(struct.calcsize("i"))
            self.dim = struct.unpack("i", d)[0]
            #size
            if self.dim ==0:
                #data
                tam = len(self.id)
                self.data = []
                for i in range(tam):
                    print("load " + str(i) + " of " + str(tam))
                    d= f.read(struct.calcsize("i i"))
                    nlin,ncol =struct.unpack("i i", d)
                    data = np.ones((nlin,ncol),np.float32)
                    for l in range(nlin):
                        for c in range(ncol):
                            d = f.read(struct.calcsize("f"))
                            data[l,c] = struct.unpack("f", d)[0]
                    self.data.append(data.tolist())
            else:
                self.sizes = None
            
    def save(self,sfile):
        file = open(sfile,"wb")
        pk.dump([self.id,self.pheno,self.data,self.batch,self.painel,self.dim,self.sizes], file)
        file.close()
        
    def load(self,sfile):
        file = open(sfile,"rb")
        self.id,self.pheno,self.data,self.batch,self.painel,self.dim,self.sizes = pk.load(file)
        file.close()
        self.map_batch = {}
        tam = len(self.id)
        for i in range(tam):
            self.map_batch[self.id[i]] = self.batch[i]
        
        
    def start_transform(self,folder):
        batch_f = os.listdir(folder)
        print(batch_f)
        self.id = []
        self.pheno = []
        self.data = []
        self.batch = []
        self.painel = []
        self.map_batch = {}
        self.dim = 0
        self.sizes = None
        for b in batch_f:
            files = os.listdir(folder + "/"+b)
            #if b=="ST1_20200123":
            print(b)
            count = 1
            for f in files:
                print(str(count) + " of " + str(len(files)))
                count+=1
                s = f.replace(" ", "").replace("-", "").replace("_", "")
                #print(s)
                x = re.findall(r"AD(\d\d\d)([CP])",s )[0]
                idd = "AD"+x[0]+x[1]
                self.id.append(idd)
                self.batch.append(b)
                self.map_batch[idd] = b
                if x[1]=="C":
                    self.pheno.append(0)
                else:
                    self.pheno.append(1)
                file = folder + "/" + b + "/" + f
                df = pd.read_csv(file)
                my_cols = list(df.columns)
                for c in df.columns:
                    if c.startswith("-"):
                        my_cols.remove(c)
                my_cols.remove("Time")
                if "livedead" in my_cols:
                    my_cols.remove("livedead")
                if "Livedead" in my_cols:
                    my_cols.remove("Livedead")
                df = df[my_cols]
                
                if len(self.painel)<1:
                    self.painel = my_cols
                # else:
                #     for i in range(len(my_cols)):
                #         if my_cols[i].upper() != self.painel[i].upper():
                #             print("painel "+ self.painel[i] + " \t col " + my_cols[i])
                        
                self.data.append(df.to_numpy())
                
                
                
    def _sample_data(self,index ,num_lin):
        df = self.data[index]
        
        if num_lin > len(df):
            print("sample " + str(index) + " num cells large " + str(len(df)))
            return
        else:
            ilin = np.arange(len(df))
            np.random.shuffle(ilin)
            df = df[ilin[:num_lin],:]
            self.data[index] = df
    
    def sample_all(self,num_lin):
        for i in range(len(self.id)):
            self._sample_data(i, num_lin)
        
            
    def split_data_test(self,perc_train = 0.7):
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
        data_train.id = [self.id[i] for i in train]
        data_test.id = [self.id[i] for i in test]
        
        data_train.pheno = [self.pheno[i] for i in train]
        data_test.pheno = [self.pheno[i] for i in test]
        
        data_train.data = [self.data[i] for i in train]
        data_test.data = [self.data[i] for i in test]

        data_train.batch = [self.batch[i] for i in train]
        data_test.batch = [self.batch[i] for i in test]
        
        data_train.painel= self.painel
        data_test.painel =self.painel
        
        data_train.map_batch = {}
        data_test.map_batch = {}
        for i in train:
            data_train.map_batch[self.id[i]] = self.batch[i]
        for i in test:
            data_test.map_batch[self.id[i]] = self.batch[i]
        
        data_train.dim = self.dim
        data_test.dim = self.dim
        data_train.sizes = self.sizes
        data_test.sizes = self.sizes
        
        return data_train,data_test

        
    def augmentation(self,factor):
        #balanciate
        pos = [i for i in range(len(self.id)) if self.pheno[i]==1]
        neg = [i for i in range(len(self.id)) if self.pheno[i]==0]
        num_neg = len(neg)
        num_pos = len(pos)
        while(num_neg!=num_pos):
            if num_neg < num_pos:
               samp1 = neg[random.randint(0, len(neg))]
               samp2 = neg[random.randint(0, len(neg))]
               num_neg+=1
            else:
               samp1 = pos[random.randint(0, len(neg))]
               samp2 = pos[random.randint(0, len(neg))]
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
            
            self.data.append(df) 
            batch = self.batch[samp1]+self.batch[samp2]
            self.batch.append(batch)
            self.map_batch[idd] = batch
        #aumentation
        fac = int(num_pos*factor)
        while(num_pos< fac):
            #neg
            samp1 = neg[random.randint(0, len(neg))]
            samp2 = neg[random.randint(0, len(neg))]
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
            self.map_batch[idd] = batch
            #pos
            samp1 = pos[random.randint(0, len(neg))]
            samp2 = pos[random.randint(0, len(neg))]
            split = random.uniform(0, 1)
            idd = self.id[samp1]+self.id[samp2]
            self.id.append(idd)
            df1 = self.data[samp1]
            df2 = self.data[samp2]
            n_lin = np.min(len(df1),len(df2))
            aux1 = int(n_lin*split)
            aux2 = n_lin -aux1
            np.random.shuffle(df1)
            np.random.shuffle(df2)
            df = np.concatenate((df1[:aux1], df2[:aux2]), axis=0)
            self.pheno.append(self.pheno[samp1])
            self.data.append(df) 
            batch = self.batch[samp1]+self.batch[samp2]
            self.batch.append(batch)
            self.map_batch[idd] = batch
            num_pos+=1
            
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
            
            self.data.append(df) 
            batch = self.batch[samp1]
            self.batch.append(batch)
            self.map_batch[idd] = batch
        #aumentation
        print("aumentation")
        fac = int(num_pos*factor)
        diff = abs(fac-num_pos)
        i=1
        while(num_pos< fac):
            print(str(i), " out of ", str(diff))
            i+=1
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
            self.map_batch[idd] = batch
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
            self.map_batch[idd] = batch
            num_pos+=1
             
            
               
                