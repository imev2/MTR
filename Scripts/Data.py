# -*- coding: utf-8 -*-

## Script to perform random data augmentation

# 0. IMPORT STATEMENTS 

import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from joblib import Parallel, delayed
#import plotly.express as px
#import umap
#import pickle as pk
import os
#from os import listdir
#from os.path import isfile, join
#import random
import numpy as np
import struct
#import subprocess

class Data:

    def __init__(self):
        self.id = None
        self.pheno = None
        self.data = None 
        self.batch = None
        self.painel = None
        self.map_batch = None
        self.dim = None
        self.sizes = None
    def save(self,file):
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
            if self.dim != 0:
                #size
                mask = str(len(self.dim))+"i"
                d = struct.pack(mask, bytearray(self.sizes))
                f.write(d)
                #data
                tam = len(self.id)
                for i in range(tam):
                    #nlin, ncol
                    nlin = len(self.data[i])
                    ncol = len(self.data[i][0])
                    d = struct.pack("i i", nlin,ncol)
                    f.write(d)
                    #value
                    for l in range(nlin):
                        for c in range(ncol):
                            f.write(struct.pack("f", self.data[i][l][c]))
                 
            
            
    def load(self,file):
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
            #dim
            d = f.read(struct.calcsize("i"))
            self.dim = struct.unpack("i", d)[0]
            #size
            if self.dim !=0:
                mask = str(len(self.dim))+"i"
                d = f.read(struct.calcsize(mask))
                self.size = struct.unpack(mask, d)
                #data
                tam = len(self.id)
                self.data = []
                for i in range(tam):
                    nlin,ncol = f.read(struct.calcsize("i i"))
                    data = np.ones((nlin,ncol),np.float32)
                    for l in range(nlin):
                        for c in range(ncol):
                            d = f.read(struct.calcsize("f"))
                            var = struct.unpack("f", d)[0]
                            data[l,c] = var
                    self.data.append(data.tolist())
            else:
                self.size = None
            
    
    def start_transform(self,folder):
        batch_f = os.listdir(folder)
        print(batch_f)
        self.id = []
        self.pheno = []
        self.data = []
        self.batch = []
        self.painel = []
        self.map_batch = {}
        self.dim = None
        self.sizes = None
        for b in batch_f:
            files = os.listdir(folder + "/"+b)
            for f in files:
                idd = f
                
        
        
            
                 
        
           
            