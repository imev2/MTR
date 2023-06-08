# -*- coding: utf-8 -*-

## Script to perform random data augmentation

# 0. IMPORT STATEMENTS 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import plotly.express as px
import umap
import pickle as pk
import os
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import subprocess

seed = 123571113
random.seed(seed)
n_thread = 15
fold = os.getcwd()
perc_split = 0.7


# 1. FUNCTIONS 
def dataRecombination(file1, file2, perc1, outputDir="Data/Expanded"):
    print(file1 + " " + file2)
    file = open(outputDir+file1+".csv","r")
    lin1 = file.readlines()
    file.close()
    file = open(outputDir+file2+".csv","r")
    lin2 = file.readlines()
    file.close()
    d = lin1[0].split("\t")[1]
    del lin1[0]
    del lin2[0]
    random.shuffle(lin1)
    random.shuffle(lin2)
    tam = min((len(lin1),len(lin2)))
    n = round(tam*perc1)
    p = tam-n
    lin1 = lin1[:n]
    lin2 = lin2[:p]
    total = lin1+lin2
    random.shuffle(total)
    total.insert(0,str(len(total))+ "\t"+ d)
    file = open(outputDir+file1+file2+".csv","w")
    file.writelines(total)
    file.close()
    

def dataExpansion(directory, n, perc1, perc2):

    # Get a list of all csv files in the directory
    csvFiles = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')] #COULD POTENTIALLY FILTER ON C OR P HERE
    print(csvFiles)
    
    # Sampling without because otherwise we are skewing the density of the points (ex. getting a lot of repeats)
    for _ in range(n):
        file1, file2 = random.sample(csvFiles, 2)
        print(f'Combining {file1} and {file2}')

        # Combine the selected files
        dataRecombination(file1, file2, perc1, perc2)

# 2. USING FUNCTIONS
def split_data():
    folds = 5
    
    #ST1
    table = pd.read_csv(fold +"/data/"+"ST1.csv",sep="\t",header=None,names=("file_name","id","batch")) # Read in all ST1 file names. 
    
    #split train_test
    y = []
    for i in table["id"]:
        i = i.upper()
        if "P" in i:
            y.append(1) # Positives = 1
        elif "C" in i:
            y.append(0) # Controls = 0
        else:
            y.append(pd.NA)
            print(i)
    # y's are the outcomes
    
    del table["file_name"]   
    x_train, x_test, y_train, y_test = train_test_split(table, y, test_size=1/folds, random_state=seed,stratify=y) # balances number of positives and controls in the test and train groups to match the overall groups
    x_train["y"] = y_train
    x_test["y"] = y_test
    x_train.to_csv(fold +"/data/"+"ST1_train.csv",sep="\t",header=None,index = False)
    x_test.to_csv(fold +"/data/"+"ST1_test.csv",sep="\t",header=None,index = False)
    
    #ST2
    table = pd.read_csv(fold +"/data/"+"ST1.csv",sep="\t",header=None,names=("file_name","id","batch"))
    y = []
    for i in table["id"]:
        i = i.upper()
        if "P" in i:
            y.append(1)
        elif "C" in i:
            y.append(0)
        else:
            y.append(pd.NA)
            print(i)
    del table["file_name"]
    x_train, x_test, y_train, y_test = train_test_split(table, y, test_size=1/folds, random_state=seed,stratify=y)
    x_train["y"] = y_train
    x_test["y"] = y_test
    x_train.to_csv(fold +"/data/"+"ST1_train.csv",sep="\t",header=None,index = False)
    x_test.to_csv(fold +"/data/"+"ST1_test.csv",sep="\t",header=None,index = False)
    
    #ST2
    table = pd.read_csv(fold +"/data/"+"ST2.csv",sep="\t",header=None,names=("file_name","id","batch"))
    #split train_test
    y = []
    for i in table["id"]:
        i = i.upper()
        if "P" in i:
            y.append(1)
        elif "C" in i:
            y.append(0)
        else:
            y.append(pd.NA)
            print(i)
    
    del table["file_name"]    
    x_train, x_test, y_train, y_test = train_test_split(table, y, test_size=1/folds, random_state=seed,stratify=y)
    x_train["y"] = y_train
    x_test["y"] = y_test
    x_train.to_csv(fold +"/data/"+"ST2_train.csv",sep="\t",header=None,index = False)
    x_test.to_csv(fold +"/data/"+"ST2_test.csv",sep="\t",header=None,index = False)
    
    #ST2
    table = pd.read_csv(fold +"/data/"+"ST2.csv",sep="\t",header=None,names=("file_name","id","batch"))
    y = []
    for i in table["id"]:
        i = i.upper()
        if "P" in i:
            y.append(1)
        elif "C" in i:
            y.append(0)
        else:
            y.append(pd.NA)
            print(i)
    del table["file_name"]
    x_train, x_test, y_train, y_test = train_test_split(table, y, test_size=1/folds, random_state=seed,stratify=y)
    x_train["y"] = y_train
    x_test["y"] = y_test
    x_train.to_csv(fold +"/data/"+"ST2_train.csv",sep="\t",header=None,index = False)
    x_test.to_csv(fold +"/data/"+"ST2_test.csv",sep="\t",header=None,index = False)
    

#batch normalization
def batch_normalization():
    sample_size = 1000
    #ST1
    print("\nST1\n")
    train = pd.read_csv(fold +"/data/"+"ST1_train.csv",sep="\t",header=None,names=("id","batch","y")) 
    test = pd.read_csv(fold +"/data/"+"ST1_test.csv",sep="\t",header=None,names=("id","batch","y"))
    scale = StandardScaler(with_mean= True,with_std= True)
    mapa = {}
    l_batchs = pd.unique(train.batch)
    
    for b in l_batchs:
        print(b)
        f_idd = train.loc[train["batch"]==b,"id"]
        n_df = pd.DataFrame()
        for idd in f_idd:
            df = pd.read_csv(fold + "/data/ProcessData/ST1/"+idd+".csv",sep="\t")    
            n_df = pd.concat([n_df,df.sample(sample_size,replace=False)],ignore_index=True)
        colum = n_df.columns
        #n_df = n_df.to_numpy()
        #mi = np.min(n_df)
        #n_df = n_df - mi
        #n_df = np.log1p(n_df)
        scale.fit(n_df)
        mapa[b] = [scale.scale_,scale.mean_,[string.upper().split("(CD")[0] for string in colum]]
    
    scale = StandardScaler()
    print("\ntransform\n")
    for i in range(len(train.id)):
        idd = train.id.iloc[i]
        print(idd)
        batch = train.batch.iloc[i]
        df = pd.read_csv(fold + "/data/ProcessData/ST1/"+idd+".csv",sep="\t")
        df.columns = [string.upper().split("(CD")[0] for string in df.columns]
        col_name = df.columns
        scale.scale_,scale.mean_,scale.feature_names_in_ = mapa[batch]
        df = df.to_numpy()
        #mi = np.min(df)
        #df = df - mi
        #df = np.log1p(df)
        #df = pd.DataFrame(df,columns=col_name)
        df = scale.transform(df)
        df = pd.DataFrame(df,columns=col_name)
        df.to_csv(fold + "/data/ProcessData/ST1_train/"+idd+".csv",sep="\t",index=False)
    
    for i in range(len(test.id)):
        idd = test.id.iloc[i]
        print(idd)
        batch = train.batch.iloc[i]
        df = pd.read_csv(fold + "/data/ProcessData/ST1/"+idd+".csv",sep="\t")
        df.columns = [string.upper().split("(CD")[0] for string in df.columns]
        col_name = df.columns
        scale.scale_,scale.mean_,scale.feature_names_in_ = mapa[batch]
        df = df.to_numpy()
        #mi = np.min(df)
        #df = df - mi
        #df = np.log1p(df)
        #df = pd.DataFrame(df,columns=col_name)
        df = scale.transform(df)
        df = pd.DataFrame(df,columns=col_name)
        df.to_csv(fold + "/data/ProcessData/ST1_test/"+idd+".csv",sep="\t",index=False)
    
    #ST2
    print("\nST2\n")
    train = pd.read_csv(fold +"/data/"+"ST2_train.csv",sep="\t",header=None,names=("id","batch","y")) 
    test = pd.read_csv(fold +"/data/"+"ST2_test.csv",sep="\t",header=None,names=("id","batch","y")) 
    scale = StandardScaler(with_mean= True,with_std= True)
    mapa = {}
    l_batchs = pd.unique(train.batch)
    for b in l_batchs:
        print(b)
        f_idd = train.loc[train["batch"]==b,"id"]
        n_df = pd.DataFrame()
        scale = StandardScaler()
        for idd in f_idd:
            df = pd.read_csv(fold + "/data/ProcessData/ST2/"+idd+".csv",sep="\t")    
            n_df = pd.concat([n_df,df.sample(sample_size,replace=False)],ignore_index=True)
        colum = n_df.columns
        #n_df = n_df.to_numpy()
        #mi = np.min(n_df)
        #n_df = n_df - mi
        #n_df = np.log1p(n_df)
        scale.fit(n_df)
        mapa[b] = [scale.scale_,scale.mean_,[string.upper().split("(CD")[0] for string in colum]]
    
    scale = StandardScaler()
    print("\ntransform\n")
    for i in range(len(train.id)):
        idd = train.id.iloc[i]
        print(idd)
        batch = train.batch.iloc[i]
        df = pd.read_csv(fold + "/data/ProcessData/ST2/"+idd+".csv",sep="\t")
        df.columns = [string.upper().split("(CD")[0] for string in df.columns]
        col_name = df.columns
        scale.scale_,scale.mean_, scale.feature_names_in_ = mapa[batch]
        df = df.to_numpy()
        #mi = np.min(df)
        #df = df - mi
        #df = np.log1p(df)
        #df = pd.DataFrame(df,columns=col_name)
        df = scale.transform(df)
        df = pd.DataFrame(df,columns=col_name)
        df.to_csv(fold + "/data/ProcessData/ST2_train/"+idd+".csv",sep="\t",index=False)
    
    for i in range(len(test.id)):
        idd = test.id.iloc[i]
        print(idd)
        batch = train.batch.iloc[i]
        df = pd.read_csv(fold + "/data/ProcessData/ST2/"+idd+".csv",sep="\t")
        df.columns = [string.upper().split("(CD")[0] for string in df.columns]
        col_name = df.columns
        scale.scale_,scale.mean_,scale.feature_names_in_ = mapa[batch]
        df = df.to_numpy()
        #mi = np.min(df)
        #df = df - mi
        #df = np.log1p(df)
        #df = pd.DataFrame(df,columns=col_name)
        df = scale.transform(df)
        df = pd.DataFrame(df,columns=col_name)
        df.to_csv(fold + "/data/ProcessData/ST2_test/"+idd+".csv",sep="\t",index=False)
    


def umap_apply():    
    sample_size = 1000
    print("\nST1\n")
    train = pd.read_csv(fold +"/data/"+"ST1_train.csv",sep="\t",header=None,names=("id","batch","y")) 
    test = pd.read_csv(fold +"/data/"+"ST1_test.csv",sep="\t",header=None,names=("id","batch","y"))
    #train ST1
    
    f_idd = train["id"]
    i=1
    n_df = pd.DataFrame()
    mini = []
    for idd in f_idd:
        print(str(i)+" de " + str(len(f_idd)))
        idd = f_idd[0]
        df = pd.read_csv(fold + "/data/ProcessData/ST1_train/"+idd+".csv",sep="\t")
        mini.append(len(df))    
        n_df = pd.concat([n_df,df.sample(sample_size,replace=False)],ignore_index=True)
        i = i+1
    print("fit")
    dens_mapper = umap.UMAP(densmap=False, random_state=seed,low_memory=False,min_dist = 0.0)
    dens_mapper = dens_mapper.fit(n_df)
    f = open(fold+'/data/st1.pk', 'wb')
    pk.dump(dens_mapper, f, pk.HIGHEST_PROTOCOL)
    f.close()
    
    df = pd.DataFrame(dens_mapper.embedding_)
    df.to_csv(fold + "/data/ProcessData/ST1_umap_space.csv",sep="\t",index=False)
    f1 = open(fold + "/data/ProcessData/ST1_umap_space.csv","a+")
    f1.seek(0)
    f1.write(str(len(df))+"\t2"+"\n")
    f1.close()
    def cal_st1(v):
        i,n_i,file,file_out = v
        print(str(i)+" de " + str(len(f_idd)))
        f = open(fold+'/data/st1.pk', 'rb')
        model = pk.load(f)
        f.close()
        df = pd.read_csv(file,sep="\t")
        df = pd.DataFrame(model.transform(df))
        df.to_csv(file_out,sep="\t",index=False,header=False)
        f1 = open(file_out,"a+")
        f1.seek(0)
        f1.write(str(len(df))+"\t2"+"\n")
        f1.close()
    print("transform train")
    i=0
    v = []
    n_i = len(f_idd)
    for i in range(n_i):
        v.append((i,n_i,fold + "/data/ProcessData/ST1_train/"+f_idd[i]+".csv",fold + "/data/ProcessData/ST1_umap_train/"+f_idd[i]+".csv"))
        
    Parallel(n_jobs=n_thread,verbose=0)(delayed(cal_st1)(p) for p in v)
    
    print("transform test")
    f_idd = test["id"]
    i=0
    v = []
    n_i = len(f_idd)
    for i in range(n_i):
        v.append((i,n_i,fold + "/data/ProcessData/ST1_test/"+f_idd[i]+".csv",fold + "/data/ProcessData/ST1_umap_test/"+f_idd[i]+".csv"))
    
    Parallel(n_jobs=n_thread,verbose=0)(delayed(cal_st1)(p) for p in v)
    
    print("\nST2\n")
    train = pd.read_csv(fold +"/data/"+"ST2_train.csv",sep="\t",header=None,names=("id","batch","y")) 
    test = pd.read_csv(fold +"/data/"+"ST2_test.csv",sep="\t",header=None,names=("id","batch","y"))
    #train ST2
    
    f_idd = train["id"]
    i=1
    n_df = pd.DataFrame()
    mini = []
    for idd in f_idd:
        print(str(i)+" de " + str(len(f_idd)))
        idd = f_idd[0]
        df = pd.read_csv(fold + "/data/ProcessData/ST2_train/"+idd+".csv",sep="\t")
        mini.append(len(df))    
        n_df = pd.concat([n_df,df.sample(sample_size,replace=False)],ignore_index=True)
        i = i+1
    print("fit")
    dens_mapper = umap.UMAP(densmap=False, random_state=seed,low_memory=False,min_dist = 0.0)
    dens_mapper = dens_mapper.fit(n_df)
    f = open(fold+'/data/st2.pk', 'wb')
    pk.dump(dens_mapper, f, pk.HIGHEST_PROTOCOL)
    f.close()
    
    df = pd.DataFrame(dens_mapper.embedding_)
    df.to_csv(fold + "/data/ProcessData/ST2_umap_space.csv",sep="\t",index=False)
    f1 = open(fold + "/data/ProcessData/ST2_umap_space.csv","a+")
    f1.seek(0)
    f1.write(str(len(df))+"\t2"+"\n")
    f1.close()
    def cal_st2(v):
       i,n_i,file,file_out = v
       print(str(i)+" de " + str(len(f_idd)))
       f = open(fold+'/data/st2.pk', 'rb')
       model = pk.load(f)
       f.close()
       df = pd.read_csv(file,sep="\t")
       df = pd.DataFrame(model.transform(df))
       df.to_csv(file_out,sep="\t",index=False,header=False)
       f1 = open(file_out,"a+")
       f1.seek(0)
       f1.write(str(len(df))+"\t2"+"\n")
       f1.close()
    print("transform train")
    i=0
    v = []
    n_i = len(f_idd)
    for i in range(n_i):
        v.append((i,n_i,fold + "/data/ProcessData/ST2_train/"+f_idd[i]+".csv",fold + "/data/ProcessData/ST2_umap_train/"+f_idd[i]+".csv"))
        
    Parallel(n_jobs=n_thread,verbose=0)(delayed(cal_st2)(p) for p in v)
    
    print("transform test")
    f_idd = test["id"]
    i=0
    v = []
    n_i = len(f_idd)
    for i in range(n_i):
        v.append((i,n_i,fold + "/data/ProcessData/ST2_test/"+f_idd[i]+".csv",fold + "/data/ProcessData/ST2_umap_test/"+f_idd[i]+".csv"))
    
    Parallel(n_jobs=n_thread,verbose=0)(delayed(cal_st2)(p) for p in v)


def change_lines():
    #ST1
    print("ST1")
    i=0
    fold_train = fold+"/data/ProcessData/ST1_umap_train/"   
    fold_test = fold+"/data/ProcessData/ST1_umap_test/"
    f_total = os.listdir(fold_train)
    for f in f_total:
        print(i)
        i+=1
        file = open(fold_train+f,"r")
        lines = file.readlines()
        file.close()
        lines.insert(0, lines.pop(len(lines)-1))
        file = open(fold_train+f,"w")
        file.writelines(lines)
        file.close()
    f_total = os.listdir(fold_test)
    i-0
    for f in f_total:
        print(i)
        i+=1
        file = open(fold_test+f,"r")
        lines = file.readlines()
        file.close()
        lines.insert(0, lines.pop(len(lines)-1))
        file = open(fold_test+f,"w")
        file.writelines(lines)
        file.close()
    #st2
    print("ST2")
    fold_train = fold+"/data/ProcessData/ST2_umap_train/"   
    fold_test = fold+"/data/ProcessData/ST2_umap_test/"
    f_total = os.listdir(fold_train)
    i=0
    for f in f_total:
        print(i)
        i+=1
        file = open(fold_train+f,"r")
        lines = file.readlines()
        file.close()
        lines.insert(0, lines.pop(len(lines)-1))
        file = open(fold_train+f,"w")
        file.writelines(lines)
        file.close()
    f_total = os.listdir(fold_test)
    i=0
    for f in f_total:
        print(i)
        i+=1
        file = open(fold_test+f,"r")
        lines = file.readlines()
        file.close()
        lines.insert(0, lines.pop(len(lines)-1))
        file = open(fold_test+f,"w")
        file.writelines(lines)
        file.close()


## First only balance dataset based on positive and control cases. 

# def generate_sample():
#     files = os.listdir(fold+"/data/ProcessData/ST1_umap_train/") 
#     files = [l.split(".")[0] for l in files ]
#     table = pd.read_csv(fold +"/data/"+"ST1_train.csv",sep="\t",header=None,names=("id","batch","outcome"))
#     u_batch = pd.unique(table["batch"])
#     i = 1
#     print("ST1")
#     for b in u_batch:
#         #print("positive")
#         #print(str(i)+ " of " +str(len(u_batch)))
#         #i = i+1
#         lf_pos = list(table.loc[(table["batch"]==b) & (table["outcome"]==1),"id"])
#         lf_pos = [l.upper() for l in lf_pos]
        
#         for f in lf_pos:
#             if not (f in files):
#                 print(f)
#                 lf_pos.remove(f)
#         ###
#         num_pos = len(lf_pos)
#         ###
        
#         lf_neg = list(table.loc[(table["batch"]==b) & (table["outcome"]==0),"id"])
#         lf_neg = [l.upper() for l in lf_neg]
#         for f in lf_neg:
#             if not (f in files):
#                 print(f)
#                 lf_pos.remove(f)

#         ###
#         num_neg = len(lf_neg)
#         ###

#         diff = abs(num_pos-num_neg)
        
#         if num_pos < num_neg:
#             print("Expanding positive cases." )   
#             for i in range(int(diff)):  
#                 print(i, "out of", diff)
#                 file1, file2 = random.sample(list(lf_pos), 2)
#                 ############### random.uniform(0,1) OR random.random()
#                 dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
#         elif num_pos > num_neg:
#             print("Expanding negative cases." ) 
#             for i in range(int(diff)):    
#                 print(i, "out of", diff)     
#                 file1, file2 = random.sample(list(lf_neg), 2)
#                 ############### random.uniform(0,1) OR random.random()
#                 dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
        
#         elif num_pos == num_neg:
#             print("Equal number of positive and control cases, no data augmentation will be performed.")

#         # for i in range(int(len(lf_pos)*factor)): 
#         #     print("file " + str(i) + " of " + str(len(lf_pos)*2))           
#         #     file1, file2 = random.sample(list(lf_pos), 2)
#         #     ############### random
#         #     dataRecombination(file1, file2, 0.7, outputDir=fold+"/data/ProcessData/ST1_umap_train/")
#         # #print("negative")
#         #print(str(i)+ " of " +str(len(u_batch)))
#         #i = i+1
#         # lf_pos = list(table.loc[(table["batch"]==b) & (table["outcome"]==0),"id"])
#         # lf_pos = [l.upper() for l in lf_pos]
#         # for f in lf_pos:
#         #     if not (f in files):
#         #         print(f)
#         #         lf_pos.remove(f)
#         # for i in range(int(len(lf_pos)*factor)): 
#         #     print("file " + str(i) + " of " + str(len(lf_pos)*2))           
#         #     file1, file2 = random.sample(list(lf_pos), 2)
#         #     ############### random
#         #     dataRecombination(file1, file2, 0.7, outputDir=fold+"/data/ProcessData/ST1_umap_train/")
    
#     files = os.listdir(fold+"/data/ProcessData/ST2_umap_train/") 
#     files = [l.split(".")[0] for l in files ]
#     table = pd.read_csv(fold +"/data/"+"ST2_train.csv",sep="\t",header=None,names=("id","batch","outcome"))
#     u_batch = pd.unique(table["batch"])
#     i = 1
#     print("ST2")
#     for b in u_batch:
#         #print("positive")
#         #print(str(i)+ " of " +str(len(u_batch)))
#         #i = i+1
#         lf_pos = list(table.loc[(table["batch"]==b) & (table["outcome"]==1),"id"])
#         lf_pos = [l.upper() for l in lf_pos]
#         for f in lf_pos:
#             if not (f in files):
#                 print(f)
#                 lf_pos.remove(f)
#         ###
#         num_pos = len(lf_pos)
#         ###
#         lf_neg = list(table.loc[(table["batch"]==b) & (table["outcome"]==0),"id"])
#         lf_neg = [l.upper() for l in lf_neg]
#         for f in lf_neg:
#             if not (f in files):
#                 print(f)
#                 lf_pos.remove(f)
#         ###
#         num_neg = len(lf_neg)
#         ###
#         diff = abs(num_pos-num_neg)
#         ### Expand the smaller dataset 
#         if num_pos < num_neg:
#             for i in range(int(diff)):  
#                 print("Expanding positive cases" )      
#                 file1, file2 = random.sample(list(lf_pos), 2)
#                 ############### random.uniform(0,1) OR random.random()
#                 dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
#         elif num_pos > num_neg:
#             for i in range(int(diff)):    
#                 print("Expanding negative cases" )          
#                 file1, file2 = random.sample(list(lf_neg), 2)
#                 ############### random.uniform(0,1) OR random.random()
#                 dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
        
#         elif num_pos == num_neg:
#             print("Equal number of positive and control cases, no data augmentation will be performed.")
        
        
            
#     print("")
###########################################################################################################################
    
## Balance dataset based on positive and control cases and expand both by a given factor. 

def generate_expansion_and_balance(factor):
    files = os.listdir(fold+"/data/ProcessData/ST1_umap_train/") 
    files = [l.split(".")[0] for l in files ]
    table = pd.read_csv(fold +"/data/"+"ST1_train.csv",sep="\t",header=None,names=("id","batch","outcome"))
    u_batch = pd.unique(table["batch"])
    i = 1
    print("ST1")
    for b in u_batch:
        lf_pos = list(table.loc[(table["batch"]==b) & (table["outcome"]==1),"id"])
        lf_pos = [l.upper() for l in lf_pos]
        
        for f in lf_pos:
            if not (f in files):
                print(f)
                lf_pos.remove(f)
        ###
        num_pos = len(lf_pos)
        ###
        lf_neg = list(table.loc[(table["batch"]==b) & (table["outcome"]==0),"id"])
        lf_neg = [l.upper() for l in lf_neg]
        for f in lf_neg:
            if not (f in files):
                print(f)
                lf_pos.remove(f)

        ###
        num_neg = len(lf_neg)
        ###
        diff = abs(num_pos-num_neg)
        
        ### Balancing the data set (positive and control cases)
        if num_pos < num_neg:
            print("Expanding positive cases." )   
            for i in range(int(diff)):  
                print(i, "out of", diff)
                file1, file2 = random.sample(list(lf_pos), 2)
                ############### random.uniform(0,1) OR random.random()
                dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
            
        elif num_pos > num_neg:
            print("Expanding control cases." ) 
            for i in range(int(diff)):    
                print(i, "out of", diff)     
                file1, file2 = random.sample(list(lf_neg), 2)
                ############### random.uniform(0,1) OR random.random()
                dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
        
        elif num_pos == num_neg:
            print("Equal number of positive and control cases, no data augmentation will be performed.")

        ### Expanding the data set (positive and control cases)
        #while isinstance(num_pos*factor, int) is False:
        while num_pos <= num_pos*factor:
            ## Create 1 new negative file 
            file1, file2 = random.sample(list(lf_neg), 2)
            ############### random.uniform(0,1) OR random.random()
            dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")

            ## Create 1 new positive file 
            file1, file2 = random.sample(list(lf_pos), 2)
            ############### random.uniform(0,1) OR random.random()
            dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")

            num_pos += 1
            print("Increased to", num_pos, "cases.")
        print("There are", num_pos, "positive and negative cases.")

    files = os.listdir(fold+"/data/ProcessData/ST2_umap_train/") 
    files = [l.split(".")[0] for l in files ]
    table = pd.read_csv(fold +"/data/"+"ST2_train.csv",sep="\t",header=None,names=("id","batch","outcome"))
    u_batch = pd.unique(table["batch"])
    i = 1
    print("ST2")
    for b in u_batch:
        lf_pos = list(table.loc[(table["batch"]==b) & (table["outcome"]==1),"id"])
        lf_pos = [l.upper() for l in lf_pos]
        
        for f in lf_pos:
            if not (f in files):
                print(f)
                lf_pos.remove(f)
        ###
        num_pos = len(lf_pos)
        ###
        lf_neg = list(table.loc[(table["batch"]==b) & (table["outcome"]==0),"id"])
        lf_neg = [l.upper() for l in lf_neg]
        for f in lf_neg:
            if not (f in files):
                print(f)
                lf_pos.remove(f)

        ###
        num_neg = len(lf_neg)
        ###
        diff = abs(num_pos-num_neg)
        
        ### Balancing the data set (positive and control cases)
        if num_pos < num_neg:
            print("Expanding positive cases." )   
            for i in range(int(diff)):  
                print(i, "out of", diff)
                file1, file2 = random.sample(list(lf_pos), 2)
                ############### random.uniform(0,1) OR random.random()
                dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
            
        elif num_pos > num_neg:
            print("Expanding control cases." ) 
            for i in range(int(diff)):    
                print(i, "out of", diff)     
                file1, file2 = random.sample(list(lf_neg), 2)
                ############### random.uniform(0,1) OR random.random()
                dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")
        
        elif num_pos == num_neg:
            print("Equal number of positive and control cases, no data augmentation will be performed.")

        ### Expanding the data set (positive and control cases)
        #while isinstance(num_pos*factor, int) is False:
        while num_pos <= num_pos*factor:
            ## Create 1 new negative file 
            file1, file2 = random.sample(list(lf_neg), 2)
            ############### random.uniform(0,1) OR random.random()
            dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")

            ## Create 1 new positive file 
            file1, file2 = random.sample(list(lf_pos), 2)
            ############### random.uniform(0,1) OR random.random()
            dataRecombination(file1, file2, perc1=random.uniform(0,1), outputDir=fold+"/data/ProcessData/ST1_umap_train/")

            num_pos += 1
            print("Increased to", num_pos, "cases.")
        print("There are", num_pos, "positive and negative cases.")           
    print("")
###########################################################################################################################
   
#split_data()
#batch_normalization()
umap_apply()
#change_lines()
#generate_expansion_and_balance(2)
#subprocess.Popen(["Grid_file.exe","C:\\repos\\flow_C\\data\\ProcessData\\ST2_umap_train\\","C:\\repos\\flow_C\\data\\ProcessData\\ST2_umap_space.csv","C:\\repos\\flow_C\\data\\ProcessData\\ST2_grid_train\\","200","15"]).wait()
print("end")
