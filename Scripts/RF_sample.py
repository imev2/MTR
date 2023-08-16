# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:37:56 2023

@author: rafae
"""

from Data import Data
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import plotly.express as px
import plotly.io as pio
import pickle as pk

import numpy as np
seed = 12357
fold = os.getcwd()
data = Data()
n_jobs = 15
pio.renderers.default = 'browser'
#st3
#data.load(fold +"/data/ST3/ST3_base")
#data.split_data_test(fold_train=fold +"/data/ST3/ST3_train_po" , fold_test=fold +"/data/ST3/ST3_test_po",perc_train=0.80)

#data.load(fold +"/data/ST3/ST3_train_po")
#df,y = data.get_poll_cells(fold=fold +"/data/ST3/", filename="ST3_train.csv", balanciate=True,num_cells=1000, save=True,sam_bach=True)

#data.load(fold +"/data/ST3/ST3_test_po")
#df,y = data.get_poll_cells(fold=fold +"/data/ST3/", filename="ST3_test.csv", balanciate=True,num_cells=1000, save=True,sam_bach=True)
train = pd.read_csv(fold +"/data/ST3/ST3_train.csv",index_col=0)
train.index = train.iloc[:,-2]
y = train.iloc[:,-1]
train = train.iloc[:,:-2]
train["y"] = y
test = pd.read_csv(fold +"/data/ST3/ST3_test.csv",index_col=0)
test.index = test.iloc[:,-2]
y = test.iloc[:,-1]
test = test.iloc[:,:-2]
test["y"] = y
#cv

def cv_sample(train,cv=3):
    cv=3
    train
    sample = train.index
    uni = list(np.unique(list(sample)))
    np.random.shuffle(uni)
    a = []
    for i in range(cv):
        a.append([])
    i=0
    for s in uni:
        if i==cv:
            i=0
        a[i].append(s)
        i=i+1
    b = []
    data_x = []
    data_y = []
    for i in range(cv):
        data_x.append([])
        data_y.append([])
    for i in range(cv):
        b.append([])
    for s in uni:
        for i in range(cv):
            if not (s in a[i]):
                b[i].append(s)               
    for i in range(cv):
        l = 0
        data_x[i].append(train.loc[train.index==b[i][l]].iloc[:,:-1])
        data_y[i].append(train.loc[train.index==b[i][l]].iloc[:,-1])  
        for l in range(1,len(b[i])):
            data_x[i][0] = pd.concat((data_x[i][0],train.loc[train.index==b[i][l]].iloc[:,:-1]))
            data_y[i][0] = pd.concat((data_y[i][0],train.loc[train.index==b[i][l]].iloc[:,-1]))
        l = 0
        data_x[i].append(train.loc[train.index==a[i][l]].iloc[:,:-1])
        data_y[i].append(train.loc[train.index==a[i][l]].iloc[:,-1])  
        for l in range(1,len(a[i])):
            data_x[i][1] = pd.concat((data_x[i][1],train.loc[train.index==a[i][l]].iloc[:,:-1]))
            data_y[i][1] = pd.concat((data_y[i][1],train.loc[train.index==a[i][l]].iloc[:,-1]))
    return data_x,data_y
####################################################################################
def get_mean(data,y_pred):
    y_prob = []
    y_ = []
    df = data.copy()
    df["y_pred"] = [a[1] for a in y_pred]
    uni = np.unique(df.index)
    for s in uni:
        aux = df.loc[df.index==s]
        y_.append(aux.iloc[0,-2])
        y_prob.append(np.mean(aux.iloc[:,-1]))
    df = pd.DataFrame({"y_true":y_,"y_pred":y_prob},index=uni)
    return df




def cutof(true_y,pred_y,p_max,p_min,num):
    a = (p_max+p_min)/2
    y_hat = [(0 if v<a else 1) for v in pred_y]
    fnp = 0
    fpn = 0
    for i in range(len(true_y)):
        if true_y[i] == 0:
            if y_hat[i]==1:
                fnp+=1
        else:
            if y_hat[i]==0:
                fpn+=1
    print(str(fnp) + " " + str(fpn))
    if num==0:
        return 1-(fnp+fpn)/len(true_y) , fnp,fpn, (a,p_min,p_max)
    if fnp==fpn:
        return 1-(fnp+fpn)/len(true_y) , fnp,fpn, (a,p_min,p_max)
    
    if fpn>fnp:
        print("<-- " + str((p_min,a)))
        return cutof(true_y, pred_y, a, p_min, num-1)
    else:
        print("--> " + str((a,p_max)))
        return cutof(true_y, pred_y, p_max, a, num-1)

















####################################################################################
## LR
# n_jobs = 16
# cv=3
# data_x,data_y = cv_sample(train,cv)
# v = 1
# reg = []
# for a in range(100):
#     reg.append(v)
#     v = v-v*1/10
# v = []
# for i in range(cv):
#     for r in reg:
#         v.append((i,data_x[i][0],data_y[i][0],data_x[i][1],data_y[i][1],r))

# def mult(v):
#     i,x_train,y_train,x_test,y_test,reg = v
#     model = LogisticRegression(C=reg)
#     model.fit(x_train,y_train)
#     y_hat = model.predict(x_test)
#     acu = accuracy_score(y_test,y_hat)
#     return acu
# print("start")
# res = Parallel(n_jobs=n_jobs,verbose=10)(delayed(mult)(p) for p in v)
# print("stop")
# cv=[]
# reg = []
# ac = []
# for i in range(len(res)):
#     cv.append(v[i][0])
#     reg.append(v[i][5])
#     ac.append(res[i])
# df = pd.DataFrame({"cv":cv,"reg":reg,"acuracy":ac})

# uni = list(np.unique(list(df["reg"])))
# acu = []
# for r in uni:
#     acu.append(np.mean(df.loc[df["reg"]==r,"acuracy"]))
# df = pd.DataFrame({"reg":uni,"acuracy":acu})

# fig = px.line(data_frame=df,x="reg",y="acuracy",log_x=True)
# fig.show()
# best = 0.6
#####################################################################
# model = LogisticRegression(C=0.6)
# model.fit(train.iloc[:,:-1],train.iloc[:,-1])
# y_hat = model.predict_proba(train.iloc[:,:-1])

# df = get_mean(train, y_hat)


# res = cutof(df["y_true"],df["y_pred"],1,0,15)
# y_hat = model.predict_proba(test.iloc[:,:-1])

# df["y_hat"] = [(0 if a<res[3][0] else 1) for a in df["y_pred"]] 
# df.to_csv("logistic_regression_sample.csv")

##############################################################################

   
##RF
##best par max_depth = 10 ; max_feature = log2
model = RandomForestClassifier(n_estimators=1001,max_depth = 10,max_features="log2",verbose=10,n_jobs=n_jobs)
model.fit(train.iloc[:,:-1],train.iloc[:,-1])
y_hat = model.predict_proba(train.iloc[:,:-1])

df = get_mean(train, y_hat)


res = cutof(df["y_true"],df["y_pred"],1,0,15)
y_hat = model.predict_proba(test.iloc[:,:-1])

df["y_hat"] = [(0 if a<res[3][0] else 1) for a in df["y_pred"]] 
df.to_csv("Random_forest_sample.csv")

