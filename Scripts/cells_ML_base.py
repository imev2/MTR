# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:00 2023

@author: nina working on RF
"""
import pickle as pk
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
#generate data ST1
seed = 1235711
fold = os.getcwd()
fold
v = []
files = []

# train_path = "/data/ST2/pooled/ST2_train_logscale_pool"
# lab = "logscaled"
# file = pd.read_csv(fold+train_path)
# file=file.iloc[:,1:]
# x_train= file.iloc[:, :-1]
# y_train = file.iloc[:, -1]

# val_path = "/data/ST2/pooled/ST2_val_logscale_pool"
# file = pd.read_csv(fold+val_path)
# file=file.iloc[:,1:]
# x_val= file.iloc[:, :-1]
# y_val = file.iloc[:, -1]

# files.append((x_train.copy(), x_val.copy(), y_train.copy(), y_val.copy(),lab))

# train_path = "/data/ST2/pooled/ST2_train_scale_pool"
# lab = "scaled"
# file = pd.read_csv(fold+train_path)
# file=file.iloc[:,1:]
# x_train= file.iloc[:, :-1]
# y_train = file.iloc[:, -1]

# val_path = "/data/ST2/pooled/ST2_val_scale_pool"
# file = pd.read_csv(fold+val_path)
# file=file.iloc[:,1:]
# x_val= file.iloc[:, :-1]
# y_val = file.iloc[:, -1]

# files.append((x_train.copy(), x_val.copy(), y_train.copy(), y_val.copy(),lab))


# v = []
# for data in files:
#     for model in ["RF","LR","SVM"]:
        
#         if model=="RF":
#             res_max_features = ["sqrt","log2"]
#             res_max_depth = [10,15,20,30]
#             for max_features in res_max_features:
#                 for max_depth in res_max_depth:
#                     par= {}
#                     par["max_features"]=max_features
#                     par["max_depth"]=max_depth
#                     v.append((data,model,par)) 
#         if model=="LR":    
#             c = 1
#             l_12 = ["l1", "l2"]
#             for a in range(20):
#                 par = {}
#                 par["c"]=c
#                 c = c-c*1/5
#                 for l in l_12:
#                     par['l']=l
#                     v.append((data,model,par)) 
#         # if model=="SVM":
#         #     # res_c = [0.8]
#         #     # res_gamma = [0.1]
#         #     res_c = [1,0.7,0.4,0.1]
#         #     #res_gamma = [1,0.5,0.1,0.01,0.001,0.0001]
#         #     res_kernel = ["linear",'rbf', 'sigmoid']
#         #     for c in res_c:
#         #         # for gamma in res_gamma:
#         #         for kernel in res_kernel:
#         #             par = {}
#         #             par["c"]=c
#         #             par["kernel"]=kernel
#         #             # par["gamma"]=gamma
#         #             v.append((data,model,par)) 
                    
# def calcu(v):
#     data,model,par = v
#     x_train, x_val, y_train, y_val, lab=data
#     if model=="RF":
#         rf = RandomForestClassifier(n_estimators=501,max_features=par["max_features"],max_depth=par["max_depth"],random_state=seed,oob_score=False)
#         rf.fit(x_train, y_train)
#         y_pred=rf.predict(x_val)
#         return balanced_accuracy_score(y_true=y_val,y_pred=y_pred)
#     if model=="LR":
#         lr = LogisticRegression(random_state=seed,C=par["c"],penalty=par['l'],solver="liblinear",max_iter=100)
#         lr.fit(x_train, y_train)
#         y_pred=lr.predict(x_val)
#         return balanced_accuracy_score(y_true=y_val,y_pred=y_pred)
#     # if model=="SVM":
#     #     svc = SVC(C=par["c"],kernel=par["kernel"],cache_size=3000)
#     #     svc.fit(x_train, y_train)
#     #     if svc.fit_status_==1:
#     #         print("not fit")
#     #         return 0
#     #     y_pred=svc.predict(x_val)
#     #     return balanced_accuracy_score(y_true=y_val,y_pred=y_pred)
#     return None
# print("start")
# res = Parallel(n_jobs=15,verbose=10)(delayed(calcu)(p) for p in v)
# print("stop")              

# output = {"v":v,"res":res}

# lab =[]
# par = []
# model = []
# for i in range(len(v)):
#     data,m,p = v[i]
#     lab.append(data[4])
#     par.append(p)
#     model.append(m)
# data = pd.DataFrame({"data":lab,"model":model,"par":par,"acuracy":res})

# data.to_csv(fold+"/data/ST2/pooled/ST2_RFLRFINAL_parameters.csv",index_label=False)

# ### SCALED ###
# train_path = "/data/ST2/pooled/ST2_train_scale_pool"
# lab = "scaled"
# file = pd.read_csv(fold+train_path)
# file=file.iloc[:,1:]
# x_train= file.iloc[:, :-1]
# y_train = file.iloc[:, -1]

# val_path = "/data/ST2/pooled/ST2_val_scale_pool"
# file = pd.read_csv(fold+val_path)
# file=file.iloc[:,1:]
# x_val= file.iloc[:, :-1]
# y_val = file.iloc[:, -1]

# x_train_val= x_train.append(x_val)
# y_train_val= y_train.append(y_val)

# test_path = "/data/ST2/pooled/ST2_test_scale_pool_unbalanced"
# file = pd.read_csv(fold+test_path)
# file=file.iloc[:,1:]
# x_test= file.iloc[:, :-1]
# y_test = file.iloc[:, -1]

# rf = RandomForestClassifier(n_estimators=2001,max_features="sqrt",max_depth=30,random_state=seed,oob_score=False,n_jobs=15, verbose=2)
# rf.fit(x_train, y_train)
# y_pred_rf=rf.predict(x_test)
# y_prob_rf=rf.predict_proba(x_test)
# print("RF Finished")
# lr = LogisticRegression(random_state=seed,C= 0.014411518807585589,penalty="l2",solver="liblinear", verbose=2)
# lr.fit(x_train, y_train)
# y_pred_lr=lr.predict(x_test)
# y_prob_lr=lr.predict_proba(x_test)
# print("LR Finished")
# a3 = roc_auc_score(y_test, y_prob_rf[:,0])
# b3 = roc_auc_score(y_test, y_prob_rf[:,0])
# comb = {"y_true":y_test,  "RF_y_pred":y_pred_rf, "RF_y_prob":y_prob_rf, "LR_y_pred":y_pred_lr, "LR_y_prob":y_prob_lr}
# file = open(fold+"/data/ST2/scaled_ST2_RFLR_2001.dat","wb")
# pk.dump(comb, file)
# file.close()

### SCALED ###
train_path = "/data/ST2/pooled/ST2_train_scale_pool"
lab = "scaled"
file = pd.read_csv(fold+train_path)
file=file.iloc[:,1:]
x_train= file.iloc[:, :-1]
y_train = file.iloc[:, -1]

val_path = "/data/ST2/pooled/ST2_val_scale_pool"
file = pd.read_csv(fold+val_path)
file=file.iloc[:,1:]
x_val= file.iloc[:, :-1]
y_val = file.iloc[:, -1]

x_train_val= x_train.append(x_val)
y_train_val= y_train.append(y_val)

test_path = "/data/ST2/pooled/ST2_test_scale_pool_unbalanced"
file = pd.read_csv(fold+test_path)
file=file.iloc[:,1:]
x_test= file.iloc[:, :-1]
y_test = file.iloc[:, -1]

rf = RandomForestClassifier(n_estimators=2001,max_features="log2",max_depth=30,random_state=seed,oob_score=False,n_jobs=15, verbose=2)
rf.fit(x_train, y_train)
y_pred_rf=rf.predict(x_test)
y_prob_rf=rf.predict_proba(x_test)
print("RF Finished")
lr = LogisticRegression(random_state=seed,C= 1,penalty="l2",solver="liblinear", verbose=2)
lr.fit(x_train, y_train)
y_pred_lr=lr.predict(x_test)
y_prob_lr=lr.predict_proba(x_test)
print("LR Finished")
comb = {"y_true":y_test,  "RF_y_pred":y_pred_rf, "RF_y_prob":y_prob_rf, "LR_y_pred":y_pred_lr, "LR_y_prob":y_prob_lr}
a2 = roc_auc_score(y_test, y_prob_rf[:,0])
b2 = roc_auc_score(y_test, y_prob_rf[:,0])
file = open(fold+"/data/ST2/scaled_ST2_RFLR_2001.dat","wb")
pk.dump(comb, file)
file.close()


### LOG SCALED ###
train_path = "/data/ST2/pooled/ST2_train_logscale_pool"
lab = "scaled"
file = pd.read_csv(fold+train_path)
file=file.iloc[:,1:]
x_train= file.iloc[:, :-1]
y_train = file.iloc[:, -1]

val_path = "/data/ST2/pooled/ST2_val_logscale_pool"
file = pd.read_csv(fold+val_path)
file=file.iloc[:,1:]
x_val= file.iloc[:, :-1]
y_val = file.iloc[:, -1]

x_train_val= x_train.append(x_val)
y_train_val= y_train.append(y_val)

test_path = "/data/ST2/pooled/ST2_test_logscale_pool_unbalanced"
file = pd.read_csv(fold+test_path)
file=file.iloc[:,1:]
x_test= file.iloc[:, :-1]
y_test = file.iloc[:, -1]

# rf = RandomForestClassifier(n_estimators=2001,max_features="log2",max_depth=10,random_state=seed,oob_score=False,n_jobs=15, verbose=2)
rf = RandomForestClassifier(n_estimators=2001,max_features="sqrt",max_depth=30,random_state=seed,oob_score=False,n_jobs=15, verbose=2)
rf.fit(x_train, y_train)
y_pred_rf=rf.predict(x_test)
y_prob_rf=rf.predict_proba(x_test)
print("RF Finished")
a1 = roc_auc_score(y_test, y_prob_rf[:,0])
b1 = roc_auc_score(y_test, y_prob_rf[:,0])

lr = LogisticRegression(random_state=seed,C= 1,penalty="l2",solver="liblinear", verbose=2)
lr.fit(x_train, y_train)
y_pred_lr=lr.predict(x_test)
y_prob_lr=lr.predict_proba(x_test)
print("LR Finished")
comb = {"y_true":y_test,  "RF_y_pred":y_pred_rf, "RF_y_prob":y_prob_rf, "LR_y_pred":y_pred_lr, "LR_y_prob":y_prob_lr}
file = open(fold+"/data/ST2/logscaled_ST2_RFLR_2001.dat","wb")
pk.dump(comb, file)
file.close()

# ### LOG SCALED ###
# train_path = "/data/ST2/pooled/ST2_train_logscale_pool"
# lab = "scaled"
# file = pd.read_csv(fold+train_path)
# file=file.iloc[:,1:]
# x_train= file.iloc[:, :-1]
# y_train = file.iloc[:, -1]

# val_path = "/data/ST2/pooled/ST2_val_logscale_pool"
# file = pd.read_csv(fold+val_path)
# file=file.iloc[:,1:]
# x_val= file.iloc[:, :-1]
# y_val = file.iloc[:, -1]

# x_train_val= x_train.append(x_val)
# y_train_val= y_train.append(y_val)

# test_path = "/data/ST2/pooled/ST2_test_logscale_pool_unbalanced"
# file = pd.read_csv(fold+test_path)
# file=file.iloc[:,1:]
# x_test= file.iloc[:, :-1]
# y_test = file.iloc[:, -1]

# # rf = RandomForestClassifier(n_estimators=2001,max_features="log2",max_depth=10,random_state=seed,oob_score=False,n_jobs=15, verbose=2)
# rf = RandomForestClassifier(n_estimators=2001,max_features="sqrt",max_depth=30,random_state=seed,oob_score=False,n_jobs=15, verbose=2)
# rf.fit(x_train, y_train)
# y_pred_rf=rf.predict(x_test)
# y_prob_rf=rf.predict_proba(x_test)
# print("RF Finished")
# a = roc_auc_score(y_test, y_prob_rf[:,0])
# b = roc_auc_score(y_test, y_prob_rf[:,0])
# lr = LogisticRegression(random_state=seed,C= 0.014411518807585589,penalty="l2",solver="liblinear", verbose=2)
# lr.fit(x_train, y_train)
# y_pred_lr=lr.predict(x_test)
# y_prob_lr=lr.predict_proba(x_test)
# print("LR Finished")
# comb = {"y_true":y_test,  "RF_y_pred":y_pred_rf, "RF_y_prob":y_prob_rf, "LR_y_pred":y_pred_lr, "LR_y_prob":y_prob_lr}
# file = open(fold+"/data/ST2/logscaled_ST2_RFLR_2001.dat","wb")
# pk.dump(comb, file)
# file.close()