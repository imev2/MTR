#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:44:52 2023

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

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, auc, precision_recall_curve

def calculate_metrics(y_true, y_pred_scores, y_labels):
    balanced_acc = balanced_accuracy_score(y_true, y_labels)
    auroc = roc_auc_score(y_true, y_pred_scores)
    sensitivity = recall_score(y_true, y_labels, average='weighted')
    precision = precision_score(y_true, y_labels, average='weighted')
    f1 = f1_score(y_true, y_labels)
    f2 = 2 * (precision * sensitivity) / (precision + sensitivity)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_scores)
    auprc = auc(recall_curve, precision_curve)

    return balanced_acc, auroc, sensitivity, precision, f1, f2, auprc

fold = os.getcwd()

print("#### ST1 ####")
ST1 = '/data/ResultsOld/ST1/Model_Cell_2_ST1_0408_1200_bs64_noaug_tested.dat'

with open(fold+ST1, 'rb') as f: 
    a = torch.load(f)
    f.close()
print(a.keys())

# Extract true and predicted labels from the dictionary
ytest_true = a['ytest_true']
print(ytest_true)
ypred = a['ytest_pred']
# print(ypred)
ytest = torch.sigmoid(torch.as_tensor(a['ytest_pred']))
# print(ytest)
ytest_pred = torch.round(ytest)
# print(ytest_pred)
# Calculate metrics
print("#### ST1 ####")
balanced_acc, auroc, sensitivity, precision, f1, f2, auprc = calculate_metrics(ytest_true, ytest, ytest_pred)

print("Balanced Accuracy:", balanced_acc)
print("AUROC:", auroc)
print("Sensitivity:", sensitivity)
print("Precision:", precision)
print("F1 Score:", f1)
print("F2 Score:", f2)
print("AUPRC:", auprc)


ST2 = '/data/Results/ST2/Model_Cell_2_ST2_0408_bs64_tested_noaug.dat'
with open(fold+ST2, 'rb') as f: 
    b = torch.load(f)
    f.close()
print(b.keys())

print("#### ST2 ####")
# Extract true and predicted labels from the dictionary
ytest_true = b['ytest_true']
print(ytest_true)
ypred = b['ytest_pred']
# print(ypred)
ytest = torch.sigmoid(torch.as_tensor(b['ytest_pred']))
# print(ytest)
ytest_pred = torch.round(ytest)
# print(ytest_pred)
# Calculate metrics
balanced_acc, auroc, sensitivity, precision, f1, f2, auprc = calculate_metrics(ytest_true, ytest, ytest_pred)

print("Balanced Accuracy:", balanced_acc)
print("AUROC:", auroc)
print("Sensitivity:", sensitivity)
print("Precision:", precision)
print("F1 Score:", f1)
print("F2 Score:", f2)
print("AUPRC:", auprc)


ST3 = '/data/Results/ST3/Model_Cell_2_ST3_0508_bs64_tested_noaug.dat'
with open(fold+ST3, 'rb') as f: 
    c= torch.load(f)
    f.close()
print(c.keys())
print("#### ST3 ####")
# Extract true and predicted labels from the dictionary
ytest_true = c['ytest_true']
print(ytest_true)
ypred = c['ytest_pred']
# print(ypred)
ytest = torch.sigmoid(torch.as_tensor(c['ytest_pred']))
# print(ytest)
ytest_pred = torch.round(ytest)
# print(ytest_pred)
# Calculate metrics
balanced_acc, auroc, sensitivity, precision, f1, f2, auprc = calculate_metrics(ytest_true, ytest, ytest_pred)

print("Balanced Accuracy:", balanced_acc)
print("AUROC:", auroc)
print("Sensitivity:", sensitivity)
print("Precision:", precision)
print("F1 Score:", f1)
print("F2 Score:", f2)
print("AUPRC:", auprc)