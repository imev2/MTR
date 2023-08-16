# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:26:31 2023

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
from sklearn.preprocessing import StandardScaler

import numpy as np
seed = 12357
fold = os.getcwd()
data = Data()
n_jobs = 15
pio.renderers.default = 'browser'
#st3
data.load(fold +"/data/ST3/ST3_base")
df,y = data.get_poll_cells(fold=fold +"/data/ST3/", filename="ST3_base.csv", balanciate=True,num_cells=1000, save=True,sam_bach=True)

train = pd.read_csv(fold +"/data/ST3/ST3_base.csv",index_col=0)
train.index = train.iloc[:,-2]
y = train.iloc[:,-1]
train = train.iloc[:,:-2]
stand = StandardScaler()
train = stand.fit_transform(train)

model = RandomForestClassifier(n_estimators=1001,max_depth = 10,max_features="log2",verbose=10)
model.fit(train,y)

#################################################
i=7 # AD003C
x = data._get_data(i)[0]

x = stand.transform(x)
y_hat = model.predict_proba(x)
y_hat = y_hat[:,1]
df = pd.DataFrame(x,columns=data.painel)
df["y_hat"] = y_hat
fig =px.histogram(data_frame=df,x="y_hat",title=data.id[i],labels=dict(y_hat="Probability"),color_discrete_sequence=['indianred'])#indianred
fig.update_xaxes(range=[0.3, 0.65])
fig.update_layout(autosize=False,width=800,height=500)
fig.show()
################################################
df["y_pred"] = [(0 if a<0.5 else 1) for a in df["y_hat"]]
df.to_csv("RF_cell.csv",index=False)

