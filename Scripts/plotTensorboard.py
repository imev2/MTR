#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:30:29 2023

@author: listonlab
"""
import os
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal
fold = os.getcwd()
fold

# Read the CSV files into pandas DataFrames
val_loss_df = pd.read_csv(fold+'/data/ST1/tensorboard/CV1_scale/Loss_validation-tag-scale_modelCV1_2207_1730.csv')
train_loss_df = pd.read_csv(fold+'/data/ST1/tensorboard/CV1_scale/Loss_train-tag-scale_modelCV1_2207_1730.csv')
val_accuracy_df = pd.read_csv(fold+'/data/ST1/tensorboard/CV1_scale/Accuracy_validation-tag-scale_modelCV1_2207_1730.csv')
train_accuracy_df = pd.read_csv(fold+'/data/ST1/tensorboard/CV1_scale/Accuracy_train-tag-scale_modelCV1_2207_1730.csv')

# Merge the DataFrames based on the "Step" column
merged_df = val_loss_df.merge(train_loss_df, on='Step', suffixes=('_val_loss', '_train_loss'))
merged_df = merged_df.merge(val_accuracy_df, on='Step', suffixes=('_val_loss', '_val_accuracy'))
merged_df = merged_df.merge(train_accuracy_df, on='Step', suffixes=('_val_accuracy', '_train_accuracy'))

# Plot using plotnine
plot = ggplot(merged_df, aes(x='Step')) + theme_minimal()

# Add the curves for each CSV file
plot += geom_line(aes(y='Value_val_loss'), color='blue', size=1, linetype='solid')
plot += geom_line(aes(y='Value_train_loss'), color='red', size=1, linetype='dashed')
plot += geom_line(aes(y='Value_val_accuracy'), color='green', size=1, linetype='solid')
plot += geom_line(aes(y='Value_train_accuracy'), color='orange', size=1, linetype='dashed')

# Add labels and title
plot += labs(x='Step', y='Value', title='Change in Value at Each Step')

# Display the plot
print(plot)
