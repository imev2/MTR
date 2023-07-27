#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:30:29 2023

@author: listonlab
"""
import os
import pandas as pd
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

# from plotnine import ggplot, aes, geom_line, labs, theme_minimal
fold = os.getcwd()
fold

# Read the CSV files into pandas DataFrames
val_loss_df = pd.read_csv(fold+'/data/ST2/tensorboard/Loss_validation-tag-modelCVsimp_dense__24_07__0001bs16.csv')
train_loss_df = pd.read_csv(fold+'/data/ST2/tensorboard/Loss_train-tag-modelCVsimp_dense__24_07__0001bs16.csv')
val_accuracy_df = pd.read_csv(fold+'/data/ST2/tensorboard/Accuracy_validation-tag-modelCVsimp_dense__24_07__0001bs16.csv')
train_accuracy_df = pd.read_csv(fold+'/data/ST2/tensorboard/Accuracy_train-tag-modelCVsimp_dense__24_07__0001bs16.csv')

# Merge the DataFrames based on the "Step" column
merged_df = val_loss_df.merge(train_loss_df, on='Step', suffixes=('_val_loss', '_train_loss'))
merged_df = merged_df.merge(val_accuracy_df, on='Step', suffixes=('_val_loss', '_val_accuracy'))
merged_df = merged_df.merge(train_accuracy_df, on='Step', suffixes=('_val_accuracy', '_train_accuracy'))

# Plot using seaborn and matplotlib
plt.figure(figsize=(10, 6))

# Add the curves for each CSV file
plt.plot(merged_df['Step'], merged_df['Value_val_loss'], label='Validation Loss', color='darkred', linestyle='solid')
plt.plot(merged_df['Step'], merged_df['Value_val_accuracy'], label='Validation Accuracy', color='pink', linestyle='solid')
plt.plot(merged_df['Step'], merged_df['Value_train_loss'], label='Train Loss', color='darkblue', linestyle='solid')
plt.plot(merged_df['Step'], merged_df['Value_train_accuracy'], label='Train Accuracy', color='lightblue', linestyle='solid')

# Add labels and title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('UMAP-Grid')
plt.legend()

# Display the plot
plt.show()

