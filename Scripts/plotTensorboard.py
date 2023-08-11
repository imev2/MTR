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

# tag = "Model_Cell_1_ST1_0408_1200_bs64_noaug_tested.csv"
# tag = "Model_Cell_2_ST1_0408_1200_bs64_noaug_tested.csv"
# tag = "Model_Cell_1_ST3_3007_1500_bs60_noaug.csv"
tag = "Model_Cell_2_ST3_0408_bs64_tested.csv"



# Read the CSV files into pandas DataFrames
val_loss_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Loss_validation-tag-" + tag)
train_loss_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Loss_train-tag-" + tag)
val_accuracy_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Accuracy_validation-tag-" + tag)
train_accuracy_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Accuracy_train-tag-" + tag)

# Merge the DataFrames based on the "Step" column
merged_df = val_loss_df.merge(train_loss_df, on='Step', suffixes=('_val_loss', '_train_loss'))
merged_df = merged_df.merge(val_accuracy_df, on='Step', suffixes=('_val_loss', '_val_accuracy'))
merged_df = merged_df.merge(train_accuracy_df, on='Step', suffixes=('_val_accuracy', '_train_accuracy'))

# Calculate moving averages
window_size = 5  # Adjust this value to control the degree of smoothing
merged_df['Smoothed_val_loss'] = merged_df['Value_val_loss'].rolling(window=window_size).mean()
merged_df['Smoothed_val_accuracy'] = merged_df['Value_val_accuracy'].rolling(window=window_size).mean()
merged_df['Smoothed_train_loss'] = merged_df['Value_train_loss'].rolling(window=window_size).mean()
merged_df['Smoothed_train_accuracy'] = merged_df['Value_train_accuracy'].rolling(window=window_size).mean()

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot for unaugmented data
axs[0].plot(merged_df['Step'], merged_df['Smoothed_val_loss'], label='Validation Loss', color='darkred', linestyle='solid')
axs[0].plot(merged_df['Step'], merged_df['Smoothed_val_accuracy'], label='Validation AUCROC', color='pink', linestyle='solid')
axs[0].plot(merged_df['Step'], merged_df['Smoothed_train_loss'], label='Training Loss', color='darkblue', linestyle='solid')
axs[0].plot(merged_df['Step'], merged_df['Smoothed_train_accuracy'], label='Training AUCROC', color='lightblue', linestyle='solid')
axs[0].set_ylim(0.35, 0.80)
axs[0].set_xlabel('Epoch', fontsize=14)
axs[0].set_ylabel('Value', fontsize=14)
axs[0].set_title("Unaugmented Data", fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=12)  # Increase tick label size
axs[0].xaxis.set_label_coords(0.5, -0.12)  # Adjust the position of the x-axis label

# GET MEAN VALUES 
# Step 1: Filter DataFrames to include only the last 100 steps
last_100_val_loss = val_loss_df.tail(100)
last_100_train_loss = train_loss_df.tail(100)
last_100_val_accuracy = val_accuracy_df.tail(100)
last_100_train_accuracy = train_accuracy_df.tail(100)

# Step 2: Extract the "Value" column
val_loss_values = last_100_val_loss['Value']
train_loss_values = last_100_train_loss['Value']
val_accuracy_values = last_100_val_accuracy['Value']
train_accuracy_values = last_100_train_accuracy['Value']

# Step 3: Compute the mean of the extracted "Value" columns
mean_val_loss = val_loss_values.mean()
mean_train_loss = train_loss_values.mean()
mean_val_accuracy = val_accuracy_values.mean()
mean_train_accuracy = train_accuracy_values.mean()

# Print the mean values
print(tag)
print("Mean Validation Loss:", mean_val_loss)
print("Mean Training Loss:", mean_train_loss)
print("Mean Validation Accuracy:", mean_val_accuracy)
print("Mean Training Accuracy:", mean_train_accuracy)

# tag = "Model_Cell_1_ST1_0608_bs64_tested.csv"
# tag = "Model_Cell_2_ST1_0608_bs64_tested.csv"
# tag = "Model_Cell_1_ST3_0408_bs64_tested.csv"
tag = "Model_Cell_2_ST3_3007_1500_bs60_noaug.csv"

# /data/ST2/ST2_tensorboard_models
# /data/ST1/ST1_tensorboard_models
# data/ST3/ST3_tensorboard_models

# Read the CSV files into pandas DataFrames
val_loss_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Loss_validation-tag-" + tag)
train_loss_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Loss_train-tag-" + tag)
val_accuracy_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Accuracy_validation-tag-" + tag)
train_accuracy_df = pd.read_csv(fold+"/data/ST3/ST3_tensorboard_models/Accuracy_train-tag-" + tag)

# Merge the DataFrames based on the "Step" column
merged_df = val_loss_df.merge(train_loss_df, on='Step', suffixes=('_val_loss', '_train_loss'))
merged_df = merged_df.merge(val_accuracy_df, on='Step', suffixes=('_val_loss', '_val_accuracy'))
merged_df = merged_df.merge(train_accuracy_df, on='Step', suffixes=('_val_accuracy', '_train_accuracy'))

# Calculate moving averages
window_size = 5  # Adjust this value to control the degree of smoothing
merged_df['Smoothed_val_loss'] = merged_df['Value_val_loss'].rolling(window=window_size).mean()
merged_df['Smoothed_val_accuracy'] = merged_df['Value_val_accuracy'].rolling(window=window_size).mean()
merged_df['Smoothed_train_loss'] = merged_df['Value_train_loss'].rolling(window=window_size).mean()
merged_df['Smoothed_train_accuracy'] = merged_df['Value_train_accuracy'].rolling(window=window_size).mean()

# Plot for augmented data
axs[1].plot(merged_df['Step'], merged_df['Smoothed_val_loss'], label='Validation Loss', color='darkred', linestyle='solid')
axs[1].plot(merged_df['Step'], merged_df['Smoothed_val_accuracy'], label='Validation AUCROC', color='pink', linestyle='solid')
axs[1].plot(merged_df['Step'], merged_df['Smoothed_train_loss'], label='Training Loss', color='darkblue', linestyle='solid')
axs[1].plot(merged_df['Step'], merged_df['Smoothed_train_accuracy'], label='Training AUCROC', color='lightblue', linestyle='solid')
axs[1].set_ylim(0.35, 0.80)
axs[1].set_xlabel('Epoch', fontsize=14)
axs[1].set_title("Augmented Data", fontsize=14)
axs[1].tick_params(axis='y', which='major', labelsize=0)  # Increase tick label size
axs[1].tick_params(axis='x', which='major', labelsize=12)  # Increase tick label size
axs[1].xaxis.set_label_coords(0.5, -0.12)  # Adjust the position of the x-axis label

# Add a main title
plt.suptitle("CellCNN (ST1)", fontsize=18)

# Create a single legend from the first subplot and place it at the center of the figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=14)
plt.subplots_adjust(bottom=0.2)  # Increase bottom margin for the legend

# Adjust layout to make space for the legend
plt.tight_layout()

# Display the plot
plt.show()

# GET MEAN VALUES 
# Step 1: Filter DataFrames to include only the last 100 steps
last_100_val_loss = val_loss_df.tail(100)
last_100_train_loss = train_loss_df.tail(100)
last_100_val_accuracy = val_accuracy_df.tail(100)
last_100_train_accuracy = train_accuracy_df.tail(100)

# Step 2: Extract the "Value" column
val_loss_values = last_100_val_loss['Value']
train_loss_values = last_100_train_loss['Value']
val_accuracy_values = last_100_val_accuracy['Value']
train_accuracy_values = last_100_train_accuracy['Value']

# Step 3: Compute the mean of the extracted "Value" columns
mean_val_loss = val_loss_values.mean()
mean_train_loss = train_loss_values.mean()
mean_val_accuracy = val_accuracy_values.mean()
mean_train_accuracy = train_accuracy_values.mean()

# Print the mean values
print(tag)
print("Mean Validation Loss:", mean_val_loss)
print("Mean Training Loss:", mean_train_loss)
print("Mean Validation Accuracy:", mean_val_accuracy)
print("Mean Training Accuracy:", mean_train_accuracy)
