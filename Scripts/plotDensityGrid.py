#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:53:07 2023

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fold = os.getcwd()
def plot_heatmap_for_marker(fold, csv_file_path, marker_name, meta_path, title):
    # Read the CSV file into a DataFrame, skipping the first row
    df = pd.read_csv(fold+csv_file_path, delimiter=" ", header=None)

    # Read the meta.txt file to get marker names
    with open(fold+meta_path, 'r') as file:
        lines = file.readlines()

    # Extract marker names from the third row
    if len(lines) >= 3:
        third_row = lines[2].strip()
        marker_names = third_row.split(' ')
    else:
        marker_names = []

    # Append "X" and "Y" to the marker names list
    marker_names.append("X")
    marker_names.append("Y")

    # Set column names for the DataFrame
    df.columns = marker_names

    # Select data for the specified marker

    selected_data = df[["X", "Y", marker_name]]

    # Plot the scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=selected_data, x="X", y="Y", hue=marker_name, palette="viridis", sizes=(10, 100))
    plt.title(title + f" ({marker_name})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title=marker_name)
    plt.show()

# Path to your CSV file
csv_file_path = '/data/ST2/UMAP/ST2_space_umap_points_noheader.csv'

# Color the points by marker expression, using meta.txt file for ST2
meta_path = '/data/ST2/UMAP/meta.txt'

# Call the function to plot a heatmap for a specific marker
marker_name = "CD4"  # Replace with the marker you want to plot
plot_heatmap_for_marker(fold=fold, csv_file_path=csv_file_path, marker_name="CD4", meta_path=meta_path, title="ST2 UMAP Embedding Space")

data = Data()
data.load(fold+"/data/ST2/UMAP/Cells_ST2_test_1")
data._get_data(0)[0].shape

