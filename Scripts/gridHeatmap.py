from Data import Data,Log_transformer,Standard_tranformer, Oversample
import pickle as pk
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score, fbeta_score, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools
import matplotlib.pyplot as plt
import csv
import os

fold = os.getcwd()

#### PLOT ALL POINTS IN UMAP EMBEDDING SPACE #### 

# Path to your CSV file
csv_file_path = '/data/ST2/UMAP/ST2_space_umap_points_noheader.csv'

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv(fold+csv_file_path, delimiter=" ", header=None)

# Color the points by marker expression, using meta.txt file for ST2
meta_path = '/data/ST2/UMAP/meta.txt'

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

# Print the modified marker names list
print(marker_names)

df2 = df.set_axis(marker_names, axis=1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap_for_marker(csv_file_path, marker_name, meta_path, title):
    # Read the CSV file into a DataFrame, skipping the first row
    df = pd.read_csv(csv_file_path, delimiter=" ", header=None)

    # Read the meta.txt file to get marker names
    with open(meta_path, 'r') as file:
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
    selected_data = df[[marker_name, "X", "Y"]]

    # Create a pivot table for the heatmap
    pivot_table = selected_data.pivot_table(index="Y", columns="X", values=marker_name, aggfunc=np.mean)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, ccmap='Blues', vmin=0, vmax=1, annot=True, fmt=".2f")
    plt.title(title + f"({marker_name})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Path to your CSV file
csv_file_path = '/data/ST2/UMAP/ST2_space_umap_points.csv'

# Color the points by marker expression, using meta.txt file for ST2
meta_path = '/data/ST2/UMAP/meta.txt'

title = "ST2 UMAP Embedding"

# Call the function to plot a heatmap for a specific marker
selected_marker = "CD4"  # Replace with the marker you want to plot
plot_heatmap_for_marker(fold+csv_file_path, selected_marker, fold+meta_path)


# #### PLOT DENSITY FILE ####
# seed = 1235711
# data = Data(seed=seed)
# data.load(fold+"/data/ST2/UMAP/ST2_2D_test_dens")
# data._get_data(0)[0]


# # Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)

# # Get a list of all CSV files in the input directory
# csv_files = glob.glob(os.path.join(input_directory, '*.csv'))

# i=0
# # Iterate over each CSV file
# for file_path in csv_files:
#     # Load the CSV file into a pandas DataFrame
#     df = pd.read_csv(file_path, sep='\t')
    
#     # Create a heatmap plot using seaborn
#     plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
#     sns.heatmap(df, cmap='Blues', vmin=0, vmax=1)  # Adjust the colormap as needed
    
#     # Get the filename without the directory path or extension
#     file_name = os.path.splitext(os.path.basename(file_path))[0]
    
#     # Save the heatmap plot as an image in the output directory
#     output_path = os.path.join(output_directory, f'{file_name}_heatmap.png')
#     plt.savefig(output_path)
    
#     # Close the plot to free up memory
#     plt.close()

# print('Heatmap generation complete!')


data = Data()
data._get_data(0)[0]