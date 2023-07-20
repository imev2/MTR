#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 20:52:12 2023

@author: listonlab
"""

import os
import pandas as pd

# Get the current working directory
cwd = os.getcwd()
path = "/ST1_transformed"
# List all files in the directory
files_list = [file for file in os.listdir(cwd+path)]

# Initialize an empty DataFrame for results
cell_counts = pd.DataFrame(columns=["batch","fileName", "rowCount"])

# Loop over each file
for file in files_list:
    # Full file path
    
    sample_list = [s for s in os.listdir(cwd+path+"/"+file)]
    for s in sample_list:
        file_path = cwd+path+"/"+file+"/"+s
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Get the number of rows
        row_count = len(lines) - 1
    
        # Patient ID
        idd = s.replace(".csv", "")
    
        # Add the file name and row count to the DataFrame
        cell_counts = cell_counts.append({"batch":file, "fileName": idd, "rowCount": row_count}, ignore_index=True)


# Save the DataFrame to a CSV file
dataset = pd.DataFrame({"Batch":cell_counts["batch"], "Sample": cell_counts["fileName"], "Number of cells (live)": cell_counts["rowCount"]})
dataset.to_csv("./ST1Counts.csv", index=False)
