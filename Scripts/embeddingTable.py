## Script to read in the first line of a C++ embedding file to create an empty table 

# 0. IMPORT STATEMENTS
import pandas as pd
import numpy as np
import os

# 1. FUNCTIONS
def embeddingTable(path):

    # Read in the first line of a tab-delimited file 
    with open(path, 'r') as f:
        firstLine = f.readline().strip('\n')
    
    # Split dimensions
    params = firstLine.split('\t')

    # Create list of dimensions
    try:
        dimensions = list(map(int, params))
    except ValueError:
        print("Error: The first line of the file does not contain integer values separated by tabs.")
        return None

    # Generate a single 2D array with the first two dimensions
    singleArray = np.empty(shape=(dimensions[0], dimensions[1]))
    
    concatenatedArray = singleArray

    if len(dimensions)>2: #OR if is not (dimensions[2]==0)?
        for dimSize in range(1, dimensions[2]):
            concatenatedArray = np.concatenate([concatenatedArray, singleArray], axis=1)
    
    finalArray = concatenatedArray
    df = pd.DataFrame(finalArray)
    print(df.shape)

    # File path name
    pathName = os.path.splitext(path)[0]

    # Save to TSV
    df.to_csv(pathName+"_embedded.tsv", sep="\t")