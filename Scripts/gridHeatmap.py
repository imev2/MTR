import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
fold = os.getcwd()
print(fold)

# Define the input and output directories
input_directory = fold+'/../data' # Specify the directory containing the CSV files
output_directory = fold+'/../heatmaps'  # Specify the directory to save the heatmaps

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get a list of all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_directory, '*.csv'))

i=0
# Iterate over each CSV file
for file_path in csv_files:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\t')
    
    # Create a heatmap plot using seaborn
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.heatmap(df, cmap='Blues', vmin=0, vmax=1)  # Adjust the colormap as needed
    
    # Get the filename without the directory path or extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save the heatmap plot as an image in the output directory
    output_path = os.path.join(output_directory, f'{file_name}_heatmap.png')
    plt.savefig(output_path)
    
    # Close the plot to free up memory
    plt.close()

print('Heatmap generation complete!')
