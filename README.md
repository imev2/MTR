# Master's Thesis Research
## MPhil in Computational Biology with the Liston Laboratory 

### 0. Data availability
The flow cytometry dataset for this project was provided by J. Neumann, Dr. S. Humblet-Baron, and Dr. L. Yshii (KU Leuven). The participant population comprised 143 males and 158 females aged between 40-83 years old. Participants were further characterized by clinical measures including age, age at diagnosis, and Mini‐Mental State Examination (MMSE) score at the time of diagnosis.

Initial data processing was performed by Dr. O. Burton (University of Cambridge) using FlowJo™ Software (BD Life Sciences). Processing steps included the removal of artifacts, doublets, and dead cells and compensation correction by Autospill. 
### 1. Data pre-processing

#### 1.1 Statistical analyses (linear mixed modeling)

#### 1.2 Creating ML-friendly files

##### 1.2.1 Pre-processing
LMM.R
##### 1.2.2 Test/train/validation split
preprocess.py - Define percentages to split data for training, validation, and testing.
##### 1.2.3 Data transformations
main/mainST2/mainST3.py - Log transform, standard scaling, and data augmentation functions. Pools defined number of cells. 
### 2. Machine learning classifiers 

#### 2.1 Cell level models
cells_ML_base/cells_ML_base_ST2/cells_ML_base_ST3.py

#### 2.2 Sample-level models

### 3. UMAP+ML classifier pipeline
plotDensityGrid.py
Model_density
plotTensorboard.py - view Tensorboard results during training and validation
### 4. Cell-subset gating 
wekaCSV.py
