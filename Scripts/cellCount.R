## Script to count cells per individual for a given panel directory 

library(tidyverse)

# Directory 
#dirPath <- "D:\toNina" #Ex. ST1, ST2, etc.  



# List all CSV files in the directory
filesList <- list.files(path = "~/Cambridge/Thesis/Data/Pre-processing", pattern = "*.csv")

# Initializes an empty data frame for results 
cellCounts <- data.frame(fileName = character(), rowCount = integer())
i = 1
len = rep(NA,length(filesList))
idd = rep(NA,length(filesList))
# Loop over each file
for(file in filesList){
  # Full file path
  #filePath <- file.path(dirP, file)
  print(i)
  
  # Read the file
  #data <- read.csv(paste0("./ST1/",file), header=TRUE)
  l = readLines(paste0("~/Cambridge/Thesis/Data/Pre-processing/",file))
  #text = readChar(text_file, file.info(text_file)$size)
  # Get the number of rows
  len[i] <- length(l)-1
  
  # Patient ID
  idd[i] <- sub(".csv", "", file)
  
  i= i+1  
  # Add the file name and row count to the dataframe
  #cellCounts <- rbind(cellCounts, data.frame(fileName = fileID, rowCount = numRows))
}
dataset = data.frame(ID = idd,Number = len)
#print(cellCounts)
#write.csv(cellCounts, file="~/Cambridge/Thesis/Data/ST1Counts.csv")
write.csv(dataset, file="./ST1Counts.csv",row.names = FALSE,quote = FALSE)