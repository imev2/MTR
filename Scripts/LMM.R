csv_unbalanced <- read.csv("~/MPHIL/MTR/Scripts/data/pooled/ST1_base_LMM_10.csv")
csv_balanced <- read.csv("~/MPHIL/MTR/Scripts/data/pooled/ST1_base_LMM_10_balanced.csv")


# Function to concatenate rows for a given header column in all CSV files
concatenate_csv_headers_with_meta <- function(base_folder, header_name) {
  # Read meta.txt to get sample names and their corresponding batch names
  meta_file <- file.path(base_folder, "meta.txt")
  meta_names <- read.table(meta_file, header = FALSE, sep = " ", stringsAsFactors = FALSE, skip = 1, nrows = 1)
  sample_names <- unlist(meta_names)
  
  meta_batch <- read.table(meta_file, header = FALSE, sep = " ", stringsAsFactors = FALSE, skip = 3, nrows = 1)
  batch_names <- unlist(meta_batch)
  
  # Initialize an empty data frame to store the concatenated data
  concatenated_data <- data.frame()
  
  # Loop through each batch and read all CSV files
  for (i in seq_along(batch_names)) {
    # Get the batch name and sample names for this iteration
    batch_name <- batch_names[i]
    samples_in_batch <- unlist(strsplit(sample_names[i], ","))
    
    # Loop through each sample and read its corresponding CSV file
    for (sample_name in samples_in_batch) {
      # Construct the path to the CSV file for this sample
      csv_file <- file.path(base_folder, paste0(sample_name, ".dat"))
      
      # Read the CSV file
      csv_data <- read.csv(csv_file, stringsAsFactors = FALSE)
      
      # Extract the header column for the given sample
      header_column <- csv_data[[header_name]]
      
      # Create a data frame with batch name, sample name, and header column
      sample_data <- data.frame(Batch = batch_name, Sample = sample_name, Value = header_column)
      
      # Append the sample data to the concatenated data
      concatenated_data <- bind_rows(concatenated_data, sample_data)
    }
  }
  
  return(concatenated_data)
}

# Replace 'path/to/ST1_base' with your actual base folder path
base_folder <- "~/MPHIL/MTR/Scripts/data/ST1_base"
header_name <- "CD28"  # Replace 'HeaderName' with the actual column header you want to concatenate

ST1 <- concatenate_csv_headers(base_folder, header_name)

head(ST1)

hist(ST1$Value)

###### Without scaling ######

###### Without scaling ######
dragons$ValueSca <- scale(ST1$Value, center = TRUE, scale = TRUE)

