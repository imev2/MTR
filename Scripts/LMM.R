csv_unbalanced <- read.csv("~/MPHIL/MTR/Scripts/data/pooled/ST1_base_LMM_10.csv")
csv_balanced <- read.csv("~/MPHIL/MTR/Scripts/data/pooled/ST1_base_LMM_10_balanced.csv")
# Get meta data 
base_folder = "~/MPHIL/MTR/Scripts/data/ST1_base"
meta_file <- file.path(base_folder, "meta.txt")

meta_names <- read.table(meta_file, header = FALSE, sep = " ", stringsAsFactors = TRUE, skip = 1, nrows = 1)
sample_names <- unlist(meta_names)
meta_batch <- read.table(meta_file, header = FALSE, sep = " ", stringsAsFactors = TRUE, skip = 3, nrows = 1)
batch_names <- unlist(meta_batch)

meta_markers <- read.table(meta_file, header = FALSE, sep = " ", stringsAsFactors = FALSE, skip = 2, nrows = 1)
marker_names <- unlist(meta_markers)

# Add meta data 
csv_unbalanced <- csv_unbalanced[,2:ncol(csv_unbalanced)]
# colnames(csv_unbalanced)[ncol(csv_unbalanced)] <- "Phenotype"
csv_unbalanced <- csv_unbalanced[,1:ncol(csv_unbalanced)-1]
colnames(csv_unbalanced) <- marker_names
repetitions = nrow(csv_unbalanced)/length(sample_names)
samples <- rep(sample_names, each=repetitions)
batches <- rep(batch_names, each=repetitions)
csv_pooled <- cbind(csv_unbalanced, batches, samples)

hist(csv_pooled$CD4)
# dragons$bodyLength2 <- scale(dragons$bodyLength, center = TRUE, scale = TRUE)

basic.lm <- lm(CD4 ~ batches, data = csv_pooled)
summary(basic.lm)

library(tidyverse)  # load the package containing both ggplot2 and dplyr

(prelim_plot <- ggplot(csv_pooled, aes(x = batches, y = CD4)) +
    geom_point() +
    geom_smooth(method = "lm"))
