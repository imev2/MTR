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
outcome <- csv_unbalanced[,ncol(csv_unbalanced)]
# colnames(csv_unbalanced)[ncol(csv_unbalanced)] <- "Phenotype"
csv_unbalanced <- csv_unbalanced[,1:ncol(csv_unbalanced)-1]
colnames(csv_unbalanced) <- marker_names
repetitions = nrow(csv_unbalanced)/length(sample_names)
samples <- rep(sample_names, each=repetitions)
batches <- rep(batch_names, each=repetitions)
csv_pooled <- cbind(csv_unbalanced, batches, samples, outcome)

#### Data - not standardized ####
## Fit to data ##
# Do you use outcome or batch at first level?
# Remove random effect?

hist(csv_pooled$CD4)

basic.lm <- lm(CD4 ~ batches, data = csv_pooled) # Data from within each batch is more similar to each other than from different batches 
summary(basic.lm)

(prelim_plot <- ggplot(csv_pooled, aes(x = batches, y = CD4)) +
    geom_point() +
    geom_smooth(method = "lm"))
plot(basic.lm, which=1)
plot(basic.lm, which=2)

basic.lm <- lm(CD4 ~ samples, data = csv_pooled) # Data from within each SAMPLE is more similar to each other than from different batches 
summary(basic.lm)

boxplot(CD4 ~ batches, data = csv_pooled)

(color_plot <- ggplot(csv_pooled, aes(x=batches, y=CD4, color = batches)) +
    geom_point(size=2) +
    theme_classic() +
    theme(legend.position="none"))

(prelim_plot <- ggplot(csv_pooled, aes(x = samples, y = CD4)) +
    geom_point() +
    geom_smooth(method = "lm"))
plot(basic.lm, which=1)
plot(basic.lm, which=2)

# csv_pooled$CD4_scaled <- scale(csv_pooled$CD4, center = TRUE, scale = TRUE)
# 
# basic.lm.scaled <- lm(CD4_scaled ~ batches, data = csv_pooled)
# summary(basic.lm.scaled)
# 
# (prelim_plot <- ggplot(csv_pooled, aes(x = batches, y = CD4_scaled)) +
#     geom_point() +
#     geom_smooth(method = "lm"))
# plot(basic.lm, which=1)
# plot(basic.lm, which=2)

(color_plot <- ggplot(csv_pooled, aes(x=samples, y=CD4, color = batches)) +
  geom_point(size=2) +
  theme_classic() +
  theme(legend.position="none"))

(split_plot <- ggplot(aes(samples, CD4), data=csv_pooled)+
    geom_point() +
    facet_wrap(~ batches) +
    xlab("batch") +
    ylab ("CD4"))


# basic.lm.outcome <- lm(CD4 ~ outcome, data = csv_pooled)
# summary(basic.lm.outcome)

## Fit to individual data ##

#### PCA ####

#### Standardize by batch data ####

##### Mixed Effects Models ##### Is there an association between CD4/14 and samples, after controlling for batch. 
library(lme4)
mixed.lmer <- lmer(samples ~ CD4 + (1|batches), data = csv_pooled)
summary(mixed.lmer)

#####
mixed.lmer <- lmer(samples ~ CD14 + (1|batches), data = csv_pooled)
summary(mixed.lmer)



