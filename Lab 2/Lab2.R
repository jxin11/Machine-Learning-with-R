# Read dataset
data <- read.csv('aug_train.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
# data <- read.csv(file.choose(), header = T)
# data <- read.table(url(""),sep=',')

# View dataset
View(data)

# View top n rows of dataset
head(data, 5)

# Get dimension of the dataset
dim(data)

# Install packages
install.packages('dplyr')

# Preview data type & sample data
library(dplyr)
glimpse(data)
str(data)

# Summary of data
summary(data)

# Select columns
select(data, c(1,2,3))

# Delete columns
data2 <- data[,-1]  # remove first columns
data$enrollee_id <- NULL  # set columns to null to remove
data2 <- select(data, -1)  # select columns, remove first col

# Change col name
names(data)[names(data) == "relevent_experience"] <- "exp"

# Add new col
data$newCol = 1
data$newCol <- NULL 

# Data explorer
install.packages('DataExplorer')
library(DataExplorer)
plot_str(data, fontSize=20)   # plot structure
plot_intro(data)   # plot the basic info
plot_histogram(data)
plot_histogram(data$training_hours)
plot_bar(data)
plot_density(data)
plot_boxplot(data, by='target')
barplot(table(data$target), main="Target Distribution", col=c("skyblue","red"))
install.packages('ggplot2')
library(ggplot2)
# ggplot(data, aes(target, fill= Approved))+ geom_bar() #Fill by category
plot_correlation(data,'continuous', cor_args = list("use" = "pairwise.complete.obs"))

# Plotting
plot(table(data$gender))
plot(data$gender)
# Plot a barchart for categorical var
barplot(table(data$gender))
table(data$gender)
hist(data$city_development_index)

# Count na
plot_missing(data)
colSums(is.na(data))
sum(is.na(data))
which (is.na(data$experience)) # Which are the rows with missing values in a column





# ###-------------------------------
# ### Example from lecturer
# 
# plot_str(df, fontSize=18)
# plot_histogram(df)
# plot_hist(df$Gender)
# plot_bar(df$Gender)
# plot_bar(df)
# plot_density(df)
# plot_boxplot(df, by= 'Diabetic' )
# plot_boxplot(df, 'Diabetic' )
# boxplot(df)
# plot_bar(df$Diabetic)
# 
# #####-----------
# # More plots
# barplot(table(df$Diabetic), main="Diabetic Distribution", col=c("skyblue","red", "lightgreen"))
# ggplot(mydata, aes(Ethnicity, fill= Approved))+ geom_bar() #Fill by category
# plot_correlation(ds,'continuous', cor_args = list("use" = "pairwise.complete.obs"))
# 
# #----------------------------------
# # Missing Data
# plot_missing(df)
# plot_missing(df$Pregnancies)
# sum(is.na(df)) # sum of missing values in df
# colSums(sapply(df,is.na)) # missing values by columns
# sum(is.na(df$Pregnancies)) # missing value in a column
# which (is.na(df$Pregnancies)) # Which are the rows with missing values in a column
# #-------------
# 
# # -------------------------------
