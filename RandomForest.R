library(RCurl)
# Dataset
fileURL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
download.file(fileURL, destfile="breast-cancer-wisconsin.data", method="libcurl")
# read the data
df <- read.table("breast-cancer-wisconsin.data", na.strings = "?", sep=",")
str(df)
# Name the columns. 
# These names are displayed in the tree to facilitate semantic interpretation

df <- df [ ,-1]
df$V11 <- factor(df$V11, levels=c(2,4), labels=c("1", "2"))

# Removing columns with missing Values
sum (is.na(df))
colSums(sapply(df,is.na))
df$V7 <- NULL

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(df$V11, SplitRatio = 0.7)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)
str(training_set)

# Model building
library(randomForest)
set.seed(345)
rf <- randomForest(V11~.,data = training_set )
print(rf)
attributes(rf)
p1 <- predict(rf, training_set)
p1
cm_train <- table (p1, training_set$V11)
cm_train
train_accuracy = sum(diag(cm_train)/sum(cm_train))
train_accuracy

"""
Use the following code for regression
library(caret)
summary(p1)
RMSE(p1, training_set$V11) # Root mean squared error
MAE(y_pred, training_set$V11) # Mean Absolute Error
"""

p2 <- predict(rf, test_set)
cm_test <- table(p2, test_set$V11)
cm_test
test_accuracy = sum(diag(cm_test)/sum(cm_test))
test_accuracy

plot(rf)

# In the plot black solid line for overall OOB error and the colour lines, one for each class' error.
# Tuning mtry
library(caret)
str(training_set)
tuneRF(training_set[ ,-9], training_set$V11,
      stepFactor=0.5,
      plot = TRUE,
      ntreeTry = 400,
      trace = TRUE,
      improve = 0.05)

rf1 <- randomForest(V11~.,data = training_set,
                   ntreeTry = 400,
                   mtry=2,
                   importance = TRUE,
                   proximity = TRUE)
print(rf1)

p1 <- predict(rf1, training_set)
cm1 <- table(p1, training_set$V11)
cm1
p2 <- predict(rf1, test_set)
cm2 <- table(p2, test_set$V11)
cm2
# Number of nodes for trees
hist(treesize(rf),
     main = "No. of nodes for trees",
     col = "green")

varImpPlot(rf)
importance(rf)
varUsed(rf)


