# Logistic Regression

# Objective 1 - Perform Logistic Regression with and without preproccessing and compare the results.
# Objective 2 - Compute sensitivity, specificity and AUC.  draw ROC curves  
# Objective 3 - Perform various data cleaning and  normalization operations and compare results

fileURL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
download.file(fileURL, destfile="breast-cancer-wisconsin.data", method="curl")
# read the data
df <- read.table("breast-cancer-wisconsin.data", na.strings = "?", sep=",")
str(df)
# Name the columns. 
# These names are displayed in the tree to facilitate semantic interpretation

library(DataExplorer)
library(ggplot2)
library(data.table)

# Remove unique column
df <- df [ ,-1]  

# Convert to factor dataset
df$V11 <- factor(df$V11, levels=c(2,4), labels=c("0", "1"))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)

# Note that this is Startified sampling method
split = sample.split(df$V11, SplitRatio = 0.7)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Checking Class distribution
table(df$V11)
prop.table(table(df$V11))
prop.table(table(training_set$V11))
prop.table(table(test_set$V11))

# Building classifier
classifier = glm(V11 ~.,
                 training_set,
                 family = binomial)
summary(classifier)

# Predicting the Training set results
pred_prob_training <- predict(classifier, type = 'response', training_set[ ,-10] )
pred_prob_training
pred_class_training = ifelse(pred_prob_training > 0.5, 1, 0)
pred_class_training
cbind(pred_prob_training, pred_class_training)
cm_training = table(training_set$V11, pred_class_training)
cm_training

# Evaluation metrics using formula
# accuracy = (cm[1,1] + cm[2,2])/ (cm[1,1] + cm [1,2] + cm [2,1] +cm [2,2])
# accuracy

accuracy_training <- sum(diag(cm_training))/sum(cm_training)


# Predicting the Test set results
pred_prob_test <- predict(classifier, type = 'response', test_set[ ,-10] )
pred_prob_test
pred_class_test = ifelse(prob_pred_test > 0.5, 1, 0)
pred_class_test
cm_test = table(test_set$V11, pred_class_test)
cm_test

# Evaluation metrics using formula
# accuracy = (cm[1,1] + cm[2,2])/ (cm[1,1] + cm [1,2] + cm [2,1] +cm [2,2])
# accuracy

accuracy_test <- sum(diag(cm_test))/sum(cm_test)

# Using formulae compute all other evaluation metrics

# ROC curve on test set
# install.packages("ROCR")
library(ROCR)
# install.packages("gplots")

# To draw ROC we need to predict the prob values. 

pred = prediction(pred_prob_test, test_set$V11)
perf = performance(pred, "tpr", "fpr")
pred
perf
plot(perf, colorize = T)
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

# Area Under Curve
auc <- as.numeric(performance(pred, "auc")@y.values)
auc <-  round(auc, 3)
auc




# Task 1: Closely examine cm on training and test sets and comment on model fitness

# Task 2: Variation 2 - Run the Expt by imputing the missing values. comment on this experiment

# Task 3: Varaition 3 - Run the experiment after scaling the values

# Task 4: Compare results of all experiments. You may do this after appropriate tabulation to summarise the results

