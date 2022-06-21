# SVM in R

library(ggplot2)
library(e1071)

# Dataset
fileURL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
download.file(fileURL, destfile="breast-cancer-wisconsin.data", method="libcurl")
# read the data
df <- read.table("breast-cancer-wisconsin.data", na.strings = "?", sep=",")
str(df)
table(df$V11) 
# Name the columns. 
# These names are displayed in the tree to facilitate semantic interpretation
df <- df [ ,-1]

# Removing columns with missing Values
sum (is.na(df))
colSums(sapply(df,is.na))
df$V7 <- NULL
colSums(sapply(df,is.na))
dim(df)

# labelling the target variable values
df$V11 <- factor(df$V11, levels=c(2,4), labels=c("1", "2"))
table(df$V11)

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(df$V11, SplitRatio = 0.7)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)
table(training_set$V11)
table(test_set$V11)

# https://stats.stackexchange.com/questions/237382/difference-between-the-types-of-svm

# ~~~~~~~~~~~~~~~~~~~~  Default SVM Model using the RBF kernel ~~~~~~~~~~~~~~~~~~~~~
svm_rbf <- svm(V11~., data = training_set)
summary(svm_rbf)
svm_rbf$gamma

# Confusion Matrix on the training set


# Pred on test set
pred = predict (svm_rbf, test_set)
pred

length(pred)
length(test_set$V11)
test_set$V11

cm = table(Predicted = pred, Actual = test_set$V11)
cm
accuracy = sum(diag(cm))/sum(cm)*100
accuracy

misclassification = (1-sum(diag(cm))/sum(cm))*100
misclassification

"""
library(caret)
confusionMatrix(table(pred, test_set$V11))
"""


# ~~~~~~~~~~~~~~~~~~~~   SVM model using the Linear model  ~~~~~~~~~~~~~~~~~~~~~
svm_linear = svm (V11~., data = training_set, kernel = "linear")
summary (svm_linear)

# Confusion Matrix
pred = predict (svm_linear, test_set)
pred
cm = table(Predicted = pred, Actual = test_set$V11)
cm
1-sum(diag(cm))/sum(cm)


# ~~~~~~~~~~~~~~~~~~~~   SVM model using sigmoid kernal  ~~~~~~~~~~~~~~~~~~~~~
svm_sigmoid = svm (V11~., data = training_set, kernel = "sigmoid")
summary (svm_sigmoid)

# Confusion Matrix
pred = predict (svm_sigmoid, test_set)
cm = table(Predicted = pred, Actual = test_set$V11)
cm
1-sum(diag(cm))/sum(cm)


# ~~~~~~~~~~~~~~~~~~~~   SVM model using polynomial kernal  ~~~~~~~~~~~~~~~~~~~~~
svm_polynomial = svm (V11~., data = training_set, kernel = "poly")
summary (svm_polynomial)

# Confusion Matrix
pred = predict (svm_polynomial, test_set)
cm_poly = table(Predicted = pred, Actual = test_set$V11)
cm_poly

accuracy = sum(diag(cm_poly))/sum(cm_poly)*100
accuracy

misclassificarion = (1-sum(diag(cm_poly))/sum(cm_poly))*100
misclassificarion


# ~~~~~~~~~~~~~~~~~~~~  Model Tuning  ~~~~~~~~~~~~~~~~~~~~
set.seed(123)
# tune function tunes the hyperparameters of the model using grid search method
tuned_model = tune(svm, V11~., data=training_set,
     ranges = list(epsilon = seq (0, 1, 0.1), cost = 2^(0:2)))
plot (tuned_model)
summary (tuned_model)
tuned_model$best.parameters
opt_model = tuned_model$best.model
summary(opt_model)

# Building the best model
svm_best <- svm (V11~., data = training_set, epsilon = 0, cost = 1)
summary(svm_best)


