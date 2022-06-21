# SVM_REGRESSION in R

library(ggplot2)
library(e1071)

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Checking missing Values
sum (is.na(dataset))
colSums(sapply(dataset,is.na))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# ~~~~~~~~~~~~~~~~~~~~  Default SVM Model using the RBF kernel ~~~~~~~~~~~~~~~~~~~~~
svm_rbf <- svm(Profit~., data = training_set)
summary(svm_rbf)

pred = predict (svm_rbf, test_set)
pred
# table(pred, test_set$Profit)

library(caret)
summary(pred)
RMSE(pred, test_set$Profit) # Root mean squared error
MAE(pred, test_set$Profit) # Mean Absolute Error


# ~~~~~~~~~~~~~~~~~~~~   SVM model using the Linear model  ~~~~~~~~~~~~~~~~~~~~~
svm_linear = svm (Profit~., data = training_set, kernel = "linear")
summary (svm_linear)

pred2 = predict (svm_linear, test_set)
pred2
# table(pred2, test_set$Profit)

library(caret)
summary(pred2)
RMSE(pred2, test_set$Profit) # Root mean squared error
MAE(pred2, test_set$Profit) # Mean Absolute Error

# ~~~~~~~~~~~~~~~~~~~~   SVM model using sigmoid kernal  ~~~~~~~~~~~~~~~~~~~~~
svm_sigmoid = svm (Profit~., data = training_set, kernel = "sigmoid")
summary (svm_sigmoid)

pred3 = predict (svm_sigmoid, test_set)
pred3
table(pred3, test_set$Profit)

library(caret)
summary(pred3)
RMSE(pred3, test_set$Profit) # Root mean squared error
MAE(pred3, test_set$Profit) # Mean Absolute Error


# ~~~~~~~~~~~~~~~~~~~~   SVM model using polynomial kernal  ~~~~~~~~~~~~~~~~~~~~~
svm_polynomial = svm (Profit~., data = training_set, kernel = "poly")
summary (svm_polynomial)

pred4 = predict (svm_polynomial, test_set)
pred4
table(pred4, test_set$Profit)

library(caret)
summary(pred4)
RMSE(pred4, test_set$Profit) # Root mean squared error
MAE(pred4, test_set$Profit) # Mean Absolute Error

# ~~~~~~~~~~~~~~~~~~~ Kernel comparison ~~~~~~~~~~~~~~~~~~~ 
cat("RMSE using RBF Kernal is ", RMSE(pred, test_set$Profit))
cat("RMSE using LINEAR Kernal is ", RMSE(pred2, test_set$Profit))
cat("RMSE using SIGMOID Kernal is ", RMSE(pred3, test_set$Profit))
cat("RMSE using POLYNOMIAL Kernal is ", RMSE(pred4, test_set$Profit))

