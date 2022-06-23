library(plyr)
library(readr)
library(dplyr)
library(caret)
library(caretEnsemble)

# ~~~~~~~~~~~~~~~~~~~~~~ data ~~~~~~~~~~~~~~~~~~~~~~  
data = read.csv("diabetes.csv")
View(data)
str(data)
sum(is.na(data))

data$Outcome <- factor(data$Outcome, levels=c(0,1), labels=c("N", "Y"))
table(data$Outcome)
str(data)

# ~~~~~~~~~~~~~~~~~~~~~~ data split~~~~~~~~~~~~~~~~~~~~~~  
set.seed(100)
library(caTools)
spl = sample.split(data$Outcome, SplitRatio = 0.7)
train = subset(data, spl==TRUE)
test = subset(data, spl==FALSE)
dim(train)
dim(test)


# ~~~~~~~~~~~~~~~~~~~~~~ Fit the logistic regression model ~~~~~~~~~~~~~~~~~~~~~~ 
model_glm = glm(Outcome ~ ., family=binomial, data = train)
summary(model_glm)

library(jtools)
summ(model_glm) # option for the data scientist to select the significant features

# Predictions on the test set
predictTest = predict(model_glm, newdata = test, type = "response")

# Confusion matrix on test set
cm = table(test$Outcome, predictTest >= 0.5)
cm

accuracy = sum(diag(cm))/sum(cm)
accuracy

# ~~~~~~~~~~~~~~~~~~~~~~ Train Control ~~~~~~~~~~~~~~~~~~~~~~ 
set.seed(100)
control <- trainControl(method="repeatedcv", number=5, repeats=5)


# ~~~~~~~~~~~~~~~~~~~~~~ GLM with repeated cv ~~~~~~~~~~~~~~~~~~~~~~ 
lr_model <- train(Outcome ~., data=train, method="glm", metric="Accuracy", trControl=control)
predictTest = predict(lr_model, newdata = test)
cm_lr = table(test$Outcome, predictTest)
cm_lr
accuracy_lr = sum(diag(cm_lr))/sum(cm_lr)
accuracy_lr


# ~~~~~~~~~~~~~~~~~~~~~~ Bagging ~~~~~~~~~~~~~~~~~~~~~~ 
# Bagged Decision Tree
bagDT_model <- train(Outcome ~., data=train, method="treebag", metric="Accuracy", trControl=control)

#Predictions on the test set
predictTest = predict(bagDT_model, newdata = test)

# Confusion matrix on test set
cm1 = table(test$Outcome, predictTest)
cm1

accuracy1 = sum(diag(cm1))/sum(cm1)
accuracy1


# ~~~~~~~~~~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~~~~~~~~ 
set.seed(100)

rf_model <- train(Outcome ~., data=train, method="rf", metric="Accuracy", trControl=control)

predictTest = predict(rf_model, newdata = test, type = "raw")

# Confusion matrix on test set
cm2 = table(test$Outcome, predictTest)
cm2

accuracy2 = sum(diag(cm2))/sum(cm2)
accuracy2


# ~~~~~~~~~~~~~~~~~~~~~~ Boosting ~~~~~~~~~~~~~~~~~~~~~~ 
# Stochastic Gradient Boosting
set.seed(100)
gbm_model <- train(Outcome ~., data=train, method="gbm", metric="Accuracy", trControl=control)

predictTest = predict(gbm_model, newdata = test)

cm3 = table(test$Outcome, predictTest)
cm3

accuracy3 = sum(diag(cm3))/sum(cm3)
accuracy3

# ~~~~~~~~~~~~~~~~~~~~~~ Stacking ~~~~~~~~~~~~~~~~~~~~~~ 
set.seed(100)

control_stacking <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions='final', classProbs=TRUE,
                                 index = createResample(train$Outcome, 10))

# Possible CARET inbuilt algorithms, and the data scientist can select the more suitable (no need all)
# algorithms_to_use <- c('knn', 'rpart', 'treebag', 'glm', 'gbm', 'rf', 'svmRadial', 'adaboost', 'xgbDART', 'nb', 'nnet')

# algorithms_to_use <- c('rpart', 'knn', 'svmRadial', 'adaboost', 'xgbDART')
algorithms_to_use <- c('rpart', 'knn', 'svmRadial')

stacked_models <- caretList(Outcome ~., data=train, trControl=control_stacking, methodList=algorithms_to_use)

stacking_results <- resamples(stacked_models)

summary(stacking_results)


# ~~~~~~~~~~~~~~~~~~~~~~ GLM using Stack ~~~~~~~~~~~~~~~~~~~~~~
stackControl <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions='final', classProbs=TRUE)

set.seed(100)
glm_stack <- caretStack(stacked_models, method="glm", metric="Accuracy", trControl=stackControl)
print(glm_stack)


# Compare Models - Model Accuracies
# Basic_LR = 0.8
# LR_with_rcv = 0.8
# Bagged_DT = 0.79
# Random_Forest = 0.78
# GBM = 0.77
# LR_with_Stack = 0.77
# Based on the experiments, bagged DT (under Bagging) is better among all the ensembled models for this dataset.
# Check the models with some other datasets for more practice
