
# Random Forest has its own tuning algorithm RFtune
# As you may have seen, there are many ways to tune any model
# This example illustrates some of the ways. 
# This is the only example, that I have come across, which has all the methods quoted in a single file.
# REF: https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/


library(randomForest)
library(mlbench)
library(caret)
library(e1071)

# Load Dataset
data(Sonar)
dataset <- Sonar
x <- dataset[,1:60]
y <- dataset[,61]


# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
mtry
tunegrid <- expand.grid(mtry=mtry)
rf_default <- train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)


# fromgithub
# http://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/


# load dataset
data(Sonar)

dataset <- Sonar
x <- dataset[,1:60]   # predictors
y <- dataset[,61]     # labels

# Create model with default paramters


# 1 - Tune using 'caret' package
# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
# mtry <- sqrt(ncol(x))
metric <- "Accuracy"
rf_random <- train(Class ~ ., data=dataset, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)


# Grid Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(Class ~ ., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

# 2 - Tune using algorithm tools
# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry <- tuneRF(x, y, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)

# 3 - Tune using own paramaters search
# Tune Manually

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(x))))
modellist <- list()
for (ntree in c(1000, 1500, 2000, 2500)) {
  set.seed(seed)
  fit <- train(Class ~ ., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)



