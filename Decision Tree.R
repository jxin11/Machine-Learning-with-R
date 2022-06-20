# Decision Tree Classification on Breast cancer dataset

library(ggplot2)
# Downloading the file
library(RCurl)
fileURL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
download.file(fileURL, destfile="breast-cancer-wisconsin.data", method="libcurl")

 # read the data
data <- read.table("breast-cancer-wisconsin.data", na.strings = "?", sep=",")
str(data)

# Remove ID column, col = 1
data <- data[,-1]
dim(data)

library(formattable)
formattable(data)
# Name the columns. 
# These names are displayed in the tree to facilitate semantic interpretation

names(data) <- c("ClumpThickness",
                 "UniformityCellSize",
                 "UniformityCellShape",
                 "MarginalAdhesion",
                 "SingleEpithelialCellSize",
                 "BareNuclei",
                 "BlandChromatin",
                 "NormalNucleoli",
                 "Mitoses",
                 "Class")

# Numerical values in the response variable are converted to labels
formattable(data)

data$Class <- factor(data$Class, levels=c(2,4), labels=c("benign", "malignant"))

print(summary(data))

# Proportions of the class values
table(data$Class)
prop.table(table(data$Class)) 
 

# Note that there are 16 missing values in BareNuclei
# Later you will see that there is no imputation of these missing values. 
# Investigate how decision trees handle missing values
# Read rpart documentation from this.
# This link has some extra information: 
# https://stats.stackexchange.com/questions/96025/how-do-decision-tree-learning-algorithms-deal-with-missing-values-under-the-hoo


# Dividing the dataset into training and validation sets. There are many ways to do this.
# Alternate method is also listed here.

set.seed(123)
ind <- sample(2, nrow(data), replace=TRUE, prob=c(0.7, 0.3))
train_Data <- data[ind==1,]
validation_Data <- data[ind==2,]
table(train_Data$Class)
table(validation_Data$Class)

# Proportions of the class values
prop.table(table(train_Data$Class)) 


# Stratified Sampling -Alternate methods for data split
# Stratified Sampling ensures that the raio of the classes in teh target variable after the split is the same as that in the oroginal dataset
# Create training and testing sets
library(caTools)
set.seed(123)
split = sample.split(data$Class, SplitRatio = 0.7)
dataTrain = subset(data, split == TRUE)
dataTest = subset(data, split == FALSE)

table(dataTrain$Class)
prop.table(table(dataTrain$Class)) 

table(dataTest$Class)
prop.table(table(dataTest$Class)) 

library(ROCR)

cm <- function(model, train, test, targetCol){
        
# cm - train set
pred_train <- predict(model, type='class', newdata=train[ ,-which(names(train) %in% c(targetCol))])
cm_training = table(train[[targetCol]], pred_train)
print(cm_training)

print(ggplot(data = as.data.frame(cm_training),
             mapping = aes(x = pred_train, y = Var1)) + ggtitle("Prediction on Train Set") +
              geom_tile(aes(fill = Freq)) +
              geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 0.5, size = 13) +
              labs(x = "Prediction", y = "Actual") + scale_y_discrete(limits = rev(levels(train[[targetCol]]))))

# accuracy - train set
accuracy_training <- round(sum(diag(cm_training))/sum(cm_training),4)
print(paste("Accuracy for training set: ", accuracy_training))

# cm - test set
pred_test <- predict(model, type='class', newdata=test[ ,-which(names(test) %in% c(targetCol))])
cm_testing = table(test[[targetCol]], pred_test)
print(cm_testing)

print(ggplot(data = as.data.frame(cm_testing),
             mapping = aes(x = pred_test, y = Var1)) + ggtitle("Prediction on Test Set") +
              geom_tile(aes(fill = Freq)) +
              geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 0.5, size = 13) +
              labs(x = "Prediction", y = "Actual") + scale_y_discrete(limits = rev(levels(test[[targetCol]]))))

# accuracy - test set
accuracy_testing <- round(sum(diag(cm_testing))/sum(cm_testing),4)
print(paste("Accuracy for testing set: ", accuracy_testing))

}

roc <- function(model, test, targetCol){
        Predict_ROC = predict(model, test)
        Predict_ROC
        Predict_ROC[,2]
        
        pred = prediction(Predict_ROC[,2], test[[targetCol]])
        perf = performance(pred, "tpr", "fpr")
        pred
        perf
        plot(perf, colorize = T)
        plot(perf, colorize=T, 
             main = "ROC curve",
             ylab = "Sensitivity",
             xlab = "1-Specificity",
             print.cutoffs.at=seq(0,1,0.3),
             text.adj= c(-0.2,1.7))
        
        # Area Under Curve
        auc = as.numeric(performance(pred, "auc")@y.values)
        auc = round(auc, 3)
        auc
        print(paste("AUC: ", auc))
}

# install.packages('rpart') --> (Recursive Partitioning And Regression Trees) and the R implementation of the CART algorithm
# install.packages("rpart.plot")

library(rpart)
library(rpart.plot)
library(party)

"Can generate different types of trees with rpart
Default split is with Gini index"

tree = rpart(Class~ ., data=dataTrain,method="class")
tree
prp(tree) # plot Rpart Model
prp (tree, type = 5, extra = 100)
rpart.plot(tree, extra = 101, nn = TRUE)
plotcp(tree)

cm(tree, dataTrain, dataTest, "Class")
roc(tree, dataTest, "Class")

# DT using Party Package
tree = ctree(Class~ ., data=dataTrain)
tree
plot(tree)



# Split with entropy information
ent_Tree = rpart(Class ~ ., data=dataTrain, method="class", parms=list(split="information"))
ent_Tree
prp(ent_Tree)
prp(ent_Tree)
rpart.plot(ent_Tree, extra = 101, nn = TRUE)
plotcp(ent_Tree)

cm(ent_Tree, dataTrain, dataTest, "Class")
roc(ent_Tree,dataTest,"Class")

library(rpart.plot)
plotcp(tree)

# Here we use tree with parameter settings.
# This code generates the tree with training data
tree_with_params = rpart(Class ~ ., data=dataTrain, method="class", minsplit = 1, minbucket = 10, cp = -1)
prp (tree_with_params)
rpart.plot(tree_with_params, extra = 101, nn = TRUE)
print(tree_with_params)
summary(tree_with_params)
plot(tree_with_params)
text(tree_with_params)
plotcp(tree_with_params)

cm(tree_with_params, dataTrain, dataTest, "Class")
roc(tree_with_params,dataTest,"Class")

# Now we predict and evaluate the performance of the trained tree model 
Predict = predict(tree_with_params, dataTest)
# Now examine the values of Predict. These are the class probabilities
Predict

# pred <= predict (mymodel, dataset, type = 'prob')
# To produce classes only, without the probabilities, run the next command.
# By default threshold is set at 0.5 to produce the classes

Predict = predict(tree_with_params, dataTest, type = "class")
Predict


# Producing confusion matrix
Confusion_matrix = table(Predict, validation_Data$Class)
Confusion_matrix

# Calculating the accuracy using the cofusion matrix
Accuracy = sum(diag(Confusion_matrix))/sum(Confusion_matrix)
Accuracy

# Performance of the DT model
library(caret)
confusionMatrix(Predict, validation_Data$Class)

# ROC curve
# install.packages("ROCR")
library(ROCR)
# install.packages("gplots")

# To draw ROC we need to predict the prob values. So we run predict again
# Note that PredictROC is same as Predict with "type = prob"

Predict_ROC = predict(tree_with_params, validation_Data)
Predict_ROC
Predict_ROC[,2]
validation_Data$Class

pred = prediction(Predict_ROC[,2], validation_Data$Class)
pred
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
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc

