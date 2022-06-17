# Logistic Regression

# Objective 1 - Perform Logistic Regression with and without preproccessing and compare the results.
# Objective 2 - Compute sensitivity, specificity and AUC. Draw ROC curves.
# Objective 3 - Perform various data cleaning and  normalization operations and compare results.

# Import library
library(DataExplorer)
library(ggplot2)
library(data.table)


# Read different sets of data
df <- read.csv('aug_train2.csv', 
               header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_impute_mice <- read.csv('aug_train_mice.csv', 
                           header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mF <- read.csv('aug_train_mF.csv', 
                  header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mice_encoded <- read.csv('aug_train_mice_encoded.csv', 
                            header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mice_encoded_norm <- read.csv('aug_train_clean.csv', 
                                 header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)

# Convert to factor dataset
dfList = c(df, df_impute_mice, df_mF, df_mice_encoded, df_mice_encoded)
for (data in dfList){
  dfList[["target"]] <- factor(dfList[["target"]], levels=c(0,1), labels=c("0", "1"))
}


# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(0)

# Note that this is Startified sampling method
df2 = df_mice_encoded_norm
split = sample.split(df2$target, SplitRatio = 0.7)
training_set = subset(df2, split == TRUE)
training_set$X <- NULL
training_set$X.1 <- NULL
test_set = subset(df2, split == FALSE)
test_set$X <- NULL
test_set$X.1 <- NULL

# Checking Class distribution
table(df2$target)
prop.table(table(df2$target))
prop.table(table(training_set$target))
prop.table(table(test_set$target))


# Building classifier
classifier = glm(target ~.,training_set,family = binomial)
summary(classifier)


# Predicting the Training set results
# install.packages("MoEClust")
library(MoEClust)
# pred_prob_training <- predict(classifier, type = 'response', training_set[ ,-8] )
pred_prob_training <- predict(classifier, type = 'response', 
                              newdata=drop_levels(classifier, training_set[ ,-8]))
# Ref: https://rdrr.io/cran/MoEClust/man/drop_levels.html
pred_class_training = ifelse(pred_prob_training > 0.5, 1, 0)
cbind(pred_prob_training, pred_class_training)
cm_training = table(training_set$target, pred_class_training)
cm_training

accuracy_training <- sum(diag(cm_training))/sum(cm_training)
accuracy_training

# Predicting the Test set results
# pred_prob_test <- predict(classifier, type = 'response', test_set[ ,-12] )
pred_prob_test <- predict(classifier, type = 'response', 
                          newdata=drop_levels(classifier, test_set[ ,-8]))
pred_class_test = ifelse(pred_prob_test > 0.5, 1, 0)
cm_test = table(test_set$target, pred_class_test)
cm_test

accuracy_test <- sum(diag(cm_test))/sum(cm_test)
accuracy_test

# Using formulae compute all other evaluation metrics

# ROC curve on test set
# install.packages("ROCR")
library(ROCR)
# install.packages("gplots")

# To draw ROC we need to predict the prob values. 

pred = prediction(pred_prob_test, test_set$target)
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







