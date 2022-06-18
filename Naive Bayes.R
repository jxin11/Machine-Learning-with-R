# Naive Bayes

fileURL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
download.file(fileURL, destfile="breast-cancer-wisconsin.data", method="curl")
# read the data
df <- read.table("breast-cancer-wisconsin.data", na.strings = "?", sep=",")
str(df)

# The ID Number column which is not a useful feature for this analysis, is dropped. 
df <- df[ , -1]

# Columns are renamed to facilitate semantic interpretation

ds <- df
names(ds) <- c("ClumpThickness",
               "UniformityCellSize",
               "UniformityCellShape",
               "MarginalAdhesion",
               "SingleEpithelialCellSize",
               "BareNuclei",
               "BlandChromatin",
               "NormalNucleoli",
               "Mitoses",
               "Class")

# Checking for class balance
table(ds$Class)
prop.table(table(ds$Class))

barplot(table(ds$Class),
        xlab="Class (2 = Benign, 4 = Malignant)", ylab="Count", col=c("darkblue","red"),
        legend = levels(ds$Class), beside=TRUE)


# Checking for correlation among independent variables
corrTable <- cor(df[,c("V2","V3","V4","V5","V6","V7","V8","V9","V10")])
corrTable
library(DataExplorer)
plot_correlation(ds,'continuous', cor_args = list("use" = "pairwise.complete.obs"))

# Gaussianness test - Check if the density plot is Gaussian
plot_density(ds)

# Missing Values
# Check for missing values
plot_missing(ds)

# Total missing Values
sum (is.na(ds))

# Missing values by the column
colSums(sapply(ds,is.na))


# Replace the missing value in BareNuckei variable with its mean

ds$BareNuclei <- ifelse(is.na(ds$BareNuclei),
                        ave(ds$BareNuclei, FUN = function(x) mean(x, na.rm = TRUE)),
                        ds$BareNuclei)
plot_missing(ds)

summary(ds)


# Tasks
# 1. Perform any otehr preprocessing you see necessary
ds$Class = as.factor(ds$Class)

# 2. Use stratified sampling divide the dataset into trining and validation datasets
library(caTools)
set.seed(0)
ds2 = ds
split = sample.split(ds$Class, SplitRatio = 0.7)
train_ds = subset(ds2, split == TRUE)
test_ds = subset(ds2, split == FALSE)

# 3. Check class distrbution of training and validation datasets


# 4. Install, load and read about "naivebayes" package
library(naivebayes)

# 5. Use the following command to build a basic naive bayes model and evaluate it. 

Naive_Bayes_basic = naive_bayes(x = train_ds[ , -10],
                        y = train_ds$Class , laplace = 0 )
                        
# 6. Evaluate the performance of this model, using the exmple of logistic regression model.
library(MoEClust)
y_pred_train_raw <- predict(Naive_Bayes_basic, type = 'prob', 
                              newdata=train_ds[ ,-10])
y_pred_train_raw
y_pred_train_class <- predict(Naive_Bayes_basic, type = 'class', 
                            newdata=train_ds[ ,-10])
y_pred_train_class

# confusion matrix
cm_training = table(train_ds$Class, y_pred_train_class)
cm_training

library(ggplot2)
ggplot(data = as.data.frame(cm_training),
       mapping = aes(x = y_pred_train_class,
                     y = Var1)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 0.5, size = 13) +
  labs(x = "Prediction", y = "Actual") + scale_y_discrete(limits = rev(levels(train_ds$Class)))

accuracy_training <- sum(diag(cm_training))/sum(cm_training)
accuracy_training

# Predicting the Test set results
# pred_prob_test <- predict(classifier, type = 'response', test_set[ ,-12] )
y_pred_test_raw <- predict(Naive_Bayes_basic, type = 'prob', 
                            newdata=test_ds[ ,-10])
y_pred_test_raw
y_pred_test_class <- predict(Naive_Bayes_basic, type = 'class', 
                              newdata=test_ds[ ,-10])
y_pred_test_class

cm_test = table(test_ds$Class, y_pred_test_class)
cm_test

ggplot(data = as.data.frame(cm_test),
       mapping = aes(x = y_pred_test_class,
                     y = Var1)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 0.5, size = 13) +
  labs(x = "Prediction", y = "Actual") + scale_y_discrete(limits = rev(levels(test_ds$Class)))

accuracy_test <- sum(diag(cm_test))/sum(cm_test)
accuracy_test

# Using formulae compute all other evaluation metrics

# ROC curve on test set
# install.packages("ROCR")
library(ROCR)
# install.packages("gplots")

# To draw ROC we need to predict the prob values. 

pred = prediction(as.numeric(y_pred_test_class), as.numeric(test_ds$Class))
perf = performance(pred, "tpr", "fpr")
pred
perf
plot(perf, colorize = T)
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1.0),
     text.adj= c(-0.2,1.0))

# Area Under Curve
auc <- as.numeric(performance(pred, "auc")@y.values)
auc <-  round(auc, 3)
auc

# 7. Buiild a second model with laplacian smoothing using the following code:


# Naive Bayes with Laplacian Smoothing

Naive_Bayes_Laplace = naive_bayes(x = train_ds[ , -10],
                                y = train_ds$Class , laplace = 1 )

# 8. Once again evaluate this second model.

y2_pred_train_raw <- predict(Naive_Bayes_Laplace, type = 'prob', 
                            newdata=train_ds[ ,-10])
y2_pred_train_raw
y2_pred_train_class <- predict(Naive_Bayes_Laplace, type = 'class', 
                              newdata=train_ds[ ,-10])
y2_pred_train_class

# confusion matrix
cm_training2 = table(train_ds$Class, y2_pred_train_class)
cm_training2

ggplot(data = as.data.frame(cm_training2),
       mapping = aes(x = y2_pred_train_class,
                     y = Var1)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 0.5, size = 13) +
  labs(x = "Prediction", y = "Actual") + scale_y_discrete(limits = rev(levels(train_ds$Class)))

accuracy_training2 <- sum(diag(cm_training2))/sum(cm_training2)
accuracy_training2

# Predicting the Test set results
# pred_prob_test <- predict(classifier, type = 'response', test_set[ ,-12] )
y2_pred_test_raw <- predict(Naive_Bayes_Laplace, type = 'prob', 
                           newdata=test_ds[ ,-10])
y2_pred_test_raw
y2_pred_test_class <- predict(Naive_Bayes_Laplace, type = 'class', 
                             newdata=test_ds[ ,-10])
y2_pred_test_class

cm_test2 = table(test_ds$Class, y2_pred_test_class)
cm_test2

ggplot(data = as.data.frame(cm_test2),
       mapping = aes(x = y2_pred_test_class,
                     y = Var1)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 0.5, size = 13) +
  labs(x = "Prediction", y = "Actual") + scale_y_discrete(limits = rev(levels(test_ds$Class)))


accuracy_test2 <- sum(diag(cm_test2))/sum(cm_test2)
accuracy_test2

# Using formulae compute all other evaluation metrics

# ROC curve on test set
# install.packages("ROCR")
library(ROCR)
# install.packages("gplots")

# To draw ROC we need to predict the prob values. 

pred2 = prediction(as.numeric(y2_pred_test_class), as.numeric(test_ds$Class))
perf2 = performance(pred2, "tpr", "fpr")
pred2
perf2
plot(perf2, colorize = T)
plot(perf2, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

# Area Under Curve
auc2 <- as.numeric(performance(pred2, "auc")@y.values)
auc2 <-  round(auc, 3)
auc2

# 9. Compile your results and present your analysis
# 10. You may extend this experimentation to your dataset
# 11. Load your work in to OneNote.
