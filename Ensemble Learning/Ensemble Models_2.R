# Loading the required libraries
library(caret)
library(DataExplorer)

#Loading the hackathon dataset
data<-read.csv(url('https://datahack-prod.s3.ap-south-1.amazonaws.com/train_file/train_u6lujuX_CVtuZ9i.csv'))
View(data)

# The structure of dataset
str(data)
summary(data)

# Check missing values
sum(is.na(data))
plot_missing(data) 
colSums(is.na(data))

# Imputing missing values using median
preProcValues <- preProcess(data, method = "medianImpute")

# install.packages("RANN")
library('RANN')
data_processed <- predict(preProcValues, data)
sum(is.na(data_processed))
View(data_processed)


# Data Split
index <- createDataPartition(data_processed$Loan_Status, p=0.75, list=FALSE)
trainSet <- data_processed[ index,]
testSet <- data_processed[-index,]
str(trainSet)
str(testSet)

# Checking if there is any bias in sampling
prop.table(table(trainSet$Loan_Status))
prop.table(table(testSet$Loan_Status))

# Defining the training controls for multiple models
Trian_Control <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)

#Training the knn model
model_knn<-train(Loan_Status~., data = trainSet, method='knn', trControl=Trian_Control, tuneLength=3)
testSet$pred_knn<-predict(object = model_knn, testSet)
confusionMatrix(testSet$Loan_Status,testSet$pred_knn)


#Training the Logistic decision tree model

model_dt<-train(Loan_Status~., data = trainSet, method='rpart', trControl=Trian_Control, tuneLength=3)
testSet$pred_dt<-predict(object = model_dt,testSet)
confusionMatrix(testSet$Loan_Status,testSet$pred_dt)


#Training the random forest model
model_rf<-train(Loan_Status~., data = trainSet, method='rf', trControl=Trian_Control, tuneLength=3)
testSet$pred_rf<-predict(object = model_rf, testSet)
confusionMatrix(testSet$Loan_Status,testSet$pred_rf)

# ~~~~~~~~~~~~~~~~~~~~ Averaging: 

#Predicting the probabilities
testSet$pred_rf_prob<-predict(object = model_rf,testSet,type='prob')
testSet$pred_knn_prob<-predict(object = model_knn,testSet,type='prob')
testSet$pred_dt_prob<-predict(object = model_dt,testSet,type='prob')

#Taking average of predictions
testSet$pred_avg<-(testSet$pred_rf_prob$Y+testSet$pred_knn_prob$Y+testSet$pred_dt_prob$Y)/3

#Splitting into binary classes at 0.5
testSet$pred_avg<-as.factor(ifelse(testSet$pred_avg>0.5,'Y','N'))


# ~~~~~~~~~~~~~~~~~~~~ Majority Voting: 
#The majority vote
testSet$pred_majority<-as.factor(ifelse(testSet$pred_rf=='Y' & testSet$pred_knn=='Y','Y',ifelse(testSet$pred_rf=='Y' & testSet$pred_dt=='Y','Y',ifelse(testSet$pred_knn=='Y' & testSet$pred_dt=='Y','Y','N'))))


# ~~~~~~~~~~~~~~~~~~~~ Weighted Average: 
#Taking weighted average of predictions
testSet$pred_weighted_avg<-(testSet$pred_rf_prob$Y*0.25)+(testSet$pred_knn_prob$Y*0.25)+(testSet$pred_dt_prob$Y*0.5)

#Splitting into binary classes at 0.5
testSet$pred_weighted_avg<-as.factor(ifelse(testSet$pred_weighted_avg>0.5,'Y','N'))
