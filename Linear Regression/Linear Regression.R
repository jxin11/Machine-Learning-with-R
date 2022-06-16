# Linear Regression

# Read dataset
df = read.csv('50_Startups.csv')

# Check NA
colSums(is.na(df))

# Split into train & test
# install.packages("caTools")
library(caTools)

set.seed(0)
ind <- sample(2, nrow(df), replace=TRUE, prob=c(0.7,0.3))
training_set <- df[ind==1,]
test_set <- df[ind==2,]

# Fit Multiple Linear Regression on Training Set

regressor = lm(formula = Profit ~., data = training_set)

summary(regressor)

#  Predict Test Set
y1 = predict(regressor, training_set)
data.frame(y1, training_set$Profit)

y_pred = predict(regressor, test_set)
data.frame(y_pred, test_set$Profit)

# RMSE on Train Set
RMSE_train <- sqrt(mean((training_set$Profit-y1)^2))
RMSE_train

# RMSE on Test Set
RMSE_test <- sqrt(mean((test_set$Profit-y_pred)^2))
RMSE_test

# Model Evaluation
# Plotting
# Ref: https://cran.r-project.org/web/packages/jtools/vignettes/summ.html
install.packages('jtools')
library(jtools)
effect_plot(regressor, pred = R.D.Spend, interval = TRUE, plot.points = TRUE)
effect_plot(regressor, pred = Marketing.Spend, interval = TRUE, plot.points = TRUE)
effect_plot(regressor, pred = Administration, interval = TRUE, plot.points = TRUE)
effect_plot(regressor, pred = State, interval = TRUE, plot.points = TRUE)


# Backward Elimination
regressor2 = lm(formula = Profit ~ R.D.Spend + Marketing.Spend + State,
                data = training_set)
summary(regressor2)
y2_pred = predict(regressor2, test_set)
RMSE_test2 <- sqrt(mean((test_set$Profit-y2_pred)^2))
RMSE_test2

regressor3 = lm(formula = Profit ~ R.D.Spend + Marketing.Spend + Administration,
                data = training_set)
summary(regressor3)
y3_pred = predict(regressor3, test_set)
RMSE_test3 <- sqrt(mean((test_set$Profit-y3_pred)^2))
RMSE_test3

regressor4 = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = training_set)
summary(regressor4)
y4_pred = predict(regressor4, test_set)
RMSE_test4 <- sqrt(mean((test_set$Profit-y4_pred)^2))
RMSE_test4

regressor5 = lm(formula = Profit ~ R.D.Spend,
                data = training_set)
summary(regressor5)
y5_pred = predict(regressor5, test_set)
RMSE_test5 <- sqrt(mean((test_set$Profit-y5_pred)^2))
RMSE_test5

#glm
glmmodel <- glm(Profit ~ ., data=training_set)
summary(glmmodel)
