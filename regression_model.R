#Load libraries
library(readxl)
library(skimr)
library(dplyr)
library(car)
library(Metrics)
library(leaps)
library(MASS)
library(glmnet)

#Load data
data <- read_excel("C:\\cars_data.xlsx")
View(data)

#EDA----------------------------------------------------------------------------
dim(data)
skim(data)
summary(data)

#Examine unique values
table(data$Brand)
table(data$Engine)
table(data$Gears)
table(data$Region)

#Remove incorrect values in 'Brand'
data <- data[!data$Brand %in% c("2016", "2021", "VW", "XC60", "Söker"), ]
table(data$Brand)

#Handle missing values
sum(is.na(data)) #645
sapply(data, function(x) sum(is.na(x)))
data$Dealer[is.na(data$Dealer)] <- "Private Dealers"
sapply(data, function(x) sum(is.na(x)))
data$Dealer[data$Dealer != "Private Dealers"] <- "Corporate Dealers"
table(data$Dealer)

#Remove duplicate rows
sum(duplicated(data)) #85
data <- data[!duplicated(data), ]
sum(duplicated(data))

#Convert columns to appropriate data types
data <- data %>%
  mutate_at(c("ModelYear", "Miles", "Price"), as.numeric) %>%
  mutate_at(c("Brand", "Model", "Engine", "Gears", "Region", "Dealer"), as.factor)

str(data)

#Drop columns with high variability in values
data <- data[, !(names(data) %in% c("Model", "Region"))]

#Handle outliers
numerical_columns <- c('ModelYear', 'Price', 'Miles')

remove_outliers <- function(data, columns) {
  for (col in columns) {
    q1 <- quantile(data[[col]], 0.25)
    q3 <- quantile(data[[col]], 0.75)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    data <- data[data[[col]] >= lower_bound & data[[col]] <= upper_bound, ]
  }
  return(data)
}

clean_data <- remove_outliers(data, numerical_columns)

#Add log-transformed columns for 'Price' and 'Miles'
clean_data$LogPrice <- log(clean_data$Price)
clean_data$LogMiles <- log(clean_data$Miles)

summary(clean_data)
skim(clean_data)

#View relationships among numerical variables
correlation_matrix <- cor(clean_data[, sapply(clean_data, is.numeric)])
print(correlation_matrix)

#Model Building-----------------------------------------------------------------
#Split data
spec = c(train = .6, validate = .2, test = .2)
set.seed(123) 
g = sample(cut(
  seq(nrow(clean_data)),
      nrow(clean_data)*cumsum(c(0,spec)),
      labels = names(spec)
))

res = split(clean_data, g)

train <- res$train    #Training set
val <- res$validate   #Validation set
test <- res$test      #Test set

#Model 1
model_1 <- lm(Price ~ . -LogPrice -LogMiles, train)
summary(model_1)
par(mfrow=c(2,2))
plot(model_1)
vif(model_1)

#Model 2
model_2 <- lm(LogPrice ~ . -Price -Miles, train)
summary(model_2)
par(mfrow=c(2,2))
plot(model_2)
vif(model_2)

#Model 3
#Detect and remove high leverage points using Cook's Distance
cook_threshold <- 4 /nrow(train)
high_leverage <- which(cooks.distance(model_2) > cook_threshold)
clean_train <- train[-high_leverage,]

model_3 <- lm(LogPrice ~ . -Price -Miles, clean_train)
summary(model_3)
par(mfrow = c(2, 2))
plot(model_3)
vif(model_3)

#Lasso and Ridge
x_train <- model.matrix(~ . -Price -Miles, clean_train)[, -1]
y_train <- clean_train$LogPrice

# Model 4
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0) 
ridge_lambda <- ridge_cv$lambda.min  
model_4 <- glmnet(x_train, y_train, alpha = 0, lambda = ridge_lambda)

# Model 5
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1) 
lasso_lambda <- lasso_cv$lambda.min
model_5 <- glmnet(x_train, y_train, alpha = 1, lambda = lasso_lambda)

# Model evaluation on validation set--------------------------------------------
calculate_mape <- function(actual, predicted) {
  mean(abs((actual - predicted) / actual)) * 100  #Calculate MAPE
}

x_val <- model.matrix(~ . -Price -Miles, val)[, -1]
y_val <- val$LogPrice

models <- list(model_1, model_2, model_3, model_4, model_5)
predictions <- list(
  predict(model_1, newdata = val),
  predict(model_2, newdata = val),
  predict(model_3, newdata = val),
  predict(model_4, s = ridge_lambda, newx = x_val),
  predict(model_5, s = lasso_lambda, newx = x_val)
)
results_val <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3", "Model 4", "Model 5"),
  RMSE = sapply(predictions, function(pred) rmse(val$LogPrice, pred)),
  MAPE = sapply(predictions, function(pred) calculate_mape(val$LogPrice, pred))
)

results_val

#Predictions on test data (best model)
x_test <- model.matrix(~ . -Price -Miles, test)[, -1]
y_test <- test$LogPrice

test_pred <- predict(model_5, s = lasso_lambda, newx = x_test)

results_test <- data.frame(
  RMSE_test = c(rmse(test$LogPrice, test_pred)),
  MAPE_test = c(calculate_mape(test$LogPrice, test_pred))
)

results_test

#Compare the predictated value and the actual value
test_pred <- predict(model_5, s = lasso_lambda, newx = x_test)

comparison <- data.frame(
  Actual_Price = exp(y_test),  
  Predicted_Price = exp(test_pred)
)

print("First 10 rows:")
print(head(comparison, 10))

print("Last 10 rows:")
print(tail(comparison, 10))


