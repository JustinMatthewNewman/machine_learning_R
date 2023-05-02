library(e1071)
library(ggplot2)
library(dplyr)

par(mfrow = c(1, 1)) # Default value for mf

## SVM MTCARS

# Load the mtcars dataset
data(mtcars)
# look for clusters
pairs(mtcars)
# Prepare the data
mtcars$am <- as.factor(mtcars$am)
features <- c("disp", "qsec")
mtcars_subset <- mtcars[, c(features, "am")]
# Split the data into training and test sets
set.seed(1)
train_index <- sample(1:nrow(mtcars_subset), 0.7 * nrow(mtcars_subset))
train_data <- mtcars_subset[train_index, ]
test_data <- mtcars_subset[-train_index, ]
# Train the SVM classifier using a linear kernel
svmfit <- svm(am ~ .,
              data = train_data,
              kernel = "linear",
              cost = 10,
              scale = FALSE)
# Perform cross-validation to choose the right cost
tune_out <- tune(svm,
                 am ~ .,
                 data = train_data,
                 kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_linear <- table(true = test_data$am, predicted = test_pred)
print(cm_linear)
# Calculate the error rate for the linear kernel
error_rate_linear <- 1 - sum(diag(cm_linear)) / sum(cm_linear)
cat("Error rate (linear kernel):", error_rate_linear, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)
# Train the SVM classifier using a radial kernel
svmfit <- svm(am ~ .,
              data = train_data,
              kernel = "radial",
              gamma = 1,
              cost = 1)
# Perform cross-validation to choose the right cost and gamma
tune_out <- tune(svm,
                 am ~ .,
                 data = train_data,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_radial <- table(true = test_data$am, predicted = test_pred)
print(cm_radial)
# Calculate the error rate for the radial kernel
error_rate_radial <- 1 - sum(diag(cm_radial)) / sum(cm_radial)
cat("Error rate (radial kernel):", error_rate_radial, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)















## SPAM SVM
spam <- read.csv("~/Documents/math358/kernlab_spam.csv")
# Recode the 'type' column
spam <- spam %>%
  mutate(type = ifelse(type == "spam", 1, 0))

# Prepare the data
spam$type <- as.factor(spam$type)
features <- c("internet", "free")
spam_subset <- spam[, c(features, "type")]
# Split the data into training and test sets
set.seed(1)
train_index <- sample(1:nrow(spam_subset), 0.7 * nrow(spam_subset))
train_data <- spam_subset[train_index, ]
test_data <- spam_subset[-train_index, ]
# Train the SVM classifier using a linear kernel
svmfit <- svm(type ~ .,
              data = train_data,
              kernel = "linear",
              cost = 10,
              scale = FALSE)
# Perform cross-validation to choose the right cost
tune_out <- tune(svm,
                 type ~ .,
                 data = train_data,
                 kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_linear <- table(true = test_data$type, predicted = test_pred)
print(cm_linear)
# Calculate the error rate for the linear kernel
error_rate_linear <- 1 - sum(diag(cm_linear)) / sum(cm_linear)
cat("Error rate (linear kernel):", error_rate_linear, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)
# Train the SVM classifier using a radial kernel
svmfit <- svm(type ~ .,
              data = train_data,
              kernel = "radial",
              gamma = 1,
              cost = 1)
# Perform cross-validation to choose the right cost and gamma
tune_out <- tune(svm,
                 type ~ .,
                 data = train_data,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_radial <- table(true = test_data$type, predicted = test_pred)
print(cm_radial)
# Calculate the error rate for the radial kernel
error_rate_radial <- 1 - sum(diag(cm_radial)) / sum(cm_radial)
cat("Error rate (radial kernel):", error_rate_radial, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)











## HEART SVM
heart <- read.csv("~/Documents/math358/HeartFailure.csv")
heart

pairs(heart)

# Prepare the data
heart$DEATH_EVENT <- as.factor(heart$DEATH_EVENT)
features <- c("ejection_fraction", "platelets")
heart_subset <- heart[, c(features, "DEATH_EVENT")]
# Split the data into training and test sets
set.seed(1)
train_index <- sample(1:nrow(heart_subset), 0.7 * nrow(heart_subset))
train_data <- heart_subset[train_index, ]
test_data <- heart_subset[-train_index, ]
# Train the SVM classifier using a linear kernel
svmfit <- svm(DEATH_EVENT ~ .,
              data = train_data,
              kernel = "linear",
              cost = 10,
              scale = FALSE)
# Perform cross-validation to choose the right cost
tune_out <- tune(svm,
                 DEATH_EVENT ~ .,
                 data = train_data,
                 kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_linear <- table(true = test_data$DEATH_EVENT, predicted = test_pred)
print(cm_linear)
# Calculate the error rate for the linear kernel
error_rate_linear <- 1 - sum(diag(cm_linear)) / sum(cm_linear)
cat("Error rate (linear kernel):", error_rate_linear, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)
# Train the SVM classifier using a radial kernel
svmfit <- svm(DEATH_EVENT ~ .,
              data = train_data,
              kernel = "radial",
              gamma = 1,
              cost = 1)
# Perform cross-validation to choose the right cost and gamma
tune_out <- tune(svm,
                 DEATH_EVENT ~ .,
                 data = train_data,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_radial <- table(true = test_data$DEATH_EVENT, predicted = test_pred)
print(cm_radial)
# Calculate the error rate for the radial kernel
error_rate_radial <- 1 - sum(diag(cm_radial)) / sum(cm_radial)
cat("Error rate (radial kernel):", error_rate_radial, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)








## WINE SVM
wine <- read.csv("~/Documents/math358/wine.csv")
pairs(wine)

# Prepare the data
wine$Wine <- as.factor(wine$Wine)
features <- c("Alcohol", "Mg")
wine_subset <- wine[, c(features, "Wine")]
wine_subset
# Split the data into training and test sets
set.seed(1)
train_index <- sample(1:nrow(wine_subset), 0.7 * nrow(wine_subset))
train_data <- wine_subset[train_index, ]
test_data <- wine_subset[-train_index, ]
# Train the SVM classifier using a linear kernel
svmfit <- svm(Wine ~ .,
              data = train_data,
              kernel = "linear",
              cost = 10,
              scale = FALSE)
# Perform cross-validation to choose the right cost
tune_out <- tune(svm,
                 Wine ~ .,
                 data = train_data,
                 kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_linear <- table(true = test_data$Wine, predicted = test_pred)
print(cm_linear)
# Calculate the error rate for the linear kernel
error_rate_linear <- 1 - sum(diag(cm_linear)) / sum(cm_linear)
cat("Error rate (linear kernel):", error_rate_linear, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)
# Train the SVM classifier using a radial kernel
svmfit <- svm(Wine ~ .,
              data = train_data,
              kernel = "radial",
              gamma = 1,
              cost = 1)
# Perform cross-validation to choose the right cost and gamma
tune_out <- tune(svm,
                 Wine ~ .,
                 data = train_data,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_radial <- table(true = test_data$Wine, predicted = test_pred)
print(cm_radial)
# Calculate the error rate for the radial kernel
error_rate_radial <- 1 - sum(diag(cm_radial)) / sum(cm_radial)
cat("Error rate (radial kernel):", error_rate_radial, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)






## FOODs SVM
food <- read.csv("~/Documents/math358/fastfood.csv")
food <- food[, !(names(food) %in% c("item", "salad"))]
food <- na.omit(food)
unique(food$restaurant)
food$restaurant <- recode(food$restaurant, 
                          "Mcdonalds" = 1,
                          "Chick Fil-A" = 3,
                          "Sonic" = 3,
                          "Arbys" = 3,
                          "Dairy Queen" = 3,
                          "Subway" = 0,
                          "Taco Bell" = 3)
food <- food[food$restaurant != 3, ]

food

# Prepare the data
food$restaurant <- as.factor(food$restaurant)
features <- c("vit_a", "cal_fat")
food_subset <- food[, c(features, "restaurant")]
food_subset
# Split the data into training and test sets
set.seed(1)
train_index <- sample(1:nrow(food_subset), 0.7 * nrow(food_subset))
train_data <- food_subset[train_index, ]
test_data <- food_subset[-train_index, ]
# Train the SVM classifier using a linear kernel
svmfit <- svm(restaurant ~ .,
              data = train_data,
              kernel = "linear",
              cost = 10,
              scale = FALSE)
# Perform cross-validation to choose the right cost
tune_out <- tune(svm,
                 restaurant ~ .,
                 data = train_data,
                 kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_linear <- table(true = test_data$restaurant, predicted = test_pred)
print(cm_linear)
# Calculate the error rate for the linear kernel
error_rate_linear <- 1 - sum(diag(cm_linear)) / sum(cm_linear)
cat("Error rate (linear kernel):", error_rate_linear, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)
# Train the SVM classifier using a radial kernel
svmfit <- svm(restaurant ~ .,
              data = train_data,
              kernel = "radial",
              gamma = 1,
              cost = 1)
# Perform cross-validation to choose the right cost and gamma
tune_out <- tune(svm,
                 restaurant ~ .,
                 data = train_data,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)))
# Get the best model
bestmod <- tune_out$best.model
# Make predictions on the test data
test_pred <- predict(bestmod, test_data)
# Display the confusion matrix
cm_radial <- table(true = test_data$restaurant, predicted = test_pred)
print(cm_radial)
# Calculate the error rate for the radial kernel
error_rate_radial <- 1 - sum(diag(cm_radial)) / sum(cm_radial)
cat("Error rate (radial kernel):", error_rate_radial, "\n")
# Plot the decision boundary of the best model
plot(bestmod, train_data)






