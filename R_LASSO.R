# Load required libraries
library(ISLR2)
library(glmnet)
root = "https://raw.githubusercontent.com/JustinMatthewNewman/machine_learning_R/main/datasets/"


# =========== MTCARS LASSO =============

data(mtcars)

#response variable
y = mtcars$hp

#matrix of predictor variables
x = data.matrix(mtcars[, c('mpg', 'wt', 'hp', 'cyl')])


#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=4) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

# Print the predicted value
prediction

# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
test_mse












# ========= SPAM regression ===========


# Load the mtcars dataset
spam <- read.csv(paste0(root, "kernlab_spam.csv"))

# Define the response variable
y <- spam$free

# Define the matrix of predictor variables
x <- data.matrix(spam[, c('make', 'address', 'all', 'our')])

#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=4) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

# Print the predicted value
prediction

# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
test_mse







# ======== Heart regression =========


# Load the mtcars dataset
heart <- read.csv(paste0(root, "HeartFailure.csv"))

# Define the response variable
y <- heart$age

# Define the matrix of predictor variables
x <- data.matrix(heart[, c('platelets', 'creatinine_phosphokinase', 'serum_sodium', 'serum_creatinine')])

#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=4) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

# Print the predicted value
prediction

# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
sqrt(test_mse)








# ========= Wine regression ===========


# Load the mtcars dataset
wine <- read.csv(paste0(root, "wine.csv"))

# Define the response variable
y <- wine$Malic.acid

# Define the matrix of predictor variables
x <- data.matrix(wine[, c('Alcohol', 'Mg', 'Proanth', 'OD', "Hue", "Proline")])

#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5, 3, 5), nrow=1, ncol=6) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

# Print the predicted value
prediction

# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
sqrt(test_mse)










# ======== Food regression ==========


# Load the mtcars dataset
food <- read.csv(paste0(root,   "fastfood.csv"))
food <- na.omit(food)

# Define the response variable
y <- food$calories

# Define the matrix of predictor variables
x <- data.matrix(food[, c('cal_fat', 'total_fat', 'sodium', 'cholesterol', "sugar", "vit_c")])


#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5, 3, 5), nrow=1, ncol=6) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

# Print the predicted value
prediction

# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
sqrt(test_mse)


