head(mtcars)
summary(mtcars)

install.packages("glmnet")
library(glmnet)

#response variable
y = mtcars$hp

#matrix of predictor variables
x = data.matrix(mtcars[, c('mpg', 'wt', 'drat', 'qsec')])


#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 0)

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
