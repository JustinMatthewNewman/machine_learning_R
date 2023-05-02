data(mtcars)
head(mtcars)
summary(mtcars)

install.packages("glmnet")
library(glmnet)

y = mtcars$hp
x = data.matrix(mtcars[, c('mpg', 'wt', 'drat', 'qsec')])

cv_model <- cv.glmnet(x, y, alpha = 0)

best_lambda <- cv_model$lambda.min
best_lambda

plot(cv_model) 

best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

new = matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=4) 

predict(best_model, s = best_lambda, newx = new)
