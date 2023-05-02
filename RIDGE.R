# Project work
#fast food data set
fastfood <- read.csv('fastfood.csv')
getwd()
setwd("C:/Users/Dana/Desktop")
getwd()


fastfood$restaurant
library(ISLR2)
library(glmnet)
library(dplyr)
library(tidyr)
fastfood=na.omit(fastfood) # remove missing observations
test=fastfood$restaurant
x = model.matrix(calories~restaurant, fastfood)[,-1]
# trim off the first column of 1s
# leaving only the predictors
# turn string variables into dummy variable
y = fastfood$calories
cv.out <- cv.glmnet(x,y,alpha=1,nfolds=5)
# alpha=1 for ridge regression, alpha=0 for Lasso
print(cv.out$cvm)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.out=glmnet(x,y,alpha=1,lambda=bestlam)
ridge.out$beta [ridge.out$beta != 0]
