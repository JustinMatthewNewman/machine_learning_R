# Project work
#fast food data set
fastfood <- read.csv('fastfood.csv')
getwd()
setwd("C:/Users/Dana/Desktop")
getwd()


library(ISLR2)
library(glmnet)
library(dplyr)
library(tidyr)

fastfood <- read.csv('fastfood.csv')


fastfood=na.omit(fastfood) # remove missing observations

x = model.matrix(restaurant~., fastfood)[,-1]
y = fastfood$restaurant

set.seed(1)
train = fastfood %>%
  sample_frac(0.67)
test = fastfood %>%
  setdiff(train)

xtrain = x[1:nrow(train), ]
ytrain = y[1:nrow(train)]
xtest = x[(nrow(train)+1):nrow(x), ]

cv.out <- cv.glmnet(xtrain, ytrain, alpha=1, nfolds=5)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.out=glmnet(xtrain, ytrain, alpha=1, lambda=bestlam)
pred = predict(ridge.out, newx= xtest)
mean((test$restaurant-pred)^2)

#mtcars data set
library(tidyverse)
mtcars

library(ISLR2)
library(glmnet)
library(dplyr)
library(tidyr)

mtcars=na.omit(mtcars) # remove missing observations

x = model.matrix(cyl~., mtcars)[,-1]
y = mtcars$cyl

set.seed(1)
train = mtcars %>%
  sample_frac(0.67)
test = mtcars %>%
  setdiff(train)

xtrain = x[1:nrow(train), ]
ytrain = y[1:nrow(train)]
xtest = x[(nrow(train)+1):nrow(x), ]

cv.out <- cv.glmnet(xtrain, ytrain, alpha=1, nfolds=5)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.out=glmnet(xtrain, ytrain, alpha=1, lambda=bestlam)
pred = predict(ridge.out, newx= xtest)
mean((test$cyl-pred)^2)


#wine data set
wine <- read.csv('wine.csv')
wine=na.omit(wine) # remove missing observations

x = model.matrix(Wine~., wine)[,-1]
y = wine$Wine

set.seed(1)
train = wine %>%
  sample_frac(0.67)
test = wine %>%
  setdiff(train)

xtrain = x[1:nrow(train), ]
ytrain = y[1:nrow(train)]
xtest = x[(nrow(train)+1):nrow(x), ]

cv.out <- cv.glmnet(xtrain, ytrain, alpha=1, nfolds=5)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.out=glmnet(xtrain, ytrain, alpha=1, lambda=bestlam)
pred = predict(ridge.out, newx= xtest)
mean((test$Wine-pred)^2)



#spam data set
spam <- read.csv('kernlab_spam.csv')
spam=na.omit(spam) # remove missing observations

x = model.matrix(type~., spam)[,-1]
y = spam$type

set.seed(1)
train = spam %>%
  sample_frac(0.67)
test = spam %>%
  setdiff(train)

xtrain = x[1:nrow(train), ]
ytrain = y[1:nrow(train)]
xtest = x[(nrow(train)+1):nrow(x), ]

cv.out <- cv.glmnet(xtrain, ytrain, alpha=1, nfolds=5)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.out=glmnet(xtrain, ytrain, alpha=1, lambda=bestlam)
pred = predict(ridge.out, newx= xtest)
mean((test$type-pred)^2)


#heart data set

heart <- read.csv('HeartFailure.csv')
heart=na.omit(heart) # remove missing observations

x = model.matrix(DEATH_EVENT~., heart)[,-1]
y = heart$DEATH_EVENT

set.seed(1)
train = heart %>%
  sample_frac(0.67)
test = heart %>%
  setdiff(train)

xtrain = x[1:nrow(train), ]
ytrain = y[1:nrow(train)]
xtest = x[(nrow(train)+1):nrow(x), ]

cv.out <- cv.glmnet(xtrain, ytrain, alpha=1, nfolds=5)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.out=glmnet(xtrain, ytrain, alpha=1, lambda=bestlam)
pred = predict(ridge.out, newx= xtest)
mean((test$DEATH_EVENT-pred)^2)

