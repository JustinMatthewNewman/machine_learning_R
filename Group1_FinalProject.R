
#       ,----.                                     ,--. 
#       '  .-./   ,--.--. ,---. ,--.,--. ,---.     /   | 
#       |  | .---.|  .--'| .-. ||  ||  || .-. |    `|  | 
#       '  '--'  ||  |   ' '-' ''  ''  '| '-' '     |  | 
#         `------' `--'    `---'  `----' |  |-'     `--' 
#                                       `--'             
#        ,------.,--.                ,--.                  
#        |  .---'`--',--,--,  ,--,--.|  |                  
#        |  `--, ,--.|      \' ,-.  ||  |                  
#        |  |`   |  ||  ||  |\ '-'  ||  |                  
#        `--'    `--'`--''--' `--`--'`--'                  
#        ,------.                ,--.               ,--.   
#        |  .--. ',--.--. ,---.  `--' ,---.  ,---.,-'  '-. 
#        |  '--' ||  .--'| .-. | ,--.| .-. :| .--''-.  .-' 
#        |  | --' |  |   ' '-' ' |  |\   --.\ `--.  |  |   
#        `--'     `--'    `---'.-'  / `----' `---'  `--'   o
#                              '---'                       
#
#        MATH 358 FINAL_PROJECT - May 2nd 2023.
#
#   Authors: Justin Newman, Hannah Dalakate Phommachanthone
#                       Grace Dwindel and Dana Stufin.
# 
#   Instructor: Dr. Chen
# ==========================================================================
# ======================    DATASETS LOCATION   ==========================
# ==========================================================================
str1 <- "https://raw.githubusercontent.com/"
str2 <-  "JustinMatthewNewman/machine_learning_R/main/datasets/"
root = paste0(str1,str2)
# ==========================================================================

# Installations and packages

#BOOST
#install.packages("readr")
#install.packages("pROC")
#install.packages("dplyr")
#install.packages("gbm")
#
##KNN
#install.packages("readr")
#install.packages("pROC")
#install.packages("dplyr")
#install.packages("gbm")
#
##TREE
#install.packages("tree")
#
##LASSO
#install.packages("ISLR2")
#install.packages("glmnet")
#
##Ridge
#install.packages("ISLR2")
#install.packages("glmnet")
#install.packages("dplyr")
#install.packages("tidyr")
#
##FOREST
#install.packages("tree")
#install.packages("ISLR2") # find data sets
#install.packages("dplyr") # split data
#install.packages("ggplot2") # plot
#install.packages("randomForest")
#
##NETWORK
#install.packages("neuralnet")


#         USE (ctrl + F), or (cmd + F) on mac to quickly navigate.
#
#   OPTIONS: ( BOOST_FIND, KNN_FIND, TREE_FIND, 
#              RIDGE_FIND, LASSO_FIND, SVM_FIND,
#              NETWORK_FIND, FOREST_FIND )
 
# ==========================================================================
# ,-----.   ,-----.  ,-----.  ,---. ,--------. 
# |  |) /_ '  .-.  ''  .-.  ''   .-''--.  .--' 
# |  .-.  \|  | |  ||  | |  |`.  `-.   |  |    
# |  '--' /'  '-'  ''  '-'  '.-'    |  |  |    
# `------'  `-----'  `-----' `-----'   `--'o
# BOOST_FIND
# ==========================================================================

library(readr)
library(pROC)
library(dplyr)
library(gbm)


# ========== mtcars boosting regression ==========

# Commented out because N was too small to run boosting.

# mtcars=mtcars
# mtcars
# set.seed(1)
# 
# mtcars_train = mtcars %>%
#   sample_frac(0.5)
# mtcars_test = mtcars %>%
#   setdiff(mtcars_train)
# 
# set.seed(1)
# boost_mtcars = gbm(am~., 
#                    data=mtcars_train, 
#                    distribution = "gaussian",
#                    n.trees=15,
#                    interaction.depth = 5)
# summary(boost_mtcars)
# 
# boost_estimate = predict(boost_mtcars,
#                          newdata= mtcars_test,
#                          n.trees = 5000)
# 
# mean((boost_estimate - mtcars_test$am)^2)
# 
# boost_mtcars2 = gbm(am~., data = mtcars_train,
#                     distribution = "gaussian",
#                     n.trees = 5000,
#                     interaction.depth = 4,
#                     shrinkage = 0.01,
#                     verbose = F)
# 
# boost_estimate2 = predict(boost_mtcars2, newdata = mtcars_test, n.trees = 5000)
# mean((boost_estimate2 - mtcars_test$am)^2)
# 
# # ========== mtcars boosting on classification ==========
# 
# mtcars$am = as.numeric(mtcars$am)-1
# 
# mtcars_train = mtcars %>%
#   sample_frac(0.67)
# mtcars_test = mtcars %>%
#   setdiff(mtcars_train)
# 
# out=gbm(am~., data=mtcars_train, distribution="bernoulli", n.trees=5000)
# 
# pred=predict(out, newdata=mtcars_test, n.trees=5000)
# 
# predProb = exp(pred) / (1+exp(pred))
# 
# auc(mtcars_test$am, predProb)
# 
# 
# 
# # ========== KERN boosting regression ==========
# 
# url = "kernlab_spam.csv"
# kern = read.csv(paste0(root, url))
# kern
# set.seed(1)
# 
# kern_train = kern %>%
#   sample_frac(0.5)
# kern_test = kern %>%
#   setdiff(kern_train)
# 
# set.seed(1)
# boost_kern = gbm(free~., 
#                  data=kern_train, 
#                  distribution = "gaussian",
#                  n.trees=5000,
#                  interaction.depth = 4)
# summary(boost_kern)
# 
# boost_estimate = predict(boost_kern,
#                          newdata= kern_test,
#                          n.trees = 5000)
# 
# mean((boost_estimate - kern_test$free)^2)
# 
# boost_kern2 = gbm(free~., data = kern_train,
#                   distribution = "gaussian",
#                   n.trees = 5000,
#                   interaction.depth = 4,
#                   shrinkage = 0.01,
#                   verbose = F)
# 
# boost_estimate2 = predict(boost_kern2, newdata = kern_test, n.trees = 5000)
# mean((boost_estimate2 - kern_test$free)^2)
# 
# # ========== KERN boosting on classification ==========
# 
# set.seed(1)
# 
# kern <- kern %>%
#   mutate(type= ifelse(type = "spam", 1, 0))
# kern$type = as.numeric(kern$type)-1
# 
# kern_train = kern %>%
#   sample_frac(0.67)
# kern_test = kern %>%
#   setdiff(kern_train)
# 
# out=gbm(type~., data=kern_train, distribution="bernoulli", n.trees=5000)
# out
# 
# pred=predict(out, newdata=kern_test, n.trees=5000)
# pred
# 
# predProb = exp(pred) / (1+exp(pred))
# predProb
# 
# auc(kern_test$type, predProb)
# 
# 

# ========== Heart Failure boosting regression ==========

url = "HeartFailure.csv"
Heart = read.csv(paste0(root, url))
set.seed(1)
Heart_train = Heart %>%
  sample_frac(0.5)
Heart_test = Heart %>%
  setdiff(Heart_train)

set.seed(1)
boost_Heart = gbm(platelets~., 
                  data=Heart_train, 
                  distribution = "gaussian",
                  n.trees=5000,
                  interaction.depth = 4)
summary(boost_Heart)

boost_estimate = predict(boost_Heart,
                         newdata= Heart_test,
                         n.trees = 5000)

mean((boost_estimate - Heart_test$platelets)^2)

boost_Heart2 = gbm(platelets~., data = Heart_train,
                   distribution = "gaussian",
                   n.trees = 5000,
                   interaction.depth = 4,
                   shrinkage = 0.01,
                   verbose = F)

boost_estimate2 = predict(boost_Heart2, newdata = Heart_test, n.trees = 5000)
mean((boost_estimate2 - Heart_test$platelets)^2)

# ========== Heart Failure boosting on Classification ==========

# Heart$DEATH_EVENT = as.numeric(Heart$DEATH_EVENT)-1
# 
# set.seed(1)
# Heart_train = Heart %>%
#   sample_frac(0.67)
# Heart_test = Heart %>%
#   setdiff(Heart_train)
# 
# out=gbm(DEATH_EVENT~., data=Heart_train, distribution="bernoulli", n.trees=5000)
# out
# 
# pred=predict(out, newdata=Heart_test, n.trees=5000)
# pred
# 
# predProb = exp(pred) / (1+exp(pred))
# predProb
# 
# auc(Heart_test$DEATH_EVENT, predProb)

# ========== Wine Dataset ==========

url = "wine.csv"
WineDat=read.csv(paste0(root, url))

#Wine boosting regression

set.seed(1)

wine_train = WineDat %>%
  sample_frac(0.5)
wine_test = WineDat %>%
  setdiff(wine_train)

set.seed(1)
boost_wine = gbm(Malic.acid~., 
                 data=wine_train, 
                 distribution = "gaussian",
                 n.trees=5000,
                 interaction.depth = 4)
summary(boost_wine)

boost_estimate = predict(boost_wine,
                         newdata= wine_test,
                         n.trees = 5000)

mean((boost_estimate - wine_test$Malic.acid)^2)

boost_wine2 = gbm(Malic.acid~., data = wine_train,
                  distribution = "gaussian",
                  n.trees = 5000,
                  interaction.depth = 4,
                  shrinkage = 0.01,
                  verbose = F)

boost_estimate2 = predict(boost_wine2, newdata = wine_test, n.trees = 5000)
mean((boost_estimate2 - wine_test$Malic.acid)^2)

# ========== Wine boosting on classification ==========

# WineDat$Wine = as.numeric(WineDat$Wine)-1
# 
# wine_train = WineDat %>%
#   sample_frac(0.67)
# wine_test = WineDat %>%
#   setdiff(wine_train)
# 
# out=gbm(Wine~., data=wine_train, distribution="bernoulli", n.trees=5000)
# out
# 
# pred=predict(out, newdata=wine_test, n.trees=5000)
# pred
# 
# predProb = exp(pred) / (1+exp(pred))
# predProb
# 
# auc(wine_test$Wine, predProb)
# 


# ========== Fast Food boosting Regression ==========

fastfood <- read.csv(paste0(root, "fastfood.csv"))
fastfood <- fastfood[, !(names(fastfood) %in% c("item", "salad"))]
fastfood <- na.omit(fastfood)
unique(fastfood$restaurant)
fastfood$restaurant <- recode(fastfood$restaurant, 
                          "Mcdonalds" = 1,
                          "Chick Fil-A" = 3,
                          "Sonic" = 3,
                          "Arbys" = 3,
                          "Dairy Queen" = 3,
                          "Subway" = 0,
                          "Taco Bell" = 3)
fastfood <- fastfood[fastfood$restaurant != 3, ]

fastfood

# Prepare the data
fastfood$restaurant <- as.factor(fastfood$restaurant)

set.seed(1)

Food_train = fastfood %>%
  sample_frac(0.5)
Food_test = fastfood %>%
  setdiff(Food_train)

set.seed(1)
boost_Food = gbm(calories~., 
                 data=Food_train, 
                 distribution = "gaussian",
                 n.trees=5000,
                 interaction.depth = 4)
summary(boost_Food)

boost_estimate = predict(boost_Food,
                         newdata= Food_test,
                         n.trees = 5000)

mean((boost_estimate - Food_test$calories)^2)

boost_Food2 = gbm(calories~., data = Food_train,
                  distribution = "gaussian",
                  n.trees = 5000,
                  interaction.depth = 4,
                  shrinkage = 0.01,
                  verbose = F)

boost_estimate2 = predict(boost_Food2, newdata = Food_test, n.trees = 5000)
mean((boost_estimate2 - Food_test$calories)^2)

# ========== Fast food boosting on classification ==========

fastfood$restaurant = as.numeric(fastfood$restaurant)-1

Food_train = fastfood %>%
  sample_frac(0.67)
Food_test = fastfood %>%
  setdiff(Food_train)

out=gbm(restaurant~., data=Food_train, distribution="bernoulli", n.trees=5000)
out

pred=predict(out, newdata=Food_test, n.trees=5000)
pred

predProb = exp(pred) / (1+exp(pred))
predProb

auc(Food_test$restaurant, predProb)









# ==========================================================================
# --------.                      
# --.  .--',--.--. ,---.  ,---.  
#   |  |   |  .--'| .-. :| .-. : 
#   |  |   |  |   \   --.\   --. 
#   `--'   `--'    `----' `----' o
# TREE_FIND
# ==========================================================================


library(tree)


# ========== MTCARS TREE ==========

data(mtcars)
mtcars$cyl = as.factor(mtcars$cyl)
tree_out = tree(cyl ~ ., data=mtcars)
summary(tree_out)
plot(tree_out)
text(tree_out, pretty=0)
cv_tree = cv.tree(tree_out, FUN=prune.misclass)
plot(cv_tree$size, cv_tree$dev, type="b")
prune_out = prune.misclass(tree_out, best=2)
plot(prune_out)
text(prune_out)
summary(prune_out)
set.seed(1)
train_indices = sample(1:nrow(mtcars), round(0.7 * nrow(mtcars)))
train = mtcars[train_indices, ]
test = mtcars[-train_indices, ]
tree_out = tree(cyl ~ ., data=train)
tree_pred = predict(tree_out, test, type="class")
table(tree_pred, test$cyl)
mean(test$cyl != tree_pred)



# ========== SPAM EMAIL TREE ==========
spam <- read.csv(paste0(root, "kernlab_spam.csv"))
spam$type <- as.factor(spam$type)
tree_out = tree(type ~ ., data=spam)
summary(tree_out)
plot(tree_out)
text(tree_out, pretty=0)
cv_tree = cv.tree(tree_out, FUN=prune.misclass)
plot(cv_tree$size, cv_tree$dev, type="b")
prune_out = prune.misclass(tree_out, best=8)
plot(prune_out)
text(prune_out)
summary(prune_out)
set.seed(1)
train_indices = sample(1:nrow(spam), round(0.7 * nrow(spam)))
train = spam[train_indices, ]
test = spam[-train_indices, ]
tree_out = tree(type ~ ., data=train)
tree_pred = predict(tree_out, test, type="class")
table(tree_pred, test$type)
mean(test$type != tree_pred)
misclass_rate <- mean(test$type != tree_pred)
cat("Misclassification rate:", misclass_rate, "\n")




# ========== Heart Tree ==========

heart <- read.csv(paste0(root, "HeartFailure.csv"))
heart$DEATH_EVENT <- as.factor(heart$DEATH_EVENT)
tree_out = tree(DEATH_EVENT ~ ., data=heart)
summary(tree_out)
plot(tree_out)
text(tree_out, pretty=0)
cv_tree = cv.tree(tree_out, FUN=prune.misclass)
plot(cv_tree$size, cv_tree$dev, type="b")
prune_out = prune.misclass(tree_out, best=8)
plot(prune_out)
text(prune_out)
summary(prune_out)
set.seed(1)
train_indices = sample(1:nrow(heart), round(0.7 * nrow(heart)))
train = heart[train_indices, ]
test = heart[-train_indices, ]
tree_out = tree(DEATH_EVENT ~ ., data=train)
tree_pred = predict(tree_out, test, type="class")
table(tree_pred, test$DEATH_EVENT)
mean(test$DEATH_EVENT != tree_pred)
misclass_rate <- mean(test$DEATH_EVENT != tree_pred)
cat("Misclassification rate:", misclass_rate, "\n")


#  ========== FastFood tree ==========
fastfood <- read.csv(paste0(root, "fastfood.csv"))
fastfood$restaurant <- as.factor(fastfood$restaurant)
fastfood <- na.omit(fastfood)
fastfood <- fastfood[, -2]
fastfood <- fastfood[, -16]

tree_out = tree(restaurant ~ ., data=fastfood)
summary(tree_out)
plot(tree_out)
text(tree_out, pretty=0)
cv_tree = cv.tree(tree_out, FUN=prune.misclass)
plot(cv_tree$size, cv_tree$dev, type="b")

prune_out = prune.misclass(tree_out, best=8)
plot(prune_out)
text(prune_out)
summary(prune_out)
set.seed(1)
train_indices = sample(1:nrow(fastfood), round(0.7 * nrow(fastfood)))
train = fastfood[train_indices, ]
test = fastfood[-train_indices, ]
tree_out = tree(restaurant ~ ., data=train)
tree_pred = predict(tree_out, test, type="class")
table(tree_pred, test$restaurant)
mean(test$restaurant != tree_pred)
misclass_rate <- mean(test$restaurant != tree_pred)
cat("Misclassification rate:", misclass_rate, "\n")










# ==========================================================================
# ,--.                                 
# |  |    ,--,--. ,---.  ,---.  ,---.  
# |  |   ' ,-.  |(  .-' (  .-' | .-. | 
# |  '--.\ '-'  |.-'  `).-'  `)' '-' ' 
# `-----' `--`--'`----' `----'  `---'  o
# LASSO_FIND
# ==========================================================================



# Load required libraries
library(ISLR2)
library(glmnet)



# =========== MTCARS LASSO =============

data(mtcars)

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
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=4) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)


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
cv_model <- cv.glmnet(x, y, alpha = 0)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=4) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

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
cv_model <- cv.glmnet(x, y, alpha = 0)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=4) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
test_mse








# ========= Wine regression ===========


# Load the mtcars dataset
wine <- read.csv(paste0(root, "wine.csv"))

# Define the response variable
y <- wine$Malic.acid

# Define the matrix of predictor variables
x <- data.matrix(wine[, c('Alcohol', 'Mg', 'Proanth', 'OD', "Hue", "Proline")])

#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 0)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5, 3, 5), nrow=1, ncol=6) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)


# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
test_mse










# ======== Food regression ==========


# Load the mtcars dataset
food <- read.csv(paste0(root,   "fastfood.csv"))
food <- na.omit(food)

# Define the response variable
y <- food$calories

# Define the matrix of predictor variables
x <- data.matrix(food[, c('cal_fat', 'total_fat', 'sodium', 'cholesterol', "sugar", "vit_c")])


#k-fold cross-validation
cv_model <- cv.glmnet(x, y, alpha = 0)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda


#plot of test MSE
plot(cv_model) 

#coefficients of best model
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

#new observation
new = matrix(c(24, 2.5, 3.5, 18.5, 3, 5), nrow=1, ncol=6) 

#lasso regression model to predict response value
predict(best_model, s = best_lambda, newx = new)

# Calculate the test MSE
test_mse <- cv_model$cvm[cv_model$lambda == best_lambda]
test_mse









# ==========================================================================
# ,------. ,--.   ,--.               
# |  .--. '`--' ,-|  | ,---.  ,---.  
# |  '--'.',--.' .-. || .-. || .-. : 
# |  |\  \ |  |\ `-' |' '-' '\   --. 
# `--' '--'`--' `---' .`-  /  `----' o
# RIDGE_FIND
# ==========================================================================

# Ridge Regression work by DANA Stufin.

library(ISLR2)
library(glmnet)
library(dplyr)
library(tidyr)



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




#spam data set
spam <- read.csv(paste0(root, 'kernlab_spam.csv'))
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

heart <- read.csv(paste0(root, 'HeartFailure.csv'))
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






#wine data set
wine <- read.csv(paste0(root, 'wine.csv'))
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








fastfood <- read.csv(paste0(root,  'fastfood.csv'))

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













# ==========================================================================
# ,------. ,-----. ,------. ,------. ,---. ,--------. 
# |  .---''  .-.  '|  .--. '|  .---''   .-''--.  .--' 
# |  `--, |  | |  ||  '--'.'|  `--, `.  `-.   |  |    
# |  |`   '  '-'  '|  |\  \ |  `---..-'    |  |  |    
# `--'     `-----' `--' '--'`------'`-----'   `--'   
# FOREST_FIND
# ==========================================================================


library(tree)
library(ISLR2) # find data sets
library(dplyr) # split data
library(ggplot2) # plot
library(randomForest)






# ========== MTCARS FOREST ==========

data(mtcars)
mtcars$cyl = as.numeric(mtcars$cyl)


set.seed(1)
mtcars_train = mtcars %>%
  sample_frac(.5)
mtcars_test = mtcars %>%
  setdiff(mtcars_train) # randomly split data into train and test data
bag_mtcars = randomForest(cyl~.,
                          data = mtcars_train,
                          mtry = 4,
                          importance = TRUE)
bag_mtcars # choose mtry=13 (use all predictors) so this is bagging
bagged_estimate = predict(bag_mtcars,
                          newdata = mtcars_test)
ggplot() +
  geom_point(aes(x = mtcars_test$cyl, y = bagged_estimate)) +
  geom_abline()
# check the performance of bagging using the scatter plot, the line is 45 degrees line.
bag_mtcars_25_trees = randomForest(cyl~., data = mtcars_train, mtry =
                                     4, ntree = 25)
bagged_estimate_25_trees = predict(bag_mtcars_25_trees, newdata =
                                     mtcars_test)
mean((bagged_estimate_25_trees - mtcars_test$cyl)^2) # mean square prediction error
#ntree = number of trees
rf_mtcars = randomForest(cyl~.,
                         data = mtcars_train,
                         mtry = 4,
                         importance = TRUE)
random_forest_estimate = predict(rf_mtcars,
                                 newdata = mtcars_test)
mean((random_forest_estimate - mtcars_test$cyl)^2) # m=6, default
ntree=500
importance(rf_mtcars)
# 1st measure: mean decrease in prediciton accuracy if excluded.
# 2nd measure total decrease in node impurity over this variable.
varImpPlot(rf_mtcars)










# ========== SPAM FOREST ==========

spam <- read.csv(paste0(root, "kernlab_spam.csv"))
# Recode the 'type' column
spam <- spam %>%
  mutate(type = ifelse(type == "spam", 1, 0))


set.seed(1)
spam_train = spam %>%
  sample_frac(.5)
spam_test = spam %>%
  setdiff(spam_train) # randomly split data into train and test data
bag_spam = randomForest(type~.,
                        data = spam_train,
                        mtry = 4,
                        importance = TRUE)
bag_spam # choose mtry=13 (use all predictors) so this is bagging
bagged_estimate = predict(bag_spam,
                          newdata = spam_test)
ggplot() +
  geom_point(aes(x = spam_test$type, y = bagged_estimate)) +
  geom_abline()
# check the performance of bagging using the scatter plot, the line is 45 degrees line.
bag_spam_25_trees = randomForest(type~., data = spam_train, mtry =
                                   4, ntree = 25)
bagged_estimate_25_trees = predict(bag_spam_25_trees, newdata =
                                     spam_test)
mean((bagged_estimate_25_trees - spam_test$type)^2) # mean square prediction error
#ntree = number of trees
rf_spam = randomForest(type~.,
                       data = spam_train,
                       mtry = 4,
                       importance = TRUE)
random_forest_estimate = predict(rf_spam,
                                 newdata = spam_test)
mean((random_forest_estimate - spam_test$type)^2) # m=6, default
ntree=500
importance(rf_spam)
# 1st measure: mean decrease in prediciton accuracy if excluded.
# 2nd measure total decrease in node impurity over this variable.
varImpPlot(rf_spam)














# ========== HEART FOREST ==========

heart <- read.csv(paste0(root, "HeartFailure.csv"))
heart

set.seed(1)
heart_train = heart %>%
  sample_frac(.5)
heart_test = heart %>%
  setdiff(heart_train) # randomly split data into train and test data
bag_heart = randomForest(DEATH_EVENT~.,
                         data = heart_train,
                         mtry = 4,
                         importance = TRUE)
bag_heart # choose mtry=13 (use all predictors) so this is bagging
bagged_estimate = predict(bag_heart,
                          newdata = heart_test)
ggplot() +
  geom_point(aes(x = heart_test$DEATH_EVENT, y = bagged_estimate)) +
  geom_abline()
# check the performance of bagging using the scatter plot, the line is 45 degrees line.
bag_heart_25_trees = randomForest(DEATH_EVENT~., data = heart_train, mtry =
                                    4, ntree = 25)
bagged_estimate_25_trees = predict(bag_heart_25_trees, newdata =
                                     heart_test)
mean((bagged_estimate_25_trees - heart_test$DEATH_EVENT)^2) # mean square prediction error
#ntree = number of trees
rf_heart = randomForest(DEATH_EVENT~.,
                        data = heart_train,
                        mtry = 4,
                        importance = TRUE)
random_forest_estimate = predict(rf_heart,
                                 newdata = heart_test)
mean((random_forest_estimate - heart_test$DEATH_EVENT)^2) # m=6, default
ntree=500
importance(rf_heart)
# 1st measure: mean decrease in prediciton accuracy if excluded.
# 2nd measure total decrease in node impurity over this variable.
varImpPlot(rf_heart)










# ========== Wine FOREST ==========

wine <- read.csv(paste0(root, "wine.csv"))
wine

set.seed(1)
wine_train = wine %>%
  sample_frac(.5)
wine_test = wine %>%
  setdiff(wine_train) # randomly split data into train and test data
bag_wine = randomForest(Wine~.,
                        data = wine_train,
                        mtry = 4,
                        importance = TRUE)
bag_wine # choose mtry=13 (use all predictors) so this is bagging
bagged_estimate = predict(bag_wine,
                          newdata = wine_test)
ggplot() +
  geom_point(aes(x = wine_test$Wine, y = bagged_estimate)) +
  geom_abline()
# check the performance of bagging using the scatter plot, the line is 45 degrees line.
bag_wine_25_trees = randomForest(Wine~., data = wine_train, mtry =
                                   4, ntree = 25)
bagged_estimate_25_trees = predict(bag_wine_25_trees, newdata =
                                     wine_test)
mean((bagged_estimate_25_trees - wine_test$Wine)^2) # mean square prediction error
#ntree = number of trees
rf_wine = randomForest(Wine~.,
                       data = wine_train,
                       mtry = 4,
                       importance = TRUE)
random_forest_estimate = predict(rf_wine,
                                 newdata = wine_test)
mean((random_forest_estimate - wine_test$Wine)^2) # m=6, default
ntree=500
importance(rf_wine)
# 1st measure: mean decrease in prediciton accuracy if excluded.
# 2nd measure total decrease in node impurity over this variable.
varImpPlot(rf_wine)









# ========== food FOREST ==========

food <- read.csv(paste0(root,"fastfood.csv"))
food <- na.omit(food)
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
set.seed(1)
food_train = food %>%
  sample_frac(.5)
food_test = food %>%
  setdiff(food_train) # randomly split data into train and test data
bag_food = randomForest(restaurant~.,
                        data = food_train,
                        mtry = 4,
                        importance = TRUE)
bag_food # choose mtry=13 (use all predictors) so this is bagging
bagged_estimate = predict(bag_food,
                          newdata = food_test)
ggplot() +
  geom_point(aes(x = food_test$restaurant, y = bagged_estimate)) +
  geom_abline()
# check the performance of bagging using the scatter plot, the line is 45 degrees line.
bag_food_25_trees = randomForest(restaurant~., data = food_train, mtry =
                                   4, ntree = 25)
bagged_estimate_25_trees = predict(bag_food_25_trees, newdata =
                                     food_test)
mean((bagged_estimate_25_trees - food_test$restaurant)^2) # mean square prediction error
#ntree = number of trees
rf_food = randomForest(restaurant~.,
                       data = food_train,
                       mtry = 4,
                       importance = TRUE)
random_forest_estimate = predict(rf_food,
                                 newdata = food_test)
mean((random_forest_estimate - food_test$restaurant)^2) # m=6, default
ntree=500
importance(rf_food)
# 1st measure: mean decrease in prediciton accuracy if excluded.
# 2nd measure total decrease in node impurity over this variable.
varImpPlot(rf_food)











# ==========================================================================
# ,--.                     
# |  |,-. ,--,--, ,--,--,  
# |     / |      \|      \ 
# |  \  \ |  ||  ||  ||  | 
# `--'`--'`--''--'`--''--' o
# KNN_FIND
# ==========================================================================

## KNN FILE BY Hannah Dalakate Phommachanthone

#========== mtcars data set ==========

mtcars=mtcars
mtcars

# ========== KNN Classification ==========

set.seed(1)

CarX=model.matrix(am~., data=mtcars)[,-1]
CarX=scale(CarX)
idx2=sample(1:nrow(mtcars))
npart=floor(0.5*nrow(mtcars))
idx=idx2[1:npart]

CarX_train = CarX[idx,]
CarY_train = mtcars$am[idx]

CarX_test = CarX[-idx,]
CarY_test = mtcars$am[-idx]

predicted = knn(train = CarX_train, test = CarX_test, cl= CarY_train, k=5)
mean(CarY_test != predicted)

# Using test error to find optimal K

set.seed(1)

n=nrow(mtcars)
npart=floor(0.5*n)

M=15
miserror=rep(0, M)

for (i in 1: M){
  pred=knn(train=CarX_train,test=CarX_test,cl=CarY_train,k=i)
  miserror[i]=mean(pred != CarY_test)
}
which(miserror==min(miserror))

# ========== KNN Regression ==========

set.seed(1)

predout=knn.reg(train=CarX_train, test=CarX_test, y=CarY_train, k=5)
pred=predout$pred
mean((CarY_test - pred)^2)

#========== KERN Dataset ==========

kern = read.csv(paste0(root, "kernlab_spam.csv"))
kern

# ========== KNN Classification ==========

set.seed(1)

KernX=model.matrix(type~., data=kern)[,-1]
KernX=scale(KernX)
idx2=sample(1:nrow(kern))
npart=floor(0.5*nrow(kern))
idx=idx2[1:npart]

KernX_train = KernX[idx,]
KernY_train = kern$type[idx]

KernX_test = KernX[-idx,]
KernY_test = kern$type[-idx]

predicted = knn(train = KernX_train, test = KernX_test, cl= KernY_train, k=5)
mean(KernY_test != predicted)

#Using test error to find optimal K

set.seed(1)

n=nrow(kern)
npart=floor(0.5*n)

M=20
miserror=rep(0, M)

for (i in 1: M){
  pred=knn(train=KernX_train,test=KernX_test,cl=KernY_train,k=i)
  miserror[i]=mean(pred != KernY_test)
}
which(miserror==min(miserror))

# ========== KNN Regression ==========

set.seed(1)

predout=knn.reg(train=KernX_train, test=KernX_test, y=KernY_train, k=5)
pred=predout$pred
mean((KernY_test - pred)^2)

# ========== Heart Failure Dataset ==========

Heart=read.csv(paste0(root, "HeartFailure.csv"))
Heart

# ========== KNN Classification ==========

set.seed(1)

HeartX=model.matrix(DEATH_EVENT~., data=Heart)[,-1]
HeartX=scale(HeartX)
idx2=sample(1:nrow(Heart))
npart=floor(0.5*nrow(Heart))
idx=idx2[1:npart]

HeartX_train = HeartX[idx,]
HeartY_train = Heart$DEATH_EVENT[idx]

HeartX_test = HeartX[-idx,]
HeartY_test = Heart$DEATH_EVENT[-idx]

predicted = knn(train = HeartX_train, test = HeartX_test, cl= HeartY_train, k=5)
mean(HeartY_test != predicted)

#Using test error to find optimal K

set.seed(1)

M=20
miserror=rep(0, M)

for (i in 1: M){
  pred=knn(train=HeartX_train,test=HeartX_test,cl=HeartY_train,k=i)
  miserror[i]=mean(pred != HeartY_test)
}
which(miserror==min(miserror))



# ========== KNN Regression ==========

set.seed(1)

predout=knn.reg(train=HeartX_train, test=HeartX_test, y=HeartY_train, k=5)
pred=predout$pred
mean((HeartY_test - pred)^2)

#Wine Dataset

wine=read.csv(paste0(root, "wine.csv"))
wine

# ========== KNN Classification ==========

set.seed(1)

WineX=model.matrix(Wine~., data=wine)[,-1]
WineX=scale(WineX)
idx2=sample(1:nrow(wine))
npart=floor(0.5*nrow(wine))
idx=idx2[1:npart]

WineX_train = WineX[idx,]
WineY_train = wine$Wine[idx]

WineX_test = WineX[-idx,]
WineY_test = wine$Wine[-idx]

predicted = knn(train = WineX_train, test = WineX_test, cl= WineY_train, k=5)
mean(WineY_test != predicted)

#Using test error to find optimal K

set.seed(1)

M=20
miserror=rep(0, M)

for (i in 1: M){
  pred=knn(train=WineX_train,test=WineX_test,cl=WineY_train,k=i)
  miserror[i]=mean(pred != WineY_test)
}
which(miserror==min(miserror))

# ========== KNN Regression ==========

set.seed(1)

predout=knn.reg(train=WineX_train, test=WineX_test, y=WineY_train, k=5)
pred=predout$pred
mean((WineY_test - pred)^2)

#Fast food data set

fastfood=read.csv(paste0(root, "fastfood.csv"))
fastfood
fastfood$restaurant <- as.factor(fastfood$restaurant)
fastfood <- na.omit(fastfood)
fastfood <- fastfood[,-2]
fastfood <- fastfood[,-16]

fastfood$restaurant <- recode(fastfood$restaurant, 
                              "Mcdonalds" = 1,
                              "Chick Fil-A" = 3,
                              "Sonic" = 3,
                              "Arbys" = 3,
                              "Dairy Queen" = 3,
                              "Subway" = 0,
                              "Taco Bell" = 3)
fastfood <- fastfood[fastfood$restaurant != 3, ]

fastfood$restaurant

# ========== KNN Classification ==========

set.seed(1)

FoodX=model.matrix(restaurant~., data=fastfood)[,-1]
FoodX=scale(FoodX)
idx2=sample(1:nrow(fastfood))
npart=floor(0.5*nrow(fastfood))
idx=idx2[1:npart]

FoodX_train = FoodX[idx,]
FoodY_train = fastfood$restaurant[idx]
nrow(FoodX_train)
length(FoodY_train)


FoodX_test = FoodX[-idx,]
FoodY_test = fastfood$restaurant[-idx]

#stops working here
predicted = knn(train = FoodX_train, test = FoodX_test, cl= FoodY_train, k=5)
mean(FoodY_test != predicted)

#Using test error to find optimal K

set.seed(1)

n=nrow(fastfood)
npart=floor(0.5*n)

M=20
miserror=rep(0, M)

for (i in 1: M){
  pred=knn(train=FoodX_train,test=FoodX_test,cl=FoodY_train,k=i)
  miserror[i]=mean(pred != FoodY_test)
}
which(miserror==min(miserror))

# ========== KNN Regression ==========

set.seed(1)

predout=knn.reg(train=FoodX_train, test=FoodX_test, y=FoodY_train, k=5)
pred=predout$pred
mean((FoodY_test - pred)^2)









# ==========================================================================
# ,---.,--.   ,--.,--.   ,--. 
# '   .-'\  `.'  / |   `.'   | 
# `.  `-. \     /  |  |'.'|  | 
# .-'    | \   /   |  |   |  | 
# `-----'   `-'    `--'   `--' 
# SVM_FIND
# ==========================================================================





library(e1071)
library(ggplot2)
library(dplyr)


par(mfrow = c(1, 1)) # Default value for mf

## ========== SVM MTCARS ==========

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















## ========== SPAM SVM ==========
spam <- read.csv(paste0(root, "kernlab_spam.csv"))
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











##  ========== HEART SVM ==========
heart <- read.csv(paste0(root, "HeartFailure.csv"))
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








## ========== WINE SVM ==========
wine <- read.csv(paste0(root, "wine.csv"))
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






## ========== FOODs SVM ==========
food <- read.csv(paste0(root, "fastfood.csv"))
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





# ==========================================================================

#,--.  ,--.,------.,--. ,--.,------.   ,---.  ,--.   ,--.  ,--.,------.,--------
#|  ,'.|  ||  .---'|  | |  ||  .--. ' /  O  \ |  |   |  ,'.|  ||  .---''--.  .--
#|  |' '  ||  `--, |  | |  ||  '--'.'|  .-.  ||  |   |  |' '  ||  `--,    |  |    
#|  | `   ||  `---.'  '-'  '|  |\  \ |  | |  ||  '--.|  | `   ||  `---.   |  |    
#`--'  `--'`------' `-----' `--' '--'`--' `--'`-----'`--'  `--'`------'   `--'    
# NETWORK_FIND                                                                                 
# ==========================================================================




#install.packages("neuralnet")

#IRIS NN EXAMPLE




library(neuralnet)
data(iris)
iris$setosa <- ifelse(iris$Species == "setosa", 1, 0)
set.seed(1)
train_idx <- sample(1:nrow(iris), nrow(iris)*0.7)
train_data <- iris[train_idx, ]
test_data <- iris[-train_idx, ]
model <- neuralnet(setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                   train_data, 
                   hidden = 3)
plot(model)
test_pred <- compute(model, test_data[,1:4])
test_pred <- ifelse(test_pred$net.result > 0.5, 1, 0)
accuracy <- sum(test_pred == test_data$setosa) / nrow(test_data)
accuracy
library(caret)
NN = neuralnet (Species ~., iris, hidden = 3 )
# plot neural network
plot (NN)



# DATA MTCARS NN
data(mtcars)
mtcars$am <- as.factor(mtcars$am)
set.seed(1)
train_idx <- sample(1:nrow(mtcars), nrow(mtcars)*0.7)
train_data <- mtcars[train_idx, ]
test_data <- mtcars[-train_idx, ]
library(neuralnet)
model <- neuralnet(am ~ mpg + cyl + disp + hp + drat + wt + qsec + vs + carb, train_data, hidden = c(5, 3))
plot(model)
test_pred <- predict(model, test_data[,1:9])
test_pred <- max.col(test_pred)
accuracy <- sum(test_pred == test_data$am) / nrow(test_data)
accuracy
test_pred





#SPAM DATA NN
spamdata <- read.csv(paste0(root, "kernlab_spam.csv"))
spamdata$type <- as.factor(spamdata$type)
set.seed(1)
train_idx <- sample(1:nrow(spamdata), nrow(spamdata)*0.7)
train_data <- spamdata[train_idx, ]
test_data <- spamdata[-train_idx, ]
model <- neuralnet(type ~ capitalTotal + charExclamation + your, train_data, hidden = 3)
plot(model)
test_pred <- compute(model, test_data[,1:3])
test_pred_result <- test_pred$net.result
test_pred <- max.col(test_pred_result)
accuracy <- sum(test_pred == test_data$DEATH_EVENT) / nrow(test_data)
accuracy
test_pred



#HEART FAILURE DATA NN
heart <- read.csv(paste0(root,  "HeartFailure.csv"))
heart$DEATH_EVENT <- as.factor(heart$DEATH_EVENT)
set.seed(1)
train_idx <- sample(1:nrow(heart), nrow(heart)*0.7)
train_data <- heart[train_idx, ]
test_data <- heart[-train_idx, ]
model <- neuralnet(DEATH_EVENT ~ ., train_data, hidden = c(12, 2))
plot(model)
test_pred <- compute(model, test_data[,1:12])
test_pred_result <- test_pred$net.result
test_pred <- max.col(test_pred_result)
accuracy <- sum(test_pred == test_data$DEATH_EVENT) / nrow(test_data)
accuracy
test_pred




# WINE DATA NN 
wine <- read.csv(paste0(root, "wine.csv"))
wine$Wine <- as.factor(wine$Wine)
set.seed(1)
train_idx <- sample(1:nrow(wine), nrow(wine)*0.7)
train_data <- wine[train_idx, ]
test_data <- wine[-train_idx, ]
model <- neuralnet(Wine ~ Alcohol + Malic.acid + Ash + Acl + Mg + Phenols + Flavanoids + Nonflavanoid.phenols + Proanth + Color.int + Hue + OD + Proline, train_data, hidden = 3)
plot(model)
test_pred <- compute(model, test_data[,2:14])
test_pred_result <- test_pred$net.result
test_pred <- max.col(test_pred_result)
accuracy <- sum(test_pred == test_data$Wine) / nrow(test_data)
accuracy
test_pred



# FAST FOOD NN
fastfood <- read.csv(paste0(root, "fastfood.csv"))
fastfood$restaurant <- as.factor(fastfood$restaurant)
fastfood <- na.omit(fastfood)
fastfood <- fastfood[, -2]
fastfood <- fastfood[, -16]
set.seed(1)
train_idx <- sample(1:nrow(fastfood), nrow(fastfood)*0.7)
train_data <- fastfood[train_idx, ]
test_data <- fastfood[-train_idx, ]
model <- neuralnet(restaurant ~ ., train_data, hidden = c(15, 3))
plot(model)
test_pred <- compute(model, test_data[,1:15])
test_pred_result <- test_pred$net.result
test_pred <- max.col(test_pred_result)
accuracy <- sum(test_pred == test_data$restaurant) / nrow(test_data)
accuracy
test_pred




