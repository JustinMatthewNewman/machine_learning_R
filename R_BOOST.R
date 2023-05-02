library(readr)
library(pROC)
library(dplyr)
library(gbm)

root = "https://raw.githubusercontent.com/JustinMatthewNewman/machine_learning_R/main/datasets/"


# ========== mtcars boosting regression ==========

mtcars=mtcars
mtcars
set.seed(1)

mtcars_train = mtcars %>%
  sample_frac(0.5)
mtcars_test = mtcars %>%
  setdiff(mtcars_train)

set.seed(1)
boost_mtcars = gbm(am~., 
                   data=mtcars_train, 
                   distribution = "gaussian",
                   n.trees=15,
                   interaction.depth = 5)
summary(boost_mtcars)

boost_estimate = predict(boost_mtcars,
                         newdata= mtcars_test,
                         n.trees = 5000)

mean((boost_estimate - mtcars_test$am)^2)

boost_mtcars2 = gbm(am~., data = mtcars_train,
                    distribution = "gaussian",
                    n.trees = 5000,
                    interaction.depth = 4,
                    shrinkage = 0.01,
                    verbose = F)

boost_estimate2 = predict(boost_mtcars2, newdata = mtcars_test, n.trees = 5000)
mean((boost_estimate2 - mtcars_test$am)^2)

# ========== mtcars boosting on classification ==========

mtcars$am = as.numeric(mtcars$am)-1

mtcars_train = mtcars %>%
  sample_frac(0.67)
mtcars_test = mtcars %>%
  setdiff(mtcars_train)

out=gbm(am~., data=mtcars_train, distribution="bernoulli", n.trees=5000)

pred=predict(out, newdata=mtcars_test, n.trees=5000)

predProb = exp(pred) / (1+exp(pred))

auc(mtcars_test$am, predProb)



# ========== KERN boosting regression ==========

url = "kernlab_spam.csv"
kern = read.csv(paste0(root, url))
kern
set.seed(1)

kern_train = kern %>%
  sample_frac(0.5)
kern_test = kern %>%
  setdiff(kern_train)

set.seed(1)
boost_kern = gbm(free~., 
                   data=kern_train, 
                   distribution = "gaussian",
                   n.trees=5000,
                   interaction.depth = 4)
summary(boost_kern)

boost_estimate = predict(boost_kern,
                         newdata= kern_test,
                         n.trees = 5000)

mean((boost_estimate - kern_test$free)^2)

boost_kern2 = gbm(free~., data = kern_train,
                    distribution = "gaussian",
                    n.trees = 5000,
                    interaction.depth = 4,
                    shrinkage = 0.01,
                    verbose = F)

boost_estimate2 = predict(boost_kern2, newdata = kern_test, n.trees = 5000)
mean((boost_estimate2 - kern_test$free)^2)

# ========== KERN boosting on classification ==========

set.seed(1)

kern <- kern %>%
  mutate(type= ifelse(type = "spam", 1, 0))
kern$type = as.numeric(kern$type)-1

kern_train = kern %>%
  sample_frac(0.67)
kern_test = kern %>%
  setdiff(kern_train)

out=gbm(type~., data=kern_train, distribution="bernoulli", n.trees=5000)
out

pred=predict(out, newdata=kern_test, n.trees=5000)
pred

predProb = exp(pred) / (1+exp(pred))
predProb

auc(kern_test$type, predProb)



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

Heart$DEATH_EVENT = as.numeric(Heart$DEATH_EVENT)-1

set.seed(1)
Heart_train = Heart %>%
  sample_frac(0.67)
Heart_test = Heart %>%
  setdiff(Heart_train)

out=gbm(DEATH_EVENT~., data=Heart_train, distribution="bernoulli", n.trees=5000)
out

pred=predict(out, newdata=Heart_test, n.trees=5000)
pred

predProb = exp(pred) / (1+exp(pred))
predProb

auc(Heart_test$DEATH_EVENT, predProb)

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

WineDat$Wine = as.numeric(WineDat$Wine)-1

wine_train = WineDat %>%
  sample_frac(0.67)
wine_test = WineDat %>%
  setdiff(wine_train)

out=gbm(Wine~., data=wine_train, distribution="bernoulli", n.trees=5000)
out

pred=predict(out, newdata=wine_test, n.trees=5000)
pred

predProb = exp(pred) / (1+exp(pred))
predProb

auc(wine_test$Wine, predProb)



# ========== Fast Food boosting Regression ==========

url = "fastfood.csv"
fastfood=read.csv(paste0(root, url))
fastfood

fastfood$calories <- as.factor(fastfood$calories)
fastfood <- na.omit(fastfood)
fastfood <- fastfood[,-2]
fastfood <- fastfood[,-16]

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

