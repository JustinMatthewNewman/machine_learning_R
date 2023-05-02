library(tree)
library(ISLR2) # find data sets
library(dplyr) # split data
library(ggplot2) # plot
library(randomForest)
root = "https://raw.githubusercontent.com/JustinMatthewNewman/machine_learning_R/main/datasets/"





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

