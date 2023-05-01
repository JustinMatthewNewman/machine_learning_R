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
model <- neuralnet(am ~ ., train_data, hidden = 3)
plot(model)

test_pred <- compute(model, test_data[,1:(ncol(mtcars) - 1)])
test_pred_result <- test_pred$net.result
test_pred <- max.col(test_pred_result)
accuracy <- sum(test_pred == test_data$am) / nrow(test_data)
accuracy
test_pred





#SPAM DATA NN
spamdata <- read.csv("~/Documents/math358/kernlab_spam.csv")
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
heart <- read.csv("~/Documents/math358/HeartFailure.csv")
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
wine <- read.csv("~/Documents/math358/wine.csv")
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
fastfood <- read.csv("~/Documents/math358/fastfood.csv")
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














