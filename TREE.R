
# MTCARS TREE

data(mtcars)
mtcars$cyl = as.factor(mtcars$cyl)
library(tree)
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



# SPAM EMAIL TREE
spam <- read.csv("~/kernlab_spam.csv")
spam$type <- as.factor(spam$type)
library(tree)
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




# Heart Tree

heart <- read.csv("~/HeartFailure.csv")
heart$DEATH_EVENT <- as.factor(heart$DEATH_EVENT)
library(tree)
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
fastfood <- read.csv("~/fastfood.csv")
fastfood$restaurant <- as.factor(fastfood$restaurant)
fastfood <- na.omit(fastfood)
fastfood <- fastfood[, -2]
fastfood <- fastfood[, -16]


# FastFood tree

library(tree)
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
