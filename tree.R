


spam <- read.csv("~/Desktop/kernlab_spam.csv")
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







bike <- read.csv("~/Downloads/SeoulBikeData.csv")
bike$type <- as.factor(bike$Seasons)

library(tree)
tree_out = tree(type ~ ., data=bike)
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
train_indices = sample(1:nrow(bike), round(0.7 * nrow(bike)))
train = bike[train_indices, ]
test = bike[-train_indices, ]
tree_out = tree(type ~ ., data=train)
tree_pred = predict(tree_out, test, type="class")
table(tree_pred, test$type)
mean(test$type != tree_pred)
misclass_rate <- mean(test$type != tree_pred)
cat("Misclassification rate:", misclass_rate, "\n")
