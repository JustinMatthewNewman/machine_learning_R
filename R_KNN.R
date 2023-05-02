library(dplyr)
library(FNN)
library(ISLR)
library(class)
library(readr)

## KNN FILE BY Hannah Dalakate Phommachanthone
root = "https://raw.githubusercontent.com/JustinMatthewNewman/machine_learning_R/main/datasets/"
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

