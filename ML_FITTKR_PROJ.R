# ML Final Project
# load libraries
library(caret); library(dplyr); library(ggplot2); library(kernlab); library(MASS)
library(e1071)
# read in data

DataTr <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
DataTst <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")


write.csv(DataTr, file = "DataTr.csv")
write.csv(DataTst, file = "DataTst.csv")

#summary data
summary(DataTr)
str(DataTr)
head(DataTr)


# select for variables to predict on
DataTr <- dplyr::select(DataTr, roll_belt, pitch_belt, yaw_belt, total_accel_belt,
                 gyros_belt_x, gyros_belt_y, gyros_belt_z,
                 accel_belt_x, accel_belt_y, accel_belt_z,
                 magnet_belt_x, magnet_belt_y, magnet_belt_z,
                 roll_arm, pitch_arm, yaw_arm, total_accel_arm,
                 gyros_arm_x, gyros_arm_y, gyros_arm_z,
                 magnet_arm_x, magnet_arm_y, magnet_arm_z,
                 roll_dumbbell, pitch_dumbbell, yaw_dumbbell, total_accel_dumbbell,
                 gyros_dumbbell_x, gyros_dumbbell_y, gyros_dumbbell_z, 
                 accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z,
                 magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z,
                 roll_forearm, pitch_forearm, pitch_forearm, yaw_forearm, total_accel_forearm,
                 gyros_forearm_x, gyros_forearm_y, gyros_forearm_z, 
                 accel_forearm_x, accel_forearm_y, accel_forearm_z, 
                 magnet_forearm_x, magnet_forearm_y, magnet_forearm_z, classe)




set.seed(353)

# data split
spTraining <- createDataPartition(DataTr$classe,
                                  p=0.70, list=FALSE)
spTrain <- DataTr[ spTraining,]
spTest <- DataTr[-spTraining, ]


# fit prediction models

##glm(formula, family=gaussian(link=identity), data=)

gbm.fit <- train(classe ~., method = "gbm", data = spTrain)
pred.gbm <- predict(gbm.fit, spTest)


glm.fit <- train(classe ~., method = "glm", data = spTrain)
pred.glm <- predict(lda.glm, spTest)


rf.fit <- train(classe ~., method = "rf", data = spTrain)
pred.rf <- predict(rf.fit, spTest)



# k fold cross validation
# define training control
train_control5<- trainControl(method="cv", number=5,  classProbs = TRUE)
train_control10<- trainControl(method="cv", number=10, classProbs = TRUE)
train_control15<- trainControl(method="cv", number=15, classProbs = TRUE)

# train K Fold Models


kfGbm5 <- train(classe ~ ., data = spTrain,
                 method = "gbm",
                 trControl = train_control5,
                 verbose = FALSE)

kfGbm10 <- train(classe ~ ., data = spTrain,
                 method = "gbm",
                 trControl = train_control10,
                 verbose = FALSE)

kfGbm15 <- train(classe ~ ., data = spTrain,
                 method = "gbm",
                 trControl = train_control15,
                 verbose = FALSE)


nbFit <- naiveBayes(classe~., data = spTrain, trControl = train_control10,
                    verbose = FALSE)

rf.fit <- train(classe ~., method = "rf", data = spTrain)

rpart.fit <- train(classe ~., method = "rpart", data = spTrain)

# summarize  models
kfGbm5
kfGbm10
kfGbm15

nbFit
rf.fit
rpart.fit


# predict and print confusion matrix

predkfGbm5 <- predict(kfGbm5, spTest)
table(predkfGbm5, spTest$classe)

predkfGbm10 <- predict(kfGbm10, spTest)
table(predkfGbm10, spTest$classe)

predkfGbm15 <- predict(kfGbm15, spTest)
table(predkfGbm15, spTest$classe)

prednbFit <- predict(nbFit, spTest)
table(prednbFit, spTest$classe)

predrpart.fit <- predict(rpart.fit, spTest)
table(predrpart.fit, spTest$classe)

predrf.fit <- predict(rf.fit, spTest)
table(predrf.fit, spTest$classe)





