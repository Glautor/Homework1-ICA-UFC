
library(caret)
library(MASS)

load("grantData_HW3.RData")
set.seed(5)

fullSet = rbind(training,testing)
amostraTreino = (1:dim(training)[1]);

ctrl <- trainControl(method = "LOOCV",
                    summaryFunction = twoClassSummary,
                    classProbs = TRUE,
                    index = list(TrainSet = amostraTreino),
                    savePredictions = TRUE)

#Logistic Regression with reduced set,the testing sample is used to teste the model and the training sample is used to train it
LogRegRed <- train(fullSet[,reducedSet], 
                    y = fullSet$Class,
                    method = "glm", 
                    metric = "ROC", 
                    trControl = ctrl)

confusionMatrix(data = LogRegRed$pred$pred,
                reference = LogRegRed$pred$obs)
