library(e1071)
library(ggplot2)
library(corrplot)
library(factoextra)
library(Metrics)
library(glmnet)
library(tidyverse)
library(caret)
library(foba)
library(ridge)

setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2")
#                                               Parte 0 - Pre Processamento
# o treino_y e Teste_y é a solubilidade de cada composto
treino_x = read.table("solTrainX.txt")
treino_y = read.table("solTrainY.txt")
Teste_x = read.table("solTesteX.txt")
Teste_y = read.table("solTesteY.txt")
dim(treino_x)
dim(treino_y)
dim(Teste_x)
dim(Teste_y)
dataset = rbind(treino_x,Teste_x)
dataset["y"] = rbind(treino_y,Teste_y)
# são 208 Fingerprints(indicadores de moléculas), então as Features de interesse são as 20 restantes
Cor = cor(dataset[209:228])
dataTreino = treino_x
dataTreino["y"] = treino_y
dataTeste = Teste_x
dataTeste["y"] = Teste_y

RemoverCorr = findCorrelation(cor(dataTreino), .9) # ==> 34 elementos 
dataTreino = dataTreino[, -RemoverCorr]
dataTeste = dataTeste[, -RemoverCorr]
get_eig(pca.data)[3]>90 # para ter mais de 90% da variância explicada, precisa de 58 componentes
# a matriz de correlação, onde há 34 colunas com correlação > 0.9
png("Matriz_Correlação_upper.png")
corrplot(Cor, type = "upper")
dev.off()

# Colunas finais removidas : NumAtoms, NumNonHAtoms, NumBonds, NumNonHBonds, NumMultBonds, NumHalogen, SurfaceArea2



#PCA
pca.data = prcomp(treino_x[1:228],scale=TRUE)
pdf("Variância vs Componentes.pdf")
fviz_eig(pca.data,geom = "line", ncp = 10,xlab="Componente",ylab="Variância",addlabels = TRUE,main="Variância vs Componentes")
dev.off()

ggplot(dataset,aes(dataset[,1])) + geom_histogram(color="black",fill="springgreen2")+theme_gray()+
    labs(x=names(dataset[1]),y="Frequência")
# Histograma
setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2/Histograma")
for(i in 1:228){
  
  ggplot(dataset,aes(dataset[,i])) + geom_histogram(color="black",fill="springgreen2")+theme_gray()+
    labs(x=names(dataset[i]),y="Frequência")
  ggsave(paste(names(dataset[i]),".png"))
}
  dev.off()

# Scatter plot(dispersão)
setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2/ScatterPlot")
for(i in 209:228){
  for(j in 209:228){
    if(i>j){
      ggplot(dataset, aes(dataset[,i],dataset[,j])) + geom_point(aes(color=y)) +theme_gray() + labs( x=names(dataset[i]), y=names(dataset[j]))
      ggsave(paste(names(dataset[i])," vs ",names(dataset[j]),".png"))
      ggsave(paste(names(dataset[i])," vs ",names(dataset[j]),".pdf"))
    }
  }
}
dev.off()

setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2")


#                                               Parte 1 - Ordinary Linear Regression

# Build the model
set.sed(5)
olsCV5 <- train(y ~., data = dataTreino, method = "lm",  trControl = trainControl("cv", number = 5),  tuneLength = 13,  preProc=c("center","scale","YeoJohnson"))
olsCV10 <- train(y ~., data = dataTreino, method = "lm",  trControl = trainControl("cv", number = 10),  tuneLength = 13,  preProc=c("center","scale","YeoJohnson"))
# Model coefficients
coef(olsCV5$finalModel, olsCV5$bestTune$lambda)
coef(olsCV10$finalModel, olsCV10$bestTune$lambda)
# Make predictions
  #CV5
predictionsOLSTreinoCV5 <- predict(olsCV5, dataTreino)
predictionsOLSTesteCV5 <- predict(olsCV5, dataTeste)
  #CV10
predictionsOLSTreinoCV10 <- predict(olsCV10, dataTreino)
predictionsOLSTesteCV10 <- predict(olsCV10, dataTeste)
# Model prediction performance

# Performace via RMSE e Rsquare do Treino 
  #CV5
perfTreinoOLSCV5 = data.frame(
  RMSE = RMSE(predictionsOLSTreinoCV5, dataTreino$y),
  Rsquare = R2(predictionsOLSTreinoCV5, dataTreino$y)
)
  #CV10
perfTreinoOLSCV10 = data.frame(
  RMSE = RMSE(predictionsOLSTreinoCV10, dataTreino$y),
  Rsquare = R2(predictionsOLSTreinoCV10, dataTreino$y)
)

# Performace via RMSE e Rsquare do Teste
  #CV5
perfTesteOLSCV5 = data.frame(
  RMSE = RMSE(predictionsOLSTesteCV5, dataTeste$y),
  Rsquare = R2(predictionsOLSTesteCV5, dataTeste$y)
)
  #CV10
perfTesteOLSCV10 = data.frame(
  RMSE = RMSE(predictionsOLSTesteCV10, dataTeste$y),
  Rsquare = R2(predictionsOLSTesteCV10, dataTeste$y)
)

  #CV5
dfTreinoOLSCV5 = data.frame(dataTreino$y,predictionsOLSTreinoCV5)
dfTesteOLSCV5 = data.frame(dataTeste$y,predictionsOLSTesteCV5)
  #CV10
dfTreinoOLSCV10 = data.frame(dataTreino$y,predictionsOLSTreinoCV10)
dfTesteOLSCV10 = data.frame(dataTeste$y,predictionsOLSTesteCV10)

setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2/OLS")
pdf("olsTrain5fold.pdf")
ggplot(dfTreinoOLSCV5,aes(dataTreino.y,predictionsOLSTreinoCV5)) + geom_point(color="green") + labs(title="OLS Treino: 5-fold",x="Predito",y="Observado")
dev.off()
pdf("olsTeste5fold.pdf")
ggplot(dfTesteOLSCV5,aes(dataTeste.y,predictionsOLSTesteCV5)) + geom_point(color="green") + labs(title="OLS Teste: 5-fold",x="Predito",y="Observado")
dev.off()


pdf("olsTrain10fold.pdf")
ggplot(dfTreinoOLSCV10,aes(dataTreino.y,predictionsOLSTreinoCV10)) + geom_point(color="green") + labs(title="OLS Treino: 10-fold",x="Predito",y="Observado")
dev.off()
pdf("olsTeste10fold.pdf")
ggplot(dfTesteOLSCV10,aes(dataTeste.y,predictionsOLSTesteCV10)) + geom_point(color="green") + labs(title="OLS Teste: 10-fold",x="Predito",y="Observado")
dev.off()

#                                               Parte 2 - L² Penalized Linear Regression (Ridge)

# Build the model
set.seed(25)
ridgeCV5 <- train( y ~., data = dataTreino, method = "ridge", trControl = trainControl("cv", number = 5), tuneLength = 13, preProc=c("center","scale","YeoJohnson"))
ridgeCV10 <- train( y ~., data = dataTreino, method = "ridge", trControl = trainControl("cv", number = 10), tuneLength = 13, preProc=c("center","scale","YeoJohnson"))
# Model coefficients
coef(ridge$finalModel, ridge$bestTune$lambda)
# Make predictions

  #CV5
predictionsL2TreinoCV5 <- predict(ridgeCV5, dataTreino)
predictionsL2TesteCV5 <- predict(ridgeCV5, dataTeste)
  #CV10
predictionsL2TreinoCV10 <- predict(ridgeCV10, dataTreino)
predictionsL2TesteCV10 <- predict(ridgeCV10, dataTeste)

# Model prediction performance

# Performace via RMSE e Rsquare do Treino 
  #CV5
perfL2TreinoCV5 = data.frame( 
  RMSE = RMSE(predictionsL2TreinoCV5, dataTreino$y), 
  Rsquare = R2(predictionsL2TreinoCV5, dataTreino$y)
)
  #CV10
perfL2TreinoCV10 = data.frame( 
  RMSE = RMSE(predictionsL2TreinoCV10, dataTreino$y), 
  Rsquare = R2(predictionsL2TreinoCV10, dataTreino$y)
)

# Performace via RMSE e Rsquare do Teste
  #CV5
perfL2TesteCV5 = data.frame( 
  RMSE = RMSE(predictionsL2TesteCV5, dataTeste$y), 
  Rsquare = R2(predictionsL2TesteCV5, dataTeste$y)
)
  #CV10
perfL2TesteCV10 = data.frame( 
  RMSE = RMSE(predictionsL2TesteCV10, dataTeste$y), 
  Rsquare = R2(predictionsL2TesteCV10, dataTeste$y)
)

  #CV5
dfTreinoL2CV5 = data.frame(dataTreino$y,predictionsL2TreinoCV5)
dfTesteL2CV5 = data.frame(dataTeste$y,predictionsL2TesteCV5)
  #CV10
dfTreinoL2CV10 = data.frame(dataTreino$y,predictionsL2TreinoCV10)
dfTesteL2CV10 = data.frame(dataTeste$y,predictionsL2TesteCV10)

setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2/L2")
pdf("L2Train5fold.pdf")
ggplot(dfTreinoL2CV5,aes(dataTreino.y,predictionsL2TreinoCV5)) + geom_point(color="blue") + labs(title="L2_Treino: 5-fold",x="Predito",y="Observado")
dev.off()
pdf("L2Teste5fold.pdf")
ggplot(dfTesteL2CV5,aes(dataTeste.y,predictionsL2TesteCV5)) + geom_point(color="blue") + labs(title="L2_Teste: 5-fold",x="Predito",y="Observado")
dev.off()

pdf("L2Train10fold.pdf")
ggplot(dfTreinoL2CV10,aes(dataTreino.y,predictionsL2TreinoCV10)) + geom_point(color="blue") + labs(title="L2_Treino: 10-fold",x="Predito",y="Observado")
dev.off()
pdf("L2Teste10fold.pdf")
ggplot(dfTesteL2CV10,aes(dataTeste.y,predictionsL2TesteCV10)) + geom_point(color="blue") + labs(title="L2_Teste: 10-fold",x="Predito",y="Observado")
dev.off()

#                                               Parte 3.1 - PLS

# Build the model
set.seed(50)
plsCV5 <- train( y ~., data = dataTreino, method = "pls", trControl = trainControl("cv", number = 5), tuneLength = 13, preProc=c("center","scale","YeoJohnson"))
plsCV10 <- train( y ~., data = dataTreino, method = "pls", trControl = trainControl("cv", number = 10), tuneLength = 13, preProc=c("center","scale","YeoJohnson"))

# Make predictions
  #CV5
predictionsPLSTreinoCV5 <- predict(plsCV5, dataTreino)
predictionsPLSTesteCV5 <- predict(plsCV5, dataTeste)
  #CV10
predictionsPLSTreinoCV10 <- predict(plsCV10, dataTreino)
predictionsPLSTesteCV10 <- predict(plsCV10, dataTeste)
# Model prediction performance

# Performace via RMSE e Rsquare do Treino 
  #CV5
perfPLSTreinoCV5 = data.frame(
  RMSE = RMSE(predictionsPLSTreinoCV5, dataTreino$y),
  Rsquare = R2(predictionsPLSTreinoCV5, dataTreino$y)
)
  #CV10
perfPLSTreinoCV10 = data.frame(
  RMSE = RMSE(predictionsPLSTreinoCV10, dataTreino$y),
  Rsquare = R2(predictionsPLSTreinoCV10, dataTreino$y)
)

# Performace via RMSE e Rsquare do Teste
  #CV5
perfPLSTesteCV5 = data.frame(
  RMSE = RMSE(predictionsPLSTesteCV5, dataTeste$y),
  Rsquare = R2(predictionsPLSTesteCV5, dataTeste$y)
)
  #CV10
perfPLSTesteCV10 = data.frame(
  RMSE = RMSE(predictionsPLSTesteCV10, dataTeste$y),
  Rsquare = R2(predictionsPLSTesteCV10, dataTeste$y)
)
  #CV5
dfTreinoPLSCV5 = data.frame(dataTreino$y,predictionsPLSTreinoCV5)
dfTestePLSCV5 = data.frame(dataTeste$y,predictionsPLSTesteCV5)
  #CV10
dfTreinoPLSCV10 = data.frame(dataTreino$y,predictionsPLSTreinoCV10)
dfTestePLSCV10 = data.frame(dataTeste$y,predictionsPLSTesteCV10)

setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2/PLS")
pdf("PLSTrain5fold.pdf")
ggplot(dfTreinoPLSCV5,aes(dataTreino.y,predictionsPLSTreinoCV5)) + geom_point(color="red") + labs(title="PLS-Train: 5-fold",x="Predito",y="Observado")
dev.off()
pdf("PLSTeste5fold.pdf")
ggplot(dfTestePLSCV5,aes(dataTeste.y,predictionsPLSTesteCV5)) + geom_point(color="red") + labs(title="PLS-Teste: 5-fold",x="Predito",y="Observado")
dev.off()
pdf("PLSTrain10fold.pdf")
ggplot(dfTreinoPLSCV10,aes(dataTreino.y,predictionsPLSTreinoCV10)) + geom_point(color="red") + labs(title="PLS-Train: 10-fold",x="Predito",y="Observado")
dev.off()
pdf("PLSTeste10fold.pdf")
ggplot(dfTestePLSCV10,aes(dataTeste.y,predictionsPLSTesteCV10)) + geom_point(color="red") + labs(title="PLS-Teste: 10-fold",x="Predito",y="Observado")
dev.off()

#                                               Parte 3.2 - PCR

# Build the model
# Build the model
set.seed(100)
PCRCV5 <- train( y ~., data = dataTreino, method = "pcr", trControl = trainControl("cv", number = 5), tuneLength = 13, preProc=c("center","scale","YeoJohnson"))
PCRCV10 <- train( y ~., data = dataTreino, method = "pcr", trControl = trainControl("cv", number = 10), tuneLength = 13, preProc=c("center","scale","YeoJohnson"))

# Make predictions
  #CV5
predictionsPCRTreinoCV5 <- predict(PCRCV5, dataTreino)
predictionsPCRTesteCV5 <- predict(PCRCV5, dataTeste)
  #CV10
predictionsPCRTreinoCV10 <- predict(PCRCV10, dataTreino)
predictionsPCRTesteCV10 <- predict(PCRCV10, dataTeste)
# Model prediction performance

# Performace via RMSE e Rsquare do Treino 
  #CV5
perfPCRTreinoCV5 = data.frame(
  RMSE = RMSE(predictionsPCRTreinoCV5, dataTreino$y),
  Rsquare = R2(predictionsPCRTreinoCV5, dataTreino$y)
)
  #CV10
perfPCRTreinoCV10 = data.frame(
  RMSE = RMSE(predictionsPCRTreinoCV10, dataTreino$y),
  Rsquare = R2(predictionsPCRTreinoCV10, dataTreino$y)
)

# Performace via RMSE e Rsquare do Teste
  #CV5
perfPCRTesteCV5 = data.frame(
  RMSE = RMSE(predictionsPCRTesteCV5, dataTeste$y),
  Rsquare = R2(predictionsPCRTesteCV5, dataTeste$y)
)
  #CV10
perfPCRTesteCV10 = data.frame(
  RMSE = RMSE(predictionsPCRTesteCV10, dataTeste$y),
  Rsquare = R2(predictionsPCRTesteCV10, dataTeste$y)
)
  #CV5
dfTreinoPCRCV5 = data.frame(dataTreino$y,predictionsPCRTreinoCV5)
dfTestePCRCV5 = data.frame(dataTeste$y,predictionsPCRTesteCV5)
  #CV10
dfTreinoPCRCV10 = data.frame(dataTreino$y,predictionsPCRTreinoCV10)
dfTestePCRCV10 = data.frame(dataTeste$y,predictionsPCRTesteCV10)

setwd("D:/Eng.Comp/2019/2019.2/ICA/HW2/PCR")
pdf("PCRTrain5fold.pdf")
ggplot(dfTreinoPCRCV5,aes(dataTreino.y,predictionsPCRTreinoCV5)) + geom_point(color="red") + labs(title="PCR-Train: 10-fold",x="Predito",y="Observado")
dev.off()
pdf("PCRTeste5fold.pdf")
ggplot(dfTestePCRCV5,aes(dataTeste.y,predictionsPCRTesteCV5)) + geom_point(color="red") + labs(title="PCR-Teste: 10-fold",x="Predito",y="Observado")
dev.off()
pdf("PCRTrain10fold.pdf")
ggplot(dfTreinoPCRCV10,aes(dataTreino.y,predictionsPCRTreinoCV10)) + geom_point(color="red") + labs(title="PCR-Train: 10-fold",x="Predito",y="Observado")
dev.off()
pdf("PCRTeste10fold.pdf")
ggplot(dfTestePCRCV10,aes(dataTeste.y,predictionsPCRTesteCV10)) + geom_point(color="red") + labs(title="PCR-Teste: 10-fold",x="Predito",y="Observado")
dev.off()

