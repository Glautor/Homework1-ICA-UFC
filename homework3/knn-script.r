library(ISLR)
library(dplyr)
library(class)
library(ggplot2)

setwd("C:/Users/Glauton/Dropbox/Glauton/UFC/6-Sexto Semestre/ICA/homeworks/homework3")
load("grantData_HW3.RData")

# o comando acima nos permite usar dois conjuntos de dados, um de treinamento e um de teste, facilitando o processo, ja que nao e necessario dividir
# conjuntos: training e testing

# vamos verificar a variancia de algumas variaveis preditoras para saber se ha necessidade de colocar os 
# dados numa mesma escala, para obtermos um algoritmo mais performatico
var(training[,1])
# => 1.016723

var(training[,2])
# => 0.03638266

var(training[,3])
# => 0.8112048

var(training[,4])
# => 0.0006102024

var(training[,5])
# => 0.003409142

# aqui nao pegamos a ultima variavel pois ela corresponde a 'successfull' e 'unsuccessfull'
new_training <- scale(training[,reducedSet])

# vamos repetir esses passos para o s dados de teste tambem

var(testing[,1])
# => 1.271028

var(testing[,2])
# => 0.02634743

var(testing[,3])
# => 0.8651225

var(testing[,4])
# => 0

var(testing[,5])
# => 0.005769102

# aqui repetimos o mesmo processo dos dados de treinamento
new_testing <- scale(testing[,reducedSet])

# para testar ambos os conjuntos, vamos verificar a variancia de uma variavel preditiva aleatoria em cada um
var(new_training[,3])
var(new_testing[,5])

# vamos comecar com um K (numero de vizinhos) igual a 1
set.seed(1)
previsoes <- knn(train = training[,reducedSet], test = testing[,reducedSet], cl = training[,1882], k = 11)
head(previsoes)

table(previsoes, testing[,1882])
# acima fizemos uma matriz de confusao para analisar os acertos e erros

mean(testing[,1882] != previsoes)
# => 0.3552124
# isso significa que temos um erro de 35% com um K = 1, o que pode ser considerado um erro alto, entao vamos aumentar o K e testar de novo

set.seed(1)
previsoes <- knn(train = training[,reducedSet], test = testing[,reducedSet], cl = training[,1882], k = 26)
head(previsoes)

table(previsoes, testing[,1882])
mean(testing[,1882] != previsoes)

# importante notar que gastariamos muita energia verificando o melhor valor de K, entao podemos criar um loop para verificar isso para a gente
# no loop a seguir vamos analisar o melhor valor de K entre 1 e 100

previsoes = NULL
perc_erro = NULL

for(i in 1:100){
  set.seed(1)
  previsoes <- knn(train = training[,reducedSet], test = testing[,reducedSet], cl = training[,1882], k = i)
  perc_erro[i] = mean(testing[,1882] != previsoes)
}

print(perc_erro)

# para facilitar a leitura, podemos colocar os resultados em um grafico

k_values <- 1:100
error_df <- data.frame(perc_erro, k_values)

ggplot(error_df,aes(x = k_values, y = perc_erro)) + geom_point() + geom_line(lty="dotted", color='red')

# apos essa analise, podemos encontrar o melhor valor de K (26) entre 1 e 499 e aplicar uma matriz de confusao
# encontramos o valor 26

set.seed(1)
previsoes <- knn(train = training[,reducedSet], test = testing[,reducedSet], cl = training[,1882], k = 26)
head(previsoes)

table(previsoes, testing[,1882])
mean(testing[,1882] != previsoes)
