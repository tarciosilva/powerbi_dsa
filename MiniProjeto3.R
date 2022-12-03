#*******************************************************
#* Mini_Projeto de Machine Learning
#* 
#* Tarcio Pereira da Silva
#*
#* Curso de Power BI para Ciência de Dados
#*
#* Data Science Academy - 2022
#*
#********************************************************#

#Pasta de Trabalho
setwd("C:/Users/tarci/Desktop/DSA/powerbi/cap15")
getwd()

#Pacotes para trabalhos com Machine Learning
##########################################
install.packages("Amelia")
install.packages("caret")
#install.packages("ggplot2")
#install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")
##########################################

#Carregando Pacotes
library(Amelia) #funções pra tratar valores ausentes
library(ggplot2)
library(caret) #permite construir modelos de machine learning e pré-processar os dados
library(dplyr) #permite manipular dados
library(reshape) #permite modificar o formato dos dados
library(randomForest) #permite trabalhar com machine learning PS: olhar documentação deste pacote
library(e1071) #permite trabalhar com machine learning PS: olhar documentação deste pacote

##########################################

#Carregando DataSet
dados_clientes <- read.csv("dados/dataset.csv")

#Visualizando dados e sua estrutura
View(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

###############################################
###########Análise Exploratória, Limpeza e Transformação########

#Removendo a primeira coluna ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

#Renomeando a coluna de classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)
View(dados_clientes)

#Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
?missmap
missmap(dados_clientes, main = "valores Missing observados")
dados_clientes <- na.omit(dados_clientes)

############################################################

#Convertendo os atributos genero, escolaridade, estado civil e
#idade para fatores (categorias)

#Renomeando colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)

#Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut #converte variável numérica pra variável do tipo categórica e pode mudar seu valor também
dados_clientes$Genero <- cut(dados_clientes$Genero, 
                             c(0, 1, 2),
                             labels = c("Masculino",
                                        "Feminino"))
View(dados_clientes$Genero)
str(dados_clientes$Genero)
View(dados_clientes)

#Escolaridade
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade, 
                                   c(0, 1, 2, 3, 4, 5, 6),
                                   labels = c("Outros", 
                                              "Pos_Graduado",
                                              "Graduado",
                                              "Ensino_Medio_Completo",
                                              "Outros", "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
View(dados_clientes)

#Estado civil
View(dados_clientes$Estado_civil)
str(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)
dados_clientes$Estado_civil <- cut(dados_clientes$Estado_civil,
                                   c(-1, 0, 1, 2, 3),
                                   labels = c("Desconhecido",
                                              "Casado", "Solteiro",
                                              "Outro"))
View(dados_clientes$Estado_civil)
str(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)
View(dados_clientes)

#Coonvertendo a variável Idade para o tipo fator/categórica com faixa etária
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade,
                            c(0, 30, 50, 100),
                            labels = c("Jovem",
                                       "Adulto", "Idoso"))
View(dados_clientes)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)

##############################################################

#Convertendo a variável que indica pagamentos para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

#dataset após as conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observado")
dim(dados_clientes)

#Total de inadimplentes vs adimplentes
table(dados_clientes$inadimplente)

#porcentagem entre as classes
prop.table(table(dados_clientes$inadimplente))

#plot da distribuição usando o ggplo2
qplot(inadimplente, data = dados_clientes, geom = "bar") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#set seed
set.seed(12345)

#amostragem estratificada
#seleciona as linha de acordo com a variável inadimplente como strata
?createDataPartition
indice <- createDataPartition(dados_clientes$inadimplente, p = 0.75, list = FALSE)
dim(indice)

#Definimos os dados de treinamento como subconjunto do conjunto de dados original
#com números de índice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,]
table(dados_treino$inadimplente)

#porcentagem entre as classes
prop.table(table(dados_treino$inadimplente))

#número de registros no dataset de treinamento
dim(dados_treino)

#comparando as porcentagens enntre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                       prop.table(table(dados_clientes$inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

#Melt Data - Convernte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

#Plot pra ver a distribuição do treinamento vs Ooriginal
ggplot(melt_compara_dados, aes(x = X1, y = value)) +
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Tudo que não está no Dataset de treinamento será Dataset de teste. Feito pelo sinal de - em indice
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)

##########################################################
#############Modelos de Machine Learning
#Modelo 1
?randomForest
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1

#Avaliando o modelo
plot(modelo_v1)

#previsoes com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)


#####################Trecho de código com problemas 
#Confusion matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1

#Caloculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#####################################################################
#install.packages(c("zoo","xts","quantmod"))
#install.packages( "C:/Users/tarci/Desktop/pck/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
#install.packages("ROCR")
library(DMwR)
?SMOTE

#av <- available.packages(filters=list())
#av[av[, "Package"] == pkg, ]




#Aplicando o SMOTE - Synthetic Minority Over-sampling Techinique
#cria registros sintéticos dos dados
#https://arxiv.org/pdf/1106.1813.pdf
table(dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))
set.seed(9560)

dados_treino2 <- dados_treino
dados_treino2$inadimplente <- as.factor(dados_treino2$inadimplente)


dados_treino_bal <- SMOTE(inadimplente ~ ., data = dados_treino2)
table(dados_treino_bal$inadimplente)
warnings()
prop.table(table(dados_treino_bal$inadimplente))


#Modelo 2
modelo_v2 <- randomForest(inadimplente ~ ., data = dados_treino_bal)
modelo_v2

#Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            importance = round(imp_var[ , 'MeanDecreaseGini'], 2))

#Criando o Rank das vari[aveis
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(importance))))

#Usando o ggplot2 para visualizar a importância relativa das variáveis
ggplot(rankImportance, aes(x = reorder(Variables, Importance),
                           y = Importance, fill = Importance)) +
                           geom_bar(stat = 'identity') + 
                           geom_text(aes(x = Variables, y = 0.5,
                                         label = Rank),
                                     hjust = 0,
                                     vjust = 0.55,
                                     size = 4,
                                     colour = 'red') +
                             labels(x = 'Variables') +
                             coord_flip()

######################################################################
#Modelo 3
colnames(dados_treino_bal)
modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + 
                            PAY_5 + BILL_AMT1, data = dados_treino_bal)
modelo_v3

#Avaliando o modelo
plot(modelo_v3)

#Previsões com dados de texte
previsoes_v3 <- predict(modelo_v3, dados_teste)

#Confusion Matrix
?caret::confusionMatrix
teste <- dados_teste
teste$inadimplente <- as.factor(teste$inadimplente)

cm_v3 <- caret::confusionMatrix(previsoes_v3, teste$inadimplente, positive = "1")
cm_v3

#Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- teste$inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

##########################################################

#Salvando o modelo em disco
saveRDS(modelo_v3, file = "modelo_v3.rds")

#Carregando modelo
modelo_final <- readRDS("modelo_v3.rds")
#########################################################

#Previsões com novos dados de 3 clientes

#Dados de clientes
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

#Concatenando um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)
str(novos_clientes)

novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels(dados_treino$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels(dados_treino$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels(dados_treino$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels(dados_treino$PAY_5))

#Previsões
previsoes_novos_clientes <- predict(modelo_v3, novos_clientes)
previsoes_novos_clientes

plot(previsoes_novos_clientes)
