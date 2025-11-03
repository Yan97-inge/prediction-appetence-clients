#####################################################
########   Project : Data analysis        ###########
########   Customer appetite prediction   ###########
########   Awadi BEDJA MROINKODO SAID     ###########
########          Yannick KOULONI         ###########
#####################################################

#-----------------------------------------------#
# DOWNLOADING AND INSTALLING REQUIRED PACKAGES  # 
#-----------------------------------------------#

# Install packages
install.packages("readr")
install.packages("cluster")
install.packages("dbscan")
install.packages("ggplot2")
install.packages("fpc")
install.packages("rpart")
install.packages("C50")
install.packages("tree")
install.packages("rpart.plot")
install.packages("ROCR")
install.packages("regclass")
install.packages("questionr")
install.packages("CORElearn")

# Activate packages
library(readr)
library(cluster)
library(dbscan)
library(ggplot2)
library(fpc)
library(rpart)
library(C50)
library(tree)
library(rpart.plot) 
library(ROCR)
library(regclass)
library(questionr)
library(CORElearn)


#------------------------#
# CHARGEMENT DES DONNÉES #
#------------------------#
projet <- read.csv("Data_Projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T)
projet_New <- read.csv("Data_Projet_New.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T)
View(projet)

#-----------------------------#
# PRÉ-TRAITEMENTS DES DONNÉES #
#-----------------------------#

# Manipulation de base
a <- str(projet)
b <- names(projet)
c <- summary(projet)

# Sélection des lignes RESPONSE = Oui
d <- projet[projet$RESPONSE == "Oui",]
e <- nrow(projet[projet$RESPONSE == "Oui",])

# Histogramme d"effectifs des variables
qplot(RESPONSE, data = projet)
qplot(MARITAL, data = projet)

qplot(AGE, data = projet, color = 'blue') 
qplot(INCOME, data = projet, color ='red') 

# Boites à moustache
#--------------------
# boxplot de Age et Icome dans projet
boxplot(projet$AGE, data =projet, main = "Distribution de Age", ylab = "Valeur de Age")
summary(projet$AGE) 
boxplot(projet$INCOME, data =projet, main = "Distribution de Revenus", ylab = "Valeur de Revenus")
summary(projet$INCOME) 

# Proportions des classes pour Revenus
boxplot(INCOME~RESPONSE, data = projet, main = "Revenus selon une réponse positive à l'offre", xlab = "RESPONSE", ylab ="INCOME") 

# Tables de contingence
table(projet$INCOME, projet$RESPONSE) 
table(projet$RESIDE, projet$RESPONSE) 
table(projet$MARITAL, projet$RESPONSE)

#Définition du nombre de décimales affichées
options(digits=2)

# Tables de contingences en proportions et pourcentage de MARITAL
prop.table(table(projet$MARITAL, projet$RESPONSE, dnn=c("MARITAL", "RESPONSE"))) 
prop.table(table(projet$MARITAL, projet$RESPONSE, dnn=c("MARITAL", "RESPONSE")))*100 

# Tables de contingences en proportions et pourcentage de RESIDE
prop.table(table(projet$RESIDE, projet$RESIDE, dnn=c("RESIDE", "RESPONSE")))

prop.table(table(projet$RESIDE, projet$RESPONSE, dnn=c("RESIDE", "RESPONSE")))*100

# Tables de contingences en proportions et pourcentage de RESPONSE
prop.table(table(projet$INCOME, projet$RESPONSE, dnn=c("INCOME", "RESPONSE")))
prop.table(table(projet$INCOME, projet$RESPONSE, dnn=c("INCOME", "RESPONSE")))*100


# TESTS DE DEPENDANCE ENTRE VARIABLES DISCRETES #
#-----------------------------------------------#

# Test du Chi2
chisq.test(projet$NEWS, projet$RESPONSE)
chisq.test(projet$CARCAT, projet$RESPONSE) 
chisq.test(projet$INTERNET, projet$RESPONSE)

# Test de Fisher
fisher.test(projet$NEWS, projet$RESPONSE)
fisher.test(projet$CARCAT, projet$RESPONSE)
fisher.test(projet$INTERNET, projet$RESPONSE)

# Affichage des residus du chi2 pour chaque classe et chaque valeur de variable predictive
chisq.residuals(rprop(table(projet$NEWS, projet$RESPONSE)))
chisq.residuals(rprop(table(projet$CARCAT, projet$RESPONSE)))
chisq.residuals(rprop(table(projet$INTERNET, projet$RESPONSE)))

# GRAPHIQUES EN MOSAIQUE DES RESIDUS DU CHI² #
#--------------------------------------------#

mosaicplot(NEWS ~ RESPONSE, data = projet, shade = TRUE, main = "Graphe en mosaique")
mosaicplot(CARCAT ~ RESPONSE, data = projet, shade = TRUE, main = "Graphe en mosaique")
mosaicplot(INTERNET ~ RESPONSE, data = projet, shade = TRUE, main = "Graphe en mosaique")

#---------------------------------------------------#
# MESURES DE CORRELATION ENTRE VARIABLES NUMERIQUES #
#---------------------------------------------------#
cor.test(projet$CAR, projet$AGE, method="kendall")
cor.test(projet$CAR, projet$INCOME, method="kendall")
cor.test(projet$CAR, projet$ADDRESS, method="kendall")
cor.test(projet$CAR, projet$EMPLOY, method="kendall")


#--------------------------------------------------------------------#
# MESURES D'EVALUATION D'UTILITE PREDICTIVE DE VARIABLES HETEROGENES #
#--------------------------------------------------------------------#

# Activation de la librairie CORElearn
# Liste des mesures d'evaluation pour la classification supervis?e
infoCore(what="attrEval")

# Liste des mesures d'evaluation pour la regression
infoCore(what="attrEvalReg")

# Calcul des Information Gain pour la prediction de RESPONSE
infg <- attrEval(RESPONSE~., projet, estimator = "InfGain")
View(infg)

# Liste des noms de variables
ListeVar = attr(attrEval(RESPONSE~., projet, estimator = "InfGain"), "names")

# Liste des valeurs d'Information Gain
ListeIG = as.vector(attrEval(RESPONSE~., projet, estimator = "InfGain"))

# Affichage par ordre croissant d'IG
data.frame(ListeVar, ListeIG)[order(ListeIG),]

# Affichage par ordre decroissant d'IG
## data.frame(ListeVar, ListeIG)[order(ListeIG, decreasing = T),]


# Fonction de calcul et affichage par ordre decroissant des mesures 
affEval <- function(mesure) {
  result <- attrEval(RESPONSE~., projet, estimator = mesure)
  ListeVar = attr(result, "names")
  ListeVal = as.vector(result)
  data.frame(ListeVar, ListeVal)[order(ListeVal, decreasing = T),]
}


# Calcul des mesures de Gain Ratio, Index Gini, Relief et MDL
affEval("GainRatio")
affEval("Gini")
affEval("Relief")
affEval("MDL")

# Calcul de la matrice de distance par la fonction daisy() pour variables hétérogènes
matr <- daisy(projet)
summary(matr)

#------------------------#
# CLUSTERING DES DONNÉES #
#------------------------#
# Calcul de la matrice de distance par la fonction daisy() pour variables hétérogènes

km4 <- kmeans(matr,4) # K-means pour K = 4

#Répartition des classes par cluster
table(km4$cluster, projet$RESPONSE)

# Histogramme des effectifs des cluster
qplot(km4$cluster, data = projet, fill = RESPONSE)

# Ajout de la colonne du numéro de cluster
projet_km4 <- data.frame(projet,km4$cluster)
View(projet_km4)

# Boucle for pour varier K dans km_k
for (k in 4:10){
  km <- kmeans(matr,k)
  print(table(km$cluster, projet$RESPONSE))
  print(qplot(km$cluster, data = projet, fill = RESPONSE))
}


#---------------------------#
# CLASSIFICATION SUPERVISÉE #
#---------------------------#
# Suppression des accents pour les variables suivantes 
projet$MARITAL <- gsub("\\é","e",projet$MARITAL)
projet$MARITAL <- as.factor(projet$MARITAL)
projet$EMPCAT <- gsub("\\à","a",projet$EMPCAT)
projet$EMPCAT <- as.ordered(projet$EMPCAT)
projet$JOBSAT=gsub("\\è","e",projet$JOBSAT)
projet$JOBSAT <- as.ordered(projet$JOBSAT)
projet$INCCAT=gsub("\\à","a",projet$INCCAT)
projet$INCCAT <- as.ordered(projet$INCCAT)
projet$INTERNET=gsub("\\é","e",projet$INTERNET)
projet$INTERNET <- as.factor(projet$INTERNET)

#Construction des ensembles d'apprentissage et de test

projet_EA_initial <- projet[1:6100,]
# Repartition de la variable RESPONSE 
Response_Oui <- which(projet_EA_initial$RESPONSE =="Oui") 
Response_Non <- which(projet_EA_initial$RESPONSE =="Non")
# Longeur des deux classes

length(Response_Oui)
length(Response_Non)

# Réquilibrage des Classes
Response_Non.downsample <- sample(Response_Non, length(Response_Oui))

# Autre manière :Response_Oui.upsample <- sample(Response_Non, length(Response_Non))

projet_EA <- projet_EA_initial[c(Response_Non.downsample, Response_Oui),] 
# Autre manière : projet_EA <- projet_EA_initial[c(Response_oui.upsample, Response_Non),] 
projet_ET <- projet[6101:6400,]

# APPRENTISSAGE DE L'ARBRE RPART #
#--------------------------------#
# Construction de l'arbre de decision rpart
tree1 <- rpart(RESPONSE~., projet_EA, parms = list(split = "gini"), control= rpart.control(minbucket = 5))

# Affichage de l'arbre par les fonctions de base de R
plot(tree1)
text(tree1, pretty = 0)
prp(tree1, type = 4, extra = 8, box.palette = "auto")


#----------------------------------------#
# APPRENTISSAGE ARBRE DE DECISION 'C5.0' #
#----------------------------------------#

# Construction de l'arbre de decision C5.0

tree2 <- C5.0(RESPONSE~.,projet_EA, control = C5.0Control(minCases = 10, noGlobalPruning = T))

# Affichage de l'arbre par les fonctions de base de R

plot(tree2, type = "simple")

#----------------------------------------#
# APPRENTISSAGE ARBRE DE DECISION 'tree' #
#----------------------------------------#

# Construction de l'arbre de decision tree

tree3 <- tree(RESPONSE~., projet_EA)

# Affichage de l'arbre par les fonctions de base de R

plot(tree3)
text(tree3, pretty = 0)


#---------------------#
# TEST SUR LES ARBRES #
#---------------------#
# Application de l'arbre de decision a l'ensemble de test 'projet_ET'

test_tree1 <- predict(tree1, projet_ET, type = "class")
test_tree2 <- predict(tree2, projet_ET, type = "class")
test_tree3 <- predict(tree3, projet_ET, type = "class")

# Affichage du vecteur de predictions de la classe des exemples de test
test_tree1
test_tree2
test_tree3

# Affichage du nombre de predictions pour chacune des classes
table(test_tree1)
table(test_tree2)
table(test_tree3)

# Génération des probabilités de prediction pour chaque arbre de decision

prob_tree1 <- predict(tree1, projet_ET, type ="prob")
print(prob_tree1)
prob_tree2 <- predict(tree2, projet_ET, type ="prob") 
print(prob_tree2)
prob_tree3 <- predict(tree3, projet_ET, type ="vector") 
print(prob_tree3)

#---------------------------------------------------------#
# DEFINITION DE LA METHODE D'EVALUATION DES CLASSIFIEURS  #
#---------------------------------------------------------#
projet_ET$Tree1 <- test_tree1
projet_ET$Tree2 <- test_tree2
projet_ET$Tree3 <- test_tree3
View(projet_ET[,c("RESPONSE","Tree1", "Tree2","Tree3")])

# Calcul des taux de succes pour chaque arbre de décision

ts1 <- nrow(projet_ET[projet_ET$RESPONSE == projet_ET$Tree1,])/nrow(projet_ET)
ts2 <- nrow(projet_ET[projet_ET$RESPONSE == projet_ET$Tree2,])/nrow(projet_ET)
ts3 <- nrow(projet_ET[projet_ET$RESPONSE == projet_ET$Tree3,])/nrow(projet_ET)

# le classifieur d'arbre de décisions le plus performant est 'C5.0' d'après le calcul des taux de succès

#Calcul des matrices de confusion

#Matrice de confusion de rpart()
mc1 <- table(projet_ET$RESPONSE, test_tree1)
R1 <- mc1[2,2]/(mc1[2,2]+mc1[2,1]) # Mesure du Rappel(Sensibilté)
S1 <- mc1[1,1]/(mc1[1,1]+mc1[1,2]) # Mesure de Spécifité
P1 <- mc1[2,2]/(mc1[2,2]+mc1[1,2]) # Mesure de Précision
TVN1 <- mc1[1,1]/(mc1[1,1]+mc1[2,1]) # Mesure du taux de vrais négatifs

#Matrice de confusion de C5.0()
mc2 <- table(projet_ET$RESPONSE, test_tree2)
R2 <- mc2[2,2]/(mc1[2,2]+mc1[2,1]) # Mesure du Rappel(Sensibilté)

S2 <- mc2[1,1]/(mc1[1,1]+mc1[1,2]) # Mesure de Spécifité
P2 <- mc2[2,2]/(mc1[2,2]+mc1[1,2]) # Mesure de Précision

TVN2 <- mc2[1,1]/(mc1[1,1]+mc1[2,1]) # Mesure du taux de vrais négatifs

#Matrice de confusion de tree()
mc3 <- table(projet_ET$RESPONSE, test_tree3)
R3 <- mc3[2,2]/(mc1[2,2]+mc1[2,1]) # Mesure du Rappel(Sensibilté)
S3 <- mc3[1,1]/(mc1[1,1]+mc1[1,2]) # Mesure de Spécifité
P3 <- mc3[2,2]/(mc1[2,2]+mc1[1,2]) # Mesure de Précision
TVN3 <- mc3[1,1]/(mc1[1,1]+mc1[2,1]) # Mesure du taux de vrais négatifs

# D'après le calcul des mesures, l'arbre C5.0 est le Classifieur qui donne une bonne précision,par suite c'est le classifieur le plus performant 

###############
# Courbes ROC #
###############

# Génération des données necessaires pour la courbe ROC
roc1 <- prediction(prob_tree1[,2], projet_ET$RESPONSE)  
print(roc1)
roc2 <- prediction(prob_tree2[,2], projet_ET$RESPONSE)  
print(roc2)

roc3 <- prediction(prob_tree3[,2], projet_ET$RESPONSE)  
print(roc3)

# Calcul des vrais positifs et taux de faux positifs
roc_p1 <- performance(roc1, "tpr","fpr")  # pour rpart()
roc_p2 <- performance(roc2, "tpr","fpr")  # pour C50
roc_p3 <- performance(roc3, "tpr","fpr")  # pour tree()
# Traçage de la courbe
plot(roc_p1, col = "red")  # pour rpart()
plot(roc_p2, add = TRUE, col = "blue")  # pour C50()
plot(roc_p3, add = TRUE, col = "magenta")  # pour tree()

# CALCUL DES INDICES AUC DES COURBES ROC 
#----------------------------------------
# Calcul de l'AUC à partir des données générées : arbre 'rpart()'
auc_tree1 <- performance(roc1, "auc")
str(auc_tree1)  # Affichage de la structure de l'objet 'auc_tree1' généré 
attr(auc_tree1, "y.values")  # Affichage de la valeur de l'AUC stockee dans l'attribut 'y.values' de 'auc_tree1'

# auc_tree1 = 0.54

# Calcul de l'AUC de l'arbre C5.0() 
auc_tree2 <- performance(roc2, "auc")  
attr(auc_tree2, "y.values")  # Affichage de la valeur de l'AUC = 0.62 

# Calcul de l'AUC de l'arbre tree()
auc_tree3 <- performance(roc3, "auc")
attr(auc_tree3, "y.values")  # Affichage de la valeur de l'AUC = 0.53 

# D'après le calcul de l'AUC pour chaque arbre de décison, on constate que l'arbre C50 a l'AUC le plus grand c'est le classifieur le plus performant 



##########################################
# Prédictions de l'appétence des clients #
##########################################
projet_New$MARITAL <- gsub("\\é","e",projet_New$MARITAL)
projet_New$MARITAL <- as.factor(projet_New$MARITAL)
projet_New$EMPCAT <- gsub("\\à","a",projet_New$EMPCAT)
projet_New$EMPCAT <- as.ordered(projet_New$EMPCAT)
projet_New$JOBSAT=gsub("\\è","e",projet_New$JOBSAT)
projet_New$JOBSAT <- as.ordered(projet_New$JOBSAT)
projet_New$INCCAT=gsub("\\à","a",projet_New$INCCAT)
projet_New$INCCAT <- as.ordered(projet_New$INCCAT)
projet_New$INTERNET=gsub("\\é","e",projet_New$INTERNET)
projet_New$INTERNET <- as.factor(projet_New$INTERNET)


#----------------------------------------#
# PREDICTIONS DU CLASSIFIEUR SELECTIONNE #
#----------------------------------------#

# Generation de la classe prédite pour chaque exemple de test pour le classifieur le plus performant C50

pred_C50 <- predict(tree2, projet_New, type = "class")
table(pred_C50)


#Ajout des colones des prédiction
projet_New$Predition <- pred_C50
View(projet_New)

# Construction d'un data frame contenant classe reelle, prediction et probabilités des predictions
df_result <- data.frame(projet_ET$RESPONSE,pred_C50, prob_tree2[,2], prob_tree2[,1])
View(df_result)

# Rennomage des colonnes afin d'en faciliter la lecture et les manipulations
colnames(df_result) = list("RESPONSE","CLASSE PREDITE", "PROBABILITE(Oui)","PROBABILITE(Non)")
View(df_result)


#--------------------------------
# Enregistrement du fichier de resultats au format csv
write.table(projet_New, file='resultatsprojet.csv', sep="\t", dec=".", row.names = F)

write.table(df_result, file='df_result.csv', sep="\t", dec=".", row.names = F) 

