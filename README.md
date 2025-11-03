# 🧩 Prédiction de l'appétence de clients

## 🎯 Objectif
Projet réalisé dans le cadre du module **Data Mining & Machine Learning** (Université Côte d'Azur).  
But : analyser les caractéristiques clients, segmenter la clientèle et construire un modèle de classification pour prédire l'appétence (probabilité de réponse positive) à une offre promotionnelle.

---

## 📊 Jeu de données
- **Data_Projet.csv** : 6400 clients (variable cible `RESPONSE` connue).  
- **Data_Projet_New.csv** : 300 clients (à prédire).  
- **Variables** : 28 variables socio-démographiques et comportementales (AGE, INCOME, MARITAL, INTERNET, NEWS, CAR, CARCAT, ED, EMPLOY, OWNPC, MULTLINE, …).  
- **Cible** : `RESPONSE` (1 = réponse positive, 0 = non).

---

## 🧭 Méthodologie
1. **Exploration** : tests d'indépendance (chi², Fisher), corrélations (Pearson/Spearman), importance des variables (CORElearn / information gain).  
2. **Clustering** : K-means (partitionnement) pour explorer la segmentation (K testé de 4 à 10).  
3. **Gestion du déséquilibre** : sous/échantillonnage aléatoire pour rééquilibrer les classes avant apprentissage.  
4. **Classification supervisée** : entraînement et comparaison de classifieurs d'arbres (`rpart`, `C5.0`, `tree`) et évaluation par matrice de confusion, rappel, précision, spécificité et AUC (ROC).  
5. **Sélection du modèle** : choix selon métriques (préférer minimiser le risque de faux négatifs selon l'objectif métier).

---

## 🧾 Résultats clés
- Taille de l’échantillon d’apprentissage : ≈ 1304 observations.  
- Performances (AUC pour les arbres) :
  - `rpart()` : AUC ≈ 0.54  
  - `C5.0()`  : AUC ≈ 0.62  ← **meilleure AUC**  
  - `tree()`  : AUC ≈ 0.53  
- **Modèle final** : `C5.0`  
- Difficultés rencontrées : incompatibilités de packages, nettoyage des variables, rééchantillonnage.

---

## 🧰 Environnement
- **Langage :** R  
- **Bibliothèques :** tidyverse, caret, C50, rpart, tree, CORElearn, e1071, kknn, nnet, pROC  
- **IDE :** RStudio

---

## 📁 Organisation du dépôt

prediction-appetence-clients/
│
├── data/
│ ├── Data_Projet.csv
│ ├── Data_Projet_New.csv
│ └── README.md
│
├── src/
│ └── modele_classification.R
│
├── notebooks/
│ └── exploration.Rmd
│
├── results/
│ ├── predictions.csv
│ ├── confusion_matrix_C5.0.png
│ └── performance_metrics.txt
│
├── LICENSE
├── .gitignore
└── README.md


---

## 👩‍💻 Auteurs
**Yannick KOULONI** — Ingénieur Mathématicien, spécialité Ingénierie Numérique
**BEDJA M. S. Awadi** — Chef de Projet IA, spécialité Ingénierie Numérique

---

## 🏫 Contexte académique
Projet réalisé entre **octobre et décembre 2022**,  
sous la supervision de **Nicolas Pasquier**,  
dans le cadre du cours *Data Mining & Machine Learning* — Université Côte d’Azur.
