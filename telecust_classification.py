# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:20:36 2018

@author: Koffi Moïse AGBENYA

Classification avec les K plus proches voisins (KNN)


Dans cet atelier,nous allons charger un jeu de données client lié à une 
entreprise de télécommunication, le nettoyer, utiliser KNN 
(K-Nearest Neighbours) pour prédire la catégorie de clients. Et évaluer 
la précision de notre modèle. 


Un fournisseur de télécommunications segmente sa clientèle en 
fonction des modèles d'utilisation des services, classant les clients en 
quatre groupes. Si les données démographiques peuvent être utilisées pour 
prévoir l'appartenance à un groupe, l'entreprise peut personnaliser les 
offres pour les clients potentiels individuels. C'est un problème de 
classification. En d’autres termes, étant donné le jeu de données et les 
étiquettes prédéfinies, nous devons créer un modèle à utiliser pour prédire 
la classe d’un cas nouveau ou inconnu.

L'exemple se concentre sur l'utilisation de données démographiques, telles 
que la région, l'âge et les relations matrimoniales, pour prédire les modèles
d'utilisation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

"""
Le champ cible, appelé __custcat__, a quatre valeurs possibles correspondant 
aux quatre groupes de clients, comme suit:
   1- Service de base
   2- E-Service
   Service 3- Plus
   4- Service total

Notre objectif est de construire un classificateur, afin de prédire la classe 
des cas inconnus. Nous allons utiliser un type spécifique de classification 
appelé K plus proche voisin.
"""

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv"

df = pd.read_csv(path)

#Affichage des 5 premières lignes pour voir la forme des données
print(df.head())

#Data visualization and analysis
# Let’s see how many of each class is in our data set
print(df['custcat'].value_counts())

#Lets defind feature sets, X:
X = df[['region', 'tenure','age', 'marital', 'address', 'income',
        'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5]) # Juste vérifier que tout c'est bien passé

y = df['custcat'].values
print(y[0:5])

#Normalisation des données

#La normalisation des données donne des données à moyenne nulle et à variance 
#d'unité. Il s'agit d'une bonne pratique, en particulier pour des algorithmes 
#tels que KNN, basés sur la distance des cas:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#Entraînement de la partie des tests

#La précision hors échantillon est le pourcentage de prédictions correctes 
#effectuées par le modèle sur des données sur lesquelles le modèle n'a PAS 
#été formé. Faire une formation et un test sur le même jeu de données aura 
#très probablement une faible précision hors échantillon, en raison de la 
#probabilité d'être surajusté.
#Il est important que nos modèles aient une précision élevée, hors 
#échantillon, car le but de tout modèle est bien sûr de faire des 
#prédictions correctes sur des données inconnues. Alors, comment 
#pouvons-nous améliorer la précision hors échantillon? L’une des méthodes 
#consiste à utiliser une méthode d’évaluation appelée répartition par 
#train / test. La division Train / Test consiste à fractionner le jeu de 
#données en ensembles de formation et de test, qui s'excluent mutuellement. 
#Après quoi, vous vous entraînez avec le kit de formation et testez avec 
#le kit de test.
#Cela fournira une évaluation plus précise de la précision hors échantillon 
#car le jeu de données de test ne fait pas partie du jeu de données utilisé 
#pour former les données. C'est plus réaliste pour les problèmes du monde réel.


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#CLASSIFICATION

#K nearest neighbor (K-NN)

#Importer la bibliothèque

#Classificateur implémentant le vote des k-voisins les plus proches.

# Entraînement

# Commençons l’algorithme avec k = 4 pour le moment:

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#Prédiction

# nous pouvons utiliser le modèle pour prédire l'ensemble de tests:

yhat = neigh.predict(X_test)
print(yhat[0:5]) # Test de bon fonctionnement

# Évaluation de la précision

#Dans la classification multilabel, la fonction de score de la classification 
#de précision calcule la précision du sous-ensemble. Cette fonction est égale 
#à la fonction jaccard_similarity_score. Essentiellement, il calcule la 
#correspondance entre les étiquettes réelles et les étiquettes prédites dans 
#l'ensemble de test.


print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# A propos des autres K

#Nous pouvons calculer la précision de KNN pour différents K

#Nous allons utiliser différente valeur de k de 1 à 9
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

# Précision du modèle de tracé pour un nombre différent de voisins

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Naighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

