# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:05:44 2018

@author: koffi Moïse Agbenya

Nous allons utiliser ici  l'algorithme de l'arbre de décision. Il servira 
de classification pour créer un modèle à partir de données historiques de 
patients et de leurs réponses à différents médicaments. Ensuite, nous 
utiliserons l’arbre décisionnel formé pour prédire la classe d’un patient 
inconnu ou pour trouver le médicament approprié pour un nouveau patient.



Imaginons que je suis un chercheur médical en train de compiler des 
données pour une étude. Nous avons collecté des données sur un ensemble
de patients souffrant tous de la même maladie. Au cours de leur traitement, 
chaque patient a répondu à l’un des cinq médicaments suivants: médicament A, 
médicament B, médicament c, médicament x et y. 

Une partie du travail consiste à créer un modèle pour déterminer quel 
médicament pourrait être approprié pour un futur patient souffrant de la même 
maladie. Les ensembles de caractéristiques de cet ensemble de données sont l’âge, 
le sexe, la pression artérielle et le cholestérol des patients, et la cible est 
le médicament auquel chaque patient a répondu. Il s'agit d'un exemple de 
classificateur binaire. Nous utiliserons la partie formation de l'ensemble de 
données pour créer un arbre de décision, puis l'utiliser pour prédire la classe 
d'un patient inconnu ou pour le prescrire à un nouveau patient.

"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv"

df = pd.read_csv(path)

print(df.head())

# Data preprocessing

#Create the matrix X

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Sex, BP and Cholesterol are categorical variable. Let's convert these features
# to numerical values

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_chol.transform(X[:,3])


y = df["Drug"]

print("preprocessing Done!")

#Setting up the decision tree

#We will be using train/test split on our decision tree.

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#Modeling
#Instance of DTC
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

#we will fit the data with the training feature matrix X_trainset and training
#response vector y_trainset
drugTree.fit(X_trainset,y_trainset)

#Prediction
#Let's make some predictions on the testing dataset and store it into a 
#variable called predTree.
predTree = drugTree.predict(X_testset)

#Comparison of the result
print (predTree [0:5])
print (y_testset [0:5])

#Evaluation
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#Visualization

dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames,
                         out_file=dot_data, class_names= np.unique(y_trainset),
                         filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')