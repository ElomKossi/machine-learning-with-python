# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 13:38:17 2018

@author: Koffi Moïse AGBENYA

Ici nous allons créer un modèle pour une entreprise de télécommunication,
afin de prédire le moment où ses clients partiront chez un concurrent,
afin qu'ils puissent prendre des mesures pour conserver leurs clients.

Sujet : 
    Une entreprise de télécommunications s'inquiète du nombre de clients 
    qui abandonnent leurs activités de ligne terrestre (téléphonie fixe)
    pour leurs concurrents. Ils ont besoin de comprendre ceux qui partent.

A propos du jeu de données :
Nous utiliserons des données de télécommunication pour prévoir le désabonnement
des clients. Il s'agit d'une donnée client historique dans laquelle chaque 
ligne représente un client. Les données sont relativement faciles à comprendre 
et vous pouvez découvrir des informations que vous pouvez utiliser 
immédiatement. Il est généralement moins coûteux de conserver des clients que 
d’en acquérir de nouveaux. Par conséquent, cette analyse a pour objectif de 
prédire les clients qui resteront au sein de la société.

Cet ensemble de données fournit des informations pour vous aider à prévoir le 
comportement permettant de fidéliser les clients. Vous pouvez analyser toutes 
les données client pertinentes et développer des programmes ciblés de 
fidélisation de la clientèle.

L'ensemble de données comprend des informations sur:

les clients ayant quitté le mois dernier
- la colonne s'appelle Churn
Services auxquels chaque client s'est abonné 
- téléphone, lignes multiples, Internet, sécurité en ligne, sauvegarde en ligne, 
protection des appareils, support technique, streaming TV et films
Informations sur le compte client 
- depuis combien de temps est-il client, contrat, méthode de paiement, 
facturation sans papier, frais mensuels et frais totaux
Informations démographiques sur les clients 
- sexe, tranche d'âge et s'ils ont des partenaires et des personnes à charge
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Load data from path
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv"
churn_df = pd.read_csv(path)
churn_df.head()

#Data Pre-processing and selection
#Lets select some features for the modeling.
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
#Useful for scikit learn
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

#Lets define X, and y for our dataset:

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5])

y = np.asarray(churn_df['churn'])
print(y[0:5])

#Normalization of the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#Lets split our dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Modeling
#Lets build our model using LogisticRegression from Scikit-learn package. This 
#function implements logistic regression and can use different numerical 
#optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, 
#‘sag’, ‘saga’ solvers.

from sklearn.linear_model import LogisticRegression
#C=0,01 is use for regularization
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

#Now we can predict using our test set:
yhat = LR.predict(X_test)
yhat
#predict_proba returns estimates for all classes, ordered by the label of 
#classes. So, the first column is the probability of class 1, P(Y=1|X), and 
#second column is probability of class 0, P(Y=0|X):
yhat_prob = LR.predict_proba(X_test)
yhat_prob

#Evaluation
#Jacard index
#Lets try jaccard index for accuracy evaluation. we can define jaccard as the 
#size of the intersection divided by the size of the union of two label sets. 
#If the entire set of predicted labels for a sample strictly match with the 
#true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


#Confusion matrix
#Another way of looking at accuracy of classifier is to look at confusion matrix.
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

#Look at first row. The firsr row is for customers whose actual churn value in 
#test set is 1. As you can calculate, out of 40 customers, the churn value of 
#15 of them is 1. And out of these 15, the classifier correctly predicted 6 of 
#them as 1, and 9 of them as 0.
#It means, for 6 customers, the actual churn value were 1 in test set, and 
#classifier also correctly predicted those as 1. However, while the actual 
#label of 9 customers were 1, the classifier predicted those as 0, which is not 
#very good. We can consider it as error of the model for first row.
#What about the customers with churn value 0? Lets look at the second row. It 
#looks like there were 25 customers whom their churn value were 0.
#The classifier correctly predicted 24 of them as 0, and one of them wrongly as 
#1. So, it has done a good job in predicting the customers with churn value 0. 
#A good thing about confusion matrix is that shows the model’s ability to 
#correctly predict or separate the classes. In specific case of binary 
#classifier, such as this example, we can interpret these numbers as the count 
#of true positives, false positives, true negatives, and false negatives.

print (classification_report(y_test, yhat))

#Based on the count of each section, we can calculate precision and recall of 
#each label:
#Precision is a measure of the accuracy provided that a class label has been 
#predicted. It is defined by: precision = TP / (TP + FP)
#Recall is true positive rate. It is defined as: Recall = TP / (TP + FN)
#So, we can calculate precision and recall of each class.
#F1 score: Now we are in the position to calculate the F1 scores for each label 
#based on the precision and recall of that label.
#The F1score is the harmonic average of the precision and recall, where an F1 
#score reaches its best value at 1 (perfect precision and recall) and worst at 
#0. It is a good way to show that a classifer has a good value for both recall 
#and precision.
#And finally, we can tell the average accuracy for this classifier is the 
#average of the f1-score for both labels, which is 0.72 in our case.


#log loss
#Now, lets try log loss for evaluation. In logistic regression, the output can 
#be the probability of customer churn is yes (or equals to 1). This probability 
#is a value between 0 and 1. Log loss( Logarithmic loss) measures the 
#performance of a classifier where the predicted output is a probability value 
#between 0 and 1.

from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)

