# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 18:39:53 2018

@author: Koffi Moïse AGBENYA

Customer segmentation with K-Means

Some real-world applications of k-means:

Customer segmentation 
Understand what the visitors of a website are trying to accomplish
Pattern recognition
Machine learning
Data compression

Here, we practice k-means clustering with 2 examples:

k-means on a random generated dataset
Using k-means for customer segmentation


"""


import random 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs

#k-Means on a randomly generated dataset

#Lets create our own dataset
np.random.seed(0)
#Next we will be making random clusters of points by using the make_blobs class. 
#The make_blobs class can take in many inputs, but we will be using these 
#specific ones.

#Input

# n_samples: The total number of points equally divided among clusters.
# Value will be: 5000
# centers: The number of centers to generate, or the fixed center locations.
# Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
# cluster_std: The standard deviation of the clusters.
# Value will be: 0.9

#Output
# X: Array of shape [n_samples, n_features]. (Feature Matrix)
# The generated samples.
# y: Array of shape [n_samples]. (Response Vector)
# The integer labels for cluster membership of each sample.

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')


#Setting up K-Means
#Now that we have our random data, let's set up our K-Means Clustering.
#The KMeans class has many parameters that can be used, but we will be using 
#these three:
#init: Initialization method of the centroids.
#Value will be: "k-means++"
#k-means++: Selects initial cluster centers for k-mean clustering in a smart 
#way to speed up convergence.
#n\_clusters: The number of clusters to form as well as the number of centroids 
#to generate.
#Value will be: 4 (since we have 4 centers)
#n\_init: Number of time the k-means algorithm will be run with different 
#centroid seeds. The final results will be the best output of n\_init 
#consecutive runs in terms of inertia.
#Value will be: 12
#Initialize KMeans with these parameters, where the output parameter 
#is called k_means.

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
#Now let's fit the KMeans model with the feature matrix we created above, X
k_means.fit(X)
#Now let's grab the labels for each point in the model using KMeans' .labels_ 
#attribute and save it as k_means_labels
k_means_labels = k_means.labels_
k_means_labels
#We will also get the coordinates of the cluster centers using KMeans' 
#.cluster_centers_ and save it as k_means_cluster_centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#Creating the Visual Plot
#So now that we have the random data generated and the KMeans model 
#initialized, let's plot them and see what it looks like!
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()



"""
#Customer Segmentation with K-Means

We have a customer dataset, and we need to apply customer segmentation on this 
historical data. Customer segmentation is the practice of partitioning a 
customer base into groups of individuals that have similar characteristics. It 
is a significant strategy as a business can target these specific groups of 
customers and effectively allocate marketing resources. For example, one group 
might contain customers who are high-profit and low-risk, that is, more likely 
to purchase products, or subscribe for a service. A business task is to 
retaining those customers. Another group might include customers from 
non-profit organizations. And so on.

"""
import pandas as pd
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv"
cust_df = pd.read_csv(path)
print(cust_df.head())

#Pre-processing
#Address in this dataset is a categorical variable. k-means algorithm isn't 
#directly applicable to categorical variables because Euclidean distance 
#function isn't really meaningful for discrete variables. So, lets drop this 
#feature and run clustering.

df = cust_df.drop('Address', axis=1)
print(df.head())

#Normalizing over the standard deviation
#Now let's normalize the dataset. But why do we need normalization in the first 
#place? Normalization is a statistical method that helps mathematical-based 
#algorithms to interpret features with different magnitudes and distributions 
#equally. We use tandardScaler() to normalize our dataset.

from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

#Modeling
#In our example (if we didn't have access to the k-means algorithm), it would 
#be the same as guessing that each customer group would have certain age, 
#income, education, etc, with multiple tests and experiments. However, using 
#the K-means clustering we can do all this process much easier.
#Lets apply k-means on our dataset, and take look at cluster labels.

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

#Insights
#Insights
df["Clus_km"] = labels
df.head(5)
#We can easily check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()
#Now, lets look at the distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

#k-means will partition your customers into mutually exclusive groups, for 
#example, into 3 clusters. The customers in each cluster are similar to each 
#other demographically. Now we can create a profile for each group, considering 
#the common characteristics of each cluster. For example, the 3 clusters can be:

# AFFLUENT, EDUCATED AND OLD AGED
# MIDDLE AGED AND MIDDLE INCOME
# YOUNG AND LOW INCOME

