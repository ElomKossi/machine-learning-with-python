# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:12:26 2019

@author: Koffi Moïse AGBENYA

Filtering Collaraborative

ABOUT DATASET

This dataset (ml-latest) describes 5-star rating and free-text tagging activity 
from [MovieLens](http://movielens.org), a movie recommendation service. It 
contains 22884377 ratings and 586994 tag applications across 34208 movies. 
These data were created by 247753 users between January 09, 1995 and January 
29, 2016. This dataset was generated on January 29, 2016.

Users were selected at random for inclusion. All selected users had rated at 
least 1 movies. No demographic information is included. Each user is 
represented by an id, and no other information is provided.

The data are contained in four files, `links.csv`, `movies.csv`, `ratings.csv` 
and `tags.csv`. More details about the contents and use of all these files 
follows.

This is a *development* dataset. As such, it may change over time and is not an 
appropriate dataset for shared research results.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

#Storing movies informations in pandas dataframe
movies_df = pd.read_csv('movies.csv')

#Storing ratings informations in pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

print(movies_df.head())

#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())


print(movies_df.head())

#Dropping the genres column
movies_df = movies_df.drop('genres', 1)
print(movies_df.head())

# let's look at the ratings dataframe.
print(ratings_df.head())

#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())

"""
Collaborative Filtering

The first technique we're going to take a look at is called Collaborative 
Filtering, which is also known as User-User Filtering. As hinted by its 
alternate name, this technique uses other users to recommend items to the input 
user. It attempts to find users that have similar preferences and opinions as 
the input and then recommends items that they have liked to the input. There 
are several methods of finding similar users (Even some making use of Machine 
Learning), and the one we will be using here is going to be based on the 
Pearson Correlation Function.


The process for creating a User Based recommendation system is as follows:

Select a user with the movies the user has watched
Based on his rating to movies, find the top X neighbours
Get the watched movie record of the user for each neighbour.
Calculate a similarity score using some formula
Recommend the items with the highest score


"""


userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
print(inputMovies)


#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
print(inputMovies)

#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userSubset.head())

#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

#lets look at one of the users, e.g. the one with userID=1130
userSubsetGroup.get_group(1130)

#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

#Lets look at the first user
print(userSubsetGroup[0:3])

#Similarity of users to input user
#Next, we are going to compare all users (not really all !!!) to our specified 
#user and find the one that is most similar.
#we're going to find out how similar each user is to the input through the 
#Pearson Correlation Coefficient. It is used to measure the strength of a 
#linear association between two variables. The formula for finding this 
#coefficient between sets X and Y with N values can be seen in the image below.

#We will select a subset of users to iterate through. This limit is imposed
#because we don't want to waste too much time going through every single user.

userSubsetGroup = userSubsetGroup[0:100]

#Now, we calculate the Pearson Correlation between input user and subset group, 
#and store it in a dictionary, where the key is the user Id and the value is 
#the coefficient
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonCorrelationDict.items()

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print(pearsonDF.head())

#The top x similar users to input user
#Now let's get the top 50 users that are most similar to the input.

topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
print(topUsers.head())

#Now, let's start recommending movies to the input user.
#Rating of selected users to all movies
#We're going to do this by taking the weighted average of the ratings of the 
#movies using the Pearson Correlation as the weight. But to do this, we first 
#need to get the movies watched by the users in our pearsonDF from the ratings 
#dataframe and then store their correlation in a new column 
#called _similarityIndex". This is achieved below by merging of these two tables.

topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
print(topUsersRating.head())

#Now all we need to do is simply multiply the movie rating by its weight (The 
#similarity index), then sum up the new ratings and divide it by the sum 
#of the weights.

#We can easily do this by simply multiplying two columns, then grouping up the 
#dataframe by movieId and then dividing two columns:

#It shows the idea of all similar users to candidate movies for the input user:

#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
print(topUsersRating.head())

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
print(tempTopUsersRating.head())

#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
print(recommendation_df.head())

#Now let's sort it and see the top 20 movies that the algorithm recommended!

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))

movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]


"""

Advantages and Disadvantages of Collaborative Filtering

Advantages

Takes other user's ratings into consideration
Doesn't need to study or extract information from the recommended item
Adapts to the user's interests which might change over time

Disadvantages

Approximation function can be slow
There might be a low of amount of users to approximate
Privacy issues when trying to learn the user's preferences

"""