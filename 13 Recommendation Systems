import pandas as pd
import numpy as np

#naming the columns of the data
column_names=['user_id','item_id','rating','timestamp']
#reading the data part1
df=pd.read_csv('u.data',sep='\t',names=column_names)
df.head()
#reading data part2
movie_titles=pd.read_csv('movie_id_title.csv')
movie_titles.head()
#merging two data sets based on the title id
df=pd.merge(df,movie_titles,on='item_id')
df.head()

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

#grouping the data by title and checking the ratings mean
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
#grouping the data by title and checking rtaings count
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

#creating the ratings into a df by means
ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
#creating a num_of ratings col by count of ratings
ratings['num_of_ratings']=pd.DataFrame(df.groupby('title')['rating'].count())

ratings['num_of_ratings'].hist(bins=70)
ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num_of_ratings',data=ratings,alpha=0.5)

#creating a matrix by pivot table for title by user id ratings
moviemat=df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()
ratings.sort_values('num_of_ratings',ascending=False).head(10)

#creating subsets for two diff movies
starwars_u_ratings=moviemat['Star Wars (1977)']
liarliar_u_rating=moviemat['Liar Liar (1997)']
starwars_u_ratings.head()

#checking correlation based on ratings bu userid
similar_to_starwars=moviemat.corrwith(starwars_u_ratings)
similar_to_starwars
#creating subset of correlation with liarliar
similar_to_liarliar=moviemat.corrwith(liarliar_u_rating)
similar_to_liarliar

#creating subset ceated above to dataframe
corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])
#dropping na
corr_starwars.dropna(inplace=True)
corr_starwars.head()
#adding num of ratings column to the df correlation 
corr_starwars=corr_starwars.join(ratings['num_of_ratings'])
corr_starwars.head()
#filtering the  correlation based on the min number of ratings to a movie
corr_starwars[corr_starwars['num_of_ratings']>100].sort_values('correlation',ascending=False).head()

corr_liarliar=pd.DataFrame(similar_to_liarliar,columns=['correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar=corr_liarliar.join(ratings['num_of_ratings'])
corr_liarliar[corr_liarliar['num_of_ratings']>90].sort_values('correlation',ascending=False).head()
