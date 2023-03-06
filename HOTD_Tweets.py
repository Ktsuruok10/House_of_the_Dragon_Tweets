#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:29:16 2022

@author: kellentsuruoka
Homework 2: Semi-structured Data
Analyzing Tweets about HBO House of the Dragon
"""

import snscrape.modules
import snscrape.modules.twitter as sntwitter
import pandas as pd


## Collecting Tweets using snscrape
tweets_list2=[]
query2 = '''(House of Dragon OR #HOTD) until:2022-08-22 since:2022-08-21  lang:en'''
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query=query2).get_items()): 
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.lang]) 
    

## Adding tweets to dataframe for analysis 
tweets_df3 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet_Id', 'Text', 'User','Language'])

print(tweets_df3)


import re
## Using regular expressions package to remove usernames and links from tweets. Created function that inputs tweets and removes usernames and urls
def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    return tweet
tweets_df3['Text'] = tweets_df3['Text'].apply(remove_usernames_links)

print(tweets_df3['Text'])


#Calculating how many different languages tweeted about hotd
tweets_per_language = tweets_df3.groupby(['Language'])['Text'].count()
#exporting to CSV file 
tweets_per_language.to_csv('HOTD_Tweets_per_lang.csv')

#Using resample to calculate tweets perminute 
tweets_per_min = tweets_df3.resample('T', on='Datetime').Tweet_Id.count()

len(tweets_per_min)
print(tweets_per_min)
tweets_per_min.to_csv('tweets_per_min.csv')


##Tokenizing Tweets 

from nltk.tokenize import word_tokenize
word_tokens = []
for tweet in tweets_df3['Text']:
    lower = tweet.lower()
    word_tokens.append(word_tokenize(lower))

## For loop to count the number of characters mentioned in the tweets e

viserys = 0
daemon = 0
rhaenyra = 0
alicent = 0
otto = 0 

for word in word_tokens:
    for w in word: 
        if w == 'viserys':
            viserys +=1
        elif w == 'daemon':
            daemon +=1
        elif w == 'rhaenyra':
            rhaenyra +=1
        elif w == 'alicent':
            alicent +=1
        elif w == 'otto':
            otto +=1



#Analyzing sentiment from tweets list and storing it in tweet_sentiment list
from textblob import TextBlob, Word, Blobber
tweet_sentiment = []
for tweet in tweets_df3['Text']:
    s = TextBlob(tweet)
    analysis = s.sentiment
    polarity = analysis.polarity
    tweet_sentiment.append(polarity)
    
#For loop to count how many tweets were positive, negative or neutral 
pos = 0
neutral = 0
neg = 0
for tweet in tweet_sentiment:
    if tweet > 0:
        pos +=1 
    elif tweet == 0:
        neutral +=1
    else:
        neg+=1
        
##Creating wordcloud on House of Dragon Tweets 

import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
# Create stopword list:
stop_words = set (stopwords.words( 'english' ))
stop_words.update(["br", "href"])
## 
textt = " ".join(tweet for tweet in tweets_df3.Text)
wordcloud = WordCloud(stopwords=stop_words).generate(textt)
plt.imshow(wordcloud, interpolation='mitchell')
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()

#Creating percentages of positive, neutral, or negative tweets 
positive = (round((pos/len(tweet_sentiment)),1)*100)
negative = (round((neg/ len(tweet_sentiment)),1)*100)
neutral2 = (round((neutral/len(tweet_sentiment)),1)*100)

#Creating labels for pie chart using fstrings 
pos_label = f"Positive {str(positive)}%"
neutral_label = f"Neutral {str(neutral2)}%"
neg_label = f"Negative {str(negative)}%"

import matplotlib.pyplot as plt

#Creating PieCart for sentiment analysis 
labels = [ 'Positive 28%','Neutral 56%','Negative 15%']
sizes = [pos, neutral, neg]
colors = ['yellowgreen', 'blue','red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title('Sentiment Analysis Result for House of the Dragon' )
plt.axis('equal')
plt.show()

