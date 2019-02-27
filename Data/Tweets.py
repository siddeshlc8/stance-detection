import tweepy
from tweepy import OAuthHandler
import numpy as np
import pandas as pd
#TextBlob perform simple natural language processing tasks.
#from textblob import TextBlob


consumer_key = 'CMde5m2cvi0FqRq7FGdcUMkbW'
consumer_secret = 'Tk0huoPBhqa7torgw5t9I7tDKBCF5XdGAL43b1hHzHaA2uUdWz'
access_token = "953993604015579136-zj0S88WcSbCcN2nQCYCxdmmrXa9Tk4g"
access_token_secret = "dVsJExjJWqdHBE4QX0iupvQgiCt7asCUJ9z22wOitG8Jk"

# create OAuthHandler object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# set access token and secret
auth.set_access_token(access_token, access_token_secret)
# create tweepy API object to fetch tweets
api = tweepy.API(auth)




def get_tweets(query, count = 300):

    # empty list to store parsed tweets
    tweets = []
    
    # call twitter api to fetch tweets
    q=str(query)
    a=str(q+" Pulwama")
    b=str(q+" attack")
    c=str(q+" CRPF")
    fetched_tweets = api.search(a, count = count)+ api.search(b, count = count)+ api.search(c, count = count)
    # parsing tweets one by one
    print(len(fetched_tweets))

    for tweet in fetched_tweets:

        tweets.append(tweet.text)
    return tweets

    # creating object of TwitterClient Class
    # calling function to get tweets
tweets = get_tweets(query ="", count = 20000)
tmp = np.array(tweets)
pd.DataFrame(tmp).to_csv("file.csv")