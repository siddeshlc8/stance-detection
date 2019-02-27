import tweepy 
import numpy as np
import pandas as pd

# Fill the X's with the credentials obtained by 
# following the above mentioned procedure. 
consumer_key = 'CMde5m2cvi0FqRq7FGdcUMkbW'
consumer_secret = 'Tk0huoPBhqa7torgw5t9I7tDKBCF5XdGAL43b1hHzHaA2uUdWz'
access_key = "953993604015579136-zj0S88WcSbCcN2nQCYCxdmmrXa9Tk4g"
access_secret = "dVsJExjJWqdHBE4QX0iupvQgiCt7asCUJ9z22wOitG8Jk"

# Function to extract tweets 
def get_tweets(username): 
		
		# Authorization to consumer key and consumer secret 
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 

		# Access to user's access key and access secret 
		auth.set_access_token(access_key, access_secret) 

		# Calling api 
		api = tweepy.API(auth) 

		# 200 tweets to be extracted 
		number_of_tweets=200
		tweets = api.user_timeline(screen_name=username) 

		# Empty Array 
		tmp=[] 

		# create array of tweet information: username, 
		# tweet id, date/time, text 
		tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created 
		for j in tweets_for_csv: 

			# Appending tweets to the empty array tmp 
			tmp.append(j) 

		# Printing the tweets 
		tmp = np.array(tmp)

		pd.DataFrame(tmp).to_csv("file.csv")



# Driver code 
if __name__ == '__main__': 

	# Here goes the twitter handle for the user 
	# whose tweets are to be extracted. 
	get_tweets("victorjames663") 