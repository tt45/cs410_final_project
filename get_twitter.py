import tweepy
import csv
import pandas as pd
####input your credentials here
consumer_key =  'AwZGVU4TtZwg07Jfr2jjU8D7d'
consumer_secret = 'iCRFJEvj0vmb4E2dL2syjxjg9lGzEOEGd3M2tbjha8cBZBZ4ZT'
access_token = '919999923663360000-OjZOyOm6i5KQPp1gmBzACk5rOCmJqV2'
access_token_secret = 'DtBh2dfDVvlFEVRmYgRSoMvpvfyaeNyf0RzWNWFAPK74w'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('ua.csv', 'w+')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#unitedAIRLINES",count=100,
                           lang="en",
                           since="2017-04-03").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
