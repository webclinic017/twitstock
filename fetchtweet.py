'''A module to fetch tweets'''
import time
import pandas as pd
import GetOldTweets3 as got
import snscrape.modules.twitter

def tweet_fetch(name, dates):
    ''' Function to fetch tweets for an individual
    over a certain time frame, then seperate into
    a pandas dataframe with all tweet attributes in seperate columns

    inputs:
        name: twitter handle of individual
        dates: dates in the format of an encapusualted list
        i.e. [('YYYY-MM-DD','YYYY-MM-DD),('YYYY-MM-DD','YYYY-MM-DD)...]
    '''
    all_tweets = pd.DataFrame()
    for since,until in dates:
        tweet_criteria = got.manager.TweetCriteria().setUsername(name)\
                                                    .setSince(since)\
                                                    .setUntil(until)\
                                                    .setMaxTweets(10000)\
                                                    .setEmoji("unicode")

        # Fetches 10,000 tweets at a time to prevent both timeout
        # and too many request errors from twitter.
        #
        # Dataframe is created below with tweets,
        # favourites and replies in seperate columns'
        #
        # The data and time of the tweet is included in a seperate
        # column to allow for indexing, and to later have a refernce
        # for the timing of the movements relative to the tweets

        tweets = got.manager.TweetManager.getTweets(tweet_criteria)
        tweets_df = pd.DataFrame([tweet.text for tweet in tweets],
            columns=['Tweets'])

        favorites_list = list([tweet.favorites for tweet in tweets])
        tweets_df['Favorites'] = favorites_list
        tweets_df['Favorites'] = pd.to_numeric(tweets_df['Favorites'])

        retweets_list = list([tweet.retweets for tweet in tweets])
        tweets_df['Retweets'] = retweets_list
        tweets_df['Retweets'] = pd.to_numeric(tweets_df['Retweets'])

        replies_list = list([tweet.replies for tweet in tweets])
        tweets_df['Replies'] = replies_list
        tweets_df['Replies'] = pd.to_numeric(tweets_df['Replies'])

        dates_list = list(tweet.formatted_date for tweet in tweets)
        tweets_df['Date and Time'] = dates_list

        # Making the string of the tweets lowercase so that
        # when passing through for key words none are missed

        tweets_df.Tweets = tweets_df['Tweets'].map(lambda x: x.lower())

        # Reformating date and time into seperate columns so that
        # we can sum the amount of tweets in a single day

        tweets_df['Date'] = tweets_df['Date and Time'].map(lambda x: x[:10]+x[-5:])
        tweets_df['Time'] = tweets_df['Date and Time'].map(lambda x: x[11:19])

        all_tweets = all_tweets.append(tweets_df, ignore_index=True)
        time.sleep(60)

        print("Twitter data fetch successful for the period {} - {}".format(since,until))
    return all_tweets
    
#def new_fetch_tweet(name, dates):
#   '''New twitter scraper as GetOldTweets3 has been made obsolete
#    - pending update may allow for new endpoint for GOT fix
#    '''
#    all_tweets = pd.DataFrame()
#    
#    for tweet in snscrape.modules.twitter.TwitterSearchScraper('corona lang:en').get_items):
        
    
    
