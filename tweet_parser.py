'''This is a module designed to parse through tweets'''
import pandas as pd
from customdatetimecreator import CustomTime

def tweet_adequacy_parser(tweets, words):
    ''' Parse through a set of tweets adding those that match a
    given set of criteria to a new dataframe
    '''
    numberframe = pd.DataFrame()
    tweetframe = pd.DataFrame()
    for word in words:

        parsed_tweets=\
                        tweets[tweets['Tweets'].str.contains\
                        (r'\b{}\b'.format(word),na=False)]
        # Creating a Date column
        parsed_tweets['Date'] = parsed_tweets.Date.map(CustomTime.date_creator2)
        # Creating the number of tweets per day column (NTPD)
        tweets_dataframe = parsed_tweets.Date.value_counts().to_frame('{}'.format(word))
        tweets_dataframe.sort_index(ascending=True, inplace = True)
        numberframe = pd.concat([numberframe,tweets_dataframe], axis=1)
        tweetframe = pd.concat([tweetframe, parsed_tweets], axis=0)

    # Sum of retweets in a day
    retweet_sum = tweetframe.groupby('Date')['Retweets'].sum()
    replies_sum = tweetframe.groupby('Date')['Replies'].sum()
    favourites_sum = tweetframe.groupby('Date')['Favorites'].sum()
    # Temporary retweet sum df
    rts = pd.DataFrame(retweet_sum)
    rpls = pd.DataFrame(replies_sum)
    favs = pd.DataFrame(favourites_sum)
    rts.columns = ['RetweetSum']
    rpls.columns = ['RepliesSum']
    favs.columns = ['FavoritesSum']
    tweetframe.set_index('Date', inplace=True)
    tweetframe.sort_index(ascending=True, inplace=True)
    tweetframe.drop_duplicates(subset="Tweets", keep=False, inplace=True,)
    dateandtime = tweetframe['Date and Time']
    dandt = pd.DataFrame(dateandtime)
    dandt.columns = ['Date and Time']
    # Joining the temporary frame with retweets with the FinalTweetsDF
    numberframe = numberframe.join(rts)
    numberframe = numberframe.join(rpls)
    numberframe = numberframe.join(favs)
    numberframe = pd.concat([numberframe, dandt], axis=1, join='inner')
    numberframe.fillna(0, inplace=True)
    numberframe = numberframe.loc[~numberframe.index.duplicated(keep='first')]
    print("\n")
    print("Tweet dataframe parsed and sorted")
    return numberframe
