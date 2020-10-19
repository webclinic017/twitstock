#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
'''Program to predict stock market movements using twitter

Author - Joseph Franks
email - josephoefranks@icloud.com
'''
import os
import os.path
import warnings
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.tree import DecisionTreeRegressor
from IPython.core.interactiveshell import InteractiveShell

import fetchtweet
import stockdatafetch
import tweet_parser
import centralbankdata
from customdatetimecreator import CustomTime

InteractiveShell.ast_node_interactivity = "all"
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('max_rows', 10)
pd.set_option('max_columns', 10)
pd.set_option('max_colwidth', 100)
np.random.seed(12345)

print("Packages and modules successfully imported\n")
print("Program running\n")

# TWITTER DATA

# checking to see if tweet data has been fetched previously
# and if it hasn't, pulling the tweets for the specified account and date
# period and storing them in a .csv file for speed, readability and repition
'''
[('2017-01-27','2018-01-27'),
('2018-01-27','2019-01-27'),
('2019-01-27','2020-01-27'),
('2020-01-27', todays_date())]
'''
handle = input("Enter twitter handle: ") # realdonaldtrump
print("\n")

try:
    os.path.exists('./{}_tweet_data.csv'.format(handle))
    print("A data set for the chosen handle already exists\n")
    AllTweets = pd.read_csv("{}_tweet_data.csv".format(handle))
    #append to current tweet df with any new tweets captured after algo was last run
except:
    print("No data set currently exists for the chosen handle")
    print("Program will fetch Twitter data for the chosen handle and store for later use")
    print("Fetching {} twitter data".format(handle))
    dategroups = input("Enter dategroups: ")
    AllTweets = fetchtweet.tweet_fetch(handle, dategroups)
    AllTweets.to_csv(r'./{}_tweet_data.csv'.format(handle), header=True)

words = (
    'business', 'tax', 'employment', 'jobs', 'regulation',
    'derergulation', 'trillion', 'billion', 'export', 'import',
    'unemployment', 'interest rates', 'the fed', 'the sec', 'the dow',
    'stocks', 'stock', 'stock market', 'nasdaq', 'dow jones',
    'federal reserve', 'tarrifs', 's&p 500',
    )

filtered_trump_tweets = tweet_parser.tweet_adequacy_parser(AllTweets, words)

# STOCK INDEX DATA

print("Attempting to fetch stock data from ALPHAVANTAGE API\n")
symbols = ['DJI','GSPC','IXIC']
AVDF = stockdatafetch.DailyFetch(symbols,'TIME_SERIES_DAILY_ADJUSTED').alphavantage_stock_data() # alphavantage dataframe

AVDF[1][['success','Ticker']] = AVDF[1][0].str.split(" ", 1, expand=True)
successDF = AVDF[1]
SuccessDF = successDF.drop(columns=[0])

symbols2 = ''
sum1 = SuccessDF['success'].sum()
for x in SuccessDF.iloc[0][0]:
    if x == '0':
        symbols2 += '^DJI'
for x in SuccessDF.iloc[1][0]:
    if x == '0':
        symbols2 += ' ^GSPC'
for x in SuccessDF.iloc[2][0]:
    if x == '0':
        symbols2 += ' ^IXIC'

# Index pricing data through yfinance if Alpha Vantage encounters problems

print("Attempting to fetch stock data from yfinance\n")

stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
# Series of 'if' logic that takes the data from both sources
# 'either ALPHAVANTAGE or yfinance' and combine them into one
# dataframe contianing the same info regardless of its source

if sum1 == '111':
    Complete_daily_df = AVDF[0]
    #if ALPHAVANTAGE has failed to pull the index data, the status code 0 will have been added to the success dataframe,
    #this will allow the following if statements to assess which data needs to be pulled from yfinance
elif sum1 == '100':
    Complete_daily_df = AVDF[0].set_index('Date')
    stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
    SP500 = stockdata['^GSPC']
    NASDAQ = stockdata['^IXIC']
    SP500['Symbol'] = 'GSPC'
    NASDAQ['Symbol'] = 'IXIC'
    SP500 = SP500.drop(columns=['Volume'])
    NASDAQ = NASDAQ.drop(columns=['Volume'])
    Complete_daily_df = Complete_daily_df.append(SP500, ignore_index=False)
    Complete_daily_df = Complete_daily_df.append(NASDAQ, ignore_index=False)
    gb = Complete_daily_df.groupby('Symbol')
    DOW = gb.get_group('DJI')
    SP500 = gb.get_group('GSPC')
    NASDAQ = gb.get_group('IXIC')
    SP500 = SP500.reset_index()
    DOW = DOW.reset_index()
    NASDAQ = NASDAQ.reset_index()
elif sum1 == '110':
    Complete_daily_df = AVDF[0].set_index('Date')
    stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
    NASDAQ = stockdata['^IXIC']
    NASDAQ['Symbol'] = 'IXIC'
    NASDAQ = NASDAQ.drop(columns=['Volume'])
    Complete_daily_df = Complete_daily_df.append(NASDAQ, ignore_index=False)
    gb = Complete_daily_df.groupby('Symbol')
    DOW = gb.get_group('DJI')
    SP500 = gb.get_group('GSPC')
    NASDAQ = gb.get_group('IXIC')
    SP500 = SP500.reset_index()
    DOW = DOW.reset_index()
    NASDAQ = NASDAQ.reset_index()
elif sum1 == '010':
    Complete_daily_df = AVDF[0].set_index('Date')
    stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
    DJI = stockdata['^DJI']
    NASDAQ = stockdata['^IXIC']
    DJI['Symbol'] = 'DJI'
    NASDAQ['Symbol'] = 'IXIC'
    DJI = DJI.drop(columns=['Volume'])
    NASDAQ = NASDAQ.drop(columns=['Volume'])
    Complete_daily_df = Complete_daily_df.append(DJI, ignore_index=False)
    Complete_daily_df = Complete_daily_df.append(NASDAQ, ignore_index=False)
    gb = Complete_daily_df.groupby('Symbol')
    DOW = gb.get_group('DJI')
    SP500 = gb.get_group('SPX')
    NASDAQ = gb.get_group('IXIC')
    SP500 = SP500.reset_index()
    DOW = DOW.reset_index()
    NASDAQ = NASDAQ.reset_index()
elif sum1 == '011':
    Complete_daily_df = AVDF[0].set_index('Date')
    stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
    DJI = stockdata['^DJI']
    DJI['Symbol'] = 'DJI'
    DJI = DJI.drop(columns=['Volume'])
    Complete_daily_df = Complete_daily_df.append(DJI, ignore_index=False)
    gb = Complete_daily_df.groupby('Symbol')
    DOW = gb.get_group('DJI')
    SP500 = gb.get_group('SPX')
    NASDAQ = gb.get_group('IXIC')
    SP500 = SP500.reset_index()
    DOW = DOW.reset_index()
    NASDAQ = NASDAQ.reset_index()
elif sum1 == '001':
    Complete_daily_df = AVDF[0].set_index('Date')
    stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
    DJI = stockdata['^DJI']
    SP500 = stockdata['^GSPC']
    DJI['Symbol'] = 'DJI'
    SP500['Symbol'] = 'SP500'
    DJI = DJI.drop(columns=['Volume'])
    SP500 = SP500.drop(columns=['Volume'])
    Complete_daily_df = Complete_daily_df.append(DJI, ignore_index=False)
    Complete_daily_df = Complete_daily_df.append(SP500, ignore_index=False)
    gb = Complete_daily_df.groupby('Symbol')
    DOW = gb.get_group('DJI')
    SP500 = gb.get_group('SPX')
    NASDAQ = gb.get_group('IXIC')
    SP500 = SP500.reset_index()
    DOW = DOW.reset_index()
    NASDAQ = NASDAQ.reset_index()
elif sum1 == '101':
    Complete_daily_df = AVDF[0].set_index('Date')
    stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
    SP500 = stockdata['^GSPC']
    SP500['Symbol'] = 'GSPC'
    SP500 = SP500.drop(columns=['Volume'])
    Complete_daily_df = Complete_daily_df.append(SP500, ignore_index=False)
    gb = Complete_daily_df.groupby('Symbol')
    DOW = gb.get_group('DJI')
    SP500 = gb.get_group('SPX')
    NASDAQ = gb.get_group('IXIC')
    SP500 = SP500.reset_index()
    DOW = DOW.reset_index()
    NASDAQ = NASDAQ.reset_index()
else:
    stockdata = stockdatafetch.DailyFetch(symbols2,'').yfinance_stock_data()
    SP500 = stockdata['^GSPC']
    DOW = stockdata['^DJI']
    NASDAQ = stockdata['^IXIC']
    SP500 = SP500.drop(columns=['Volume'])
    DOW = DOW.drop(columns=['Volume'])
    NASDAQ = NASDAQ.drop(columns=['Volume'])
    SP500 = SP500.reset_index()
    DOW = DOW.reset_index()
    NASDAQ = NASDAQ.reset_index()

start = CustomTime.date_creator('2017-01-27')
end = CustomTime.todays_date()
# masking the indices in accordance with the periods we want to display
DOW = DOW[DOW['Date'].between(start, end, inclusive=True)]
SP500 = SP500[SP500['Date'].between(start, end, inclusive=True)]
NASDAQ = NASDAQ[NASDAQ['Date'].between(start, end, inclusive=True)]
DOW = DOW.set_index(['Date'])
SP500 = SP500.set_index(['Date'])
NASDAQ = NASDAQ.set_index(['Date'])

print("Daily stock data fetch successful and dataframe completed, parsed and sorted\n")
# yfinance intraday stock data from the last 60 days
print("Fetching intraday stock data from yfinance\n")
# bring in and initialize yfinance intraday data
intradaylast60 = stockdatafetch.IntradayFetch(symbols2, '').yfinance_intraday_stock_data()
DOWID = intradaylast60['^DJI']
SP500ID = intradaylast60['^GSPC']
NASDAQID = intradaylast60['^IXIC']

# note: DJI = dow jones, GSPC = S&P500, IXIC = nasdaq
SP500ID = SP500ID.drop(columns=['Volume'])
DOWID = DOWID.drop(columns=['Volume'])
NASDAQID = NASDAQID.drop(columns=['Volume'])
SP500ID = SP500ID.reset_index()
DOWID = DOWID.reset_index()
NASDAQID = NASDAQID.reset_index()
SP500ID = SP500ID.astype(str)
# Mapping the time of stock market trading times
SP500ID['Date'] = SP500ID['Datetime'].map(lambda x: x[:10])
SP500ID['Time'] = SP500ID['Datetime'].map(lambda x: x[11:19])
SP500ID = SP500ID.drop(columns=['Datetime'])
DOWID = DOWID.astype(str)
DOWID['Date'] = DOWID['Datetime'].map(lambda x: x[:10])
DOWID['Time'] = DOWID['Datetime'].map(lambda x: x[11:19])
DOWID = DOWID.drop(columns=['Datetime'])
NASDAQID = NASDAQID.astype(str)
NASDAQID['Date'] = NASDAQID['Datetime'].map(lambda x: x[:10])
NASDAQID['Time'] = NASDAQID['Datetime'].map(lambda x: x[11:19])
NASDAQID = NASDAQID.drop(columns=['Datetime'])

print("Intraday stock data fetch successful and dataframe completed, parsed and sorted\n")

# FED FUNDS RATE (FED API)

print("Fetching data from the Federal Reserve St Louis API\n")
FEDDATA = centralbankdata.fed_api("EFFR") # Effective Fed Funds Rate
print("Federal Reserve data fetch successful\n")

# FED ANNOUNCEMENTS (WEBSCRAPING)

print("Webscraping data from the Federal Reserve website\n")

years = ['2017','2018','2019','2020'] #the years we are interested in looking at

fedframe = centralbankdata.fed_data_func(years)
fedframe.sort_index(ascending=True, inplace = True) #sorting the bizarre date progression
fedframe[fedframe['AdequacyAnn'] == 1].count() #the number of adequate announcements, just for reference

print("Webscraping successful\n")

# THE POST-INAUGURATION TO PRESENT FED DATAFRAME

print("Creating ultimate dataframe for use in regression models\n")

def post_inaug(df1,df2):
    ''' Merges the fed annoucements frame with the fed funds rate frame

    Input: Two dataframes
    Output: a combined dataframe that contains the announcements, the FedFundRate and fills in NaN values
    for the time after Donald Trump's inaugauration
    '''
    ult_frame = df1.join(df2) # Creating a dataframe that contains both the announcements as well as the FFR
    ult_frame['AuxDate'] = pd.to_datetime(ult_frame.index)
    startpi = CustomTime.date_creator('2017-01-27')
    endpi = CustomTime.date_creator(CustomTime.todays_date())
    ult_frame = ult_frame[ult_frame['AuxDate'].between(startpi, endpi, inclusive=True)] # Selecting the period between
    # Trump's candidacy announcement and cut-off date
    # The funds rate is automatically interpolated with the prior data point, wherever it's missing
    ult_frame = ult_frame.drop(['AuxDate','RealtimeStart','RealtimeEnd'],1)
    ult_frame['FedFundsRate'].fillna(method='ffill', inplace = True) #replacing NaNs with prior data points
    return ult_frame
post_inaug(fedframe,FEDDATA)
post_inaug = post_inaug(fedframe,FEDDATA)

# THE LAST 60 DAYS DATAFRAME

def last_60_days(df1,df2):
    ''' Merges the fed annoucements frame with the fed funds rate frame for
    the last 60 days due to the limitations on pulling intraday stock data

    Input: Two dataframes
    Output: a combined dataframe that contains the announcements,
    the FedFundRate and fills in NaN values
    for the last 60 days
    '''
    # Creating an 'ultimate' fed dataframe that contains
    # both the announcements as well as the FFR
    ult_frame = df1.join(df2)
    ult_frame['AuxDate'] = pd.to_datetime(ult_frame.index)
    startl60 = datetime.datetime.strftime(CustomTime.date60daysago(),'%Y-%m-%d')
    endl60 = CustomTime.todays_date()
    ult_frame = ult_frame[ult_frame['AuxDate'].between(startl60, endl60, inclusive=True)] # Selecting the period between
    # Trump's candidacy announcement and cut-off date
    # The funds rate is automatically interpolated with the prior data point, wherever it's missing
    ult_frame = ult_frame.drop(['AuxDate','RealtimeStart','RealtimeEnd'],1)
    ult_frame['FedFundsRate'].fillna(method='ffill', inplace = True) #replacing NaNs with prior data points
    return ult_frame

last_60_days(fedframe,FEDDATA)
# creating a variable out of the function so that it can be manipulated later
last_60_days = last_60_days(fedframe,FEDDATA)
# setting the name of the index
last_60_days.index.name = 'Date'

start = datetime.datetime.strftime(CustomTime.date60daysago(),'%Y-%m-%d') # creating the time boundaries for the tweets
end = CustomTime.todays_date() # datetime 'today's date' object
FinalTweetsDF60 = filtered_trump_tweets
FinalTweetsDF601 = FinalTweetsDF60.loc[(start):(end)] #masking to display the desirable period
FinalTweetsDF601['Date and Time'] = FinalTweetsDF601['Date and Time'].map(CustomTime.time_creator) #Date and Time column using mapping and datetime function
# created earlier
FinalTweetsDF601['TimeRU'] = FinalTweetsDF601['Date and Time'].dt.ceil('15min') #creating the 15 minute intervals we're interested in
FinalTweetsDF60FF = FinalTweetsDF601.TimeRU.value_counts().to_frame('NTPQ') #counting the number of tweets per quarter hour
FinalTweetsDF60FF2 = FinalTweetsDF60FF.sort_index(ascending=True)
FinalTweetsDF60FF3 = FinalTweetsDF60FF2.reset_index()
FinalTweetsDF60FF3 = FinalTweetsDF60FF3.astype(str)
FinalTweetsDF60FF3['Date'] = FinalTweetsDF60FF3['index'].map(lambda x: x[:10]) #mapping the date
FinalTweetsDF60FF3['Time'] = FinalTweetsDF60FF3['index'].map(lambda x: x[11:19]) #mapping the time of stock market trading times
FinalTweetsDF60FF3 = FinalTweetsDF60FF3.astype(str)
FinalTweetsDF60FF3 = FinalTweetsDF60FF3.drop(columns=['index'])  #Final Tweets Dataframe that will be merged with the fed dataframes to create the ultimate frames used in regressions

retweets_sum = FinalTweetsDF601.groupby('TimeRU')['RetweetSum'].sum().to_frame('RetweetSum') #summing the retweets in the 15 mins intervals
#and creating a dataframe out of it
retweets_sum = retweets_sum.reset_index()
retweets_sum = retweets_sum.astype(str)
retweets_sum['Date'] = retweets_sum['TimeRU'].map(lambda x: x[:10])
retweets_sum['Time'] = retweets_sum['TimeRU'].map(lambda x: x[11:19]) #mapping the time of stock market trading times
retweets_sum = retweets_sum.drop(columns=['TimeRU'])
retweets_sum = retweets_sum.astype(str)
FinalTweetsDF60FF4 = FinalTweetsDF60FF3.merge(retweets_sum) #merging the Final Tweets with the retweet sum dataframe
FinalTweetsDF60FF4 = FinalTweetsDF60FF4.set_index(['Date','Time'])

#

DOWID = DOWID.set_index('Date') # creating the dataframe with all the necessary components for the regression
DOWIDANDFED = DOWID.join(last_60_days)
DOWIDANDFED = DOWIDANDFED.reset_index()
DOWIDANDFED = DOWIDANDFED.set_index(['Date','Time'])
DOWIDANDFEDANDTWEETS = DOWIDANDFED.join(FinalTweetsDF60FF4)# joining the index frame with the final tweets frame
DOWIDANDFEDANDTWEETS['FedFundsRate'].fillna(method='ffill', inplace = True) # Filling NaNs with prior values
DOWIDANDFEDANDTWEETS['RetweetSum'].fillna(0,inplace =True) # Filling NaNs with 0s
DOWIDANDFEDANDTWEETS['NTPQ'].fillna(0,inplace =True)
DOWIDANDFEDANDTWEETS.ffill()
DOWIDANDFEDANDTWEETS.fillna(0, inplace = True) # masking all the leftover NaNs that somehow were not fully dealt with before

SP500ID = SP500ID.set_index('Date') # creating the dataframe with all the necessary components for the regression
SP500IDFED = SP500ID.join(last_60_days)
SP500IDFED = SP500IDFED.reset_index()
SP500IDFED = SP500IDFED.set_index(['Date','Time']) #setting the index to Date and Time
SP500IDFEDTWEETS = SP500IDFED.join(FinalTweetsDF60FF4) #joining the index frame with the final tweets frame
SP500IDFEDTWEETS['FedFundsRate'].fillna(method='ffill', inplace = True) #Filling NaNs with prior values
SP500IDFEDTWEETS['RetweetSum'].fillna(0,inplace =True)#Filling NaNs with 0s
SP500IDFEDTWEETS['NTPQ'].fillna(0,inplace =True)
SP500IDFEDTWEETS.ffill()
SP500IDFEDTWEETS.fillna(0, inplace = True) #masking all the leftover NaNs that somehow were not fully dealt with before

NASDAQID = NASDAQID.set_index('Date') #creating the dataframe with all the necessary components for the regression
NASDAQIDFED = NASDAQID.join(last_60_days)
NASDAQIDFED = NASDAQIDFED.reset_index()
NASDAQIDFED = NASDAQIDFED.set_index(['Date','Time']) #setting the index to date and time
NASDAQIDFEDTWEETS = NASDAQIDFED.join(FinalTweetsDF60FF4)
NASDAQIDFEDTWEETS['FedFundsRate'].fillna(method='ffill', inplace = True) #Filling NaNs with prior values
NASDAQIDFEDTWEETS['RetweetSum'].fillna(0,inplace =True) #Filling NaNs with 0s
NASDAQIDFEDTWEETS['NTPQ'].fillna(0,inplace =True)
NASDAQIDFEDTWEETS.ffill()
NASDAQIDFEDTWEETS.fillna(0, inplace = True) #masking all the leftover NaNs that somehow were not fully dealt with before


#


print("Dataframe construction complete\n")

#DOW INTRADAY REGRESSION

print("Running regressions")
print("Dow Jones Intraday Regression")

collist = DOWIDANDFEDANDTWEETS.columns.tolist()  # creating a list from the columns of the df
X = np.asarray(DOWIDANDFEDANDTWEETS[['AdequacyAnn','FedFundsRate','RetweetSum', 'NTPQ']])  #running the regressions with dfs as arrays because otherwise it returns errors
y = np.asarray(DOWIDANDFEDANDTWEETS['Close'])
#X = sm.add_constant(X)
model = sm.OLS(y.astype(float), X.astype(float)) # running a regression with y as dependant variable, and X as independant variables
print(model.fit().summary())

olsres = model.fit()
ypred = olsres.predict(X.astype(float))
print(ypred)

#S&P500 INTRADAY REGRESSION

print("S&P 500 Intraday Regression")

X = np.asarray(SP500IDFEDTWEETS[['AdequacyAnn','FedFundsRate','RetweetSum', 'NTPQ']])
y = np.asarray(SP500IDFEDTWEETS['Close'])
#X = sm.add_constant(X)
model = sm.OLS(y.astype(float), X.astype(float)) # running a regression with y as dependant variable, and X as independant variables
print(model.fit().summary())

#NASDAQ INTRADAY REGRESSION

print("NASDAQ Intraday Regression")

X = np.asarray(NASDAQIDFEDTWEETS[['AdequacyAnn','FedFundsRate','RetweetSum', 'NTPQ']])
y = np.asarray(NASDAQIDFEDTWEETS['Close'])
#X = sm.add_constant(X)
model = sm.OLS(y.astype(float), X.astype(float)) # running a regression with y as dependent variable, and X as independent variables
print(model.fit().summary())


#

# DAILY CLOSING PRICE REGRESSIONS

# DOW REGRESSION

print("Dow Jones Daily Regression")
FinalTweetsDF = filtered_trump_tweets

dow_impact_fedfunds = DOW.join(post_inaug) #creating the dataframe with all the necessary components for the regression
comp_close_DOW = dow_impact_fedfunds.join(FinalTweetsDF)
comp_close_DOW['FedFundsRate'].fillna(method='ffill', inplace = True) #Filling NaNs with prior values
comp_close_DOW.fillna(0,inplace =True) #Filling NaNs with 0s
comp_close_DOW = comp_close_DOW.ffill() #masking all the leftover NaNs that somehow were not fully dealt with before
comp_close_DOW.drop(columns=['Announcements', 'Date and Time', 'Category'], inplace=True)
collistDOW = comp_close_DOW.columns.tolist()  # creating a list from the columns of the df
collistDOW = collistDOW[4:]

X = comp_close_DOW[collistDOW]
y = comp_close_DOW['Close']
X = sm.add_constant(X)
model = sm.OLS(y, X.astype(float)) # running a regression with y as dependant variable, and X as independant variables              # showing the residual degrees of freedom
print(model.fit().summary())
#Forming dataframe composed of predictions from the regression model just run

#ypred = olsres.predict(X.astype(float))
#print(ypred)
#
#print("Dow Jones Daily Decision Tree Regression")
#
#regressor = DecisionTreeRegressor(random_state = 0)
#regressor.fit(y, X.astype(float)).summary()
#print(regressor.fit(y, X.astype(float)).summary())

# S&P REGRESSION


print("S&P 500 Daily Regression")

SP_impact_fedfunds = SP500.join(post_inaug) #creating the dataframe with all the necessary components for the regression
comp_close_SP = SP_impact_fedfunds.join(FinalTweetsDF)
comp_close_SP['FedFundsRate'].fillna(method='ffill', inplace = True) #Filling NaNs with prior values
comp_close_SP.fillna(0,inplace =True) #Filling NaNs with 0s
comp_close_SP = comp_close_SP.ffill() #masking all the leftover NaNs that somehow were not fully dealt with before
comp_close_SP.drop(columns=['Announcements', 'Date and Time', 'Category'], inplace=True)
collistSP = comp_close_SP.columns.tolist()
collistSP = collistSP[4:]

X = comp_close_SP[collistSP]
y = comp_close_SP['Close']
X = sm.add_constant(X)
model = sm.OLS(y, X.astype(float))           # running a regression with y as dependant variable, and X as independant variables              # showing the residual degrees of freedom
print(model.fit().summary())

# NASDAQ REGRESSION

print("NASDAQ Daily Regression")

nasdaq_impact_fedfunds = NASDAQ.join(post_inaug) #creating the dataframe with all the necessary components for the regression
comp_close_NASDAQ = nasdaq_impact_fedfunds.join(FinalTweetsDF)
comp_close_NASDAQ['FedFundsRate'].fillna(method='ffill', inplace = True)#Filling NaNs with prior values
comp_close_NASDAQ.fillna(0,inplace =True) #Filling NaNs with 0s
comp_close_NASDAQ = comp_close_NASDAQ.ffill() #masking all the leftover NaNs that somehow were not fully dealt with before
comp_close_NASDAQ.drop(columns=['Announcements', 'Date and Time', 'Category'], inplace=True)
collistNDQ = comp_close_NASDAQ.columns.tolist()
collistNDQ = collistNDQ[4:]

X = comp_close_NASDAQ[collistNDQ]
y = comp_close_NASDAQ['Close']
X = sm.add_constant(X)
model = sm.OLS(y, X.astype(float)) # running a regression with y as dependent variable, and X as independent variables              # showing the residual degrees of freedom
print(model.fit().summary())

# DATA VISUALIZATION


# Dow Jones vs. Retweets Sum

# Defining the figures Size and giving the whole plot a title
fig = plt.figure(figsize=(15, 8))
fig.suptitle('Dow Jones vs. Sum of Retweets', fontsize=24)

# Overlaying the two datasources using subplots
# Reducing line size to avoid overcramping and adding some colourcoding
ax1 = fig.add_subplot(111)
ax1.plot(FinalTweetsDF["RetweetSum"],linewidth=.3)
ax1.set_ylabel('Number of Retweets')

# adding a second layer indexing to keep data dimensions
ax2 = ax1.twinx()
ax2.plot(DOW["Close"],'b-',linewidth=.8)
ax2.set_ylabel('Dow Jones Closing Prices', color='b')

# Show the graph
plt.show()

# SP500 vs. Retweets Sum

# Defining the figures Size and giving the whole plot a title
fig = plt.figure(figsize=(15, 8))
fig.suptitle('S&P 500 vs. Sum of Retweets', fontsize=24)

# Overlaying the two datasources using subplots
# Reducing line size to avoid overcramping and adding some colourcoding
ax1 = fig.add_subplot(111)
ax1.plot(FinalTweetsDF["RetweetSum"],linewidth=.3)
ax1.set_ylabel('Number of Retweets')

# adding a second layer indexing to keep data dimensions
ax2 = ax1.twinx()
ax2.plot(SP500["Close"],'r-',linewidth=.8)
ax2.set_ylabel('S&P 500 Closing Prices', color='r')

# Show the graph
plt.show()

# NASDAQ Composite Index vs. Retweets Sum

# Defining the figures Size and giving the whole plot a title
fig = plt.figure(figsize=(15, 8))
fig.suptitle('NASDAQ Index vs. Sum of Retweets', fontsize=24)

# Overlaying the two datasources using subplots
# Reducing line size to avoid overcramping and adding some colourcoding
ax1 = fig.add_subplot(111)
ax1.plot(FinalTweetsDF["RetweetSum"],linewidth=.3)
ax1.set_ylabel('Number of Retweets')

# adding a second layer indexing to keep data dimensions
ax2 = ax1.twinx()
ax2.plot(NASDAQ["Close"],'g-',linewidth=.8)
ax2.set_ylabel('NASDAQ Closing Prices', color='g')

# Show the graph
plt.show()

print("Application complete")
