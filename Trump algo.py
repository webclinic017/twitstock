#IMPORTING NECESSARY PACKAGES

#!/usr/bin/env python3

print("Application loading")
import sys, os, os.path, sklearn, json, requests, re, time, datetime, fetchtweet, stockdatafetch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import GetOldTweets3 as got
import yfinance as yf
from statsmodels.stats.diagnostic import het_breuschpagan
from bs4 import BeautifulSoup
from datetime import timedelta
from pandas import DataFrame
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('max_rows', 10)
pd.set_option('max_columns', 10)
pd.set_option('max_colwidth', 100)
np.random.seed(12345)

#Defining Global Time Functions

print("Program running")

def todays_date(): #Creating necessary datetime objects with functions later functions
  return datetime.date.today().strftime('%Y-%m-%d')

def date_creator(str_date):
  return datetime.datetime.strptime(str_date, '%Y-%m-%d')

def date_creator2(str_date):
  return datetime.datetime.strptime(str_date, '%a %b %d %Y')

def time_creator(str_time):
  return datetime.datetime.strptime(str_time, '%a %b %d %H:%M:%S %z %Y')

def gregorian_date_creator(str_date):
  return datetime.datetime.strftime(datetime.datetime.strptime(str_date, '%m/%d/%Y'),'%Y-%m-%d')
  
date60daysago = datetime.date.today() - timedelta(days=59)

#TWITTER DATA

#checking to see if trump tweet data has been fetched previously
#realdonaldtrump
print("Enter twitter handle")
handle = input()
dategroups = [('2017-01-27','2018-01-27'),
             ('2018-01-27','2019-01-27'),
             ('2019-01-27','2020-01-27'),
             ('2020-01-27', todays_date())]
np.savetxt (r'C:\Users\joe\documents\python programs\{}_dates_last_run.txt'.format(handle), dategroups, delimiter='\n', fmt='%s')
try:
    os.path.exists('./{}_tweet_data.csv'.format(handle))
    print("A data set for the chosen handle already exists")
    AllTweets = pd.read_csv("{}_tweet_data.csv".format(handle))
    #append to current tweet df with any new tweets captured after algo was last run
except:
    print("No data set currently exists for the chosen handle")
    print("Program will fetch Twitter data for the chosen handle and store for later use")
    print("Fetching {} twitter data".format(handle))
    AllTweets = fetchtweet.Tweet_Fetch(handle, dategroups)
    AllTweets.to_csv (r'C:\Users\joe\documents\python programs\{}_tweet_data.csv'.format(handle), header=True)

# Counting the number of times he tweets on individual days

filteredtweetsonce =\
                    AllTweets[AllTweets['Tweets'].str.contains\
                    (r'\bstock\b|\bstock market\b|\bnasdaq\b|\bdow jones\b|\bfederal reserve\b|\btariffs\b|\bs&p 500\b|\bimport\b',na=False)]
filteredtweetstwice =\
                    AllTweets[AllTweets['Tweets'].str.contains\
                    (r'\bexport\b|\bunemployment\b|\binterest rates\b|\bthe fed\b|\bthe sec\b|\bthe dow\b',na=False)]
filteredtweetsthrice =\
                    AllTweets[AllTweets['Tweets'].str.contains\
                    (r'\bbusiness\b|\btax\b|\bemployment\b|\bjobs\b|\bregulation\b|\btrillion|\bbillion\b',na=False)]
                    #the reason for filtering thrice is simply the limitation of regex functions concerning the number of keywords

## 21 words


temp1 = filteredtweetsonce.append(filteredtweetstwice, ignore_index=True)

filtered_trump_tweets = temp1.append(filteredtweetsthrice, ignore_index=True) #joining the frames with the keywords together
filtered_trump_tweets['Date'] = filtered_trump_tweets.Date.map(date_creator2) #creating a Date column

FinalTweetsDF = filtered_trump_tweets.Date.value_counts().to_frame('NTPD') #creating the number of tweets per day column (NTPD) 

FinalTweetsDF = FinalTweetsDF.sort_index(ascending=True)
retweet_sum = filtered_trump_tweets.groupby('Date')['Retweets'].sum() #sum of retweets in a day

rts = pd.DataFrame(retweet_sum) # temporary retweet sum df
rts.columns = ['RetweetSum'] 

FinalTweetsDF = FinalTweetsDF.join(rts) #joining the temporary frame with retweets with the FinalTweetsDF
FinalTweetsDF['RetweetSum'].fillna(method='bfill', inplace=True) #Filling NaNs with prior values

#Due Trump's multiple tweets in a day, the duplicated days have to be sorted, so that only the one date per day is shown
FinalTweetsDF= FinalTweetsDF.loc[~FinalTweetsDF.index.duplicated(keep='first')]

print("Tweet dataframe parsed and sorted")

#STOCK INDEX DATA

# Historical stock market data using alphavantage API

# avoiding warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Attempting to fetch stock data from ALPHAVANTAGE API")

symbols = ['DJI','GSPC','IXIC']
AVDF = stockdatafetch.ALPHAVANTAGE_stock_data(symbols) #alphavantage dataframe

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

#Index pricing data through yfinance if Alpha Vantage encounters problems

print("attempting to fetch stock data from yfinance")

stockdata = stockdatafetch.yfinance_stock_data(symbols2)
##Bring in yfinance module and initialize

if sum1 == '111': # series of 'if' statements that instruct which stock indices to pull and make dataframes out of if the stock data pulling encounters given status codes.
  Complete_daily_df = AVDF[0]
  pass
  '''
  if ALPHAVANTAGE has failed to pull the index data, the status code 0 will have been added to the success dataframe,
  this will allow the following if statements to assess which data needs to be pulled from yfinance
  '''
elif sum1 == '100':
  Complete_daily_df = AVDF[0].set_index('Date')
  stockdata = yfinance_stock_data(symbols2)
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
  stockdata = yfinance_stock_data(symbols2)
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
  stockdata = yfinance_stock_data(symbols2)
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
  stockdata = yfinance_stock_data(symbols2)
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
  stockdata = yfinance_stock_data(symbols2)
  DJI = stockdata['^DJI']
  SP500 = stockdata['^GSPC']
  DJI['Symbol'] = 'DJI'
  SP500['Symbol'] = 'SP500'
  DJI = DJI.drop(columns=['Volume'])
  SP500 = SP5000.drop(columns=['Volume'])
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
  stockdata = yfinance_stock_data(symbols2)
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
  stockdata = stockdatafetch.yfinance_stock_data(symbols2)
  SP500 = stockdata['^GSPC']
  DOW = stockdata['^DJI']
  NASDAQ = stockdata['^IXIC']
  SP500 = SP500.drop(columns=['Volume'])
  DOW = DOW.drop(columns=['Volume'])
  NASDAQ = NASDAQ.drop(columns=['Volume'])
  SP500 = SP500.reset_index()
  DOW = DOW.reset_index()
  NASDAQ = NASDAQ.reset_index()

start = date_creator('2017-01-27')  #datetime objects start and end  
end = todays_date()
# masking the indices in accordance with the periods we want to display
DOW = DOW[DOW['Date'].between(start, end, inclusive=True)]
SP500 = SP500[SP500['Date'].between(start, end, inclusive=True)]
NASDAQ = NASDAQ[NASDAQ['Date'].between(start, end, inclusive=True)]
DOW = DOW.set_index(['Date']); SP500 = SP500.set_index(['Date']); NASDAQ = NASDAQ.set_index(['Date'])

print("Daily stock data successful and dataframe completed, parsed and sorted")
#yfinance intraday stock data from the last 60 days

print("Fetching intraday stock data from yfinance")

## bring in and initialize yfinance intraday data
intradaylast60 = stockdatafetch.yfinance_intraday_stock_data()
DOWID = intradaylast60['^DJI']
SP500ID = intradaylast60['^GSPC']
NASDAQID = intradaylast60['^IXIC']

# note: DJI = dow jones, GSPC = S&P500, IXIC = nasdaq
SP500ID = SP500ID.drop(columns=['Volume']) #dropping unnecessary columns
DOWID = DOWID.drop(columns=['Volume'])
NASDAQID = NASDAQID.drop(columns=['Volume'])
SP500ID = SP500ID.reset_index()
DOWID = DOWID.reset_index()
NASDAQID = NASDAQID.reset_index()
SP500ID = SP500ID.astype(str)
SP500ID['Date'] = SP500ID['Datetime'].map(lambda x: x[:10]) 
SP500ID['Time'] = SP500ID['Datetime'].map(lambda x: x[11:19]) #mapping the time of stock market trading times
SP500ID = SP500ID.drop(columns=['Datetime'])
DOWID = DOWID.astype(str)
DOWID['Date'] = DOWID['Datetime'].map(lambda x: x[:10])
DOWID['Time'] = DOWID['Datetime'].map(lambda x: x[11:19])
DOWID = DOWID.drop(columns=['Datetime'])
NASDAQID = NASDAQID.astype(str)
NASDAQID['Date'] = NASDAQID['Datetime'].map(lambda x: x[:10])
NASDAQID['Time'] = NASDAQID['Datetime'].map(lambda x: x[11:19])
NASDAQID = NASDAQID.drop(columns=['Datetime']) 

print("Intraday stock data successful and dataframe completed, parsed and sorted")

#FED FUNDS RATE (FED API)

'''
Federal Reserve Data API
https://fred.stlouisfed.org/categories pick category, when looking at graph take code from URL and put into Series ID - params may need to
be changed

Effective Fed Funds Rate (EFFR) chosen as it is the measure provided by the fed that has the biggest impact on the US stock market as a whole
including the major stock indicies that we are focusing on 
'''

print("Fetching data from the Federal Reserve St Louis API")

api = 'https://api.stlouisfed.org/fred/series/observations'

parameters = {
    "file_type": "json",
    "series_id": "EFFR", # Effective Fed Funds Rate
    "realtime_start": "2017-01-27",
    "realtime_end": "9999-12-31",
    "limit": "100000",
    "offset": "0",
    "sort_order": "asc",
    "observation_start": "2017-01-27",
    "observation_end": "{}".format(todays_date()),
    "units": "lin",
    "aggregation_method": "avg",
    "output_type": "1",
    "api_key": "e3f70e5a440482ef1fc456be3dec47da",
    }
response = requests.get(api, params=parameters) 
dict_ = json.loads(response.content) #Pulling data through API, using json to create a dict

FEDDATA = pd.DataFrame(dict_['observations'])
FEDDATA = FEDDATA.set_index('date') 
FEDDATA.columns = ['RealtimeStart','RealtimeEnd','FedFundsRate'] #creating a dataframe with the fed funds rate, listing the columns

print("Federal Reserve data fetch successful")

#FED ANNOUNCEMENTS (WEBSCRAPING)

print("Webscraping data from the Federal Reserve website")

years = ['2017','2018','2019','2020'] #the years we are interested in looking at
def fed_data_func(x):
  '''
  Input: taking a list of years in the format 'YYYY' + relevant html-snippet
  Output: a dataframe containing the announcements of the US Federal Reserve Bank
  with Date, Announcement and Category of Announcement, indexed by date
  '''
  fdframe = pd.DataFrame()
  for year in x:
  # get HTML file and format content
    url = 'https://www.federalreserve.gov/newsevents/pressreleases/{}-press.htm'.format(year)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
  # parse through page for divisions needed, further sort by class
    dates = soup.find_all('div', class_='col-xs-3 col-md-2 eventlist__time')
  # scrape text from derived data and format into a dataframe
    time = soup.find_all('time')
    fed_df = pd.DataFrame([date.find('time').text for date in dates], columns=['Date'])
    announcements = soup.find_all('div', class_='col-xs-9 col-md-10 eventlist__event')
    fed_df['Announcements'] = pd.DataFrame([announcement.find('em').text for announcement in announcements])
    categories = soup.find_all('p', class_='eventlist__press') #general scraping for necessary information
    fed_df['Category'] = pd.DataFrame([category.find('strong').text for category in categories])
    fdframe = fdframe.append(fed_df, ignore_index=True) 
  # change US date format to Gregorian datetime object
  fdframe['Date'] = fdframe.Date.map(gregorian_date_creator) 
  fdframe['AdequacyAnn'] = fdframe['Announcements'].map(lambda x: +1 if 'Federal Open Market Committee' in x else +1 if 'discount rate' in x else +1 if 'deposit' in x else +1 if 'inflation' in x else +1 if 'rate' in x else 0)
  fdframe = fdframe.set_index('Date')
  #using a lambda function to create a new binary adequacy column that marks the announcements that contain the specific keywords 
  fdframe = pd.DataFrame(fdframe)
  return fdframe
fedframe = fed_data_func(years)
fedframe = fedframe.sort_index(ascending=True) #sorting the bizarre date progression
fedframe[fedframe['AdequacyAnn'] == 1].count() #the number of adequate announcements, just for reference

print("Webscraping successful")

#THE POST-INAUGURATION TO PRESENT FED DATAFRAME

print("Creating ultimate dataframe for use in regression models")

def post_inaug(df1,df2): #a function that merges the fed annoucements frame with the fed funds rate frame
  '''
  Input: Two dataframes 
  Output: a combined dataframe that contains the announcements, the FedFundRate and fills in NaN values
  for the time after Donald Trump's inaugauration
  '''
  ult_frame = df1.join(df2) # Creating a dataframe that contains both the announcements as well as the FFR
  ult_frame['AuxDate'] = pd.to_datetime(ult_frame.index)
  start = date_creator('2017-01-27')
  end = date_creator(todays_date())   
  ult_frame = ult_frame[ult_frame['AuxDate'].between(start, end, inclusive=True)] # Selecting the period between 
  # Trump's candidacy announcement and cut-off date
  # The funds rate is automatically interpolated with the prior data point, wherever it's missing
  ult_frame = ult_frame.drop(['AuxDate','RealtimeStart','RealtimeEnd'],1)
  ult_frame['FedFundsRate'].fillna(method='ffill', inplace = True) #replacing NaNs with prior data points
  return ult_frame
post_inaug(fedframe,FEDDATA)
post_inaug = post_inaug(fedframe,FEDDATA)

#THE LAST 60 DAYS DATAFRAME

def last_60_days (df1,df2):
  '''
  a function that does the same as the post_inaug function, only for the last 60 days
  and that is because of the existing limitations on pulling intraday stock prices

  Input: Two dataframes 
  Output: a combined dataframe that contains the announcements, the FedFundRate and fills in NaN values
  for the last 60 days
  '''
  ult_frame = df1.join(df2) # Creating an 'ultimate' fed dataframe that contains both the announcements as well as the FFR
  ult_frame['AuxDate'] = pd.to_datetime(ult_frame.index)
  start = datetime.datetime.strftime(date60daysago,'%Y-%m-%d')
  end = todays_date() 
  ult_frame = ult_frame[ult_frame['AuxDate'].between(start, end, inclusive=True)] # Selecting the period between
  # Trump's candidacy announcement and cut-off date
  # The funds rate is automatically interpolated with the prior data point, wherever it's missing
  ult_frame = ult_frame.drop(['AuxDate','RealtimeStart','RealtimeEnd'],1)
  ult_frame['FedFundsRate'].fillna(method='ffill', inplace = True) #replacing NaNs with prior data points
  return ult_frame
  
last_60_days(fedframe,FEDDATA)
#creating a variable out of the function so that it can be manipulated later
last_60_days = last_60_days(fedframe,FEDDATA)
#setting the name of the index
last_60_days.index.name = 'Date'

start = datetime.datetime.strftime(date60daysago,'%Y-%m-%d') # creating the time boundaries for the tweets
end = todays_date() #datetime 'today's date' object
FinalTweetsDF60 = filtered_trump_tweets.set_index('Date')
FinalTweetsDF601 = FinalTweetsDF60.loc[(start):(end)] #masking to display the desirable period
FinalTweetsDF601['Date and Time'] = FinalTweetsDF601['Date and Time'].map(time_creator) #Date and Time column using mapping and datetime function
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

retweets_sum = FinalTweetsDF601.groupby('TimeRU')['Retweets'].sum().to_frame('RetweetSum') #summing the retweets in the 15 mins intervals 
#and creating a dataframe out of it
retweets_sum = retweets_sum.reset_index()
retweets_sum = retweets_sum.astype(str)
retweets_sum['Date'] = retweets_sum['TimeRU'].map(lambda x: x[:10])
retweets_sum['Time'] = retweets_sum['TimeRU'].map(lambda x: x[11:19]) #mapping the time of stock market trading times
retweets_sum = retweets_sum.drop(columns=['TimeRU'])
retweets_sum = retweets_sum.astype(str)
FinalTweetsDF60FF4 = FinalTweetsDF60FF3.merge(retweets_sum) #merging the Final Tweets with the retweet sum dataframe
FinalTweetsDF60FF4 = FinalTweetsDF60FF4.set_index(['Date','Time'])

DOWID = DOWID.set_index('Date') #creating the dataframe with all the necessary components for the regression
DOWIDANDFED = DOWID.join(last_60_days)
DOWIDANDFED = DOWIDANDFED.reset_index()
DOWIDANDFED = DOWIDANDFED.set_index(['Date','Time'])
DOWIDANDFEDANDTWEETS = DOWIDANDFED.join(FinalTweetsDF60FF4)#joining the index frame with the final tweets frame
DOWIDANDFEDANDTWEETS['FedFundsRate'].fillna(method='ffill', inplace = True) #Filling NaNs with prior values
DOWIDANDFEDANDTWEETS['RetweetSum'].fillna(0,inplace =True) #Filling NaNs with 0s
DOWIDANDFEDANDTWEETS['NTPQ'].fillna(0,inplace =True)
DOWIDANDFEDANDTWEETS.ffill()
DOWIDANDFEDANDTWEETS.fillna(0, inplace = True) #masking all the leftover NaNs that somehow were not fully dealt with before

SP500ID = SP500ID.set_index('Date') #creating the dataframe with all the necessary components for the regression
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

print("Dataframe construction complete")

#DOW INTRADAY REGRESSION

print("Running regressions")
print("Dow Jones Intraday Regression")

collist = DOWIDANDFEDANDTWEETS.columns.tolist()  # creating a list from the columns of the df
X = np.asarray(DOWIDANDFEDANDTWEETS[['AdequacyAnn','FedFundsRate','RetweetSum', 'NTPQ']])  #running the regressions with dfs as arrays because otherwise it returns errors  
y = np.asarray(DOWIDANDFEDANDTWEETS['Close'])# 

model = sm.OLS(y.astype(float), X.astype(float)) # running a regression with y as dependant variable, and X as independant variables
model.df_model                 # showing the degrees of freedom of the regression
model.df_resid                 # showing the residual degrees of freedom
model.endog_names              # names of the endogenous variables
model.exog_names               # names of the exogenous variables
model.fit().summary()   
print(model.fit().summary())

olsres = model.fit()
ypred = olsres.predict(X.astype(float))
print(ypred)

#S&P500 INTRADAY REGRESSION

print("S&P 500 Intraday Regression")

X = np.asarray(SP500IDFEDTWEETS[['AdequacyAnn','FedFundsRate','RetweetSum', 'NTPQ']])                       
y = np.asarray(SP500IDFEDTWEETS['Close'])                       

model = sm.OLS(y.astype(float), X.astype(float))                # running a regression with y as dependant variable, and X as independant variables
model.df_model                 # showing the degrees of freedom of the regression
model.df_resid                 # showing the residual degrees of freedom
model.endog_names              # names of the endogenous variables
model.exog_names               # names of the exogenous variables
model.fit().summary()
print(model.fit().summary())

#NASDAQ INTRADAY REGRESSION

print("NASDAQ Intraday Regression")

X = np.asarray(NASDAQIDFEDTWEETS[['AdequacyAnn','FedFundsRate','RetweetSum', 'NTPQ']])                       
y = np.asarray(NASDAQIDFEDTWEETS['Close'])                       

model = sm.OLS(y.astype(float), X.astype(float))                # running a regression with y as dependent variable, and X as independent variables
model.df_model                 # showing the degrees of freedom of the regression
model.df_resid                 # showing the residual degrees of freedom
model.endog_names              # names of the endogenous variables
model.exog_names               # names of the exogenous variables
model.fit().summary() 
print(model.fit().summary())  

# DAILY CLOSING PRICE REGRESSIONS

# DOW REGRESSION

print("Dow Jones Daily Regression")

dow_impact_fedfunds = DOW.join(post_inaug) #creating the dataframe with all the necessary components for the regression
comp_close_DOW = dow_impact_fedfunds.join(FinalTweetsDF)
comp_close_DOW['FedFundsRate'].fillna(method='ffill', inplace = True) #Filling NaNs with prior values
comp_close_DOW['RetweetSum'].fillna(0,inplace =True) #Filling NaNs with 0s
comp_close_DOW['NTPD'].fillna(0,inplace =True)
comp_close_DOW = comp_close_DOW.ffill() #masking all the leftover NaNs that somehow were not fully dealt with before
collist = comp_close_DOW.columns.tolist()  # creating a list from the columns of the df
X = comp_close_DOW[['AdequacyAnn', 'FedFundsRate', 'RetweetSum', 'NTPD']]
y = comp_close_DOW['Close']
model = sm.OLS(y, X.astype(float))           # running a regression with y as dependant variable, and X as independant variables              # showing the residual degrees of freedom
model.endog_names              # names of the endogenous variables
model.exog_names               # names of the exogenous variables
model.fit().summary()
print(model.fit().summary())

# S&P REGRESSION

print("S&P 500 Daily Regression")

SP_impact_fedfunds = SP500.join(post_inaug) #creating the dataframe with all the necessary components for the regression
comp_close_SP = SP_impact_fedfunds.join(FinalTweetsDF)
comp_close_SP['FedFundsRate'].fillna(method='ffill', inplace = True) #Filling NaNs with prior values
comp_close_SP['RetweetSum'].fillna(0,inplace =True) #Filling NaNs with 0s
comp_close_SP['NTPD'].fillna(0,inplace =True)
comp_close_SP = comp_close_SP.ffill() #masking all the leftover NaNs that somehow were not fully dealt with before

X = comp_close_SP[['RetweetSum', 'FedFundsRate', 'AdequacyAnn', 'NTPD']]                       
y = comp_close_SP['Close']                      
model = sm.OLS(y, X.astype(float))           # running a regression with y as dependant variable, and X as independant variables              # showing the residual degrees of freedom
model.endog_names              # names of the endogenous variables
model.exog_names               # names of the exogenous variables
model.fit().summary()
print(model.fit().summary())

# NASDAQ REGRESSION

print("NASDAQ Daily Regression")

nasdaq_impact_fedfunds = NASDAQ.join(post_inaug) #creating the dataframe with all the necessary components for the regression
comp_close_NASDAQ = nasdaq_impact_fedfunds.join(FinalTweetsDF)
comp_close_NASDAQ['FedFundsRate'].fillna(method='ffill', inplace = True)#Filling NaNs with prior values
comp_close_NASDAQ['RetweetSum'].fillna(0,inplace =True) #Filling NaNs with 0s
comp_close_NASDAQ['NTPD'].fillna(0,inplace =True)
comp_close_NASDAQ = comp_close_NASDAQ.ffill() #masking all the leftover NaNs that somehow were not fully dealt with before

X = comp_close_NASDAQ[['RetweetSum', 'FedFundsRate', 'AdequacyAnn', 'NTPD']]                      
y = comp_close_NASDAQ['Close']                      
model = sm.OLS(y, X.astype(float))           # running a regression with y as dependent variable, and X as independent variables              # showing the residual degrees of freedom
model.endog_names              # names of the endogenous variables
model.exog_names               # names of the exogenous variables
model.fit().summary()
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