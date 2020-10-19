'''This module fetches stock data from multiple sources'''
import json
import csv
import requests

import pandas as pd
import yfinance as yf

from customdatetimecreator import CustomTime

class FetchCriteria:
    '''Data fetch criteria class'''

    timeperiod = '5'
    sliceoption = 'year1month1'
    output = pd.DataFrame()

    def __init__(self, symbols, function):
        '''Setting the default criteria for a search'''
        self.symbols = symbols
        self.function = function

    @classmethod
    def set_timeperiod(cls, timeperiod):
        '''Setting the timeperiod

        timeperiod is only used by yfinance Daily
        can either pass:
        a period i.e. 1y, 5y...
        '''
        cls.timeperiod = timeperiod

    @classmethod
    def set_sliceoption(cls, sliceoption):
        '''setting the slice to be collected from AV'''
        cls.sliceoption = sliceoption

class DailyFetch(FetchCriteria):
    '''Subclass to fetch daily data for the given FetchCriteria'''
    outputsize = 'full'

    @classmethod
    def set_outputsize(cls, outputsize):
        '''setting the outputsize

        options are:
        full or compact
        '''
        cls.outputsize = outputsize

    def alphavantage_stock_data(self):
        '''Input: list of stock symbols in capital letters
        Output: dataframe of stock price data, indexed by date
        '''
        frame = pd.DataFrame()
        success = pd.DataFrame()
        # Success dataframe that is appended accordingly if the proper status code is returned
        if not self.symbols:
            print('ERROR: List is empty')
        for symbol in self.symbols:
            if not symbol.isalpha():
                print('ERROR: Symbol {} is not made up of letters'.format(symbol))
            elif not symbol.isupper():
                print('ERROR: Symbol {} is not made up of Capital letters'.format(symbol))

            else:
                api = 'https://www.alphavantage.co/query'
                data = {
                        'function': self.function,
                        'symbol': symbol,
                        'outputsize': self.outputsize,
                        'apikey': "3KTUERI46K71KH85", # 3KTUERI46K71KH85, TDO6ET6NSGZF1MZ5, #B4Y78SVZIZ1HEF88
                        }
                try:
                    test = requests.head(api, params=data)
                    print('Succesful connection with response {}'.format(test.status_code))
                except requests.ConnectionError:
                    print('ERROR: failed to connect')
                response = requests.get(api, params=data)
                return_dictionary = json.loads(response.content)
                # a series of commands that add 1 to success dataframe
                # if a correct status code is returned and 0 if it isn't
                if any(['Time Series (Daily)'for x in return_dictionary]):
                    print('ALPHAVANTAGE has returned the results for {} correctly'.format(symbol))
                    success1 = pd.Series('1 {}'.format(symbol))
                    success = success.append(success1, ignore_index=True)
                else:
                    print('ERROR: ALPHAVANTAGE has encountered problems fetching the required data for {}, more information from ALPHAVANTAGE may be avaliable below:'.format(symbol))
                    print(return_dictionary)
                    success1 = pd.Series('0 {}'.format(symbol))
                    success = success.append(success1, ignore_index=True)
                    continue
                dataframe = pd.DataFrame(return_dictionary['Time Series (Daily)']).T
                dataframe.name = "{} stock price".format(symbol)
                dataframe.columns = [
                    'Open','High','Low','Close',
                    'adjusted close','volume','dividend amount','split coefficient',
                    ]
                dataframe.Open = dataframe.Open.astype(float)
                dataframe.Close = dataframe.Close.astype(float)
                dataframe['Symbol'] = '{}'.format(symbol)
                dataframe['Date'] = dataframe.index.map(CustomTime.date_creator)
                frame = frame.append(dataframe, ignore_index=True)
                frame = frame.drop(
                    [
                    'volume','adjusted close',
                    'dividend amount','split coefficient',
                    ],
                    axis=1,
                    )
        return frame, success

    def yfinance_stock_data(self):
        '''Input: List of symbols dependent on previous success variable O'''
        stockdata =\
        yf.download(
                tickers = '{}'.format(self.symbols),
                period = self.timeperiod + 'y',
                interval = '1d',
                group_by = 'ticker',
                auto_adjust = True,
                prepost = False,
                threads = True,
                proxy = None
                )
        stockdata = stockdata.reindex(index=stockdata.index[::-1])
        return stockdata

class IntradayFetch(FetchCriteria):
    '''Subclass to fetch intradaydata given the FetchCriteria'''
    interval = '15'
        
    @classmethod
    def set_interval(cls, interval):
        '''setting the interval for intraday requests

        (1, 5, 15, 30, 60)
        '''
        cls.interval = interval

    def alphavantage_intraday_stock_data(self):
        '''Input: list of stock symbols in capital letters
        Output: dataframe of stock price data, indexed by date
        '''
        datalist = []
        frame = pd.DataFrame()
        success = pd.DataFrame()
        # Success dataframe that is appended accordingly if the proper status code is returned
        if not self.symbols:
            print('ERROR: List is empty')
        for symbol in self.symbols:
            if not symbol.isalpha():
                print('ERROR: Symbol {} is not made up of letters'.format(symbol))
            elif not symbol.isupper():
                print('ERROR: Symbol {} is not made up of Capital letters'.format(symbol))
            else:
                api = 'https://www.alphavantage.co/query'
                data = {
                        'function': self.function,
                        'symbol': symbol,
                        'interval': self.interval + 'min',
                        'slice': self.sliceoption,
                        'apikey': "3KTUERI46K71KH85", # 3KTUERI46K71KH85, TDO6ET6NSGZF1MZ5, #B4Y78SVZIZ1HEF88
                        }
                # a series of commands that add 1 to success dataframe
                # if a correct status code is returned and 0 if it isn't
                try:
                    test = requests.head(api, params=data)
                    print('Succesful connection with response {}'.format(test.status_code))
                except requests.ConnectionError:
                    print('ERROR: failed to connect')
                response = requests.get(api, params=data)
                with response as file:
                    lines = (line.decode('utf-8') for line in file.iter_lines())
                    for row in csv.reader(lines):
                        datalist.append(row)
                dataframe = pd.DataFrame(
                    datalist, columns = [
                        'Time','Open','High',
                        'Low','Close','volume',
                        ]
                    )
                dataframe.drop(dataframe.index[0], inplace=True)
                dataframe.set_index('Time', inplace=True)
                #if any(['Time Series (Daily)'for x in return_dictionary]):
                    #print('ALPHAVANTAGE has returned the results for {} correctly'.format(symbol))
                    #success1 = pd.Series('1 {}'.format(symbol))
                    #success = success.append(success1, ignore_index=True)
                #else:
                    #print('ERROR: ALPHAVANTAGE has encountered problems fetching the required data for {}, more information from ALPHAVANTAGE may be avaliable below:'.format(symbol))
                    #print(return_dictionary)
                    #success1 = pd.Series('0 {}'.format(symbol))
                    #success = success.append(success1, ignore_index=True)
                    #continue
                #dataframe = pd.DataFrame(return_dictionary['Time Series (Daily)']).T
                dataframe.name = "{} stock price".format(symbol)
                dataframe.Open = dataframe.Open.astype(float)
                dataframe.Close = dataframe.Close.astype(float)
                dataframe['Symbol'] = '{}'.format(symbol)
                dataframe.index = dataframe.index.map(CustomTime.time_creator2)
                frame = frame.append(dataframe, ignore_index=True)
        return frame, success

    def yfinance_intraday_stock_data(self):
        '''gives data from past 60 days including the day that it is run'''
        stockdata =\
        yf.download(
                tickers = '{}'.format(self.symbols),
                end = CustomTime.todays_date(),
                start = CustomTime.date60daysago(),
                interval = self.interval + 'm',
                group_by = 'ticker',
                auto_adjust = True,
                prepost = False,
                threads = True,
                proxy = None
                )
        stockdata = stockdata.reindex(index=stockdata.index[::-1])
        return stockdata

class Fetch:
    '''Class to Fetch stock data'''
    def __init__(self):
        pass

    @staticmethod
    def get_stock_data(fetch_criteria):
        '''fetch stock data'''
        if fetch_criteria.function == 'Daily':
            try:
                fetch_criteria.function = 'TIME_SERIES_DAILY_ADJUSTED'
                dailydata = FetchMethods.alphavantage_stock_data(fetch_criteria)
            except:
                fetch_criteria.symbols = ['^' + symbol for symbol in fetch_criteria.symbols]
                fetch_criteria.interval = [interval + 'y' for interval in fetch_criteria.interval]
                dailydata =  FetchMethods.yfinance_stock_data(fetch_criteria)
        elif fetch_criteria.function == 'Intraday':
            try:
                fetch_criteria.function = 'TIME_SERIES_INTRADAY_EXTENDED'
                fetch_criteria.interval = [fetch_criteria.interval + 'min']
                intradaydata = FetchMethods.alphavantage_intraday_stock_data(fetch_criteria)
            except:
                fetch_criteria.symbols = ['^' + symbol for symbol in fetch_criteria.symbols]
                fetch_criteria.interval = [interval.replace('min', 'm') for interval in fetch_criteria.interval]
                print(fetch_criteria.interval)
                intradaydata = FetchMethods.yfinance_intraday_stock_data(fetch_criteria)
        else:
            print('ERROR: You have not chosen a valid fuction to call')
        return intradaydata
