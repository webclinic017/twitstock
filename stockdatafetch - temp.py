import pandas as pd
import yfinance as yf
import requests
import json

from customdatetimecreator import CustomTime

class FetchCriteria:
    '''A class to outline the parameters by which the stock data should
    be fetched
    '''
    def __init__(self):
        '''Setting the default output size of the ALPHAVANTAGE data fetch'''
        self.timeparameter = '5'
        self.outputsize = "full"
        self.interval = '15'
        self.slice = 'year1month1'

    def set_symbols(self, symbols):
        '''Setting the symbols for the stock data to be fetched
        can be a string or a singular symbol'''
        self.symbols = symbols
        return self

    def set_timeparameter(self, timeparameter):
        '''Setting the timeparameter

        timeparameter is only used by yfinance Daily
        can either pass:
        a period i.e. 1y, 5y...
        '''
        self.timeparameter = timeparameter
        return self

    def set_outputsize(self, outputsize):
        '''setting the outputsize

        outputsize is used only for ALPHAVANTAGE Daily
        options are:
        full or compact
        '''
        self.outputsize = outputsize
        return self

    def set_interval(self, interval):
        '''setting the interval

        (1, 5, 15, 30, 60)

        '''
        self.interval = interval
        return self

    def set_function(self, function):
        '''setting the fuction
        fuction is used for AV only
        AV key:
        Daily = TIME_SERIES_DAILY_ADJUSTED
        Intraday = TIME_SERIES_INTRADAY_EXTENDED
        '''
        self.function = function
        return self

    def set_slice(self, slice):
        '''setting the slice to be collected from AV'''
        self.slice = slice
        return self

class FetchMethods:
    '''Methods by which stock data can be fetched'''
    def __init__(self):
        pass

    def ALPHAVANTAGE_stock_data(fetchCriteria):
        '''Input: list of stock symbols in capital letters
        Output: dataframe of stock price data, indexed by date
        '''
        frame = pd.DataFrame()
        success = pd.DataFrame()
        # Success dataframe that is appended accordingly if the proper status code is returned
        if not fetchCriteria.symbols:
            print('ERROR: List is empty')
        for x in fetchCriteria.symbols:
            if x.isalpha() == False:
                print('ERROR: Symbol {} is not made up of letters'.format(x))
            elif x.isupper() == False:
                print('ERROR: Symbol {} is not made up of Capital letters'.format(x))

            else:
                api = 'https://www.alphavantage.co/query'
                data = {
                        'function': fetchCriteria.function,
                        'symbol': x,
                        'outputsize': fetchCriteria.outputsize,
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
                    print('ALPHAVANTAGE has returned the results for {} correctly'.format(x))
                    success1 = pd.Series('1 {}'.format(x))
                    success = success.append(success1, ignore_index=True)
                else:
                    print('ERROR: ALPHAVANTAGE has encountered problems fetching the required data for {}, more information from ALPHAVANTAGE may be avaliable below:'.format(x))
                    print(return_dictionary)
                    success1 = pd.Series('0 {}'.format(x))
                    success = success.append(success1, ignore_index=True)
                    continue
                df = pd.DataFrame(return_dictionary['Time Series (Daily)']).T
                df.name = "{} stock price".format(x)
                df.columns = [
                    'Open','High','Low','Close',
                    'adjusted close','volume','dividend amount','split coefficient',
                    ]
                df.Open = df.Open.astype(float)
                df.Close = df.Close.astype(float)
                df['Symbol'] = '{}'.format(x)
                df['Date'] = df.index.map(CustomTime.date_creator)
                frame = frame.append(df)
                frame = frame.drop(
                    [
                    'volume','adjusted close',
                    'dividend amount','split coefficient',
                    ], axis=1, inplace=True
                    )
        return frame, success

    def ALPHAVANTAGE_intraday_stock_data(fetchCriteria):
        '''Input: list of stock symbols in capital letters
        Output: dataframe of stock price data, indexed by date
        '''
        frame = pd.DataFrame()
        success = pd.DataFrame()
        # Success dataframe that is appended accordingly if the proper status code is returned
        if not fetchCriteria.symbols:
            print('ERROR: List is empty')
        for x in fetchCriteria.symbols:
            if x.isalpha() == False:
                print('ERROR: Symbol {} is not made up of letters'.format(x))

            elif x.isupper() == False:
                print('ERROR: Symbol {} is not made up of Capital letters'.format(x))

            else:
                api = 'https://www.alphavantage.co/query'
                data = {
                        'function': fetchCriteria.function,
                        'symbol': x,
                        'interval': fetchCriteria.interval,
                        'slice': fetchCriteria.slice,
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
                return_dictionary = json.loads(response.content)
                if any(['Time Series (Daily)'for x in return_dictionary]):
                    print('ALPHAVANTAGE has returned the results for {} correctly'.format(x))
                    success1 = pd.Series('1 {}'.format(x))
                    success = success.append(success1, ignore_index=True)
                else:
                    print('ERROR: ALPHAVANTAGE has encountered problems fetching the required data for {}, more information from ALPHAVANTAGE may be avaliable below:'.format(x))
                    print(return_dictionary)
                    success1 = pd.Series('0 {}'.format(x))
                    success = success.append(success1, ignore_index=True)
                    continue
                df = pd.DataFrame(return_dictionary['Time Series (Daily)']).T
                df.name = "{} stock price".format(x)
                df.columns = [
                    'Open','High','Low','Close',
                    'adjusted close','volume','dividend amount','split coefficient',
                    ]
                df.Open = df.Open.astype(float)
                df.Close = df.Close.astype(float)
                df['Symbol'] = '{}'.format(x)
                df['Date'] = df.index.map(CustomTime.date_creator)
                frame.append(df)
                frame.drop(
                    [
                    'volume','adjusted close',
                    'dividend amount','split coefficient',
                    ], axis=1, inplace=True
                    )
        return frame, success

    def yfinance_stock_data(fetchCriteria):
        '''Input: List of symbols dependent on previous success variable O'''
        stockdata =\
        yf.download(
                tickers = '{}'.format(fetchCriteria.symbols),
                period = '{}'.format(fetchCriteria.timeperameter),
                interval = '{}'.format(fetchCriteria.interval),
                group_by = 'ticker',
                auto_adjust = True,
                prepost = False,
                threads = True,
                proxy = None
                )
        stockdata = stockdata.reindex(index=stockdata.index[::-1])
        return stockdata

    def yfinance_intraday_stock_data(fetchCriteria):
        '''gives data from past 60 days including the day that it is run'''
        stockdata =\
        yf.download(
                tickers = '{}'.format(fetchCriteria.symbols),
                end = CustomTime.todays_date(),
                start = CustomTime.date60daysago(),
                interval = "{}".format(fetchCriteria.interval),
                group_by = 'ticker',
                auto_adjust = True,
                prepost = False,
                threads = True,
                proxy = None
                )
        stockdata = stockdata.reindex(index=stockdata.index[::-1])
        return stockdata

class Fetch:

    def __init__(self):
        pass

    def get_stock_data(fetchCriteria):
        if fetchCriteria.function == 'Daily':
            try:
                fetchCriteria.function = 'TIME_SERIES_DAILY_ADJUSTED'
                dailydata = FetchMethods.ALPHAVANTAGE_stock_data(fetchCriteria)
            except:
                fetchCriteria.symbols = ['^' + symbol for symbol in fetchCriteria.symbols]
                fetchCriteria.interval = [interval + 'y' for interval in fetchCriteria.interval]
                dailydata =  FetchMethods.yfinance_stock_data(fetchCriteria)
        elif fetchCriteria.function == 'intraday':
            try:
                fetchCriteria.function = 'TIME_SERIES_INTRADAY_EXTENDED'
                fetchCriteria.interval = [interval + 'min' for interval in fetchCriteria.interval]
                intradaydata = FetchMethods.ALPHAVANTAGE_intraday_stock_data(fetchCriteria)
            except:
                fetchCriteria.symbols = ['^' + symbol for symbol in fetchCriteria.symbols]
                fetchCriteria.interval = [interval + 'm' for interval in fetchCriteria.interval]
                intradaydata = FetchMethods.yfinance_intraday_stock_data(fetchCriteria)
        else:
            print('ERROR: You have not chosen a valid fuction to call')
        return dailydata
print('All Done')
