'''A module to fetch central bank data from a variety of sources'''
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from customdatetimecreator import CustomTime

def fed_api(category):
    ''' Federal Reserve Data API
    https://fred.stlouisfed.org/categories pick category, when looking at
    graph take code from URL and put into Series ID - params may need to
    be changed

    Effective Fed Funds Rate (EFFR) chosen as it is the measure provided
    by the fed that has the biggest impact on the US stock market as a whole
    including the major stock indicies that we are focusing on
    '''

    api = 'https://api.stlouisfed.org/fred/series/observations'

    parameters = {
        "file_type": "json",
        "series_id": category,
        "realtime_start": "2017-01-27",
        "realtime_end": "9999-12-31",
        "limit": "100000",
        "offset": "0",
        "sort_order": "asc",
        "observation_start": "2017-01-27",
        "observation_end": "{}".format(CustomTime.todays_date()),
        "units": "lin",
        "aggregation_method": "avg",
        "output_type": "1",
        "api_key": "e3f70e5a440482ef1fc456be3dec47da",
        }
    response = requests.get(api, params=parameters)
    # Pulling data through API, using json to create a dict
    dict_ = json.loads(response.content)

    fed_data = pd.DataFrame(dict_['observations'])
    fed_data = fed_data.set_index('date')
    # Creating a dataframe with the fed funds rate, listing the columns
    fed_data.columns = ['RealtimeStart','RealtimeEnd','FedFundsRate']
    return fed_data

def fed_data_func(years):
    '''Input: taking a list of years in the format 'YYYY' + relevant html-snippet
    Output: a dataframe containing the announcements of the US Federal Reserve Bank
    with Date, Announcement and Category of Announcement, indexed by date
    '''
    fdframe = pd.DataFrame()
    for year in years:
    # get HTML file and format content
        url = 'https://www.federalreserve.gov/newsevents/pressreleases/{}-press.htm'.format(year)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        # parse through page for divisions needed, further sort by class
        dates = soup.find_all('div', class_='col-xs-3 col-md-2 eventlist__time')
        # scrape text from derived data and format into a dataframe
        fed_df = pd.DataFrame([date.find('time').text for date in dates], columns=['Date'])
        announcements = soup.find_all('div', class_='col-xs-9 col-md-10 eventlist__event')
        fed_df['Announcements'] = pd.DataFrame(
            [announcement.find('em').text
            for announcement in announcements]
            )
        #general scraping for necessary information
        categories = soup.find_all('p', class_='eventlist__press')
        fed_df['Category'] = pd.DataFrame(
            [category.find('strong').text
            for category in categories]
            )
        fdframe = fdframe.append(fed_df, ignore_index=True)
    # change US date format to Gregorian datetime object
    fdframe['Date'] = fdframe.Date.map(CustomTime.gregorian_date_creator)
    fdframe['AdequacyAnn'] = fdframe['Announcements'].map(
                                                          lambda x: +1
                                                          if 'Federal Open Market Committee' in x
                                                          else +1 if 'discount rate' in x
                                                          else +1 if 'deposit' in x
                                                          else +1 if 'inflation' in x
                                                          else +1 if 'rate' in x
                                                          else 0
                                                          )
    fdframe = fdframe.set_index('Date')
    # using a lambda function to create
    # a new binary adequacy column that marks
    # the announcements that contain the specific keywords
    fdframe = pd.DataFrame(fdframe)
    return fdframe
