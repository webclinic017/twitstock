'''A module to create custom, reusable datetime objects'''
import datetime
from datetime import timedelta

class CustomTime:
    '''Create custom time and datetime objects'''
    def __init__(self):
        pass

    @staticmethod
    def todays_date():
        '''A function to return todays date as a datetime object in the format:
        YYYY-MM-DD
        '''
        return datetime.date.today().strftime('%Y-%m-%d')

    @staticmethod
    def date_creator(str_date):
        ''' A function to take a date in string format:
        YYYY-MM-DD
        and output a datetime oject in the format:
        YYYY-MM-DD
        '''
        return datetime.datetime.strptime(str_date, '%Y-%m-%d')

    @staticmethod
    def date_creator2(str_date):
        '''A function to take a date in string format:
        WEEKDAY(abbr) MONTH(abbr) DAYOFTHEMONTH YEAR
        and output a datetime object in the format:
        WEEKDAY(abbr) MONTH(abbr) DAYOFTHEMONTH YEAR
        '''
        return datetime.datetime.strptime(str_date, '%a %b %d %Y')

    @staticmethod
    def time_creator(str_time):
        '''A function to take a time in a string format:
        WEEKDAY(abbr) MONTH DAYOFTHEMONTH HOURS:MINUTES:SECONDS TIMEZONE YEAR
        and return a datetime object in the format:
        WEEKDAY(abbr) MONTH DAYOFTHEMONTH HOURS:MINUTES:SECONDS TIMEZONE YEAR
        '''
        return datetime.datetime.strptime(str_time, '%a %b %d %H:%M:%S %z %Y')
    
    @staticmethod
    def time_creator2(str_time):
        '''A function to take a time in a string format:
        YYYY-MM-DD HOURS:MINUTES:SECONDS
        and return a datetime object in the format:
        YYYY-MM-DD HOURS:MINUTES:SECONDS
        '''
        return datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def gregorian_date_creator(str_date):
        '''A function to take a date in a string format:
        MM-DD-YYYY
        and return a datetime object in the format:
        YYYY-MM-DD
        '''
        return datetime.datetime.strftime(
            datetime.datetime.strptime(str_date, '%m/%d/%Y'),'%Y-%m-%d'
            )

    @staticmethod
    def date60daysago():
        '''A function to return the date 60 days ago in the format:
        YYYY-MM-DD
        '''
        return datetime.date.today() - timedelta(days=59)
