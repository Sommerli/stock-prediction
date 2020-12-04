import pandas as pd
from pytrends import dailydata
import os.path
from os import path
import pageviewapi as wik
from datetime import date, datetime as dt, timedelta
import pandas as pd

import GoogleData
import WikiData
import MarkedData

ticker = 'AAPL'
from_date = '2020-11-01'


#df = pd.concat([GoogleData.get_daily_google_data(MarkedData.get_kw_list(ticker), 2016, 1), WikiData.get_daily_wiki_data(ticker, '2016-01-01'), MarkedData.get_daily_marked_data(ticker, '2016-01-01')], axis=1)
#print(df)


# shorttest = pd.concat([GoogleData.get_daily_google_data(MarkedData.get_kw_list(ticker), int(from_date.split('-')[0]), int(from_date.split('-')[1])), WikiData.get_daily_wiki_data(ticker, from_date), MarkedData.get_daily_marked_data(ticker, from_date)], axis=1)
#print(shorttest)
# df = pd.DataFrame()
# GoogleData.get_daily_google_data(MarkedData.get_kw_list(ticker), 2020, 10))
# print(WikiData.get_daily_wiki_data(MarkedData.get_kw_list(ticker)[1], '2020-10-01'))
# print(MarkedData.get_daily_marked_data(ticker, '2020-10-01'))
