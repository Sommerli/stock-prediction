import pandas as pd
from pytrends import dailydata
import os.path
from os import path
from datetime import date, datetime as dt


def get_company_name(comp_name):
    if len(comp_name) > 2:
        return comp_name[0:1]
    else:
        return comp_name[0]


def get_daily_google_data(kw_list, from_year, from_month):
    # kw_list - [0] ticker, [1] full name, [2] name without corporation classification

    if path.exists(f'googledata/{kw_list[0]}.csv'):
        return pd.read_csv(f'googledata/{kw_list[0]}.csv', index_col="Date")  # .set_index('date', inplace=True)
    else:
        df = pd.DataFrame()
        for kw in kw_list:
            df[kw] = dailydata.get_daily_data(kw, from_year, from_month, dt.now().year, dt.now().month)[kw]
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index)
        df.to_csv(f'googledata/{kw_list[0]}.csv')
        return df

# print(df.columns)
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df.set_index('date', inplace=True)
# return df

# # TEST of above code - gets google data from january 2019 for given stock
# kw_list = 'AAPL', 'apple'
# df = get_daily_google_data(kw_list, 2020, 11)
# print(df)
