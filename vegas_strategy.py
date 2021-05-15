def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import talib

import matplotlib.lines as mlines
import datetime
import pandas as pd
import math
import copy
from functools import reduce
import numpy as np
import sys
import os

from util import *
import gzip

from datetime import datetime, timedelta
import math

from optparse import OptionParser
import matplotlib.ticker as ticker

import urllib.request

from io import StringIO


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


import warnings
warnings.filterwarnings("ignore")

root_folder = "C:\\Forex\\trading"

currency_file = os.path.join(root_folder, "currency.csv")

currency_df = pd.read_csv(currency_file)

currencies = list(currency_df['currency'])


currency_folders = []
data_folders = []
for currency in currencies:
    currency_folder = os.path.join(root_folder, currency)
    if not os.path.exists(currency_folder):
        os.makedirs(currency_folder)

    data_folder = os.path.join(currency_folder, "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    currency_folders += [currency_folder]
    data_folders += [data_folder]

#This is the trial app_id
app_id = "162083550794289"

url = "http://api.forexfeed.net/data/162083550794289/n-240/f-csv/i-3600/s-EURUSD,USDJPY"

def convert_to_time(timestamp):

    return datetime.fromtimestamp(timestamp)



def get_bar_data(currency, bar_number = 240, start_timestamp = -1, is_convert_to_time = True):

    global app_id

    query = "http://api.forexfeed.net/data/[app_id]/n-[bar_number]/f-csv/i-3600/s-[currency]"

    query = query.replace("[app_id]", app_id).replace("[bar_number]", str(bar_number)).replace("[currency]", currency)

    if start_timestamp != -1:
        query = query + "/st-" + str(start_timestamp)

    print("query:")
    print(query)

    with urllib.request.urlopen(query) as response:
        reply = response.read().decode("utf-8")

        #print("reply:")
        #print(reply)

        start_idx = reply.find("QUOTE START")
        end_idx = reply.find("QUOTE END")

        data_str = reply[(start_idx + len("QUOTE START ")) : end_idx]

        # print("")
        # print("data_str:")
        # print(data_str)

        #print("")
        #print("data:")

        #print(data_str)

        data_str = "currency,dummy,time,open,high,low,close\n" + data_str

        data_df = pd.read_csv(StringIO(data_str), sep=',')

        #print("data_df:")
        #print(data_df)

        #print("columns:")
        #print(data_df.columns)

        if is_convert_to_time:
            data_df['time'] = data_df['time'].apply(lambda x: convert_to_time(x))

        data_df = data_df.drop(columns = ['dummy'])

        # print("final data_df:")
        # print(data_df)


        print("data number: " + str(data_df.shape[0]))

        return data_df

    return None



initial_bar_number = 400

for currency,data_folder in list(zip(currencies, data_folders)):

    print("Receive data for currency pair " + currency)

    currency_file = os.path.join(data_folder, currency + ".csv")

    data_df = None
    if not os.path.exists(currency_file):
        temp_data_df = get_bar_data(currency, bar_number = 2, is_convert_to_time = False)
        last_timestamp = temp_data_df.iloc[-1]['time']
        start_timestamp = last_timestamp - 3600 * initial_bar_number

        print("last_timestamp = " + str(last_timestamp))
        print("start_timestamp = " + str(start_timestamp))

        data_df = get_bar_data(currency, bar_number = initial_bar_number, start_timestamp = start_timestamp)


        data_df.to_csv(currency_file, index = False)
    else:
        data_df = pd.read_csv(currency_file)
        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))

        last_timestamp = int(datetime.timestamp(data_df.iloc[-1]['time']))
        next_timestamp = last_timestamp + 3600

        incremental_data_df = get_bar_data(currency, bar_number = initial_bar_number, start_timestamp = next_timestamp)
        print("incremental_data_df:")
        print(incremental_data_df)

        if incremental_data_df.shape[0] > 0:
            data_df = pd.concat([data_df, incremental_data_df])

            data_df.reset_index(inplace = True)
            print("data_df:")
            print(data_df.head(10))

            data_df = data_df.drop(columns = ['index'])

            data_df.to_csv(currency_file, index = False)


    print("data_df:")
    print(data_df.tail(30))



















