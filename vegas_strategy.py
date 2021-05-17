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
import time
from instrument_trader import *

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

#currencies = ['CADJPY']




currency_folders = []
data_folders = []
chart_folders = []
log_files = []
for currency in currencies:
    currency_folder = os.path.join(root_folder, currency)
    if not os.path.exists(currency_folder):
        os.makedirs(currency_folder)

    print("currency_folder:")
    print(currency_folder)
    data_folder = os.path.join(currency_folder, "data")
    print("data_folder:")
    print(data_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    chart_folder = os.path.join(currency_folder, "chart")
    if not os.path.exists(chart_folder):
        os.makedirs(chart_folder)

    log_file = os.path.join(currency_folder, currency + "_log.txt")
    #if not os.path.exists(log_file):
    fd = open(log_file, 'w')
    fd.close()

    currency_folders += [currency_folder]
    data_folders += [data_folder]
    chart_folders += [chart_folder]
    log_files += [log_file]



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

is_simulate = False

# windows = [12, 30, 35, 40, 45, 50, 60, 144, 169]
# high_low_window = 100
# bolling_width = 20
#
# bar_low_percentile = 0.3
# bar_high_percentile = 0.1
#
# initial_bar_number = 400
#
# distance_to_vegas_threshold = 0.14

symbol_id = 0

query_incremental_if_stock_file_exists = True

#Query initial data for each currency pair
currency_traders = []
for currency,data_folder,chart_folder,log_file in list(zip(currencies, data_folders, chart_folders, log_files)):

    symbol_id += 1

    print_prefix = "[Currency " + currency + " " + str(symbol_id) + "] "

    print("Query initial for currency pair " + currency)

    currency_file = os.path.join(data_folder, currency + ".csv")

    data_df = None
    if not os.path.exists(currency_file):
        temp_data_df = get_bar_data(currency, bar_number = 2, is_convert_to_time = False)
        last_timestamp = temp_data_df.iloc[-1]['time']
        start_timestamp = last_timestamp - 3600 * initial_bar_number

        print("last_timestamp = " + str(last_timestamp))
        print("start_timestamp = " + str(start_timestamp))

        data_df = get_bar_data(currency, bar_number = initial_bar_number, start_timestamp = start_timestamp)


        #data_df.to_csv(currency_file, index = False)
    else:
        data_df = pd.read_csv(currency_file)
        data_df = data_df[['currency', 'time','open','high','low','close']]
        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))

        if query_incremental_if_stock_file_exists:
            last_timestamp = int(datetime.timestamp(data_df.iloc[-1]['time']))
            #next_timestamp = last_timestamp + 3600

            incremental_data_df = get_bar_data(currency, bar_number = initial_bar_number, start_timestamp = last_timestamp)
            print("incremental_data_df:")
            print(incremental_data_df)

            incremental_data_df = incremental_data_df.iloc[1:-1]

            print("Critical incremental_data_df length = " + str(incremental_data_df.shape[0]))

            if incremental_data_df.shape[0] > 0:
                data_df = pd.concat([data_df, incremental_data_df])

                data_df.reset_index(inplace = True)
                print("data_df:")
                print(data_df.head(10))

                data_df = data_df.drop(columns = ['index'])

                #data_df.to_csv(currency_file, index = False)



    # print("Before data_df:")
    # print(data_df.tail(50))


    if is_simulate:
        data_df = data_df.iloc[0:-48]

    # print("After data_df:")
    # print(data_df.tail(50))

    #sys.exit(0)




    print("data_df:")
    print(data_df.tail(10))


    currency_trader = CurrencyTrader(threading.Condition(), currency, data_folder, chart_folder, log_file)
    currency_traders += [currency_trader]

    currency_trader.feed_data(data_df)
    currency_trader.start()



incremental_bar_number = 10


def wait_for_trigger():

    current_time = datetime.now()

    temp_time = current_time + timedelta(seconds=3600)
    next_hour = datetime(temp_time.year, temp_time.month, temp_time.day, temp_time.hour, 0, 0)

    print("Next hour: " + str(next_hour))

    seconds_remaining = (next_hour - current_time).seconds

    print("Waiting for " + str(seconds_remaining) + " seconds")

    sleep_seconds = 5
    while seconds_remaining > 0:
        actual_sleep_seconds = seconds_remaining if seconds_remaining < sleep_seconds else sleep_seconds
        print("Sleep " + str(actual_sleep_seconds))
        time.sleep(actual_sleep_seconds)
        now = datetime.now()
        print("Current time: " + now.strftime("%Y-%m-%d %H:%M:%S"))
        seconds_remaining -= actual_sleep_seconds
        print("seconds_remaining = " + str(seconds_remaining))

    now = datetime.now()
    while (now - next_hour).seconds < 2:
        print("Now is " + now.strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(1)
        now = datetime.now()

    return


def wait_for_trigger_simulate():

    time.sleep(1)

    return

maximum_trial_number = 2
while True:

    print("Waiting for the next trigger")

    if is_simulate:
        wait_for_trigger_simulate()
    else:
        wait_for_trigger()

    print("Triggered")

    is_new_data_received = [False] * len(currency_traders)
    trial_numbers = [0] * len(currency_traders)

    is_all_received = False

    while not is_all_received:
        is_all_received = True
        for i in range(len(currency_traders)):
            if not is_new_data_received[i]:
                currency_trader = currency_traders[i]
                currency = currencies[i]
                last_time = currency_trader.get_last_time() #Python thinks this as UTC time, while it is actually HK time
                last_timestamp = int(last_time.timestamp())
                last_timestamp -= 28800
                #next_timestamp = last_timestamp + 3600


                print("last_timestamp = " + str(last_timestamp))
                print("last_time = " + str(last_time))
                print(last_time)


                incremental_data_df = get_bar_data(currency, bar_number=incremental_bar_number, start_timestamp=last_timestamp)

                if is_simulate:
                    if incremental_data_df is not None and incremental_data_df.shape[0] > 2:
                        incremental_data_df = incremental_data_df.iloc[0:3]

                print("incremental_data_df:")
                print(incremental_data_df)
                print("")

                if incremental_data_df is not None and incremental_data_df.shape[0] > 2:
                    incremental_data_df = incremental_data_df.iloc[1:-1]
                    currency_trader.feed_data(incremental_data_df)
                    is_new_data_received[i] = True
                    print("Received data update for " + currency)
                else:

                    if trial_numbers[i] <= maximum_trial_number + 1:
                        if trial_numbers[i] <= maximum_trial_number:
                            is_all_received = False
                            print("Not received data update for " + currency + ", will try again")
                            trial_numbers[i] += 1
                        else:
                            print("Reached maximum number of trials for " + currency + ", give up")

        if not is_all_received:
            time.sleep(2)


















##################################


    # data_df['date'] = pd.DatetimeIndex(data_df['time']).normalize()
    #
    # all_days = pd.Series(data_df['date'].unique()).dt.to_pydatetime()
    #
    # calc_high_Low(data_df, "close", high_low_window)
    # calc_jc_lines(data_df, "close", windows)
    # calc_bolling_bands(data_df, "close", bolling_width)
    # calc_macd(data_df, "close")
    #
    #
    # data_df['min_price'] = data_df[['open', 'close']].min(axis=1)
    # data_df['max_price'] = data_df[['open', 'close']].max(axis=1)
    #
    # data_df['price_range'] = data_df['max_price'] - data_df['min_price']
    #
    # data_df['low_pct_price_buy'] = data_df['min_price'] + (data_df['price_range']) * bar_low_percentile
    # data_df['high_pct_price_buy'] = data_df['max_price'] - (data_df['price_range']) * bar_high_percentile
    #
    # data_df['low_pct_price_sell'] = data_df['min_price'] + (data_df['price_range']) * bar_high_percentile
    # data_df['high_pct_price_sell'] = data_df['max_price'] - (data_df['price_range']) * bar_low_percentile
    #
    #
    # data_df['upper_vegas'] = data_df[['ma_close144', 'ma_close169']].max(axis=1)
    # data_df['lower_vegas'] = data_df[['ma_close144', 'ma_close169']].min(axis=1)
    #
    # guppy_lines = ['ma_close30', 'ma_close35', 'ma_close40', 'ma_close45', 'ma_close50', 'ma_close60']
    #
    # aligned_long_conditions = [data_df[guppy_lines[i]] > data_df[guppy_lines[i+1]] for i in range(len(guppy_lines) - 1)]
    # data_df['is_guppy_aligned_long'] = reduce(lambda left, right: left & right, aligned_long_conditions)
    #
    # aligned_short_conditions = [data_df[guppy_lines[i]] < data_df[guppy_lines[i + 1]] for i in range(len(guppy_lines) - 1)]
    # data_df['is_guppy_aligned_short'] = reduce(lambda left, right: left & right, aligned_short_conditions)
    #
    # df_temp = data_df[guppy_lines]
    # df_temp = df_temp.apply(sorted, axis=1).apply(pd.Series)
    # df_temp.columns = ['guppy1', 'guppy2', 'guppy3', 'guppy4', 'guppy5', 'guppy6']
    # data_df = pd.concat([data_df, df_temp], axis=1)
    #
    # data_df['highest_guppy'] = data_df['guppy6']
    # data_df['lowest_guppy'] = data_df['guppy1']
    #
    # data_df['ma12_gradient'] = data_df['ma_close12'].diff()
    #
    # data_df['is_vegas_up_trend'] = data_df['ma_close144'] > data_df['ma_close169']
    # data_df['is_vegas_down_trend'] = data_df['ma_close144'] < data_df['ma_close169']
    #
    # data_df['is_above_vegas'] = data_df['ma_close12'] > data_df['lower_vegas']
    # data_df['is_above_vegas_strict'] = data_df['ma_close12'] > data_df['upper_vegas']
    #
    # data_df['is_below_vegas'] = data_df['ma_close12'] < data_df['upper_vegas']
    # data_df['is_below_vegas_strict'] = data_df['ma_close12'] < data_df['lower_vegas']
    #
    #
    #
    #
    # data_df['distance_to_upper_vegas'] = data_df['ma_close12'] - data_df['upper_vegas']
    # data_df['distance_to_lower_vegas'] = data_df['ma_close12'] - data_df['lower_vegas']
    #
    # data_df['high_low_range'] = data_df['period_high' + str(high_low_window)] - data_df['period_low' + str(high_low_window)]
    #
    # data_df['pct_to_upper_vegas'] = data_df['distance_to_upper_vegas'] / data_df['high_low_range']
    # data_df['pct_to_lower_vegas'] = data_df['distance_to_lower_vegas'] / data_df['high_low_range']
    #
    # data_df['buy_weak_ready'] = data_df['is_above_vegas'] & (data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold) & (data_df['high_pct_price_buy'] < data_df['ma_close12'])
    # data_df['buy_weak_fire'] = data_df['is_above_vegas'] & (data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold) & (data_df['low_pct_price_buy'] > data_df['ma_close12']) \
    #                                 & (data_df['ma12_gradient'] >= 0)
    #
    # data_df['buy_ready'] = data_df['buy_weak_ready'] & data_df['is_vegas_up_trend']
    # data_df['buy_fire'] = data_df['buy_weak_fire'] & data_df['is_vegas_up_trend']
    #
    #
    # data_df['sell_weak_ready'] = data_df['is_below_vegas'] & (data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold) & (data_df['low_pct_price_sell'] > data_df['ma_close12'])
    # data_df['sell_weak_fire'] = data_df['is_below_vegas'] & (data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold) & (data_df['high_pct_price_sell'] < data_df['ma_close12']) \
    #                            & (data_df['ma12_gradient'] <= 0)
    #
    # data_df['sell_ready'] = data_df['sell_weak_ready'] & data_df['is_vegas_down_trend']
    # data_df['sell_fire'] = data_df['sell_weak_fire'] & data_df['is_vegas_down_trend']
    #
    #
    # data_df.to_csv(currency_file, index = False)
    #
    #
    # plot_candle_bar_charts(currency, data_df, all_days,
    #                        num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
    #                        is_plot_candle_buy_sell_points = True,
    #                        print_prefix=print_prefix,
    #                        bar_fig_folder=chart_folder)




























