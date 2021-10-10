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


class CurrencyPair:

    def __init__(self, currency, lot_size, exchange_rate):
        self.currency = currency
        self.lot_size = lot_size
        self.exchange_rate = exchange_rate

print("Child process starts")

is_gege_server = False

is_do_portfolio_trading = True

if is_gege_server:
    root_folder = "/home/min/forex/formal_trading"
else:
    root_folder = "C:\\Forex\\formal_trading"


communicate_files = [file for file in os.listdir(root_folder) if "communicate" in file]
communicate_nums = [int(communicate_file[len('communicate'):-len('.txt')]) for communicate_file in communicate_files]
if len(communicate_nums) > 0:
    max_idx = np.array(communicate_nums).argmax()
else:
    max_idx = 0
    communicate_file = os.path.join(root_folder, "communicate1.txt")
    fd = open(communicate_file, 'w')
    fd.close()
    communicate_files += [communicate_file]
communicate_file = os.path.join(root_folder, communicate_files[max_idx])

currency_file = os.path.join(root_folder, "currency.csv")

currency_df = pd.read_csv(currency_file)

# Hutong
#currency_df = currency_df[currency_df['currency'].isin(['NZDUSD'])]
#currency_df = currency_df[currency_df['currency'].isin(['AUDCHF', 'AUDJPY', 'CADCHF'])]
#currency_df = currency_df[currency_df['currency'].isin(['AUDCHF', 'AUDJPY', 'CADCHF', 'AUDCAD'])]
#currency_df = currency_df[currency_df['currency'].isin(['USDJPY', 'CADJPY', 'EURJPY', 'CADCHF', 'GBPJPY', 'USDCHF'])]

# currency_df = currency_df[~currency_df['currency'].isin(['AUDNZD', 'EURCHF', 'EURNZD','GBPAUD',
#                                                         'GBPCAD', 'GBPCHF', 'USDCAD'])]

#currency_df = currency_df[currency_df['currency'].isin(['AUDJPY'])]

#currency_df = currency_df[currency_df['currency'].isin(['AUDJPY', 'EURCAD', 'NZDJPY', 'USDCAD', 'NZDUSD'])]

currency_df = currency_df[currency_df['currency'].isin(['CADCHF', 'USDJPY', 'GBPJPY'])]


#currency_df = currency_df[currency_df['currency'].isin(['AUDCAD', 'AUDCHF', 'CADCHF', 'CADJPY', 'EURGBP','GBPCHF'])]

#currency_df = currency_df[currency_df['currency'].isin(['CADJPY', 'EURAUD', 'EURCAD', 'EURJPY', 'AUDUSD', 'EURUSD', 'NZDJPY'])]


# print("currency_df:")
# print(currency_df)

# currency_df = currency_df[currency_df['currency'].isin(['CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'EURJPY',
#                                                         'EURCHF', 'GBPCHF', 'USDCHF'])]

#selected_ones = ['AUDCHF', 'AUDJPY','AUDUSD','EURGBP','EURJPY','EURUSD','GBPJPY','GBPUSD','NZDCAD','NZDUSD']
#selected_ones = ['AUDJPY']

#currency_df = currency_df[currency_df['currency'].isin(selected_ones)]


# start_id = which(currency_df['currency'] == 'CADCHF')[0]
# currency_df = currency_df.iloc[start_id:]



print("currency_df:")
print(currency_df)

sendEmail("Trader process starts", "")

currency_pairs = []
for i in range(currency_df.shape[0]):
    row = currency_df.iloc[i]
    currency_pairs += [CurrencyPair(row['currency'], row['lot_size'], row['exchange_rate'])]

# currencies = list(currency_df['currency'])

# currencies = ['CADJPY']


currency_folders = []
data_folders = []
chart_folders = []
simple_chart_folders = []
log_files = []
for currency_pair in currency_pairs:

    currency = currency_pair.currency

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

    simple_chart_folder = os.path.join(currency_folder, "simple_chart")
    if not os.path.exists(simple_chart_folder):
        os.makedirs(simple_chart_folder)

    log_file = os.path.join(currency_folder, currency + "_log.txt")
    if not os.path.exists(log_file):
        fd = open(log_file, 'w')
        fd.close()


    currency_folders += [currency_folder]
    data_folders += [data_folder]
    chart_folders += [chart_folder]
    simple_chart_folders += [simple_chart_folder]
    log_files += [log_file]

# This is the trial app_id
app_id = "162083550794289"

url = "http://api.forexfeed.net/data/162083550794289/n-240/f-csv/i-3600/s-EURUSD,USDJPY"


def convert_to_time(timestamp):
   #return datetime.fromtimestamp(timestamp+28800)
    return datetime.fromtimestamp(timestamp)


def get_bar_data(currency, bar_number=240, start_timestamp=-1, is_convert_to_time=True):
    global app_id

    query = "http://api.forexfeed.net/data/[app_id]/n-[bar_number]/f-csv/i-3600/s-[currency]"

    query = query.replace("[app_id]", app_id).replace("[bar_number]", str(bar_number)).replace("[currency]", currency)

    if start_timestamp != -1:
        query = query + "/st-" + str(start_timestamp)

    print("query:")
    print(query)

    with urllib.request.urlopen(query) as response:
        reply = response.read().decode("utf-8")

        # print("reply:")
        # print(reply)

        start_idx = reply.find("QUOTE START")
        end_idx = reply.find("QUOTE END")

        data_str = reply[(start_idx + len("QUOTE START ")): end_idx]

        # print("")
        # print("data_str:")
        # print(data_str)

        # print("")
        # print("data:")

        # print(data_str)



        data_str = "currency,dummy,time,open,high,low,close\n" + data_str

        data_df = pd.read_csv(StringIO(data_str), sep=',')

        # print("strange data_df:")
        # print(data_df)

        # print("My My data_df:")
        # print(data_df)
        #
        # print("columns:")
        # print(data_df.columns)

        # print("data_df:")
        # print(data_df)

        if is_convert_to_time:
            data_df['time'] = data_df['time'].apply(lambda x: convert_to_time(x))

        data_df = data_df.drop(columns=['dummy'])

        # print("final data_df:")
        # print(data_df)

        print("data number: " + str(data_df.shape[0]))

        return data_df

    return None


is_simulate = False



symbol_id = 0

query_incremental_if_stock_file_exists = True



currency_traders = []

is_new_data_received = [False] * len(currency_pairs)
is_traded_first_time = [False] * len(currency_pairs)
trial_numbers = [0] * len(currency_pairs)

is_all_received = False

maximum_trial_number = 3

for currency_pair, data_folder, chart_folder, simple_chart_folder, log_file in list(
        zip(currency_pairs, data_folders, chart_folders, simple_chart_folders, log_files)):

    currency = currency_pair.currency
    lot_size = currency_pair.lot_size
    exchange_rate = currency_pair.exchange_rate

    currency_trader = CurrencyTrader(threading.Condition(), currency, lot_size, exchange_rate, data_folder,
                                     chart_folder, simple_chart_folder, log_file)
    currency_trader.daemon = True

    currency_traders += [currency_trader]

print("data_folders:")
print(data_folders)

is_first_time = True
original_data_df100 = None
original_data_df200 = None

is_do_trading = True

is_append_new_data = True

if is_do_trading:
    while not is_all_received:
        is_all_received = True
        for i in range(len(currency_traders)):
            if not is_new_data_received[i]:
                currency_trader = currency_traders[i]

                data_folder = data_folders[i]

                currency = currency_trader.currency

                print_prefix = "[Currency " + currency + "] "

                print("Query initial for currency pair " + currency)

                currency_file = os.path.join(data_folder, currency + ".csv")
                currency_file100 = os.path.join(data_folder, currency + "100.csv")
                currency_file200 = os.path.join(data_folder, currency + "200.csv")

                data_df = None

                # print("currency_file:")
                # print(currency_file)

                if (os.path.exists(currency_file100) and os.path.exists(currency_file200)) or os.path.exists(currency_file):

                    if os.path.exists(currency_file100) and os.path.exists(currency_file200):
                        data_df100 = pd.read_csv(currency_file100)
                        #data_df100 = data_df100.iloc[0:-20]

                        data_df100['time'] = data_df100['time'].apply(lambda x: preprocess_time(x))

                        data_df200 = pd.read_csv(currency_file200)
                        #data_df200 = data_df200.iloc[0:-20]

                        data_df200['time'] = data_df200['time'].apply(lambda x: preprocess_time(x))

                        original_data_df100 = data_df100.copy()
                        original_data_df200 = data_df200.copy()

                        #print("Initial column number = " + str(len(data_df.columns)))
                        data_df = data_df100[['currency', 'time', 'open', 'high', 'low', 'close']]

                        # print("Fuck data_df:")
                        # print(data_df.tail(3))
                        #sys.exit(0)

                    else:
                        data_df = pd.read_csv(currency_file)
                        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))
                        data_df = data_df[['currency', 'time', 'open', 'high', 'low', 'close']]


                    print("data_df:")
                    print(data_df.tail(10))

                    last_time = data_df.iloc[-1]['time']
                    last_timestamp = int(datetime.timestamp(last_time)) #- 28800
                    # next_timestamp = last_timestamp + 3600

                    print("Here last time = " + str(last_time))
                    print("last_timestamp = " + str(last_timestamp))
                    # time.sleep(15)


                    if is_append_new_data:
                        incremental_data_df = get_bar_data(currency, bar_number=initial_bar_number, start_timestamp=last_timestamp)
                        print("incremental_data_df:")
                        print(incremental_data_df.tail(10))

                        incremental_data_df = incremental_data_df[incremental_data_df['time'] > last_time].iloc[0:-1]

                        # incremental_data_df = incremental_data_df.iloc[1:-1]

                        print("Critical incremental_data_df length = " + str(incremental_data_df.shape[0]))

                    if is_append_new_data and incremental_data_df.shape[0] > 0:

                        # print("cruise incremental_data_df:")
                        # print(incremental_data_df)
                        # print("before concat data_df:")
                        # print(data_df[['time','close','period_high100']].tail(10))
                        data_df = pd.concat([data_df, incremental_data_df])

                        # print("Just after concat")
                        # print(data_df[['time','close','period_high100']].tail(50))
                        #
                        # data_df.reset_index(inplace = True)
                        # data_df = data_df.drop(columns = ['index'])
                        # print("data_df after concat")
                        # print(data_df[['time','close','period_high100']].tail(50))

                        data_df.reset_index(inplace=True)
                        data_df = data_df.drop(columns=['index'])

                        print("Fuck data_df:")
                        print(data_df.tail(3))
                        #sys.exit(0)

                        # print("data_df:")
                        # print("Correct column number = " + str(len(data_df.columns)))
                        # print(data_df.tail(10))

                else:
                    print("Currency file does not exit, query initial data from web")
                    temp_data_df = get_bar_data(currency, bar_number=2, is_convert_to_time=False)
                    last_timestamp = temp_data_df.iloc[-1]['time']
                    start_timestamp = last_timestamp - 3600 * initial_bar_number

                    print("last_timestamp = " + str(last_timestamp))
                    print("start_timestamp = " + str(start_timestamp))

                    data_df = get_bar_data(currency, bar_number=initial_bar_number, start_timestamp=start_timestamp)


                if data_df is not None and (not is_traded_first_time[i]):
                    currency_trader.feed_data(data_df, original_data_df100, original_data_df200)
                    currency_trader.start()
                    is_traded_first_time[i] = True


                if data_df is not None and data_df.shape[0] > 0:
                    last_time = data_df.iloc[-1]['time']
                else:
                    last_time = None

                print("last_time = " + str(last_time))
                if last_time is not None:
                    delta = last_time - datetime.now()
                else:
                    delta = None
                if (delta is not None and delta.seconds < 7200 and delta.days == 0) or is_first_time:
                    print("Received update-to-date data")

                    delta = last_time - datetime.now()

                    #print((last_time - datetime.now()))
                    currency_trader.feed_data(data_df, original_data_df100, original_data_df200)
                    is_new_data_received[i] = True

                else:
                    print("Go here 2")
                    if trial_numbers[i] <= maximum_trial_number + 1:
                        if trial_numbers[i] <= maximum_trial_number:
                            is_all_received = False
                            print("Not received data update for " + currency + ", will try again")
                            trial_numbers[i] += 1
                        else:
                            print("Reached maximum number of trials for " + currency + ", give up")

        is_first_time = False

    sendEmail("Trader process ends", "")

    print("Finished trading *********************************")

if True:
    print("Sleeping")
    time.sleep(10)
    #dest_folder = "C:\\Users\\User\\Dropbox\\forex_real_time_new4_check_2barContinuous"

    #dest_folder = "C:\\Users\\User\\Dropbox\\forex_real_time_new2_improve_filter_vegas_guppy_other_side_fixBug_15"

    #dest_folder = "C:\\Forex\\new_experiments\\0803\\forex_innovativeFire2new_clean_entry_second_entry_Improve2"



    #dest_folder = "C:\\Forex\\new_experiments\\0924\\forex_innovativeFire2new_trend_relaxVegas_includeMore_guppyAligned_closeLogic_twoClose_corrected_upToDate_fixBug2"



    dest_folder = "C:\\Forex\\new_experiments\\1010\\final\\forex_noTrendFollowing_SpecialExclude_selected3"
    #dest_folder = "C:\\Forex\\new_experiments\\0924\\forex_noTrendFollowing_selected"


    #dest_folder = "C:\\Forex\\new_experiments\\0918\\forex_innovativeFire2new_quickLossDelayed_reentryrequire4GuppyLines_reentry_improve_fire2_partialBelow_removeSpecial_simpleQuickStop_trend_relaxVegas_includeMore_guppyAligned_closeLogic_twoClose_corrected"

    #dest_folder = "C:\\Forex\\new_experiments\\0914\\forex_innovativeFire2new_quickLossDelayed_reentryrequire4GuppyLines_reentry_improve_fire2"

    #dest_folder = "C:\\Forex\\new_experiments\\0914\\forex_innovativeFire2new_quickLossDelayed_reentryrequire4GuppyLines_reentry_improve_fire2_smallPortfolio"


    #dest_folder = "C:\\Forex\\new_experiments\\0904\\forex_innovativeFire2new_quickLossDelayed_reentryrequire4GuppyLines"




    #dest_folder = "C:\\Forex\\new_experiments\\0627\\not_support_half_close"

    print("Wakeup")

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in os.listdir(dest_folder):
        file_path = os.path.join(dest_folder, file)
        os.remove(file_path)


    symbol_folders = [os.path.join(root_folder, file) for file in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, file)) and 'pnl' not in file and 'portfolio' not in file]
    import shutil

    currency_list = list(currency_df['currency'])
    #print("currency_list*************************:")
    print(currency_list)

    for symbol_folder in symbol_folders:

        #print('symbol_folder =' + symbol_folder)

        if symbol_folder[-6:] not in currency_list:
            continue


        # if symbol_folder[-6:] not in selected_ones:
        #     continue

        print(symbol_folder)
        chart_folder = os.path.join(symbol_folder, "simple_chart")

        files = os.listdir(chart_folder)
        if len(files) == 6:
            files = files[1:]

        #files = files[-1:]

        for file in files:
            file_path = os.path.join(chart_folder, file)

            print("file_path = " + file_path)
            print("dest_folder = " + dest_folder)
            shutil.copy2(file_path, dest_folder)


    sendEmail("Charts sent!", "")


if is_do_portfolio_trading:
    print("1 is_do_portfolio_trading = " + str(is_do_portfolio_trading))
    os.system('python plot_pnl_curve.py')
else:
    print("2 is_do_portfolio_trading = " + str(is_do_portfolio_trading))

print("All finished")
sys.exit(0)


