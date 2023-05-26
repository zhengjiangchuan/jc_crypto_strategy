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

import shutil

from io import StringIO
import time
from instrument_trader import *

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

import warnings

warnings.filterwarnings("ignore")

parser = OptionParser()
parser.add_option("-c", "--currency", dest="currency_pair", default = "all",
                  help="Currency Pair to run")


(options, args) = parser.parse_args()

currency_to_run = options.currency_pair

app_id = "168180645499516"

class CurrencyPair:

    def __init__(self, currency, lot_size, exchange_rate, coefficient):
        self.currency = currency
        self.lot_size = lot_size
        self.exchange_rate = exchange_rate
        self.coefficient = coefficient


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



        start_idx = reply.find("QUOTE START")
        end_idx = reply.find("QUOTE END")

        data_str = reply[(start_idx + len("QUOTE START ")): end_idx]


        data_str = "currency,dummy,time,open,high,low,close\n" + data_str

        data_df = pd.read_csv(StringIO(data_str), sep=',')



        if is_convert_to_time:
            data_df['time'] = data_df['time'].apply(lambda x: convert_to_time(x))

        data_df = data_df.drop(columns=['dummy'])

        # print("final data_df:")
        # print(data_df)

        print("data number: " + str(data_df.shape[0]))

        return data_df

    return None



def start_do_trading():

    print("Child process starts")

    is_gege_server = False

    is_real_time_trading = True

    is_weekend = False

    is_do_portfolio_trading = False

    if is_gege_server:
        root_folder = "/home/min/forex/formal_trading"
    else:
        # if is_real_time_trading:
        #     root_folder = "C:\\JCForex_prod"
        # else:
        root_folder = "C:\\JCForex_prod"


    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

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


    currencies_to_run = ['USDCAD'] #'AUDCHF', 'EURAUD', 'GBPAUD', 'NZDCAD', 'NZDUSD'
    #currencies_to_run = ['NZDUSD', 'AUDUSD','AUDCAD','AUDCHF','NZDCAD','NZDCHF', 'GBPNZD']
    #currencies_to_run = ['NZDUSD', 'AUDCAD', 'EURUSD', 'NZDCAD', 'NZDcurrencies_toCHF']

    #if currency_to_run != 'all':
    if len(currencies_to_run) > 0:
        currency_df = currency_df[currency_df['currency'].isin(currencies_to_run)]


    currency_list = currency_df['currency'].tolist()

    print("currency_df:")
    print(currency_df)

    #sendEmail("Trader process starts", "")

    currency_pairs = []
    for i in range(currency_df.shape[0]):
        row = currency_df.iloc[i]
        currency_pairs += [CurrencyPair(row['currency'], row['lot_size'], row['exchange_rate'], row['close_position_coefficient'])]

    # currencies = list(currency_df['currency'])

    # currencies = ['CADJPY']


    currency_folders = []
    data_folders = []
    chart_folders = []
    simple_chart_folders = []
    log_files = []
    data_files = []
    trade_files = []
    performance_files = []

    #chart_folder_name = "chart_ratio1Adjust_USDCAD2_newStuff_April_EURJPY2_noConsecutive_0512_correct2_filter"
    chart_folder_name = "chart_ratio1Adjust_0512_correct2_filter2_realTime_w2_erase2"
    for currency_pair in currency_pairs:

        currency = currency_pair.currency

        currency_folder = os.path.join(root_folder, currency)
        currency_data_folder = os.path.join(root_folder, currency, 'data')
        if not os.path.exists(currency_folder):
            os.makedirs(currency_folder)

        if not os.path.exists(currency_data_folder):
            os.makedirs(currency_data_folder)

        print("currency_folder:")
        print(currency_folder)
        data_folder = os.path.join(currency_folder, "data")
        print("data_folder:")
        print(data_folder)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        chart_folder = os.path.join(currency_folder, chart_folder_name)
        if not os.path.exists(chart_folder):
            os.makedirs(chart_folder)

        simple_chart_folder = os.path.join(currency_folder, "simple_chart")
        if not os.path.exists(simple_chart_folder):
            os.makedirs(simple_chart_folder)

        log_file = os.path.join(currency_folder, currency + "_log.txt")
        if not os.path.exists(log_file):
            fd = open(log_file, 'w')
            fd.close()

        data_file = os.path.join(currency_data_folder, currency + ".csv")
        trade_file = os.path.join(currency_folder, currency + "_all_trades.csv")
        performance_file = os.path.join(currency_folder, currency + "_performance.csv")
        print("Fuck performance_file " + performance_file)

        currency_folders += [currency_folder]
        data_folders += [data_folder]
        chart_folders += [chart_folder]
        simple_chart_folders += [simple_chart_folder]
        log_files += [log_file]
        data_files += [data_file]
        trade_files += [trade_file]
        performance_files += [performance_file]

    currency_traders = []

    is_new_data_received = [False] * len(currency_pairs)
    is_traded_first_time = [False] * len(currency_pairs)
    trial_numbers = [0] * len(currency_pairs)

    is_all_received = False

    maximum_trial_number = 3

    for currency_pair, data_folder, chart_folder, simple_chart_folder, log_file, data_file, trade_file, performance_file in list(
            zip(currency_pairs, data_folders, chart_folders, simple_chart_folders, log_files, data_files, trade_files, performance_files)):

        currency = currency_pair.currency
        lot_size = currency_pair.lot_size
        exchange_rate = currency_pair.exchange_rate
        coefficient = currency_pair.coefficient

        #print("Here performance_file = " + performance_file)
        currency_trader = CurrencyTrader(threading.Condition(), currency, lot_size, exchange_rate, coefficient, data_folder,
                                         chart_folder, simple_chart_folder, log_file, data_file, trade_file, performance_file)
        currency_trader.daemon = True

        currency_traders += [currency_trader]

    print("data_folders:")
    print(data_folders)


    is_do_trading = True

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

                    #if is_real_time_trading:
                    data_file = os.path.join(data_folder, currency + ".csv")
                    print("data_file:")
                    print(data_file)
                    # else:
                    #     data_file = os.path.join(data_folder, currency + "100.csv") #Temporary
                    #data_file = currency_trader.data_file  #Permanent

                    # #Temp
                    # if os.path.exists(data_file):
                    #     os.remove(data_file)
                    #     print("Remove data_file " + data_file)
                    #
                    # continue

                    data_df = None

                    if os.path.exists(data_file):

                        data_df = pd.read_csv(data_file)
                        #data_df100 = data_df100.iloc[0:-20]

                        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))


                        data_df = data_df[['currency', 'time', 'open', 'high', 'low', 'close']]

                        #data_df = data_df.iloc[0:-50]

                        # print("data_df:")
                        # print(data_df.tail(10))

                        last_time = data_df.iloc[-1]['time']
                        print("last_time = " + str(last_time))
                        last_timestamp = int(datetime.timestamp(last_time)) #- 28800
                        # next_timestamp = last_timestamp + 3600

                        print("Here last time = " + str(last_time))
                        print("last_timestamp = " + str(last_timestamp))
                        # time.sleep(15)

                        if is_real_time_trading:
                            incremental_data_df = get_bar_data(currency, bar_number=initial_bar_number, start_timestamp=last_timestamp)
                            # print("incremental_data_df:")
                            # print(incremental_data_df)

                            # if is_weekend:
                            #     incremental_data_df = incremental_data_df[incremental_data_df['time'] > last_time]
                            # else:
                            incremental_data_df = incremental_data_df[incremental_data_df['time'] > last_time].iloc[0:-1]

                            print("incremental_data_df length = " + str(incremental_data_df.shape[0]))

                            # print("incremental_data_df after:")
                            # print(incremental_data_df)

                            # incremental_data_df = incremental_data_df.iloc[1:-1]

                            print("Critical incremental_data_df length = " + str(incremental_data_df.shape[0]))

                        if is_real_time_trading and incremental_data_df.shape[0] > 0:


                            data_df = pd.concat([data_df, incremental_data_df])


                            data_df.reset_index(inplace=True)
                            data_df = data_df.drop(columns=['index'])

                    else:
                        print("Currency file does not exit, query initial data from web")
                        temp_data_df = get_bar_data(currency, bar_number=2, is_convert_to_time=False)
                        last_timestamp = temp_data_df.iloc[-1]['time']
                        print("last_timestamp: " + str(last_timestamp))
                        print(datetime.fromtimestamp(last_timestamp))
                        start_timestamp = last_timestamp - 3600 * (initial_bar_number-1)

                        print("last_timestamp = " + str(last_timestamp))
                        print("start_timestamp = " + str(start_timestamp))

                        data_df = get_bar_data(currency, bar_number=initial_bar_number, start_timestamp=start_timestamp)

                        data_df = data_df.iloc[:-1]


                    if is_real_time_trading and not is_weekend:

                        if data_df is not None and data_df.shape[0] > 0:
                            last_time = data_df.iloc[-1]['time']
                        else:
                            last_time = None

                        if last_time is not None:
                            delta = datetime.now() - last_time
                            if delta is not None and delta.seconds > 0 and delta.seconds < 7200 and delta.days == 0:
                                print("Received up-to-date data for currency pair " + currency)

                                is_new_data_received[i] = True
                                currency_trader.feed_data(data_df)
                                currency_trader.trade()
                            else:
                                if trial_numbers[i] <= maximum_trial_number:
                                    is_all_received = False
                                    print("Not received data update for " + currency + ", will try again")
                                    trial_numbers[i] += 1
                                else:
                                    print("Reached maximum number of trials for " + currency + ", give up")
                    else:

                        if data_df is not None:

                            is_new_data_received[i] = True

                            print("Start trading")
                            currency_trader.feed_data(data_df)
                            currency_trader.trade()



        #sendEmail("Trader process ends", "")

        print("Finished trading *********************************")


        print("Collecting Results....")

        perf_dfs = []
        trade_dfs = []
        for currency in currency_list:
            perf_file = os.path.join(root_folder, currency, currency + "_performance.csv")
            perf_dfs += [pd.read_csv(perf_file)]

            trade_file = os.path.join(root_folder, currency, currency + "_all_trades.csv")
            trade_df = pd.read_csv(trade_file)
            trade_dfs += [trade_df]

        perf_df = pd.concat(perf_dfs)
        trade_df = pd.concat(trade_dfs)
        trade_df = trade_df.sort_values(by = ['entry_time'])

        print("Final Performance Result:")
        perf_df.reset_index(inplace = True)
        perf_df = perf_df.drop(columns = ['index'])
        print(perf_df)

        perf_df.to_csv(os.path.join(root_folder, chart_folder_name + ".csv"), index = False)

        des_pnl_folder = os.path.join(root_folder, 'all_pnl_' + chart_folder_name)
        if not os.path.exists(des_pnl_folder):
            os.makedirs(des_pnl_folder)

        old_pnl_files = os.listdir(des_pnl_folder)
        for file in old_pnl_files:
            os.remove(os.path.join(des_pnl_folder, file))


        des_bar_folder = os.path.join(root_folder, 'all_bars_' + chart_folder_name)
        if not os.path.exists(des_bar_folder):
            os.makedirs(des_bar_folder)

        old_bar_files = os.listdir(des_bar_folder)
        for file in old_bar_files:
            os.remove(os.path.join(des_bar_folder, file))


        print("Copying bar charts and pnl charts...")
        trade_df = trade_df.drop(columns = ['id', 'pnl', 'cum_pnl', 'reverse_pnl', 'cum_reverse_pnl'])
        trade_df.to_csv(os.path.join(des_pnl_folder, "all_trades.csv"), index = False)

        for currency in currency_list:
            #print("currency = " + str(currency))
            pic_path = os.path.join(root_folder, currency, chart_folder_name, currency + '_pnl.png')
            if os.path.exists(pic_path):
                shutil.copy2(pic_path, des_pnl_folder)

            currency_chart_folder = os.path.join(root_folder, currency, chart_folder_name)
            chart_files = os.listdir(currency_chart_folder)
            for chart_file in chart_files:
                if 'pnl' not in chart_file:
                    shutil.copy2(os.path.join(currency_chart_folder, chart_file), des_bar_folder)


        #shutil.copy2(file_path, dest_folder)


    if False:
        # print("Sleeping")
        # time.sleep(10)
        #dest_folder = "C:\\Users\\User\\Dropbox\\forex_real_time_new4_check_2barContinuous"

        #dest_folder = "C:\\Users\\User\\Dropbox\\forex_real_time_new2_improve_filter_vegas_guppy_other_side_fixBug_15"

        #dest_folder = "C:\\Forex\\new_experiments\\0803\\forex_innovativeFire2new_clean_entry_second_entry_Improve2"

        dest_folder = "C:\\Forex\\formal_trading\\All_Charts"

        #dest_folder = "C:\\Forex\\new_experiments\\0924\\forex_innovativeFire2new_trend_relaxVegas_includeMore_guppyAligned_closeLogic_twoClose_corrected_upToDate_fixBug2"



        #dest_folder = "C:\\Forex\\new_experiments\\0529\\final\\original_strategy"
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

            if currency_to_run in file:
                file_path = os.path.join(dest_folder, file)
                os.remove(file_path)


        symbol_folders = [os.path.join(root_folder, file) for file in os.listdir(root_folder)
                          if os.path.isdir(os.path.join(root_folder, file)) and 'pnl' not in file and 'portfolio' not in file]

        print("symbol_folders:")
        print(symbol_folders)



        currency_list = list(currency_df['currency'])
        #print("currency_list*************************:")
        print(currency_list)

        for symbol_folder in symbol_folders:

            #print('symbol_folder =' + symbol_folder)

            if symbol_folder[-6:] not in currency_list:
                continue


            # if symbol_folder[-6:] not in selected_ones:
            #     continue

            print("Process symbol folder " + symbol_folder)
            chart_folder = os.path.join(symbol_folder, "chart")

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
    #sys.exit(0)


#start_do_trading()