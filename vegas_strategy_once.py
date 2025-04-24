def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
#import talib

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

from twelvedata import TDClient

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

use_dynamic_TP = True

use_short_data_for_prod = False #This should always be FALSE on my own machine!!!

is_run_individual_good_ones = False
is_run_aggregated_good_ones = False

profit_loss_ratio = 1

read_5min_data = True

if use_dynamic_TP:
    profit_loss_ratio = 10

class CurrencyPair:

    def __init__(self, currency, lot_size, exchange_rate, coefficient, actual_maxdrawdown, optimal_gradient_num, optimal_gradient_num_execution, decimal):
        self.currency = currency
        self.lot_size = lot_size
        self.exchange_rate = exchange_rate
        self.coefficient = coefficient
        self.actual_maxdrawdown = actual_maxdrawdown
        self.optimal_gradient_num = optimal_gradient_num
        self.optimal_gradient_num_execution = optimal_gradient_num_execution
        self.decimal = decimal


def convert_to_time(timestamp):
   #return datetime.fromtimestamp(timestamp+28800)
    return datetime.fromtimestamp(timestamp)

def get_bar_data2(currency, bar_number=240, interval = "1h", end_date = None, start_timestamp=-1, is_convert_to_time = True):
    # Initialize client - apikey parameter is requiered
    td = TDClient(apikey="dbc2c6a6a33840d4b2a11a371def5973")

    print("")
    print("Now = " + str(datetime.now()))
    print("initial_bar_number = " + str(initial_bar_number))
    # Construct the necessary time series
    ts = td.time_series(
        symbol=currency[:-3] + '/' + currency[-3:],
        interval=interval,
        outputsize=bar_number, #initial_bar_number
        end_date=end_date,
        timezone="Asia/Singapore",
    )

    # Returns pandas.DataFrame
    data_df = ts.as_pandas()

    data_df = data_df.iloc[::-1]

    data_df.reset_index(inplace=True)

    data_df = data_df.rename(columns = {'datetime' : 'time'})

    data_df['currency'] = currency

    data_df = data_df[['time', 'currency', 'open', 'high', 'low', 'close']]

    print("Row number = " + str(data_df.shape[0]) + " &&")
    #
    print("here printing")
    print(data_df.iloc[-20:])

    return data_df





def get_bar_data(currency, bar_number=240, start_timestamp=-1, is_convert_to_time=True):
    global app_id

    query = "http://api.forexfeed.net/data/[app_id]/n-[bar_number]/f-csv/i-3600/s-[currency]"

    query = query.replace("[app_id]", app_id).replace("[bar_number]", str(bar_number)).replace("[currency]", currency)

    # if start_timestamp != -1:
    #     query = query + "/st-" + str(start_timestamp)

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




def preprocess_data(data_df):
    #data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))

    data_df['prev_time'] = data_df['time'].shift(1)

    data_df['time_delta'] = data_df['time'] - data_df['prev_time']

    data_df['delta_seconds'] = data_df['time_delta'].apply(lambda x: x.seconds).fillna(0).astype(int)
    data_df['delta_days'] = data_df['time_delta'].apply(lambda x: x.days).fillna(0).astype(int)

    data_df['total_seconds'] = data_df['delta_days'] * 24 * 3600 + data_df['delta_seconds']

    #print(type(data_df.iloc[-1]['time_delta']))
    #print(data_df.iloc[-1]['time_delta'].seconds)

    # print("###########")
    # print("Temp data")
    # print(data_df.iloc[1500:1510])
    # print("###########")

    critical_index = list(which(data_df['total_seconds'] > 3600)) + [data_df.shape[0]]

    sub_dfs = []

    print(critical_index)

    print("critical_index length = " + str(len(critical_index)))
    print("")

    start = 0
    for i in range(len(critical_index)):

        # print("i = " + str(i))
        # print("start = " + str(start))
        # print("end = " + str(critical_index[i]))
        sub_df = data_df.iloc[start:critical_index[i]]
        # print("sub_df length = " + str(sub_df.shape[0]))
        # print("")
        start = critical_index[i]

        sub_dfs += [sub_df]

    last_close_price = None
    price_cols = ['open', 'high', 'low', 'close']
    new_sub_dfs = []
    for j in range(len(sub_dfs)):

       # print("j = " + str(j))

        sub_df = sub_dfs[j]

        # print("now sub_df.columns = ")
        # print(sub_df.columns)
        # print("length = " + str(sub_df.shape[0]))

        ########Added Code ##########
        if sub_df.shape[0] < 2:

            continue

        #############################

        #print("sub df size = " + str(sub_df.shape[0]))

        # sub_df.at[sub_df.index[0], 'open'] = 0.0

        currency = sub_df.iloc[0]['currency']
        first_time = sub_df.iloc[0]['time']
        last_time = sub_df.iloc[-1]['time']

        # print("first_time = " + str(first_time))
        # print("last_time = " + str(last_time))
        # print("")

        #     print("Old head:")
        #     display(sub_df.iloc[0:5])

        #     print("Old tail:")
        #     display(sub_df.iloc[-5:])


        if last_close_price is not None:

            # if j == 32:
            #     print("j = " + str(j))
            #
            #     print("Before sub_df:")
            #     print(sub_df)

            if first_time.hour < 5:
                sub_df = sub_df.iloc[1:]

            # if j == 32:
            #     print("After sub_df:")
            #     print(sub_df)
            #
            #     print("")

            if sub_df.iloc[0]['time'].hour == 5:
                for col in price_cols:
                    sub_df.at[sub_df.index[0], col] = last_close_price

        if j < len(sub_dfs) - 1:

            last_close_price = sub_df.iloc[-1]['close']

            if last_time.hour < 5:
                add_row_num = 5 - last_time.hour

                added_data = []
                for i in range(1, add_row_num + 1):
                    new_time = last_time + timedelta(hours=i)
                    added_data += [[currency, new_time] + [last_close_price] * 4]

                added_df = pd.DataFrame(data=added_data, columns=['currency', 'time'] + price_cols)

                sub_df = sub_df[['currency', 'time'] + price_cols]

                sub_df = pd.concat([sub_df, added_df])

                # print("sub_df.columns = ")
                # print(sub_df.columns)
                # print("added_df.columns = ")
                # print(added_df.columns)
            else:
                sub_df = sub_df[['currency', 'time'] + price_cols]

        else:
            sub_df = sub_df[['currency', 'time'] + price_cols]

            #     print("New head:")
        #     display(sub_df.iloc[0:5])

        #     print("New tail:")
        #     display(sub_df.iloc[-5:])

        new_sub_dfs += [sub_df]

    new_data_df = pd.concat(new_sub_dfs)

    new_data_df.reset_index(inplace=True)
    new_data_df = new_data_df.drop(columns=['index'])

    # print("new_data_df.columns = ")
    # print(new_data_df.columns)
    return new_data_df


def start_do_trading():

    print("")
    print("")
    print("###########################################")
    print("start do trading!")
    #print("Child process starts")

    is_gege_server = False


    data_source = 2 if is_crypto else 1

    #data_source = 2

    is_real_time_trading = True
    #is_weekend = False

    is_real_time_trading_5min = True
    #is_weekend_5min = False

    is_do_portfolio_trading = False

    if is_gege_server:
        root_folder = "/home/min/forex/formal_trading"
    else:
        # if is_real_time_trading:
        #     root_folder = "C:\\JCForex_prod"
        # else:
        root_folder1 = "C:\\JCForex_prod"

        #root_folder = "C:\\Users\\admin\\Desktop\\old data\\JCForex_prod" if data_source == 1 else "C:\\Uesrs\\admin\\JCForex_prod2"

        root_folder = "C:\\Users\\admin\\JCForex_prod" if data_source == 1 else "C:\\Users\\admin\\JCForex_prod2"  #2

        #root_folder = "C:\\JCForex_prod2"


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

    currency_file = os.path.join(root_folder, "currency_instrument.csv") if not is_crypto else os.path.join(root_folder, "crypto.csv")

    currency_df = pd.read_csv(currency_file)



    #currencies_to_run = ['GBPUSD', 'EURGBP', 'USDCAD', 'CADCHF', 'NZDJPY', 'CADJPY', 'EURCHF', 'EURCAD'] #'AUDCHF', 'EURAUD', 'GBPAUD', 'NZDCAD', 'NZDUSD'
    #currencies_to_run = ['NZDUSD', 'AUDUSD','AUDCAD','AUDCHF','NZDCAD','NZDCHF', 'GBPNZD']
    #currencies_to_run = ['NZDUSD', 'AUDCAD', 'EURUSD', 'NZDCAD', 'NZDcurrencies_toCHF']

    #currencies_to_run = ['GBPUSD', 'EURGBP', 'USDCAD', 'CADCHF', 'NZDJPY', 'CADJPY', 'EURCHF', 'EURCAD']
    #currencies_to_run = ['GBPJPY', 'GBPNZD', 'USDJPY', 'CADJPY']


    #currencies_to_run = ['USDCHF', 'CHFJPY', 'AUDCHF', 'EURJPY']
    #currencies_to_run = ['EURNZD', 'EURJPY', 'USDCAD',  'CADCHF', 'GBPUSD', 'AUDJPY'] + ['GBPCHF', 'EURCAD', 'USDCHF', 'GBPAUD']  + ['NZDCHF']
    #currencies_to_run = ['EURUSD','GBPUSD','USDJPY','USDCAD','EURGBP','EURJPY','GBPJPY','USDCHF']
    #currencies_to_run = ['NZDCHF']

    #currencies_to_run =  ['EURNZD', 'EURJPY', 'USDCAD',  'CADCHF', 'GBPUSD', 'AUDJPY'] + ['GBPCHF', 'EURCAD', 'USDCHF', 'GBPAUD']  + ['NZDCHF']

    #currencies_to_run = ['USDJPY', 'GBPJPY', 'CADCHF', 'EURJPY']

    #currencies_to_run = ['USDJPY', 'GBPJPY', 'CADJPY', 'CHFJPY', 'AUDUSD', 'EURAUD', 'NZDCHF', 'NZDJPY', 'GBPCHF', 'GBPAUD']
    #currencies_to_run = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOGEUSD']

    #currencies_to_run = ['BTCUSD','ETHUSD','ADAUSD', 'DOGEUSD', 'XRPUSD', 'SOLUSD', 'AVAXUSD', 'LTCUSD']
    currencies_to_run = ['ADAUSD']

    #currencies_to_run = []

    #currencies_to_run = ['XRPUSD']

    #currencies_to_run = ['EURAUD', 'GBPAUD', 'USDCAD', 'GBPUSD'] #['CHFJPY', 'AUDJPY', 'USDCAD', 'NZDUSD']
    raw_currencies = currency_df['instrument'].tolist()



    # currencies_str = ','.join([currency[:3] + '/' + currency[3:] for currency in raw_currencies])
    #
    # print("currencies_str:")
    # print(currencies_str)
    # sys.exit(0)

    # critical_one = 'GBPAUD'
    # critical_idx = 0
    # for i in range(len(raw_currencies)):
    #     if raw_currencies[i] == critical_one:
    #         critical_idx = i
    #         break
    #
    # currencies_to_run = raw_currencies[critical_idx:]


    #currencies_to_run = [currency for currency in raw_currencies if currency not in ['GBPUSD']]

    # currencies_to_notify = ['CADCHF', 'GBPUSD', 'EURJPY', 'EURCAD', 'NZDCHF', 'AUDJPY', 'EURNZD']
    # currencies_to_remove = ['NZDJPY', 'NZDCAD', 'AUDUSD', 'EURUSD', 'NZDUSD', 'AUDCAD', 'GBPNZD', 'GBPAUD', 'EURGBP',
    #                         'GBPCAD', 'GBPCHF'] + ['CHFJPY']

    #currencies_to_remove = ['NZDJPY', 'NZDCAD', 'EURUSD', 'AUDCAD', 'GBPNZD', 'GBPAUD', 'EURGBP',
    #                        'GBPCAD', 'GBPCHF'] + ['CHFJPY'] + ['AUDCHF', 'AUDNZD']

    currencies_to_remove = []

    #good_currencies = ['XRPUSD', 'DOGEUSD']
    good_currencies = []


    #currencies_to_notify = [currency for currency in raw_currencies if currency not in currencies_to_remove]
    currencies_to_notify = good_currencies if len(good_currencies) > 0 else [currency for currency in raw_currencies if currency not in currencies_to_remove]

    print("currencies_to_notify:")
    print(currencies_to_notify)
    print("Num = " + str(len(currencies_to_notify)))


    print("good_currencies:")
    print(good_currencies)


    raw_data_folders = []
    for currency in raw_currencies:
        currency_data_folder = os.path.join(root_folder, currency, 'data')
        raw_data_folders += [currency_data_folder]

    #if currency_to_run != 'all':
    if len(currencies_to_run) > 0:
        currency_df = currency_df[currency_df['instrument'].isin(currencies_to_run)]


    currency_list = currency_df['instrument'].tolist()

    pre_run_currency_list = [currency for currency in good_currencies if currency in currency_list]

    post_run_currency_list = [currency for currency in currency_list if currency not in good_currencies]

    print("pre_run_currency_list:")
    print(pre_run_currency_list)
    print("post_run_currency_list:")
    print(post_run_currency_list)

    currency_list = pre_run_currency_list + post_run_currency_list

    # if len(currency_list) == 0:
    #     currency_list = currencies_to_run

    print("final currency_list:")
    print(currency_list)

    sorted_currency_df = pd.DataFrame({'instrument' : currency_list, 'cid' : list(range(len(currency_list)))})
    currency_df = pd.merge(currency_df, sorted_currency_df, on = ['instrument'], how='inner')
    currency_df = currency_df.sort_values(by = ['cid'])
    currency_df = currency_df.drop(columns = ['cid'])



    ################### Temp Copy Currency data outside ##################
    # print("root_folder: ")
    # print(root_folder)
    # temp_data_folder = os.path.join(root_folder, "all_data")
    # if not os.path.exists(temp_data_folder):
    #     os.makedirs(temp_data_folder)
    # for currency in currency_list:
    #     print("Copy data of " + currency)
    #     file_path = os.path.join(root_folder, currency, "data", currency + ".csv")
    #     file_path2 = os.path.join(root_folder, currency, "data", currency + "_lastRow.csv")
    #     file_path3 = os.path.join(root_folder, currency, "data", currency + "_5min.csv")
    #     out_folder = os.path.join(temp_data_folder, currency, "data")
    #     if not os.path.exists(out_folder):
    #         os.makedirs(out_folder)
    #
    #     print("Copy from " + file_path + " to " + out_folder)
    #     shutil.copy2(file_path, out_folder)
    #     shutil.copy2(file_path2, out_folder)
    #     shutil.copy2(file_path3, out_folder)
    #
    # sys.exit(0)

    # print("root_folder: ") #Never run this, keep alearted, running this will make your data lost
    # print(root_folder)
    # temp_data_folder = os.path.join(root_folder, "all_data")
    # if not os.path.exists(temp_data_folder):
    #     os.makedirs(temp_data_folder)
    # for currency in currency_list:
    #     print("Copy data of " + currency)
    #     file_path = os.path.join(root_folder, currency, "data")
    #
    #     if not os.path.exists(file_path):
    #         os.makedirs(file_path)
    #
    #     out_folder = os.path.join(temp_data_folder, currency, "data")
    #     out_folder_path = os.path.join(out_folder, currency + ".csv")
    #
    #     if not os.path.exists(out_folder):
    #         os.makedirs(out_folder)
    #
    #     print("Copy from " + out_folder_path + " to " + file_path)
    #     shutil.copy2(out_folder_path, file_path)
    #
    # sys.exit(0)



    ######################################################################



    print("currency_df:")
    print(currency_df)



    #sendEmail("Trader process starts", "")

    currency_pairs = []
    for i in range(currency_df.shape[0]):
        row = currency_df.iloc[i]
        currency_pairs += [CurrencyPair(row['instrument'], row['lot_size'], row['exchange_rate'], row['close_position_coefficient'],
                                        row['actual_maxdrawdown'], row['optimal_gradient_num'], row['optimal_gradient_num_execution'], row['decimal'])]

    print("currencies:")
    print([currencyPair.currency for currencyPair in currency_pairs])
    #sys.exit(0)

    # currencies = list(currency_df['currency'])

    # currencies = ['CADJPY']


    currency_folders = []
    data_folders = []
    chart_folders = []
    simple_chart_folders = []
    log_files = []
    data_files = []

    if read_5min_data:
        data_files_5min = []

    trade_files = []
    performance_files = []

    email_message_files = []

    selected_currencies = [] #currencies_to_notify #['CADCHF', 'GBPUSD', 'EURJPY', 'EURCAD', 'NZDCHF', 'AUDJPY', 'EURNZD']



    #chart_folder_name = "short_macd_strategy_3gradients_close"

    #chart_folder_name = "3gradients_entry_1gradient_exit"

    #chart_folder_name = "3gradients_entry_3gradients_exit_shortmacd_exit"

    #chart_folder_name = "8gradients_entry_8gradients_exit"

    #general_chart_folder_name = "n_gradients_entry_n_gradients_exit_execution_xpctDrawDown"

    current_date = "_0424_dataTest"

    general_chart_folder_name = "n_gradients_entry_n_gradients_exit"

    if do_smart_execution:
        general_chart_folder_name += "_execution"

    if read_5min_data and use_5min_in_smart_execution:
        general_chart_folder_name += "_5min"

    if do_reentry:
        general_chart_folder_name += "_reentry"

    if use_slow_macd:
        general_chart_folder_name += "_slowMACD"
    else:
        general_chart_folder_name += "_fastMACD"

    if printed_figure_num == -1:
        general_chart_folder_name += "_allPics"

    general_chart_folder_name += current_date

    #general_chart_folder_name += "_regression"

    #chart_folder_name = "3gradients_entry_3gradients_exit"

    #chart_folder_name = "cross_entry_cross_exit"
    #chart_folder_name = "4gradients_entry_4gradients_or_cross_exit"



    chart_folder_names = []
    for currency_pair in currency_pairs:

        drawdown = currency_pair.actual_maxdrawdown * 100
        #chart_folder_name = str(currency_pair.optimal_gradient_num) + "gradients_entry_" + str(currency_pair.optimal_gradient_num) + "gradients_exit_execution_" + str(drawdown) + "pctDrawDown"

        gradient_num_str = str(currency_pair.optimal_gradient_num_execution) if do_smart_execution else str(currency_pair.optimal_gradient_num)
        chart_folder_name = gradient_num_str + \
                            "gradients_entry_" + gradient_num_str + "gradients_exit" + ("_" + str(drawdown) + "pctDrawDown" if do_smart_execution else "")

        if read_5min_data and use_5min_in_smart_execution:
            chart_folder_name += "_5min"

        if do_reentry:
            chart_folder_name += "_reentry"

        if use_slow_macd:
            chart_folder_name += "_slowMACD"
        else:
            chart_folder_name += "_fastMACD"

        if printed_figure_num == -1:
            chart_folder_name += "_allPics"

        chart_folder_name += current_date

        #chart_folder_name += "_regression"


        chart_folder_names += [chart_folder_name]

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

        #
        #data_file = os.path.join(data_folder, currency + ".csv")
        #data_file_5min = os.path.join(data_folder, currency + "_5min.csv")
        #

        data_file = os.path.join(currency_data_folder, currency + ".csv")

        if read_5min_data:
            data_file_5min = os.path.join(currency_data_folder, currency + "_5min.csv")

        #trade_file = os.path.join(currency_folder, currency + "_all_trades_" + str(profit_loss_ratio) + ".csv")

        trade_file = os.path.join(currency_folder, currency + "_" + chart_folder_name + "_all_trades.csv")
        #performance_file = os.path.join(currency_folder, currency + "_performance_" + str(profit_loss_ratio) + ".csv")

        performance_file = os.path.join(currency_folder, currency + "_" + chart_folder_name + "_performance.csv")

        #email_message_file = os.path.join(currency_folder, currency + "_emails.txt")

        email_message_file = os.path.join(currency_folder, currency + "_" + chart_folder_name + "_emails.txt")

        #print("Fuck performance_file " + performance_file)

        currency_folders += [currency_folder]
        data_folders += [data_folder]
        chart_folders += [chart_folder]
        simple_chart_folders += [simple_chart_folder]
        log_files += [log_file]
        data_files += [data_file]
        if read_5min_data:
            data_files_5min += [data_file_5min]
        trade_files += [trade_file]
        performance_files += [performance_file]

        email_message_files += [email_message_file]

    currency_traders = []

    is_new_data_received = [False] * len(currency_pairs)
    is_traded_first_time = [False] * len(currency_pairs)
    trial_numbers = [0] * len(currency_pairs)

    waiting_next_time = None

    is_all_received = False

    maximum_trial_number = 100



    ################

    close_prices = []

    currencies = []
    fx_currencies = []
    fx_raw = []
    reciprocal = []
    fx = []

    #raw_currencies = [raw_currency for raw_currency in raw_currencies if raw_currency[0:-3] in ['USD','EUR','GBP','CHF','CAD','JPY','AUD','NZD']]

    if not is_crypto:
        for i in range(len(raw_currencies)):

            currency = raw_currencies[i]

            # if currency == 'GBPUSD':
            #     continue

            data_folder = raw_data_folders[i]

            data_file = os.path.join(data_folder, currency + "_lastRow.csv")

            if not os.path.exists(data_file):
                data_file = os.path.join(data_folder, currency + ".csv")


            print("Read: " + data_file)

            df = pd.read_csv(data_file)
            close_prices += [float(df.iloc[-1]['close'])]

    for i in range(len(currency_list)):

        currency = currency_list[i]

        #print("Processing currency " + currency)

        currencies += [currency]

        main_currency = currency[-3:]

        use_reciprocal = False

        if main_currency in ['EUR', 'GBP', 'AUD', 'NZD']:
            fx_currency = main_currency + 'USD'
        elif main_currency != 'USD':
            use_reciprocal = True
            fx_currency = 'USD' + main_currency
        else:
            fx_currencies += ['USD']
            reciprocal += [False]
            fx_raw += [1]
            fx += [1]
            continue


        fx_currencies += [fx_currency]
        reciprocal += [use_reciprocal]

        for j in range(len(raw_currencies)):

            if fx_currency == raw_currencies[j]:

                target_fx = close_prices[j]

                fx_raw += [target_fx]

                if use_reciprocal:
                    target_fx = 1.0 / target_fx

                fx += [target_fx]

                print("Found target currency " + fx_currency)
                break

        print("")

    print("currencies = " + str(len(currencies)))
    print("fx_currencies = " + str(len(fx_currencies)))
    print("fx_raw = " + str(len(fx_raw)))
    print("reciprocal = " + str(len(reciprocal)))
    print("fx = " + str(len(fx)))

    final_summary_data = pd.DataFrame({'currency' : currencies, 'fx_currency': fx_currencies, 'raw_fx' : fx_raw, 'reciprocal' : reciprocal, 'fx' : fx})

    print("final_summary_data:")
    print(final_summary_data)

    #sys.exit(0)




    ##############



    i = 0
    for currency_pair, data_folder, chart_folder, simple_chart_folder, log_file, data_file, trade_file, performance_file, usdfx, email_message_file in list(
            zip(currency_pairs, data_folders, chart_folders, simple_chart_folders, log_files, data_files, trade_files, performance_files, fx, email_message_files)):

        currency = currency_pair.currency
        lot_size = currency_pair.lot_size
        exchange_rate = currency_pair.exchange_rate
        coefficient = currency_pair.coefficient
        actual_maxdrawdown = currency_pair.actual_maxdrawdown
        optimal_gradient_num = currency_pair.optimal_gradient_num if not do_smart_execution else currency_pair.optimal_gradient_num_execution
        decimal = currency_pair.decimal

        data_file_5min = None
        if read_5min_data:
            data_file_5min = data_files_5min[i]
            i += 1

        #print("optimal_gradient_num = " + str(optimal_gradient_num))

        #print("Here performance_file = " + performance_file)
        currency_trader = CurrencyTrader(threading.Condition(), currency, lot_size, exchange_rate, coefficient, actual_maxdrawdown, optimal_gradient_num, data_folder,
                                         chart_folder, simple_chart_folder, log_file, data_file, trade_file, performance_file, usdfx,
                                         email_message_file, currency in currencies_to_notify, data_file_5min if read_5min_data else None, decimal)
        currency_trader.daemon = True

        currency_traders += [currency_trader]

    print("data_folders:")
    print(data_folders)


    is_do_trading = True

    running_round = 0
    waiting_round = 0

    if is_do_trading:
        while not is_all_received:

            if running_round > 0:
                print("running_round = " + str(running_round))

                now = datetime.now()
                print("now = " + str(now))
                print("waiting_time = " + str(waiting_next_time))
                if now < waiting_next_time:
                    seconds_remaining = (waiting_next_time - now).seconds
                    sleep_seconds = 5
                    while seconds_remaining > 0:
                        actual_sleep_seconds = seconds_remaining if seconds_remaining < sleep_seconds else sleep_seconds
                        time.sleep(actual_sleep_seconds)
                        now = datetime.now()

                        seconds_remaining = (waiting_next_time - now).seconds if now < waiting_next_time else 0
                        print("seconds_remaining = " + str(seconds_remaining))


            is_all_received = True
            running_round += 1

            for i in range(len(currency_traders)):
                if not is_new_data_received[i]:
                    currency_trader = currency_traders[i]

                    data_folder = data_folders[i]

                    currency = currency_trader.currency

                    print_prefix = "[Currency " + currency + "] "

                    print("Query initial for currency pair " + currency)


                    data_file = os.path.join(data_folder, currency + ".csv")
                    data_file_5min = os.path.join(data_folder, currency + "_5min.csv")
                    print("data_file:")
                    print(data_file)

                    data_df = None

                    if os.path.exists(data_file):

                        data_df = pd.read_csv(data_file)
                        #data_df100 = data_df100.iloc[0:-20]

                        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))


                        data_df = data_df[['currency', 'time', 'open', 'high', 'low', 'close']]

                        #data_df = data_df[data_df['time'] <= datetime(2025, 4, 22, 8, 0, 0)]

                        if use_short_data_for_prod:
                            data_df = data_df[data_df['time'] >= datetime(2023, 11, 30, 2, 0, 0)]
                            data_df.reset_index(inplace=True)
                            data_df = data_df.drop(columns=['index'])



                        last_time = data_df.iloc[-1]['time']
                        print("last_time = " + str(last_time))
                        last_timestamp = int(datetime.timestamp(last_time)) #- 28800
                        # next_timestamp = last_timestamp + 3600

                        print("Here last time = " + str(last_time))
                        print("last_timestamp = " + str(last_timestamp))
                        # time.sleep(15)

                        if is_real_time_trading:

                            if data_source == 1:
                                incremental_data_df = get_bar_data2(currency, bar_number=initial_bar_number)
                            else:
                                incremental_data_df = get_bar_data2(currency, bar_number=initial_bar_number)



                            if incremental_data_df.iloc[0]['time'] > last_time:
                                print("last_time = " + str(last_time) + ", but queried starting time is even after that" + str(incremental_data_df.iloc[0]['time']), file = sys.stderr)

                            #if is_weekend:
                            incremental_data_df = incremental_data_df[incremental_data_df['time'] > last_time]
                            # else:
                            #     incremental_data_df = incremental_data_df[incremental_data_df['time'] > last_time].iloc[0:-1]


                        if is_real_time_trading and incremental_data_df.shape[0] > 0:


                            data_df = pd.concat([data_df, incremental_data_df])


                            data_df.reset_index(inplace=True)
                            data_df = data_df.drop(columns=['index'])

                    else:
                        print("Currency file does not exit, query initial data from web")

                        # if data_source == 1:
                        #     temp_data_df = get_bar_data2(currency, bar_number=2, is_convert_to_time=False)
                        # else:
                        #     temp_data_df = get_bar_data2(currency, bar_number=2, is_convert_to_time=False)
                        #
                        # last_timestamp = temp_data_df.iloc[-1]['time']
                        #
                        #
                        # print("last_timestamp: " + str(last_timestamp))
                        # #print(datetime.fromtimestamp(last_timestamp))
                        # #start_timestamp = last_timestamp - 3600 * (initial_bar_number-1)
                        # print("initial_bar_num = " + str(initial_bar_number))
                        # start_timestamp = last_timestamp - timedelta(hours = initial_bar_number - 1)
                        #
                        #
                        # print("last_timestamp = " + str(last_timestamp))
                        # print("start_timestamp = " + str(start_timestamp))

                        if data_source == 1:
                            data_df = get_bar_data2(currency, bar_number=initial_bar_number)
                        else:
                            data_df = get_bar_data2(currency, bar_number=initial_bar_number)

                        data_df = data_df.iloc[:-1]


                    if read_5min_data:

                        print("Read 5 min data")

                        if os.path.exists(data_file_5min):

                            data_df_5min = pd.read_csv(data_file_5min)

                            data_df_5min['time'] = data_df_5min['time'].apply(lambda x: preprocess_time(x))

                            data_df_5min = data_df_5min[['currency', 'time', 'open', 'high', 'low', 'close']]


                            last_time = data_df_5min.iloc[-1]['time']
                            print("last_time = " + str(last_time))
                            last_timestamp = int(datetime.timestamp(last_time)) #- 28800
                            # next_timestamp = last_timestamp + 3600

                            print("Here last time = " + str(last_time))
                            print("last_timestamp = " + str(last_timestamp))
                            # time.sleep(15)

                            if is_real_time_trading_5min:

                                if data_source == 1:
                                    incremental_data_df_5min = get_bar_data2(currency, bar_number=initial_bar_number_5min, interval='5min')
                                else:
                                    incremental_data_df_5min = get_bar_data2(currency, bar_number=initial_bar_number_5min, interval='5min')



                                if incremental_data_df_5min.iloc[0]['time'] > last_time:
                                    print("5min bar: last_time = " + str(last_time) + ", but queried starting time is even after that" + str(incremental_data_df_5min.iloc[0]['time']), file = sys.stderr)

                                #if is_weekend_5min:
                                #    incremental_data_df_5min = incremental_data_df_5min[incremental_data_df_5min['time'] > last_time]
                                #else:
                                incremental_data_df_5min = incremental_data_df_5min[incremental_data_df_5min['time'] > last_time].iloc[0:-1]


                            if is_real_time_trading_5min and incremental_data_df_5min.shape[0] > 0:


                                data_df_5min = pd.concat([data_df_5min, incremental_data_df_5min])


                                data_df_5min.reset_index(inplace=True)
                                data_df_5min = data_df_5min.drop(columns=['index'])

                                # for col in ['open', 'high', 'low', 'close']:
                                #     data_df_5min[col] = data_df_5min[col].apply(lambda x: round(x, currency_trader.decimal))

                        else:
                            print("Currency file does not exit, query initial data from web")

                            # if data_source == 1:
                            #     temp_data_df = get_bar_data2(currency, bar_number=2, interval='5min',  is_convert_to_time=False)
                            # else:
                            #     temp_data_df = get_bar_data2(currency, bar_number=2, interval='5min',  is_convert_to_time=False)
                            #
                            # last_timestamp = temp_data_df.iloc[-1]['time']
                            #
                            #
                            # print("last_timestamp: " + str(last_timestamp))
                            # #print(datetime.fromtimestamp(last_timestamp))
                            # #start_timestamp = last_timestamp - 3600 * (initial_bar_number-1)
                            # print("initial_bar_num = " + str(initial_bar_number))
                            # start_timestamp = last_timestamp - timedelta(hours = initial_bar_number - 1)
                            #
                            #
                            # print("last_timestamp = " + str(last_timestamp))
                            # print("start_timestamp = " + str(start_timestamp))

                            if data_source == 1:
                                data_df_5min = get_bar_data2(currency, bar_number=initial_bar_number_5min, interval='5min')
                            else:
                                data_df_5min = get_bar_data2(currency, bar_number=initial_bar_number_5min, interval='5min')

                            data_df_5min = data_df_5min.iloc[:-1]

                            # for col in ['open', 'high', 'low', 'close']:
                            #     data_df_5min[col] = data_df_5min[col].apply(lambda x: round(x, currency_trader.decimal))

                    # print("Initial data_df:")
                    # print(data_df.iloc[-20:])


                    if data_source == 2:
                        print("preprocess data")
                        data_df = preprocess_data(data_df)  #Preprocess data to de-noise bars at weekends
                        print("preprocess finished")
                        # print("preprocessed data:")
                        # print(data_df.iloc[1500:1510])

                    #if is_real_time_trading and not is_weekend:
                    if is_real_time_trading:

                        if data_df is not None and data_df.shape[0] > 1:
                            #last_time = data_df.iloc[-1]['time']
                            last_time = data_df.iloc[-2]['time']
                        else:
                            last_time = None

                        if read_5min_data:

                            print("Now data_df_5min..........:")
                            print(data_df_5min.iloc[-5:])

                            if data_df_5min is not None and data_df_5min.shape[0] > 0:
                                last_time_5min = data_df_5min.iloc[-1]['time']
                                print("Here last_time_5min = " + str(last_time_5min))
                            else:
                                last_time_5min = None

                        if last_time is not None and ((not read_5min_data) or last_time_5min is not None):
                            delta = datetime.now() - last_time

                            # if read_5min_data:
                            #     delta_5min = datetime.now() - last_time_5min

                            print("last_time = " + str(last_time))
                            print("now = " + str(datetime.now()))

                            if read_5min_data:
                                print('last_time_5min = ' + str(last_time_5min))



                            if (delta is not None and delta.seconds > 0 and delta.seconds < 7200 and delta.days == 0): #7200
                                    #and (
                                    #(not read_5min_data) or (delta_5min.seconds > 0 and delta_5min.seconds < 1200 and delta_5min.days == 0)

                                print("Received up-to-date data for currency pair " + currency)

                                is_new_data_received[i] = True

                                final_data_df = data_df.iloc[0:-1]
                                if read_5min_data:
                                    currency_trader.feed_data(final_data_df, data_df_5min)
                                else:
                                    currency_trader.feed_data(final_data_df)

                                currency_trader.trade()
                            else:

                                print("Not received finalized data for " + currency + ", wait 1 minute to try again")

                                if read_5min_data:
                                    currency_trader.feed_data(data_df, data_df_5min)
                                else:
                                    currency_trader.feed_data(data_df)

                                currency_trader.trade()


                                if trial_numbers[i] <= maximum_trial_number:
                                    is_all_received = False
                                    print("Not received data update for " + currency + ", will try again")
                                    trial_numbers[i] += 1

                                    if waiting_round < running_round:
                                        waiting_next_time = data_df.iloc[-1]['time'] + timedelta(seconds = 3600 + running_round * 60 + 10)  #-1
                                        print("waiting_next_time = " + str(waiting_next_time))
                                        waiting_round += 1
                                        print("running_round = " + str(running_round) + ", waiting_round = " + str(waiting_round))

                                else:
                                    print("Reached maximum number of trials for " + currency + ", give up")
                    else:

                        if data_df is not None:

                            is_new_data_received[i] = True

                            print("Start trading")
                            if read_5min_data:
                                currency_trader.feed_data(data_df, data_df_5min)
                            else:
                                currency_trader.feed_data(data_df)


                            currency_trader.trade()

        for i in range(len(currency_traders)):
            if is_new_data_received[i]:
                currency_trader = currency_traders[i]
                currency_trader.post_processing()

        #sendEmail("Trader process ends", "")

        print("Finished trading *********************************")


        print("Collecting Results....")

        perf_dfs = []
        trade_dfs = []
        i = 0
        for currency in currency_list:
            #perf_file = os.path.join(root_folder, currency, currency + "_performance_" + str(profit_loss_ratio) + ".csv")
            chart_folder_name = chart_folder_names[i]
            i += 1
            perf_file = os.path.join(root_folder, currency, currency + "_" + chart_folder_name + "_performance.csv")
            perf_dfs += [pd.read_csv(perf_file)]

            #trade_file = os.path.join(root_folder, currency, currency + "_all_trades_" + str(profit_loss_ratio) + ".csv")
            trade_file = os.path.join(root_folder, currency, currency + "_" + chart_folder_name + "_all_trades.csv")
            trade_df = pd.read_csv(trade_file)
            trade_dfs += [trade_df]

        perf_df = pd.concat(perf_dfs)
        trade_df = pd.concat(trade_dfs)
        trade_df = trade_df.sort_values(by = ['entry_time'])

        print("Final Performance Result:")
        perf_df.reset_index(inplace = True)
        perf_df = perf_df.drop(columns = ['index'])
        print(perf_df)

        print("")
        print("Selected currencies Performance Result:")
        selected_perf_df = perf_df[perf_df['Currency'].isin(selected_currencies)]
        print(selected_perf_df)

        perf_df.to_csv(os.path.join(root_folder, general_chart_folder_name + ".csv"), index = False)

        des_pnl_folder = os.path.join(root_folder, 'all_pnl_' + general_chart_folder_name)
        if not os.path.exists(des_pnl_folder):
            os.makedirs(des_pnl_folder)


        old_pnl_files = os.listdir(des_pnl_folder)
        for file in old_pnl_files:
            target_file = os.path.join(des_pnl_folder, file)
            if os.path.isdir(target_file):
                shutil.rmtree(target_file)
            else:
                os.remove(target_file)

        des_selected_pnl_folder = os.path.join(des_pnl_folder, 'selected')
        if not os.path.exists(des_selected_pnl_folder):
            os.makedirs(des_selected_pnl_folder)



        des_bar_folder = os.path.join(root_folder, 'all_bars_' + general_chart_folder_name)
        if not os.path.exists(des_bar_folder):
            os.makedirs(des_bar_folder)



        old_bar_files = os.listdir(des_bar_folder)
        for file in old_bar_files:
            target_file = os.path.join(des_bar_folder, file)
            if os.path.isdir(target_file):
                shutil.rmtree(target_file)
            else:
                os.remove(target_file)

        des_selected_bar_folder = os.path.join(des_bar_folder, 'selected')
        print("des_selected_bar_folder")
        print(des_selected_bar_folder)
        if not os.path.exists(des_selected_bar_folder):
            os.makedirs(des_selected_bar_folder)



        print("Copying bar charts and pnl charts...")
        #trade_df = trade_df.drop(columns = ['id', 'pnl', 'cum_pnl', 'reverse_pnl', 'cum_reverse_pnl'])

        trade_df = trade_df.drop(columns=['trade_id', 'long_trade_id', 'short_trade_id', 'cum_pnl'])
        trade_df.to_csv(os.path.join(des_pnl_folder, "all_trades.csv"), index = False)

        i = 0
        for currency in currency_list:

            chart_folder_name = chart_folder_names[i]
            i += 1

            #print("currency = " + str(currency))
            pic_path = os.path.join(root_folder, currency, chart_folder_name, currency + '_pnl.png')
            if os.path.exists(pic_path):
                shutil.copy2(pic_path, des_pnl_folder)

                if currency in selected_currencies:
                    shutil.copy2(pic_path, des_selected_pnl_folder)


            currency_chart_folder = os.path.join(root_folder, currency, chart_folder_name)
            chart_files = os.listdir(currency_chart_folder)
            for chart_file in chart_files:
                if 'pnl' not in chart_file:

                    #print(os.path.join(currency_chart_folder, chart_file))
                    #print(des_bar_folder)

                    shutil.copy2(os.path.join(currency_chart_folder, chart_file), des_bar_folder)

                    #print("des_bar_folder:")
                    #print(des_bar_folder)

                    if currency in selected_currencies:
                        # print("currency_chart_folder:")
                        # print(currency_chart_folder)
                        # print("source file:")
                        # print(os.path.join(currency_chart_folder, chart_file))
                        # print("des folder:")
                        # print(des_selected_bar_folder)

                        source_exists = os.path.exists(os.path.join(currency_chart_folder, chart_file))
                        des_exists = os.path.exists(des_selected_bar_folder)

                        # print("source_exist = " + str(source_exists))
                        # print("des_exist = " + str(des_exists))

                        shutil.copy2(os.path.join(currency_chart_folder, chart_file), des_selected_bar_folder)


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