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
parser.add_option("-a", "--alternative", dest="alternative", default = "n",
                 help="Use alternative account")

(options, args) = parser.parse_args()

currency_to_run = options.currency_pair
alternative = options.alternative

print("currency_to_run = " + currency_to_run)
print("alternative = " + alternative)

global_log_file = "algo_log.txt"

#log_msg("currency_to_run = " + currency_to_run)

if currency_to_run != 'all':
    global_log_file = currency_to_run + "_algo_log.txt"

root_folder = os.getenv("CRYPTO_PROD")
if alternative == 'y':
    root_folder += "_alternative"

if alternative != 'y':
    alternative_root_folder = root_folder + '_alternative'

#if currency_to_run != "all":
#    root_folder += "_" + currency_to_run

if not os.path.exists(root_folder):
    os.makedirs(root_folder)


global_log_path = os.path.join(root_folder, global_log_file)
global_log_fd = open(global_log_path, "a")


def log_msg(msg):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # current_time = (datetime.now() + timedelta(seconds = 28800)).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(msg, pd.DataFrame):
        print('[' + current_time + ']  \n' + str(msg), file=global_log_fd)
    else:
        print('[' + current_time + ']  ' + str(msg), file=global_log_fd)

    global_log_fd.flush()

    if print_to_console:
        if isinstance(msg, pd.DataFrame):
            print('[' + current_time + ']  \n' + str(msg))
        else:
            print('[' + current_time + ']  ' + str(msg))





app_id = "168180645499516"

use_dynamic_TP = True

use_short_data_for_prod = False #This should always be FALSE on my own machine!!!

is_run_individual_good_ones = False
is_run_aggregated_good_ones = False

profit_loss_ratio = 1

read_5min_data = False #True

use_coinbase_data_source = False

if use_dynamic_TP:
    profit_loss_ratio = 10



client = None
while True:
    try:
        if do_real_money_trading:
            api_key, api_secret = get_api_keys(is_alternative=True if alternative == 'y' else False)
            client = RESTClient(api_key=api_key,
                                api_secret=api_secret)

        td = TDClient(apikey=get_twelvedata_api_keys())
        break
    except Exception as e:

        emsg = str(e)
        log_msg("Exception: " + emsg)

        if 'HTTPSConnection' in emsg:
            log_msg("Probably network connection exception, trying again after 10 seconds.")
            time.sleep(10)
        else:
            raise



class CurrencyPair:

    def __init__(self, currency, lot_size, exchange_rate, coefficient, actual_maxdrawdown, optimal_gradient_num, optimal_gradient_num_execution, decimal, reverse_strategy,
                 use_slow_macd, use_guppy_filter, use_guppy_filter_for_exit, guppy_force_out, use_rsi_to_exit, do_stop_loss, reentry_after_stop_loss, also_filter_too_late, use_guppy_condition,
                 init_entry_value, coinbase_decimal):
        self.currency = currency
        self.lot_size = lot_size
        self.exchange_rate = exchange_rate
        self.coefficient = coefficient
        self.actual_maxdrawdown = actual_maxdrawdown
        self.optimal_gradient_num = optimal_gradient_num
        self.optimal_gradient_num_execution = optimal_gradient_num_execution
        self.decimal = decimal
        self.reverse_strategy = True if reverse_strategy == 1 else False
        self.use_slow_macd = True if use_slow_macd == 1 else False
        self.use_guppy_filter = True if use_guppy_filter == 1 else False
        self.use_guppy_filter_for_exit = True if use_guppy_filter_for_exit == 1 else False
        self.guppy_force_out = True if guppy_force_out == 1 else False
        self.use_rsi_to_exit = True if use_rsi_to_exit == 1 else False
        self.do_stop_loss = True if do_stop_loss == 1 else False
        self.reentry_after_stop_loss = True if reentry_after_stop_loss == 1 else False
        self.also_filter_too_late = True if also_filter_too_late == 1 else False
        self.use_guppy_condition = True if use_guppy_condition == 1 else False
        self.init_entry_value = init_entry_value
        self.coinbase_decimal = coinbase_decimal

        print("slow_macd = " + str(self.use_slow_macd))
        print("use_guppy_filter = " + str(self.use_guppy_filter))
        print("use_guppy_filter_for_exit = " + str(self.use_guppy_filter_for_exit))
        print("guppy_force_out = " + str(self.guppy_force_out))
        #sys.exit(0)

def convert_to_time(timestamp):
   #return datetime.fromtimestamp(timestamp+28800)
    return datetime.fromtimestamp(timestamp)

def get_close_price(currency):

    global td

    while True:
        try:
            ts = td.price(symbol = currency[:-3] + '/' + currency[-3:])
            close_price = float(ts.as_json()['price'])
            break
        except Exception as e:

            emsg = str(e)
            log_msg("Exception: " + emsg)

            if 'API credits' in emsg:
                wait_seconds = 80
                log_msg("Running out of API credits, waiting " + str(wait_seconds) + " seconds to proceed")
                time.sleep(wait_seconds)
            else:
                raise



    return close_price



def get_bar_data2(currency, bar_number=240, interval = "1h", end_date = None, start_timestamp=-1, is_convert_to_time = True):
    # Initialize client - apikey parameter is requiered
    global td

    global client

    if client is None:
        api_key, api_secret = get_api_keys(is_alternative=True if alternative == 'y' else False)
        client = RESTClient(api_key=api_key,
                            api_secret=api_secret)

    log_msg("")
    log_msg("Now = " + str(datetime.now()))
    log_msg("initial_bar_number = " + str(initial_bar_number))
    # Construct the necessary time series


    if use_coinbase_data_source:

        coinbase_bar_num = min(300, initial_bar_number)

        start_time = datetime.now() - timedelta(hours=coinbase_bar_num)
        end_time = datetime.now()
        print("end_time = " + str(end_time.isoformat()))

        print("final start_time = " + str(start_time))
        print("final end_time = " + str(end_time))

        start_time = int(start_time.timestamp())
        end_time = int(end_time.timestamp())


        while True:
            try:

                print("product_id = " + str(currency[:-3]+'-USDC'))
                print("start_time = " + str(start_time))
                print("end_time = " + str(end_time))
                coinbase_interval = 'ONE_HOUR' if interval == '1h' else 'FIVE_MINUTE'
                response = client.get_candles(product_id=currency[:-3]+'-USDC', start=start_time, end=end_time, granularity=coinbase_interval)

                break

            except Exception as e:
                print(f"Order failed: {e}")

                print("Enter exception processing here:")
                emsg = str(e)
                log_msg("Exception: " + emsg)

                if 'API credits' in emsg or 'Connection aborted' in emsg:
                    wait_seconds = 80
                    log_msg("Running out of API credits, waiting " + str(wait_seconds) + " seconds to proceed")
                    time.sleep(wait_seconds)
                else:
                    raise



        candles = response['candles']

        columns = ['datetime', 'open', 'high', 'low', 'close']
        data = []

        for candle in candles:
            data += [[datetime.fromtimestamp(int(candle['start'])), float(candle['open']), float(candle['high']),
                      float(candle['low']), float(candle['close'])]]

        data_df = pd.DataFrame(data=data, columns=columns)

    else:

        while True:
            try:
                ts = td.time_series(
                    symbol=currency[:-3] + '/' + currency[-3:],
                    interval=interval,
                    outputsize=bar_number, #initial_bar_number
                    end_date=end_date,
                    timezone="Asia/Singapore",
                )

                data_df = ts.as_pandas()

                break
            except Exception as e:
                print("Enter exception processing here:")
                emsg = str(e)
                log_msg("Exception: " + emsg)

                if 'API credits' in emsg or 'Connection aborted' in emsg:
                    wait_seconds = 80
                    log_msg("Running out of API credits, waiting " + str(wait_seconds) + " seconds to proceed")
                    time.sleep(wait_seconds)
                else:
                    raise



    data_df = data_df.iloc[::-1]

    data_df.reset_index(inplace=True)

    data_df = data_df.rename(columns = {'datetime' : 'time'})

    data_df['currency'] = currency

    data_df = data_df[['time', 'currency', 'open', 'high', 'low', 'close']]

    log_msg("Row number = " + str(data_df.shape[0]) + " &&")
    #
    log_msg("here printing")

    #log_msg(data_df.iloc[0:20])

    log_msg(data_df.iloc[-20:])

    return data_df





def get_bar_data(currency, bar_number=240, start_timestamp=-1, is_convert_to_time=True):
    global app_id

    query = "http://api.forexfeed.net/data/[app_id]/n-[bar_number]/f-csv/i-3600/s-[currency]"

    query = query.replace("[app_id]", app_id).replace("[bar_number]", str(bar_number)).replace("[currency]", currency)

    # if start_timestamp != -1:
    #     query = query + "/st-" + str(start_timestamp)

    log_msg("query:")
    log_msg(query)

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

        # log_msg("final data_df:")
        # log_msg(data_df)

        log_msg("data number: " + str(data_df.shape[0]))

        return data_df

    return None




def preprocess_data(data_df):
    #data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))

    data_df['prev_time'] = data_df['time'].shift(1)

    data_df['time_delta'] = data_df['time'] - data_df['prev_time']

    data_df['delta_seconds'] = data_df['time_delta'].apply(lambda x: x.seconds).fillna(0).astype(int)
    data_df['delta_days'] = data_df['time_delta'].apply(lambda x: x.days).fillna(0).astype(int)

    data_df['total_seconds'] = data_df['delta_days'] * 24 * 3600 + data_df['delta_seconds']

    #log_msg(type(data_df.iloc[-1]['time_delta']))
    #log_msg(data_df.iloc[-1]['time_delta'].seconds)

    # log_msg("###########")
    # log_msg("Temp data")
    # log_msg(data_df.iloc[1500:1510])
    # log_msg("###########")

    critical_index = list(which(data_df['total_seconds'] > 3600)) + [data_df.shape[0]]

    sub_dfs = []

    log_msg(critical_index)

    log_msg("critical_index length = " + str(len(critical_index)))
    log_msg("")

    start = 0
    for i in range(len(critical_index)):

        # log_msg("i = " + str(i))
        # log_msg("start = " + str(start))
        # log_msg("end = " + str(critical_index[i]))
        sub_df = data_df.iloc[start:critical_index[i]]
        # log_msg("sub_df length = " + str(sub_df.shape[0]))
        # log_msg("")
        start = critical_index[i]

        sub_dfs += [sub_df]

    last_close_price = None
    price_cols = ['open', 'high', 'low', 'close']
    new_sub_dfs = []
    for j in range(len(sub_dfs)):

       # log_msg("j = " + str(j))

        sub_df = sub_dfs[j]

        # log_msg("now sub_df.columns = ")
        # log_msg(sub_df.columns)
        # log_msg("length = " + str(sub_df.shape[0]))

        ########Added Code ##########
        if sub_df.shape[0] < 2:

            continue

        #############################

        #log_msg("sub df size = " + str(sub_df.shape[0]))

        # sub_df.at[sub_df.index[0], 'open'] = 0.0

        currency = sub_df.iloc[0]['currency']
        first_time = sub_df.iloc[0]['time']
        last_time = sub_df.iloc[-1]['time']

        # log_msg("first_time = " + str(first_time))
        # log_msg("last_time = " + str(last_time))
        # log_msg("")

        #     log_msg("Old head:")
        #     display(sub_df.iloc[0:5])

        #     log_msg("Old tail:")
        #     display(sub_df.iloc[-5:])


        if last_close_price is not None:

            # if j == 32:
            #     log_msg("j = " + str(j))
            #
            #     log_msg("Before sub_df:")
            #     log_msg(sub_df)

            if first_time.hour < 5:
                sub_df = sub_df.iloc[1:]

            # if j == 32:
            #     log_msg("After sub_df:")
            #     log_msg(sub_df)
            #
            #     log_msg("")

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

                # log_msg("sub_df.columns = ")
                # log_msg(sub_df.columns)
                # log_msg("added_df.columns = ")
                # log_msg(added_df.columns)
            else:
                sub_df = sub_df[['currency', 'time'] + price_cols]

        else:
            sub_df = sub_df[['currency', 'time'] + price_cols]

            #     log_msg("New head:")
        #     display(sub_df.iloc[0:5])

        #     log_msg("New tail:")
        #     display(sub_df.iloc[-5:])

        new_sub_dfs += [sub_df]

    new_data_df = pd.concat(new_sub_dfs)

    new_data_df.reset_index(inplace=True)
    new_data_df = new_data_df.drop(columns=['index'])

    # log_msg("new_data_df.columns = ")
    # log_msg(new_data_df.columns)
    return new_data_df


def start_do_trading(wakeup = 0):

    global my_log_file

    log_msg("")
    log_msg("")
    log_msg("###########################################")
    log_msg("start do trading!")
    #log_msg("Child process starts")

    is_real_time_trading = True
    #is_weekend = False

    is_real_time_trading_5min = False
    #is_weekend_5min = False

    manual_delay = 7 if is_real_time_trading else 0  #manual_delay = 10  #Darren

    is_do_portfolio_trading = False



    currency_file = os.path.join(root_folder, "currency_instrument.csv") if not is_crypto else os.path.join(root_folder, "crypto_fast.csv")

    currency_df = pd.read_csv(currency_file)

    # print("currency_df:")
    # print(currency_df)
    # sys.exit(0)


    raw_currencies = currency_df['instrument'].tolist()

    currency_close_prices = {}

    currency_coinbase_close_prices = {}

    #currencies_to_run = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'LTCUSD', 'XRPUSD', 'AVAXUSD', 'DOGEUSD'] + ['LINKUSD', 'DOTUSD', 'UNIUSD', 'XTZUSD']
    #currencies_to_run = ['LINKUSD', 'DOTUSD', 'UNIUSD', 'XTZUSD']

    if currency_to_run != 'all':
        #currencies_to_run = [currency_to_run]
        currencies_to_run = currency_to_run.split(',')
    else:
        currencies_to_run = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'LTCUSD', 'XRPUSD', 'AVAXUSD', 'DOGEUSD'] + ['LINKUSD', 'DOTUSD', 'UNIUSD', 'XTZUSD']
        #currencies_to_run = ['DOGEUSD', 'XRPUSD']
        #currencies_to_run = ['AVAXUSD','DOGEUSD', 'XRPUSD']

    print("currencies_to_run:")
    print(currencies_to_run)

    log_msg("wakeup = " + str(wakeup))

    portfolio_id = None

    if wakeup == 1:
        for currency in currencies_to_run:
            log_msg("Get close price for " + currency)
            close_price = get_close_price(currency)
            log_msg("close_price = " + str(close_price))
            currency_close_prices[currency] = close_price

        try:
            if do_real_money_trading:
                coinbase_currencies = []
                for currency in currencies_to_run:
                    coinbase_currency = currency[:-len('USD')] + '-PERP-INTX'
                    coinbase_currencies += [coinbase_currency]
                    log_msg("Get current price for " + coinbase_currency)
                    product = client.get_product(coinbase_currency)
                    coinbase_price = float(product['price'])
                    log_msg("Current price = " + str(coinbase_price))

                    currency_coinbase_close_prices[currency] = coinbase_price
        except Exception as e:
            print("Enter exception processing here:")
            emsg = str(e)
            log_msg("Exception: " + emsg)

            if 'Remote end closed connection' in emsg:
                wait_seconds = 80
                log_msg("Remote end connection closed, waiting " + str(wait_seconds) + " seconds to proceed")
                time.sleep(wait_seconds)
            else:
                raise


    if do_real_money_trading:
        accounts = client.get_accounts()
        account = accounts.accounts[0]
        portfolio_id = str(account['retail_portfolio_id'])

    log_msg("Sleep 1 seconds")
    time.sleep(1)







    currencies_to_remove = []

    #good_currencies = ['XRPUSD', 'DOGEUSD']
    good_currencies = []


    #currencies_to_notify = [currency for currency in raw_currencies if currency not in currencies_to_remove]
    currencies_to_notify = good_currencies if len(good_currencies) > 0 else [currency for currency in raw_currencies if currency not in currencies_to_remove]

    log_msg("currencies_to_notify:")
    log_msg(currencies_to_notify)
    log_msg("Num = " + str(len(currencies_to_notify)))


    log_msg("good_currencies:")
    log_msg(good_currencies)


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

    log_msg("pre_run_currency_list:")
    log_msg(pre_run_currency_list)

    print("good_currencies:")
    print(good_currencies)
    log_msg("post_run_currency_list:")
    log_msg(post_run_currency_list)



    currency_list = pre_run_currency_list + post_run_currency_list

    # if len(currency_list) == 0:
    #     currency_list = currencies_to_run

    log_msg("final currency_list:")
    log_msg(currency_list)

    sorted_currency_df = pd.DataFrame({'instrument' : currency_list, 'cid' : list(range(len(currency_list)))})
    currency_df = pd.merge(currency_df, sorted_currency_df, on = ['instrument'], how='inner')
    currency_df = currency_df.sort_values(by = ['cid'])
    currency_df = currency_df.drop(columns = ['cid'])



    ################### Temp Copy Currency data outside ##################
    # log_msg("root_folder: ")
    # log_msg(root_folder)
    # # temp_data_folder = os.path.join(root_folder, "all_data")
    # # if not os.path.exists(temp_data_folder):
    # #     os.makedirs(temp_data_folder)
    # for currency in currency_list:
    #     log_msg("Copy data of " + currency)
    #     #file_path = os.path.join(root_folder, currency, "data", currency + ".csv")
    #     #file_path2 = os.path.join(root_folder, currency, "data", currency + "_lastRow.csv")
    #     file_path3 = os.path.join(root_folder, currency, "data", currency + "_5min.csv")
    #     out_folder = os.path.join(alternative_root_folder, currency, "data")
    #     if not os.path.exists(out_folder):
    #         os.makedirs(out_folder)
    #
    #     log_msg("Copy from " + file_path3 + " to " + out_folder)
    #     #shutil.copy2(file_path, out_folder)
    #     #shutil.copy2(file_path2, out_folder)
    #     shutil.copy2(file_path3, out_folder)
    #
    # sys.exit(0)



    # log_msg("root_folder: ")
    # log_msg(root_folder)
    # temp_data_folder = os.path.join(root_folder, "all_data")
    # if not os.path.exists(temp_data_folder):
    #     os.makedirs(temp_data_folder)
    # for currency in currency_list:
    #     log_msg("Copy data of " + currency)
    #     file_path = os.path.join(root_folder, currency, "data", currency + ".csv")
    #     file_path2 = os.path.join(root_folder, currency, "data", currency + "_lastRow.csv")
    #     file_path3 = os.path.join(root_folder, currency, "data", currency + "_5min.csv")
    #     out_folder = os.path.join(temp_data_folder, currency, "data")
    #     if not os.path.exists(out_folder):
    #         os.makedirs(out_folder)
    #
    #     log_msg("Copy from " + file_path + " to " + out_folder)
    #     #shutil.copy2(file_path, out_folder)
    #     #shutil.copy2(file_path2, out_folder)
    #     shutil.copy2(file_path3, out_folder)
    #
    # sys.exit(0)

    # log_msg("root_folder: ") #Never run this, keep alearted, running this will make your data lost
    # log_msg(root_folder)
    # temp_data_folder = os.path.join(root_folder, "all_data")
    # if not os.path.exists(temp_data_folder):
    #     os.makedirs(temp_data_folder)
    # for currency in currency_list:
    #     log_msg("Copy data of " + currency)
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
    #     log_msg("Copy from " + out_folder_path + " to " + file_path)
    #     shutil.copy2(out_folder_path, file_path)
    #
    # sys.exit(0)



    ######################################################################



    log_msg("currency_df:")
    log_msg(currency_df)



    #sendEmail("Trader process starts", "")

    currency_pairs = []
    for i in range(currency_df.shape[0]):
        row = currency_df.iloc[i]

        # if row['instrument'] == 'DOGEUSD':
        #     log_msg("row:")
        #     log_msg(row)
        #     sys.exit(0)
        currency_pairs += [CurrencyPair(row['instrument'], row['lot_size'], row['exchange_rate'], row['close_position_coefficient'],
                                        row['actual_maxdrawdown'], row['optimal_gradient_num'], row['optimal_gradient_num_execution'], row['decimal'],
                                        row['reverse_strategy'], row['use_slow_macd'], row['use_guppy_filter'], row['use_guppy_filter_for_exit'], row['guppy_force_out'], row['use_rsi_to_exit'], row['do_stop_loss'],
        row['reentry_after_stop_loss'],row['also_filter_too_late'],row['use_guppy_condition'], row['init_entry_value'], row['coinbase_decimal'])]

    log_msg("currencies:")
    log_msg([currencyPair.currency for currencyPair in currency_pairs])
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
    trade_prod_files = []

    delay_cost_files = []

    performance_files = []

    email_message_files = []

    selected_currencies = [] #currencies_to_notify #['CADCHF', 'GBPUSD', 'EURJPY', 'EURCAD', 'NZDCHF', 'AUDJPY', 'EURNZD']



    #chart_folder_name = "short_macd_strategy_3gradients_close"

    #chart_folder_name = "3gradients_entry_1gradient_exit"

    #chart_folder_name = "3gradients_entry_3gradients_exit_shortmacd_exit"

    #chart_folder_name = "8gradients_entry_8gradients_exit"

    #general_chart_folder_name = "n_gradients_entry_n_gradients_exit_execution_xpctDrawDown"

    #current_date = "_realtime_0523"  #0521
    #current_date = "_final_prodction_0621_noforceOut_execution_withExtra_refactorTest"  #_final_prod  _UATTest

    current_date = "_final_prodction_0621_noforceOut_checkData"
    #current_date = "_final_prodction_mytest"

    general_chart_folder_name = "n_gradients_entry_n_gradients_exit"


    if use_global:

        if do_smart_execution:
            general_chart_folder_name += "_execution"

        if read_5min_data and use_5min_in_smart_execution:
            general_chart_folder_name += "_5min"

        if do_reentry:
            general_chart_folder_name += "_reentry"

        if global_use_slow_macd:
            general_chart_folder_name += "_slowMACD"
        else:
            general_chart_folder_name += "_fastMACD"

        if global_use_guppy_filter:
            general_chart_folder_name += "_guppyFilter"

        if global_use_guppy_filter_for_exit:
            general_chart_folder_name += "_guppyFilterForExit"

        if global_guppy_force_out:
            general_chart_folder_name += "_guppyForceOut"

        if global_use_rsi_to_exit:
            general_chart_folder_name += "_rsiExit"

        if global_also_filter_too_late:
            general_chart_folder_name += "_filterTooLate"

        if global_use_guppy_condition:
            general_chart_folder_name += "_guppyCondition"

        if global_do_stop_loss:
            general_chart_folder_name += "_stopLoss"

        if global_do_stop_loss and not global_reentry_after_stop_loss:
            general_chart_folder_name += "_notReentryAfterSL"

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

        if use_global:

            if global_use_slow_macd:
                chart_folder_name += "_slowMACD"
            else:
                chart_folder_name += "_fastMACD"

            if global_use_guppy_filter:
                chart_folder_name += "_guppyFilter"

            if global_use_guppy_filter_for_exit:
                chart_folder_name += "_guppyFilterForExit"

            if global_guppy_force_out:
                chart_folder_name += "_guppyForceOut"

            if global_use_rsi_to_exit:
                chart_folder_name += "_rsiExit"

            if global_also_filter_too_late:
                chart_folder_name += "_filterTooLate"

            if global_use_guppy_condition:
                chart_folder_name += "_guppyCondition"

            if global_do_stop_loss:
                chart_folder_name += "_stopLoss"

            if (global_do_stop_loss and not global_reentry_after_stop_loss):
                chart_folder_name += "_notReentryAfterSL"

        else:

            if currency_pair.use_slow_macd:
                chart_folder_name += "_slowMACD"
            else:
                chart_folder_name += "_fastMACD"

            if currency_pair.use_guppy_filter:
                chart_folder_name += "_guppyFilter"

            if currency_pair.use_guppy_filter_for_exit:
                chart_folder_name += "_guppyFilterForExit"

            if currency_pair.guppy_force_out:
                chart_folder_name += "_guppyForceOut"

            if currency_pair.use_rsi_to_exit:
                chart_folder_name += "_rsiExit"

            if currency_pair.also_filter_too_late:
                chart_folder_name += "_filterTooLate"

            if currency_pair.use_guppy_condition:
                chart_folder_name += "_guppyCondition"

            if currency_pair.do_stop_loss:
                chart_folder_name += "_stopLoss"

            if currency_pair.do_stop_loss and not currency_pair.reentry_after_stop_loss:
                chart_folder_name += "_notReentryAfterSL"






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

        log_msg("currency_folder:")
        log_msg(currency_folder)
        data_folder = os.path.join(currency_folder, "data")
        log_msg("data_folder:")
        log_msg(data_folder)
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
        trade_prod_file = os.path.join(currency_folder, currency + "_" + chart_folder_name + "_all_trades_prod.csv")

        delay_cost_file = os.path.join(currency_folder, currency + "_" + chart_folder_name + "_delay_cost_prod.csv")

        #performance_file = os.path.join(currency_folder, currency + "_performance_" + str(profit_loss_ratio) + ".csv")

        performance_file = os.path.join(currency_folder, currency + "_" + chart_folder_name + "_performance.csv")

        #email_message_file = os.path.join(currency_folder, currency + "_emails.txt")

        email_message_file = os.path.join(currency_folder, currency + "_" + chart_folder_name + "_emails.txt")

        #log_msg("Fuck performance_file " + performance_file)

        currency_folders += [currency_folder]
        data_folders += [data_folder]
        chart_folders += [chart_folder]
        simple_chart_folders += [simple_chart_folder]
        log_files += [log_file]
        data_files += [data_file]
        if read_5min_data:
            data_files_5min += [data_file_5min]
        trade_files += [trade_file]
        trade_prod_files += [trade_prod_file]
        delay_cost_files += [delay_cost_file]

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


            log_msg("Read: " + data_file)

            df = pd.read_csv(data_file)
            close_prices += [float(df.iloc[-1]['close'])]

    for i in range(len(currency_list)):

        currency = currency_list[i]

        #log_msg("Processing currency " + currency)

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

                log_msg("Found target currency " + fx_currency)
                break

        log_msg("")

    log_msg("currencies = " + str(len(currencies)))
    log_msg("fx_currencies = " + str(len(fx_currencies)))
    log_msg("fx_raw = " + str(len(fx_raw)))
    log_msg("reciprocal = " + str(len(reciprocal)))
    log_msg("fx = " + str(len(fx)))

    final_summary_data = pd.DataFrame({'currency' : currencies, 'fx_currency': fx_currencies, 'raw_fx' : fx_raw, 'reciprocal' : reciprocal, 'fx' : fx})

    log_msg("final_summary_data:")
    log_msg(final_summary_data)

    #sys.exit(0)




    ##############



    i = 0
    for currency_pair, data_folder, chart_folder, simple_chart_folder, log_file, data_file, trade_file, trade_prod_file, delay_cost_file, performance_file, usdfx, email_message_file in list(
            zip(currency_pairs, data_folders, chart_folders, simple_chart_folders, log_files, data_files, trade_files, trade_prod_files, delay_cost_files, performance_files, fx, email_message_files)):

        currency = currency_pair.currency
        lot_size = currency_pair.lot_size
        exchange_rate = currency_pair.exchange_rate
        coefficient = currency_pair.coefficient
        actual_maxdrawdown = currency_pair.actual_maxdrawdown
        optimal_gradient_num = currency_pair.optimal_gradient_num if not do_smart_execution else currency_pair.optimal_gradient_num_execution
        decimal = currency_pair.decimal
        reverse_strategy = currency_pair.reverse_strategy
        use_slow_macd = currency_pair.use_slow_macd
        use_guppy_filter = currency_pair.use_guppy_filter
        use_guppy_filter_for_exit = currency_pair.use_guppy_filter_for_exit
        guppy_force_out = currency_pair.guppy_force_out
        use_rsi_to_exit = currency_pair.use_rsi_to_exit
        do_stop_loss = currency_pair.do_stop_loss
        reentry_after_stop_loss = currency_pair.reentry_after_stop_loss
        also_filter_too_late = currency_pair.also_filter_too_late
        use_guppy_condition = currency_pair.use_guppy_condition
        init_entry_value = currency_pair.init_entry_value
        coinbase_decimal = currency_pair.coinbase_decimal

        data_file_5min = None
        if read_5min_data:
            data_file_5min = data_files_5min[i]
            i += 1

        #log_msg("optimal_gradient_num = " + str(optimal_gradient_num))

        #log_msg("Here performance_file = " + performance_file)



        currency_trader = CurrencyTrader(threading.Condition(), currency, lot_size, exchange_rate, coefficient, actual_maxdrawdown, optimal_gradient_num, data_folder,
                                         chart_folder, simple_chart_folder, log_file, data_file, trade_file, trade_prod_file, delay_cost_file, performance_file, usdfx,
                                         email_message_file, currency in currencies_to_notify, data_file_5min if read_5min_data else None, decimal, reverse_strategy, wakeup,
                                         coinbase_client = client if do_real_money_trading else None,
                                         currency_coinbase = currency[:-len('USD')] + '-PERP-INTX' if do_real_money_trading else None,
                                         coinbase_portfolio_id = portfolio_id,
                                         crypto_last_price = currency_coinbase_close_prices[currency] if do_real_money_trading and currency in currency_coinbase_close_prices else 0,
                                         use_slow_macd = use_slow_macd, use_guppy_filter = use_guppy_filter, use_guppy_filter_for_exit = use_guppy_filter_for_exit,
                                         guppy_force_out = guppy_force_out, use_rsi_to_exit = use_rsi_to_exit,
                                         do_stop_loss = do_stop_loss, reentry_after_stop_loss = reentry_after_stop_loss,
                                         also_filter_too_late = also_filter_too_late,
                                         use_guppy_condition = use_guppy_condition, init_entry_value = init_entry_value, coinbase_decimal = coinbase_decimal,
                                         is_alternative=True if alternative == 'y' else False)
        currency_trader.daemon = True

        currency_traders += [currency_trader]

    log_msg("data_folders:")
    log_msg(data_folders)


    is_do_trading = True

    running_round = 0
    waiting_round = 0

    if is_do_trading:
        while not is_all_received:

            if running_round > 0:
                log_msg("running_round = " + str(running_round))

                now = datetime.now()
                log_msg("now = " + str(now))
                log_msg("waiting_time = " + str(waiting_next_time))
                if now < waiting_next_time:
                    seconds_remaining = (waiting_next_time - now).seconds
                    sleep_seconds = 5
                    while seconds_remaining > 0:
                        actual_sleep_seconds = seconds_remaining if seconds_remaining < sleep_seconds else sleep_seconds
                        time.sleep(actual_sleep_seconds)
                        now = datetime.now()

                        seconds_remaining = (waiting_next_time - now).seconds if now < waiting_next_time else 0
                        log_msg("seconds_remaining = " + str(seconds_remaining))


            is_all_received = True
            running_round += 1

            for i in range(len(currency_traders)):
                if not is_new_data_received[i]:
                    currency_trader = currency_traders[i]

                    data_folder = data_folders[i]

                    currency = currency_trader.currency

                    print_prefix = "[Currency " + currency + "] "

                    log_msg("Query initial for currency pair " + currency)


                    data_file = os.path.join(data_folder, currency + ".csv")
                    data_file_5min = os.path.join(data_folder, currency + "_5min.csv")
                    log_msg("data_file:")
                    log_msg(data_file)

                    data_df = None

                    if os.path.exists(data_file):

                        data_df = pd.read_csv(data_file)

                        #data_df100 = data_df100.iloc[0:-20]

                        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))

                        #data_df = data_df[data_df['time'] <= datetime(2025, 2, 10, 5, 0, 0)]  # Temp

                        final_time = data_df.iloc[-1]['time']
                        begin_time = data_df.iloc[0]['time']
                        expected_bar_num = calc_bar_num(begin_time, final_time)
                        actual_bar_num = data_df.shape[0]

                        if expected_bar_num != actual_bar_num:
                            print("expected_bar_num = " + str(expected_bar_num))
                            print("actual_bar_num = " + str(actual_bar_num))

                            data_df['expected_bar_id'] = data_df['time'].apply(lambda x: calc_bar_num(begin_time, x) - 1)
                            data_df['actual_bar_id'] = list(range(data_df.shape[0]))
                            unequal_ids = which(data_df['expected_bar_id'] != data_df['actual_bar_id'])
                            if len(unequal_ids) > 0:
                                start_wrong_time = data_df.iloc[unequal_ids[0]]['time']
                                print("start_wrong_time = " + str(start_wrong_time))
                                sys.exit(1)




                        data_df = data_df[['currency', 'time', 'open', 'high', 'low', 'close']]



                        if use_short_data_for_prod:
                            data_df = data_df[data_df['time'] >= datetime(2023, 11, 30, 2, 0, 0)]
                            data_df.reset_index(inplace=True)
                            data_df = data_df.drop(columns=['index'])



                        last_time = data_df.iloc[-1]['time']
                        log_msg("last_time = " + str(last_time))
                        last_timestamp = int(datetime.timestamp(last_time)) #- 28800
                        # next_timestamp = last_timestamp + 3600

                        log_msg("Here last time = " + str(last_time))
                        log_msg("last_timestamp = " + str(last_timestamp))
                        # time.sleep(15)

                        if is_real_time_trading:


                            incremental_data_df = get_bar_data2(currency, bar_number=initial_bar_number, end_date = until_date)

                            if incremental_data_df.iloc[0]['time'] > last_time:
                                log_msg("last_time = " + str(last_time) + ", but queried starting time is even after that" + str(incremental_data_df.iloc[0]['time']))

                            #if is_weekend:
                            incremental_data_df = incremental_data_df[incremental_data_df['time'] > last_time]
                            # else:
                            #     incremental_data_df = incremental_data_df[incremental_data_df['time'] > last_time].iloc[0:-1]


                        if is_real_time_trading and incremental_data_df.shape[0] > 0:


                            data_df = pd.concat([data_df, incremental_data_df])


                            data_df.reset_index(inplace=True)
                            data_df = data_df.drop(columns=['index'])

                    else:
                        log_msg("Currency file does not exit, query initial data from web")

                        data_df = get_bar_data2(currency, bar_number=initial_bar_number, end_date = until_date)

                        data_df = data_df.iloc[:-1]


                    if read_5min_data:

                        log_msg("Read 5 min data")

                        if os.path.exists(data_file_5min):

                            data_df_5min = pd.read_csv(data_file_5min)

                            data_df_5min['time'] = data_df_5min['time'].apply(lambda x: preprocess_time(x))

                            #data_df_5min = data_df_5min[data_df_5min['time'] <= datetime(2024,9,7,0,0,0)]  #Darren

                            data_df_5min = data_df_5min[['currency', 'time', 'open', 'high', 'low', 'close']]


                            last_time = data_df_5min.iloc[-1]['time']
                            log_msg("last_time = " + str(last_time))
                            last_timestamp = int(datetime.timestamp(last_time)) #- 28800
                            # next_timestamp = last_timestamp + 3600

                            log_msg("Here last time = " + str(last_time))
                            log_msg("last_timestamp = " + str(last_timestamp))
                            # time.sleep(15)

                            if is_real_time_trading_5min:


                                incremental_data_df_5min = get_bar_data2(currency, bar_number=initial_bar_number_5min, interval='5min', end_date = until_date_5min)



                                if incremental_data_df_5min.iloc[0]['time'] > last_time:
                                    print("5min bar: last_time = " + str(last_time) + ", but queried starting time is even after that" + str(incremental_data_df_5min.iloc[0]['time']), file = sys.stderr)

                                #if is_weekend_5min:
                                incremental_data_df_5min = incremental_data_df_5min[incremental_data_df_5min['time'] > last_time]  #Stupid Fucking Bug, wasting my whole night!!!
                                #else:

                                if until_date_5min is None or datetime.today() < preprocess_date(until_date_5min):
                                    #incremental_data_df_5min = incremental_data_df_5min[incremental_data_df_5min['time'] > last_time].iloc[0:-1]

                                    incremental_data_df_5min = incremental_data_df_5min.iloc[0:-1]


                            if is_real_time_trading_5min and incremental_data_df_5min.shape[0] > 0:


                                data_df_5min = pd.concat([data_df_5min, incremental_data_df_5min])


                                data_df_5min.reset_index(inplace=True)
                                data_df_5min = data_df_5min.drop(columns=['index'])

                                # for col in ['open', 'high', 'low', 'close']:
                                #     data_df_5min[col] = data_df_5min[col].apply(lambda x: round(x, currency_trader.decimal))

                        else:
                            log_msg("Currency file does not exit, query initial data from web")

                            data_df_5min = get_bar_data2(currency, bar_number=initial_bar_number_5min, interval='5min', end_date = until_date_5min)

                            if until_date_5min is None or datetime.today() < preprocess_date(until_date_5min):
                                data_df_5min = data_df_5min.iloc[:-1]

                            # for col in ['open', 'high', 'low', 'close']:
                            #     data_df_5min[col] = data_df_5min[col].apply(lambda x: round(x, currency_trader.decimal))



                    #if is_real_time_trading and not is_weekend:
                    if is_real_time_trading and (until_date is None or datetime.today() < preprocess_date(until_date)):

                        if data_df is not None and data_df.shape[0] > 1:
                            #last_time = data_df.iloc[-1]['time']
                            last_time = data_df.iloc[-2]['time']
                        else:
                            last_time = None

                        if read_5min_data:

                            log_msg("Now data_df_5min..........:")
                            log_msg(data_df_5min.iloc[-5:])

                            if data_df_5min is not None and data_df_5min.shape[0] > 0:
                                last_time_5min = data_df_5min.iloc[-1]['time']
                                log_msg("Here last_time_5min = " + str(last_time_5min))
                            else:
                                last_time_5min = None

                        if last_time is not None and ((not read_5min_data) or last_time_5min is not None):
                            delta = datetime.now() - last_time

                            # if read_5min_data:
                            #     delta_5min = datetime.now() - last_time_5min

                            log_msg("last_time = " + str(last_time))
                            log_msg("now = " + str(datetime.now()))

                            if read_5min_data:
                                log_msg('last_time_5min = ' + str(last_time_5min))

                            # testing_seconds = 7200
                            # if wakeup == 1:
                            #     testing_seconds = 3600

                            #log_msg("testing_seconds = " + str(testing_seconds))

                            if (delta is not None and delta.seconds > 0 and delta.seconds < 7200 and delta.days == 0):


                                log_msg("Received up-to-date data for currency pair " + currency)


                                is_new_data_received[i] = True

                                final_data_df = data_df.iloc[0:-1] #The last bar is the current hour, which has not been completed and we don't use as well
                                if read_5min_data:
                                    currency_trader.feed_data(final_data_df, data_df_5min)
                                else:
                                    currency_trader.feed_data(final_data_df)

                                if currency in currency_close_prices:
                                    close_price = currency_close_prices[currency]
                                    real_close_price = final_data_df.iloc[-1]['close']
                                    log_msg("Close Price checking: last_price = " + str(close_price) + ", close = " + str(real_close_price))
                                    difference = abs((close_price - real_close_price)/real_close_price)
                                    log_msg("difference = " + str(difference))


                                currency_trader.trade()  #Darren
                            else:

                                log_msg("Not received finalized data for " + currency + ", wait 1 minute to try again")


                                #data_df = data_df.iloc[0:-1] #Temp for testing

                                if running_round == 1:
                                    if currency in currency_close_prices:
                                        if currency in currency_close_prices:
                                            close_price = currency_close_prices[currency]
                                            log_msg(currency + " real time last price = " + str(close_price))
                                            data_df.at[data_df.index[-1], 'close'] = close_price

                                            log_msg("Real time data:")
                                            log_msg(data_df.iloc[-5:])

                                            log_msg("")

                                        if read_5min_data:
                                            currency_trader.feed_data(data_df, data_df_5min)
                                        else:
                                            currency_trader.feed_data(data_df)

                                        currency_trader.trade(print_ready=False, temporary_decision=True)  #Darren


                                if trial_numbers[i] <= maximum_trial_number:
                                    is_all_received = False
                                    log_msg("Not received data update for " + currency + ", will try again")
                                    trial_numbers[i] += 1

                                    if waiting_round < running_round:
                                        #waiting_next_time = data_df.iloc[-1]['time'] + timedelta(seconds = 3600 + running_round * 60 + 10)  #-1
                                        now_time = datetime.now()
                                        log_msg("now is " + str(now_time))
                                        waiting_next_time = datetime(now_time.year, now_time.month, now_time.day, now_time.hour, now_time.minute, now_time.second, 0) + timedelta(seconds = 120)
                                        log_msg("waiting_next_time = " + str(waiting_next_time))
                                        waiting_round += 1
                                        log_msg("running_round = " + str(running_round) + ", waiting_round = " + str(waiting_round))

                                else:
                                    log_msg("Reached maximum number of trials for " + currency + ", give up")
                    else:

                        if data_df is not None:

                            is_new_data_received[i] = True

                            #data_df = data_df.iloc[0:-1] #Temp

                            log_msg("Start trading without checking if data up-to-date as not necessary")
                            if read_5min_data:
                                currency_trader.feed_data(data_df, data_df_5min)
                            else:
                                currency_trader.feed_data(data_df)


                            currency_trader.trade()  #Darren

                    if manual_delay > 0 and len(currency_pairs) > 4:
                        log_msg("Sleep " + str(manual_delay) + " seconds ")
                        time.sleep(manual_delay)



        if do_real_money_trading and wakeup == 1:

            log_msg("")
            log_msg("Checking fill status ......................")

            is_open_order_filled = [False] * len(currency_pairs)

            is_long_open_order_filled = [False] * len(currency_pairs)
            is_short_open_order_filled = [False] * len(currency_pairs)
            is_close_long_open_order_filled = [False] * len(currency_pairs)
            is_close_short_open_order_filled = [False] * len(currency_pairs)

            long_filled_sizes = [-1] * len(currency_pairs)
            short_filled_sizes = [-1] * len(currency_pairs)
            close_long_filled_sizes = [-1] * len(currency_pairs)
            close_short_filled_sizes = [-1] * len(currency_pairs)

            long_unfilled_sizes = [-1] * len(currency_pairs)
            short_unfilled_sizes = [-1] * len(currency_pairs)
            close_long_unfilled_sizes = [-1] * len(currency_pairs)
            close_short_unfilled_sizes = [-1] * len(currency_pairs)

            long_filled_prices = [-1] * len(currency_pairs)
            short_filled_prices = [-1] * len(currency_pairs)
            close_long_filled_prices = [-1] * len(currency_pairs)
            close_short_filled_prices = [-1] * len(currency_pairs)


            is_all_filled = False

            max_trials = 60
            trial_id = 0

            use_market_order = False

            while not is_all_filled:

                is_all_filled = True

                for i in range(len(currency_traders)):

                    log_msg("Checking crypto " + currency_traders[i].currency)

                    if not is_open_order_filled[i]:

                        currency_trader = currency_traders[i]
                        log_msg("")
                        log_msg("Checking fill status of " + currency_trader.currency + " open orders if any")

                        if currency_trader.long_order_id is None:
                            log_msg("No open long order.")
                            is_long_open_order_filled[i] = True
                        else:
                            if currency_trader.long_order_id is not None:
                                log_msg("long_order_id = " + currency_trader.long_order_id)
                                fully_filled = False
                                orderResponse = client.get_order(order_id=currency_trader.long_order_id)
                                if hasattr(orderResponse, "order"):
                                    order = orderResponse.order
                                    if order is not None:
                                        status = order['status']
                                        filled_size = float(order['filled_size'])
                                        filled_price = float(order['average_filled_price'])
                                        if not use_market_order:
                                            long_filled_sizes[i] = filled_size
                                            long_filled_prices[i] = filled_price

                                        log_msg("filled_size = " + str(filled_size) + ", attempt_size = " + str(currency_trader.long_attempt_size))
                                        if status == 'FILLED' and filled_size == currency_trader.long_attempt_size:
                                            fully_filled = True
                                        long_unfilled_sizes[i] = currency_trader.long_attempt_size - filled_size


                                if fully_filled:
                                    log_msg("long order fully filled!")
                                    is_long_open_order_filled[i] = True
                                    currency_trader.reset_long()

                                    if use_market_order:
                                        log_msg("Fully filled by market order")
                                        log_msg("Passive fill size=" + str(long_filled_sizes[i]) + ", passive fill price=" + str(long_filled_prices[i]) +
                                              ", market fill size=" + str(filled_size) + ", market fill price=" + str(filled_price))
                                        average_fill_price = (long_filled_sizes[i] * long_filled_prices[i] + filled_size * filled_price)/(long_filled_sizes[i] + filled_size)
                                        currency_trader.set_long_fill(average_fill_price, long_filled_sizes[i] + filled_size)

                                        log_msg("average_fill_price=" + str(average_fill_price) + ", total_fill_size=" + str(long_filled_sizes[i] + filled_size))
                                    else:
                                        currency_trader.set_long_fill(filled_price, filled_size)
                                        log_msg("average fill price=" + str(filled_price) + ", filled_size=" + str(filled_size))
                                else:
                                    log_msg("long order not fully filled.")
                                    is_all_filled = False

                        if currency_trader.short_order_id is None:
                            log_msg("No open short order.")
                            is_short_open_order_filled[i] = True
                        else:
                            if currency_trader.short_order_id is not None:
                                log_msg("short_order_id = " + currency_trader.short_order_id)
                                fully_filled = False
                                orderResponse = client.get_order(order_id=currency_trader.short_order_id)
                                if hasattr(orderResponse, "order"):
                                    order = orderResponse.order
                                    if order is not None:
                                        status = order['status']
                                        filled_size = float(order['filled_size'])
                                        filled_price = float(order['average_filled_price'])
                                        if not use_market_order:
                                            short_filled_sizes[i] = filled_size
                                            short_filled_prices[i] = filled_price

                                        log_msg("filled_size = " + str(filled_size) + ", attempt_size = " + str(currency_trader.short_attempt_size))
                                        if status == 'FILLED' and filled_size == currency_trader.short_attempt_size:
                                            fully_filled = True
                                        short_unfilled_sizes[i] = currency_trader.short_attempt_size - filled_size


                                if fully_filled:
                                    log_msg("short order fully filled!")
                                    is_short_open_order_filled[i] = True
                                    currency_trader.reset_short()

                                    if use_market_order:
                                        log_msg("Fully filled by market order")
                                        log_msg("Passive fill size=" + str(short_filled_sizes[i]) + ", passive fill price=" + str(short_filled_prices[i]) +
                                              ", market fill size=" + str(filled_size) + ", market fill price=" + str(filled_price))
                                        average_fill_price = (short_filled_sizes[i] * short_filled_prices[i] + filled_size * filled_price)/(short_filled_sizes[i] + filled_size)
                                        currency_trader.set_short_fill(average_fill_price, short_filled_sizes[i] + filled_size)

                                        log_msg("average_fill_price=" + str(average_fill_price) + ", total_fill_size=" + str(short_filled_sizes[i] + filled_size))
                                    else:
                                        currency_trader.set_short_fill(filled_price, filled_size)
                                        log_msg("average fill price=" + str(filled_price) + ", filled_size=" + str(filled_size))

                                else:
                                    log_msg("short order not fully filled.")
                                    is_all_filled = False

                        if currency_trader.close_long_order_id is None:
                            log_msg("No open close long order.")
                            is_close_long_open_order_filled[i] = True
                        else:
                            if currency_trader.close_long_order_id is not None:
                                log_msg("close_long_order_id = " + currency_trader.close_long_order_id)
                                fully_filled = False
                                orderResponse = client.get_order(order_id=currency_trader.close_long_order_id)
                                if hasattr(orderResponse, "order"):
                                    order = orderResponse.order
                                    if order is not None:
                                        status = order['status']
                                        filled_size = float(order['filled_size'])
                                        filled_price = float(order['average_filled_price'])
                                        if not use_market_order:
                                            close_long_filled_sizes[i] = filled_size
                                            close_long_filled_prices[i] = filled_price

                                        log_msg("filled_size = " + str(filled_size) + ", attempt_size = " + str(currency_trader.close_long_attempt_size))
                                        if status == 'FILLED' and filled_size == currency_trader.close_long_attempt_size:
                                            fully_filled = True
                                        close_long_unfilled_sizes[i] = currency_trader.close_long_attempt_size - filled_size


                                if fully_filled:
                                    log_msg("close_long order fully filled!")
                                    is_close_long_open_order_filled[i] = True
                                    currency_trader.reset_close_long()

                                    if use_market_order:
                                        log_msg("Fully filled by market order")
                                        log_msg("Passive fill size=" + str(close_long_filled_sizes[i]) + ", passive fill price=" + str(close_long_filled_prices[i]) +
                                              ", market fill size=" + str(filled_size) + ", market fill price=" + str(filled_price))
                                        average_fill_price = (close_long_filled_sizes[i] * close_long_filled_prices[i] + filled_size * filled_price)/(close_long_filled_sizes[i] + filled_size)
                                        currency_trader.set_close_long_fill(average_fill_price, close_long_filled_sizes[i] + filled_size)

                                        log_msg("average_fill_price=" + str(average_fill_price) + ", total_fill_size=" + str(close_long_filled_sizes[i] + filled_size))
                                    else:
                                        currency_trader.set_close_long_fill(filled_price, filled_size)
                                        log_msg("average fill price=" + str(filled_price) + ", filled_size=" + str(filled_size))


                                else:
                                    log_msg("close_long order not fully filled.")
                                    is_all_filled = False


                        if currency_trader.close_short_order_id is None:
                            log_msg("No open close_short order.")
                            is_close_short_open_order_filled[i] = True
                        else:
                            if currency_trader.close_short_order_id is not None:
                                log_msg("close_short_order_id = " + currency_trader.close_short_order_id)
                                fully_filled = False
                                orderResponse = client.get_order(order_id=currency_trader.close_short_order_id)
                                if hasattr(orderResponse, "order"):
                                    order = orderResponse.order
                                    if order is not None:
                                        status = order['status']
                                        filled_size = float(order['filled_size'])
                                        filled_price = float(order['average_filled_price'])
                                        if not use_market_order:
                                            close_short_filled_sizes[i] = filled_size
                                            close_short_filled_prices[i] = filled_price

                                        log_msg("filled_size = " + str(filled_size) + ", attempt_size = " + str(currency_trader.close_short_attempt_size))
                                        if status == 'FILLED' and filled_size == currency_trader.close_short_attempt_size:
                                            fully_filled = True
                                        close_short_unfilled_sizes[i] = currency_trader.close_short_attempt_size - filled_size


                                if fully_filled:
                                    log_msg("close_short order fully filled!")
                                    is_close_short_open_order_filled[i] = True
                                    currency_trader.reset_close_short()

                                    if use_market_order:
                                        log_msg("Fully filled by market order")
                                        log_msg("Passive fill size=" + str(close_short_filled_sizes[i]) + ", passive fill price=" + str(close_short_filled_prices[i]) +
                                              ", market fill size=" + str(filled_size) + ", market fill price=" + str(filled_price))
                                        average_fill_price = (close_short_filled_sizes[i] * close_short_filled_prices[i] + filled_size * filled_price)/(close_short_filled_sizes[i] + filled_size)
                                        currency_trader.set_close_short_fill(average_fill_price, close_short_filled_sizes[i] + filled_size)

                                        log_msg("average_fill_price=" + str(average_fill_price) + ", total_fill_size=" + str(close_short_filled_sizes[i] + filled_size))
                                    else:
                                        currency_trader.set_close_short_fill(filled_price, filled_size)
                                        log_msg("average fill price=" + str(filled_price) + ", filled_size=" + str(filled_size))


                                else:
                                    log_msg("close_short order not fully filled.")
                                    is_all_filled = False


                        is_open_order_filled[i] = is_long_open_order_filled[i] and is_short_open_order_filled[i] and is_close_long_open_order_filled[i] and is_close_short_open_order_filled[i]



                if not is_all_filled:

                    if trial_id < max_trials:
                        log_msg("")
                        wait_seconds = 10
                        log_msg("Not all cryptos have fully filled their open orders, wait for " + str(wait_seconds) + " seconds and check again.")
                        time.sleep(wait_seconds)
                        trial_id += 1
                    else:
                        log_msg("")
                        use_market_order = True
                        log_msg("Reached max waiting time. Now use market orders to fill all")
                        for i in range(len(currency_traders)):
                            log_msg("")
                            log_msg("Crypto " + str(currency_traders[i].currency))

                            if not is_long_open_order_filled[i]:
                                currency_trader = currency_traders[i]
                                unfilled_size = long_unfilled_sizes[i]

                                try:
                                    log_msg("Cancel long order " + currency_trader.long_order_id)
                                    cancel_response = client.cancel_orders(order_ids=[currency_trader.long_order_id])
                                    log_msg("Cancel Response:")
                                    log_msg(cancel_response)
                                except Exception as e:
                                    log_msg("Error:", e)

                                try:
                                    log_msg("Place market long order of " + str(unfilled_size) + " units")
                                    client_order_id = f"order_{uuid.uuid4()}"
                                    response = client.create_order(product_id=currency_trader.currency_coinbase,
                                                                   client_order_id=client_order_id,
                                                                   side="BUY",
                                                                   order_configuration={
                                                                       "market_market_ioc":{
                                                                           "base_size" : str(unfilled_size)
                                                                       }
                                                                   },
                                                                   leverage=str(default_leverage),
                                                                   margin_type = "CROSS",
                                                                   retail_portfolio_id=currency_trader.coinbase_portfolio_id
                                                                   )
                                    log_msg(f"Order placed: {response}")
                                except Exception as e:
                                    log_msg(f"Order failed: {e}")

                                currency_trader.set_long(response['success_response']['order_id'], unfilled_size)


                            if not is_short_open_order_filled[i]:
                                currency_trader = currency_traders[i]
                                unfilled_size = short_unfilled_sizes[i]

                                try:
                                    log_msg("Cancel short order " + currency_trader.short_order_id)
                                    cancel_response = client.cancel_orders(order_ids=[currency_trader.short_order_id])
                                    log_msg("Cancel Response:")
                                    log_msg(cancel_response)
                                except Exception as e:
                                    log_msg("Error:", e)

                                try:
                                    log_msg("Place market short order of " + str(unfilled_size) + " units")
                                    client_order_id = f"order_{uuid.uuid4()}"
                                    response = client.create_order(product_id=currency_trader.currency_coinbase,
                                                                   client_order_id=client_order_id,
                                                                   side="SELL",
                                                                   order_configuration={
                                                                       "market_market_ioc":{
                                                                           "base_size" : str(unfilled_size)
                                                                       }
                                                                   },
                                                                   leverage=str(default_leverage),
                                                                   margin_type = "CROSS",
                                                                   retail_portfolio_id=currency_trader.coinbase_portfolio_id
                                                                   )
                                    log_msg(f"Order placed: {response}")
                                except Exception as e:
                                    log_msg(f"Order failed: {e}")

                                currency_trader.set_short(response['success_response']['order_id'], unfilled_size)

                            if not is_close_long_open_order_filled[i]:
                                currency_trader = currency_traders[i]
                                unfilled_size = close_long_unfilled_sizes[i]

                                try:
                                    log_msg("Cancel close_long order " + currency_trader.close_long_order_id)
                                    cancel_response = client.cancel_orders(order_ids=[currency_trader.close_long_order_id])
                                    log_msg("Cancel Response:")
                                    log_msg(cancel_response)
                                except Exception as e:
                                    log_msg("Error:", e)

                                try:
                                    log_msg("Place market close_long order of " + str(unfilled_size) + " units")
                                    client_order_id = f"order_{uuid.uuid4()}"
                                    response = client.create_order(product_id=currency_trader.currency_coinbase,
                                                                   client_order_id=client_order_id,
                                                                   side="SELL",
                                                                   order_configuration={
                                                                       "market_market_ioc":{
                                                                           "base_size" : str(unfilled_size)
                                                                       }
                                                                   },
                                                                   leverage=str(default_leverage),
                                                                   margin_type = "CROSS",
                                                                   retail_portfolio_id=currency_trader.coinbase_portfolio_id
                                                                   )
                                    log_msg(f"Order placed: {response}")
                                except Exception as e:
                                    log_msg(f"Order failed: {e}")

                                currency_trader.set_close_long(response['success_response']['order_id'], unfilled_size)

                            if not is_close_short_open_order_filled[i]:
                                currency_trader = currency_traders[i]
                                unfilled_size = close_short_unfilled_sizes[i]

                                try:
                                    log_msg("Cancel close_short order " + currency_trader.close_short_order_id)
                                    cancel_response = client.cancel_orders(order_ids=[currency_trader.close_short_order_id])
                                    log_msg("Cancel Response:")
                                    log_msg(cancel_response)
                                except Exception as e:
                                    log_msg("Error:", e)

                                try:
                                    log_msg("Place market close_short order of " + str(unfilled_size) + " units")
                                    client_order_id = f"order_{uuid.uuid4()}"
                                    response = client.create_order(product_id=currency_trader.currency_coinbase,
                                                                   client_order_id=client_order_id,
                                                                   side="BUY",
                                                                   order_configuration={
                                                                       "market_market_ioc":{
                                                                           "base_size" : str(unfilled_size)
                                                                       }
                                                                   },
                                                                   leverage=str(default_leverage),
                                                                   margin_type = "CROSS",
                                                                   retail_portfolio_id=currency_trader.coinbase_portfolio_id
                                                                   )
                                    log_msg(f"Order placed: {response}")
                                except Exception as e:
                                    log_msg(f"Order failed: {e}")

                                currency_trader.set_close_short(response['success_response']['order_id'], unfilled_size)



                else:
                    log_msg("")
                    log_msg("All cryptos have their open orders fully filled, bye bye!")

        #Darren
        for i in range(len(currency_traders)):
            if is_new_data_received[i]:
                currency_trader = currency_traders[i]
                currency_trader.post_processing()

        sendEmail("Trader process ends", "")

        log_msg("Finished trading *********************************")


        log_msg("Collecting Results....")

        perf_dfs = []
        trade_dfs = []
        prod_trade_dfs = []
        i = 0
        #sys.exit(0)  #Darren
        for currency in currency_list:
            #perf_file = os.path.join(root_folder, currency, currency + "_performance_" + str(profit_loss_ratio) + ".csv")
            chart_folder_name = chart_folder_names[i]
            i += 1
            perf_file = os.path.join(root_folder, currency, currency + "_" + chart_folder_name + "_performance.csv")
            perf_dfs += [pd.read_csv(perf_file)]

            #trade_file = os.path.join(root_folder, currency, currency + "_all_trades_" + str(profit_loss_ratio) + ".csv")
            trade_file = os.path.join(root_folder, currency, currency + "_" + chart_folder_name + "_all_trades.csv")

            if do_real_money_trading and production_running:
                prod_trade_file = os.path.join(root_folder, currency, currency + "_" + chart_folder_name + "_all_trades_prod.csv")

            trade_df = pd.read_csv(trade_file)
            trade_dfs += [trade_df]

            if do_real_money_trading and production_running:
                prod_trade_df = pd.read_csv(prod_trade_file)
                prod_trade_dfs += [prod_trade_df]

        perf_df = pd.concat(perf_dfs)

        trade_df = pd.concat(trade_dfs)
        trade_df = trade_df.sort_values(by = ['exit_time']) #entry_time
        print("entry_time type: " + str(type(trade_df.iloc[0]['entry_time'])))
        print("exit_time type: " + str(type(trade_df.iloc[0]['exit_time'])))

        if do_real_money_trading and production_running:
            prod_trade_df = pd.concat(prod_trade_dfs)
            prod_trade_df = prod_trade_df.sort_values(by=['exit_time']) #entry_time

            print("prod entry_time type: " + str(type(prod_trade_df.iloc[0]['entry_time'])))
            print("prod exit_time type: " + str(type(prod_trade_df.iloc[0]['exit_time'])))


        log_msg("Final Performance Result:")
        perf_df.reset_index(inplace = True)
        perf_df = perf_df.drop(columns = ['index'])
        log_msg(perf_df)

        # log_msg("")
        # log_msg("Selected currencies Performance Result:")
        # selected_perf_df = perf_df[perf_df['Currency'].isin(selected_currencies)]
        # log_msg(selected_perf_df)

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
        # log_msg("des_selected_bar_folder")
        # log_msg(des_selected_bar_folder)
        if not os.path.exists(des_selected_bar_folder):
            os.makedirs(des_selected_bar_folder)



        log_msg("Copying bar charts and pnl charts...")
        #trade_df = trade_df.drop(columns = ['id', 'pnl', 'cum_pnl', 'reverse_pnl', 'cum_reverse_pnl'])

        trade_df = trade_df.drop(columns=['trade_id', 'long_trade_id', 'short_trade_id', 'cum_pnl'])
        trade_df['cum_pnl'] = trade_df['pnl'].cumsum()
        trade_df['cum_pnl'] = trade_df['cum_pnl'].apply(lambda x: round(x, 2))
        trade_df.to_csv(os.path.join(des_pnl_folder, "all_trades.csv"), index = False)

        if do_real_money_trading and production_running:
            prod_trade_df = prod_trade_df.drop(columns=['long_trade_id', 'short_trade_id', 'cum_pnl', 'prod_cum_pnl'])
            prod_trade_df['cum_pnl'] = prod_trade_df['pnl'].cumsum()
            prod_trade_df['cum_pnl'] = prod_trade_df['cum_pnl'].apply(lambda x: round(x, 2))
            prod_trade_df['prod_cum_pnl'] = prod_trade_df['prod_pnl'].cumsum()
            prod_trade_df['prod_cum_pnl'] = prod_trade_df['prod_cum_pnl'].apply(lambda x: round(x, 2))

            prod_trade_df['execution_cost'] = prod_trade_df['prod_pnl'] - prod_trade_df['pnl']
            prod_trade_df['cum_execution_cost'] = prod_trade_df['execution_cost'].cumsum()

            prod_trade_df['execution_cost'] = prod_trade_df['execution_cost'].apply(lambda x: round(x, 2))
            prod_trade_df['cum_execution_cost'] = prod_trade_df['cum_execution_cost'].apply(lambda x: round(x, 2))


            prod_trade_df.to_csv(os.path.join(des_pnl_folder, "all_trades_prod.csv"), index=False)


        i = 0
        for currency in currency_list:

            chart_folder_name = chart_folder_names[i]
            i += 1

            #log_msg("currency = " + str(currency))
            pic_path = os.path.join(root_folder, currency, chart_folder_name, currency + '_pnl.png')
            if os.path.exists(pic_path):
                shutil.copy2(pic_path, des_pnl_folder)

                if currency in selected_currencies:
                    shutil.copy2(pic_path, des_selected_pnl_folder)


            currency_chart_folder = os.path.join(root_folder, currency, chart_folder_name)
            chart_files = os.listdir(currency_chart_folder)
            for chart_file in chart_files:
                if 'pnl' not in chart_file:

                    #log_msg(os.path.join(currency_chart_folder, chart_file))
                    #log_msg(des_bar_folder)

                    shutil.copy2(os.path.join(currency_chart_folder, chart_file), des_bar_folder)

                    #log_msg("des_bar_folder:")
                    #log_msg(des_bar_folder)

                    if currency in selected_currencies:
                        # log_msg("currency_chart_folder:")
                        # log_msg(currency_chart_folder)
                        # log_msg("source file:")
                        # log_msg(os.path.join(currency_chart_folder, chart_file))
                        # log_msg("des folder:")
                        # log_msg(des_selected_bar_folder)

                        source_exists = os.path.exists(os.path.join(currency_chart_folder, chart_file))
                        des_exists = os.path.exists(des_selected_bar_folder)

                        # log_msg("source_exist = " + str(source_exists))
                        # log_msg("des_exist = " + str(des_exists))

                        shutil.copy2(os.path.join(currency_chart_folder, chart_file), des_selected_bar_folder)


        #shutil.copy2(file_path, dest_folder)

    # if is_do_portfolio_trading:
    #     log_msg("1 is_do_portfolio_trading = " + str(is_do_portfolio_trading))
    #     os.system('python plot_pnl_curve.py')
    # else:
    #     log_msg("2 is_do_portfolio_trading = " + str(is_do_portfolio_trading))

    log_msg("All finished")
    #sys.exit(0)


#start_do_trading()