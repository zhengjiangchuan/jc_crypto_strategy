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
from util import *
from datetime import datetime, timedelta
import math

from optparse import OptionParser
import matplotlib.ticker as ticker

import urllib.request

from io import StringIO



pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)


import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import timedelta
from matplotlib.ticker import Formatter
import  matplotlib.ticker as plticker
import matplotlib.dates as mdates

def which(bool_array):

    a = np.arange(len(bool_array))
    return a[bool_array]

def preprocess_time(t):

    if t[0] == "\'":
        t = t[1:]

    if len(t) < 19:
        t = t + ' 00:00:00'

    return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

is_subtract_commission = True

recalculate_margin_level = True
stop_level = 0.5
# symbol = 'AUDUSD'
# contract_size = 100000
# exchange_rate = 1.0
# lot_size = contract_size * exchange_rate
# spread = 15
# principal = 10000
# deposit_per_lot = 1000
selected_symbols = []

is_gege_server = False

if is_gege_server:
    data_folder = "/home/min/forex/formal_trading"
else:
    data_folder = "C:\\Forex\\formal_trading"

#data_folder = "C:\\Forex\\formal_trading\\"

meta_file = os.path.join(data_folder, 'symbols_meta.csv')
meta_df = pd.read_csv(meta_file)

# meta_df = meta_df[~meta_df['symbol'].isin(['AUDNZD', 'EURCHF', 'EURNZD','GBPAUD',
#                                                         'GBPCAD', 'GBPCHF', 'USDCAD', 'GBPUSD', 'GBPNZD'])]


#meta_df = meta_df[meta_df['symbol'].isin(['AUDJPY', 'EURCAD', 'GBPUSD', 'NZDJPY', 'USDCAD', 'NZDUSD', 'CADCHF', 'USDJPY'])]

#meta_df = meta_df[meta_df['symbol'].isin(['AUDJPY', 'EURCAD', 'NZDJPY', 'USDCAD', 'NZDUSD'])]
#weights = {'AUDJPY' : 1, 'EURCAD' : 1, 'NZDJPY' : 1, 'USDCAD' : 1, 'NZDUSD' : 1}


meta_df = meta_df[meta_df['symbol'].isin(['CADCHF', 'USDJPY'])]
weights = {'CADCHF' : 1, 'USDJPY' : 1}


#weights = {'AUDJPY' : 1, 'EURCAD' : 1, 'GBPUSD' : 1, 'NZDJPY' : 1, 'USDCAD' : 1, 'NZDUSD' : 1, 'CADCHF' : 3, 'USDJPY' : 3}



#meta_df = meta_df[meta_df['symbol'].isin(['CADCHF', 'USDJPY', 'AUDJPY', 'EURCAD', 'GBPUSD', 'NZDJPY', 'USDCAD', 'NZDUSD'])]

if len(selected_symbols) > 0:
    meta_df = meta_df[meta_df['symbol'].isin(selected_symbols)]

if is_gege_server:
    pnl_folder = os.path.join(data_folder, 'pnl')
else:
    pnl_folder = os.path.join(data_folder, 'pnl', 'pnl0924', 'final', 'pnl_summary_spread15_innovativeFire2new_maxPnl_25000_quickLossDelayed_noTrendFollow_SpecialExclude_selected_portfolio')

#pnl_folder = os.path.join(data_folder, 'pnl', 'pnl0723', 'pnl_summary_spread15_innovativeFire2new_11pm')
if not os.path.exists(pnl_folder):
    os.makedirs(pnl_folder)


symbols = []
min_margin_levels = []
restart_min_margin_levels = []
drawdown_min_margin_levels = []
pnl = []
return_rate = []
max_drawdown = []
max_drawdown_rate = []

initial_principal = 2500  #2500

use_correct_positioning = True

draw_figure = True

draw_intraday_pnl = False

####################### Portfolio trading ####################################

is_portfolio = True

plot_hk_pnl = True
initial_deposit_hk = 25000   #31000

is_send_email = True

symbols = list(meta_df['symbol'])

total_cum_positions = []
total_cum_abs_positions = []

is_do_strategy_average = False

data_file_suffix = 'only_second_entry_trend_follow'  #'only_second_entry_trend_follow'

if is_portfolio:

    max_exposure = 2 #12 #6
    initial_principal_magnifier = 2 #6.435 #8



    print("Prepare overall_data_df")
    symbol_data_dfs = []
    for i in range(meta_df.shape[0]):

        row = meta_df.iloc[i]
        symbol = row['symbol']
        print("Read symbol " + symbol)

        weight = weights[symbol]

        #data_file = os.path.join(data_folder, symbol, 'data', symbol + '100.csv')
        data_file = os.path.join(data_folder, symbol, 'data', symbol + '100.csv')

        if is_do_strategy_average:
            data_file2 = os.path.join(data_folder, symbol, 'data', symbol + '100' + data_file_suffix + '.csv')


        data_df = pd.read_csv(data_file)
        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))
        simple_data_df = data_df[['time', 'position', 'cum_position']]
        simple_data_df['position'] = simple_data_df['position'] * weight
        simple_data_df['cum_position'] = simple_data_df['cum_position'] * weight


        if is_do_strategy_average:
            data_df2 = pd.read_csv(data_file2)
            data_df2['time'] = data_df2['time'].apply(lambda x: preprocess_time(x))
            simple_data_df2 = data_df2[['time', 'position', 'cum_position']]

            simple_data_df2 = simple_data_df2.rename(columns = {"position" : 'position2', "cum_position" : 'cum_position2'})

            simple_data_df = pd.merge(simple_data_df, simple_data_df2, on = ['time'], how = 'inner')
            simple_data_df['position'] = (simple_data_df['position'] + simple_data_df['position2'])/2.0
            simple_data_df['cum_position'] = (simple_data_df['cum_position'] + simple_data_df['cum_position2'])/2.0

            simple_data_df = simple_data_df.drop(columns = ['position2', 'cum_position2'])


        simple_data_df = simple_data_df.rename(columns = {
            'position' : symbol + '_position',
            'cum_position' : symbol + '_cum_position'
        })

        print("Raw symbol = " + str(symbol))
        print("data length = " + str(simple_data_df.shape[0]))
        symbol_data_dfs += [simple_data_df]

    overall_data_df = reduce(lambda left, right: pd.merge(left, right, on = ['time'], how = 'outer'), symbol_data_dfs)

    overall_data_df = overall_data_df.fillna(0)

    overall_data_df.reset_index(inplace = True)
    overall_data_df = overall_data_df.drop(columns = ['index'])

    print("overall_data_df created")

    print("Overall_data_df length = " + str(overall_data_df.shape[0]))

    print("start_time = " + str(overall_data_df.iloc[0]['time']))
    print("end_time = " + str(overall_data_df.iloc[-1]['time']))

    #print(overall_data_df[['time', 'EURUSD_position', 'EURUSD_cum_position', 'AUDUSD_position', 'AUDUSD_cum_position']].head(400))

    #sys.exit(0)

    symbol_actual_positions = {}
    symbol_actual_cum_positions = {}
    for symbol in symbols:
        symbol_actual_positions[symbol] = []
        symbol_actual_cum_positions[symbol] = []

    symbol_factors = {}
    for symbol in symbols:
        symbol_factors[symbol] = 1

    total_cum_position = 0

    total_cum_abs_position = 0

    for i in range(overall_data_df.shape[0]):
        #print("Process row " + str(i))
        row = overall_data_df.iloc[i]
        for symbol in symbols:
            #print("    Process symbol " + symbol)


            start = False
            if abs(row[symbol + '_position']) > 1e-5 and (i == 0 or abs(overall_data_df.iloc[i-1][symbol + '_cum_position']) < 1e-5):
                start = True

            if start:
                position = row[symbol + '_position']

                if not use_correct_positioning:
                    attempt_total_cum_position = total_cum_position + position
                else:

                    attempt_total_cum_position = total_cum_abs_position + abs(position)
                    assert(attempt_total_cum_position > 0)

                is_exceed_max_exposure = False
                if not use_correct_positioning:
                    is_exceed_max_exposure = total_cum_position * position > 0 and abs(attempt_total_cum_position) > max_exposure
                else:
                    is_exceed_max_exposure = attempt_total_cum_position > max_exposure

                if is_exceed_max_exposure:

                    already_exceed_max_exposure = False

                    if not use_correct_positioning:
                        already_exceed_max_exposure = abs(total_cum_position) > max_exposure
                    else:
                        already_exceed_max_exposure = total_cum_abs_position >= max_exposure

                    if already_exceed_max_exposure:
                        capped_position = 0
                        symbol_factor = 0

                        # print("")
                        # print("i = " + str(i))
                        # print("symbol = " + symbol)
                        # print("total_cum_abs_position = " + str(total_cum_abs_position))
                        # print("Here 00 position = " + str(position))
                        # print("Here 00 capped_position = " + str(capped_position))
                    else:

                        # print("")
                        # print("i = " + str(i))
                        # print("symbol = " + symbol)
                        #
                        # print("total_cum_position = " + str(total_cum_position))
                        # print("position = " + str(position))
                        # print("attempt_total_cum_position = " + str(attempt_total_cum_position))
                        # print("max_exposure = " + str(max_exposure))

                        if not use_correct_positioning:
                            max_position = max_exposure if position > 0 else -max_exposure
                            print("max_position = " + str(max_position))
                            capped_position = max_position - total_cum_position

                            print("capped_position = " + str(capped_position))

                            symbol_factor = capped_position / position

                        else:
                            capped_abs_position = max_exposure - total_cum_abs_position
                            assert(capped_abs_position > 0)

                            #print("capped_abs_position = " + str(capped_abs_position))

                            symbol_factor = capped_abs_position / position if position > 0 else -capped_abs_position / position

                            if position > 0:
                                capped_position = capped_abs_position
                            else:
                                capped_position = -capped_abs_position




                        #print("symbol_factor = " + str(symbol_factor))

                    assert(symbol_factor >= 0 and symbol_factor <= 1)
                    symbol_factors[symbol] = symbol_factor

                    # print("")
                    # print("i = " + str(i))
                    # print("symbol = " + symbol)
                    # print("total_cum_abs_position = " + str(total_cum_abs_position))
                    # print("Here 0 position = " + str(position))
                    # print("Here 0 capped_position = " + str(capped_position))

                else:

                    # if i == 262:
                    #     print("262 1:symbol = " + symbol)
                    #     print("position = " + str(position))
                    #     print("")

                    capped_position = position

                    # print("")
                    # print("i = " + str(i))
                    # print("symbol = " + symbol)
                    # print("total_cum_abs_position = " + str(total_cum_abs_position))
                    # print("Here 1 position = " + str(position))
                    # print("Here 1 capped_position = " + str(capped_position))
            else:
                position = row[symbol + '_position']
                capped_position = position * symbol_factors[symbol]


                # if i == 262:
                #     print("262 2:symbol = " + symbol)
                #     print("position = " + str(position))
                #     print("factor = " + str(symbol_factors[symbol]))
                #     print("capped_position = " + str(capped_position))
                #     print("")

                # print("Here 2 position = " + str(position))
                # print("Here 2 capped_position = " + str(capped_position))

            capped_position = round(capped_position, 2)


            # if (position != 0 and capped_position == 0):
            #     print("i = " + str(i))
            #     print("position = " + str(position))
            #     print("capped_position = " + str(capped_position))
            #     print("symbol = " + symbol)
            #     sys.exit(0)

            # print("position = " + str(position))
            # print("capped_position = " + str(capped_position))

            assert(capped_position * position >= 0)

            # if i == 262:
            #     print('Before total_cum_position = ' + str(total_cum_position))

            total_cum_position += capped_position
            #total_cum_abs_position += abs(capped_position)

            if abs(row[symbol + '_cum_position']) < 1e-5:
                if len(symbol_actual_cum_positions[symbol]) > 0:
                    capped_position = -symbol_actual_cum_positions[symbol][-1]



            # if i == 262:
            #     print('After total_cum_position = ' + str(total_cum_position))


            #print("Calculated total_cum_position = " + str(total_cum_position))

            symbol_actual_positions[symbol] += [capped_position]

            if len(symbol_actual_cum_positions[symbol]) > 0:
                symbol_last_cum_position = symbol_actual_cum_positions[symbol][-1]
            else:
                symbol_last_cum_position = 0

            symbol_cur_cum_position = symbol_last_cum_position + capped_position

            symbol_position_delta = abs(symbol_cur_cum_position) - abs(symbol_last_cum_position)

            symbol_actual_cum_positions[symbol] += [symbol_cur_cum_position]

            total_cum_abs_position += symbol_position_delta


            # if abs(capped_position) > 0:
            #     print("")
            #     print("i = " + str(i))
            #     print("symbol = " + symbol)
            #     print("capped_position = " + str(capped_position))
            #     print("symbol_last_cum_position = " + str(symbol_last_cum_position))
            #     print("symbol_cur_cum_position = " + str(symbol_cur_cum_position))
            #     print("symbol_position_delta = " + str(symbol_position_delta))
            #
            #     print("Update total_cum_abs_position = " + str(total_cum_abs_position))



            if abs(row[symbol + '_cum_position']) < 1e-5:
                if symbol_factors[symbol] < 1:
                    symbol_factors[symbol] = 1







        total_cum_positions += [total_cum_position]
        total_cum_abs_positions += [total_cum_abs_position]

    final_data_df = overall_data_df[['time']]

    for symbol in symbols:
        #print("Final process symbol " + symbol)
        #print("final_data_df length = " + str(final_data_df.shape[0]))
        final_data_df[symbol+'_position'] = overall_data_df[symbol+'_position']
        final_data_df[symbol+'_cum_position'] = overall_data_df[symbol+'_cum_position']

        #print("actual_positions length = " + str(len(symbol_actual_positions[symbol])))
        final_data_df[symbol+'_actual_position'] = symbol_actual_positions[symbol]
        final_data_df[symbol+'_actual_cum_position'] = final_data_df[symbol+'_actual_position'].cumsum()


    #IMportant assertions
    for symbol in symbols:
        cum_position_zero = np.abs(final_data_df[symbol+'_cum_position']) < 1e-5
        actual_cum_position_not_zero = np.abs(final_data_df[symbol+'_actual_cum_position']) > 1e-5

        problem_idx = which(cum_position_zero & actual_cum_position_not_zero)
        if len(problem_idx) > 0:
            print("")
            print("Problem detected!")
            print("symbol " + str(symbol))
            print("idx:")
            print(problem_idx)
            assert(False)


    final_data_df['actual_cum_position'] = 0
    for symbol in symbols:
        final_data_df['actual_cum_position'] += np.abs(final_data_df[symbol + '_actual_cum_position'])

    # check_sum = 0
    # check_row = final_data_df.iloc[747]
    # for symbol in symbols:
    #     check_cum_position = check_row[symbol + '_actual_cum_position']
    #     print(symbol + ":")
    #     check_sum += check_cum_position
    #     print("check_cum_position = " + str(check_cum_position))
    #     print("check_sum = " + str(check_sum))
    # print("final check_sum = " + str(check_sum))
    # sys.exit(0)



    final_data_df['total_cum_position'] = total_cum_positions
    final_data_df['total_cum_abs_position'] = total_cum_abs_positions

    start_time = final_data_df.iloc[0]['time']
    end_time = final_data_df.iloc[-1]['time']


    final_data_df.to_csv(os.path.join(pnl_folder, 'portfolio_position.csv'), index = False)


#sys.exit(0)
if is_portfolio and is_send_email:
    lot_per_unit = round(initial_deposit_hk / (initial_principal * initial_principal_magnifier * 7.77), 2)

    last_row = final_data_df.iloc[-1]

    current_time = final_data_df.iloc[-1]['time'] + timedelta(seconds=3600)

    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    message_array = []
    for symbol in symbols:
        symbol_position = last_row[symbol + '_actual_position']
        if abs(symbol_position) > 1e-5:
            symbol_position = round(symbol_position * lot_per_unit, 2)
            operation = "Long" if symbol_position > 1e-5 else "Short"

            symbol_position = abs(symbol_position)

            message_array += [operation + ' ' + symbol + ' ' + str(symbol_position) + ' lot']

    if len(message_array) > 0:
        trading_message = ', '.join(message_array)

        print("Trading_message:")
        print(trading_message)

        sendEmail("Trading Operations at " + current_time, trading_message)




##############################################################################

if is_portfolio:
    pnl_df = final_data_df[['time']]


sample_data_df = None


# print("meta_df:")
# print(meta_df)

for i in range(meta_df.shape[0] + 1):

    if i == meta_df.shape[0] and not is_portfolio:
        break

    if i < meta_df.shape[0]:

        row = meta_df.iloc[i]
        symbol = row['symbol']
        exchange_rate = row['exchange_rate']
        principal = initial_principal
        deposit_per_lot = row['deposit_per_lot']
        contract_size = row['contract_size']
        spread = row['spread']

        weight = weights[symbol]

        #symbols += [symbol]

        lot_size = contract_size * exchange_rate





        #data_df = pd.read_csv(data_file)
        #data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))



        #data_file = os.path.join(data_folder, symbol, 'data', symbol + '100.csv')
        data_file = os.path.join(data_folder, symbol, 'data', symbol + '100.csv')

        if is_do_strategy_average:
            data_file2 = os.path.join(data_folder, symbol, 'data', symbol + '100' + data_file_suffix + '.csv')


        data_df = pd.read_csv(data_file)
        data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))
        #simple_data_df = data_df[['time', 'position', 'cum_position']]


        if is_do_strategy_average:
            data_df2 = pd.read_csv(data_file2)
            data_df2['time'] = data_df2['time'].apply(lambda x: preprocess_time(x))
            simple_data_df2 = data_df2[['time', 'position', 'cum_position']]

            simple_data_df2 = simple_data_df2.rename(columns = {"position" : 'position2', "cum_position" : 'cum_position2'})

            data_df = pd.merge(data_df, simple_data_df2, on = ['time'], how = 'inner')
            data_df['position'] = (data_df['position'] + data_df['position2'])/2.0
            data_df['cum_position'] = (data_df['cum_position'] + data_df['cum_position2'])/2.0

            data_df = data_df.drop(columns = ['position2', 'cum_position2'])






        if is_portfolio:

            print("symbol = " + symbol)
            print("Old length = " + str(data_df.shape[0]))

            data_df = data_df[(data_df['time'] >= start_time) & (data_df['time'] <= end_time)]

            print("New length = " + str(data_df.shape[0]))

        data_df = data_df[['time','id','buy_point_id', 'sell_point_id', 'close', 'buy_position','cum_buy_position','sell_position','cum_sell_position',
                           'position', 'cum_position']]

        data_df['position'] = data_df['position'] * weight
        data_df['cum_position'] = data_df['cum_position'] * weight

        if is_portfolio and sample_data_df is None and data_df.shape[0] == final_data_df.shape[0]:
            sample_data_df = data_df.copy()
            print("Copy sample data_df")
            #sys.exit(0)

        # if symbol == 'AUDUSD':
        #     print("AUDUSD data_df length = " + str(data_df.shape[0]))
        #     print("final_data_df length = " + str(final_data_df.shape[0]))
            #sys.exit(0)

        if is_portfolio and sample_data_df is not None and data_df.shape[0] < final_data_df.shape[0]:
            data_df = pd.merge(sample_data_df[['time']], data_df, on = ['time'], how = 'outer')
            data_df = data_df.fillna(0)

            data_df.reset_index(inplace=True)
            data_df = data_df.drop(columns=['index'])

            #
            # print("Critical symbol = " + symbol)
            # print(data_df.head(300))
            # sys.exit(0)



        if is_portfolio:
            data_df['position'] = final_data_df[symbol+'_actual_position']
            data_df['cum_position'] = final_data_df[symbol+'_actual_cum_position']
            pnl_df[symbol + '_cum_position'] = data_df['cum_position']

            # if symbol == 'USDJPY':
            #     print("Check USDJPY:")
            #     print(data_df.iloc[1277:1287][['time','position','cum_position']])
            #     sys.exit(0)




        data_df['time_id'] = list(range(data_df.shape[0]))

        data_df['price'] = data_df['close']

        data_df['pre_pos'] = data_df['cum_position'].shift(1)
        data_df['pre_price'] = data_df['price'].shift(1)
        data_df['pre_pos'].iloc[0] = 0

        data_df['pnl'] = (data_df['price'] - data_df['pre_price']) * lot_size * data_df['pre_pos']

        if is_subtract_commission:
            data_df['pre_pos_increment'] = data_df['pre_pos'].diff()
            data_df['pre_pos_increment'].iloc[0] = 0
            data_df['pnl'] = np.where(data_df['pre_pos_increment'] > 0,
                                       data_df['pnl'] - spread * data_df['pre_pos_increment'],
                                       data_df['pnl']
                                      )


        if is_portfolio:
            pnl_df[symbol + '_pnl'] = data_df['pnl']

    else:

        data_df = final_data_df[['time', 'actual_cum_position']]
        data_df['pnl'] = 0
        for symbol in symbols:
            data_df['pnl'] += pnl_df[symbol + '_pnl']

        data_df['time_id'] = list(range(data_df.shape[0]))

        data_df = data_df.rename(columns = {'actual_cum_position' : 'cum_position'})


        deposit_per_lot = 1000
        principal = initial_principal * initial_principal_magnifier






    data_df['acc_pnl'] = data_df['pnl'].cumsum()
    data_df['acc_return'] = data_df['acc_pnl'] / float(principal)
    data_df['acc_return_bps'] = data_df['acc_return'] * 10000.0

    data_df['acc_pnl_hk'] = data_df['acc_pnl'] * initial_deposit_hk/principal

    lot_per_unit = round(initial_deposit_hk / (principal * 7.77), 2)




    pnl += [data_df.iloc[-1]['acc_pnl']]
    return_rate += [data_df.iloc[-1]['acc_return']]

    data_df['abs_cum_position'] = np.abs(data_df['cum_position'])

    data_df['abs_cum_position_hk'] = data_df['abs_cum_position'] * lot_per_unit


    #data_df['remaining_deposit'] = principal + data_df['acc_pnl'] - data_df['abs_cum_position'] * deposit_per_lot

    #data_df['remaining_deposit_pct'] = data_df['remaining_deposit'] / float(principal)

    data_df['equity'] = principal + data_df['acc_pnl']
    data_df['used_margin'] = data_df['abs_cum_position'] * deposit_per_lot

    temp_df = data_df[['abs_cum_position', 'equity', 'used_margin']]
    temp_df['margin_level'] = np.nan
    temp_df['margin_level'] = np.where(
        np.abs(temp_df['abs_cum_position']) < 1e-5,
        temp_df['margin_level'],
        temp_df['equity'] / temp_df['used_margin']
    )

    temp_df2 = temp_df.copy()

    # temp_df = temp_df.fillna(method='ffill').fillna(1)
    # temp_df2 = temp_df2.fillna(method='ffill').fillna(1e9)

    temp_df = temp_df.fillna(1)
    temp_df2 = temp_df2.fillna(1e9)

    data_df['margin_level'] = temp_df['margin_level']
    data_df['real_margin_level'] = temp_df2['margin_level']

    data_df['pre_margin_level'] = data_df['margin_level'].shift(1)
    data_df['cross_down_stop_level'] = np.where(
        (data_df['pre_margin_level'] > stop_level) & (data_df['margin_level'] <= stop_level),
        1,
        0
    )
    data_df['cross_up_stop_level'] = np.where(
        (data_df['pre_margin_level'] <= stop_level) & (data_df['margin_level'] > stop_level),
        1,
        0
    )

    cross_down_points = which(data_df['cross_down_stop_level'] == 1)
    cross_up_points = which(data_df['cross_up_stop_level'] == 1)

    min_margin_levels += [data_df['real_margin_level'].min()]





    data_df['pre_cum_position'] = data_df['cum_position'].shift(1)
    data_df['create_position'] = np.where(
        (np.abs(data_df['pre_cum_position']) <= 1e-5) & (np.abs(data_df['cum_position']) > 1e-5),
        1,
        0
    )

    data_df['create_position_temp'] = np.nan
    data_df['create_position_temp'] = np.where(
        data_df['create_position'] == 1,
        data_df['time_id'],
        data_df['create_position_temp']
    )

    temp_df = data_df[['time_id', 'create_position_temp', 'acc_pnl']]
    acc_pnl_df = data_df[data_df['create_position_temp'].notnull()][['time_id', 'acc_pnl']]
    acc_pnl_df.reset_index(inplace=True)
    acc_pnl_df = acc_pnl_df.drop(columns=['index'])

    temp_df = temp_df.drop(columns = ['acc_pnl'])
    temp_df = temp_df.fillna(method = 'ffill').fillna(0)


    temp_df['time_id'] = temp_df['create_position_temp']
    temp_df = pd.merge(temp_df, acc_pnl_df, on = ['time_id'], how = 'left')
    temp_df = temp_df.fillna(0)

    data_df['acc_pnl_for_restart'] = temp_df['acc_pnl']
    data_df['acc_pnl_restart'] = data_df['acc_pnl'] - data_df['acc_pnl_for_restart']


    data_df['restart_equity'] = principal + data_df['acc_pnl_restart']
    #data_df['used_margin'] = data_df['abs_cum_position'] * deposit_per_lot

    temp_df = data_df[['abs_cum_position', 'restart_equity', 'used_margin']]
    temp_df['restart_margin_level'] = np.nan
    temp_df['restart_margin_level'] = np.where(
        np.abs(temp_df['abs_cum_position']) < 1e-5,
        temp_df['restart_margin_level'],
        temp_df['restart_equity'] / temp_df['used_margin']
    )

    temp_df2 = temp_df.copy()

    # temp_df = temp_df.fillna(method='ffill').fillna(1)
    # temp_df2 = temp_df2.fillna(method='ffill').fillna(1e9)

    temp_df = temp_df.fillna(1)
    temp_df2 = temp_df2.fillna(1e9)

    data_df['restart_margin_level'] = temp_df['restart_margin_level']
    data_df['restart_real_margin_level'] = temp_df2['restart_margin_level']

    data_df['pre_restart_margin_level'] = data_df['restart_margin_level'].shift(1)
    data_df['restart_cross_down_stop_level'] = np.where(
        (data_df['pre_restart_margin_level'] > stop_level) & (data_df['restart_margin_level'] <= stop_level),
        1,
        0
    )
    data_df['restart_cross_up_stop_level'] = np.where(
        (data_df['pre_restart_margin_level'] <= stop_level) & (data_df['restart_margin_level'] > stop_level),
        1,
        0
    )

    restart_cross_down_points = which(data_df['restart_cross_down_stop_level'] == 1)
    restart_cross_up_points = which(data_df['restart_cross_up_stop_level'] == 1)

    restart_min_margin_levels += [data_df['restart_real_margin_level'].min()]









    ######## Calculate Max-drawdown ###########

    data_df['max_acc_pnl'] = data_df['acc_pnl'].cummax()
    data_df['drawdown'] = data_df['max_acc_pnl'] - data_df['acc_pnl']
    symbol_max_drawdown = data_df['drawdown'].max()

    max_drawdown_end = np.array(data_df.iloc[1:]['drawdown']).argmax()+1

    # print(np.array(data_df.iloc[-10:-1]['drawdown']))
    #
    # print("symbol_max_drawdown = " + str(symbol_max_drawdown))
    # print("max_drawdown_end = " + str(max_drawdown_end))
    # print("critical max_acc_pnl = " + str(data_df.iloc[max_drawdown_end]['max_acc_pnl']))

    print("symbol = " + symbol)
    print("data_df length = " + str(data_df.shape[0]))
    print("max_drawdown_end = " + str(max_drawdown_end))

    # if i == meta_df.shape[0]:
    #     pnl_file = os.path.join(data_folder, 'portfolio', 'data', 'portfolio' + '_pnl_debug.csv')
    #     data_df.to_csv(pnl_file, index = False)

    max_drawdown_start = which(np.abs(data_df['acc_pnl'] - data_df.iloc[max_drawdown_end]['max_acc_pnl']) < 1e-5)[0]

    print("max_drawdown_start = " + str(max_drawdown_start))
    #
    # sys.exit(0)


    max_drawdown += [symbol_max_drawdown]
    max_drawdown_rate += [symbol_max_drawdown/float(principal)]



    ##############

    create_position_ids = which(data_df.iloc[0:max_drawdown_start]['create_position'] == 1)
    if len(create_position_ids) > 0:
        start_create_position_id = create_position_ids[-1]
    else:
        start_create_position_id = 0

    acc_pnl_to_subtract = 0
    if start_create_position_id > 0:
        acc_pnl_to_subtract = data_df.iloc[start_create_position_id]['acc_pnl']
    data_df['acc_pnl_drawdown'] = data_df['acc_pnl'] - acc_pnl_to_subtract
    data_df['acc_pnl_drawdown'] = np.where(
        data_df['time_id'] < start_create_position_id,
        0,
        data_df['acc_pnl_drawdown']
    )

    data_df['abs_cum_position_drawdown'] = np.where(
        data_df['time_id'] < start_create_position_id,
        0,
        data_df['abs_cum_position']
    )


    data_df['drawdown_equity'] = principal + data_df['acc_pnl_drawdown']
    data_df['drawdown_used_margin'] = data_df['abs_cum_position_drawdown'] * deposit_per_lot

    temp_df = data_df[['abs_cum_position_drawdown', 'drawdown_equity', 'drawdown_used_margin']]
    temp_df['drawdown_margin_level'] = np.nan
    temp_df['drawdown_margin_level'] = np.where(
        np.abs(temp_df['abs_cum_position_drawdown']) < 1e-5,
        temp_df['drawdown_margin_level'],
        temp_df['drawdown_equity'] / temp_df['drawdown_used_margin']
    )

    temp_df2 = temp_df.copy()

    # temp_df = temp_df.fillna(method='ffill').fillna(1)
    # temp_df2 = temp_df2.fillna(method='ffill').fillna(1e9)

    temp_df = temp_df.fillna(1)
    temp_df2 = temp_df2.fillna(1e9)

    data_df['drawdown_margin_level'] = temp_df['drawdown_margin_level']
    data_df['drawdown_real_margin_level'] = temp_df2['drawdown_margin_level']

    data_df['drawdown_pre_margin_level'] = data_df['drawdown_margin_level'].shift(1)
    data_df['drawdown_cross_down_stop_level'] = np.where(
        (data_df['drawdown_pre_margin_level'] > stop_level) & (data_df['drawdown_margin_level'] <= stop_level),
        1,
        0
    )
    data_df['drawdown_cross_up_stop_level'] = np.where(
        (data_df['drawdown_pre_margin_level'] <= stop_level) & (data_df['drawdown_margin_level'] > stop_level),
        1,
        0
    )

    drawdown_cross_down_points = which(data_df['drawdown_cross_down_stop_level'] == 1)
    drawdown_cross_up_points = which(data_df['drawdown_cross_up_stop_level'] == 1)

    drawdown_min_margin_levels += [data_df['drawdown_real_margin_level'].min()]




    ########### Calculate intraday pnl and floating pnl ###################

    if i <= meta_df.shape[0]:
        data_df['date'] = pd.DatetimeIndex(data_df['time']).normalize()

        data_df['date'] = data_df['date'].shift(2)
        data_df.at[0, 'date'] = data_df.iloc[2]['date']
        data_df.at[1, 'date'] = data_df.iloc[2]['date']
        #
        # data_df['date'] = data_df['date'].shift(1)
        # data_df.at[0, 'date'] = data_df.iloc[1]['date']

        data_df['prev_acc_pnl'] = data_df['acc_pnl'].shift(1)

        acc_pnl_df = data_df[['date', 'acc_pnl', 'prev_acc_pnl']]
        acc_pnl_df = acc_pnl_df.fillna(0)

        acc_pnl_df_summary = acc_pnl_df.groupby(['date']).agg(
            {
                'acc_pnl' : 'last',
                'prev_acc_pnl' : 'first'
            }
        )

        acc_pnl_df_summary = acc_pnl_df_summary.rename(columns = {
            'prev_acc_pnl' : 'last_day_acc_pnl',
            'acc_pnl' : 'current_day_acc_pnl'
        })

        acc_pnl_df_summary['daily_pnl'] = acc_pnl_df_summary['current_day_acc_pnl'] - acc_pnl_df_summary['last_day_acc_pnl']

        # print("acc_pnl_df_summary:")
        # print(acc_pnl_df_summary)

        acc_pnl_df_summary.reset_index(inplace=True)

        # print("acc_pnl_df_summary2:")
        # print(acc_pnl_df_summary)
        #acc_pnl_df_summary = acc_pnl_df_summary.drop(columns=['index'])

        data_df = pd.merge(data_df, acc_pnl_df_summary[['date', 'last_day_acc_pnl']], on = ['date'], how = 'left')
        data_df['intraday_acc_pnl'] = data_df['acc_pnl'] - data_df['last_day_acc_pnl']

        if i < meta_df.shape[0] and is_portfolio: ######################################IMportant  IMportant IMportant IMportant IMportant
            pnl_df[symbol + '_intraday_acc_pnl'] = data_df['intraday_acc_pnl']

        intraday_pnl_df = data_df[['date', 'intraday_acc_pnl']]
        intraday_pnl_df['intraday_min_pnl'] = intraday_pnl_df['intraday_acc_pnl']
        intraday_pnl_df['intraday_max_pnl'] = intraday_pnl_df['intraday_acc_pnl']
        intraday_pnl_df['intraday_final_pnl'] = intraday_pnl_df['intraday_acc_pnl']

        intraday_pnl_df_summary = intraday_pnl_df.groupby(['date']).agg(
            {
                'intraday_min_pnl' : 'min',
                'intraday_max_pnl' : 'max',
                'intraday_final_pnl' : 'last'
            }
        )

        intraday_pnl_df_summary.reset_index(inplace=True)

        data_df = pd.merge(data_df, intraday_pnl_df_summary[['date', 'intraday_min_pnl', 'intraday_max_pnl', 'intraday_final_pnl']], on = ['date'], how = 'left')

        #data_df['acc_pnl_hk'] = data_df['acc_pnl'] * initial_deposit_hk / principal

        data_df['intraday_acc_pnl_hk'] = data_df['intraday_acc_pnl'] * initial_deposit_hk / principal
        data_df['intraday_min_pnl_hk'] = data_df['intraday_min_pnl'] * initial_deposit_hk / principal
        data_df['intraday_max_pnl_hk'] = data_df['intraday_max_pnl'] * initial_deposit_hk / principal
        data_df['intraday_final_pnl_hk'] = data_df['intraday_final_pnl'] * initial_deposit_hk / principal

        data_df['is_intraday_min_pnl'] = np.abs(data_df['intraday_acc_pnl'] - data_df['intraday_min_pnl']) < 1e-5
        data_df['is_intraday_max_pnl'] = np.abs(data_df['intraday_acc_pnl'] - data_df['intraday_max_pnl']) < 1e-5


        intraday_pnl_df_summary['intraday_min_pnl_hk'] = intraday_pnl_df_summary['intraday_min_pnl'] * initial_deposit_hk / principal
        intraday_pnl_df_summary['intraday_max_pnl_hk'] = intraday_pnl_df_summary['intraday_max_pnl'] * initial_deposit_hk / principal
        intraday_pnl_df_summary['intraday_final_pnl_hk'] = intraday_pnl_df_summary['intraday_final_pnl'] * initial_deposit_hk / principal



        intraday_min_pnl_df = data_df[data_df['is_intraday_min_pnl']][['date', 'time', 'intraday_acc_pnl_hk', 'intraday_min_pnl_hk', 'intraday_max_pnl_hk', 'intraday_final_pnl_hk']]
        intraday_max_pnl_df = data_df[data_df['is_intraday_max_pnl']][['date', 'time', 'intraday_acc_pnl_hk', 'intraday_min_pnl_hk', 'intraday_max_pnl_hk', 'intraday_final_pnl_hk']]

        intraday_min_pnl_df['count'] = 1
        intraday_min_pnl_df_summary = intraday_min_pnl_df.groupby(['date']).agg({
            'time' : 'first',
            'intraday_acc_pnl_hk' : 'first',
            'intraday_min_pnl_hk' : 'first',
            'intraday_max_pnl_hk' : 'first',
            'intraday_final_pnl_hk' : 'first',
            'count' : 'count'
        })
        intraday_min_pnl_df_summary.reset_index(inplace = True)
        intraday_min_pnl_df_summary = intraday_min_pnl_df_summary[intraday_min_pnl_df_summary['count'] == 1]

        intraday_max_pnl_df['count'] = 1
        intraday_max_pnl_df_summary = intraday_max_pnl_df.groupby(['date']).agg({
            'time' : 'first',
            'intraday_acc_pnl_hk' : 'first',
            'intraday_min_pnl_hk' : 'first',
            'intraday_max_pnl_hk' : 'first',
            'intraday_final_pnl_hk' : 'first',
            'count' : 'count'
        })
        intraday_max_pnl_df_summary.reset_index(inplace = True)
        intraday_max_pnl_df_summary = intraday_max_pnl_df_summary[intraday_max_pnl_df_summary['count'] == 1]

        intraday_min_pnl_df_summary_temp = intraday_min_pnl_df_summary[['date', 'time', 'intraday_min_pnl_hk', 'intraday_final_pnl_hk']]
        intraday_min_pnl_df_summary_temp = intraday_min_pnl_df_summary_temp.rename(columns = {
            'time' : 'min_pnl_time',
            'intraday_final_pnl_hk' : 'intraday_final_pnl_hk1'
        })

        intraday_max_pnl_df_summary_temp = intraday_max_pnl_df_summary[['date', 'time', 'intraday_max_pnl_hk', 'intraday_final_pnl_hk']]
        intraday_max_pnl_df_summary_temp = intraday_max_pnl_df_summary_temp.rename(columns = {
            'time' : 'max_pnl_time',
            'intraday_final_pnl_hk' : 'intraday_final_pnl_hk2'
        })

        merged_intraday_pnl_summary = pd.merge(intraday_min_pnl_df_summary_temp, intraday_max_pnl_df_summary_temp, on = ['date'], how = 'outer')
        merged_intraday_pnl_summary['intraday_final_pnl_hk'] = np.where(
            merged_intraday_pnl_summary['intraday_final_pnl_hk1'].notnull(),
            merged_intraday_pnl_summary['intraday_final_pnl_hk1'],
            merged_intraday_pnl_summary['intraday_final_pnl_hk2']
        )
        merged_intraday_pnl_summary = merged_intraday_pnl_summary.drop(columns = ['intraday_final_pnl_hk1', 'intraday_final_pnl_hk2'])



        if i < meta_df.shape[0]:
            min_pnl_file = os.path.join(data_folder, symbol, 'data', symbol + '_min_pnl.csv')
        else:
            min_pnl_file = os.path.join(pnl_folder, 'portfolio_min_pnl.csv')


        intraday_min_pnl_df_summary.to_csv(min_pnl_file, index = False)

        #max_pnl_file = os.path.join(pnl_folder, 'portfolio_max_pnl.csv')

        if i < meta_df.shape[0]:
            max_pnl_file = os.path.join(data_folder, symbol, 'data', symbol + '_max_pnl.csv')
        else:
            max_pnl_file = os.path.join(pnl_folder, 'portfolio_min_pnl.csv')

        intraday_max_pnl_df_summary.to_csv(max_pnl_file, index = False)


        if i == meta_df.shape[0]:
            final_pnl_file = os.path.join(pnl_folder, 'portfolio_daily_final_pnl.csv')
            intraday_pnl_df_summary[['date', 'intraday_min_pnl_hk', 'intraday_max_pnl_hk', 'intraday_final_pnl_hk']].to_csv(final_pnl_file, index = False)

        merged_intraday_pnl_summary = merged_intraday_pnl_summary.sort_values(by = ['date'])


        if i < meta_df.shape[0]:
            min_max_file = os.path.join(data_folder, symbol, 'data', symbol + '_min_max_pnl.csv')
        else:
            min_max_file = os.path.join(pnl_folder, 'portfolio_min_max_pnl.csv')

        merged_intraday_pnl_summary[['date', 'min_pnl_time','max_pnl_time','intraday_min_pnl_hk',
                                     'intraday_max_pnl_hk', 'intraday_final_pnl_hk']].to_csv(min_max_file, index = False)


        simple_data_df = data_df[['time', 'date',  'cum_position', 'intraday_acc_pnl_hk']]
        simple_data_df['cum_position_hk_calc'] = simple_data_df['cum_position'] * lot_per_unit

        simple_data_df['cum_position_hk_calc'] = simple_data_df['cum_position_hk_calc'].apply(lambda x : round(x, 2))

        simple_data_df = simple_data_df.drop(columns = ['cum_position'])

        # if i == meta_df.shape[0]:
        #     simple_data_df['intraday_acc_pnl_per_position_hk'] = np.where(
        #         np.abs(simple_data_df['cum_position_hk']) < 1e-5,
        #         0,
        #         simple_data_df['intraday_acc_pnl_hk'] / (simple_data_df['cum_position_hk']/lot_per_unit)
        #     )


        if i == meta_df.shape[0]:
            simple_data_df['cum_position_hk'] = 0.0
            for symbol in symbols:
                simple_data_df[symbol + '_intraday_acc_pnl_hk'] = pnl_df[symbol + '_intraday_acc_pnl'] * initial_deposit_hk / principal
                simple_data_df[symbol + '_cum_position_hk'] = pnl_df[symbol + '_cum_position'] * lot_per_unit

                #simple_data_df[symbol + '_cum_position'] = pnl_df[symbol + '_cum_position'] ## Temp

                # if symbol == 'USDJPY' or symbol == 'EURUSD':
                #     print("Here check data:")
                #     print("pnl_df:")
                #     print(pnl_df[['time',symbol + '_intraday_acc_pnl', symbol + '_cum_position']].tail(100))
                #
                #     print("simple_data_df:")
                #     print(simple_data_df[['time', symbol + '_intraday_acc_pnl_hk', symbol + '_cum_position_hk']].tail(100))
                #
                #     if symbol == 'USDJPY':
                #         sys.exit(0)
                simple_data_df[symbol + '_cum_position_hk_raw'] = simple_data_df[symbol + '_cum_position_hk']
                simple_data_df[symbol + '_cum_position_hk'] = simple_data_df[symbol + '_cum_position_hk'].apply(lambda x: round(x, 2))

                simple_data_df['cum_position_hk'] += np.abs(simple_data_df[symbol + '_cum_position_hk'])

        if i == meta_df.shape[0]:
            simple_data_df['intraday_acc_pnl_per_position_hk'] = np.where(
                np.abs(simple_data_df['cum_position_hk']) < 1e-5,
                0,
                simple_data_df['intraday_acc_pnl_hk'] / (simple_data_df['cum_position_hk']/lot_per_unit)
            )


        if i < meta_df.shape[0]:
            simple_file = os.path.join(data_folder, symbol, 'data', symbol + '_intraday_pnl.csv')
        else:
            simple_file = os.path.join(pnl_folder, 'portfolio_intraday_pnl.csv')


        if i == meta_df.shape[0] and is_send_email:

            last_row = simple_data_df.iloc[-1]
            # intraday_acc_pnl_hk
            # cum_position_hk

            # EURUSD_intraday_acc_pnl_hk
            # EURUSD_cum_position_hk intraday_acc_pnl_hk
            message_array = ['Symbol  Intraday Pnl($HK)  Position on hold(Lot)']
            message_array += ['Portfolio  ' + str(round(last_row['intraday_acc_pnl_hk'], 2)) + '  ' + str(round(last_row['cum_position_hk'], 2))]

            for symbol in symbols:
                symbol_intraday_acc_pnl = round(last_row[symbol + '_intraday_acc_pnl_hk'], 2)
                symbol_cum_position = round(last_row[symbol + '_cum_position_hk'], 2)

                if abs(symbol_intraday_acc_pnl) > 1e-5 or abs(symbol_cum_position) > 1e-5:
                    message_array += [symbol + '  ' + str(symbol_intraday_acc_pnl) + '  ' + str(symbol_cum_position)]

            message_body = '\n'.join(message_array)

            print("")
            print("Portfolio message:")
            print(message_body)





            ###### Temp ##########

            # message_array = ['Symbol  Intraday Pnl  Position on hold(Lot)']
            # message_array += ['Portfolio  ' + str(round(last_row['intraday_acc_pnl_hk'], 2)) + '  ' + str(
            #     round(last_row['cum_position'], 2))]
            #
            # for symbol in symbols:
            #     symbol_intraday_acc_pnl = round(last_row[symbol + '_intraday_acc_pnl_hk'], 2)
            #     symbol_cum_position = round(last_row[symbol + '_cum_position'], 2)
            #
            #     if abs(symbol_intraday_acc_pnl) > 1e-5 or abs(symbol_cum_position) > 1e-5:
            #         message_array += [symbol + '  ' + str(symbol_intraday_acc_pnl) + '  ' + str(symbol_cum_position)]
            #
            # message_body = '\n'.join(message_array)
            #
            # print("")
            # print("Portfolio message:")
            # print(message_body)

            ######################






            sendEmail("Portfolio Current Status at " + current_time, message_body)

            ############### Draw intraday pnl change figure ###################
            intraday_df = simple_data_df[['time','date','intraday_acc_pnl_hk', 'cum_position_hk', 'intraday_acc_pnl_per_position_hk']]
            unique_dates = list(intraday_df['date'].unique())
            intraday_pnl_folder = os.path.join(pnl_folder, 'intraday_pnl')
            if not os.path.exists(intraday_pnl_folder):
                os.makedirs(intraday_pnl_folder)

            for unique_date in unique_dates:

                intraday_date_df = intraday_df[intraday_df['date'] == unique_date]
                intraday_date_df['time_id'] = list(range(intraday_date_df.shape[0]))
                intraday_date_df = intraday_date_df.drop(columns = ['date', 'cum_position_hk'])
                intraday_date_df['intraday_max_acc_pnl_hk'] = intraday_date_df['intraday_acc_pnl_hk'].cummax()


                date_str = intraday_date_df.iloc[0]['time'].strftime('%y-%m%d')

                intraday_date_df['raw_time'] = intraday_date_df['time']
                intraday_date_df['time'] = intraday_date_df['time'].apply(lambda x: x.strftime('%H:%M'))
                intraday_date_df['time'] = np.where(
                    intraday_date_df['time'] == '00:00',
                    '24:00',
                    intraday_date_df['time']
                )

                if draw_intraday_pnl:
                    print("Draw intraday pnl change for date " + date_str)

                # print("intraday_date_df:")
                # print(intraday_date_df)

                y_attrs = ['intraday_acc_pnl_per_position_hk', 'intraday_max_acc_pnl_hk', 'intraday_acc_pnl_hk']
                dfs = []
                for y_attr in y_attrs:
                    single_df = intraday_date_df[['time', 'time_id', y_attr]]
                    single_df = single_df.rename(columns = {y_attr : 'pnl'})
                    single_df['type'] = y_attr
                    dfs += [single_df]

                intraday_final_acc_pnl_df = pd.concat(dfs)

                ##Hutong
                if draw_intraday_pnl:
                    fig = plt.figure(figsize=(28, 18))

                    intraday_time_number = intraday_date_df.shape[0]
                    intraday_trade_times = intraday_date_df['time']
                    tick_interval = 1
                    font_size = 20
                    angle = 30

                    def format_intraday_time(x, pos=None):
                        thisind = np.clip(int(x + 0.5), 0, intraday_time_number - 1)
                        return intraday_trade_times[thisind].strftime('%H:%M')

                    axes = fig.subplots(nrows=1, ncols=1)

                    # print("intraday_final_acc_pnl_df:")
                    # print(intraday_final_acc_pnl_df)

                    sns.lineplot(x='time', y='pnl', hue = 'type', data=intraday_final_acc_pnl_df, ax=axes, linewidth = 6)
                    axes.set_title('FX ' + "Portfolio" + " Pnl " + date_str, fontsize=18)
                    axes.set_xlabel('time', size=font_size)
                    axes.set_ylabel('pnl', size=font_size)
                    #
                    axes.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
                    #axes.xaxis.set_major_formatter(ticker.FuncFormatter(format_intraday_time))

                    #axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                    axes.tick_params(labelsize=font_size)

                    axes.axhline(0, ls='--', color='black', linewidth=1)

                    axes.axhline(250, ls='--', color='red', linewidth=1)

                    axes.axhline(300, ls='--', color='blue', linewidth=1)

                    plt.setp(axes.get_xticklabels(), rotation=angle)

                    fig.savefig(os.path.join(intraday_pnl_folder, 'intraday_pnl_' + date_str + '.png'))

                    plt.close(fig)
                    #sys.exit(0)







        simple_data_df.to_csv(simple_file, index = False)



    ###########################################


    if i == meta_df.shape[0]:
        portfolio_folder = os.path.join(data_folder, 'portfolio', 'data')
        if not os.path.exists(portfolio_folder):
            os.makedirs(portfolio_folder)
        pnl_file = os.path.join(pnl_folder, 'portfolio' + '_pnl_final.csv')

    else:
        pnl_file = os.path.join(data_folder, symbol, 'data', symbol + '_pnl.csv')

    data_df.to_csv(pnl_file, index = False)


    ################# Draw figure ###############


    if draw_figure:

        time_number = data_df.shape[0]
        trade_times = data_df['time']

        if i == meta_df.shape[0]:
            trade_dates = intraday_pnl_df_summary['date']
            date_number = intraday_pnl_df_summary.shape[0]


        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, time_number - 1)
            return trade_times[thisind].strftime('%y%m%d-%H')

        def format_actual_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, date_number - 1)
            return trade_dates[thisind].strftime('%y%m%d')

        tick_number = 20
        tick_interval = time_number / tick_number

        if i == meta_df.shape[0]:
            tick_number2 = 20
            tick_interval2 = date_number / tick_number2


        types = ['pct', 'non_pct']

        for type in types:

            fig = plt.figure(figsize=(28, 18))
            col_num = 1
            #row_num = 3 if i == meta_df.shape[0] else 4
            row_num = 4

            angle = 30

            font_size = 14

            axes = fig.subplots(nrows=row_num, ncols=col_num)

            #start_id = -1 if i == meta_df.shape[0] else 0

            start_id = 0

            if i < meta_df.shape[0]:
                ## Figure 1: market price curve and our buy/sell points
                data_df['price'] = np.where(
                    np.abs(data_df['price']) < 1e-5,
                    np.nan,
                    data_df['price']
                )

                sns.lineplot(x='time_id', y='price', color='black', data=data_df, ax=axes[0])
                axes[0].set_title('FX ' + symbol + " Market Price", fontsize=18)
                axes[0].set_xlabel('time', size=font_size)
                axes[0].set_ylabel('price', size=font_size)
                axes[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
                axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
                #axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                axes[0].tick_params(labelsize=font_size)

                plt.setp(axes[0].get_xticklabels(), rotation=angle)

            else:

                if plot_hk_pnl:
                    intraday_final_pnl = 'intraday_final_pnl_hk'
                    intraday_min_pnl = 'intraday_min_pnl_hk'
                    intraday_max_pnl = 'intraday_max_pnl_hk'
                    intraday_pnl = 'intraday_pnl_hk'
                    pnl_attr = 'pnl_hk'
                else:
                    intraday_final_pnl = 'intraday_final_pnl'
                    intraday_min_pnl = 'intraday_min_pnl'
                    intraday_max_pnl = 'intraday_max_pnl'
                    intraday_pnl = 'intraday_pnl'
                    pnl_attr = 'pnl'

                intraday_pnl_df_summary['time_id'] = list(range(intraday_pnl_df_summary.shape[0]))

                intraday_final_pnl_df = intraday_pnl_df_summary[['time_id', 'date', intraday_final_pnl]]
                intraday_min_pnl_df = intraday_pnl_df_summary[['time_id', 'date', intraday_min_pnl]]
                intraday_max_pnl_df = intraday_pnl_df_summary[['time_id', 'date', intraday_max_pnl]]

                # print("before intraday_final_pnl_df:")
                # print(intraday_final_pnl_df.head(10))

                intraday_final_pnl_df = intraday_final_pnl_df.rename(columns = {intraday_final_pnl : intraday_pnl})
                intraday_final_pnl_df[pnl_attr] = intraday_final_pnl

                # print("after intraday_final_pnl_df:")
                # print(intraday_final_pnl_df.head(10))



                intraday_min_pnl_df = intraday_min_pnl_df.rename(columns = {intraday_min_pnl : intraday_pnl})
                intraday_min_pnl_df[pnl_attr] = intraday_min_pnl

                intraday_max_pnl_df = intraday_max_pnl_df.rename(columns={intraday_max_pnl: intraday_pnl})
                intraday_max_pnl_df[pnl_attr] = intraday_max_pnl

                intraday_summary = pd.concat([intraday_min_pnl_df, intraday_max_pnl_df, intraday_final_pnl_df])

                # print("intraday_summary:")
                # print(intraday_summary)

                sns.lineplot(x='time_id', y=intraday_pnl, hue = pnl_attr, color='black', data=intraday_summary, ax=axes[0])
                axes[0].set_title('FX ' + "Portfolio" + " Intraday Pnl", fontsize=18)
                axes[0].set_xlabel('time', size=font_size)
                axes[0].set_ylabel(pnl_attr, size=font_size)
                axes[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval2))
                axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_actual_date))
                #axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                axes[0].tick_params(labelsize=font_size)

                plt.setp(axes[0].get_xticklabels(), rotation=angle)



            if i == meta_df.shape[0]:
                ticker_name = "Portfolio"
            else:
                ticker_name = symbol

            ## Figure 2: mark-to-market pnl curve
            y_attr = 'acc_return' if type == 'pct' else 'acc_pnl'
            if type != 'pct' and plot_hk_pnl:
                y_attr = 'acc_pnl_hk'

            sns.lineplot(x = 'time_id', y = y_attr, markers = 'o', color = 'red', data = data_df, ax = axes[start_id + 1])
            axes[start_id + 1].set_title('FX ' + ticker_name + " Strategy Return ", fontsize = 18)
            axes[start_id + 1].set_xlabel('time', size = font_size)
            axes[start_id + 1].set_ylabel(y_attr, size = font_size)
            axes[start_id + 1].tick_params(labelsize = font_size)
            axes[start_id + 1].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            axes[start_id + 1].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))



            if type == 'pct':
                axes[start_id + 1].axvline(max_drawdown_start, ls='--', color='blue', linewidth=1)
                axes[start_id + 1].axvline(max_drawdown_end, ls='--', color='blue', linewidth=1)

                axes[start_id + 1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

            plt.setp(axes[start_id + 1].get_xticklabels(), rotation = angle)


            ## Figure 3: remaining deposit change curve
            # y_attr = 'remaining_deposit_pct' if type == 'pct' else 'remaining_deposit'
            # y_attr_simple = 'deposit_pct' if type == 'pct' else 'deposit'
            # max_deposit = data_df[y_attr].max()
            # min_deposit = data_df[y_attr].min()
            #
            # #print("min_deposit = " + str(min_deposit))
            #
            #
            # sns.lineplot(x = 'time_id', y = y_attr, color = 'blue', data = data_df, ax = axes[2])
            # axes[2].set_title('FX ' + symbol + " Remaining Deposit ", fontsize = 18)
            # axes[2].set_xlabel('time', size = font_size)
            # axes[2].set_ylabel(y_attr_simple, size = font_size)
            # axes[2].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            # axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
            # axes[2].tick_params(labelsize = font_size)
            #
            # #y_lim_min = min([0, min_deposit])
            # y_lim_min = min_deposit
            # y_lim_max = max([0, max_deposit])
            # if y_lim_max > 0:
            #     y_lim_max = y_lim_max * 1.2
            # axes[2].set_ylim([y_lim_min, y_lim_max])
            #
            # if type == 'pct':
            #     axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            #
            # plt.setp(axes[2].get_xticklabels(), rotation = angle)



            ##### Figure 3: margine_level change curve (Very important to monitor auto-stop) ############

            y_attr = 'drawdown_margin_level' if type == 'pct' else 'margin_level'
            max_margin_level = data_df[y_attr].max()
            min_margin_level = min(0, data_df[y_attr].min())

            # print("min_deposit = " + str(min_deposit))

            sns.lineplot(x='time_id', y=y_attr, color='blue', data=data_df, ax=axes[start_id + 2])
            title = ('FX ' + ticker_name + " Drawdown Margin Level ") if type == 'pct' else ('FX ' + symbol + " Margin Level ")
            axes[start_id + 2].set_title(title, fontsize=18)
            axes[start_id + 2].set_xlabel('time', size=font_size)
            axes[start_id + 2].set_ylabel(y_attr, size=font_size)
            axes[start_id + 2].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            axes[start_id + 2].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
            axes[start_id + 2].tick_params(labelsize=font_size)

            used_cross_down_points = drawdown_cross_down_points if type == 'pct' else cross_down_points
            used_cross_up_points = drawdown_cross_up_points if type == 'pct' else cross_up_points

            axes[start_id + 2].axhline(0.5, ls='--', color='black', linewidth=1)
            for point_id in used_cross_down_points:
                axes[start_id + 2].axvline(point_id, ls = '--', color = 'red', linewidth=1)
            for point_id in used_cross_up_points:
                axes[start_id + 2].axvline(point_id, ls = '--', color = 'blue', linewidth=1)


            # y_lim_min = min([0, min_deposit])
            y_lim_min = min_margin_level
            y_lim_max = max([0, max_margin_level])
            if y_lim_max > 0:
                y_lim_max = y_lim_max * 1.2
            axes[start_id + 2].set_ylim([y_lim_min, y_lim_max])

            #if type == 'pct':
            axes[start_id + 2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

            plt.setp(axes[start_id + 2].get_xticklabels(), rotation=angle)




            ## Figure 4: position change curve

            if plot_hk_pnl:
                position_attr = 'abs_cum_position_hk'
            else:
                position_attr = 'abs_cum_position'

            max_position = data_df[position_attr].max()
            sns.lineplot(x = 'time_id', y = position_attr, color = 'purple', data = data_df, ax = axes[start_id + 3])
            axes[start_id + 3].set_title('FX ' + ticker_name + " Position ", fontsize = 18)
            axes[start_id + 3].set_xlabel('time', size = font_size)
            axes[start_id + 3].set_ylabel('position', size = font_size)
            axes[start_id + 3].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            axes[start_id + 3].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
            axes[start_id + 3].tick_params(labelsize = font_size)
            axes[start_id + 3].set_ylim([-0.5, max_position * 2])
            plt.setp(axes[start_id + 3].get_xticklabels(), rotation = angle)


            plt.subplots_adjust(hspace = 0.5)

            if i < meta_df.shape[0]:

                figure_file_path = os.path.join(data_folder, symbol, 'data', symbol + '_' + type + '_pnl.png')
                print("figure_file_path:")
                print(figure_file_path)

                fig.savefig(figure_file_path)

            prefix = symbol if i < meta_df.shape[0] else "Portfolio"

            summary_path = os.path.join(pnl_folder, prefix + '_' + type + '_pnl.png')
            print("summary_path:")
            print(summary_path)
            fig.savefig(summary_path)

            plt.close(fig)

if is_portfolio:
    symbols += ['Portfolio']

print("symbols:")
print(symbols)
print("symbols = " + str(len(symbols)))
print("min_margin_levels = " + str(len(min_margin_levels)))
print("drawdown_min_margin_levels = " + str(len(drawdown_min_margin_levels)))
print("pnl = " + str(len(pnl)))
print("return_rate = " + str(len(return_rate)))
print("max_drawdown = " + str(len(max_drawdown)))
print("max_drawdown_rate = " + str(len(max_drawdown_rate)))


performance_summary = pd.DataFrame({'symbol' : symbols,  'min_margin_level(%)' : min_margin_levels, 'drawdown_min_margin_level(%)' : drawdown_min_margin_levels,
                                    'pnl' : pnl, 'return(%)' : return_rate, 'max_drawdown' : max_drawdown,
                                    'max_drawdown_rate(%)' : max_drawdown_rate})

performance_summary['min_margin_level(%)'] *= 100.0
performance_summary['drawdown_min_margin_level(%)'] *= 100.0
performance_summary['return(%)'] *= 100.0
performance_summary['max_drawdown_rate(%)'] *= 100.0

performance_summary['drawdown_adjusted_return'] = performance_summary['return(%)'] / performance_summary['max_drawdown_rate(%)']

print("performance_summary:")
print(performance_summary)
performance_summary.to_csv(os.path.join(pnl_folder, 'performance_summary.csv'), index = False)


if is_portfolio:
    config_df = pd.DataFrame({'initial_principal' : [initial_principal], 'max_exposure' : [max_exposure], 'initial_principal_magnifier' : [initial_principal_magnifier]})
    config_df.to_csv(os.path.join(pnl_folder, 'config.csv'), index = False)


auto_stop_summary = performance_summary[performance_summary['min_margin_level(%)'] <= stop_level * 100.0]
drawdown_auto_stop_summary = performance_summary[performance_summary['drawdown_min_margin_level(%)'] <= stop_level * 100.0]

print("")

print("These currency pairs will be forced to close all positions during trading due to margin_level below stop_level at some time")
print(auto_stop_summary)


print("These currency pairs will be forced to close all positions during trading (drawdown) due to margin_level below stop_level at some time")
print(drawdown_auto_stop_summary)

if is_portfolio:
    lot_per_unit = initial_deposit_hk/(principal*7.77)
    print("Under this model, you should enter " + str(round(lot_per_unit, 2)) + " lot per signal, and the maximum position is roughly restricted to " + str(round(lot_per_unit * max_exposure, 2)) + " lot")

















