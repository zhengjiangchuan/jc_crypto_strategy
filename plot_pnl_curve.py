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

data_folder = "C:\\Forex\\formal_trading\\"

meta_file = os.path.join(data_folder, 'symbols_meta.csv')
meta_df = pd.read_csv(meta_file)

if len(selected_symbols) > 0:
    meta_df = meta_df[meta_df['symbol'].isin(selected_symbols)]

pnl_folder = os.path.join(data_folder, 'pnl', 'pnl0720', 'pnl_summary_spread15_innovativeFire2new_marginLevel2.5_3pm_check2')
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

initial_principal = 2500




for i in range(meta_df.shape[0]):
    row = meta_df.iloc[i]
    symbol = row['symbol']
    exchange_rate = row['exchange_rate']
    principal = initial_principal
    deposit_per_lot = row['deposit_per_lot']
    contract_size = row['contract_size']
    spread = row['spread']

    symbols += [symbol]

    lot_size = contract_size * exchange_rate



    data_file = os.path.join(data_folder, symbol, 'data', symbol + '100.csv')

    data_df = pd.read_csv(data_file)



    data_df['time'] = data_df['time'].apply(lambda x: preprocess_time(x))

    data_df = data_df[['time','id','buy_point_id', 'sell_point_id', 'close', 'buy_position','cum_buy_position','sell_position','cum_sell_position',
                       'position', 'cum_position']]

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

    data_df['acc_pnl'] = data_df['pnl'].cumsum()
    data_df['acc_return'] = data_df['acc_pnl'] / float(principal)
    data_df['acc_return_bps'] = data_df['acc_return'] * 10000.0

    pnl += [data_df.iloc[-1]['acc_pnl']]
    return_rate += [data_df.iloc[-1]['acc_return']]

    data_df['abs_cum_position'] = np.abs(data_df['cum_position'])

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

    max_drawdown_start = which(np.abs(data_df['acc_pnl'] - data_df.iloc[max_drawdown_end]['max_acc_pnl']) < 1e-5)[0]

    # print("max_drawdown_start = " + str(max_drawdown_start))
    #
    # sys.exit(0)


    max_drawdown += [symbol_max_drawdown]
    max_drawdown_rate += [symbol_max_drawdown/float(principal)]



    ##############

    start_create_position_id = which(data_df.iloc[0:max_drawdown_start]['create_position'] == 1)[-1]
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





    ###########################################



    pnl_file = os.path.join(data_folder, symbol, 'data', symbol + '_pnl.csv')

    data_df.to_csv(pnl_file, index = False)


    ################# Draw figure ###############


    time_number = data_df.shape[0]
    trade_times = data_df['time']


    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, time_number - 1)
        return trade_times[thisind].strftime('%y%m%d-%H')

    tick_number = 20
    tick_interval = time_number / tick_number


    types = ['pct', 'non_pct']

    for type in types:

        fig = plt.figure(figsize=(28, 18))
        col_num = 1
        row_num = 4

        angle = 30

        font_size = 14

        axes = fig.subplots(nrows=row_num, ncols=col_num)

        ## Figure 1: market price curve and our buy/sell points
        sns.lineplot(x='time_id', y='price', color='black', data=data_df, ax=axes[0])
        axes[0].set_title('FX ' + symbol + " Market Price", fontsize=18)
        axes[0].set_xlabel('time', size=font_size)
        axes[0].set_ylabel('price', size=font_size)
        axes[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        #axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        axes[0].tick_params(labelsize=font_size)

        plt.setp(axes[0].get_xticklabels(), rotation=angle)


        ## Figure 2: mark-to-market pnl curve
        y_attr = 'acc_return' if type == 'pct' else 'acc_pnl'
        sns.lineplot(x = 'time_id', y = y_attr, markers = 'o', color = 'red', data = data_df, ax = axes[1])
        axes[1].set_title('FX ' + symbol + " Strategy Return ", fontsize = 18)
        axes[1].set_xlabel('time', size = font_size)
        axes[1].set_ylabel(y_attr, size = font_size)
        axes[1].tick_params(labelsize = font_size)
        axes[1].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))



        if type == 'pct':
            axes[1].axvline(max_drawdown_start, ls='--', color='blue', linewidth=1)
            axes[1].axvline(max_drawdown_end, ls='--', color='blue', linewidth=1)

            axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.setp(axes[1].get_xticklabels(), rotation = angle)


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

        sns.lineplot(x='time_id', y=y_attr, color='blue', data=data_df, ax=axes[2])
        title = ('FX ' + symbol + " Drawdown Margin Level ") if type == 'pct' else ('FX ' + symbol + " Margin Level ")
        axes[2].set_title(title, fontsize=18)
        axes[2].set_xlabel('time', size=font_size)
        axes[2].set_ylabel(y_attr, size=font_size)
        axes[2].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        axes[2].tick_params(labelsize=font_size)

        used_cross_down_points = drawdown_cross_down_points if type == 'pct' else cross_down_points
        used_cross_up_points = drawdown_cross_up_points if type == 'pct' else cross_up_points

        axes[2].axhline(0.5, ls='--', color='black', linewidth=1)
        for point_id in used_cross_down_points:
            axes[2].axvline(point_id, ls = '--', color = 'red', linewidth=1)
        for point_id in used_cross_up_points:
            axes[2].axvline(point_id, ls = '--', color = 'blue', linewidth=1)


        # y_lim_min = min([0, min_deposit])
        y_lim_min = min_margin_level
        y_lim_max = max([0, max_margin_level])
        if y_lim_max > 0:
            y_lim_max = y_lim_max * 1.2
        axes[2].set_ylim([y_lim_min, y_lim_max])

        #if type == 'pct':
        axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.setp(axes[2].get_xticklabels(), rotation=angle)




        ## Figure 4: position change curve
        max_position = data_df['abs_cum_position'].max()
        sns.lineplot(x = 'time_id', y = 'abs_cum_position', color = 'purple', data = data_df, ax = axes[3])
        axes[3].set_title('FX ' + symbol + " Position ", fontsize = 18)
        axes[3].set_xlabel('time', size = font_size)
        axes[3].set_ylabel('position', size = font_size)
        axes[3].xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        axes[3].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        axes[3].tick_params(labelsize = font_size)
        axes[3].set_ylim([-1, max_position * 2])
        plt.setp(axes[3].get_xticklabels(), rotation = angle)


        plt.subplots_adjust(hspace = 0.5)


        figure_file_path = os.path.join(data_folder, symbol, 'data', symbol + '_' + type + '_pnl.png')
        print("figure_file_path:")
        print(figure_file_path)


        fig.savefig(figure_file_path)

        summary_path = os.path.join(pnl_folder, symbol + '_' + type + '_pnl.png')
        print("summary_path:")
        print(summary_path)
        fig.savefig(summary_path)

        plt.close(fig)




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


auto_stop_summary = performance_summary[performance_summary['min_margin_level(%)'] <= stop_level * 100.0]
drawdown_auto_stop_summary = performance_summary[performance_summary['drawdown_min_margin_level(%)'] <= stop_level * 100.0]

print("")

print("These currency pairs will be forced to close all positions during trading due to margin_level below stop_level at some time")
print(auto_stop_summary)


print("These currency pairs will be forced to close all positions during trading (drawdown) due to margin_level below stop_level at some time")
print(drawdown_auto_stop_summary)

















