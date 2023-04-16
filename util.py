def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import talib
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.ticker as ticker
import datetime
from mpl_finance import *


#from mplfinance import *
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import timedelta
from matplotlib.ticker import Formatter
import  matplotlib.ticker as plticker

import matplotlib.dates as mdates

#from instrument_trader import support_half_stop_loss

support_half_stop_loss = False

import gzip

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

pd.options.mode.chained_assignment = None

high_low_window = 100

high_low_window2 = 200

plot_both_high_low_windows = True

def which(bool_array):

    a = np.arange(len(bool_array))
    return a[bool_array]


def calc_high_Low(df, attr, window):

    df['period_high' + str(window)] = df[attr].rolling(window, min_periods = window).max()
    df['period_low' + str(window)] = df[attr].rolling(window, min_periods = window).min()


def calc_ma(df, attr, window):

    df['ma_' + attr + str(window)] = df[attr].rolling(window, min_periods = window).mean()

def plot_line(df, x, y, ax, legend, linewidth, color, label = ""):

    if legend == -1:
        df.plot(x = x, y = y, ax = ax, linewidth = linewidth, color = color, label = label)
    else:
        df.plot(x = x, y = y, ax = ax, legend = legend, linewidth = linewidth, color = color)

def calc_jc_lines(df, attr, windows):

    for window in windows:
        calc_ma(df, attr, window)

    #return(windows)

def calc_bolling_bands(df, attr, window):

    values = df[attr].values
    upper, middle, lower = talib.BBANDS(values, timeperiod = window, matype=talib.MA_Type.SMA)

    df['upper_band_' + attr] = upper
    df['lower_band_' + attr] = lower
    df['middle_band_' +attr] = middle


def calc_macd(df, attr, window):

    values = df[attr].values
    macd, macdsignal, macdhist = talib.MACD(values, fastperiod = 12, slowperiod = 26, signalperiod = 9)

    df['macd'] = macd
    df['msignal'] = macdsignal

    df['macd_period_high' + str(window)] = df['macd'].rolling(window, min_periods = window).max()
    df['macd_period_low' + str(window)] = df['macd'].rolling(window, min_periods = window).min()

    #print("In calc_macd:")
    #print(df[['time','close','macd','msignal', 'macd_period_high' + str(window), 'macd_period_low' + str(window)]])

    #
    # print("macd:")
    # print(df[['time', 'close', 'macd', 'msignal']].tail(30))
    #




def plot_group_lines(df, attr, x, ax, legend, windows, linewidth, color, label = ""):

    for window in windows:
        plot_line(df, x, "ma_" + attr + str(window), ax, legend, linewidth, color, label)


def plot_group_lines2(df, x, ax, legend, windows, linewidth, color, label = ""):

    for window in windows:
        plot_line(df, x, "period_high" + str(window),
                  ax, legend, linewidth, color, label)
        plot_line(df, x, "period_low" + str(window),
                  ax, legend, linewidth, color, label)




def plot_jc_lines(df, attr, x, ax, legend, label = "", is_plot_high_low = False):

    # short_guppy_windows = [3,5,8,10,15]
    # short_guppy_color = 'teal'
    # short_guppy_width = 0.5

    long_guppy_windows = [30,35,40,45,50,60]
    long_guppy_color = 'sienna'
    long_guppy_width = 0.5

    filter_windows = [12]
    filter_color = 'red'
    filter_width = 0.5

    vegas_windows = [144]
    vegas_color = 'darkviolet'
    vegas_width = 0.5

    vegas_windows2 = [169]
    vegas_color2 = 'blue'
    vegas_width2 = 0.5

    #plot_group_lines(df, attr, x, ax, legend, short_guppy_windows, short_guppy_width, short_guppy_color, label)
    plot_group_lines(df, attr, x, ax, legend, long_guppy_windows, long_guppy_width, long_guppy_color, label)
    plot_group_lines(df, attr, x, ax, legend, filter_windows, filter_width, filter_color, label)
    plot_group_lines(df, attr, x, ax, legend, vegas_windows, vegas_width, vegas_color, label)
    plot_group_lines(df, attr, x, ax, legend, vegas_windows2, vegas_width2, vegas_color2, label)

    if is_plot_high_low:

        if plot_both_high_low_windows:
            high_low_windows = [high_low_window2]
            high_low_color = 'darkorange'
            high_low_width = 1
            plot_group_lines2(df, x, ax, legend, high_low_windows, high_low_width, high_low_color, label)

        high_low_windows = [high_low_window]
        high_low_color = 'darkgreen'
        high_low_width = 1
        plot_group_lines2(df, x, ax, legend, high_low_windows, high_low_width, high_low_color, label)






def plot_bolling_bands(df, attr, x, ax, legend, label = ""):

    line_width = 0.5
    color = 'purple'

    plot_line(df, x, "upper_band_" + attr, ax, legend, line_width, color, label)
    plot_line(df, x, "lower_band_" + attr, ax, legend, line_width, color, label)
    plot_line(df, x, "middle_band_" + attr, ax, legend, line_width, color, label)






def time_diff_in_seconds(t1, t2):

    delta = t2.replace(tzinfo = None) - t1.replace(tzinfo = None)

    return delta.days * 3600 * 24 + delta.seconds

def time_diff_in_seconds_no_timezone(t1, t2):

    delta = t2 - t1
    return delta.days * 3600 * 24 + delta.seconds

class JCFormatter(Formatter):

    def __init__(self, dt, start, fmt = "%y-%m-%d %H:%M"):

        self.dt = dt
        self.start = start
        self.fmt = fmt

    def __call__(self, x, pos = 0):

        artificial_time = mdates.num2date(x)
        start_time = mdates.num2date(self.start)

        idx = int(time_diff_in_seconds(start_time, artificial_time)/60)

        if idx >= 0 and idx < len(self.dt):
            return self.dt[idx].strftime(self.fmt)
        else:
            return ""


def attach_ha_bars(df_bar):

    df_bar['pre_open'] = df_bar['open'].shift(1)
    df_bar['pre_close'] = df_bar['close'].shift(1)
    df_bar['ha_open'] = (df_bar['pre_open'] + df_bar['pre_close'])/2
    df_bar['ha_close'] = (df_bar['open'] + df_bar['high'] + df_bar['low'] + df_bar['close'])/4
    df_bar['ha_high'] = df_bar[['ha_open', 'ha_close', 'high']].max(axis = 1)
    df_bar['ha_low'] = df_bar[['ha_open', 'ha_close', 'low']].min(axis = 1)
    df_bar = df_bar.drop(columns = ['pre_open', 'pre_close'])

    return df_bar

def preprocess_time(t):

    if t[0] == "\'":
        t = t[1:]

    if len(t) < 19:
        t = t + ' 00:00:00'

    return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

def plot_candle_bar_charts(raw_symbol, all_data_df, trading_days, long_df, short_df,
                           num_days = 10, plot_jc = False, plot_bolling = False, is_jc_calculated = False, print_prefix = "",
                           trade_df = None, trade_buy_time = 'buy_time', trade_sell_time = 'sell_time',
                           state_df = None, is_plot_candle_buy_sell_points = False, is_plot_market_state = False, tick_interval = 0.001,
                           bar_fig_folder = None, is_plot_aux = False, file_name_suffix = '', is_plot_simple_chart = False, plot_exclude = False):

    print("In plot_candle_bar_charts:")
    print("tick_interval = " + str(tick_interval))

    windows = [12, 30, 35, 40, 45, 50, 60, 144, 169]



    remaining = len(trading_days) % num_days

    if remaining == 0:
        periods = []
    else:
        periods = [(0, remaining)]

    for i in range(0, len(trading_days) // num_days):
        periods += [(remaining + i * num_days, remaining + (i+1) * num_days)]

    y_tick_number = 10

    figs = []
    intervals = []
    for period in periods:
        start_date = trading_days[period[0]]

        if period[1] < len(trading_days):
            end_date = trading_days[period[1]]
        else:
            end_date = trading_days[-1] + timedelta(days = 1)

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        print(print_prefix + " Plot candle bar data from " + start_date_str + " until " + end_date_str)

        start_date_simple_str = start_date.strftime('%Y%m%d')
        end_date_simple_str = end_date.strftime('%Y%m%d')

        interval = start_date_simple_str + '-' + end_date_simple_str
        intervals += [interval]

        sub_data = all_data_df[(all_data_df['time'] >= start_date) & (all_data_df['time'] < end_date)]

        min_id = sub_data.iloc[0]['id']
        max_id = sub_data.iloc[-1]['id']


        # print("long_df:")
        # print(long_df.iloc[0:20])
        #
        # print("sub_data:")
        # print(sub_data.iloc[0:20])


        long_sub_data = long_df[(long_df['long_stop_profit_loss_time'] >= sub_data.iloc[0]['time']) & (long_df['time'] <= sub_data.iloc[-1]['time'])]
        short_sub_data = short_df[(short_df['short_stop_profit_loss_time'] >= sub_data.iloc[0]['time']) & (short_df['time'] <= sub_data.iloc[-1]['time'])]

        long_sub_data['entry_id'] = np.where(long_sub_data['id'] >= min_id, long_sub_data['id'], min_id)
        long_sub_data['exit_id'] = np.where(long_sub_data['long_stop_profit_loss_id'] <= max_id, long_sub_data['long_stop_profit_loss_id'], max_id)


        short_sub_data['entry_id'] = np.where(short_sub_data['id'] >= min_id, short_sub_data['id'], min_id)
        short_sub_data['exit_id'] = np.where(short_sub_data['short_stop_profit_loss_id'] <= max_id, short_sub_data['short_stop_profit_loss_id'], max_id)


        long_sub_data['entry_id'] = long_sub_data['entry_id'] - min_id
        long_sub_data['exit_id'] = long_sub_data['exit_id'] - min_id

        short_sub_data['entry_id'] = short_sub_data['entry_id'] - min_id
        short_sub_data['exit_id'] = short_sub_data['exit_id'] - min_id

        long_real_sub_data = long_sub_data[long_sub_data['id'] >= min_id]
        short_real_sub_data = short_sub_data[short_sub_data['id'] >= min_id]




        max_price = sub_data['close'].max()
        min_price = sub_data['close'].min()
        tick_interval = (max_price - min_price) / y_tick_number



        row_num = sub_data.shape[0]

        sub_data['idx'] = np.arange(0, row_num, 1)
        start_time = sub_data.iloc[0]['time']
        sub_data['artificial_time'] = sub_data['idx'].apply(lambda idx: start_time + timedelta(0, idx*60))
        sub_data.set_index(['idx'], inplace = True)

        sub_data['start'] = sub_data.index
        sub_data['end'] = sub_data.index
        all_data = sub_data[['time', 'start', 'end']]

        d_data = all_data.groupby(pd.DatetimeIndex(all_data['time']).normalize()).agg(
            {
                'start' : 'first',
                'end' : 'last'
            }
        )

        fig = plt.figure(figsize = (30, 15))

        if is_plot_aux:
            axes_list = fig.subplots(nrows = 2, ncols = 1, gridspec_kw={'height_ratios': [2, 1]})
            axes = axes_list[0]
            aux_axes = axes_list[1]
        else:
            axes = fig.subplots(nrows = 1, ncols = 1)

        candle_df = sub_data[['artificial_time', 'open', 'high', 'low', 'close', 'time']
                             + ['ma_' + 'close' + str(window) for window in windows] + ['upper_band_close', 'lower_band_close', 'middle_band_close']]
        candle_df['artificial_time'] = candle_df['artificial_time'].apply(lambda x: mdates.date2num(x))
        int_time_series = candle_df['artificial_time'].values
        candle_matrix = candle_df.values
        candlestick_ohlc(axes, candle_matrix, colordown = 'r', colorup = 'g', width = 0.0005, alpha = 1)

        # print("Reach here 1")
        #
        # print("sub_data:")
        # #print(sub_data[['time','id','open']])
        # print(sub_data.iloc[0:10][['time','id','open']])
        # print(sub_data.iloc[-10:][['time', 'id', 'open']])
        #
        # print("long_sub_data:")
        # print(long_sub_data)
        #
        # if long_sub_data.shape[0] > 0:
        #     print(type(long_sub_data.iloc[0]['entry_id']))
        #
        # print("")
        # print("")


        long_win_points = long_real_sub_data[long_real_sub_data['long_stop_profit_loss'] == 1]['entry_id'].tolist()
        long_lose_points = long_real_sub_data[long_real_sub_data['long_stop_profit_loss'] == -1]['entry_id'].tolist()

        # print("long_win_points:")
        # print(long_win_points)


        long_hit_profit = long_sub_data[long_sub_data['long_stop_profit_loss'] == 1][['entry_id', 'exit_id', 'long_stop_profit_price']]

        # if long_hit_profit.shape[0] > 0:
        #     print("long_hit_profit type:")
        #     print(long_hit_profit)
        #     print(type(long_hit_profit.iloc[0]['entry_id']))

        long_not_hit_profit = long_sub_data[long_sub_data['long_stop_profit_loss'] == -1][['entry_id', 'exit_id', 'long_stop_profit_price']]

        long_hit_loss = long_sub_data[long_sub_data['long_stop_profit_loss'] == -1][['entry_id', 'exit_id', 'long_stop_loss_price']]
        long_not_hit_loss = long_sub_data[long_sub_data['long_stop_profit_loss'] == 1][['entry_id', 'exit_id', 'long_stop_loss_price']]





        short_win_points = short_real_sub_data[short_real_sub_data['short_stop_profit_loss'] == 1]['entry_id'].tolist()
        short_lose_points = short_real_sub_data[short_real_sub_data['short_stop_profit_loss'] == -1]['entry_id'].tolist()

        short_hit_profit = short_sub_data[short_sub_data['short_stop_profit_loss'] == 1][['entry_id', 'exit_id', 'short_stop_profit_price']]
        short_not_hit_profit = short_sub_data[short_sub_data['short_stop_profit_loss'] == -1][['entry_id', 'exit_id', 'short_stop_profit_price']]

        short_hit_loss = short_sub_data[short_sub_data['short_stop_profit_loss'] == -1][['entry_id', 'exit_id', 'short_stop_loss_price']]
        short_not_hit_loss = short_sub_data[short_sub_data['short_stop_profit_loss'] == 1][['entry_id', 'exit_id', 'short_stop_loss_price']]

        long_hit_profit['entry_id'] = long_hit_profit['entry_id'].astype(int)
        long_hit_profit['exit_id'] = long_hit_profit['exit_id'].astype(int)

        # print("Fucking type:")
        # if long_hit_profit.shape[0] > 0:
        #     print(type(long_hit_profit.iloc[0]['entry_id']))

        # for my_df in [long_hit_profit, long_not_hit_profit, long_hit_loss, long_not_hit_loss, short_hit_profit, short_not_hit_profit, short_hit_loss, short_not_hit_loss]:
        #     my_df['entry_id'] = my_df['entry_id'].astype(int)
        #     my_df['exit_id'] = my_df['exit_id'].astype(int)




        if is_plot_candle_buy_sell_points:

            long_marker = '^'
            short_marker = 'v'

            # print("Plot long_win_points:")
            # print(long_win_points)
            for point in long_win_points:
                axes.plot(int_time_series[point], sub_data.iloc[point]['open'], marker = long_marker, markersize = 15, color = 'blue')

            # print("Plot long_lose_points:")
            # print(long_lose_points)
            for point in long_lose_points:
                axes.plot(int_time_series[point], sub_data.iloc[point]['open'], marker = long_marker, markersize = 15, color = 'red')

            for point in short_win_points:
                axes.plot(int_time_series[point], sub_data.iloc[point]['open'], marker = short_marker, markersize = 15, color = 'blue')

            for point in short_lose_points:
                axes.plot(int_time_series[point], sub_data.iloc[point]['open'], marker = short_marker, markersize = 15, color = 'red')


            # print("long_hit_profit:")
            # print(long_hit_profit)

            # if long_hit_profit.shape[0] > 0:
            #     print(long_hit_profit.iloc[0])
            #     print("First element: " + str(long_hit_profit.iloc[0]['entry_id']))

            for j in range(long_hit_profit.shape[0]):
                # print("j:")
                # print(j)
                # print("entry_id:")
                # print(int_time_series[long_hit_profit.iloc[j]['entry_id']])
                # print("exit_id:")
                # print(int_time_series[long_hit_profit.iloc[j]['exit_id']])
                # print("int_time_series:")
                # print(int_time_series)
                # print("len = " + str(len(int_time_series)))
                #
                # print("entry_id = " + str(long_hit_profit.iloc[j]['entry_id']))
                # print("xmin = " + str(int_time_series[int(long_hit_profit.iloc[j]['entry_id'])]))
                #
                # print("exit_id = " + str(long_hit_profit.iloc[j]['exit_id']))
                # print("xmax = " + str(int_time_series[int(long_hit_profit.iloc[j]['exit_id'])]))

                axes.hlines(y=long_hit_profit.iloc[j]['long_stop_profit_price'],
                             xmin = int_time_series[int(long_hit_profit.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(long_hit_profit.iloc[j]['exit_id'])],
                             ls = '-', color = 'blue', linewidth = 1)

            # print("long_not_hit_profit:")
            # print(long_not_hit_profit)
            for j in range(long_not_hit_profit.shape[0]):
                axes.hlines(y=long_not_hit_profit.iloc[j]['long_stop_profit_price'],
                             xmin = int_time_series[int(long_not_hit_profit.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(long_not_hit_profit.iloc[j]['exit_id'])],
                             ls = '-', color = 'black', linewidth = 1)

            # print("long_hit_loss:")
            # print(long_hit_loss)
            for j in range(long_hit_loss.shape[0]):
                axes.hlines(y=long_hit_loss.iloc[j]['long_stop_loss_price'],
                             xmin = int_time_series[int(long_hit_loss.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(long_hit_loss.iloc[j]['exit_id'])],
                             ls = '-', color = 'red', linewidth = 1)

            # print("long_not_hit_loss:")
            # print(long_not_hit_loss)
            for j in range(long_not_hit_loss.shape[0]):
                axes.hlines(y=long_not_hit_loss.iloc[j]['long_stop_loss_price'],
                             xmin = int_time_series[int(long_not_hit_loss.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(long_not_hit_loss.iloc[j]['exit_id'])],
                             ls = '-', color = 'black', linewidth = 1)





            for j in range(short_hit_profit.shape[0]):
                axes.hlines(y=short_hit_profit.iloc[j]['short_stop_profit_price'],
                             xmin = int_time_series[int(short_hit_profit.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(short_hit_profit.iloc[j]['exit_id'])],
                             ls = '-', color = 'blue', linewidth = 1)


            for j in range(short_not_hit_profit.shape[0]):
                axes.hlines(y=short_not_hit_profit.iloc[j]['short_stop_profit_price'],
                             xmin = int_time_series[int(short_not_hit_profit.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(short_not_hit_profit.iloc[j]['exit_id'])],
                             ls = '-', color = 'black', linewidth = 1)


            for j in range(short_hit_loss.shape[0]):
                axes.hlines(y=short_hit_loss.iloc[j]['short_stop_loss_price'],
                             xmin = int_time_series[int(short_hit_loss.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(short_hit_loss.iloc[j]['exit_id'])],
                             ls = '-', color = 'red', linewidth = 1)

            for j in range(short_not_hit_loss.shape[0]):
                axes.hlines(y=short_not_hit_loss.iloc[j]['short_stop_loss_price'],
                             xmin = int_time_series[int(short_not_hit_loss.iloc[j]['entry_id'])],
                             xmax = int_time_series[int(short_not_hit_loss.iloc[j]['exit_id'])],
                             ls = '-', color = 'black', linewidth = 1)






        if plot_jc:
            plot_jc_lines(candle_df, attr = 'close', x = 'artificial_time', ax = axes, legend = -1, label = '_nolegend_', is_plot_high_low = False)

        if plot_bolling:
            plot_bolling_bands(candle_df, attr = 'close', x = 'artificial_time', ax = axes, legend = -1, label = '_nolegend_')

        axes.xaxis.set_major_formatter(JCFormatter(sub_data['time'], candle_df.iloc[0]['artificial_time']))

        axes.xaxis.set_major_locator(dates.MinuteLocator(interval = 10))
        axes.yaxis.set_major_locator(plticker.MultipleLocator(tick_interval))

        plt.setp(axes.get_xticklabels(), rotation = 45)
        axes.set_xlabel('time', size = 20)
        axes.set_ylabel('price', size = 20)
        axes.tick_params(labeltop = False, labelright = True)

        for day_point in d_data['start'].values[1:]:
            axes.axvline(int_time_series[day_point], ls = '--', color = 'black', linewidth = 1)

        axes.set_title(raw_symbol + " from " + start_date_str + " to " + end_date_str, fontsize = 20)



        candle_df['time_id'] = list(range(candle_df.shape[0]))
        time_id_array = candle_df['time_id'].values
        trade_times = candle_df['time']
        time_number = candle_df.shape[0]

        # print("trade_times:")
        # print(trade_times[0:10])

        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, time_number - 1)
            my_time = trade_times[thisind].strftime('%y%m%d-%H')

            # if x == 10:
            #     print("x = " + str(x))
            #     print("my_time = " + str(my_time))

            return my_time


        if is_plot_aux:

            sub_df1 = candle_df[['time_id', 'macd']]
            sub_df2 = candle_df[['time_id', 'msignal']]

            sub_df1 = sub_df1.rename(columns={"macd": "macd_indicator"})
            sub_df2 = sub_df2.rename(columns={"msignal": "macd_indicator"})

            sub_df1['signal'] = 'macd'
            sub_df2['signal'] = 'msignal'

            sub_df = pd.concat([sub_df1, sub_df2])

            sns.lineplot(x = 'time_id', y = 'macd_indicator', hue = 'signal', data = sub_df, ax = aux_axes, legend = False)
            candle_df.plot(x="time_id", y="macd_period_high" + str(high_low_window2), ax=aux_axes, linewidth=1, color='darkorange', legend = False)
            candle_df.plot(x="time_id", y="macd_period_low" + str(high_low_window2), ax=aux_axes, linewidth=1, color='darkorange', legend = False)
            candle_df.plot(x="time_id", y="macd_period_high" + str(high_low_window), ax=aux_axes, linewidth=1, color='darkgreen', legend = False)
            candle_df.plot(x="time_id", y="macd_period_low" + str(high_low_window), ax=aux_axes, linewidth=1, color='darkgreen', legend = False)


            plt.setp(aux_axes.get_xticklabels(), rotation=45)
            for day_point in d_data['start'].values[1:]:
                aux_axes.axvline(time_id_array[day_point], ls = '--', color = 'black', linewidth = 1)

            # for buy_reverse_point in buy_real_points_reverse:
            #     aux_axes.axvline(time_id_array[buy_reverse_point], ls='--', color='blue', linewidth=1)
            #
            # for sell_reverse_point in sell_real_points_reverse:
            #     aux_axes.axvline(time_id_array[sell_reverse_point], ls='--', color='red', linewidth=1)

            aux_axes.set_xlabel('time', size = 10)
            aux_axes.set_ylabel('macd', size = 10)

            aux_axes.xaxis.set_major_locator(ticker.MultipleLocator(10))
            aux_axes.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

            aux_axes.axhline(0, ls = '--', color = 'blue', linewidth = 1)

        fig_file_name = raw_symbol + '_' + interval + file_name_suffix + '.png'
        fig_file_path = os.path.join(bar_fig_folder, fig_file_name)
        #print(print_prefix + " Save figure " + fig_file_name)

        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)

        fig.savefig(fig_file_path)
        plt.close(fig)


        figs += [fig]

    print("Plotting finishes")
    return None #list(zip(figs, intervals))




def convert_to_5min(t):

    str_time = t.strftime("%Y-%m-%d %H:%M:%S")
    temp_array = str_time.split(' ')
    year_part = temp_array[0]
    hour_part = temp_array[1]

    temp = hour_part.split(':')
    hour_str = temp[0]
    min_str = temp[1]
    sec_str = temp[2]

    minute = int(min_str)
    nor_minute = minute //5 * 5
    nor_minute_str = str(nor_minute) if nor_minute >= 0 else '0' + str(nor_minute)

    new_time_str = year_part + ' ' + ':'.join([hour_str, str(nor_minute_str), '00'])

    new_time = datetime.datetime.strptime(new_time_str, "%Y-%m-%d %H:")

    if minute % 5 > 0:

        new_time = new_time + datetime.timedelta(seconds = 300)

    return new_time


import smtplib
from email.header import Header
from email.mime.text import MIMEText


mail_host = "smtp.163.com"
mail_user = "glzxely123"
mail_pass = "10331861oO"

sender = 'glzxely123@163.com'
receivers = ['jczheng198508@gmail.com']



def sendEmail(title, content):
    # message = MIMEText(content, 'plain', 'utf-8')
    # message['From'] = "{}".format(sender)
    # message['To'] = ",".join(receivers)
    # message['Subject'] = title
    #
    # try:
    #     smtpObj = smtplib.SMTP_SSL(mail_host, 465)
    #     smtpObj.login(mail_user, mail_pass)
    #     smtpObj.sendmail(sender, receivers, message.as_string())
    #     print("mail has been send successfully.")
    # except smtplib.SMTPException as e:
    #     print(e)
    pass











































































