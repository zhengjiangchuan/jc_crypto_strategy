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

    return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

def plot_candle_bar_charts(raw_symbol, all_data_df, trading_days,
                           num_days = 10, plot_jc = False, plot_bolling = False, is_jc_calculated = False, print_prefix = "",
                           trade_df = None, trade_buy_time = 'buy_time', trade_sell_time = 'sell_time',
                           state_df = None, is_plot_candle_buy_sell_points = False, is_plot_market_state = False, tick_interval = 0.001,
                           bar_fig_folder = None, is_plot_aux = False, file_name_suffix = '', is_plot_simple_chart = False, plot_exclude = False):

    print("In plot_candle_bar_charts:")
    print("tick_interval = " + str(tick_interval))

    windows = [12, 30, 35, 40, 45, 50, 60, 144, 169]

    #high_low_window = 200

    # print("all_data_df:")
    # print(all_data_df.tail(10))


    # if is_plot_candle_buy_sell_points:
    #
    #     long_trade_df = trade_df[trade_df['trade_type'] == 1]
    #     short_trade_df = trade_df[trade_df['trade_type'] == -1]
    #
    #     long_df = long_trade_df[[trade_buy_time, 'buy_price']]
    #     close_long_df = long_trade_df[[trade_sell_time, 'sell_price']]
    #
    #     short_df = short_trade_df[[trade_sell_time, 'sell_price']]
    #     close_short_df = short_trade_df[[trade_buy_time, 'buy_price']]
    #
    #
    #     long_marker = '^'
    #     close_long_marker = 'v'
    #     short_marker = 'v'
    #     close_short_marker = '^'


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

        max_price = sub_data['close'].max()
        min_price = sub_data['close'].min()
        tick_interval = (max_price - min_price) / y_tick_number

        #period_start_time = sub_data.iloc[0]['ori_date']
        #period_end_time = sub_data.iloc[-1]['ori_date']

        # if is_plot_candle_buy_sell_points:
        #
        #     #period_end_time_next = period_end_time + timedelta(days=1)
        #
        #     # print("long_df:")
        #     # print(long_df)
        #
        #     period_long_df = long_df[(long_df[trade_buy_time] >= start_date) & (long_df[trade_buy_time] < end_date)]
        #     period_close_long_df = close_long_df[(close_long_df[trade_sell_time] >= start_date) & (close_long_df[trade_sell_time] < end_date)]
        #
        #     period_short_df = short_df[(short_df[trade_sell_time] >= start_date) & (short_df[trade_sell_time] < end_date)]
        #     period_close_short_df = close_short_df[(close_short_df[trade_buy_time] >= start_date) & (close_short_df[trade_buy_time] < end_date)]
        #
        #     long_points = []
        #     for i in range(period_long_df.shape[0]):
        #         long_points += [which(sub_data['time'] == period_long_df.iloc[i][trade_buy_time])[0]]
        #
        #     close_long_points = []
        #     for i in range(period_close_long_df.shape[0]):
        #
        #         # print("period_close_long_df:")
        #         # print(period_close_long_df)
        #         #
        #         # print("sub_data:")
        #         # print(sub_data.head(40))
        #
        #         close_long_points += [which(sub_data['time'] == period_close_long_df.iloc[i][trade_sell_time])[0]]
        #
        #     short_points = []
        #     for i in range(period_short_df.shape[0]):
        #         short_points += [which(sub_data['time'] == period_short_df.iloc[i][trade_sell_time])[0]]
        #
        #     close_short_points = []
        #     for i in range(period_close_short_df.shape[0]):
        #         close_short_points += [which(sub_data['time'] == period_close_short_df.iloc[i][trade_buy_time])[0]]
        #
        # if is_plot_market_state and state_df is not None:
        #
        #     period_state_df = state_df[(state_df['start_time'] >= start_date) & (state_df['start_time'] < end_date)]
        #
        #     state_points = []
        #     for i in range(period_state_df.shape[0]):
        #         state_points += [which(sub_data['time'] == period_state_df.iloc[i]['start_time'])[0]]
        #
        #



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

        candle_df = sub_data[['artificial_time', 'open', 'high', 'low', 'close', 'macd', 'msignal', 'time']
                             + ['period_high' + str(high_low_window), 'period_low' + str(high_low_window)] + \
                             (['period_high' + str(high_low_window2), 'period_low' + str(high_low_window2)] if plot_both_high_low_windows else []) + \
                             ['macd_period_high' + str(high_low_window), 'macd_period_low' + str(high_low_window)] + \
                             (['macd_period_high' + str(high_low_window2), 'macd_period_low' + str(high_low_window2)] if plot_both_high_low_windows else [])
                             + ['ma_' + 'close' + str(window) for window in windows] + ['upper_band_close', 'lower_band_close', 'middle_band_close']]
        candle_df['artificial_time'] = candle_df['artificial_time'].apply(lambda x: mdates.date2num(x))
        int_time_series = candle_df['artificial_time'].values
        candle_matrix = candle_df.values
        candlestick_ohlc(axes, candle_matrix, colordown = 'r', colorup = 'g', width = 0.0005, alpha = 1)

        print("Reach here 1")

        buy_ready_points = which(sub_data['buy_ready'])
        # print("buy_fire:")
        # print(sub_data['buy_fire'][0:10])
        # print("first_buy_fire:")
        # print(sub_data['first_buy_fire'][0:10])
        # dummy_buy_points = which(sub_data['buy_fire'])
        # print("dummy_buy_points:")
        # print(dummy_buy_points[0:10])



        buy_real_points_vegas = which(sub_data['first_buy_real_fire'])
        buy_real_points_reverse = which(sub_data['first_final_buy_fire']) #which(sub_data['first_buy_real_fire2'] | sub_data['first_buy_real_fire3'])
        if 'first_final_buy_fire_exclude' in sub_data.columns:
            buy_real_points_reverse_exclude = which(sub_data['first_final_buy_fire_exclude'])
        if 'first_buy_fire_magic_exclude' in sub_data.columns:
            buy_real_points_magic_exclude = which(sub_data['first_buy_fire_magic_exclude'])

        if 'first_actual_special_sell_close_position' in sub_data.columns:
            sell_special_close_points = which(sub_data['first_actual_special_sell_close_position'])
        if 'first_actual_sell_close_position_excessive' in sub_data.columns:
            sell_close_points_excessive = which(sub_data['first_actual_sell_close_position_excessive'])
        if 'first_actual_sell_close_position_conservative' in sub_data.columns:
            sell_close_points_conservative = which(sub_data['first_actual_sell_close_position_conservative'])
        if 'first_sell_stop_loss_excessive' in sub_data.columns:
            sell_stop_loss_excessive = which(sub_data['first_sell_stop_loss_excessive'])
        if 'first_sell_stop_loss_conservative' in sub_data.columns:
            sell_stop_loss_conservative = which(sub_data['first_sell_stop_loss_conservative'])


        if 'show_special_sell_close_position' in sub_data.columns:
            show_sell_special_close_points = which(sub_data['show_special_sell_close_position'])
        if 'show_sell_close_position_excessive' in sub_data.columns:
            show_sell_close_points_excessive = which(sub_data['show_sell_close_position_excessive'])
        if 'show_sell_close_position_conservative' in sub_data.columns:
            show_sell_close_points_conservative = which(sub_data['show_sell_close_position_conservative'])
        if 'show_sell_stop_loss_excessive' in sub_data.columns:
            show_sell_stop_loss_excessive = which(sub_data['show_sell_stop_loss_excessive'])
        if 'show_sell_stop_loss_conservative' in sub_data.columns:
            show_sell_stop_loss_conservative = which(sub_data['show_sell_stop_loss_conservative'])


        if 'show_sell_close_position_guppy1' in sub_data.columns:
            show_sell_close_points_guppy1 = which(sub_data['show_sell_close_position_guppy1'])
        if 'show_sell_close_position_guppy2' in sub_data.columns:
            show_sell_close_points_guppy2 = which(sub_data['show_sell_close_position_guppy2'])
        if 'show_sell_close_position_vegas' in sub_data.columns:
            show_sell_close_points_vegas = which(sub_data['show_sell_close_position_vegas'])
        if 'show_sell_close_position_final_excessive' in sub_data.columns:
            show_sell_close_points_final_excessive = which(sub_data['show_sell_close_position_final_excessive'])
        if 'show_sell_close_position_final_conservative' in sub_data.columns:
            show_sell_close_points_final_conservative = which(sub_data['show_sell_close_position_final_conservative'])





        buy_real_points_macd = which(sub_data['first_buy_real_fire4'])
        buy_real_points_simple = which(sub_data['first_buy_real_fire5'])

        buy_points = which(sub_data['first_buy_fire'] & (~sub_data['first_buy_real_fire']))

        #buy_reverse_points = which(sub_data['first_buy_real_fire2'] | sub_data['first_buy_real_fire3'])
        buy_reverse_points = which(sub_data['first_buy_real_fire4'])

        # buy_real_points = which(sub_data['first_buy_real_fire2'])
        # buy_points = which(sub_data['first_buy_fire'] & (~sub_data['first_buy_real_fire']))


        buy_weak_ready_points = which(sub_data['buy_weak_ready'] & (~sub_data['buy_ready']))
        buy_weak_points = which(sub_data['first_buy_weak_fire'] & (~sub_data['first_buy_fire']) & (~sub_data['first_buy_real_fire']))


        sell_ready_points = which(sub_data['sell_ready'])

        sell_real_points_vegas = which(sub_data['first_sell_real_fire'])
        sell_real_points_reverse = which(sub_data['first_final_sell_fire']) # which(sub_data['first_sell_real_fire2'] | sub_data['first_sell_real_fire3'])
        if 'first_final_sell_fire_exclude' in sub_data.columns:
            sell_real_points_reverse_exclude = which(sub_data['first_final_sell_fire_exclude'])
        if 'first_sell_fire_magic_exclude' in sub_data.columns:
            sell_real_points_magic_exclude = which(sub_data['first_sell_fire_magic_exclude'])


        if 'first_actual_special_buy_close_position' in sub_data.columns:
            buy_special_close_points = which(sub_data['first_actual_special_buy_close_position'])
        if 'first_actual_buy_close_position_excessive' in sub_data.columns:
            buy_close_points_excessive = which(sub_data['first_actual_buy_close_position_excessive'])
        if 'first_actual_buy_close_position_conservative' in sub_data.columns:
            buy_close_points_conservative = which(sub_data['first_actual_buy_close_position_conservative'])
        if 'first_buy_stop_loss_excessive' in sub_data.columns:
            buy_stop_loss_excessive = which(sub_data['first_buy_stop_loss_excessive'])
        if 'first_buy_stop_loss_conservative' in sub_data.columns:
            buy_stop_loss_conservative = which(sub_data['first_buy_stop_loss_conservative'])


        if 'show_special_buy_close_position' in sub_data.columns:
            show_buy_special_close_points = which(sub_data['show_special_buy_close_position'])
        if 'show_buy_close_position_excessive' in sub_data.columns:
            show_buy_close_points_excessive = which(sub_data['show_buy_close_position_excessive'])
        if 'show_buy_close_position_conservative' in sub_data.columns:
            show_buy_close_points_conservative = which(sub_data['show_buy_close_position_conservative'])
        if 'show_buy_stop_loss_excessive' in sub_data.columns:
            show_buy_stop_loss_excessive = which(sub_data['show_buy_stop_loss_excessive'])
        if 'show_buy_stop_loss_conservative' in sub_data.columns:
            show_buy_stop_loss_conservative = which(sub_data['show_buy_stop_loss_conservative'])


        if 'show_buy_close_position_guppy1' in sub_data.columns:
            show_buy_close_points_guppy1 = which(sub_data['show_buy_close_position_guppy1'])
        if 'show_buy_close_position_guppy2' in sub_data.columns:
            show_buy_close_points_guppy2 = which(sub_data['show_buy_close_position_guppy2'])
        if 'show_buy_close_position_vegas' in sub_data.columns:
            show_buy_close_points_vegas = which(sub_data['show_buy_close_position_vegas'])
        if 'show_buy_close_position_final_excessive' in sub_data.columns:
            show_buy_close_points_final_excessive = which(sub_data['show_buy_close_position_final_excessive'])
        if 'show_buy_close_position_final_conservative' in sub_data.columns:
            show_buy_close_points_final_conservative = which(sub_data['show_buy_close_position_final_conservative'])



        sell_real_points_macd = which(sub_data['first_sell_real_fire4'])
        sell_real_points_simple = which(sub_data['first_sell_real_fire5'])

        sell_points = which(sub_data['first_sell_fire'] & (~sub_data['first_sell_real_fire']))

        #sell_reverse_points = which(sub_data['first_sell_real_fire2'] | sub_data['first_sell_real_fire3'])
        sell_reverse_points = which(sub_data['first_sell_real_fire4'])

        # sell_real_points = which(sub_data['first_sell_real_fire2'])
        # sell_points = which(sub_data['first_sell_fire'] & (~sub_data['first_sell_real_fire']))

        sell_weak_ready_points = which(sub_data['sell_weak_ready'] & (~sub_data['sell_ready']))
        sell_weak_points = which(sub_data['first_sell_weak_fire'] & (~sub_data['first_sell_fire']) & (~sub_data['first_sell_real_fire']))


        if is_plot_candle_buy_sell_points:

            long_marker = '^'
            short_marker = 'v'


            # for buy_real_point in buy_real_points_vegas: #buy_real_points_vegas
            #     axes.plot(int_time_series[buy_real_point], sub_data.iloc[buy_real_point]['close'], marker = long_marker, markersize = 15, color = 'blue')

            ####################################
            for buy_real_point in buy_real_points_reverse:
                axes.plot(int_time_series[buy_real_point], sub_data.iloc[buy_real_point]['close'], marker = long_marker, markersize = 15, color = 'darkblue')

            if 'first_final_buy_fire_exclude' in sub_data.columns:
                for buy_real_point_exclude in buy_real_points_reverse_exclude:
                    axes.plot(int_time_series[buy_real_point_exclude], sub_data.iloc[buy_real_point_exclude]['close'], marker = long_marker, markersize = 15, color = 'cornflowerblue')

            if 'first_buy_fire_magic_exclude' in sub_data.columns:
                for buy_real_point_exclude in buy_real_points_magic_exclude:
                    axes.plot(int_time_series[buy_real_point_exclude], sub_data.iloc[buy_real_point_exclude]['close'], marker = long_marker, markersize = 15, color = 'darkturquoise')

            ####################################


            # if 'first_actual_special_sell_close_position' in sub_data.columns:
            #     for sell_close_point in sell_special_close_points:
            #         axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkturquoise')
            #
            # if 'first_actual_sell_close_position_excessive' in sub_data.columns:
            #     for sell_close_point in sell_close_points_excessive:
            #         axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkturquoise')
            #
            # if 'first_actual_sell_close_position_conservative' in sub_data.columns:
            #     for sell_close_point in sell_close_points_conservative:
            #         axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'cadetblue')
            #
            #
            # if 'first_sell_stop_loss_excessive' in sub_data.columns:
            #     for sell_close_point in sell_stop_loss_excessive:
            #         axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = 'o', markersize = 9, color = 'darkturquoise')
            #
            # if 'first_sell_stop_loss_conservative' in sub_data.columns:
            #     for sell_close_point in sell_stop_loss_conservative:
            #         axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = 'o', markersize = 9, color = 'cadetblue')


            ###################################################################################
            if 'show_special_sell_close_position' in sub_data.columns:
                for sell_close_point in show_sell_special_close_points:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkturquoise')

            if 'show_sell_close_position_excessive' in sub_data.columns:
                for sell_close_point in show_sell_close_points_excessive:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkturquoise')

            if 'show_sell_close_position_conservative' in sub_data.columns:
                for sell_close_point in show_sell_close_points_conservative:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'cadetblue')


            if 'show_sell_stop_loss_excessive' in sub_data.columns:
                for sell_close_point in show_sell_stop_loss_excessive:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkturquoise')

            if 'show_sell_stop_loss_conservative' in sub_data.columns:
                for sell_close_point in show_sell_stop_loss_conservative:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'cadetblue')






            if 'show_sell_close_position_guppy1' in sub_data.columns:
                for sell_close_point in show_sell_close_points_guppy1:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkgreen')

            if 'show_sell_close_position_guppy2' in sub_data.columns:
                for sell_close_point in show_sell_close_points_guppy2:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkgreen')


            if 'show_sell_close_position_vegas' in sub_data.columns:
                for sell_close_point in show_sell_close_points_vegas:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'darkgreen')

            if 'show_sell_close_position_final_excessive' in sub_data.columns:
                for sell_close_point in show_sell_close_points_final_excessive:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'black')

            if 'show_sell_close_position_final_conservative' in sub_data.columns:
                for sell_close_point in show_sell_close_points_final_conservative:
                    axes.plot(int_time_series[sell_close_point], sub_data.iloc[sell_close_point]['close'], marker = long_marker, markersize = 12, color = 'black')






            if not is_plot_simple_chart:
                for buy_point in buy_points:
                    axes.plot(int_time_series[buy_point], sub_data.iloc[buy_point]['close'], marker = long_marker, markersize = 15, color = 'blue')


                for buy_weak_point in buy_weak_points:
                    axes.plot(int_time_series[buy_weak_point], sub_data.iloc[buy_weak_point]['close'], marker = long_marker, markersize = 15, color = 'cornflowerblue')

            #
            # for sell_real_point in sell_real_points_vegas: #sell_real_points_vegas
            #     axes.plot(int_time_series[sell_real_point], sub_data.iloc[sell_real_point]['close'], marker = short_marker, markersize = 15, color = 'red')

            ####################################
            for sell_real_point in sell_real_points_reverse:
                axes.plot(int_time_series[sell_real_point], sub_data.iloc[sell_real_point]['close'], marker = short_marker, markersize = 15, color = 'darkred')

            if 'first_final_sell_fire_exclude' in sub_data.columns:
                for sell_real_point_exclude in sell_real_points_reverse_exclude:
                    axes.plot(int_time_series[sell_real_point_exclude], sub_data.iloc[sell_real_point_exclude]['close'], marker = short_marker, markersize = 15, color = 'salmon')

            if 'first_sell_fire_magic_exclude' in sub_data.columns:
                for sell_real_point_exclude in sell_real_points_magic_exclude:
                    axes.plot(int_time_series[sell_real_point_exclude], sub_data.iloc[sell_real_point_exclude]['close'], marker = short_marker, markersize = 15, color = 'violet')

            ####################################


            # if 'first_actual_special_buy_close_position' in sub_data.columns:
            #     for buy_close_point in buy_special_close_points:
            #         axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'violet')
            #
            #
            # if 'first_actual_buy_close_position_excessive' in sub_data.columns:
            #     for buy_close_point in buy_close_points_excessive:
            #         axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'violet')
            #
            # if 'first_actual_buy_close_position_conservative' in sub_data.columns:
            #     for buy_close_point in buy_close_points_conservative:
            #         axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'mediumorchid')
            #
            # if 'first_buy_stop_loss_excessive' in sub_data.columns:
            #     for buy_close_point in buy_stop_loss_excessive:
            #         axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = 'o', markersize = 9, color = 'violet')
            #
            # if 'first_buy_stop_loss_conservative' in sub_data.columns:
            #     for buy_close_point in buy_stop_loss_conservative:
            #         axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = 'o', markersize = 9, color = 'mediumorchid')



             ###################################################################################
            if 'show_special_buy_close_position' in sub_data.columns:
                for buy_close_point in show_buy_special_close_points:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'violet')

            if 'show_buy_close_position_excessive' in sub_data.columns:
                for buy_close_point in show_buy_close_points_excessive:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'violet')

            if 'show_buy_close_position_conservative' in sub_data.columns:
                for buy_close_point in show_buy_close_points_conservative:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'mediumorchid')


            if 'show_buy_stop_loss_excessive' in sub_data.columns:
                for buy_close_point in show_buy_stop_loss_excessive:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'violet')

            if 'show_buy_stop_loss_conservative' in sub_data.columns:
                for buy_close_point in show_buy_stop_loss_conservative:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'mediumorchid')





            if 'show_buy_close_position_guppy1' in sub_data.columns:
                for buy_close_point in show_buy_close_points_guppy1:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'darkgreen')

            if 'show_buy_close_position_guppy2' in sub_data.columns:
                for buy_close_point in show_buy_close_points_guppy2:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'darkgreen')


            if 'show_buy_close_position_vegas' in sub_data.columns:
                for buy_close_point in show_buy_close_points_vegas:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'darkgreen')

            if 'show_buy_close_position_final_excessive' in sub_data.columns:
                for buy_close_point in show_buy_close_points_final_excessive:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'black')

            if 'show_buy_close_position_final_conservative' in sub_data.columns:
                for buy_close_point in show_buy_close_points_final_conservative:
                    axes.plot(int_time_series[buy_close_point], sub_data.iloc[buy_close_point]['close'], marker = short_marker, markersize = 12, color = 'black')






            if not is_plot_simple_chart:
                for sell_point in sell_points:
                    axes.plot(int_time_series[sell_point], sub_data.iloc[sell_point]['close'], marker = short_marker, markersize = 15, color = 'red')


                for sell_weak_point in sell_weak_points:
                    axes.plot(int_time_series[sell_weak_point], sub_data.iloc[sell_weak_point]['close'], marker = short_marker, markersize = 15, color = 'salmon')


        # if is_plot_candle_buy_sell_points:
        #
        #     i = -1
        #     for close_long_point in close_long_points:
        #         i += 1
        #         axes.plot(int_time_series[close_long_point], period_close_long_df.iloc[i]['sell_price'], marker=close_long_marker, markersize=20, color='red')
        #
        #     i = -1
        #     for close_short_point in close_short_points:
        #         i += 1
        #         axes.plot(int_time_series[close_short_point], period_close_short_df.iloc[i]['buy_price'],
        #                   marker=close_short_marker, markersize=20, color='midnightblue')
        #
        #     i = -1
        #     for long_point in long_points:
        #         i += 1
        #         axes.plot(int_time_series[long_point], period_long_df.iloc[i]['buy_price'], marker=long_marker,
        #                   markersize=20, color='blue')
        #
        #     i = -1
        #     for short_point in short_points:
        #         i += 1
        #         axes.plot(int_time_series[short_point], period_short_df.iloc[i]['sell_price'], marker=short_marker, markersize=20, color='darkred')
        #
        #
        #
        #
        # if is_plot_market_state and state_df is not None:
        #     i = -1
        #     for state_point in state_points:
        #         i += 1
        #         axes.annotate(period_state_df.iloc[i]['state'], xy = (int_time_series[state_point], period_state_df.iloc[i]['price']), #sub_data.iloc[state_point]['close']),
        #                 color = 'black', size = 12)


        if plot_jc:
            plot_jc_lines(candle_df, attr = 'close', x = 'artificial_time', ax = axes, legend = -1, label = '_nolegend_', is_plot_high_low = True)

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



        #candle_df['signal'] = 'macd'

        # print("candle_df:")
        # print(candle_df.head(20))

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

        #Riccardo
        # macd_df = candle_df[['time_id', 'time', 'close', 'macd', 'msignal', "macd_period_high" + str(high_low_window), "macd_period_low" + str(high_low_window)]]
        # print("macd_df:")
        # print(macd_df)

        sub_df1 = candle_df[['time_id', 'macd']]
        sub_df2 = candle_df[['time_id', 'msignal']]

        # sub_df3 = candle_df[['time_id', "macd_period_high" + str(high_low_window)]]
        # sub_df4 = candle_df[['time_id', "macd_period_low" + str(high_low_window)]]

        sub_df1 = sub_df1.rename(columns = {"macd" : "macd_indicator"})
        sub_df2 = sub_df2.rename(columns={"msignal": "macd_indicator"})

        # sub_df3 = sub_df3.rename(columns={"macd_period_high" + str(high_low_window): "macd_indicator"})
        # sub_df4 = sub_df4.rename(columns={"macd_period_low" + str(high_low_window): "macd_indicator"})


        # print("sub_df1:")
        # print(sub_df1.tail(10))
        # print("sub_df2:")
        # print(sub_df2.tail(10))

        sub_df1['signal'] = 'macd'
        sub_df2['signal'] = 'msignal'
        #
        # sub_df3['signal'] = "macd_period_high" + str(high_low_window)
        # sub_df4['signal'] = "macd_period_low" + str(high_low_window)

        sub_df = pd.concat([sub_df1, sub_df2])

        #sub_df = candle_df[['time_id', 'macd', 'signal']]
        # print("sub_df:")
        # print(sub_df.head(20))
        # print("row_number = " + str(sub_df.shape[0]))

        if is_plot_aux:

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
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = "{}".format(sender)
    message['To'] = ",".join(receivers)
    message['Subject'] = title

    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("mail has been send successfully.")
    except smtplib.SMTPException as e:
        print(e)











































































