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


import threading

windows = [12, 30, 35, 40, 45, 50, 60, 144, 169]
high_low_window = 100
bolling_width = 20

bar_low_percentile = 0.3
bar_high_percentile = 0.1

initial_bar_number = 400

distance_to_vegas_threshold = 0.14


class CurrencyTrader(threading.Thread):

    def __init__(self, condition, currency, data_folder, chart_folder, log_file):
        super().__init__(name = currency)
        self.condition = condition
        self.currency = currency
        self.data_folder = data_folder
        self.chart_folder = chart_folder
        self.data_df = None
        self.last_time = None
        self.log_file = log_file

        self.currency_file = os.path.join(data_folder, currency + ".csv")

        self.log_fd = open(self.log_file, 'a')

        self.print_to_console = True

        self.log_msg("Initializing...")




    def log_msg(self, msg):

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('[' + current_time + ' ' + self.currency + ']  ' + msg, file = self.log_fd)
        self.log_fd.flush()

        if self.print_to_console:
            print('[' + current_time + ' ' + self.currency + ']  ' + msg)


    def get_last_time(self):
        return self.last_time

    def feed_data(self, new_data_df):
        self.condition.acquire()

        if new_data_df is not None and new_data_df.shape[0] > 0:

            if self.data_df is None or self.data_df.shape[0] == 0:
                self.data_df = new_data_df
                self.last_time = self.data_df.iloc[-1]['time']
                self.condition.notify()
            else:
                new_last_time = new_data_df.iloc[-1]['time']
                if new_last_time > self.last_time:
                    incremental_df = new_data_df[new_data_df['time'] > self.last_time]

                    self.log_msg("Receive new bar data from " + str(incremental_df.iloc[0]['time']) + " to " + str(incremental_df.iloc[-1]['time']))

                    self.data_df = pd.concat([self.data_df, incremental_df])
                    self.last_time = new_last_time
                    self.condition.notify()

        self.condition.release()

    def run(self):

        self.trade()

    def trade(self):

        key_features = ['is_above_vegas', 'pct_to_upper_vegas', 'high_pct_price_buy', 'low_pct_price_buy',
                        'is_vegas_up_trend',
                        'is_below_vegas', 'pct_to_lower_vegas', 'low_pct_price_sell', 'high_pct_price_sell',
                        'is_vegas_down_trend',
                        'ma_close12', 'ma12_gradient']

        signal_attrs = ['buy_weak_ready', 'buy_weak_fire',  'buy_ready', 'buy_fire',
                        'sell_weak_ready', 'sell_weak_fire',  'sell_ready', 'sell_fire']

        self.condition.acquire()

        while True:

            self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()

            all_days = pd.Series(self.data_df['date'].unique()).dt.to_pydatetime()

            calc_high_Low(self.data_df, "close", high_low_window)
            calc_jc_lines(self.data_df, "close", windows)
            calc_bolling_bands(self.data_df, "close", bolling_width)
            calc_macd(self.data_df, "close")

            self.data_df['min_price'] = self.data_df[['open', 'close']].min(axis=1)
            self.data_df['max_price'] = self.data_df[['open', 'close']].max(axis=1)

            self.data_df['price_range'] = self.data_df['max_price'] - self.data_df['min_price']

            self.data_df['low_pct_price_buy'] = self.data_df['min_price'] + (self.data_df['price_range']) * bar_low_percentile
            self.data_df['high_pct_price_buy'] = self.data_df['max_price'] - (self.data_df['price_range']) * bar_high_percentile

            self.data_df['low_pct_price_sell'] = self.data_df['min_price'] + (self.data_df['price_range']) * bar_high_percentile
            self.data_df['high_pct_price_sell'] = self.data_df['max_price'] - (self.data_df['price_range']) * bar_low_percentile

            self.data_df['upper_vegas'] = self.data_df[['ma_close144', 'ma_close169']].max(axis=1)
            self.data_df['lower_vegas'] = self.data_df[['ma_close144', 'ma_close169']].min(axis=1)

            guppy_lines = ['ma_close30', 'ma_close35', 'ma_close40', 'ma_close45', 'ma_close50', 'ma_close60']

            aligned_long_conditions = [self.data_df[guppy_lines[i]] > self.data_df[guppy_lines[i + 1]] for i in
                                       range(len(guppy_lines) - 1)]
            self.data_df['is_guppy_aligned_long'] = reduce(lambda left, right: left & right, aligned_long_conditions)

            aligned_short_conditions = [self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]] for i in
                                        range(len(guppy_lines) - 1)]
            self.data_df['is_guppy_aligned_short'] = reduce(lambda left, right: left & right, aligned_short_conditions)

            df_temp = self.data_df[guppy_lines]
            df_temp = df_temp.apply(sorted, axis=1).apply(pd.Series)
            df_temp.columns = ['guppy1', 'guppy2', 'guppy3', 'guppy4', 'guppy5', 'guppy6']
            self.data_df = pd.concat([self.data_df, df_temp], axis=1)

            self.data_df['highest_guppy'] = self.data_df['guppy6']
            self.data_df['lowest_guppy'] = self.data_df['guppy1']

            self.data_df['ma12_gradient'] = self.data_df['ma_close12'].diff()

            self.data_df['is_vegas_up_trend'] = self.data_df['ma_close144'] > self.data_df['ma_close169']
            self.data_df['is_vegas_down_trend'] = self.data_df['ma_close144'] < self.data_df['ma_close169']

            self.data_df['is_above_vegas'] = self.data_df['ma_close12'] > self.data_df['lower_vegas']
            self.data_df['is_above_vegas_strict'] = self.data_df['ma_close12'] > self.data_df['upper_vegas']

            self.data_df['is_below_vegas'] = self.data_df['ma_close12'] < self.data_df['upper_vegas']
            self.data_df['is_below_vegas_strict'] = self.data_df['ma_close12'] < self.data_df['lower_vegas']

            self.data_df['distance_to_upper_vegas'] = self.data_df['ma_close12'] - self.data_df['upper_vegas']
            self.data_df['distance_to_lower_vegas'] = self.data_df['ma_close12'] - self.data_df['lower_vegas']

            self.data_df['high_low_range'] = self.data_df['period_high' + str(high_low_window)] - self.data_df['period_low' + str(high_low_window)]

            self.data_df['pct_to_upper_vegas'] = self.data_df['distance_to_upper_vegas'] / self.data_df['high_low_range']
            self.data_df['pct_to_lower_vegas'] = self.data_df['distance_to_lower_vegas'] / self.data_df['high_low_range']


            feature_msg = ','.join([feature + "=" + str("%.4f" % self.data_df.iloc[-1][feature]) for feature in key_features])
            self.log_msg(feature_msg)

            self.data_df['buy_weak_ready'] = self.data_df['is_above_vegas'] & (
                        self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold) & (
                                                    self.data_df['high_pct_price_buy'] < self.data_df['ma_close12'])
            self.data_df['buy_weak_fire'] = self.data_df['is_above_vegas'] & (
                        self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold) & (
                                                   self.data_df['low_pct_price_buy'] > self.data_df['ma_close12']) \
                                       & (self.data_df['ma12_gradient'] >= 0)

            self.data_df['buy_ready'] = self.data_df['buy_weak_ready'] & self.data_df['is_vegas_up_trend']
            self.data_df['buy_fire'] = self.data_df['buy_weak_fire'] & self.data_df['is_vegas_up_trend']

            self.data_df['sell_weak_ready'] = self.data_df['is_below_vegas'] & (
                        self.data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold) & (
                                                     self.data_df['low_pct_price_sell'] > self.data_df['ma_close12'])
            self.data_df['sell_weak_fire'] = self.data_df['is_below_vegas'] & (
                        self.data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold) & (
                                                    self.data_df['high_pct_price_sell'] < self.data_df['ma_close12']) \
                                        & (self.data_df['ma12_gradient'] <= 0)

            self.data_df['sell_ready'] = self.data_df['sell_weak_ready'] & self.data_df['is_vegas_down_trend']
            self.data_df['sell_fire'] = self.data_df['sell_weak_fire'] & self.data_df['is_vegas_down_trend']



            signal_msg = ','.join([signal_attr + "=" + str(self.data_df.iloc[-1][signal_attr]) for signal_attr in signal_attrs])
            self.log_msg(signal_msg)
            if self.data_df.iloc[-1]['buy_weak_ready']:
                if self.data_df.iloc[-1]['buy_ready']:
                    self.log_msg("Ready to long " + self.currency)
                else:
                    self.log_msg("Ready to weakly long " + self.currency)

            if self.data_df.iloc[-1]['buy_weak_fire']:
                if self.data_df.iloc[-1]['buy_fire']:
                    self.log_msg("Long " + self.currency)
                else:
                    self.log_msg("Weakly long " + self.currency)


            if self.data_df.iloc[-1]['sell_weak_ready']:
                if self.data_df.iloc[-1]['sell_ready']:
                    self.log_msg("Ready to short " + self.currency)
                else:
                    self.log_msg("Ready to weakly short " + self.currency)

            if self.data_df.iloc[-1]['sell_weak_fire']:
                if self.data_df.iloc[-1]['sell_fire']:
                    self.log_msg("Short " + self.currency)
                else:
                    self.log_msg("Weakly short " + self.currency)



            self.data_df.to_csv(self.currency_file, index=False)

            print_prefix = "[Currency " + self.currency + "] "

            for file in os.listdir(self.chart_folder):
                file_path = os.path.join(self.chart_folder, file)
                os.remove(file_path)

            plot_candle_bar_charts(self.currency, self.data_df, all_days,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   bar_fig_folder=self.chart_folder)


            self.data_df = self.data_df[['currency', 'time','open','high','low','close']]


            self.condition.wait()


        self.condition.release()











