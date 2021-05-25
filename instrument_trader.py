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

bar_low_percentile = 0.5
bar_high_percentile = 0.1

initial_bar_number = 400

distance_to_vegas_threshold = 0.20

vegas_width_threshold = 10

maximum_loss = 400
minimum_profit = 70
minimum_profilt_loss_ratio = 1/3

ma12_lookback = 10

vegas_tolerate = 30

bolling_threshold = 10

enter_lot = 1
maximum_tolerable_loss = 300

c5_lookback = 3

enter_bar_width_threshold = 10

guppy_tolerate = 20

class CurrencyTrader(threading.Thread):

    def __init__(self, condition, currency, lot_size, exchange_rate,  data_folder, chart_folder, simple_chart_folder, log_file):
        super().__init__(name = currency)
        self.condition = condition
        self.currency = currency
        self.lot_size = lot_size
        self.exchange_rate = exchange_rate
        self.data_folder = data_folder
        self.chart_folder = chart_folder
        self.simple_chart_folder = simple_chart_folder
        self.data_df = None
        self.last_time = None
        self.log_file = log_file

        self.use_relaxed_vegas_support = True
        self.is_require_m12_strictly_above_vegas = False
        self.remove_c12 = True

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
                    self.data_df.reset_index(inplace = True)
                    self.data_df = self.data_df.drop(columns = ['index'])

                    self.last_time = new_last_time
                    self.condition.notify()

        self.condition.release()

    def run(self):

        self.trade()

    def trade(self):

        numerical_features = [ 'pct_to_upper_vegas', 'high_pct_price_buy', 'low_pct_price_buy', 'pct_to_lower_vegas', 'low_pct_price_sell', 'high_pct_price_sell',
                        'ma_close12', 'ma12_gradient']

        bool_features = ['is_above_vegas',  'is_vegas_up_trend', 'is_below_vegas', 'is_vegas_down_trend']

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


            # self.data_df['prev_open'] = self.data_df['open'].shift(1)
            # self.data_df['prev_close'] = self.data_df['close'].shift(1)

            self.data_df['is_positive'] = (self.data_df['close'] > self.data_df['open'])
            self.data_df['is_negative'] = (self.data_df['close'] < self.data_df['open'])

            self.data_df['prev1_open'] = self.data_df['open'].shift(1)
            self.data_df['prev1_close'] = self.data_df['close'].shift(1)

            for i in range(2, c5_lookback + 1):
                self.data_df['prev' + str(i) + '_open'] = self.data_df['prev' + str(i-1) + '_open'].shift(1)
                self.data_df['prev' + str(i) + '_close'] = self.data_df['prev' + str(i - 1) + '_close'].shift(1)
                #
                # self.data_df['prev' + str(i) + '_is_positive'] = self.data_df['prev' + str(i-1) + '_is_positive'].shift(1)
                # self.data_df['prev' + str(i) + '_is_negative'] = self.data_df['prev' + str(i-1) + '_is_negative'].shift(1)
                #
                # self.data_df.at[0, 'prev' + str(i) + '_is_positive'] = False
                # self.data_df['prev' + str(i) + '_is_positive'] = pd.Series(list(self.data_df['prev' + str(i) + '_is_positive']), dtype='bool')
                #
                # self.data_df.at[0, 'prev' + str(i) + '_is_negative'] = False
                # self.data_df['prev' + str(i) + '_is_negative'] = pd.Series(list(self.data_df['prev' + str(i) + '_is_negative']), dtype='bool')


            self.data_df['min_price'] = self.data_df[['open', 'close']].min(axis=1)
            self.data_df['max_price'] = self.data_df[['open', 'close']].max(axis=1)

            self.data_df['prev1_min_price'] = self.data_df['min_price'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_min_price'] = self.data_df['prev' + str(i-1) + '_min_price'].shift(1)

            self.data_df['prev1_max_price'] = self.data_df['max_price'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_max_price'] = self.data_df['prev' + str(i - 1) + '_max_price'].shift(1)


            self.data_df['price_range'] = self.data_df['max_price'] - self.data_df['min_price']

            self.data_df['low_pct_price_buy'] = self.data_df['min_price'] + (self.data_df['price_range']) * bar_low_percentile
            self.data_df['high_pct_price_buy'] = self.data_df['max_price'] - (self.data_df['price_range']) * bar_high_percentile

            self.data_df['low_pct_price_sell'] = self.data_df['min_price'] + (self.data_df['price_range']) * bar_high_percentile
            self.data_df['high_pct_price_sell'] = self.data_df['max_price'] - (self.data_df['price_range']) * bar_low_percentile

            self.data_df['upper_vegas'] = self.data_df[['ma_close144', 'ma_close169']].max(axis=1)
            self.data_df['lower_vegas'] = self.data_df[['ma_close144', 'ma_close169']].min(axis=1)

            self.data_df['upper_vegas_gradient'] = self.data_df['upper_vegas'].diff()
            self.data_df['lower_vegas_gradient'] = self.data_df['lower_vegas'].diff()

            self.data_df['prev1_upper_vegas'] = self.data_df['upper_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_upper_vegas'] = self.data_df['prev' + str(i-1) + '_upper_vegas'].shift(1)

            self.data_df['prev1_lower_vegas'] = self.data_df['lower_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_lower_vegas'] = self.data_df['prev' + str(i-1) + '_lower_vegas'].shift(1)

            self.data_df['prev_upper_band_close'] = self.data_df['upper_band_close'].shift(1)
            self.data_df['prev_lower_band_close'] = self.data_df['lower_band_close'].shift(1)


            guppy_lines = ['ma_close30', 'ma_close35', 'ma_close40', 'ma_close45', 'ma_close50', 'ma_close60']
            for guppy_line in guppy_lines:
                self.data_df[guppy_line + '_gradient'] = self.data_df[guppy_line].diff()


            aligned_long_conditions = [(self.data_df[guppy_lines[i]] > self.data_df[guppy_lines[i + 1]]) for i in
                                       range(len(guppy_lines) - 1)]
            all_up_conditions = [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines]
            self.data_df['is_guppy_aligned_long'] = reduce(lambda left, right: left & right, aligned_long_conditions + all_up_conditions)


            aligned_short_conditions = [(self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]]) for i in
                                        range(len(guppy_lines) - 1)]
            all_down_conditions = [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]
            self.data_df['is_guppy_aligned_short'] = reduce(lambda left, right: left & right, aligned_short_conditions + all_down_conditions)

            df_temp = self.data_df[guppy_lines]
            df_temp = df_temp.apply(sorted, axis=1).apply(pd.Series)
            df_temp.columns = ['guppy1', 'guppy2', 'guppy3', 'guppy4', 'guppy5', 'guppy6']
            self.data_df = pd.concat([self.data_df, df_temp], axis=1)

            self.data_df['highest_guppy'] = self.data_df['guppy6']
            self.data_df['lowest_guppy'] = self.data_df['guppy1']

            self.data_df['prev1_ma_close12'] = self.data_df['ma_close12'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_ma_close12'] = self.data_df['prev' + str(i-1) + '_ma_close12'].shift(1)

            self.data_df['ma12_gradient'] = self.data_df['ma_close12'].diff()
            self.data_df['prev1_ma12_gradient'] = self.data_df['ma12_gradient'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_ma12_gradient'] = self.data_df['prev' + str(i-1) + '_ma12_gradient'].shift(1)

            self.data_df['upper_band_close_gradient'] = self.data_df['upper_band_close'].diff()
            self.data_df['lower_band_close_gradient'] = self.data_df['lower_band_close'].diff()


            self.data_df['is_vegas_up_trend'] = self.data_df['ma_close144'] > self.data_df['ma_close169']
            self.data_df['is_vegas_down_trend'] = self.data_df['ma_close144'] < self.data_df['ma_close169']

            self.data_df['vegas_width'] = (self.data_df['ma_close144'] - self.data_df['ma_close169'])
            self.data_df['is_vegas_enough_up_trend'] = self.data_df['vegas_width'] * self.lot_size * self.exchange_rate > vegas_width_threshold
            self.data_df['is_vegas_enough_down_trend'] = self.data_df['vegas_width'] * self.lot_size * self.exchange_rate < -vegas_width_threshold



            self.data_df['is_above_vegas'] = self.data_df['ma_close12'] > self.data_df['lower_vegas']
            self.data_df['is_above_vegas_strict'] = self.data_df['ma_close12'] > self.data_df['upper_vegas']

            self.data_df['is_above_vegas_tolerated'] = (self.data_df['ma_close12'] - self.data_df['lower_vegas']) * self.lot_size * self.exchange_rate > -vegas_tolerate

            self.data_df['is_above_vegas_prev1'] = self.data_df['is_above_vegas_tolerated'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['is_above_vegas_prev' + str(i)] = self.data_df['is_above_vegas_prev' + str(i-1)].shift(1)

            self.data_df['is_above_vegas_until_prev1'] = self.data_df['is_above_vegas'] & self.data_df['is_above_vegas_prev1']
            for i in range(2, ma12_lookback + 1):
                self.data_df['is_above_vegas_until_prev' + str(i)] = self.data_df['is_above_vegas_until_prev' + str(i-1)] & self.data_df['is_above_vegas_prev' + str(i)]



            self.data_df['is_below_vegas'] = self.data_df['ma_close12'] < self.data_df['upper_vegas']
            self.data_df['is_below_vegas_strict'] = self.data_df['ma_close12'] < self.data_df['lower_vegas']

            self.data_df['is_below_vegas_tolerated'] = (self.data_df['ma_close12'] - self.data_df['upper_vegas']) * self.lot_size * self.exchange_rate < vegas_tolerate

            self.data_df['is_below_vegas_prev1'] = self.data_df['is_below_vegas_tolerated'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['is_below_vegas_prev' + str(i)] = self.data_df['is_below_vegas_prev' + str(i - 1)].shift(1)

            self.data_df['is_below_vegas_until_prev1'] = self.data_df['is_below_vegas'] & self.data_df['is_below_vegas_prev1']
            for i in range(2, ma12_lookback + 1):
                self.data_df['is_below_vegas_until_prev' + str(i)] = self.data_df['is_below_vegas_until_prev' + str(i-1)] & self.data_df['is_below_vegas_prev' + str(i)]

            self.data_df['high_low_range'] = self.data_df['period_high' + str(high_low_window)] - self.data_df['period_low' + str(high_low_window)]





            self.data_df['price_pct_to_upper_vegas'] = (self.data_df['low'] - self.data_df['upper_vegas']) / self.data_df['high_low_range']
            self.data_df['price_pct_to_lower_vegas'] = (self.data_df['high'] - self.data_df['lower_vegas']) / self.data_df['high_low_range']

            self.data_df['prev1_price_pct_to_upper_vegas'] = self.data_df['price_pct_to_upper_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_price_pct_to_upper_vegas'] = self.data_df['prev' + str(i-1) + '_price_pct_to_upper_vegas'].shift(1)

            self.data_df['prev1_price_pct_to_lower_vegas'] = self.data_df['price_pct_to_lower_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_price_pct_to_lower_vegas'] = self.data_df['prev' + str(i-1) + '_price_pct_to_lower_vegas'].shift(1)


            recent_supported_by_vegas = reduce(lambda left, right: left | right,
                                               [((self.data_df['prev' + str(i) + '_price_pct_to_upper_vegas'] < distance_to_vegas_threshold) & \
                                                (self.data_df['prev' + str(i) + '_max_price'] <= self.data_df['prev' + str(i) + '_ma_close12']))
                                                for i in range(1, ma12_lookback + 1)])

            recent_suppressed_by_vegas = reduce(lambda left, right: left | right,
                                               [((self.data_df['prev' + str(i) + '_price_pct_to_lower_vegas'] > -distance_to_vegas_threshold) & \
                                                (self.data_df['prev' + str(i) + '_min_price'] >= self.data_df['prev' + str(i) + '_ma_close12']))
                                                for i in range(1, ma12_lookback + 1)])




            self.data_df['distance_to_upper_vegas'] = self.data_df['ma_close12'] - self.data_df['upper_vegas']
            self.data_df['distance_to_lower_vegas'] = self.data_df['ma_close12'] - self.data_df['lower_vegas']

            self.data_df['price_to_upper_vegas'] = self.data_df['close'] - self.data_df['upper_vegas']
            self.data_df['price_to_lower_vegas'] = self.data_df['close'] - self.data_df['lower_vegas']

            self.data_df['price_to_bolling_upper'] = self.data_df['close'] - self.data_df['upper_band_close']
            self.data_df['price_to_bolling_lower'] = self.data_df['close'] - self.data_df['lower_band_close']



            self.data_df['pct_to_upper_vegas'] = self.data_df['distance_to_upper_vegas'] / self.data_df['high_low_range']
            self.data_df['pct_to_lower_vegas'] = self.data_df['distance_to_lower_vegas'] / self.data_df['high_low_range']

            if self.use_relaxed_vegas_support:
                final_recent_supported_by_vegas = (recent_supported_by_vegas) | (self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold)
                final_recent_suppressed_by_vegas = (recent_suppressed_by_vegas) | (self.data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold)
            else:
                final_recent_supported_by_vegas = self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold
                final_recent_suppressed_by_vegas = self.data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold


            feature_msg = ','.join([feature + "=" + str(self.data_df.iloc[-1][feature]) for feature in bool_features] + [feature + "=" + str("%.4f" % self.data_df.iloc[-1][feature]) for feature in numerical_features])
            self.log_msg(feature_msg)

            #is_above_vegas = "is_above_vegas_strict" if self.is_require_m12_strictly_above_vegas else "is_above_vegas"

            if self.is_require_m12_strictly_above_vegas:
                above_cond = self.data_df['is_above_vegas_strict']
            else:
                above_cond = self.data_df['is_above_vegas_strict'] | (self.data_df['is_above_vegas'] & (self.data_df['upper_vegas_gradient'] > 0) & (self.data_df['lower_vegas_gradient'] > 0))

            self.data_df['buy_weak_ready'] = self.data_df['is_above_vegas'] & (
                        final_recent_supported_by_vegas) & ( #self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold
                                                    self.data_df['high_pct_price_buy'] < self.data_df['ma_close12'])
            self.data_df['buy_weak_fire'] = above_cond & ( #'is_above_vegas_strict'
                        final_recent_supported_by_vegas) & (self.data_df['low_pct_price_buy'] > self.data_df['ma_close12']) \
                                       & (self.data_df['ma12_gradient'] >= 0) & (self.data_df['close'] > self.data_df['ma_close12']) \
                                            & ((self.data_df['close'] - self.data_df['open']) * self.lot_size * self.exchange_rate > enter_bar_width_threshold)

            self.data_df['buy_ready'] = self.data_df['buy_weak_ready'] & self.data_df['is_vegas_up_trend']
            self.data_df['buy_fire'] = self.data_df['buy_weak_fire'] & self.data_df['is_vegas_up_trend']
            self.data_df['buy_real_fire'] = self.data_df['buy_fire'] & self.data_df['is_vegas_enough_up_trend']

            buy_c11 = self.data_df['price_to_lower_vegas'] * self.lot_size * self.exchange_rate < maximum_loss
            buy_c12 = self.data_df['price_to_bolling_upper'] * self.lot_size * self.exchange_rate < -minimum_profit
            buy_c13 = (-self.data_df['price_to_bolling_upper'] / self.data_df['price_to_lower_vegas']) > minimum_profilt_loss_ratio

            buy_c2 = self.data_df['is_guppy_aligned_short']

            buy_c3_strong = reduce(lambda left, right: left | right,
                            [((self.data_df['prev' + str(i) + '_ma12_gradient'] < 0) & \
                              (self.data_df['prev' + str(i) + '_ma_close12'] > self.data_df['prev' + str(i) + '_upper_vegas']) & \
                              (self.data_df['is_above_vegas_until_prev' + str(i)])) for i in range(1, ma12_lookback + 1)])

            buy_c3_weak = reduce(lambda left, right: left | right,
                            [((self.data_df['prev' + str(i) + '_ma12_gradient'] < 0) & \
                              (self.data_df['prev' + str(i) + '_ma_close12'] > self.data_df['prev' + str(i) + '_lower_vegas']) & \
                              (self.data_df['is_above_vegas_until_prev' + str(i)])) for i in range(1, ma12_lookback + 1)])

            buy_c3 = buy_c3_strong | (buy_c3_weak & ((self.data_df['highest_guppy'] - self.data_df['lower_vegas']) * self.lot_size * self.exchange_rate > -guppy_tolerate))

            #buy_c3 = (self.data_df['prev1_ma12_gradient'] < 0) | (self.data_df['prev2_ma12_gradient'] < 0) | (self.data_df['prev3_ma12_gradient'] < 0) | (self.data_df['prev4_ma12_gradient'] < 0) | (self.data_df['prev5_ma12_gradient'] < 0)
            #buy_c4 = self.data_df['high'] > self.data_df['upper_band_close']
            buy_c41 = self.data_df['high'] > self.data_df['upper_band_close'] #self.data_df['prev_upper_band_close']
            buy_c42 = self.data_df['upper_band_close_gradient'] * self.lot_size * self.exchange_rate > 0# bolling_threshold
            buy_c43 = self.data_df['is_positive'] & (self.data_df['prev1_open'] < self.data_df['prev1_close'])
            buy_c4 = buy_c41 & buy_c42 & buy_c43

            buy_c5 = reduce(lambda left, right: left & right, [((self.data_df['prev' + str(i) + '_open'] - self.data_df['prev' + str(i) + '_close']) * self.lot_size * self.exchange_rate > enter_bar_width_threshold)
                                                               for i in range(1,c5_lookback + 1)])



            if not self.remove_c12:
                self.data_df['buy_real_fire'] = (self.data_df['buy_real_fire']) & (buy_c3) & (~buy_c5) & ((buy_c4) | (((buy_c12) | (buy_c13)) & (~buy_c2)))
            else:
                self.data_df['buy_real_fire'] = (self.data_df['buy_real_fire']) & (buy_c3) & (~buy_c5) & ((buy_c4) | ((buy_c13) & (~buy_c2)))


            self.data_df['buy_c11'] = buy_c11
            self.data_df['buy_c12'] = buy_c12
            self.data_df['buy_c13'] = buy_c13
            self.data_df['buy_c2'] = buy_c2
            self.data_df['buy_c3'] = buy_c3
            self.data_df['buy_c41'] = buy_c41
            self.data_df['buy_c42'] = buy_c42
            self.data_df['buy_c43'] = buy_c43
            self.data_df['buy_c5'] = buy_c5



            self.data_df['prev_buy_weak_fire'] = self.data_df['buy_weak_fire'].shift(1)
            self.data_df.at[0, 'prev_buy_weak_fire'] = False
            self.data_df['prev_buy_weak_fire'] = pd.Series(list(self.data_df['prev_buy_weak_fire']), dtype='bool')
            self.data_df['first_buy_weak_fire'] = self.data_df['buy_weak_fire'] & (~self.data_df['prev_buy_weak_fire'])


            self.data_df['prev_buy_fire'] = self.data_df['buy_fire'].shift(1)
            self.data_df.at[0, 'prev_buy_fire'] = False
            self.data_df['prev_buy_fire'] = pd.Series(list(self.data_df['prev_buy_fire']), dtype = 'bool')
            self.data_df['first_buy_fire'] = self.data_df['buy_fire'] & (~self.data_df['prev_buy_fire'])

            self.data_df['prev_buy_real_fire'] = self.data_df['buy_real_fire'].shift(1)
            self.data_df.at[0, 'prev_buy_real_fire'] = False
            self.data_df['prev_buy_real_fire'] = pd.Series(list(self.data_df['prev_buy_real_fire']), dtype='bool')
            self.data_df['first_buy_real_fire'] = self.data_df['buy_real_fire'] & (~self.data_df['prev_buy_real_fire'])


            # print(type(self.data_df.iloc[0]['first_buy_fire']))
            # print(type(self.data_df.iloc[0]['buy_fire']))
            # print(type(self.data_df.iloc[0]['prev_buy_fire']))
            #
            # print("debug first_buy_fire:")
            # print(self.data_df[['time','first_buy_fire','buy_fire','prev_buy_fire']].tail(50))
            # print("")


            #is_below_vegas = "is_below_vegas_strict" if self.is_require_m12_strictly_above_vegas else "is_below_vegas"

            if self.is_require_m12_strictly_above_vegas:
                below_cond = self.data_df['is_below_vegas_strict']
            else:
                below_cond = self.data_df['is_below_vegas_strict'] | (self.data_df['is_below_vegas'] & (self.data_df['upper_vegas_gradient'] < 0) & (self.data_df['lower_vegas_gradient'] < 0))

            self.data_df['sell_weak_ready'] = self.data_df['is_below_vegas'] & (
                        final_recent_suppressed_by_vegas) & (#self.data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold
                                                     self.data_df['low_pct_price_sell'] > self.data_df['ma_close12'])
            self.data_df['sell_weak_fire'] = below_cond & ( #is_below_vegas_strict
                        final_recent_suppressed_by_vegas) & (self.data_df['high_pct_price_sell'] < self.data_df['ma_close12']) \
                                        & (self.data_df['ma12_gradient'] <= 0) & (self.data_df['close'] < self.data_df['ma_close12']) \
                                             & ((self.data_df['close'] - self.data_df['open']) * self.lot_size * self.exchange_rate < -enter_bar_width_threshold)

            self.data_df['sell_ready'] = self.data_df['sell_weak_ready'] & self.data_df['is_vegas_down_trend']
            self.data_df['sell_fire'] = self.data_df['sell_weak_fire'] & self.data_df['is_vegas_down_trend']
            self.data_df['sell_real_fire'] = self.data_df['sell_fire'] & self.data_df['is_vegas_enough_down_trend']

            sell_c11 = self.data_df['price_to_upper_vegas'] * self.lot_size * self.exchange_rate > -maximum_loss
            sell_c12 = self.data_df['price_to_bolling_lower'] * self.lot_size * self.exchange_rate > minimum_profit
            sell_c13 = (-self.data_df['price_to_bolling_lower'] / self.data_df['price_to_upper_vegas']) > minimum_profilt_loss_ratio

            sell_c2 = self.data_df['is_guppy_aligned_long']


            sell_c3_strong = reduce(lambda left, right: left | right,
                            [((self.data_df['prev' + str(i) + '_ma12_gradient'] > 0) & \
                              (self.data_df['prev' + str(i) + '_ma_close12'] < self.data_df['prev' + str(i) + '_lower_vegas']) & \
                              (self.data_df['is_below_vegas_until_prev' + str(i)])) for i in range(1, ma12_lookback + 1)])

            sell_c3_weak = reduce(lambda left, right: left | right,
                            [((self.data_df['prev' + str(i) + '_ma12_gradient'] > 0) & \
                              (self.data_df['prev' + str(i) + '_ma_close12'] < self.data_df['prev' + str(i) + '_upper_vegas']) & \
                              (self.data_df['is_below_vegas_until_prev' + str(i)])) for i in range(1, ma12_lookback + 1)])

            sell_c3 = sell_c3_strong | (sell_c3_weak & ((self.data_df['lowest_guppy'] - self.data_df['upper_vegas']) * self.lot_size * self.exchange_rate < guppy_tolerate))

            #sell_c3 = (self.data_df['prev1_ma12_gradient'] > 0) | (self.data_df['prev2_ma12_gradient'] > 0) | (self.data_df['prev3_ma12_gradient'] > 0) | (self.data_df['prev4_ma12_gradient'] > 0) | (self.data_df['prev5_ma12_gradient'] > 0)
            #sell_c4 = self.data_df['low'] < self.data_df['lower_band_close']
            sell_c41 = self.data_df['low'] < self.data_df['lower_band_close'] # self.data_df['prev_lower_band_close']
            sell_c42 = self.data_df['lower_band_close_gradient'] * self.lot_size * self.exchange_rate < 0 #-bolling_threshold
            sell_c43 = self.data_df['is_negative'] &  (self.data_df['prev1_open'] > self.data_df['prev1_close'])
            sell_c4 = sell_c41 & sell_c42 & sell_c43

            sell_c5 = reduce(lambda left, right: left & right, [((self.data_df['prev' + str(i) + '_open'] - self.data_df['prev' + str(i) + '_close']) * self.lot_size * self.exchange_rate < -enter_bar_width_threshold )
                                                                for i in range(1,c5_lookback + 1)])

            if not self.remove_c12:
                self.data_df['sell_real_fire'] = (self.data_df['sell_real_fire']) & (sell_c3) & (~sell_c5) & ((sell_c4) | (((sell_c12) | (sell_c13)) & (~sell_c2)))
            else:
                self.data_df['sell_real_fire'] = (self.data_df['sell_real_fire']) & (sell_c3) & (~sell_c5) & ((sell_c4) | ((sell_c13) & (~sell_c2)))



            self.data_df['sell_c11'] = sell_c11
            self.data_df['sell_c12'] = sell_c12
            self.data_df['sell_c13'] = sell_c13
            self.data_df['sell_c2'] = sell_c2
            self.data_df['sell_c3'] = sell_c3
            self.data_df['sell_c41'] = sell_c41
            self.data_df['sell_c42'] = sell_c42
            self.data_df['sell_c43'] = sell_c43
            self.data_df['sell_c5'] = sell_c5


            self.data_df['prev_sell_weak_fire'] = self.data_df['sell_weak_fire'].shift(1)
            self.data_df.at[0, 'prev_sell_weak_fire'] = False
            self.data_df['prev_sell_weak_fire'] = pd.Series(list(self.data_df['prev_sell_weak_fire']), dtype='bool')
            self.data_df['first_sell_weak_fire'] = self.data_df['sell_weak_fire'] & (~self.data_df['prev_sell_weak_fire'])

            self.data_df['prev_sell_fire'] = self.data_df['sell_fire'].shift(1)
            self.data_df.at[0, 'prev_sell_fire'] = False
            self.data_df['prev_sell_fire'] = pd.Series(list(self.data_df['prev_sell_fire']), dtype='bool')
            self.data_df['first_sell_fire'] = self.data_df['sell_fire'] & (~self.data_df['prev_sell_fire'])

            self.data_df['prev_sell_real_fire'] = self.data_df['sell_real_fire'].shift(1)
            self.data_df.at[0, 'prev_sell_real_fire'] = False
            self.data_df['prev_sell_real_fire'] = pd.Series(list(self.data_df['prev_sell_real_fire']), dtype='bool')
            self.data_df['first_sell_real_fire'] = self.data_df['sell_real_fire'] & (~self.data_df['prev_sell_real_fire'])


            signal_msg = ','.join([signal_attr + "=" + str(self.data_df.iloc[-1][signal_attr]) for signal_attr in signal_attrs])
            self.log_msg(signal_msg)

            current_time = self.data_df.iloc[-1]['time'] + timedelta(seconds = 3600)
            current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

            if self.data_df.iloc[-1]['buy_weak_ready']:
                if self.data_df.iloc[-1]['buy_ready']:
                    msg = "Ready to long " + self.currency + " at " + current_time + ", last_price = " + str("%.5f" % self.data_df.iloc[-1]['close'])
                    self.log_msg(msg)
                    #sendEmail(msg, msg)
                else:
                    msg = "Ready to weakly long " + self.currency + " at " + current_time + ", last_price = " + str("%.5f" % self.data_df.iloc[-1]['close'])
                    self.log_msg(msg)
                    #sendEmail(msg, msg)



            if self.data_df.iloc[-1]['first_buy_real_fire']:

                enter_price = self.data_df.iloc[-1]['close']
                stop_loss_price = self.data_df.iloc[-1]['lower_vegas']

                expected_loss = (enter_price - stop_loss_price) * self.lot_size * self.exchange_rate

                actual_enter_lot = maximum_tolerable_loss / expected_loss * enter_lot
                if actual_enter_lot > 1:
                    actual_enter_lot = 1

                msg = "Strongly Long " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close']) + " with " + str("%.2f" % actual_enter_lot) + " lot"
                self.log_msg(msg)
                self.log_msg("enter_price = " + str(enter_price) + " stop_loss_price = " + str(stop_loss_price) + " expected_loss = " + str(expected_loss))
                self.log_msg("********************************")
                sendEmail(msg, msg)

            elif self.data_df.iloc[-1]['first_buy_fire']:
                msg = "Long " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close'])
                self.log_msg(msg)
                self.log_msg("********************************")
                #sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_buy_weak_fire']:
                msg = "Weakly Long " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close'])
                self.log_msg(msg)
                self.log_msg("********************************")
                #sendEmail(msg, msg)



            if self.data_df.iloc[-1]['sell_weak_ready']:
                if self.data_df.iloc[-1]['sell_ready']:
                    msg = "Ready to short " + self.currency + " at " + current_time + ", last_price = " + str("%.5f" % self.data_df.iloc[-1]['close'])
                    self.log_msg(msg)
                    #sendEmail(msg, msg)
                else:
                    msg = "Ready to weakly short " + self.currency + " at " + current_time + ", last_price = " + str("%.5f" % self.data_df.iloc[-1]['close'])
                    self.log_msg(msg)
                    #sendEmail(msg, msg)


            if self.data_df.iloc[-1]['first_sell_real_fire']:

                enter_price = self.data_df.iloc[-1]['close']
                stop_loss_price = self.data_df.iloc[-1]['upper_vegas']

                expected_loss = (stop_loss_price - enter_price) * self.lot_size * self.exchange_rate

                actual_enter_lot = maximum_tolerable_loss / expected_loss * enter_lot
                if actual_enter_lot > 1:
                    actual_enter_lot = 1


                msg = "Strongly Short " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close']) + " with " + str("%.2f" % actual_enter_lot) + " lot"
                self.log_msg(msg)
                self.log_msg("enter_price = " + str(enter_price) + " stop_loss_price = " + str(stop_loss_price) + " expected_loss = " + str(expected_loss))
                self.log_msg("********************************")
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_sell_fire']:
                msg = "Short " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close'])
                self.log_msg(msg)
                self.log_msg("********************************")
                #sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_sell_weak_fire']:
                msg = "Weakly Short " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close'])
                self.log_msg(msg)
                self.log_msg("********************************")
                #sendEmail(msg, msg)



            self.log_msg("\n")



            self.data_df.to_csv(self.currency_file, index=False)

            print_prefix = "[Currency " + self.currency + "] "

            for file in os.listdir(self.chart_folder):
                file_path = os.path.join(self.chart_folder, file)
                os.remove(file_path)

            for file in os.listdir(self.simple_chart_folder):
                file_path = os.path.join(self.simple_chart_folder, file)
                os.remove(file_path)

            plot_candle_bar_charts(self.currency, self.data_df, all_days,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   bar_fig_folder=self.chart_folder, is_plot_simple_chart=False)

            plot_candle_bar_charts(self.currency, self.data_df, all_days,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   bar_fig_folder=self.simple_chart_folder, is_plot_simple_chart=True)


            self.data_df = self.data_df[['currency', 'time','open','high','low','close']]


            self.condition.wait()


        self.condition.release()











