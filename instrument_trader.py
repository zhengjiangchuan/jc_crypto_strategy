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

vegas_bar_percentile = 0.2

initial_bar_number = 400

distance_to_vegas_threshold = 0.20
tight_distance_to_vegas_threshold = 0.05

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

maximum_enter_bar_length = 100

price_range_lookback_window = 3 #Change this to 15  used to be 3

bar_increase_threshold = 1.5 #1.5 for mean

large_bar_look_back = 15
skip_bar_num = 2

large_bar_consider_past_num = 2

price_to_period_range_pct = 0.10
price_to_period_range_pct_strict = 0.02

vegas_look_back = 120
vegas_trend_pct_threshold = 0.8

vegas_short_look_back = 10

vagas_fast_support_threshold = 10

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


            self.data_df['period_high_low_range'] = self.data_df['period_high' + str(high_low_window)] - self.data_df['period_low' + str(high_low_window)]
            self.data_df['price_to_period_low_pct'] = (self.data_df['close'] - self.data_df['period_low' + str(high_low_window)]) / self.data_df['period_high_low_range']
            self.data_df['price_to_period_high_pct'] = (self.data_df['period_high' + str(high_low_window)] - self.data_df['close']) / self.data_df['period_high_low_range']

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
            for i in range(2, max(ma12_lookback, large_bar_look_back) + 1):
                self.data_df['prev' + str(i) + '_min_price'] = self.data_df['prev' + str(i-1) + '_min_price'].shift(1)

            self.data_df['prev1_max_price'] = self.data_df['max_price'].shift(1)
            for i in range(2, max(ma12_lookback, large_bar_look_back) + 1):
                self.data_df['prev' + str(i) + '_max_price'] = self.data_df['prev' + str(i - 1) + '_max_price'].shift(1)


            self.data_df['price_range'] = self.data_df['max_price'] - self.data_df['min_price']
            self.data_df['prev_price_range'] = self.data_df['price_range'].shift(1)

            self.data_df['positive_price_range'] = np.where(self.data_df['close'] > self.data_df['open'],
                                                              self.data_df['price_range'],
                                                              0)
            self.data_df['negative_price_range'] = np.where(self.data_df['close'] < self.data_df['open'],
                                                              self.data_df['price_range'],
                                                              0)


            guppy_lines = ['ma_close30', 'ma_close35', 'ma_close40', 'ma_close45', 'ma_close50', 'ma_close60']
            for guppy_line in guppy_lines:
                self.data_df[guppy_line + '_gradient'] = self.data_df[guppy_line].diff()

            for guppy_line in guppy_lines:
                self.data_df[guppy_line + '_up'] = np.where(
                    self.data_df[guppy_line + '_gradient'] > 0,
                    1,
                    0
                )

            self.data_df['up_guppy_line_num'] = reduce(lambda left, right: left + right, [self.data_df[guppy_line + '_up'] for guppy_line in guppy_lines])
            self.data_df['down_guppy_line_num'] = len(guppy_lines) - self.data_df['up_guppy_line_num']


            aligned_long_conditions1 = [(self.data_df[guppy_lines[i]] > self.data_df[guppy_lines[i + 1]]) for i in
                                       range(len(guppy_lines) - 1)]
            #all_up_conditions = [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines]
            aligned_long_conditions2 = aligned_long_conditions1[0:2] + [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines[0:3]]
            #self.data_df['is_guppy_aligned_long'] = reduce(lambda left, right: left & right, aligned_long_conditions) # + all_up_conditions)
            aligned_long_condition1 = reduce(lambda left, right: left & right, aligned_long_conditions1)
            aligned_long_condition2 = reduce(lambda left, right: left & right, aligned_long_conditions2)

            half_aligned_long_condition = reduce(lambda left, right: left & right, aligned_long_conditions1[0:2])
            strongly_half_aligned_long_condition = aligned_long_condition2

            self.data_df['is_guppy_aligned_long'] = aligned_long_condition1 #| aligned_long_condition2
            self.data_df['strongly_half_aligned_long_condition'] = strongly_half_aligned_long_condition


            aligned_short_conditions1 = [(self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]]) for i in
                                        range(len(guppy_lines) - 1)]
            #all_down_conditions = [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]
            aligned_short_conditions2 = aligned_short_conditions1[0:2] + [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines[0:3]]
            #self.data_df['is_guppy_aligned_short'] = reduce(lambda left, right: left & right, aligned_short_conditions) # + all_down_conditions)
            aligned_short_condition1 = reduce(lambda left, right: left & right, aligned_short_conditions1)
            aligned_short_condition2 = reduce(lambda left, right: left & right, aligned_short_conditions2)

            half_aligned_short_condition = reduce(lambda left, right: left & right, aligned_short_conditions1[0:2])
            strongly_half_aligned_short_condition = aligned_short_condition2

            self.data_df['is_guppy_aligned_short'] = aligned_short_condition1 # | aligned_short_condition2
            self.data_df['strongly_half_aligned_short_condition'] = strongly_half_aligned_short_condition

            df_temp = self.data_df[guppy_lines]
            df_temp = df_temp.apply(sorted, axis=1).apply(pd.Series)
            sorted_guppys = ['guppy1', 'guppy2', 'guppy3', 'guppy4', 'guppy5', 'guppy6']
            df_temp.columns = sorted_guppys
            self.data_df = pd.concat([self.data_df, df_temp], axis=1)

            self.data_df['highest_guppy'] = self.data_df['guppy6']
            self.data_df['lowest_guppy'] = self.data_df['guppy1']

            self.data_df['prev1_highest_guppy'] = self.data_df['highest_guppy'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_highest_guppy'] = self.data_df['prev' + str(i-1) + '_highest_guppy'].shift(1)

            self.data_df['prev1_lowest_guppy'] = self.data_df['lowest_guppy'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_lowest_guppy'] = self.data_df['prev' + str(i-1) + '_lowest_guppy'].shift(1)


            self.data_df['upper_vegas'] = self.data_df[['ma_close144', 'ma_close169']].max(axis=1)
            self.data_df['lower_vegas'] = self.data_df[['ma_close144', 'ma_close169']].min(axis=1)

            self.data_df['upper_vegas_gradient'] = self.data_df['upper_vegas'].diff()
            self.data_df['lower_vegas_gradient'] = self.data_df['lower_vegas'].diff()

            self.data_df['prev_upper_vegas_gradient'] = self.data_df['upper_vegas_gradient'].shift(1)
            self.data_df['prev_lower_vegas_gradient'] = self.data_df['lower_vegas_gradient'].shift(1)

            self.data_df['upper_vegas_go_up'] = np.where(
                self.data_df['upper_vegas_gradient'] > 0,
                1,
                0
            )

            self.data_df['lower_vegas_go_up'] = np.where(
                self.data_df['lower_vegas_gradient'] > 0,
                1,
                0
            )

            self.data_df['upper_vegas_go_up_num'] = self.data_df['upper_vegas_go_up'].rolling(vegas_look_back, min_periods = vegas_look_back).sum()
            self.data_df['lower_vegas_go_up_num'] = self.data_df['lower_vegas_go_up'].rolling(vegas_look_back, min_periods = vegas_look_back).sum()

            self.data_df['upper_vegas_go_up_pct'] = self.data_df['upper_vegas_go_up_num'] / vegas_look_back
            self.data_df['lower_vegas_go_up_pct'] = self.data_df['lower_vegas_go_up_num'] / vegas_look_back

            self.data_df['prev_upper_vegas_go_up_pct'] = self.data_df['upper_vegas_go_up_pct'].shift(1)
            self.data_df['prev_lower_vegas_go_up_pct'] = self.data_df['lower_vegas_go_up_pct'].shift(1)





            self.data_df['prev1_upper_vegas'] = self.data_df['upper_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_upper_vegas'] = self.data_df['prev' + str(i-1) + '_upper_vegas'].shift(1)

            self.data_df['prev1_lower_vegas'] = self.data_df['lower_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_lower_vegas'] = self.data_df['prev' + str(i-1) + '_lower_vegas'].shift(1)



            #df[attr].rolling(window, min_periods = window).max()

            ##Calculate large bar condition 1
            self.data_df['price_range_mean'] = self.data_df['price_range'].rolling(price_range_lookback_window, min_periods = price_range_lookback_window).mean() #mean()
            self.data_df['prev_price_range_mean'] = self.data_df['price_range_mean'].shift(1)


            # self.data_df['positive_price_range_mean'] = self.data_df['positive_price_range'].rolling(price_range_lookback_window, min_periods = price_range_lookback_window).max() #mean()
            # self.data_df['prev_positive_price_range_mean'] = self.data_df['positive_price_range_mean'].shift(1)
            #
            # self.data_df['negative_price_range_mean'] = self.data_df['negative_price_range'].rolling(price_range_lookback_window, min_periods = price_range_lookback_window).max() #mean()
            # self.data_df['prev_negative_price_range_mean'] = self.data_df['negative_price_range_mean'].shift(1)


            # self.data_df['bar_length_increase'] = np.where(
            #     self.data_df['close'] - self.data_df['open'] > 0,
            #     np.where(
            #         self.data_df['prev_positive_price_range_mean'] > 0,
            #         (self.data_df['price_range'] - self.data_df['prev_positive_price_range_mean']) / self.data_df['prev_positive_price_range_mean'],
            #         0
            #     ),
            #     np.where(
            #         self.data_df['prev_negative_price_range_mean'] > 0,
            #         (self.data_df['price_range'] - self.data_df['prev_negative_price_range_mean']) / self.data_df['prev_negative_price_range_mean'],
            #         0
            #     )
            # )



            self.data_df['bar_length_increase'] = (self.data_df['price_range'] - self.data_df['prev_price_range_mean']) / self.data_df['prev_price_range_mean']
            self.data_df['large_bar_c1'] = self.data_df['bar_length_increase'] > bar_increase_threshold


            ##Calculate large bar condition 2
            self.data_df['middle'] = (self.data_df['open'] + self.data_df['close']) / 2

            self.data_df['prev' + str(skip_bar_num + 1) + '_has_covered_lower'] = self.data_df['prev' + str(skip_bar_num + 1) + '_min_price'] <= self.data_df['middle']
            for i in range(skip_bar_num + 2, large_bar_look_back + 1):
                self.data_df['prev' + str(i) + '_has_covered_lower'] = self.data_df['prev' + str(i-1) + '_has_covered_lower'] | \
                                                                         (self.data_df['prev' + str(i) + '_min_price'] <= self.data_df['middle'])

            self.data_df['prev'  + str(skip_bar_num + 1) +  '_has_covered_higher'] = self.data_df['prev' + str(skip_bar_num + 1) + '_max_price'] >= self.data_df['middle']
            for i in range(skip_bar_num + 2, large_bar_look_back + 1):
                self.data_df['prev' + str(i) + '_has_covered_higher'] = self.data_df['prev' + str(i-1) + '_has_covered_higher'] | \
                                                                         (self.data_df['prev' + str(i) + '_max_price'] >= self.data_df['middle'])

            self.data_df['has_been_covered_recently'] = np.where(
                self.data_df['close'] > self.data_df['open'],
                self.data_df['prev' + str(large_bar_look_back) + '_has_covered_higher'],
                self.data_df['prev' + str(large_bar_look_back) + '_has_covered_lower']
            )


            self.data_df['large_bar_c2'] = self.data_df['has_been_covered_recently']

            ##Calculate large bar condition 3
            self.data_df['positive_price_range_max'] = self.data_df['positive_price_range'].rolling(large_bar_look_back, min_periods = large_bar_look_back).max() #mean()
            self.data_df['prev_positive_price_range_max'] = self.data_df['positive_price_range_max'].shift(1)

            self.data_df['negative_price_range_max'] = self.data_df['negative_price_range'].rolling(large_bar_look_back, min_periods = large_bar_look_back).max() #mean()
            self.data_df['prev_negative_price_range_max'] = self.data_df['negative_price_range_max'].shift(1)

            self.data_df['prev_opposite_longest_bar_range'] = np.where(
                self.data_df['close'] - self.data_df['open'] > 0,
                self.data_df['prev_negative_price_range_max'],
                self.data_df['prev_positive_price_range_max']
            )

            self.data_df['large_bar_c3'] = (self.data_df['prev_opposite_longest_bar_range'] - self.data_df['price_range']) / self.data_df['price_range'] < 0.05


            self.data_df['large_temp_positive_bar'] = (self.data_df['close'] > self.data_df['open']) & \
                                                 self.data_df['large_bar_c1'] & self.data_df['large_bar_c2'] & self.data_df['large_bar_c3']

            self.data_df['large_temp_negative_bar'] = (self.data_df['close'] < self.data_df['open']) & \
                                                 self.data_df['large_bar_c1'] & self.data_df['large_bar_c2'] & self.data_df['large_bar_c3']

            self.data_df['is_cross_guppy'] = (self.data_df['min_price'] < self.data_df['lowest_guppy']) & (self.data_df['max_price'] > self.data_df['highest_guppy'])

            # self.data_df['large_positive_bar'] = self.data_df['large_temp_positive_bar'] & \
            #                                      ((~(strongly_half_aligned_long_condition & (self.data_df['lowest_guppy'] > self.data_df['upper_vegas']))) | self.data_df['is_cross_guppy'])
            # self.data_df['large_negative_bar'] = self.data_df['large_temp_negative_bar'] & \
            #                                      ((~(strongly_half_aligned_short_condition & (self.data_df['highest_guppy'] < self.data_df['lower_vegas']))) | self.data_df['is_cross_guppy'])


            self.data_df['large_positive_bar'] = self.data_df['large_temp_positive_bar'] & \
                                                 ((~strongly_half_aligned_long_condition) | self.data_df['is_cross_guppy'])
            self.data_df['large_negative_bar'] = self.data_df['large_temp_negative_bar'] & \
                                                 ((~strongly_half_aligned_short_condition) | self.data_df['is_cross_guppy'])



            #self.data_df['large_positive_bar'] = self.data_df['large_temp_positive_bar']
            #self.data_df['large_negative_bar'] = self.data_df['large_temp_negative_bar']




            self.data_df['prev1_large_positive_bar'] = self.data_df['large_positive_bar'].shift(1)
            #self.data_df.at[0, 'prev1_large_positive_bar'] = False
            #self.data_df['prev1_large_positive_bar'] = pd.Series(list(self.data_df['prev1_large_positive_bar']), dtype='bool')

            for i in range(2, large_bar_consider_past_num + 1):
                self.data_df['prev' + str(i) + '_large_positive_bar'] = self.data_df['prev' + str(i-1) + '_large_positive_bar'].shift(1)

            self.data_df['prev1_large_negative_bar'] = self.data_df['large_negative_bar'].shift(1)
            for i in range(2, large_bar_consider_past_num + 1):
                self.data_df['prev' + str(i) + '_large_negative_bar'] = self.data_df['prev' + str(i-1) + '_large_negative_bar'].shift(1)


            self.data_df['is_false_buy_signal'] = reduce(lambda left, right: left | right,
                                                          [self.data_df['prev' + str(i) + '_large_positive_bar'] for i in range(1, large_bar_consider_past_num + 1)] +
                                                          [self.data_df['large_positive_bar']])

            self.data_df['is_false_sell_signal'] = reduce(lambda left, right: left | right,
                                                          [self.data_df['prev' + str(i) + '_large_negative_bar'] for i in range(1, large_bar_consider_past_num + 1)] +
                                                          [self.data_df['large_negative_bar']])





           ############# Old code ###############
            self.data_df['prev1_bar_length_increase'] = self.data_df['bar_length_increase'].shift(1)
            for i in range(2, large_bar_consider_past_num + 1):
                self.data_df['prev' + str(i) + '_bar_length_increase'] = self.data_df['prev' + str(i-1) + '_bar_length_increase'].shift(1)


            self.data_df['is_large_bar_buy'] = reduce(lambda left, right: left | right,
                                                      [((self.data_df['prev' + str(i) + '_bar_length_increase'] > bar_increase_threshold) & \
                                                       (self.data_df['prev' + str(i) + '_close'] - self.data_df['prev' + str(i) + '_open'] > 0))
                                                                                         for i in range(1, large_bar_consider_past_num + 1)] +
                                                      [(self.data_df['bar_length_increase'] > bar_increase_threshold)])

            self.data_df['is_large_bar_sell'] = reduce(lambda left, right: left | right,
                                                      [((self.data_df['prev' + str(i) + '_bar_length_increase'] > bar_increase_threshold) & \
                                                       (self.data_df['prev' + str(i) + '_close'] - self.data_df['prev' + str(i) + '_open'] < 0))
                                                                                         for i in range(1, large_bar_consider_past_num + 1)] +
                                                      [(self.data_df['bar_length_increase'] > bar_increase_threshold)])

             ######################################










            #self.data_df['is_large_bar'] = self.data_df['bar_length_increase'] > bar_increase_threshold


            self.data_df['low_pct_price_buy'] = self.data_df['min_price'] + (self.data_df['price_range']) * bar_low_percentile
            self.data_df['high_pct_price_buy'] = self.data_df['max_price'] - (self.data_df['price_range']) * bar_high_percentile

            self.data_df['low_low_pct_price_buy'] = self.data_df['min_price'] + (self.data_df['price_range']) * vegas_bar_percentile
            self.data_df['prev_low_low_pct_price_buy'] = self.data_df['low_low_pct_price_buy'].shift(1)


            self.data_df['low_pct_price_sell'] = self.data_df['min_price'] + (self.data_df['price_range']) * bar_high_percentile
            self.data_df['high_pct_price_sell'] = self.data_df['max_price'] - (self.data_df['price_range']) * bar_low_percentile

            self.data_df['high_high_pct_price_sell'] = self.data_df['max_price'] - (self.data_df['price_range']) * vegas_bar_percentile
            self.data_df['prev_high_high_pct_price_sell'] = self.data_df['high_high_pct_price_sell'].shift(1)




            self.data_df['prev_upper_band_close'] = self.data_df['upper_band_close'].shift(1)
            self.data_df['prev_lower_band_close'] = self.data_df['lower_band_close'].shift(1)


            # self.data_df['is_real_false_buy_signal'] = self.data_df['is_false_buy_signal'] & \
            #                                            ((~strongly_half_aligned_long_condition) | self.data_df['is_cross_guppy'])
            #
            # self.data_df['is_real_false_sell_signal'] = self.data_df['is_false_sell_signal'] & \
            #                                              ((~strongly_half_aligned_short_condition) | self.data_df['is_cross_guppy'])

            self.data_df['half_aligned_long_condition'] = half_aligned_long_condition
            self.data_df['half_aligned_short_condition'] = half_aligned_short_condition





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

            self.data_df['is_vegas_enough_up_trend'] = (self.data_df['vegas_width'] * self.lot_size * self.exchange_rate > vegas_width_threshold) | half_aligned_long_condition
                                                         # | reduce(lambda left, right: left & right,
                                                         #          [(self.data_df[guppy] > self.data_df['lower_vegas']) for guppy in guppy_lines[0:3]] +
                                                         #          [(self.data_df[guppy + '_gradient'] > 0) for guppy in guppy_lines[0:3]])


            self.data_df['is_vegas_enough_down_trend'] = (self.data_df['vegas_width'] * self.lot_size * self.exchange_rate < -vegas_width_threshold) | half_aligned_short_condition
                                                           # | reduce(lambda left, right: left & right,
                                                           #        [(self.data_df[guppy] < self.data_df['upper_vegas']) for guppy in guppy_lines[0:3]] +
                                                           #        [(self.data_df[guppy + '_gradient'] < 0) for guppy in guppy_lines[0:3]])



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


            #Hutong
            recent_tightly_supported_by_vegas = reduce(lambda left, right: left | right,
                                               [((self.data_df['prev' + str(i) + '_price_pct_to_upper_vegas'] < tight_distance_to_vegas_threshold) & \
                                                (self.data_df['prev' + str(i) + '_max_price'] <= self.data_df['prev' + str(i) + '_ma_close12']))
                                                for i in range(1, ma12_lookback + 1)])

            recent_supported_by_vegas = reduce(lambda left, right: left | right,
                                           [((self.data_df['prev' + str(i) + '_price_pct_to_upper_vegas'] < distance_to_vegas_threshold) & \
                                            (self.data_df['prev' + str(i) + '_max_price'] <= self.data_df['prev' + str(i) + '_ma_close12'])) & \
                                            (self.data_df['prev' + str(i) + '_lowest_guppy'] > self.data_df['prev' + str(i) + '_upper_vegas']) & \
                                            (self.data_df['prev' + str(i) + '_min_price'] <= self.data_df['prev' + str(i) + '_lowest_guppy'])
                                            for i in range(1, ma12_lookback + 1)])




            recent_tightly_suppressed_by_vegas = reduce(lambda left, right: left | right,
                                               [((self.data_df['prev' + str(i) + '_price_pct_to_lower_vegas'] > -tight_distance_to_vegas_threshold) & \
                                                (self.data_df['prev' + str(i) + '_min_price'] >= self.data_df['prev' + str(i) + '_ma_close12']))
                                                for i in range(1, ma12_lookback + 1)])

            recent_suppressed_by_vegas = reduce(lambda left, right: left | right,
                                           [((self.data_df['prev' + str(i) + '_price_pct_to_lower_vegas'] > -distance_to_vegas_threshold) & \
                                            (self.data_df['prev' + str(i) + '_min_price'] >= self.data_df['prev' + str(i) + '_ma_close12'])) & \
                                            (self.data_df['prev' + str(i) + '_highest_guppy'] < self.data_df['prev' + str(i) + '_lower_vegas']) & \
                                            (self.data_df['prev' + str(i) + '_max_price'] >= self.data_df['prev' + str(i) + '_highest_guppy'])
                                            for i in range(1, ma12_lookback + 1)])







            self.data_df['distance_to_upper_vegas'] = self.data_df['ma_close12'] - self.data_df['upper_vegas']
            self.data_df['distance_to_lower_vegas'] = self.data_df['ma_close12'] - self.data_df['lower_vegas']

            self.data_df['price_to_upper_vegas'] = self.data_df['close'] - self.data_df['upper_vegas']
            self.data_df['price_to_lower_vegas'] = self.data_df['close'] - self.data_df['lower_vegas']

            self.data_df['price_to_bolling_upper'] = self.data_df['close'] - self.data_df['upper_band_close']
            self.data_df['price_to_bolling_lower'] = self.data_df['close'] - self.data_df['lower_band_close']



            self.data_df['pct_to_upper_vegas'] = self.data_df['distance_to_upper_vegas'] / self.data_df['high_low_range']
            self.data_df['pct_to_lower_vegas'] = self.data_df['distance_to_lower_vegas'] / self.data_df['high_low_range']

            self.data_df['prev1_pct_to_upper_vegas'] = self.data_df['pct_to_upper_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_pct_to_upper_vegas'] = self.data_df['prev' + str(i-1) + '_pct_to_upper_vegas'].shift(1)

            self.data_df['prev1_pct_to_lower_vegas'] = self.data_df['pct_to_lower_vegas'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_pct_to_lower_vegas'] = self.data_df['prev' + str(i-1) + '_pct_to_lower_vegas'].shift(1)


            recent_m12_supported_by_vegas = reduce(lambda left, right: left | right,
                                                   [  self.data_df['prev' + str(i) + '_pct_to_upper_vegas'] < distance_to_vegas_threshold
                                                       for i in range(1, ma12_lookback + 1)]
                                                   )

            recent_m12_suppressed_by_vegas = reduce(lambda left, right: left | right,
                                                   [  self.data_df['prev' + str(i) + '_pct_to_lower_vegas'] > -distance_to_vegas_threshold
                                                       for i in range(1, ma12_lookback + 1)]
                                                   )



            if self.use_relaxed_vegas_support:
                #final_recent_supported_by_vegas = (recent_supported_by_vegas) | (self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold)
                #final_recent_suppressed_by_vegas = (recent_suppressed_by_vegas) | (self.data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold)

                final_recent_supported_by_vegas = recent_tightly_supported_by_vegas | recent_supported_by_vegas | (self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold)
                final_recent_suppressed_by_vegas = recent_tightly_suppressed_by_vegas | recent_suppressed_by_vegas | (self.data_df['pct_to_lower_vegas'] > -distance_to_vegas_threshold)

                # final_recent_supported_by_vegas = recent_tightly_supported_by_vegas | recent_supported_by_vegas | recent_m12_supported_by_vegas
                # final_recent_suppressed_by_vegas = recent_tightly_suppressed_by_vegas | recent_suppressed_by_vegas | recent_m12_suppressed_by_vegas


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



            self.data_df['upper_vegas_mostly_up'] = self.data_df['prev_upper_vegas_go_up_pct'] >= vegas_trend_pct_threshold
            self.data_df['lower_vegas_mostly_up'] = self.data_df['prev_lower_vegas_go_up_pct'] >= vegas_trend_pct_threshold

            self.data_df['upper_vegas_mostly_down'] = self.data_df['prev_upper_vegas_go_up_pct'] <= 1 - vegas_trend_pct_threshold
            self.data_df['lower_vegas_mostly_down'] = self.data_df['prev_lower_vegas_go_up_pct'] <= 1 - vegas_trend_pct_threshold

            self.data_df['vegas_mostly_up'] = self.data_df['upper_vegas_mostly_up'] & self.data_df['lower_vegas_mostly_up']
            self.data_df['vegas_mostly_down'] = self.data_df['upper_vegas_mostly_down'] & self.data_df['lower_vegas_mostly_down']

            self.data_df['vegas_fast_above'] = self.data_df['ma_close144'] > self.data_df['ma_close169']

            self.data_df['vegas_layout'] = np.where(
                self.data_df['vegas_fast_above'],
                1,
                0
            )
            self.data_df['vegas_cross'] = self.data_df['vegas_layout'].diff()
            self.data_df['vegas_cross_up'] = np.where(
                self.data_df['vegas_cross'] == 1,
                1,
                0
            )
            self.data_df['vegas_cross_down'] = np.where(
                self.data_df['vegas_cross'] == -1,
                1,
                0
            )


            self.data_df['recent_vegas_cross_up_num'] = self.data_df['vegas_cross_up'].rolling(vegas_short_look_back, min_periods = vegas_short_look_back).sum()
            self.data_df['recent_vegas_cross_down_num'] = self.data_df['vegas_cross_down'].rolling(vegas_short_look_back, min_periods = vegas_short_look_back).sum()

            self.data_df['fastest_guppy_at_top'] = np.abs(self.data_df['ma_close30'] - self.data_df['highest_guppy']) < 1e-5
            self.data_df['fastest_guppy_at_btm'] = np.abs(self.data_df['ma_close30'] - self.data_df['lowest_guppy']) < 1e-5


            buy_c41 = self.data_df['high'] > self.data_df['upper_band_close'] #self.data_df['prev_upper_band_close']
            buy_c42 = self.data_df['upper_band_close_gradient'] * self.lot_size * self.exchange_rate > 0# bolling_threshold
            buy_c43 = self.data_df['is_positive'] & (self.data_df['prev1_open'] < self.data_df['prev1_close'])
            buy_c44 = (self.data_df['low_low_pct_price_buy'] > self.data_df['upper_vegas']) & (self.data_df['prev_low_low_pct_price_buy'] > self.data_df['prev1_upper_vegas'])
            buy_c4 = buy_c41 & buy_c42 & buy_c43 & buy_c44

            sell_c41 = self.data_df['low'] < self.data_df['lower_band_close'] # self.data_df['prev_lower_band_close']
            sell_c42 = self.data_df['lower_band_close_gradient'] * self.lot_size * self.exchange_rate < 0 #-bolling_threshold
            sell_c43 = self.data_df['is_negative'] &  (self.data_df['prev1_open'] > self.data_df['prev1_close'])
            sell_c44 = (self.data_df['high_high_pct_price_sell'] < self.data_df['lower_vegas']) & (self.data_df['prev_high_high_pct_price_sell'] < self.data_df['prev1_lower_vegas'])
            sell_c4 = sell_c41 & sell_c42 & sell_c43 & sell_c44


            #(self.data_df['fastest_guppy_at_top'] & self.data_df['ma_close30_gradient'] > 0)
            self.data_df['vegas_inferred_up_cond1'] = self.data_df['vegas_fast_above'] & (self.data_df['recent_vegas_cross_up_num'] == 1)
            self.data_df['vegas_inferred_up_cond2'] = (~self.data_df['vegas_fast_above']) & (
                ((self.data_df['fastest_guppy_at_top'] & self.data_df['ma_close30_gradient'] > 0)) | \
                ((self.data_df['lower_vegas_gradient']*self.lot_size*self.exchange_rate > vagas_fast_support_threshold) & (self.data_df['prev_lower_vegas_gradient']*self.lot_size*self.exchange_rate > vagas_fast_support_threshold))
            )
            self.data_df['vegas_inferred_up'] = self.data_df['vegas_inferred_up_cond1'] | self.data_df['vegas_inferred_up_cond2'] | buy_c4

            #(self.data_df['fastest_guppy_at_btm'] & self.data_df['ma_close30_gradient'] < 0)
            self.data_df['vegas_inferred_down_cond1'] = (~self.data_df['vegas_fast_above']) & (self.data_df['recent_vegas_cross_down_num'] == 1)
            self.data_df['vegas_inferred_down_cond2'] = self.data_df['vegas_fast_above'] & (
                ((self.data_df['fastest_guppy_at_btm'] & self.data_df['ma_close30_gradient'] < 0)) | \
                ((self.data_df['upper_vegas_gradient']*self.lot_size*self.exchange_rate < -vagas_fast_support_threshold) & (self.data_df['prev_upper_vegas_gradient']*self.lot_size*self.exchange_rate < -vagas_fast_support_threshold))
            )
            self.data_df['vegas_inferred_down'] = self.data_df['vegas_inferred_down_cond1'] | self.data_df['vegas_inferred_down_cond2'] | sell_c4

            self.data_df['vegas_final_inferred_up'] = self.data_df['vegas_mostly_up'] & self.data_df['vegas_inferred_up']
            self.data_df['vegas_final_inferred_down'] = self.data_df['vegas_mostly_down'] & self.data_df['vegas_inferred_down']


            #Cruise
            enter_bar_too_large = (self.data_df['price_range'] * self.lot_size * self.exchange_rate > maximum_enter_bar_length) | \
                                  (self.data_df['prev_price_range'] * self.lot_size * self.exchange_rate > maximum_enter_bar_length)

            self.data_df['buy_weak_ready'] = self.data_df['is_above_vegas'] & (
                        final_recent_supported_by_vegas) & ( #self.data_df['pct_to_upper_vegas'] < distance_to_vegas_threshold
                                                    self.data_df['high_pct_price_buy'] < self.data_df['ma_close12'])
            self.data_df['buy_weak_fire'] = above_cond & ( #'is_above_vegas_strict'  (~enter_bar_too_large)
                        final_recent_supported_by_vegas) & (self.data_df['low_pct_price_buy'] > self.data_df['ma_close12']) \
                                       & (self.data_df['ma12_gradient'] >= 0) & (self.data_df['close'] > self.data_df['ma_close12']) \
                                            & ((self.data_df['close'] - self.data_df['open']) * self.lot_size * self.exchange_rate > enter_bar_width_threshold)

            self.data_df['buy_ready'] = self.data_df['buy_weak_ready'] & (self.data_df['is_vegas_up_trend'] | self.data_df['vegas_final_inferred_up'])
            self.data_df['buy_fire'] = self.data_df['buy_weak_fire'] & (self.data_df['is_vegas_up_trend'] | self.data_df['vegas_final_inferred_up'])
            self.data_df['buy_real_fire'] = self.data_df['buy_fire'] & (self.data_df['is_vegas_enough_up_trend'] | self.data_df['vegas_final_inferred_up'])

            buy_c11 = self.data_df['price_to_lower_vegas'] * self.lot_size * self.exchange_rate < maximum_loss
            buy_c12 = self.data_df['price_to_bolling_upper'] * self.lot_size * self.exchange_rate < -minimum_profit
            buy_c13 = (-self.data_df['price_to_bolling_upper'] / self.data_df['price_to_lower_vegas']) > minimum_profilt_loss_ratio

            buy_c2 = self.data_df['is_guppy_aligned_short']

            buy_c2_aux = (half_aligned_short_condition &
                      (self.data_df['close'] < self.data_df[sorted_guppys[2]]) & (self.data_df['close'] > self.data_df['lowest_guppy'])) | self.data_df['is_guppy_aligned_short']

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

            # buy_c41 = self.data_df['high'] > self.data_df['upper_band_close'] #self.data_df['prev_upper_band_close']
            # buy_c42 = self.data_df['upper_band_close_gradient'] * self.lot_size * self.exchange_rate > 0# bolling_threshold
            # buy_c43 = self.data_df['is_positive'] & (self.data_df['prev1_open'] < self.data_df['prev1_close'])
            # buy_c44 = (self.data_df['low_low_pct_price_buy'] > self.data_df['upper_vegas']) & (self.data_df['prev_low_low_pct_price_buy'] > self.data_df['prev1_upper_vegas'])
            # buy_c4 = buy_c41 & buy_c42 & buy_c43 & buy_c44


            buy_c5 = reduce(lambda left, right: left & right, [((self.data_df['prev' + str(i) + '_open'] - self.data_df['prev' + str(i) + '_close']) * self.lot_size * self.exchange_rate > enter_bar_width_threshold)
                                                               for i in range(1,c5_lookback + 1)])

            #buy_c6 = (self.data_df['bar_length_increase'] > bar_increase_threshold) | (self.data_df['prev1_bar_length_increase'] > bar_increase_threshold) | (self.data_df['prev2_bar_length_increase'] > bar_increase_threshold)

            #buy_c6 = self.data_df['is_large_bar_buy']

            buy_c6 = self.data_df['is_false_buy_signal']

            buy_c7 = (self.data_df['close'] > self.data_df['upper_band_close']) & \
                     ((self.data_df['price_to_period_high_pct'] < price_to_period_range_pct_strict) | \
                      ((self.data_df['price_to_period_high_pct'] < price_to_period_range_pct) & (~strongly_half_aligned_long_condition)))
                      #Buy price too high, should not enter

            if not self.remove_c12:
                self.data_df['buy_real_fire'] = (self.data_df['buy_real_fire']) & (buy_c3) & (~buy_c5) & (~buy_c6) & (~buy_c7) & ((buy_c4) | (((buy_c12) | (buy_c13)) & (~buy_c2)))
            else:
                self.data_df['buy_real_fire'] = (self.data_df['buy_real_fire']) & (buy_c3) & (~buy_c5) & (~buy_c6) & (~buy_c7) & ((buy_c4) | ((buy_c13) & (~buy_c2)))


            self.data_df['buy_c11'] = buy_c11
            self.data_df['buy_c12'] = buy_c12
            self.data_df['buy_c13'] = buy_c13
            self.data_df['buy_c2'] = buy_c2
            self.data_df['buy_c3'] = buy_c3
            self.data_df['buy_c41'] = buy_c41
            self.data_df['buy_c42'] = buy_c42
            self.data_df['buy_c43'] = buy_c43
            self.data_df['buy_c44'] = buy_c44
            self.data_df['buy_c5'] = buy_c5
            self.data_df['buy_c6'] = buy_c6
            self.data_df['buy_c7'] = buy_c7



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
            self.data_df['sell_weak_fire'] = below_cond & ( #is_below_vegas_strict    (~enter_bar_too_large) &
                        final_recent_suppressed_by_vegas) & (self.data_df['high_pct_price_sell'] < self.data_df['ma_close12']) \
                                        & (self.data_df['ma12_gradient'] <= 0) & (self.data_df['close'] < self.data_df['ma_close12']) \
                                             & ((self.data_df['close'] - self.data_df['open']) * self.lot_size * self.exchange_rate < -enter_bar_width_threshold)

            self.data_df['sell_ready'] = self.data_df['sell_weak_ready'] & (self.data_df['is_vegas_down_trend'] | self.data_df['vegas_final_inferred_down'])
            self.data_df['sell_fire'] = self.data_df['sell_weak_fire'] & (self.data_df['is_vegas_down_trend'] | self.data_df['vegas_final_inferred_down'])
            self.data_df['sell_real_fire'] = self.data_df['sell_fire'] & (self.data_df['is_vegas_enough_down_trend'] | self.data_df['vegas_final_inferred_down'])

            sell_c11 = self.data_df['price_to_upper_vegas'] * self.lot_size * self.exchange_rate > -maximum_loss
            sell_c12 = self.data_df['price_to_bolling_lower'] * self.lot_size * self.exchange_rate > minimum_profit
            sell_c13 = (-self.data_df['price_to_bolling_lower'] / self.data_df['price_to_upper_vegas']) > minimum_profilt_loss_ratio

            sell_c2 = self.data_df['is_guppy_aligned_long']

            sell_c2_aux = (half_aligned_long_condition &
                       (self.data_df['close'] > self.data_df[sorted_guppys[3]]) & (self.data_df['close'] < self.data_df['highest_guppy'])) | self.data_df['is_guppy_aligned_long']


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

            # sell_c41 = self.data_df['low'] < self.data_df['lower_band_close'] # self.data_df['prev_lower_band_close']
            # sell_c42 = self.data_df['lower_band_close_gradient'] * self.lot_size * self.exchange_rate < 0 #-bolling_threshold
            # sell_c43 = self.data_df['is_negative'] &  (self.data_df['prev1_open'] > self.data_df['prev1_close'])
            # sell_c44 = (self.data_df['high_high_pct_price_sell'] < self.data_df['lower_vegas']) & (self.data_df['prev_high_high_pct_price_sell'] < self.data_df['prev1_lower_vegas'])
            # sell_c4 = sell_c41 & sell_c42 & sell_c43 & sell_c44

            sell_c5 = reduce(lambda left, right: left & right, [((self.data_df['prev' + str(i) + '_open'] - self.data_df['prev' + str(i) + '_close']) * self.lot_size * self.exchange_rate < -enter_bar_width_threshold )
                                                                for i in range(1,c5_lookback + 1)])

            #sell_c6 = (self.data_df['bar_length_increase'] > bar_increase_threshold) | (self.data_df['prev1_bar_length_increase'] > bar_increase_threshold) | (self.data_df['prev2_bar_length_increase'] > bar_increase_threshold)
            #sell_c6 = self.data_df['is_large_bar_sell']

            sell_c6 = self.data_df['is_false_sell_signal']

            sell_c7 = (self.data_df['close'] < self.data_df['lower_band_close']) & \
                     ((self.data_df['price_to_period_low_pct'] < price_to_period_range_pct_strict) | \
                      ((self.data_df['price_to_period_low_pct'] < price_to_period_range_pct) & (~strongly_half_aligned_short_condition)))

            if not self.remove_c12:
                self.data_df['sell_real_fire'] = (self.data_df['sell_real_fire']) & (sell_c3) & (~sell_c5) & (~sell_c6) & (~sell_c7) & ((sell_c4) | (((sell_c12) | (sell_c13)) & (~sell_c2)))
            else:
                self.data_df['sell_real_fire'] = (self.data_df['sell_real_fire']) & (sell_c3) & (~sell_c5) & (~sell_c6) & (~sell_c7) & ((sell_c4) | ((sell_c13) & (~sell_c2)))



            self.data_df['sell_c11'] = sell_c11
            self.data_df['sell_c12'] = sell_c12
            self.data_df['sell_c13'] = sell_c13
            self.data_df['sell_c2'] = sell_c2
            self.data_df['sell_c3'] = sell_c3
            self.data_df['sell_c41'] = sell_c41
            self.data_df['sell_c42'] = sell_c42
            self.data_df['sell_c43'] = sell_c43
            self.data_df['sell_c44'] = sell_c44
            self.data_df['sell_c5'] = sell_c5
            self.data_df['sell_c6'] = sell_c6


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

                additional_msg = " Exit if next two bars are both negative" if buy_c2_aux.iloc[-1] else ""

                additional_msg2 = " Be careful" if buy_c6[-1] else ""

                sendEmail(msg, msg + additional_msg + additional_msg2)

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

                additional_msg = " Exit if next two bars are both positive" if sell_c2_aux.iloc[-1] else ""

                additional_msg2 = " Be careful" if sell_c6[-1] else ""

                sendEmail(msg, msg + additional_msg + additional_msg2)

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



            self.data_df.to_csv(self.currency_file + 'tmp.csv', index=False)

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











