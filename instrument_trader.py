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


import threading

windows = [12, 30, 35, 40, 45, 50, 60, 144, 169]
high_low_window_options = [100, 200]
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

price_to_period_range_pct_relaxed = 0.25
price_to_period_range_pct = 0.10
price_to_period_range_pct_strict = 0.02

vegas_look_back = 120
vegas_trend_pct_threshold = 0.8

vegas_short_look_back = 10

vagas_fast_support_threshold = 10

period_lookback = 50

minimum_opposite_side_trend_num = 0
minimum_break_bolling_num = 1

reverse_threshold = 0.1

guppy_lookback = 24

vegas_angle_threshold = 3

reverse_trade_min_points_to_vegas = 150
reverse_trade_min_distance_to_vegas = 0.15

reverse_trade_look_back = 10

macd_relaxed = True

price_range_look_back = 10

is_plot_exclude = True

high_low_delta_threshold = 20

entry_risk_threshold = 0.6

close_position_look_back = 12

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

        #self.currency_file = os.path.join(data_folder, currency + "100.csv")

        self.log_fd = open(self.log_file, 'a')

        self.print_to_console = True

        self.is_cut_data = False

        self.data_df_backup100 = None
        self.data_df_backup200 = None

        self.data_dfs_backup = []


        self.log_msg("Initializing...")




    def log_msg(self, msg):

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #current_time = (datetime.now() + timedelta(seconds = 28800)).strftime("%Y-%m-%d %H:%M:%S")
        print('[' + current_time + ' ' + self.currency + ']  ' + msg, file = self.log_fd)
        self.log_fd.flush()

        if self.print_to_console:
            print('[' + current_time + ' ' + self.currency + ']  ' + msg)


    def get_last_time(self):
        return self.last_time

    def cut_data(self):

        #self.data_df_backup = self.data_df.copy()
        #self.data_df_backup = self.data_df_backup[self.data_df_backup['price_range'].notnull()]

        self.data_df = self.data_df.iloc[-200:]
        self.data_df.reset_index(inplace = True)
        self.data_df = self.data_df.drop(columns = ['index'])



    def feed_data(self, new_data_df, original_data_df100 = None, original_data_df200 = None):
        self.condition.acquire()

        self.data_df_backup100 = original_data_df100
        self.data_df_backup200 = original_data_df200

        if self.data_df_backup100 is not None:
            self.data_df_backup100['date'] = pd.DatetimeIndex(self.data_df_backup100['time']).normalize()

        if self.data_df_backup200 is not None:
            self.data_df_backup200['date'] = pd.DatetimeIndex(self.data_df_backup200['time']).normalize()

        for high_low_window in high_low_window_options:
            if high_low_window == 100:
                self.data_dfs_backup += [self.data_df_backup100]
            elif high_low_window == 200:
                self.data_dfs_backup += [self.data_df_backup200]


        if new_data_df is not None and new_data_df.shape[0] > 0:

            if self.data_df is None or self.data_df.shape[0] == 0:
                self.data_df = new_data_df

                self.last_time = self.data_df.iloc[-1]['time']

                #self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()

                if self.is_cut_data and original_data_df100 is not None and original_data_df200 is not None:
                    self.cut_data()

                #self.data_df = self.data_df[['currency', 'time', 'open', 'high', 'low', 'close']]

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

                    #self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()

                    if self.is_cut_data and original_data_df100 is not None and original_data_df200 is not None:
                        self.cut_data()

                    #self.data_df = self.data_df[['currency', 'time', 'open', 'high', 'low', 'close']]

                    self.condition.notify()

        self.condition.release()

    def run(self):
        print("Running...........")
        self.trade()


    def calculate_signals(self, high_low_window):

        numerical_features = ['pct_to_upper_vegas', 'high_pct_price_buy', 'low_pct_price_buy', 'pct_to_lower_vegas',
                              'low_pct_price_sell', 'high_pct_price_sell',
                              'ma_close12', 'ma12_gradient']

        bool_features = ['is_above_vegas', 'is_vegas_up_trend', 'is_below_vegas', 'is_vegas_down_trend']

        if True:
            print("Process data_df cut: high_low_window = " + str(high_low_window))
            print(self.data_df[['time','close']].head(10))

            self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()

            #all_days = pd.Series(self.data_df['date'].unique()).dt.to_pydatetime()

            self.data_df['price_delta'] = self.data_df['close'] - self.data_df['open']

            calc_high_Low(self.data_df, "close", high_low_window)
            calc_jc_lines(self.data_df, "close", windows)
            calc_bolling_bands(self.data_df, "close", bolling_width)
            calc_macd(self.data_df, "close", high_low_window)

            self.data_df['macd_gradient'] = self.data_df['macd'].diff()

            self.data_df['prev_macd'] = self.data_df['macd'].shift(1)
            self.data_df['prev_msignal'] = self.data_df['msignal'].shift(1)
            self.data_df['macd_cross_up'] = (self.data_df['prev_macd'] < self.data_df['prev_msignal']) & (self.data_df['macd'] >= self.data_df['msignal'])
            self.data_df['macd_cross_down'] = (self.data_df['prev_macd'] > self.data_df['prev_msignal']) & (self.data_df['macd'] <= self.data_df['msignal'])



            self.data_df['min_price'] = self.data_df[['open', 'close']].min(axis=1)
            self.data_df['max_price'] = self.data_df[['open', 'close']].max(axis=1)


            self.data_df['period_high_low_range'] = self.data_df['period_high' + str(high_low_window)] - self.data_df['period_low' + str(high_low_window)]

            self.data_df['price_to_period_low'] = self.data_df['close'] - self.data_df['period_low' + str(high_low_window)]
            self.data_df['price_to_period_high'] = self.data_df['period_high' + str(high_low_window)] - self.data_df['close']

            self.data_df['price_to_period_low_pct'] = self.data_df['price_to_period_low'] / self.data_df['period_high_low_range']
            self.data_df['price_to_period_high_pct'] = self.data_df['price_to_period_high'] / self.data_df['period_high_low_range']

            self.data_df['prev_price_to_period_low_pct'] = self.data_df['price_to_period_low_pct'].shift(1)
            self.data_df['prev_price_to_period_high_pct'] = self.data_df['price_to_period_high_pct'].shift(1)

            self.data_df['prev1_price_to_period_low_pct'] = self.data_df['prev_price_to_period_low_pct']
            self.data_df['prev1_price_to_period_high_pct'] = self.data_df['prev_price_to_period_high_pct']

            for i in range(2, close_position_look_back + 1):
                self.data_df['prev' + str(i) + '_price_to_period_low_pct'] = self.data_df['prev' + str(i-1) + '_price_to_period_low_pct'].shift(1)
                self.data_df['prev' + str(i) + '_price_to_period_high_pct'] = self.data_df['prev' + str(i-1) + '_price_to_period_high_pct'].shift(1)


            #self.data_df['prev2_price_to_period_low_pct'] = self.data_df['prev_price_to_period_low_pct'].shift(1)
            #self.data_df['prev2_price_to_period_high_pct'] = self.data_df['prev_price_to_period_high_pct'].shift(1)



            self.data_df['min_price_to_period_low_pct'] = (self.data_df['min_price'] - self.data_df['period_low' + str(high_low_window)]) / self.data_df['period_high_low_range']
            self.data_df['max_price_to_period_high_pct'] = (self.data_df['period_high' + str(high_low_window)] - self.data_df['max_price']) / self.data_df['period_high_low_range']


            self.data_df['prev1_min_price_to_period_low_pct'] = self.data_df['min_price_to_period_low_pct'].shift(1)
            self.data_df['prev1_max_price_to_period_high_pct'] = self.data_df['max_price_to_period_high_pct'].shift(1)

            for i in range(2, max(reverse_trade_look_back, close_position_look_back)+1):
                self.data_df['prev' + str(i) + '_min_price_to_period_low_pct'] = self.data_df['prev' + str(i-1) + '_min_price_to_period_low_pct'].shift(1)
                self.data_df['prev' + str(i) + '_max_price_to_period_high_pct'] = self.data_df['prev' + str(i-1) + '_max_price_to_period_high_pct'].shift(1)



            self.data_df['low_price_to_period_low_pct'] = (self.data_df['low'] - self.data_df['period_low' + str(high_low_window)]) / self.data_df['period_high_low_range']
            self.data_df['high_price_to_period_high_pct'] = (self.data_df['period_high' + str(high_low_window)] - self.data_df['high']) / self.data_df['period_high_low_range']

            self.data_df['prev1_low_price_to_period_low_pct'] = self.data_df['low_price_to_period_low_pct'].shift(1)
            self.data_df['prev1_high_price_to_period_high_pct'] = self.data_df['high_price_to_period_high_pct'].shift(1)

            for i in range(2, max(reverse_trade_look_back, close_position_look_back)+1):
                self.data_df['prev' + str(i) + '_low_price_to_period_low_pct'] = self.data_df['prev' + str(i-1) + '_low_price_to_period_low_pct'].shift(1)
                self.data_df['prev' + str(i) + '_high_price_to_period_high_pct'] = self.data_df['prev' + str(i-1) + '_high_price_to_period_high_pct'].shift(1)


            # self.data_df['prev2_min_price_to_period_low_pct'] = self.data_df['prev_min_price_to_period_low_pct'].shift(1)
            # self.data_df['prev2_max_price_to_period_high_pct'] = self.data_df['prev_max_price_to_period_high_pct'].shift(1)



            # self.data_df['prev_open'] = self.data_df['open'].shift(1)
            # self.data_df['prev_close'] = self.data_df['close'].shift(1)

            self.data_df['is_positive'] = (self.data_df['close'] > self.data_df['open'])
            self.data_df['is_negative'] = (self.data_df['close'] < self.data_df['open'])

            self.data_df['prev_high'] = self.data_df['high'].shift(1)
            self.data_df['prev_low'] = self.data_df['low'].shift(1)

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



            self.data_df['middle'] = (self.data_df['open'] + self.data_df['close']) / 2

            self.data_df['prev1_min_price'] = self.data_df['min_price'].shift(1)
            for i in range(2, max(ma12_lookback, large_bar_look_back) + 1):
                self.data_df['prev' + str(i) + '_min_price'] = self.data_df['prev' + str(i-1) + '_min_price'].shift(1)

            self.data_df['prev1_max_price'] = self.data_df['max_price'].shift(1)
            for i in range(2, max(ma12_lookback, large_bar_look_back) + 1):
                self.data_df['prev' + str(i) + '_max_price'] = self.data_df['prev' + str(i - 1) + '_max_price'].shift(1)

            self.data_df['prev1_middle'] = self.data_df['middle'].shift(1)
            for i in range(2, max(ma12_lookback, large_bar_look_back) + 1):
                self.data_df['prev' + str(i) + '_middle'] = self.data_df['prev' + str(i-1) + '_middle'].shift(1)


            self.data_df['price_range'] = self.data_df['max_price'] - self.data_df['min_price']
            self.data_df['price_volatility'] = self.data_df['high'] - self.data_df['low']

            self.data_df['is_needle_bar'] = self.data_df['price_range'] / self.data_df['price_volatility'] < 0.3

            self.data_df['prev_price_range'] = self.data_df['price_range'].shift(1)

            self.data_df['positive_price_range'] = np.where(self.data_df['close'] > self.data_df['open'],
                                                              self.data_df['price_range'],
                                                              0)
            self.data_df['negative_price_range'] = np.where(self.data_df['close'] < self.data_df['open'],
                                                              self.data_df['price_range'],
                                                              0)

            self.data_df['recent_max_price_range'] = self.data_df['price_range'].rolling(price_range_look_back, min_periods = price_range_look_back).max()
            self.data_df['prev_recent_max_price_range'] = self.data_df['recent_max_price_range'].shift(1)




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
            aligned_long_conditions_go_on = [(self.data_df[guppy_lines[i]] > self.data_df[guppy_lines[i + 1]]) for i in range(3,5)] + \
                                            [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines[4:]]
            #all_up_conditions = [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines]
            aligned_long_conditions2 = aligned_long_conditions1[0:2] + [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines[0:3]]
            aligned_long_conditions3 = aligned_long_conditions1 + [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines]
            aligned_long_conditions4 = aligned_long_conditions1 + \
                                       [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate > -1) for guppy_line in guppy_lines]
            aligned_long_conditions5 = [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines]

            #self.data_df['is_guppy_aligned_long'] = reduce(lambda left, right: left & right, aligned_long_conditions) # + all_up_conditions)
            aligned_long_condition1 = reduce(lambda left, right: left & right, aligned_long_conditions1)
            aligned_long_condition_go_on = reduce(lambda left, right: left & right, aligned_long_conditions_go_on)
            aligned_long_condition2 = reduce(lambda left, right: left & right, aligned_long_conditions2)
            aligned_long_condition3 = reduce(lambda left, right: left & right, aligned_long_conditions3)
            aligned_long_condition4 = reduce(lambda left, right: left & right, aligned_long_conditions4)
            aligned_long_condition5 = reduce(lambda left, right: left & right, aligned_long_conditions5)

            half_aligned_long_condition = reduce(lambda left, right: left & right, aligned_long_conditions1[0:2])
            strongly_half_aligned_long_condition = aligned_long_condition2
            strongly_aligned_long_condition = aligned_long_condition3
            strongly_relaxed_aligned_long_condition = aligned_long_condition4
            strongly_long_condition = aligned_long_condition5

            self.data_df['is_guppy_aligned_long'] = aligned_long_condition1 #| aligned_long_condition2
            self.data_df['aligned_long_condition_go_on'] = aligned_long_condition_go_on
            self.data_df['strongly_half_aligned_long_condition'] = strongly_half_aligned_long_condition
            self.data_df['strongly_aligned_long_condition'] = strongly_aligned_long_condition
            self.data_df['strongly_relaxed_aligned_long_condition'] = strongly_relaxed_aligned_long_condition
            self.data_df['strongly_long_condition'] = strongly_long_condition

            aligned_short_conditions1 = [(self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]]) for i in
                                        range(len(guppy_lines) - 1)]
            aligned_short_conditions_go_on = [(self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]]) for i in range(3,5)] + \
                                            [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines[4:]]
            #all_down_conditions = [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]
            aligned_short_conditions2 = aligned_short_conditions1[0:2] + [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines[0:3]]
            aligned_short_conditions3 = aligned_short_conditions1 + [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]
            aligned_short_conditions4 = aligned_short_conditions1 + \
                                       [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate < 1) for guppy_line in guppy_lines]
            aligned_short_conditions5 = [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]


            #self.data_df['is_guppy_aligned_short'] = reduce(lambda left, right: left & right, aligned_short_conditions) # + all_down_conditions)
            aligned_short_condition1 = reduce(lambda left, right: left & right, aligned_short_conditions1)
            aligned_short_condition_go_on = reduce(lambda left, right: left & right, aligned_short_conditions_go_on)
            aligned_short_condition2 = reduce(lambda left, right: left & right, aligned_short_conditions2)
            aligned_short_condition3 = reduce(lambda left, right: left & right, aligned_short_conditions3)
            aligned_short_condition4 = reduce(lambda left, right: left & right, aligned_short_conditions4)
            aligned_short_condition5 = reduce(lambda left, right: left & right, aligned_short_conditions5)

            half_aligned_short_condition = reduce(lambda left, right: left & right, aligned_short_conditions1[0:2])
            strongly_half_aligned_short_condition = aligned_short_condition2
            strongly_aligned_short_condition = aligned_short_condition3
            strongly_relaxed_aligned_short_condition = aligned_short_condition4
            strongly_short_condition = aligned_short_condition5

            self.data_df['is_guppy_aligned_short'] = aligned_short_condition1 # | aligned_short_condition2
            self.data_df['aligned_short_condition_go_on'] = aligned_short_condition_go_on
            self.data_df['strongly_half_aligned_short_condition'] = strongly_half_aligned_short_condition
            self.data_df['strongly_aligned_short_condition'] = strongly_aligned_short_condition
            self.data_df['strongly_relaxed_aligned_short_condition'] = strongly_relaxed_aligned_short_condition
            self.data_df['strongly_short_condition'] = strongly_short_condition

            df_temp = self.data_df[guppy_lines]
            df_temp = df_temp.apply(sorted, axis=1).apply(pd.Series)
            sorted_guppys = ['guppy1', 'guppy2', 'guppy3', 'guppy4', 'guppy5', 'guppy6']
            df_temp.columns = sorted_guppys
            self.data_df = pd.concat([self.data_df, df_temp], axis=1)

            self.data_df['highest_guppy'] = self.data_df['guppy6']
            self.data_df['lowest_guppy'] = self.data_df['guppy1']

            self.data_df['second_highest_guppy'] = self.data_df['guppy5']
            self.data_df['second_lowest_guppy'] = self.data_df['guppy2']

            self.data_df['prev1_highest_guppy'] = self.data_df['highest_guppy'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_highest_guppy'] = self.data_df['prev' + str(i-1) + '_highest_guppy'].shift(1)

            self.data_df['prev1_lowest_guppy'] = self.data_df['lowest_guppy'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_lowest_guppy'] = self.data_df['prev' + str(i-1) + '_lowest_guppy'].shift(1)


            self.data_df['upper_band_close_gradient'] = self.data_df['upper_band_close'].diff()
            self.data_df['lower_band_close_gradient'] = self.data_df['lower_band_close'].diff()

            self.data_df['upper_band_go_up'] = self.data_df['upper_band_close_gradient'] > 0
            self.data_df['lower_band_go_down'] = self.data_df['lower_band_close_gradient'] < 0

            self.data_df['break_upper_bolling'] = self.data_df['high'] > self.data_df['upper_band_close']
            self.data_df['break_lower_bolling'] = self.data_df['low'] < self.data_df['lower_band_close']

            self.data_df['strict_break_upper_bolling'] = self.data_df['close'] > self.data_df['upper_band_close']
            self.data_df['strict_break_lower_bolling'] = self.data_df['close'] < self.data_df['lower_band_close']


            self.data_df['upper_vegas'] = self.data_df[['ma_close144', 'ma_close169']].max(axis=1)
            self.data_df['lower_vegas'] = self.data_df[['ma_close144', 'ma_close169']].min(axis=1)

            ################# features to detect trend approaching the end ###############
            self.data_df['period_high' + str(high_low_window) + '_gradient'] = self.data_df['period_high' + str(high_low_window)].diff()
            self.data_df['period_high' + str(high_low_window) + '_go_up'] = np.where(
                self.data_df['period_high' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate >= high_low_delta_threshold,
                1,
                0
            )
            self.data_df['period_high' + str(high_low_window) + '_go_down'] = np.where(
                self.data_df['period_high' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate <= -high_low_delta_threshold,
                1,
                0
            )



            self.data_df['period_high' + str(high_low_window) + '_go_up_num'] = \
                self.data_df['period_high' + str(high_low_window) + '_go_up'].rolling(period_lookback, min_periods = period_lookback).sum()
            self.data_df['period_high' + str(high_low_window) + '_go_down_num'] = \
                self.data_df['period_high' + str(high_low_window) + '_go_down'].rolling(period_lookback, min_periods = period_lookback).sum()


            #self.data_df['period_high' + str(high_low_window) + '_go_up_num'] = self.data_df['period_high' + str(high_low_window) + '_go_up_num'].astype(int)
            #self.data_df['period_high' + str(high_low_window) + '_go_down_num'] = self.data_df['period_high' + str(high_low_window) + '_go_down_num'].astype(int)


            self.data_df['period_low' + str(high_low_window) + '_gradient'] = self.data_df['period_low' + str(high_low_window)].diff()
            self.data_df['period_low' + str(high_low_window) + '_go_up'] = np.where(
                self.data_df['period_low' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate >= high_low_delta_threshold,
                1,
                0
            )
            self.data_df['period_low' + str(high_low_window) + '_go_down'] = np.where(
                self.data_df['period_low' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate <= -high_low_delta_threshold,
                1,
                0
            )



            self.data_df['period_low' + str(high_low_window) + '_go_up_num'] = \
                self.data_df['period_low' + str(high_low_window) + '_go_up'].rolling(period_lookback, min_periods = period_lookback).sum()
            self.data_df['period_low' + str(high_low_window) + '_go_down_num'] = \
                self.data_df['period_low' + str(high_low_window) + '_go_down'].rolling(period_lookback, min_periods = period_lookback).sum()







            self.data_df['m12_to_upper_vegas'] = self.data_df['ma_close12'] - self.data_df['upper_vegas']
            self.data_df['m12_to_lower_vegas'] = self.data_df['ma_close12'] - self.data_df['lower_vegas']

            self.data_df['prev_m12_to_upper_vegas'] = self.data_df['m12_to_upper_vegas'].shift(1)
            self.data_df['prev_m12_to_lower_vegas'] = self.data_df['m12_to_lower_vegas'].shift(1)

            self.data_df['cross_up_vegas'] = np.where(
                (self.data_df['prev_m12_to_upper_vegas'] <= 0) & (self.data_df['m12_to_upper_vegas'] > 0),
                1,
                0
            )

            self.data_df['cross_up_lower_vegas'] = np.where(
                (self.data_df['prev_m12_to_lower_vegas'] <= 0) & (self.data_df['m12_to_lower_vegas'] > 0),
                1,
                0
            )



            self.data_df['cross_down_vegas'] = np.where(
                (self.data_df['prev_m12_to_lower_vegas'] >= 0) & (self.data_df['m12_to_lower_vegas'] < 0),
                -1,
                0
            )

            self.data_df['cross_down_upper_vegas'] = np.where(
                (self.data_df['prev_m12_to_upper_vegas'] >= 0) & (self.data_df['m12_to_upper_vegas'] < 0),
                -1,
                0
            )




            #self.data_df['period_low' + str(high_low_window) + '_go_up_num'] = self.data_df['period_low' + str(high_low_window) + '_go_up_num'].astype(int)
            #self.data_df['period_low' + str(high_low_window) + '_go_down_num'] = self.data_df['period_low' + str(high_low_window) + '_go_down_num'].astype(int)



            self.data_df['is_break_upper_bolling'] = np.where(
                self.data_df['strict_break_upper_bolling'],
                1,
                0
            )
            self.data_df['is_break_lower_bolling'] = np.where(
                self.data_df['strict_break_lower_bolling'],
                1,
                0
            )
            self.data_df['is_upper_band_go_up'] = np.where(
                self.data_df['upper_band_go_up'],
                1,
                0
            )
            self.data_df['is_lower_band_go_down'] = np.where(
                self.data_df['lower_band_go_down'],
                1,
                0
            )


            self.data_df['break_upper_band_temp_state'] = self.data_df['is_break_upper_bolling'] * 2 + self.data_df['is_upper_band_go_up']
            self.data_df['break_lower_band_temp_state'] = self.data_df['is_break_lower_bolling'] * 2 + self.data_df['is_lower_band_go_down']

            self.data_df['break_upper_band_state'] = np.nan
            self.data_df['break_upper_band_state'] = np.where(
                self.data_df['break_upper_band_temp_state'] == 1,
                self.data_df['break_upper_band_state'],
                self.data_df['break_upper_band_temp_state']
            )

            self.data_df['break_lower_band_state'] = np.nan
            self.data_df['break_lower_band_state'] = np.where(
                self.data_df['break_lower_band_temp_state'] == 1,
                self.data_df['break_lower_band_state'],
                self.data_df['break_lower_band_temp_state']
            )

            dummy_df = self.data_df[['break_upper_band_state', 'break_lower_band_state']]
            dummy_df = dummy_df.fillna(method = 'ffill').fillna(0)
            self.data_df['break_upper_band_state'] = dummy_df['break_upper_band_state']
            self.data_df['break_lower_band_state'] = dummy_df['break_lower_band_state']

            self.data_df['prev_break_upper_band_state'] = self.data_df['break_upper_band_state'].shift(1)
            self.data_df['prev_break_lower_band_state'] = self.data_df['break_lower_band_state'].shift(1)

            self.data_df['new_break_up_upper_band'] = (self.data_df['prev_break_upper_band_state'] == 0) & \
                                                         ((self.data_df['break_upper_band_state'] == 2) | (self.data_df['break_upper_band_state'] == 3))

            self.data_df['new_break_down_lower_band'] = (self.data_df['prev_break_lower_band_state'] == 0) & \
                                                         ((self.data_df['break_lower_band_state'] == 2) | (self.data_df['break_lower_band_state'] == 3))



            # self.data_df['prev_is_break_upper_bolling'] = self.data_df['is_break_upper_bolling'].shift(1)
            # self.data_df['prev_is_break_lower_bolling'] = self.data_df['is_break_lower_bolling'].shift(1)
            # self.data_df['prev_is_upper_band_go_up'] = self.data_df['is_upper_band_go_up'].shift(1)
            # self.data_df['prev_is_lower_band_go_down'] = self.data_df['is_lower_band_go_down'].shift(1)
            #
            # self.data_df['new_break_up_upper_band'] = (self.data_df['prev_is_break_upper_bolling'] == 0) & \
            #                                                 (self.data_df['prev_is_upper_band_go_up'] == 0) & \
            #                                                 (self.data_df['is_break_upper_bolling'] == 1)
            #
            # self.data_df['new_break_down_lower_band'] = (self.data_df['prev_is_break_lower_bolling'] == 0) & \
            #                                                 (self.data_df['prev_is_lower_band_go_down'] == 0) & \
            #                                                 (self.data_df['is_break_lower_bolling'] == 1)


            self.data_df['is_new_break_up_upper_band'] = np.where(
                self.data_df['new_break_up_upper_band'],
                1,
                0
            )

            self.data_df['is_new_break_down_lower_band'] = np.where(
                self.data_df['new_break_down_lower_band'],
                1,
                0
            )



            # self.data_df['is_breaking_up_upper_band'] = self.data_df['break_upper_bolling'] | self.data_df['upper_band_go_up']
            # self.data_df['is_breaking_down_lower_band'] = self.data_df['break_lower_bolling'] | self.data_df['lower_band_go_down']
            #
            # self.data_df['breaking_up_upper_band'] = np.where(
            #     self.data_df['is_breaking_up_upper_band'],
            #     1,
            #     0
            # )
            #
            # self.data_df['breaking_down_lower_band'] = np.where(
            #     self.data_df['is_breaking_down_lower_band'],
            #     1,
            #     0
            # )
            #
            # self.data_df['breaking_up_upper_band_diff'] = self.data_df['breaking_up_upper_band'].diff()
            # self.data_df['breaking_down_lower_band_diff'] = self.data_df['breaking_down_lower_band'].diff()
            #
            # self.data_df['is_new_break_up_upper_band'] = np.where(
            #     self.data_df['breaking_up_upper_band_diff'] == 1,
            #     1,
            #     0
            # )
            #
            # self.data_df['is_new_break_down_lower_band'] = np.where(
            #     self.data_df['breaking_down_lower_band_diff'] == 1,
            #     1,
            #     0
            # )



            # self.data_df['period_break_up_upper_band_num'] = self.data_df['is_new_break_up_upper_band'].rolling(period_lookback, min_periods = period_lookback).sum()
            # self.data_df['period_break_down_lower_band_num'] = self.data_df['is_new_break_down_lower_band'].rolling(period_lookback, min_periods = period_lookback).sum()

            #'period_high' + str(high_low_window) + '_go_up'
            self.data_df['cum_num_new_break_up'] = self.data_df['is_new_break_up_upper_band'].cumsum()
            self.data_df['cum_num_new_break_down'] = self.data_df['is_new_break_down_lower_band'].cumsum()

            self.data_df['cum_num_high_go_up'] = self.data_df['period_high' + str(high_low_window) + '_go_up'].cumsum()
            self.data_df['cum_num_high_go_down'] = self.data_df['period_high' + str(high_low_window) + '_go_down'].cumsum()
            self.data_df['cum_num_low_go_up'] = self.data_df['period_low' + str(high_low_window) + '_go_up'].cumsum()
            self.data_df['cum_num_low_go_down'] = self.data_df['period_low' + str(high_low_window) + '_go_down'].cumsum()



            self.data_df['cum_num_new_break_up'] = self.data_df['cum_num_new_break_up'].shift(1)
            self.data_df.at[0, 'cum_num_new_break_up'] = 0

            self.data_df['cum_num_new_break_down'] = self.data_df['cum_num_new_break_down'].shift(1)
            self.data_df.at[0, 'cum_num_new_break_down'] = 0


            self.data_df['cum_num_high_go_up'] = self.data_df['cum_num_high_go_up'].shift(1)
            self.data_df.at[0, 'cum_num_high_go_up'] = 0

            self.data_df['cum_num_high_go_down'] = self.data_df['cum_num_high_go_down'].shift(1)
            self.data_df.at[0, 'cum_num_high_go_down'] = 0

            self.data_df['cum_num_low_go_up'] = self.data_df['cum_num_low_go_up'].shift(1)
            self.data_df.at[0, 'cum_num_low_go_up'] = 0

            self.data_df['cum_num_low_go_down'] = self.data_df['cum_num_low_go_down'].shift(1)
            self.data_df.at[0, 'cum_num_low_go_down'] = 0

            cum_columns = ['cum_num_new_break_up', 'cum_num_new_break_down',
                           'cum_num_high_go_up', 'cum_num_high_go_down', 'cum_num_low_go_up', 'cum_num_low_go_down']


            self.data_df['id'] = list(range(self.data_df.shape[0]))



            #############################################################
            self.data_df.at[0, 'period_high' + str(high_low_window) + '_cum_gradient'] = 0
            self.data_df.at[0, 'period_low' + str(high_low_window) + '_cum_gradient'] = 0

            self.data_df['period_high' + str(high_low_window) + '_cum_gradient'] = self.data_df['period_high' + str(high_low_window) + '_gradient'].cumsum()
            self.data_df['period_low' + str(high_low_window) + '_cum_gradient'] = self.data_df['period_low' + str(high_low_window) + '_gradient'].cumsum()

            self.data_df['period_high' + str(high_low_window) + '_cum_gradient'] = self.data_df['period_high' + str(high_low_window) + '_cum_gradient'].shift(1)
            self.data_df['period_low' + str(high_low_window) + '_cum_gradient'] = self.data_df['period_low' + str(high_low_window) + '_cum_gradient'].shift(1)

            self.data_df.at[0, 'period_high' + str(high_low_window) + '_cum_gradient'] = 0
            self.data_df.at[0, 'period_low' + str(high_low_window) + '_cum_gradient'] = 0



            self.data_df['cross_vegas'] = self.data_df['cross_up_vegas'] + self.data_df['cross_down_vegas']

            # print("Previous cross_vegas:")
            # print(self.data_df[self.data_df['cross_vegas'] != 0][['time','cross_vegas']])

            temp_df = self.data_df[['id', 'cross_vegas']]
            temp_df = temp_df[temp_df['cross_vegas'] != 0]
            temp_df.reset_index(inplace = True)
            temp_df = temp_df.drop(columns = ['index'])
            temp_df['prev_cross_vegas'] = temp_df['cross_vegas'].shift(1)
            temp_df = temp_df[temp_df['cross_vegas'] != temp_df['prev_cross_vegas']]
            temp_df = temp_df[['id', 'cross_vegas']]
            temp_df = temp_df.rename(columns = {'cross_vegas' : 'actual_cross_vegas'})
            self.data_df = pd.merge(self.data_df, temp_df, on = ['id'], how = 'left')
            self.data_df['cross_vegas'] = np.where(
                self.data_df['actual_cross_vegas'].notnull(),
                self.data_df['cross_vegas'],
                0
            )

            self.data_df['actual_cross_up_vegas'] = np.where(
                self.data_df['cross_vegas'] == 1,
                1,
                0
            )

            self.data_df['actual_cross_down_vegas'] = np.where(
                self.data_df['cross_vegas'] == -1,
                1,
                0
            )



            # print("After cross_vegas:")
            # print(self.data_df[self.data_df['cross_vegas'] != 0][['time', 'cross_vegas']])


            # self.data_df['cross_vegas'] = self.data_df['cross_up_vegas'] + self.data_df['cross_down_vegas']
            # self.data_df['cross_vegas'] = np.where(
            #     self.data_df['cross_vegas'] == 2,
            #     1,
            #     self.data_df['cross_vegas']
            # )
            self.data_df['cross_vegas_temp'] = np.nan
            self.data_df['cross_vegas_temp'] = np.where(
                self.data_df['cross_vegas'] != 0,
                self.data_df['id'],
                self.data_df['cross_vegas_temp']
            )

            #Focus here
            df_cross_vegas = self.data_df[self.data_df['cross_vegas_temp'].notnull()][['id',
                                                                                       'period_high' + str(high_low_window) + '_cum_gradient',
                                                                                       'period_low' + str(high_low_window) + '_cum_gradient'
                                                                                       ]]
            df_cross_vegas.reset_index(inplace = True)
            df_cross_vegas = df_cross_vegas.drop(columns = ['index'])


            ##############################################

            temp_df = self.data_df[['id', 'cross_vegas_temp']]
            temp_df = temp_df.fillna(method='ffill').fillna(0)
            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)

            temp_df['on_one_side_vegas_duration'] = temp_df['id'] - temp_df['cross_vegas_temp'] + 1

            temp_df['id'] = temp_df['cross_vegas_temp']

            temp_df = pd.merge(temp_df, df_cross_vegas, on=['id'], how='left')
            temp_df = temp_df.rename(columns={
                'period_high' + str(high_low_window) + '_cum_gradient': 'period_high' + str(high_low_window) + '_cum_gradient_for_vegas',
                'period_low' + str(high_low_window) + '_cum_gradient': 'period_low' + str(high_low_window) + '_cum_gradient_for_vegas'
            })
            temp_df = temp_df.fillna(0)

            temp_df = temp_df[[col for col in temp_df.columns if 'cum_gradient' in col or 'duration' in col]]
            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(float)

            self.data_df = pd.concat([self.data_df, temp_df], axis = 1)
            self.data_df = self.data_df.drop(columns = [col for col in self.data_df.columns if 'temp' in col])
            self.data_df['prev_on_one_side_vegas_duration'] = self.data_df['on_one_side_vegas_duration'].shift(1)

            #Gay
            self.data_df['period_high' + str(high_low_window) + '_vegas_gradient'] = \
                self.data_df['period_high' + str(high_low_window) + '_cum_gradient'] - self.data_df['period_high' + str(high_low_window) + '_cum_gradient_for_vegas']

            self.data_df['period_low' + str(high_low_window) + '_vegas_gradient'] = \
                self.data_df['period_low' + str(high_low_window) + '_cum_gradient'] - self.data_df['period_low' + str(high_low_window) + '_cum_gradient_for_vegas']


            self.data_df['period_high_low_vegas_gradient_ratio'] = np.where(
                self.data_df['period_low' + str(high_low_window) + '_vegas_gradient'] == 0,
                10000,
                self.data_df['period_high' + str(high_low_window) + '_vegas_gradient']/self.data_df['period_low' + str(high_low_window) + '_vegas_gradient']
            )

            self.data_df['period_high_low_vegas_gradient_ratio'] = np.where(
                (self.data_df['period_low' + str(high_low_window) + '_vegas_gradient'] == 0) & (self.data_df['period_high' + str(high_low_window) + '_vegas_gradient'] == 0),
                0,
                self.data_df['period_high_low_vegas_gradient_ratio']
            )


            self.data_df['period_low_high_vegas_gradient_ratio'] = np.where(
                self.data_df['period_high' + str(high_low_window) + '_vegas_gradient'] == 0,
                10000,
                self.data_df['period_low' + str(high_low_window) + '_vegas_gradient']/self.data_df['period_high' + str(high_low_window) + '_vegas_gradient']
            )

            self.data_df['period_low_high_vegas_gradient_ratio'] = np.where(
                (self.data_df['period_low' + str(high_low_window) + '_vegas_gradient'] == 0) & (self.data_df['period_high' + str(high_low_window) + '_vegas_gradient'] == 0),
                0,
                self.data_df['period_low_high_vegas_gradient_ratio']
            )




            # print("Investigate gradient:")
            # self.data_df['high_cum_gradient'] = self.data_df['period_high' + str(high_low_window) + '_cum_gradient']
            # self.data_df['high_cum_gradient2'] = self.data_df['period_high' + str(high_low_window) + '_vegas_gradient']
            #
            # self.data_df['low_cum_gradient'] = self.data_df['period_low' + str(high_low_window) + '_cum_gradient']
            # self.data_df['low_cum_gradient2'] = self.data_df['period_low' + str(high_low_window) + '_vegas_gradient']
            #
            # self.data_df['high_low'] = self.data_df['period_high_low_vegas_gradient_ratio']
            # self.data_df['low_high'] = self.data_df['period_low_high_vegas_gradient_ratio']
            #
            # print(self.data_df[['time','cross_vegas',
            #                     'high_cum_gradient', 'high_cum_gradient2', 'low_cum_gradient', 'low_cum_gradient2',
            #                     'high_low', 'low_high']].tail(350))

            ##############################################
            #sys.exit(0)
            #############################################################







            for col in cum_columns:
                #print("col = " + col)
                self.data_df[col] = self.data_df[col].astype(int)

            self.data_df['period_high' + str(high_low_window) + '_go_up_temp'] = np.nan
            self.data_df['period_high' + str(high_low_window) + '_go_up_temp'] = np.where(
                self.data_df['period_high' + str(high_low_window) + '_go_up'] == 1,
                self.data_df['id'],
                self.data_df['period_high' + str(high_low_window) + '_go_up_temp']
            )

            df_high_go_up = self.data_df[self.data_df['period_high' + str(high_low_window) + '_go_up_temp'].notnull()]\
                                        [['id'] + cum_columns + ['period_low' + str(high_low_window)]]
            df_high_go_up.reset_index(inplace = True)
            df_high_go_up = df_high_go_up.drop(columns = ['index'])

            # print("Create df_high_go_up:")
            # print("length = " + str(df_high_go_up.shape[0]))
            # print(df_high_go_up.head(20))
            # print(df_high_go_up.tail(20))


            self.data_df['period_high' + str(high_low_window) + '_go_down_temp'] = np.nan
            self.data_df['period_high' + str(high_low_window) + '_go_down_temp'] = np.where(
                self.data_df['period_high' + str(high_low_window) + '_go_down'] == 1,
                self.data_df['id'],
                self.data_df['period_high' + str(high_low_window) + '_go_up_temp']
            )

            df_high_go_down = self.data_df[self.data_df['period_high' + str(high_low_window) + '_go_down_temp'].notnull()]\
                                          [['id'] + cum_columns + ['period_low' + str(high_low_window)]]

            df_high_go_down.reset_index(inplace = True)
            df_high_go_down = df_high_go_down.drop(columns = ['index'])


            self.data_df['period_low' + str(high_low_window) + '_go_up_temp'] = np.nan
            self.data_df['period_low' + str(high_low_window) + '_go_up_temp'] = np.where(
                self.data_df['period_low' + str(high_low_window) + '_go_up'] == 1,
                self.data_df['id'],
                self.data_df['period_low' + str(high_low_window) + '_go_up_temp']
            )

            df_low_go_up = self.data_df[self.data_df['period_low' + str(high_low_window) + '_go_up_temp'].notnull()]\
                                       [['id'] + cum_columns + ['period_high' + str(high_low_window)]]

            df_low_go_up.reset_index(inplace = True)
            df_low_go_up = df_low_go_up.drop(columns = ['index'])


            self.data_df['period_low' + str(high_low_window) + '_go_down_temp'] = np.nan
            self.data_df['period_low' + str(high_low_window) + '_go_down_temp'] = np.where(
                self.data_df['period_low' + str(high_low_window) + '_go_down'] == 1,
                self.data_df['id'],
                self.data_df['period_low' + str(high_low_window) + '_go_down_temp']
            )

            df_low_go_down = self.data_df[self.data_df['period_low' + str(high_low_window) + '_go_down_temp'].notnull()]\
                                         [['id'] + cum_columns + ['period_high' + str(high_low_window)]]


            df_low_go_down.reset_index(inplace = True)
            df_low_go_down = df_low_go_down.drop(columns = ['index'])

            #
            # print("df_low_go_down:")
            # print(df_low_go_down.tail(20))


            temp_df = self.data_df[['id',
                                    'period_high' + str(high_low_window) + '_go_up_temp',
                                    'period_high' + str(high_low_window) + '_go_down_temp',
                                    'period_low' + str(high_low_window) + '_go_up_temp',
                                    'period_low' + str(high_low_window) + '_go_down_temp']]

            temp_df = temp_df.fillna(method = 'ffill').fillna(0)

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)



            temp_df['period_high' + str(high_low_window) + '_go_up_duration'] = temp_df['id'] - temp_df['period_high' + str(high_low_window) + '_go_up_temp'] + 1
            temp_df['period_high' + str(high_low_window) + '_go_down_duration'] = temp_df['id'] - temp_df['period_high' + str(high_low_window) + '_go_down_temp']  + 1
            temp_df['period_low' + str(high_low_window) + '_go_up_duration'] = temp_df['id'] - temp_df['period_low' + str(high_low_window) + '_go_up_temp'] + 1
            temp_df['period_low' + str(high_low_window) + '_go_down_duration'] = temp_df['id'] - temp_df['period_low' + str(high_low_window) + '_go_down_temp'] + 1


            temp_df['id'] = temp_df['period_high' + str(high_low_window) + '_go_up_temp']


            temp_df = pd.merge(temp_df, df_high_go_up, on = ['id'], how = 'left')
            temp_df = temp_df.rename(columns = {
                'cum_num_new_break_up' : 'cum_num_new_break_up_for_high_go_up',
                'cum_num_new_break_down' : 'cum_num_new_break_down_for_high_go_up',
                'cum_num_high_go_up' : 'cum_num_high_go_up_for_high_go_up',
                'cum_num_high_go_down': 'cum_num_high_go_down_for_high_go_up',
                'cum_num_low_go_up': 'cum_num_low_go_up_for_high_go_up',
                'cum_num_low_go_down': 'cum_num_low_go_down_for_high_go_up',
                'period_low' + str(high_low_window) : 'period_low' + str(high_low_window) + "_for_high_go_up"
            })

            temp_df = temp_df.fillna(0)



            temp_df['id'] = temp_df['period_high' + str(high_low_window) + '_go_down_temp']
            temp_df = pd.merge(temp_df, df_high_go_down, on = ['id'], how = 'left')
            temp_df = temp_df.rename(columns = {
                'cum_num_new_break_up' : 'cum_num_new_break_up_for_high_go_down',
                'cum_num_new_break_down' : 'cum_num_new_break_down_for_high_go_down',
                'cum_num_high_go_up': 'cum_num_high_go_up_for_high_go_down',
                'cum_num_high_go_down': 'cum_num_high_go_down_for_high_go_down',
                'cum_num_low_go_up': 'cum_num_low_go_up_for_high_go_down',
                'cum_num_low_go_down': 'cum_num_low_go_down_for_high_go_down',
                'period_low' + str(high_low_window): 'period_low' + str(high_low_window) + "_for_high_go_down"
            })
            temp_df = temp_df.fillna(0)


            temp_df['id'] = temp_df['period_low' + str(high_low_window) + '_go_up_temp']
            temp_df = pd.merge(temp_df, df_low_go_up, on = ['id'], how = 'left')
            temp_df = temp_df.rename(columns = {
                'cum_num_new_break_up' : 'cum_num_new_break_up_for_low_go_up',
                'cum_num_new_break_down' : 'cum_num_new_break_down_for_low_go_up',
                'cum_num_high_go_up': 'cum_num_high_go_up_for_low_go_up',
                'cum_num_high_go_down': 'cum_num_high_go_down_for_low_go_up',
                'cum_num_low_go_up': 'cum_num_low_go_up_for_low_go_up',
                'cum_num_low_go_down': 'cum_num_low_go_down_for_low_go_up',
                'period_high' + str(high_low_window): 'period_high' + str(high_low_window) + "_for_low_go_up"
            })
            temp_df = temp_df.fillna(0)

            temp_df['id'] = temp_df['period_low' + str(high_low_window) + '_go_down_temp']
            temp_df = pd.merge(temp_df, df_low_go_down, on = ['id'], how = 'left')
            temp_df = temp_df.rename(columns = {
                'cum_num_new_break_up' : 'cum_num_new_break_up_for_low_go_down',
                'cum_num_new_break_down' : 'cum_num_new_break_down_for_low_go_down',
                'cum_num_high_go_up': 'cum_num_high_go_up_for_low_go_down',
                'cum_num_high_go_down': 'cum_num_high_go_down_for_low_go_down',
                'cum_num_low_go_up': 'cum_num_low_go_up_for_low_go_down',
                'cum_num_low_go_down': 'cum_num_low_go_down_for_low_go_down',
                'period_high' + str(high_low_window): 'period_high' + str(high_low_window) + "_for_low_go_down"
            })
            temp_df = temp_df.fillna(0)



            temp_df = temp_df[[col for col in temp_df.columns if 'duration' in col or 'cum_num' in col]]

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)

            # print("Final temp_df:")
            # print("length = " + str(temp_df.shape[0]))
            # print(temp_df.head(20))
            # print(temp_df.tail(20))
            #
            # print("data_df:")
            # print("length = " + str(self.data_df.shape[0]))
            # print(self.data_df.head(10))
            # print(self.data_df.tail(10))

            self.data_df = pd.concat([self.data_df, temp_df], axis = 1)


            self.data_df = self.data_df.drop(columns = [col for col in self.data_df.columns if 'temp' in col])  #Remove temp


            self.data_df['num_new_break_up_in_high_go_up'] = self.data_df['cum_num_new_break_up'] - self.data_df['cum_num_new_break_up_for_high_go_up']
            self.data_df['num_new_break_up_in_high_go_down'] = self.data_df['cum_num_new_break_up'] - self.data_df['cum_num_new_break_up_for_high_go_down']
            self.data_df['num_new_break_up_in_low_go_up'] = self.data_df['cum_num_new_break_up'] - self.data_df['cum_num_new_break_up_for_low_go_up']
            self.data_df['num_new_break_up_in_low_go_down'] = self.data_df['cum_num_new_break_up'] - self.data_df['cum_num_new_break_up_for_low_go_down']


            self.data_df['num_new_break_down_in_high_go_up'] = self.data_df['cum_num_new_break_down'] - self.data_df['cum_num_new_break_down_for_high_go_up']
            self.data_df['num_new_break_down_in_high_go_down'] = self.data_df['cum_num_new_break_down'] - self.data_df['cum_num_new_break_down_for_high_go_down']
            self.data_df['num_new_break_down_in_low_go_up'] = self.data_df['cum_num_new_break_down'] - self.data_df['cum_num_new_break_down_for_low_go_up']
            self.data_df['num_new_break_down_in_low_go_down'] = self.data_df['cum_num_new_break_down'] - self.data_df['cum_num_new_break_down_for_low_go_down']


            self.data_df['num_high_go_up_in_high_go_up'] = self.data_df['cum_num_high_go_up'] - self.data_df['cum_num_high_go_up_for_high_go_up']
            self.data_df['num_high_go_up_in_high_go_down'] = self.data_df['cum_num_high_go_up'] - self.data_df['cum_num_high_go_up_for_high_go_down']
            self.data_df['num_high_go_up_in_low_go_up'] = self.data_df['cum_num_high_go_up'] - self.data_df['cum_num_high_go_up_for_low_go_up']
            self.data_df['num_high_go_up_in_low_go_down'] = self.data_df['cum_num_high_go_up'] - self.data_df['cum_num_high_go_up_for_low_go_down']

            self.data_df['num_high_go_down_in_high_go_up'] = self.data_df['cum_num_high_go_down'] - self.data_df['cum_num_high_go_down_for_high_go_up']
            self.data_df['num_high_go_down_in_high_go_down'] = self.data_df['cum_num_high_go_down'] - self.data_df['cum_num_high_go_down_for_high_go_down']
            self.data_df['num_high_go_down_in_low_go_up'] = self.data_df['cum_num_high_go_down'] - self.data_df['cum_num_high_go_down_for_low_go_up']
            self.data_df['num_high_go_down_in_low_go_down'] = self.data_df['cum_num_high_go_down'] - self.data_df['cum_num_high_go_down_for_low_go_down']

            self.data_df['num_low_go_up_in_high_go_up'] = self.data_df['cum_num_low_go_up'] - self.data_df['cum_num_low_go_up_for_high_go_up']
            self.data_df['num_low_go_up_in_high_go_down'] = self.data_df['cum_num_low_go_up'] - self.data_df['cum_num_low_go_up_for_high_go_down']
            self.data_df['num_low_go_up_in_low_go_up'] = self.data_df['cum_num_low_go_up'] - self.data_df['cum_num_low_go_up_for_low_go_up']
            self.data_df['num_low_go_up_in_low_go_down'] = self.data_df['cum_num_low_go_up'] - self.data_df['cum_num_low_go_up_for_low_go_down']

            self.data_df['num_low_go_down_in_high_go_up'] = self.data_df['cum_num_low_go_down'] - self.data_df['cum_num_low_go_down_for_high_go_up']
            self.data_df['num_low_go_down_in_high_go_down'] = self.data_df['cum_num_low_go_down'] - self.data_df['cum_num_low_go_down_for_high_go_down']
            self.data_df['num_low_go_down_in_low_go_up'] = self.data_df['cum_num_low_go_down'] - self.data_df['cum_num_low_go_down_for_low_go_up']
            self.data_df['num_low_go_down_in_low_go_down'] = self.data_df['cum_num_low_go_down'] - self.data_df['cum_num_low_go_down_for_low_go_down']


            self.data_df['is_bull_dying'] = \
                (self.data_df['period_high' + str(high_low_window) + '_go_up_duration'] > period_lookback) & \
                (self.data_df['num_low_go_up_in_high_go_up'] >= minimum_opposite_side_trend_num) & \
                (self.data_df['num_new_break_up_in_high_go_up'] >= minimum_break_bolling_num) #& \
                #(self.data_df['num_low_go_down_in_high_go_up'] == 0)



            self.data_df['is_bear_dying'] = \
                (self.data_df['period_low' + str(high_low_window) + '_go_down_duration'] > period_lookback) & \
                (self.data_df['num_high_go_down_in_low_go_down'] >= minimum_opposite_side_trend_num) & \
                (self.data_df['num_new_break_down_in_low_go_down'] >= minimum_break_bolling_num) #& \
                #(self.data_df['num_high_go_up_in_low_go_down'] == 0)






            # self.data_df['is_bear_dying'] = \
            #     (self.data_df['period_low' + str(high_low_window) + '_go_down_num'] == 0) & (self.data_df['period_high' + str(high_low_window) + '_go_down_num'] >= 1) \
            #     & (self.data_df['period_break_down_lower_band_num'] >= 2)
            #
            # self.data_df['is_bull_dying'] = \
            #     (self.data_df['period_high' + str(high_low_window) + '_go_up_num'] == 0) & (self.data_df['period_low' + str(high_low_window) + '_go_up_num'] >= 1) \
            #     & (self.data_df['period_break_up_upper_band_num'] >= 2)






            ####################################################################################################################################

            # print("data_df:")
            # print(self.data_df.tail(10))





            self.data_df['middle_vegas'] = (self.data_df['upper_vegas'] + self.data_df['lower_vegas']) / 2



            self.data_df['price_to_upper_vegas_pct'] = (self.data_df['close'] - self.data_df['upper_vegas']) / self.data_df['period_high_low_range']
            self.data_df['price_to_lower_vegas_pct'] = (self.data_df['lower_vegas'] - self.data_df['close']) / self.data_df['period_high_low_range']


            self.data_df['fast_vegas'] = self.data_df['ma_close144']
            self.data_df['slow_vegas'] = self.data_df['ma_close169']

            self.data_df['fast_vegas_gradient'] = self.data_df['fast_vegas'].diff()
            self.data_df['slow_vegas_gradient'] = self.data_df['slow_vegas'].diff()

            self.data_df['ma_close144_gradient'] = self.data_df['ma_close144'].diff()
            self.data_df['ma_close169_gradient'] = self.data_df['ma_close169'].diff()

            self.data_df['prev_fast_vegas_gradient'] = self.data_df['fast_vegas_gradient'].shift(1)
            self.data_df['prev_slow_vegas_gradient'] = self.data_df['slow_vegas_gradient'].shift(1)

            self.data_df['fast_vegas_go_up'] = np.where(
                self.data_df['fast_vegas_gradient'] > 0,
                1,
                0
            )

            self.data_df['slow_vegas_go_up'] = np.where(
                self.data_df['slow_vegas_gradient'] > 0,
                1,
                0
            )

            self.data_df['fast_vegas_go_up_num'] = self.data_df['fast_vegas_go_up'].rolling(vegas_look_back, min_periods = vegas_look_back).sum()
            self.data_df['slow_vegas_go_up_num'] = self.data_df['slow_vegas_go_up'].rolling(vegas_look_back, min_periods = vegas_look_back).sum()

            self.data_df['fast_vegas_go_up_pct'] = self.data_df['fast_vegas_go_up_num'] / vegas_look_back
            self.data_df['slow_vegas_go_up_pct'] = self.data_df['slow_vegas_go_up_num'] / vegas_look_back

            self.data_df['prev_fast_vegas_go_up_pct'] = self.data_df['fast_vegas_go_up_pct'].shift(1)
            self.data_df['prev_slow_vegas_go_up_pct'] = self.data_df['slow_vegas_go_up_pct'].shift(1)








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


            self.data_df['prev' + str(skip_bar_num + 1) + '_has_covered_lower'] = self.data_df['prev' + str(skip_bar_num + 1) + '_min_price'] <= self.data_df['middle']
            for i in range(skip_bar_num + 2, large_bar_look_back + 1):
                self.data_df['prev' + str(i) + '_has_covered_lower'] = self.data_df['prev' + str(i-1) + '_has_covered_lower'] | \
                                                                         (self.data_df['prev' + str(i) + '_min_price'] <= self.data_df['min_price']) #self.data_df['middle']

            self.data_df['prev'  + str(skip_bar_num + 1) +  '_has_covered_higher'] = self.data_df['prev' + str(skip_bar_num + 1) + '_max_price'] >= self.data_df['middle']
            for i in range(skip_bar_num + 2, large_bar_look_back + 1):
                self.data_df['prev' + str(i) + '_has_covered_higher'] = self.data_df['prev' + str(i-1) + '_has_covered_higher'] | \
                                                                         (self.data_df['prev' + str(i) + '_max_price'] >= self.data_df['max_price']) #self.data_df['middle']

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

            self.data_df['price_to_upper_vegas'] = self.data_df['upper_vegas'] - self.data_df['close']
            self.data_df['price_to_lower_vegas'] = self.data_df['close'] - self.data_df['lower_vegas']

            self.data_df['min_price_to_upper_vegas'] = self.data_df['min_price'] - self.data_df['upper_vegas']
            self.data_df['max_price_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['max_price']

            self.data_df['prev_min_price_to_upper_vegas'] = self.data_df['min_price_to_upper_vegas'].shift(1)
            self.data_df['prev_max_price_to_lower_vegas'] = self.data_df['max_price_to_lower_vegas'].shift(1)

            self.data_df['guppy_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['highest_guppy']
            self.data_df['guppy_to_upper_vegas'] = self.data_df['lowest_guppy'] - self.data_df['upper_vegas']

            self.data_df['guppy_to_lower_vegas_pct'] = self.data_df['guppy_to_lower_vegas'] / self.data_df['high_low_range']
            self.data_df['guppy_to_upper_vegas_pct'] = self.data_df['guppy_to_upper_vegas'] / self.data_df['high_low_range']



            self.data_df['price_to_bolling_upper'] = self.data_df['upper_band_close'] - self.data_df['close']
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
                #above_cond = self.data_df['is_above_vegas_strict'] | (self.data_df['is_above_vegas'] & ((self.data_df['upper_vegas_gradient'] > 0) | (self.data_df['lower_vegas_gradient'] > 0)))
                above_cond = self.data_df['is_above_vegas_strict'] | \
                             (self.data_df['is_above_vegas'] & (self.data_df['ma_close169_gradient'] > 0) \
                              & self.data_df['is_vegas_up_trend'] & ((~half_aligned_short_condition) | (self.data_df['close'] > self.data_df['lowest_guppy'])))

            self.data_df['above_cond'] = above_cond

            self.data_df['fast_vegas_mostly_up'] = self.data_df['prev_fast_vegas_go_up_pct'] >= vegas_trend_pct_threshold
            self.data_df['slow_vegas_mostly_up'] = self.data_df['prev_slow_vegas_go_up_pct'] >= vegas_trend_pct_threshold

            self.data_df['fast_vegas_mostly_down'] = self.data_df['prev_fast_vegas_go_up_pct'] <= 1 - vegas_trend_pct_threshold
            self.data_df['slow_vegas_mostly_down'] = self.data_df['prev_slow_vegas_go_up_pct'] <= 1 - vegas_trend_pct_threshold

            self.data_df['vegas_mostly_up'] = self.data_df['fast_vegas_mostly_up'] & self.data_df['slow_vegas_mostly_up']
            self.data_df['vegas_mostly_down'] = self.data_df['fast_vegas_mostly_down'] & self.data_df['slow_vegas_mostly_down']

            self.data_df['vegas_mostly_up_relaxed'] = self.data_df['fast_vegas_mostly_up'] | self.data_df['slow_vegas_mostly_up']
            self.data_df['vegas_mostly_down_relaxed'] = self.data_df['fast_vegas_mostly_down'] | self.data_df['slow_vegas_mostly_down']


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



            self.data_df['prev1_ma_close30'] = self.data_df['ma_close30'].shift(1)
            for i in range(2, guppy_lookback+1):
                self.data_df['prev' + str(i) + '_ma_close30'] = self.data_df['prev' + str(i-1) + '_ma_close30'].shift(1)

            self.data_df['prev1_highest_guppy'] = self.data_df['highest_guppy'].shift(1)
            for i in range(2, guppy_lookback+1):
                self.data_df['prev' + str(i) + '_highest_guppy'] = self.data_df['prev' + str(i-1) + '_highest_guppy'].shift(1)

            self.data_df['prev1_lowest_guppy'] = self.data_df['lowest_guppy'].shift(1)
            for i in range(2, guppy_lookback+1):
                self.data_df['prev' + str(i) + '_lowest_guppy'] = self.data_df['prev' + str(i-1) + '_lowest_guppy'].shift(1)

            self.data_df['highest_guppy_gradient'] = self.data_df['highest_guppy'].diff()
            self.data_df['lowest_guppy_gradient'] = self.data_df['lowest_guppy'].diff()

            self.data_df['prev1_highest_guppy_gradient'] = self.data_df['highest_guppy_gradient'].shift(1)
            for i in range(2, guppy_lookback+1):
                self.data_df['prev' + str(i) + '_highest_guppy_gradient'] = self.data_df['prev' + str(i-1) + '_highest_guppy_gradient'].shift(1)

            self.data_df['prev1_lowest_guppy_gradient'] = self.data_df['lowest_guppy_gradient'].shift(1)
            for i in range(2, guppy_lookback+1):
                self.data_df['prev' + str(i) + '_lowest_guppy_gradient'] = self.data_df['prev' + str(i-1) + '_lowest_guppy_gradient'].shift(1)


            recent_fast_guppy_at_btm_go_down = reduce(lambda left, right: left | right,
                                                      [((np.abs(self.data_df['prev' + str(i) + '_ma_close30'] - self.data_df['prev' + str(i) + '_lowest_guppy']) < 1e-5) & \
                                                       (self.data_df['prev' + str(i) + '_lowest_guppy_gradient'] < 0))
                                                       for i in range(1, guppy_lookback+1)] +
                                                      [(self.data_df['fastest_guppy_at_btm'] & self.data_df['lowest_guppy_gradient'] < 0)])

            recent_fast_guppy_at_top_go_up = reduce(lambda left, right: left | right,
                                                      [((np.abs(self.data_df['prev' + str(i) + '_ma_close30'] - self.data_df['prev' + str(i) + '_highest_guppy']) < 1e-5) & \
                                                       (self.data_df['prev' + str(i) + '_highest_guppy_gradient'] > 0))
                                                       for i in range(1, guppy_lookback+1)] +
                                                      [(self.data_df['fastest_guppy_at_top'] & self.data_df['highest_guppy_gradient'] > 0)])





            buy_c41 = self.data_df['high'] > self.data_df['upper_band_close'] #self.data_df['prev_upper_band_close']
            buy_c42 = self.data_df['upper_band_close_gradient'] * self.lot_size * self.exchange_rate > 0# bolling_threshold
            buy_c43 = self.data_df['is_positive'] & (self.data_df['prev1_open'] < self.data_df['prev1_close'])
            buy_c44 = (self.data_df['low_low_pct_price_buy'] > self.data_df['upper_vegas']) & (self.data_df['prev_low_low_pct_price_buy'] > self.data_df['prev1_upper_vegas'])
            buy_c4 = buy_c41 & buy_c42 & ((self.data_df['close'] > self.data_df['highest_guppy']) | (buy_c43 & buy_c44))

            sell_c41 = self.data_df['low'] < self.data_df['lower_band_close'] # self.data_df['prev_lower_band_close']
            sell_c42 = self.data_df['lower_band_close_gradient'] * self.lot_size * self.exchange_rate < 0 #-bolling_threshold
            sell_c43 = self.data_df['is_negative'] &  (self.data_df['prev1_open'] > self.data_df['prev1_close'])
            sell_c44 = (self.data_df['high_high_pct_price_sell'] < self.data_df['lower_vegas']) & (self.data_df['prev_high_high_pct_price_sell'] < self.data_df['prev1_lower_vegas'])
            sell_c4 = sell_c41 & sell_c42 & ((self.data_df['close'] < self.data_df['lowest_guppy']) | (sell_c43 & sell_c44))



            ################# Additional conditions for breaking bolling band to enter ###################
            breaking_up_cond1 = (~aligned_short_condition_go_on) | (self.data_df['min_price'] > self.data_df['highest_guppy'])
            breaking_up_cond2 = recent_tightly_supported_by_vegas | (~recent_fast_guppy_at_btm_go_down)
            breaking_up_cond3 = ~self.data_df['fast_vegas_mostly_down']

            breaking_down_cond1 = (~aligned_long_condition_go_on) | (self.data_df['max_price'] < self.data_df['lowest_guppy'])
            breaking_down_cond2 = recent_tightly_suppressed_by_vegas | (~recent_fast_guppy_at_top_go_up)
            breaking_down_cond3 = ~self.data_df['fast_vegas_mostly_up']

            ###############################################################################################





            #(self.data_df['fastest_guppy_at_top'] & self.data_df['ma_close30_gradient'] > 0)
            self.data_df['vegas_inferred_up_cond1'] = self.data_df['vegas_fast_above'] & (self.data_df['recent_vegas_cross_up_num'] == 1)
            self.data_df['vegas_inferred_up_cond2'] = (~self.data_df['vegas_fast_above']) & (
                ((self.data_df['fastest_guppy_at_top'] & self.data_df['ma_close30_gradient'] > 0)) | \
                ((self.data_df['fast_vegas_gradient']*self.lot_size*self.exchange_rate > vagas_fast_support_threshold) & (self.data_df['prev_fast_vegas_gradient']*self.lot_size*self.exchange_rate > vagas_fast_support_threshold))
            )
            self.data_df['vegas_inferred_up'] = self.data_df['vegas_inferred_up_cond1'] | self.data_df['vegas_inferred_up_cond2'] | buy_c4

            #(self.data_df['fastest_guppy_at_btm'] & self.data_df['ma_close30_gradient'] < 0)
            self.data_df['vegas_inferred_down_cond1'] = (~self.data_df['vegas_fast_above']) & (self.data_df['recent_vegas_cross_down_num'] == 1)
            self.data_df['vegas_inferred_down_cond2'] = self.data_df['vegas_fast_above'] & (
                ((self.data_df['fastest_guppy_at_btm'] & self.data_df['ma_close30_gradient'] < 0)) | \
                ((self.data_df['fast_vegas_gradient']*self.lot_size*self.exchange_rate < -vagas_fast_support_threshold) & (self.data_df['prev_fast_vegas_gradient']*self.lot_size*self.exchange_rate < -vagas_fast_support_threshold))
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
            buy_c13 = (self.data_df['price_to_bolling_upper'] / self.data_df['price_to_lower_vegas']) > minimum_profilt_loss_ratio

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

            buy_c7 = self.data_df['break_upper_bolling'] & \
                     ((self.data_df['price_to_period_high_pct'] < price_to_period_range_pct_strict) | \
                      ((self.data_df['price_to_period_high_pct'] < price_to_period_range_pct) & (~strongly_half_aligned_long_condition)))
                      #Buy price too high, should not enter

            buy_c8 = self.data_df['is_bull_dying'] & (self.data_df['price_to_period_high_pct'] < price_to_period_range_pct_relaxed) & (~strongly_aligned_long_condition)

            #buy_c9 = self.data_df['upper_vegas_mostly_down']

            buy_c9 = (self.data_df['fast_vegas_gradient'] * self.lot_size * self.exchange_rate > vegas_angle_threshold) | \
                     (self.data_df['slow_vegas_gradient'] * self.lot_size * self.exchange_rate > vegas_angle_threshold) | \
                     self.data_df['vegas_mostly_up']

            buy_c10 = ~self.data_df['vegas_mostly_down_relaxed']

            if not self.remove_c12:
                self.data_df['buy_real_fire'] = (self.data_df['buy_real_fire']) & \
                                                (buy_c3) & (~buy_c5) & (~buy_c6) & (~buy_c7) & (~buy_c8) & (buy_c9) & (buy_c10) & ((buy_c4 & breaking_up_cond1 & breaking_up_cond2 & breaking_up_cond3) | (((buy_c12) | (buy_c13)) & (~buy_c2)))
            else:
                self.data_df['buy_real_fire'] = (self.data_df['buy_real_fire']) & \
                                                (buy_c3) & (~buy_c5) & (~buy_c6) & (~buy_c7) & (~buy_c8) & (buy_c9) & (buy_c10) & ((buy_c4 & breaking_up_cond1 & breaking_up_cond2 & breaking_up_cond3) | ((buy_c13) & (~buy_c2)))


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
            self.data_df['buy_c8'] = buy_c8
            self.data_df['buy_c9'] = buy_c9
            self.data_df['buy_c10'] = buy_c10
            self.data_df['breaking_up_cond1'] = breaking_up_cond1
            self.data_df['breaking_up_cond2'] = breaking_up_cond2
            self.data_df['breaking_up_cond3'] = breaking_up_cond3

            self.data_df['buy_real_fire5'] = (self.data_df['close'] > self.data_df['open']) & (self.data_df['price_range'] >= 2 * self.data_df['prev_recent_max_price_range'])


            #sell_good_cond1 = (self.data_df['prev_max_price_to_period_high_pct'] < reverse_threshold) | (self.data_df['prev2_max_price_to_period_high_pct'] < reverse_threshold)

            sell_good_cond1 = reduce(lambda left, right: left | right,
                                     [(self.data_df['prev' + str(i) + '_high_price_to_period_high_pct'] < reverse_threshold)
                                     for i in range(1, reverse_trade_look_back + 1)])

            sell_good_cond2 = (self.data_df['close'] - self.data_df['open'] < 0) & \
                             (self.data_df['close'] < self.data_df['ma_close12']) #& \
                             #(self.data_df['close'] > self.data_df['second_lowest_guppy'])
                             #(self.data_df['prev_high'] > self.data_df['ma_close12'])
            sell_good_cond3 = (self.data_df['ma12_gradient'] * self.lot_size * self.exchange_rate < 0)
            sell_good_cond4 = (self.data_df['close'] > self.data_df['upper_vegas'])
            #sell_good_cond5 = (self.data_df['price_to_lower_vegas']/self.data_df['price_to_period_high']) >= 0.7

            sell_good_cond5 = np.where(
                self.data_df['close'] < self.data_df['lowest_guppy'],
                (-self.data_df['price_to_upper_vegas'] / self.data_df['price_to_period_high']) >= entry_risk_threshold,
                (self.data_df['price_to_lower_vegas'] / self.data_df['price_to_period_high']) >= entry_risk_threshold
            )

            sell_bad_cond0 = (self.data_df['close'] - self.data_df['upper_vegas']) * self.lot_size * self.exchange_rate < reverse_trade_min_points_to_vegas
            sell_bad_cond1 = self.data_df['price_to_upper_vegas_pct'] < reverse_trade_min_distance_to_vegas
            sell_bad_cond2 = (self.data_df['fast_vegas_gradient']*self.lot_size*self.exchange_rate > -1) | \
                             (self.data_df['slow_vegas_gradient']*self.lot_size*self.exchange_rate > -1)

            #sell_bad_cond3 = strongly_relaxed_aligned_long_condition & ((self.data_df['highest_guppy'] - self.data_df['upper_vegas'])*self.lot_size*self.exchange_rate > 0)

            sell_bad_cond3 = strongly_half_aligned_long_condition & (self.data_df['highest_guppy'] > self.data_df['middle_vegas']) & (self.data_df['close'] > self.data_df['highest_guppy'])

            #sell_bad_cond4 = self.data_df['close'] <= self.data_df['lowest_guppy']
            sell_bad_cond4 = (~strongly_aligned_short_condition) & ((self.data_df['middle'] < self.data_df['lowest_guppy']) | \
                                                                  ((self.data_df['close'] < self.data_df['lowest_guppy']) & (strongly_relaxed_aligned_long_condition)))


            sell_bad_cond = (sell_bad_cond0 & sell_bad_cond1 & (sell_bad_cond2 | sell_bad_cond3)) | sell_bad_cond4


            self.data_df['sell_real_fire2'] = self.data_df['is_bull_dying'] & (self.data_df['prev_price_to_period_high_pct'] < reverse_threshold) & \
                                              (self.data_df['close'] - self.data_df['open'] < 0) & (self.data_df['close'] < self.data_df['prev1_min_price'])
            self.data_df['sell_real_fire2'] = self.data_df['sell_real_fire2'] & (~(sell_bad_cond0 & sell_bad_cond1))

            #self.data_df['sell_real_fire2'] = self.data_df['sell_real_fire2'] & ((self.data_df['macd'] < self.data_df['msignal']) | (self.data_df['macd_gradient'] < 0))


            self.data_df['sell_real_fire3'] = sell_good_cond1 & sell_good_cond2 & sell_good_cond3 & sell_good_cond4 & sell_good_cond5 & (~sell_bad_cond)

            #self.data_df['sell_real_fire3'] = self.data_df['sell_real_fire3'] & ((self.data_df['macd'] < self.data_df['msignal']) | (self.data_df['macd_gradient'] < 0))


            self.data_df['sell_good_cond1'] = sell_good_cond1
            self.data_df['sell_good_cond2'] = sell_good_cond2
            self.data_df['sell_good_cond3'] = sell_good_cond3
            self.data_df['sell_good_cond4'] = sell_good_cond4
            self.data_df['sell_good_cond5'] = sell_good_cond5

            self.data_df['sell_bad_cond0'] = sell_bad_cond0
            self.data_df['sell_bad_cond1'] = sell_bad_cond1
            self.data_df['sell_bad_cond2'] = sell_bad_cond2
            self.data_df['sell_bad_cond3'] = sell_bad_cond3
            self.data_df['sell_bad_cond4'] = sell_bad_cond4

            self.data_df['sell_real_fire4'] = (self.data_df['ma_close12'] > self.data_df['upper_vegas']) & (self.data_df['macd_cross_down']) & \
                                               (self.data_df['close'] > self.data_df['upper_vegas'])
            macd_sell_cond1 = (self.data_df['price_to_lower_vegas']/self.data_df['price_to_period_high'] >= 0.7)
            #macd_sell_cond2 = (self.data_df['ma12_gradient'] * self.lot_size * self.exchange_rate < 0)
            macd_sell_cond2 = ((self.data_df['is_vegas_down_trend']) & (self.data_df['ma_close144_gradient'] <= 0))

            macd_sell_condition = (macd_sell_cond1 | macd_sell_cond2) if macd_relaxed else macd_sell_cond1

            self.data_df['sell_real_fire4'] = self.data_df['sell_real_fire4'] & macd_sell_condition
            self.data_df['macd_sell_cond1'] = macd_sell_cond1
            self.data_df['macd_sell_cond2'] = macd_sell_cond2



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

            self.data_df['prev_buy_real_fire5'] = self.data_df['buy_real_fire5'].shift(1)
            self.data_df.at[0, 'prev_buy_real_fire5'] = False
            self.data_df['prev_buy_real_fire5'] = pd.Series(list(self.data_df['prev_buy_real_fire5']), dtype='bool')
            self.data_df['first_buy_real_fire5'] = self.data_df['buy_real_fire5'] & (~self.data_df['prev_buy_real_fire5'])

            self.data_df['prev_sell_real_fire2'] = self.data_df['sell_real_fire2'].shift(1)
            self.data_df.at[0, 'prev_sell_real_fire2'] = False
            self.data_df['prev_sell_real_fire2'] = pd.Series(list(self.data_df['prev_sell_real_fire2']), dtype='bool')
            self.data_df['first_sell_real_fire2'] = self.data_df['sell_real_fire2'] & (~self.data_df['prev_sell_real_fire2'])

            self.data_df['prev_sell_real_fire3'] = self.data_df['sell_real_fire3'].shift(1)
            self.data_df.at[0, 'prev_sell_real_fire3'] = False
            self.data_df['prev_sell_real_fire3'] = pd.Series(list(self.data_df['prev_sell_real_fire3']), dtype='bool')
            self.data_df['first_sell_real_fire3'] = self.data_df['sell_real_fire3'] & (~self.data_df['prev_sell_real_fire3'])

            self.data_df['prev_sell_real_fire4'] = self.data_df['sell_real_fire4'].shift(1)
            self.data_df.at[0, 'prev_sell_real_fire4'] = False
            self.data_df['prev_sell_real_fire4'] = pd.Series(list(self.data_df['prev_sell_real_fire4']), dtype='bool')
            self.data_df['first_sell_real_fire4'] = self.data_df['sell_real_fire4'] & (~self.data_df['prev_sell_real_fire4'])


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
                #below_cond = self.data_df['is_below_vegas_strict'] | (self.data_df['is_below_vegas'] & ((self.data_df['upper_vegas_gradient'] < 0) | (self.data_df['lower_vegas_gradient'] < 0)))
                #below_cond = self.data_df['is_below_vegas_strict'] | (self.data_df['is_below_vegas'] & (self.data_df['ma_close169_gradient'] < 0) & self.data_df['is_vegas_down_trend'])
                below_cond = self.data_df['is_below_vegas_strict'] | \
                             (self.data_df['is_below_vegas'] & (self.data_df['ma_close169_gradient'] < 0) \
                              & self.data_df['is_vegas_down_trend'] & ((~half_aligned_long_condition) | (self.data_df['close'] < self.data_df['highest_guppy'])))

            self.data_df['below_cond'] = below_cond

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
            sell_c13 = (self.data_df['price_to_bolling_lower'] / self.data_df['price_to_upper_vegas']) > minimum_profilt_loss_ratio

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

            sell_c7 = self.data_df['break_lower_bolling'] & \
                     ((self.data_df['price_to_period_low_pct'] < price_to_period_range_pct_strict) | \
                      ((self.data_df['price_to_period_low_pct'] < price_to_period_range_pct) & (~strongly_half_aligned_short_condition)))

            sell_c8 = self.data_df['is_bear_dying'] & (self.data_df['price_to_period_low_pct'] < price_to_period_range_pct_relaxed) & (~strongly_aligned_short_condition)

            #sell_c9 = self.data_df['upper_vegas_mostly_up']

            sell_c9 = (self.data_df['fast_vegas_gradient'] * self.lot_size * self.exchange_rate < -vegas_angle_threshold) | \
                      (self.data_df['slow_vegas_gradient'] * self.lot_size * self.exchange_rate < -vegas_angle_threshold) | \
                      self.data_df['vegas_mostly_down']

            sell_c10 = ~self.data_df['vegas_mostly_up_relaxed']


            if not self.remove_c12:
                self.data_df['sell_real_fire'] = (self.data_df['sell_real_fire']) & \
                                                 (sell_c3) & (~sell_c5) & (~sell_c6) & (~sell_c7) & (~sell_c8) & (sell_c9) & (sell_c10) & ((sell_c4 & breaking_down_cond1 & breaking_down_cond2 & breaking_down_cond3) | (((sell_c12) | (sell_c13)) & (~sell_c2)))
            else:
                self.data_df['sell_real_fire'] = (self.data_df['sell_real_fire']) & \
                                                 (sell_c3) & (~sell_c5) & (~sell_c6) & (~sell_c7) & (~sell_c8) & (sell_c9) & (sell_c10) & ((sell_c4 & breaking_down_cond1 & breaking_down_cond2 & breaking_down_cond3) | ((sell_c13) & (~sell_c2)))



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
            self.data_df['sell_c7'] = sell_c7
            self.data_df['sell_c8'] = sell_c8
            self.data_df['sell_c9'] = sell_c9
            self.data_df['sell_c10'] = sell_c10
            self.data_df['breaking_down_cond1'] = breaking_down_cond1
            self.data_df['breaking_down_cond2'] = breaking_down_cond2
            self.data_df['breaking_down_cond3'] = breaking_down_cond3

            self.data_df['sell_real_fire5'] = (self.data_df['close'] < self.data_df['open']) & (self.data_df['price_range'] >= 2 * self.data_df['prev_recent_max_price_range'])




            #buy_good_cond1 = (self.data_df['prev_min_price_to_period_low_pct'] < reverse_threshold) | (self.data_df['prev2_min_price_to_period_low_pct'] < reverse_threshold)

            buy_good_cond1 = reduce(lambda left, right: left | right,
                                     [(self.data_df['prev' + str(i) + '_low_price_to_period_low_pct'] < reverse_threshold)
                                     for i in range(1, reverse_trade_look_back + 1)])

            buy_good_cond2 = (self.data_df['close'] - self.data_df['open'] > 0) & \
                             (self.data_df['close'] > self.data_df['ma_close12']) #& \
                             #(self.data_df['close'] < self.data_df['second_highest_guppy'])
                             #(self.data_df['prev_low'] < self.data_df['ma_close12'])
            buy_good_cond3 = (self.data_df['ma12_gradient'] * self.lot_size * self.exchange_rate > 0)
            buy_good_cond4 = (self.data_df['close'] < self.data_df['lower_vegas'])
            #buy_good_cond5 = (self.data_df['price_to_upper_vegas'] / self.data_df['price_to_period_low']) >= 0.7

            buy_good_cond5 = np.where(
                self.data_df['close'] > self.data_df['highest_guppy'],
                (-self.data_df['price_to_lower_vegas'] / self.data_df['price_to_period_low']) >= entry_risk_threshold,
                (self.data_df['price_to_upper_vegas'] / self.data_df['price_to_period_low']) >= entry_risk_threshold
            )

            buy_bad_cond0 = (self.data_df['lower_vegas'] - self.data_df['close']) * self.lot_size * self.exchange_rate < reverse_trade_min_points_to_vegas
            buy_bad_cond1 = self.data_df['price_to_lower_vegas_pct'] < reverse_trade_min_distance_to_vegas
            buy_bad_cond2 = (self.data_df['fast_vegas_gradient']*self.lot_size*self.exchange_rate < 1) | \
                            (self.data_df['slow_vegas_gradient']*self.lot_size*self.exchange_rate < 1)
            #buy_bad_cond3 = strongly_relaxed_aligned_short_condition & ((self.data_df['lowest_guppy'] - self.data_df['lower_vegas'])*self.lot_size*self.exchange_rate < 0)

            buy_bad_cond3 = strongly_half_aligned_short_condition & (self.data_df['lowest_guppy'] < self.data_df['middle_vegas']) & (self.data_df['close'] < self.data_df['lowest_guppy'])

            #buy_bad_cond4 = self.data_df['close'] >= self.data_df['highest_guppy']

            buy_bad_cond4 = (~strongly_aligned_long_condition) & ((self.data_df['middle'] > self.data_df['highest_guppy']) | \
                                                                  ((self.data_df['close'] > self.data_df['highest_guppy']) & (strongly_relaxed_aligned_short_condition)))

            buy_bad_cond = (buy_bad_cond0 & buy_bad_cond1 & (buy_bad_cond2 | buy_bad_cond3)) | buy_bad_cond4


            self.data_df['buy_real_fire2'] = self.data_df['is_bear_dying'] & (self.data_df['prev_price_to_period_low_pct'] < reverse_threshold) & \
                                              (self.data_df['close'] - self.data_df['open'] > 0) & (self.data_df['close'] > self.data_df['prev1_max_price'])
            self.data_df['buy_real_fire2'] = self.data_df['buy_real_fire2'] & (~(buy_bad_cond0 & buy_bad_cond1))

            #self.data_df['buy_real_fire2'] = self.data_df['buy_real_fire2'] & ((self.data_df['macd'] > self.data_df['msignal']) | (self.data_df['macd_gradient'] > 0)) #added

            self.data_df['buy_real_fire3'] = buy_good_cond1 & buy_good_cond2 & buy_good_cond3 & buy_good_cond4 & buy_good_cond5 & (~buy_bad_cond)

            #self.data_df['buy_real_fire3'] = self.data_df['buy_real_fire3'] & ((self.data_df['macd'] > self.data_df['msignal']) | (self.data_df['macd_gradient'] > 0)) #added


            self.data_df['buy_good_cond1'] = buy_good_cond1
            self.data_df['buy_good_cond2'] = buy_good_cond2
            self.data_df['buy_good_cond3'] = buy_good_cond3
            self.data_df['buy_good_cond4'] = buy_good_cond4
            self.data_df['buy_good_cond5'] = buy_good_cond5

            self.data_df['buy_bad_cond0'] = buy_bad_cond0
            self.data_df['buy_bad_cond1'] = buy_bad_cond1
            self.data_df['buy_bad_cond2'] = buy_bad_cond2
            self.data_df['buy_bad_cond3'] = buy_bad_cond3
            self.data_df['buy_bad_cond4'] = buy_bad_cond4


            self.data_df['buy_real_fire4'] = (self.data_df['ma_close12'] < self.data_df['lower_vegas']) & (self.data_df['macd_cross_up']) & \
                                             (self.data_df['close'] < self.data_df['lower_vegas'])
            macd_buy_cond1 = (self.data_df['price_to_upper_vegas'] / self.data_df['price_to_period_low'] >= 0.7)
            #macd_buy_cond2 = (self.data_df['ma12_gradient'] * self.lot_size * self.exchange_rate > 0)
            macd_buy_cond2 = ((self.data_df['is_vegas_up_trend']) & (self.data_df['ma_close144_gradient'] >= 0))

            macd_buy_condition = (macd_buy_cond1 | macd_buy_cond2) if macd_relaxed else macd_buy_cond1
            self.data_df['buy_real_fire4'] = self.data_df['buy_real_fire4'] & macd_buy_condition
            self.data_df['macd_buy_cond1'] = macd_buy_cond1
            self.data_df['macd_buy_cond2'] = macd_buy_cond2




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

            self.data_df['prev_sell_real_fire5'] = self.data_df['sell_real_fire5'].shift(1)
            self.data_df.at[0, 'prev_sell_real_fire5'] = False
            self.data_df['prev_sell_real_fire5'] = pd.Series(list(self.data_df['prev_sell_real_fire5']), dtype='bool')
            self.data_df['first_sell_real_fire5'] = self.data_df['sell_real_fire5'] & (~self.data_df['prev_sell_real_fire5'])

            self.data_df['prev_buy_real_fire2'] = self.data_df['buy_real_fire2'].shift(1)
            self.data_df.at[0, 'prev_buy_real_fire2'] = False
            self.data_df['prev_buy_real_fire2'] = pd.Series(list(self.data_df['prev_buy_real_fire2']), dtype='bool')
            self.data_df['first_buy_real_fire2'] = self.data_df['buy_real_fire2'] & (~self.data_df['prev_buy_real_fire2'])

            self.data_df['prev_buy_real_fire3'] = self.data_df['buy_real_fire3'].shift(1)
            self.data_df.at[0, 'prev_buy_real_fire3'] = False
            self.data_df['prev_buy_real_fire3'] = pd.Series(list(self.data_df['prev_buy_real_fire3']), dtype='bool')
            self.data_df['first_buy_real_fire3'] = self.data_df['buy_real_fire3'] & (~self.data_df['prev_buy_real_fire3'])

            self.data_df['prev_buy_real_fire4'] = self.data_df['buy_real_fire4'].shift(1)
            self.data_df.at[0, 'prev_buy_real_fire4'] = False
            self.data_df['prev_buy_real_fire4'] = pd.Series(list(self.data_df['prev_buy_real_fire4']), dtype='bool')
            self.data_df['first_buy_real_fire4'] = self.data_df['buy_real_fire4'] & (~self.data_df['prev_buy_real_fire4'])


            #Cruise
            sell_close1_cond1 = self.data_df['fast_vegas'] > self.data_df['slow_vegas']
            sell_close1_cond2 = (self.data_df['close'] > self.data_df['highest_guppy']) & (self.data_df['open'] <= self.data_df['highest_guppy']) & \
                                (self.data_df['guppy_to_lower_vegas_pct'] > 0.15)
            sell_close1_cond3 = (self.data_df['close'] > self.data_df['upper_vegas']) & (self.data_df['open'] <= self.data_df['upper_vegas'])
            self.data_df['sell_close_position1'] = sell_close1_cond1 & (sell_close1_cond2 | sell_close1_cond3)


            sell_close2_cond1 = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) #\
                                # & (self.data_df['fast_vegas_gradient'] < 0) & \
                                # (self.data_df['slow_vegas_gradient'] < 0)
            sell_close2_cond2 = (self.data_df['period_high' + str(high_low_window) + '_vegas_gradient'] <= 0) & \
                                (self.data_df['period_low' + str(high_low_window) + '_vegas_gradient'] <= 0) & \
                                (self.data_df['period_high_low_vegas_gradient_ratio'] >= 1.0)


            #sell_close2_cond3 = (self.data_df['prev_price_to_period_low_pct'] < 0.1) #Change to previous 10 such value at least one < 0.1

            sell_close2_cond3 = reduce(lambda left, right: left | right,
                                       [(self.data_df['prev' + str(i) + '_min_price_to_period_low_pct'] < 0.1)
                                        for i in range(1, close_position_look_back)]
                                       )


            sell_close2_cond4 = (self.data_df['period_low' + str(high_low_window) + '_go_down_duration'] >= close_position_look_back)
            sell_close2_cond5 = (self.data_df['is_positive']) #& (self.data_df['price_range'] / self.data_df['price_volatility'] >= 0.5)

            sell_close2_cond6 = (self.data_df['price_to_lower_vegas_pct'] > 0.05) | \
                                ((self.data_df['ma_close12'] > self.data_df['lower_vegas']) & (self.data_df['ma12_gradient'] > 0) )

            #self.data_df['sell_close_position2'] = sell_close2_cond1 & ((sell_close2_cond2 & sell_close2_cond3 & sell_close2_cond4) | sell_close1_cond3)

            #self.data_df['sell_close_position2'] = sell_close2_cond1 & ((sell_close2_cond3 & sell_close2_cond4 & sell_close2_cond5))

            self.data_df['sell_close_position2_excessive'] = ((sell_close2_cond3 & sell_close2_cond4 & sell_close2_cond5 & sell_close2_cond6))
            self.data_df['sell_close_position_excessive'] =(self.data_df['ma_close12'] < self.data_df['upper_vegas']) & (self.data_df['sell_close_position2_excessive'])

            self.data_df['sell_close_position2_conservative'] = ((sell_close2_cond2 & sell_close2_cond3 & sell_close2_cond4 & sell_close2_cond5 & sell_close2_cond6))
            self.data_df['sell_close_position_conservative'] =(self.data_df['ma_close12'] < self.data_df['upper_vegas']) & (self.data_df['sell_close_position2_conservative'])

            self.data_df['sell_stop_loss_excessive'] = (self.data_df['cross_up_lower_vegas'] == 1) & \
                                                       (~((self.data_df['fast_vegas_gradient'] < 0) & (self.data_df['slow_vegas_gradient'] < 0))) & \
                                                       (self.data_df['prev_on_one_side_vegas_duration'] > 24)
            self.data_df['sell_stop_loss_conservative'] = (self.data_df['cross_vegas'] == 1) & (self.data_df['prev_on_one_side_vegas_duration'] > 24)


            self.data_df['sell_close1_cond1'] = sell_close1_cond1
            self.data_df['sell_close1_cond2'] = sell_close1_cond2
            self.data_df['sell_close1_cond3'] = sell_close1_cond3
            self.data_df['sell_close2_cond1'] = sell_close2_cond1
            self.data_df['sell_close2_cond2'] = sell_close2_cond2
            self.data_df['sell_close2_cond3'] = sell_close2_cond3
            self.data_df['sell_close2_cond4'] = sell_close2_cond4
            self.data_df['sell_close2_cond5'] = sell_close2_cond5



            buy_close1_cond1 = self.data_df['fast_vegas'] < self.data_df['slow_vegas']
            buy_close1_cond2 = (self.data_df['close'] < self.data_df['lowest_guppy']) & (self.data_df['open'] >= self.data_df['lowest_guppy']) & \
                                (self.data_df['guppy_to_upper_vegas_pct'] > 0.15)
            buy_close1_cond3 = (self.data_df['close'] < self.data_df['lower_vegas']) & (self.data_df['open'] >= self.data_df['lower_vegas'])
            self.data_df['buy_close_position1'] = buy_close1_cond1 & (buy_close1_cond2 | buy_close1_cond3)


            buy_close2_cond1 = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) #\
                               # & (self.data_df['fast_vegas_gradient'] > 0) & \
                               #  (self.data_df['slow_vegas_gradient'] > 0)
            buy_close2_cond2 = (self.data_df['period_high' + str(high_low_window) + '_vegas_gradient'] >= 0) & \
                                (self.data_df['period_low' + str(high_low_window) + '_vegas_gradient'] >= 0) & \
                                (self.data_df['period_low_high_vegas_gradient_ratio'] >= 1.0)

            #buy_close2_cond3 = (self.data_df['prev_price_to_period_high_pct'] < 0.1)

            buy_close2_cond3 = reduce(lambda left, right: left | right,
                                       [(self.data_df['prev' + str(i) + '_max_price_to_period_high_pct'] < 0.1)
                                        for i in range(1, close_position_look_back)]
                                       )

            buy_close2_cond4 = (self.data_df['period_high' + str(high_low_window) + '_go_up_duration'] >= close_position_look_back)
            buy_close2_cond5 = (self.data_df['is_negative']) #& (self.data_df['price_range'] / self.data_df['price_volatility'] >= 0.5)

            buy_close2_cond6 = (self.data_df['price_to_upper_vegas_pct'] > 0.05) | \
                               ((self.data_df['ma_close12'] < self.data_df['upper_vegas']) & (self.data_df['ma12_gradient'] < 0))

            #self.data_df['buy_close_position2'] = buy_close2_cond1 & ((buy_close2_cond2 & buy_close2_cond3 & buy_close2_cond4) | buy_close1_cond3)

            self.data_df['buy_close_position2_excessive'] = ((buy_close2_cond3 & buy_close2_cond4 & buy_close2_cond5 & buy_close2_cond6))
            self.data_df['buy_close_position_excessive'] = (self.data_df['ma_close12'] > self.data_df['lower_vegas']) & (self.data_df['buy_close_position2_excessive'])

            self.data_df['buy_close_position2_conservative'] = ((buy_close2_cond2 & buy_close2_cond3 & buy_close2_cond4 & buy_close2_cond5 & buy_close2_cond6))
            self.data_df['buy_close_position_conservative'] = (self.data_df['ma_close12'] > self.data_df['lower_vegas']) & (self.data_df['buy_close_position2_conservative'])

            self.data_df['buy_stop_loss_excessive'] = (self.data_df['cross_down_upper_vegas'] == -1) & \
                                                       (~((self.data_df['fast_vegas_gradient'] > 0) & (self.data_df['slow_vegas_gradient'] > 0))) & \
                                                       (self.data_df['prev_on_one_side_vegas_duration'] > 24)
            self.data_df['buy_stop_loss_conservative'] = (self.data_df['cross_vegas'] == -1) & (self.data_df['prev_on_one_side_vegas_duration'] > 24)

            self.data_df['buy_close1_cond1'] = buy_close1_cond1
            self.data_df['buy_close1_cond2'] = buy_close1_cond2
            self.data_df['buy_close1_cond3'] = buy_close1_cond3
            self.data_df['buy_close2_cond1'] = buy_close2_cond1
            self.data_df['buy_close2_cond2'] = buy_close2_cond2
            self.data_df['buy_close2_cond3'] = buy_close2_cond3
            self.data_df['buy_close2_cond4'] = buy_close2_cond4
            self.data_df['buy_close2_cond5'] = buy_close2_cond5


            ################ Hutong Gay
            self.data_df['buy_point'] = np.where(
                self.data_df['first_buy_real_fire3'] | self.data_df['first_buy_real_fire2'],
                1,
                0
            )


            self.data_df['sell_point'] = np.where(
                self.data_df['first_sell_real_fire3'] | self.data_df['first_sell_real_fire2'],
                1,
                0
            )

            self.data_df['large_positive_bar'] = self.data_df['is_positive'] & (self.data_df['price_range'] * self.exchange_rate * self.lot_size > 80)
            self.data_df['large_negative_bar'] = self.data_df['is_negative'] & (self.data_df['price_range'] * self.exchange_rate * self.lot_size > 80)


            self.data_df['cum_large_positive'] = self.data_df['large_positive_bar'].cumsum()
            self.data_df['cum_positive'] = self.data_df['is_positive'].cumsum()

            self.data_df['cum_large_negative'] = self.data_df['large_negative_bar'].cumsum()
            self.data_df['cum_negative'] = self.data_df['is_negative'].cumsum()

            self.data_df['cum_large_positive'] = self.data_df['cum_large_positive'].shift(1)
            self.data_df.at[0, 'cum_large_positive'] = 0

            self.data_df['cum_positive'] = self.data_df['cum_positive'].shift(1)
            self.data_df.at[0, 'cum_positive'] = 0

            self.data_df['cum_large_negative'] = self.data_df['cum_large_negative'].shift(1)
            self.data_df.at[0, 'cum_large_negative'] = 0

            self.data_df['cum_negative'] = self.data_df['cum_negative'].shift(1)
            self.data_df.at[0, 'cum_negative'] = 0

            for col in ['cum_large_positive', 'cum_large_negative', 'cum_positive', 'cum_negative']:
                self.data_df[col] = self.data_df[col].astype(int)

            self.data_df['buy_point_temp'] = np.nan
            self.data_df['buy_point_temp'] = np.where(
                self.data_df['buy_point'] == 1,
                self.data_df['id'],
                self.data_df['buy_point_temp']
            )

            self.data_df['sell_point_temp'] = np.nan
            self.data_df['sell_point_temp'] = np.where(
                self.data_df['sell_point'] == 1,
                self.data_df['id'],
                self.data_df['sell_point_temp']
            )


            df_buy_point = self.data_df[self.data_df['buy_point_temp'].notnull()][['id', 'cum_large_positive', 'cum_negative']]
            df_buy_point.reset_index(inplace = True)
            df_buy_point = df_buy_point.drop(columns = ['index'])

            df_sell_point = self.data_df[self.data_df['sell_point_temp'].notnull()][['id', 'cum_large_negative', 'cum_positive']]
            df_sell_point.reset_index(inplace = True)
            df_sell_point = df_sell_point.drop(columns = ['index'])

            temp_df = self.data_df[['id',
                                    'buy_point_temp', 'sell_point_temp'
                                    ]]

            temp_df = temp_df.fillna(method = 'ffill').fillna(0)

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)

            temp_df['id'] = temp_df['buy_point_temp']
            temp_df = pd.merge(temp_df, df_buy_point, on=['id'], how ='left')
            temp_df = temp_df.rename(columns = {
                'cum_large_positive': 'cum_large_positive_for_buy',
                'cum_negative': 'cum_negative_for_buy'
            })
            temp_df = temp_df.fillna(0)

            temp_df['id'] = temp_df['sell_point_temp']
            temp_df = pd.merge(temp_df, df_sell_point, on=['id'], how ='left')
            temp_df = temp_df.rename(columns = {
                'cum_large_negative': 'cum_large_negative_for_sell',
                'cum_positive' : 'cum_positive_for_sell'
            })
            temp_df = temp_df.fillna(0)

            temp_df = temp_df[[col for col in temp_df.columns if 'cum' in col]]

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)

            self.data_df = pd.concat([self.data_df, temp_df], axis = 1)

            self.data_df['num_large_positive_for_buy'] = self.data_df['cum_large_positive'] - self.data_df['cum_large_positive_for_buy']
            self.data_df['num_negative_for_buy'] = self.data_df['cum_negative'] - self.data_df['cum_negative_for_buy']

            self.data_df['num_large_negative_for_sell'] = self.data_df['cum_large_negative'] - self.data_df['cum_large_negative_for_sell']
            self.data_df['num_positive_for_sell'] = self.data_df['cum_positive'] - self.data_df['cum_positive_for_sell']


            self.data_df['special_buy_close_position'] = (self.data_df['num_large_positive_for_buy'] >= 2) & \
                                                         (self.data_df['num_negative_for_buy'] == 0) & \
                                                         self.data_df['is_negative'] & (self.data_df['low'] > self.data_df['upper_vegas'])
            self.data_df['special_sell_close_position'] = (self.data_df['num_large_negative_for_sell'] >= 2) & \
                                                          (self.data_df['num_positive_for_sell'] == 0) & \
                                                          self.data_df['is_positive'] & (self.data_df['high'] < self.data_df['lower_vegas'])



            #####################


            self.data_df['cum_special_buy_close_position'] = self.data_df['special_buy_close_position'].cumsum()
            self.data_df['cum_special_sell_close_position'] = self.data_df['special_sell_close_position'].cumsum()

            self.data_df['cum_special_buy_close_position'] = self.data_df['cum_special_buy_close_position'].shift(1)
            self.data_df.at[0, 'cum_special_buy_close_position'] = 0

            self.data_df['cum_special_sell_close_position'] = self.data_df['cum_special_sell_close_position'].shift(1)
            self.data_df.at[0, 'cum_special_sell_close_position'] = 0

            for col in ['cum_special_buy_close_position', 'cum_special_sell_close_position']:
                self.data_df[col] = self.data_df[col].astype(int)


            df_buy_point = self.data_df[self.data_df['buy_point_temp'].notnull()][['id', 'cum_special_buy_close_position']]
            df_buy_point.reset_index(inplace = True)
            df_buy_point = df_buy_point.drop(columns = ['index'])

            df_sell_point = self.data_df[self.data_df['sell_point_temp'].notnull()][['id', 'cum_special_sell_close_position']]
            df_sell_point.reset_index(inplace = True)
            df_sell_point = df_sell_point.drop(columns = ['index'])

            temp_df = self.data_df[['id',
                                    'buy_point_temp', 'sell_point_temp'
                                    ]]

            temp_df = temp_df.fillna(method = 'ffill').fillna(0)

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)





            temp_df['id'] = temp_df['buy_point_temp']
            temp_df = pd.merge(temp_df, df_buy_point, on=['id'], how ='left')
            temp_df = temp_df.rename(columns = {
                'cum_special_buy_close_position': 'cum_special_buy_close_position_for_buy'
            })
            temp_df = temp_df.fillna(0)

            temp_df['id'] = temp_df['sell_point_temp']
            temp_df = pd.merge(temp_df, df_sell_point, on=['id'], how ='left')
            temp_df = temp_df.rename(columns = {
                'cum_special_sell_close_position':'cum_special_sell_close_position_for_sell'
            })
            temp_df = temp_df.fillna(0)

            temp_df = temp_df[[col for col in temp_df.columns if 'cum' in col]]

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)

            self.data_df = pd.concat([self.data_df, temp_df], axis = 1)

            self.data_df['num_special_buy_close_position_for_buy'] = self.data_df['cum_special_buy_close_position'] - self.data_df['cum_special_buy_close_position_for_buy']
            self.data_df['num_special_sell_close_position_for_sell'] = self.data_df['cum_special_sell_close_position'] - self.data_df['cum_special_sell_close_position_for_sell']

            self.data_df['first_actual_special_buy_close_position'] = self.data_df['special_buy_close_position'] & \
                                                                      (self.data_df['num_special_buy_close_position_for_buy'] == 0)

            self.data_df['first_actual_special_sell_close_position'] = self.data_df['special_sell_close_position'] & \
                                                                      (self.data_df['num_special_sell_close_position_for_sell'] == 0)

            self.data_df['first_actual_special_buy_close_position_high'] = self.data_df['first_actual_special_buy_close_position'] & \
                                                                           (self.data_df['max_price_to_period_high_pct'] < 0.05)

            self.data_df['first_actual_special_sell_close_position_low'] = self.data_df['first_actual_special_sell_close_position'] & \
                                                                           (self.data_df['min_price_to_period_low_pct'] < 0.05)




            self.data_df['prev_sell_close_position_excessive'] = self.data_df['sell_close_position_excessive'].shift(1)
            self.data_df.at[0, 'prev_sell_close_position_excessive'] = False
            self.data_df['prev_sell_close_position_excessive'] = pd.Series(list(self.data_df['prev_sell_close_position_excessive']), dtype='bool')
            self.data_df['first_sell_close_position_excessive'] = self.data_df['sell_close_position_excessive'] & (~self.data_df['prev_sell_close_position_excessive'])
            #Special
            self.data_df['first_sell_close_position_excessive_add'] = self.data_df['first_sell_close_position_excessive'] | \
                                                                       self.data_df['first_actual_special_sell_close_position_low']
            #self.data_df['first_sell_close_position_excessive'] = self.data_df['first_sell_close_position_excessive'] | self.data_df['first_actual_special_sell_close_position']

            self.data_df['prev_sell_close_position_conservative'] = self.data_df['sell_close_position_conservative'].shift(1)
            self.data_df.at[0, 'prev_sell_close_position_conservative'] = False
            self.data_df['prev_sell_close_position_conservative'] = pd.Series(list(self.data_df['prev_sell_close_position_conservative']), dtype='bool')
            self.data_df['first_sell_close_position_conservative'] = self.data_df['sell_close_position_conservative'] & (~self.data_df['prev_sell_close_position_conservative'])

            self.data_df['prev_sell_stop_loss_excessive'] = self.data_df['sell_stop_loss_excessive'].shift(1)
            self.data_df.at[0, 'prev_sell_stop_loss_excessive'] = False
            self.data_df['prev_sell_stop_loss_excessive'] = pd.Series(list(self.data_df['prev_sell_stop_loss_excessive']), dtype='bool')
            self.data_df['first_sell_stop_loss_excessive'] = self.data_df['sell_stop_loss_excessive'] & (~self.data_df['prev_sell_stop_loss_excessive'])

            self.data_df['prev_sell_stop_loss_conservative'] = self.data_df['sell_stop_loss_conservative'].shift(1)
            self.data_df.at[0, 'prev_sell_stop_loss_conservative'] = False
            self.data_df['prev_sell_stop_loss_conservative'] = pd.Series(list(self.data_df['prev_sell_stop_loss_conservative']), dtype='bool')
            self.data_df['first_sell_stop_loss_conservative'] = self.data_df['sell_stop_loss_conservative'] & (~self.data_df['prev_sell_stop_loss_conservative'])




            self.data_df['prev_buy_close_position_excessive'] = self.data_df['buy_close_position_excessive'].shift(1)
            self.data_df.at[0, 'prev_buy_close_position_excessive'] = False
            self.data_df['prev_buy_close_position_excessive'] = pd.Series(list(self.data_df['prev_buy_close_position_excessive']), dtype='bool')
            self.data_df['first_buy_close_position_excessive'] = self.data_df['buy_close_position_excessive'] & (~self.data_df['prev_buy_close_position_excessive'])
            #Special
            self.data_df['first_buy_close_position_excessive_add'] = self.data_df['first_buy_close_position_excessive'] | \
                                                                       self.data_df['first_actual_special_buy_close_position_high']
            #self.data_df['first_buy_close_position_excessive'] = self.data_df['first_buy_close_position_excessive'] | self.data_df['first_actual_special_buy_close_position']


            self.data_df['prev_buy_close_position_conservative'] = self.data_df['buy_close_position_conservative'].shift(1)
            self.data_df.at[0, 'prev_buy_close_position_conservative'] = False
            self.data_df['prev_buy_close_position_conservative'] = pd.Series(list(self.data_df['prev_buy_close_position_conservative']), dtype='bool')
            self.data_df['first_buy_close_position_conservative'] = self.data_df['buy_close_position_conservative'] & (~self.data_df['prev_buy_close_position_conservative'])

            self.data_df['prev_buy_stop_loss_excessive'] = self.data_df['buy_stop_loss_excessive'].shift(1)
            self.data_df.at[0, 'prev_buy_stop_loss_excessive'] = False
            self.data_df['prev_buy_stop_loss_excessive'] = pd.Series(list(self.data_df['prev_buy_stop_loss_excessive']), dtype='bool')
            self.data_df['first_buy_stop_loss_excessive'] = self.data_df['buy_stop_loss_excessive'] & (~self.data_df['prev_buy_stop_loss_excessive'])

            self.data_df['prev_buy_stop_loss_conservative'] = self.data_df['buy_stop_loss_conservative'].shift(1)
            self.data_df.at[0, 'prev_buy_stop_loss_conservative'] = False
            self.data_df['prev_buy_stop_loss_conservative'] = pd.Series(list(self.data_df['prev_buy_stop_loss_conservative']), dtype='bool')
            self.data_df['first_buy_stop_loss_conservative'] = self.data_df['buy_stop_loss_conservative'] & (~self.data_df['prev_buy_stop_loss_conservative'])




            ##################################
            ## Hutong Sauna

            self.data_df['price_cross_down_vegas'] = np.where(
                (self.data_df['close'] < self.data_df['lower_vegas']) & (self.data_df['open'] >= self.data_df['lower_vegas']),
                1,
                0
            )

            self.data_df['price_cross_up_vegas'] = np.where(
                (self.data_df['close'] > self.data_df['upper_vegas']) & (self.data_df['open'] <= self.data_df['upper_vegas']),
                1,
                0
            )


            self.data_df['period_high' + str(high_low_window) + "_change"] = np.where(
                np.abs(self.data_df['period_high' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate) >= high_low_delta_threshold,
                1,
                0
            )

            self.data_df['period_high' + str(high_low_window) + "_change"] = np.where(
                (self.data_df['period_high' + str(high_low_window) + "_change"] == 1) | (self.data_df['price_cross_up_vegas'] == 1),
                1,
                0
            )


            self.data_df['period_low' + str(high_low_window) + "_change"] = np.where(
                np.abs(self.data_df['period_low' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate) >= high_low_delta_threshold,
                1,
                0
            )

            self.data_df['period_low' + str(high_low_window) + "_change"] = np.where(
                (self.data_df['period_low' + str(high_low_window) + "_change"] == 1) | (self.data_df['price_cross_down_vegas'] == 1),
                1,
                0
            )



            self.data_df['cum_first_sell_close_position_excessive'] = self.data_df['first_sell_close_position_excessive_add'].cumsum()
            self.data_df['cum_first_sell_close_position_conservative'] = self.data_df['first_sell_close_position_conservative'].cumsum()

            self.data_df['cum_first_buy_close_position_excessive'] = self.data_df['first_buy_close_position_excessive_add'].cumsum()
            self.data_df['cum_first_buy_close_position_conservative'] = self.data_df['first_buy_close_position_conservative'].cumsum()


            low_period_cum_columns = ['cum_first_sell_close_position_excessive', 'cum_first_sell_close_position_conservative']
            high_period_cum_columns = ['cum_first_buy_close_position_excessive', 'cum_first_buy_close_position_conservative']

            for col in low_period_cum_columns + high_period_cum_columns:
                self.data_df[col] = self.data_df[col].shift(1)
                self.data_df.at[0, col] = 0


            self.data_df['period_high' + str(high_low_window) + '_change_temp'] = np.nan
            self.data_df['period_high' + str(high_low_window) + '_change_temp'] = np.where(
                self.data_df['period_high' + str(high_low_window) + '_change'] == 1,
                self.data_df['id'],
                self.data_df['period_high' + str(high_low_window) + '_change_temp']
            )

            self.data_df['period_low' + str(high_low_window) + '_change_temp'] = np.nan
            self.data_df['period_low' + str(high_low_window) + '_change_temp'] = np.where(
                self.data_df['period_low' + str(high_low_window) + '_change'] == 1,
                self.data_df['id'],
                self.data_df['period_low' + str(high_low_window) + '_change_temp']
            )


            df_low_change = self.data_df[self.data_df['period_low' + str(high_low_window) + '_change_temp'].notnull()][['id'] + low_period_cum_columns]
            df_low_change.reset_index(inplace = True)
            df_low_change = df_low_change.drop(columns = ['index'])

            df_high_change = self.data_df[self.data_df['period_high' + str(high_low_window) + '_change_temp'].notnull()][['id'] + high_period_cum_columns]
            df_high_change.reset_index(inplace = True)
            df_high_change = df_high_change.drop(columns = ['index'])

            temp_df = self.data_df[['id', 'period_high' + str(high_low_window) + '_change_temp', 'period_low' + str(high_low_window) + '_change_temp']]
            temp_df = temp_df.fillna(method = 'ffill').fillna(0)

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)


            temp_df['id'] = temp_df['period_low' + str(high_low_window) + '_change_temp']
            temp_df = pd.merge(temp_df, df_low_change, on = ['id'], how = 'left')
            temp_df = temp_df.rename(columns = {
                'cum_first_sell_close_position_excessive' : 'cum_first_sell_close_position_excessive_for_low_change',
                'cum_first_sell_close_position_conservative': 'cum_first_sell_close_position_conservative_for_low_change'
            })
            temp_df = temp_df.fillna(0)


            temp_df['id'] = temp_df['period_high' + str(high_low_window) + '_change_temp']
            temp_df = pd.merge(temp_df, df_high_change, on = ['id'], how = 'left')
            temp_df = temp_df.rename(columns = {
                'cum_first_buy_close_position_excessive' : 'cum_first_buy_close_position_excessive_for_high_change',
                'cum_first_buy_close_position_conservative': 'cum_first_buy_close_position_conservative_for_high_change'
            })
            temp_df = temp_df.fillna(0)

            temp_df = temp_df[[col for col in temp_df.columns if 'cum' in col]]

            for col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(int)

            self.data_df = pd.concat([self.data_df, temp_df], axis = 1)

            self.data_df = self.data_df.drop(columns = [col for col in self.data_df.columns if 'temp' in col])

            self.data_df['num_first_sell_close_position_excessive_in_low_change'] = \
                self.data_df['cum_first_sell_close_position_excessive'] - self.data_df['cum_first_sell_close_position_excessive_for_low_change']

            self.data_df['num_first_sell_close_position_conservative_in_low_change'] = \
                self.data_df['cum_first_sell_close_position_conservative'] - self.data_df['cum_first_sell_close_position_conservative_for_low_change']


            self.data_df['num_first_buy_close_position_excessive_in_high_change'] = \
                self.data_df['cum_first_buy_close_position_excessive'] - self.data_df['cum_first_buy_close_position_excessive_for_high_change']


            self.data_df['num_first_buy_close_position_conservative_in_high_change'] = \
                self.data_df['cum_first_buy_close_position_conservative'] - self.data_df['cum_first_buy_close_position_conservative_for_high_change']



            self.data_df['first_actual_sell_close_position_excessive'] = self.data_df['first_sell_close_position_excessive'] & \
                                                                         (self.data_df['num_first_sell_close_position_excessive_in_low_change'] == 0)

            self.data_df['first_actual_sell_close_position_conservative'] = self.data_df['first_sell_close_position_conservative'] & \
                                                                         (self.data_df['num_first_sell_close_position_conservative_in_low_change'] == 0)


            self.data_df['first_actual_buy_close_position_excessive'] = self.data_df['first_buy_close_position_excessive'] & \
                                                                         (self.data_df['num_first_buy_close_position_excessive_in_high_change'] == 0)

            self.data_df['first_actual_buy_close_position_conservative'] = self.data_df['first_buy_close_position_conservative'] & \
                                                                         (self.data_df['num_first_buy_close_position_conservative_in_high_change'] == 0)



            #################################################




    def trade(self):

        print("Do trading............")


        signal_attrs = ['buy_weak_ready', 'buy_weak_fire',  'buy_ready', 'buy_fire',
                        'sell_weak_ready', 'sell_weak_fire',  'sell_ready', 'sell_fire']

        self.condition.acquire()

        data_df100 = None
        data_df200 = None

        for file in os.listdir(self.chart_folder):
            file_path = os.path.join(self.chart_folder, file)
            if 'png' in file:
                os.remove(file_path)

        for file in os.listdir(self.simple_chart_folder):
            file_path = os.path.join(self.simple_chart_folder, file)
            if 'png' in file:
                os.remove(file_path)


        ##          use2TypeSignals  filter_option
        #Option 0      False             0
        #Option 1      True              0   Benchmark 100
        #Option 2      False             1
        #Option 3      False             2
        #Option 4      True              1
        #Option 5      True              2

        use2TypeSignals = True
        filter_option = 1

        while True:

            for high_low_window, data_df_backup in list(zip(high_low_window_options, self.data_dfs_backup)):

                self.data_df = self.data_df[['currency', 'time', 'open', 'high', 'low', 'close']]

                self.calculate_signals(high_low_window)


                if self.is_cut_data:

                    increment_data_df = self.data_df[self.data_df['time'] > data_df_backup.iloc[-1]['time']]
                    if increment_data_df.shape[0] > 0:

                        self.data_df = pd.concat([data_df_backup, increment_data_df])

                        self.data_df.reset_index(inplace = True)
                        self.data_df = self.data_df.drop(columns = ['index'])


                    else:

                        self.data_df = data_df_backup



                print("to csv:")
                if high_low_window == 200:
                    self.data_df.to_csv(os.path.join(self.data_folder, self.currency + str(high_low_window) + ".csv"), index=False)
                print("after to csv:")

                if len(high_low_window_options) > 1:
                    if high_low_window == 100:
                        data_df100 = self.data_df.copy()
                    elif high_low_window == 200:
                        data_df200 = self.data_df.copy()




                print("Process data_df final:")
                print(self.data_df[['time', 'close']].head(10))



            if len(high_low_window_options) > 1:

                self.data_df = data_df100

                self.data_df['period_high' + str(high_low_window_options[-1])] = data_df200['period_high' + str(high_low_window_options[-1])]
                self.data_df['period_low' + str(high_low_window_options[-1])] = data_df200['period_low' + str(high_low_window_options[-1])]

                self.data_df['macd_period_high' + str(high_low_window_options[-1])] = data_df200['macd_period_high' + str(high_low_window_options[-1])]
                self.data_df['macd_period_low' + str(high_low_window_options[-1])] = data_df200['macd_period_low' + str(high_low_window_options[-1])]



                # print("period_high200 here:")
                # print(self.data_df[['time', 'close', 'period_high200']].tail(10))


                if use2TypeSignals:
                    data_df100['final_buy_fire'] = data_df100['buy_real_fire3'] | data_df100['buy_real_fire2']

                    data_df100['final_sell_fire'] = data_df100['sell_real_fire3'] | data_df100['sell_real_fire2']
                else:
                    data_df100['final_buy_fire'] = data_df100['buy_real_fire3']

                    data_df100['final_sell_fire'] = data_df100['sell_real_fire3']

                if filter_option > 0:
                    if use2TypeSignals:
                        data_df200['final_buy_fire'] = data_df200['buy_real_fire3'] | data_df200['buy_real_fire2']

                        data_df200['final_sell_fire'] = data_df200['sell_real_fire3'] | data_df200['sell_real_fire2']
                    else:
                        data_df200['final_buy_fire'] = data_df200['buy_real_fire3']

                        data_df200['final_sell_fire'] = data_df200['sell_real_fire3']

                    if filter_option == 1:

                        #self.data_df['final_buy_fire_exclude'] = data_df100['final_buy_fire'] & (~data_df200['final_buy_fire']) & data_df100['strongly_aligned_short_condition']
                        self.data_df['final_buy_fire_exclude'] = data_df100['buy_real_fire3'] & (~data_df200['buy_real_fire3']) & data_df100['strongly_aligned_short_condition']

                        self.data_df['final_buy_fire'] = data_df100['final_buy_fire'] & (~self.data_df['final_buy_fire_exclude'])

                        #self.data_df['final_sell_fire_exclude'] = data_df100['final_sell_fire'] & (~data_df200['final_sell_fire']) & data_df100['strongly_aligned_long_condition']
                        self.data_df['final_sell_fire_exclude'] = data_df100['sell_real_fire3'] & (~data_df200['sell_real_fire3']) & data_df100['strongly_aligned_long_condition']

                        self.data_df['final_sell_fire'] = data_df100['final_sell_fire'] & (~self.data_df['final_sell_fire_exclude'])


                    elif filter_option == 2:

                        self.data_df['final_buy_fire'] = data_df100['final_buy_fire'] | data_df200['final_buy_fire']
                        self.data_df['final_buy_fire_include'] = (data_df100['final_buy_fire'] & data_df200['final_buy_fire']) | (~data_df100['strongly_aligned_short_condition'])
                        self.data_df['final_buy_fire'] = self.data_df['final_buy_fire'] & self.data_df['final_buy_fire_include']

                        self.data_df['final_sell_fire'] = data_df100['final_sell_fire'] | data_df200['final_sell_fire']
                        self.data_df['final_sell_fire_include'] = (data_df100['final_sell_fire'] & data_df200['final_sell_fire']) | (~data_df100['strongly_aligned_long_condition'])
                        self.data_df['final_sell_fire'] = self.data_df['final_sell_fire'] & self.data_df['final_sell_fire_include']

            else:

                self.data_df['final_buy_fire'] = self.data_df['buy_real_fire3'] | self.data_df['buy_real_fire2']
                self.data_df['final_sell_fire'] = self.data_df['sell_real_fire3'] | self.data_df['sell_real_fire2']




            ############# Magic Filter, oh yeah! #####################

            if 100 in high_low_window_options:
                high_low_window = 100
                self.data_df['buy_fire_special_exclude'] = self.data_df['final_buy_fire'] &\
                    (self.data_df['period_high' + str(high_low_window) + '_go_down_duration'] < 0.2*self.data_df['period_low' + str(high_low_window) + '_go_down_duration'])

                self.data_df['buy_fire_special_exclude_exempt'] = self.data_df['buy_fire_special_exclude'] &\
                    self.data_df['buy_real_fire3'] & (self.data_df['period_high' + str(high_low_window) + '_go_down_duration'] <= 3)

                self.data_df['final_buy_fire'] = self.data_df['final_buy_fire'] & \
                                                 ((~self.data_df['buy_fire_special_exclude']) | self.data_df['buy_fire_special_exclude_exempt'])




                self.data_df['sell_fire_special_exclude'] = self.data_df['final_sell_fire'] &\
                    (self.data_df['period_low' + str(high_low_window) + '_go_up_duration'] < 0.2*self.data_df['period_high' + str(high_low_window) + '_go_up_duration'])

                self.data_df['sell_fire_special_exclude_exempt'] = self.data_df['sell_fire_special_exclude'] &\
                    self.data_df['sell_real_fire3'] & (self.data_df['period_low' + str(high_low_window) + '_go_up_duration'] <= 3)

                self.data_df['final_sell_fire'] = self.data_df['final_sell_fire'] & \
                                                   ((~self.data_df['sell_fire_special_exclude']) | self.data_df['sell_fire_special_exclude_exempt'])



            # self.data_df['final_buy_fire'] = self.data_df['final_buy_fire'] & (~self.data_df['strongly_aligned_short_condition'])
            # self.data_df['final_sell_fire'] = self.data_df['final_sell_fire'] & (~self.data_df['strongly_aligned_long_condition'])


            # if 100 in high_low_window_options:
            #     high_low_window = 100
            #     self.data_df['buy_fire_magic_exclude'] = self.data_df['final_buy_fire'] & (self.data_df['num_high_go_down_in_low_go_down'] >= 1)
            #
            #     self.data_df['buy_exclude_exempt1'] = ~self.data_df['strongly_aligned_short_condition']
            #     self.data_df['buy_exclude_exempt2'] = self.data_df['period_low' + str(high_low_window) + '_go_down_duration'] > period_lookback
            #     self.data_df['buy_exclude_exempt3'] = self.data_df['period_high' + str(high_low_window) + '_go_down_duration'] > 0.5 * self.data_df['period_low' + str(high_low_window) + '_go_down_duration']
            #     self.data_df['buy_exclude_exempt4'] = self.data_df['period_high' + str(high_low_window) + '_go_down_duration'] >= 24
            #
            #     # self.data_df['buy_exclude_exempt'] = self.data_df['buy_exclude_exempt1'] & self.data_df['buy_exclude_exempt2'] & \
            #     #                                      (self.data_df['buy_exclude_exempt3'] | self.data_df['buy_exclude_exempt4'])
            #
            #     self.data_df['buy_exclude_exempt'] = self.data_df['buy_exclude_exempt3']
            #
            #     #
            #     # self.data_df['buy_exclude_exempt1'] = self.data_df['period_low' + str(high_low_window) + '_go_down_duration'] > period_lookback
            #     # #self.data_df['buy_exclude_exempt2'] = self.data_df['period_high' + str(high_low_window) + '_go_down_duration'] > period_lookback
            #     # self.data_df['buy_exclude_exempt'] = self.data_df['buy_exclude_exempt1'] #& self.data_df['buy_exclude_exempt2']
            #
            #     self.data_df['buy_fire_magic_exclude'] = self.data_df['buy_fire_magic_exclude'] & (~self.data_df['buy_exclude_exempt'])
            #     self.data_df['final_buy_fire'] = self.data_df['final_buy_fire'] & (~self.data_df['buy_fire_magic_exclude'])
            #
            #
            #     self.data_df['sell_fire_magic_exclude'] = self.data_df['final_sell_fire'] & (self.data_df['num_low_go_up_in_high_go_up'] >= 1)
            #
            #     self.data_df['sell_exclude_exempt1'] = ~self.data_df['strongly_aligned_long_condition']
            #     self.data_df['sell_exclude_exempt2'] = self.data_df['period_high' + str(high_low_window) + '_go_up_duration'] > period_lookback
            #     self.data_df['sell_exclude_exempt3'] = self.data_df['period_low' + str(high_low_window) + '_go_up_duration'] > 0.5 * self.data_df['period_high' + str(high_low_window) + '_go_up_duration']
            #     self.data_df['sell_exclude_exempt4'] = self.data_df['period_low' + str(high_low_window) + '_go_up_duration'] >= 24
            #
            #     # self.data_df['sell_exclude_exempt'] = self.data_df['sell_exclude_exempt1'] & self.data_df['sell_exclude_exempt2'] & \
            #     #                                      (self.data_df['sell_exclude_exempt3'] | self.data_df['sell_exclude_exempt4'])
            #
            #     self.data_df['sell_exclude_exempt'] = self.data_df['sell_exclude_exempt3']
            #     #
            #     # self.data_df['sell_exclude_exempt1'] = self.data_df['period_high' + str(high_low_window) + '_go_up_duration'] > period_lookback
            #     # #self.data_df['sell_exclude_exempt2'] = self.data_df['period_low' + str(high_low_window) + '_go_up_duration'] > period_lookback
            #     # self.data_df['sell_exclude_exempt'] = self.data_df['sell_exclude_exempt1'] #& self.data_df['sell_exclude_exempt2']
            #
            #     self.data_df['sell_fire_magic_exclude'] = self.data_df['sell_fire_magic_exclude'] & (~self.data_df['sell_exclude_exempt'])
            #     self.data_df['final_sell_fire'] = self.data_df['final_sell_fire'] & (~self.data_df['sell_fire_magic_exclude'])
            #

            



            ##########################################################




            self.data_df['prev_final_buy_fire'] = self.data_df['final_buy_fire'].shift(1)
            self.data_df.at[0, 'prev_final_buy_fire'] = False
            self.data_df['prev_final_buy_fire'] = pd.Series(list(self.data_df['prev_final_buy_fire']), dtype='bool')
            self.data_df['first_final_buy_fire'] = self.data_df['final_buy_fire'] & (~self.data_df['prev_final_buy_fire'])

            self.data_df['prev_final_sell_fire'] = self.data_df['final_sell_fire'].shift(1)
            self.data_df.at[0, 'prev_final_sell_fire'] = False
            self.data_df['prev_final_sell_fire'] = pd.Series(list(self.data_df['prev_final_sell_fire']), dtype='bool')
            self.data_df['first_final_sell_fire'] = self.data_df['final_sell_fire'] & (~self.data_df['prev_final_sell_fire'])

            if is_plot_exclude:
                if 'final_buy_fire_exclude' in self.data_df.columns:
                    self.data_df['prev_final_buy_fire_exclude'] = self.data_df['final_buy_fire_exclude'].shift(1)
                    self.data_df.at[0, 'prev_final_buy_fire_exclude'] = False
                    self.data_df['prev_final_buy_fire_exclude'] = pd.Series(list(self.data_df['prev_final_buy_fire_exclude']), dtype='bool')
                    self.data_df['first_final_buy_fire_exclude'] = self.data_df['final_buy_fire_exclude'] & (~self.data_df['prev_final_buy_fire_exclude'])

                if 'final_sell_fire_exclude' in self.data_df.columns:
                    self.data_df['prev_final_sell_fire_exclude'] = self.data_df['final_sell_fire_exclude'].shift(1)
                    self.data_df.at[0, 'prev_final_sell_fire_exclude'] = False
                    self.data_df['prev_final_sell_fire_exclude'] = pd.Series(list(self.data_df['prev_final_sell_fire_exclude']), dtype='bool')
                    self.data_df['first_final_sell_fire_exclude'] = self.data_df['final_sell_fire_exclude'] & (~self.data_df['prev_final_sell_fire_exclude'])


                if 'buy_fire_magic_exclude' in self.data_df.columns:
                    self.data_df['prev_buy_fire_magic_exclude'] = self.data_df['buy_fire_magic_exclude'].shift(1)
                    self.data_df.at[0, 'prev_buy_fire_magic_exclude'] = False
                    self.data_df['prev_buy_fire_magic_exclude'] = pd.Series(list(self.data_df['prev_buy_fire_magic_exclude']), dtype='bool')
                    self.data_df['first_buy_fire_magic_exclude'] = self.data_df['buy_fire_magic_exclude'] & (~self.data_df['prev_buy_fire_magic_exclude'])


                if 'sell_fire_magic_exclude' in self.data_df.columns:
                    self.data_df['prev_sell_fire_magic_exclude'] = self.data_df['sell_fire_magic_exclude'].shift(1)
                    self.data_df.at[0, 'prev_sell_fire_magic_exclude'] = False
                    self.data_df['prev_sell_fire_magic_exclude'] = pd.Series(list(self.data_df['prev_sell_fire_magic_exclude']), dtype='bool')
                    self.data_df['first_sell_fire_magic_exclude'] = self.data_df['sell_fire_magic_exclude'] & (~self.data_df['prev_sell_fire_magic_exclude'])




            ########### Calculate Phase 1 close position points ######

            self.data_df['buy_point'] = np.where(
                    self.data_df['first_final_buy_fire'],
                    1,
                    0
                )

            self.data_df['buy_point_id'] = self.data_df['buy_point'].cumsum()

            self.data_df['sell_point'] = np.where(
                self.data_df['first_final_sell_fire'],
                1,
                0
            )

            self.data_df['sell_point_id'] = self.data_df['sell_point'].cumsum()

            self.data_df['buy_point_temp'] = np.nan
            self.data_df['buy_point_temp'] = np.where(
                self.data_df['buy_point'] == 1,
                self.data_df['id'],
                self.data_df['buy_point_temp']
            )

            self.data_df['sell_point_temp'] = np.nan
            self.data_df['sell_point_temp'] = np.where(
                self.data_df['sell_point'] == 1,
                self.data_df['id'],
                self.data_df['sell_point_temp']
            )


            if True:
                temp_df = self.data_df[['id', 'buy_point', 'sell_point', 'buy_point_id', 'sell_point_id', 'close',
                                        'period_high' + str(high_low_window) + '_change', 'period_low' + str(high_low_window) + '_change']]

                # print("temp_df:")
                # print(temp_df.tail(20))

                temp_df['buy_group'] = np.nan
                temp_df['buy_group'] = np.where(
                    temp_df['period_low' + str(high_low_window) + '_change'] == 1,
                    temp_df['id'],
                    temp_df['buy_group']
                )
                temp_df = temp_df.fillna(method = 'ffill').fillna(0)


                temp_df['sell_group'] = np.nan
                temp_df['sell_group'] = np.where(
                    temp_df['period_high' + str(high_low_window) + '_change'] == 1,
                    temp_df['id'],
                    temp_df['sell_group']
                )
                temp_df = temp_df.fillna(method='ffill').fillna(0)


                for col in temp_df.columns:
                    if col != 'close':
                        temp_df[col] = temp_df[col].astype(int)


                buy_df = temp_df[temp_df['buy_point'] == 1]
                sell_df = temp_df[temp_df['sell_point'] == 1]



                def calc_cum_min(x):
                    # print("Group buy")
                    # print(x)

                    x['group_min_price'] = x['close'].cummin()
                    return x

                def calc_cum_max(x):
                    x['group_max_price'] = x['close'].cummax()
                    return x

                buy_df = buy_df.groupby(['buy_group']).apply(lambda x: calc_cum_min(x))
                sell_df = sell_df.groupby(['sell_group']).apply(lambda x: calc_cum_max(x))

                buy_df = buy_df[['buy_point_id', 'group_min_price']]
                sell_df = sell_df[['sell_point_id', 'group_max_price']]

                temp_df = pd.merge(temp_df, buy_df, on = ['buy_point_id'], how = 'left')
                temp_df = pd.merge(temp_df, sell_df, on = ['sell_point_id'], how = 'left')

                temp_df = temp_df.fillna(0)

                self.data_df['buy_group'] = temp_df['buy_group']
                self.data_df['sell_group'] = temp_df['sell_group']

                self.data_df['group_min_price'] = temp_df['group_min_price']
                self.data_df['group_max_price'] = temp_df['group_max_price']



                temp_df = temp_df[['id','buy_point','sell_point']]
                temp_df['buy_point_support'] = np.nan
                temp_df['buy_point_support'] = np.where(
                    self.data_df['buy_point'] == 1,
                    self.data_df['period_low' + str(high_low_window)],
                    temp_df['buy_point_support']
                )



                temp_df['sell_point_support'] = np.nan
                temp_df['sell_point_support'] = np.where(
                    self.data_df['sell_point'] == 1,
                    self.data_df['period_high' + str(high_low_window)],
                    temp_df['sell_point_support']
                )


                temp_df = temp_df.fillna(method='ffill').fillna(0)


                # for col in temp_df.columns:
                #     temp_df[col] = temp_df[col].astype(int)

                self.data_df['buy_point_support'] = temp_df['buy_point_support']
                self.data_df['sell_point_support'] = temp_df['sell_point_support']




                ########################
                self.data_df['bar_above_guppy'] = self.data_df['min_price'] > self.data_df['highest_guppy']
                self.data_df['bar_below_guppy'] = self.data_df['max_price'] < self.data_df['lowest_guppy']

                self.data_df['bar_above_passive_guppy'] = self.data_df['low'] > self.data_df['lowest_guppy']
                self.data_df['bar_below_passive_guppy'] = self.data_df['high'] < self.data_df['highest_guppy']

                self.data_df['cum_bar_above_guppy'] = self.data_df['bar_above_guppy'].cumsum()
                self.data_df['cum_bar_below_guppy'] = self.data_df['bar_below_guppy'].cumsum()

                self.data_df['cum_bar_above_passive_guppy'] = self.data_df['bar_above_passive_guppy'].cumsum()
                self.data_df['cum_bar_below_passive_guppy'] = self.data_df['bar_below_passive_guppy'].cumsum()

                # self.data_df['bar_above_vegas'] = (self.data_df['min'] - self.data_df['upper_vegas']) * self.lot_size * self.exchange_rate >= 300
                # self.data_df['bar_below_vegas'] = (self.data_df['max'] - self.data_df['lower_vegas']) * self.lot_size * self.exchange_rate <= -300

                # self.data_df['cum_bar_above_vegas'] = self.data_df['bar_above_vegas'].cumsum()
                # self.data_df['cum_bar_below_vegas'] = self.data_df['bar_below_vegas'].cumsum()

                self.data_df['cum_above_vegas'] = self.data_df['is_above_vegas_strict'].cumsum()
                self.data_df['cum_below_vegas'] = self.data_df['is_below_vegas_strict'].cumsum()

                cum_buy_cols = ['cum_bar_above_guppy', 'cum_bar_above_passive_guppy',  'cum_above_vegas']
                cum_sell_cols = ['cum_bar_below_guppy', 'cum_bar_below_passive_guppy',  'cum_below_vegas']

                for cum_col in cum_buy_cols + cum_sell_cols:
                    self.data_df[cum_col] = self.data_df[cum_col].shift(1)
                    self.data_df.at[0, cum_col] = 0
                    self.data_df[cum_col] = self.data_df[cum_col].astype(int)

                df_buy = self.data_df[self.data_df['buy_point_temp'].notnull()][['id'] + cum_buy_cols]
                df_buy.reset_index(inplace = True)
                df_buy = df_buy.drop(columns = ['index'])

                df_sell = self.data_df[self.data_df['sell_point_temp'].notnull()][['id'] + cum_sell_cols]
                df_sell.reset_index(inplace = True)
                df_sell = df_sell.drop(columns = ['index'])


                temp_df = self.data_df[['id', 'buy_point_temp', 'sell_point_temp']]
                temp_df = temp_df.fillna(method = 'ffill').fillna(0)

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)


                temp_df['id'] = temp_df['buy_point_temp']
                temp_df = pd.merge(temp_df, df_buy, on = ['id'], how = 'left')
                temp_df = temp_df.rename(columns = {
                    'cum_bar_above_guppy' : 'cum_bar_above_guppy_for_buy',
                    'cum_bar_above_passive_guppy' : 'cum_bar_above_passive_guppy_for_buy',
                    'cum_above_vegas' : 'cum_above_vegas_for_buy'
                })
                temp_df = temp_df.fillna(0)


                temp_df['id'] = temp_df['sell_point_temp']
                temp_df = pd.merge(temp_df, df_sell, on = ['id'], how = 'left')
                temp_df = temp_df.rename(columns = {
                    'cum_bar_below_guppy' : 'cum_bar_below_guppy_for_sell',
                    'cum_bar_below_passive_guppy' : 'cum_bar_below_passive_guppy_for_sell',
                    'cum_below_vegas' : 'cum_below_vegas_for_sell'
                })
                temp_df = temp_df.fillna(0)

                temp_df = temp_df[[col for col in temp_df.columns if 'cum' in col]]

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)

                self.data_df = pd.concat([self.data_df, temp_df], axis = 1)

                self.data_df['num_bar_above_guppy_for_buy'] = self.data_df['cum_bar_above_guppy'] - self.data_df['cum_bar_above_guppy_for_buy']
                self.data_df['num_bar_above_passive_guppy_for_buy'] = self.data_df['cum_bar_above_passive_guppy'] - self.data_df['cum_bar_above_passive_guppy_for_buy']
                self.data_df['num_above_vegas_for_buy'] = self.data_df['cum_above_vegas'] - self.data_df['cum_above_vegas_for_buy']

                self.data_df['num_bar_below_guppy_for_sell'] = self.data_df['cum_bar_below_guppy'] - self.data_df['cum_bar_below_guppy_for_sell']
                self.data_df['num_bar_below_passive_guppy_for_sell'] = self.data_df['cum_bar_below_passive_guppy'] - self.data_df['cum_bar_below_passive_guppy_for_sell']
                self.data_df['num_below_vegas_for_sell'] = self.data_df['cum_below_vegas'] - self.data_df['cum_below_vegas_for_sell']

                #Critical
                self.data_df['buy_close_position_guppy'] = (self.data_df['open'] > self.data_df['highest_guppy']) &\
                                                           (self.data_df['ma_close12'] < self.data_df['lower_vegas']) &\
                                                           (self.data_df['close'] < self.data_df['highest_guppy']) &\
                                                           (self.data_df['num_bar_above_guppy_for_buy'] > 1) &\
                                                           (~self.data_df['strongly_half_aligned_long_condition'])

                self.data_df['sell_close_position_guppy'] = (self.data_df['open'] < self.data_df['lowest_guppy']) & \
                                                            (self.data_df['ma_close12'] > self.data_df['upper_vegas']) & \
                                                            (self.data_df['close'] > self.data_df['lowest_guppy']) & \
                                                            (self.data_df['num_bar_below_guppy_for_sell'] > 1) & \
                                                            (~self.data_df['strongly_half_aligned_short_condition'])




                self.data_df['buy_close_position_vegas'] = (self.data_df['is_negative']) & \
                                                           ((self.data_df['close'] - self.data_df['lower_vegas'])*self.lot_size*self.exchange_rate < -20) &\
                                                           ((self.data_df['high'] > self.data_df['lower_vegas']) | (self.data_df['prev_high'] > self.data_df['lower_vegas'])) &\
                                                           (self.data_df['num_above_vegas_for_buy'] == 0) &\
                                    ((self.data_df['max_price_to_lower_vegas']/self.data_df['price_range'] < 0.4) | (self.data_df['prev_max_price_to_lower_vegas']/self.data_df['prev_price_range'] < 0.4))


                self.data_df['sell_close_position_vegas'] = (self.data_df['is_positive']) & \
                                                           ((self.data_df['close'] - self.data_df['upper_vegas'])*self.lot_size*self.exchange_rate > 20) &\
                                                           ((self.data_df['low'] < self.data_df['upper_vegas']) | (self.data_df['prev_low'] < self.data_df['upper_vegas'])) & \
                                                            (self.data_df['num_below_vegas_for_sell'] == 0) &\
                                    ((self.data_df['min_price_to_upper_vegas']/self.data_df['price_range'] < 0.4) | (self.data_df['prev_min_price_to_upper_vegas']/self.data_df['prev_price_range'] < 0.4))


                # self.data_df['critical_ratio_buy'] = np.abs(self.data_df['min_price_to_upper_vegas'])/self.data_df['price_range']
                # self.data_df['critical_ratio_sell'] = np.abs(self.data_df['max_price_to_lower_vegas'])/self.data_df['price_range']
                #
                # self.data_df['critical_bool_buy'] = ((np.abs(self.data_df['min_price_to_upper_vegas'])/self.data_df['price_range'] < 0.4) | (np.abs(self.data_df['prev_min_price_to_upper_vegas'])/self.data_df['prev_price_range'] < 0.4))
                #
                # self.data_df['critical_bool_sell'] = ((np.abs(self.data_df['max_price_to_lower_vegas'])/self.data_df['price_range'] < 0.4) | (np.abs(self.data_df['prev_max_price_to_lower_vegas'])/self.data_df['prev_price_range'] < 0.4))

                # print("Fucking you:")
                # print(self.data_df.iloc[276:285][['time','id','max_price_to_lower_vegas', 'price_range', 'critical_ratio_sell', 'critical_bool_sell',
                #                                   'sell_close_position_vegas']])
                # sys.exit(0)

                self.data_df['buy_close_position_final_excessive'] = (self.data_df['close'] - self.data_df['group_min_price'])*self.lot_size*self.exchange_rate < -100
                self.data_df['sell_close_position_final_excessive'] = (self.data_df['close'] - self.data_df['group_max_price'])*self.lot_size*self.exchange_rate > 100

                self.data_df['buy_close_position_final_conservative'] = (self.data_df['close'] - self.data_df['buy_point_support'])*self.lot_size*self.exchange_rate < -300
                self.data_df['sell_close_position_final_conservative'] = (self.data_df['close'] - self.data_df['sell_point_support'])*self.lot_size*self.exchange_rate > 300



                self.data_df['prev_buy_close_position_guppy'] = self.data_df['buy_close_position_guppy'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_guppy'] = False
                self.data_df['prev_buy_close_position_guppy'] = pd.Series(list(self.data_df['prev_buy_close_position_guppy']), dtype='bool')
                self.data_df['first_buy_close_position_guppy'] = self.data_df['buy_close_position_guppy'] & (~self.data_df['prev_buy_close_position_guppy'])

                self.data_df['prev_buy_close_position_vegas'] = self.data_df['buy_close_position_vegas'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_vegas'] = False
                self.data_df['prev_buy_close_position_vegas'] = pd.Series(list(self.data_df['prev_buy_close_position_vegas']), dtype='bool')
                self.data_df['first_buy_close_position_vegas'] = self.data_df['buy_close_position_vegas'] & (~self.data_df['prev_buy_close_position_vegas'])

                self.data_df['prev_buy_close_position_final_excessive'] = self.data_df['buy_close_position_final_excessive'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_excessive'] = False
                self.data_df['prev_buy_close_position_final_excessive'] = pd.Series(list(self.data_df['prev_buy_close_position_final_excessive']), dtype='bool')
                self.data_df['first_buy_close_position_final_excessive'] = self.data_df['buy_close_position_final_excessive'] & (~self.data_df['prev_buy_close_position_final_excessive'])

                self.data_df['prev_buy_close_position_final_conservative'] = self.data_df['buy_close_position_final_conservative'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_conservative'] = False
                self.data_df['prev_buy_close_position_final_conservative'] = pd.Series(list(self.data_df['prev_buy_close_position_final_conservative']), dtype='bool')
                self.data_df['first_buy_close_position_final_conservative'] = self.data_df['buy_close_position_final_conservative'] & (~self.data_df['prev_buy_close_position_final_conservative'])


                self.data_df['prev_sell_close_position_guppy'] = self.data_df['sell_close_position_guppy'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_guppy'] = False
                self.data_df['prev_sell_close_position_guppy'] = pd.Series(list(self.data_df['prev_sell_close_position_guppy']), dtype='bool')
                self.data_df['first_sell_close_position_guppy'] = self.data_df['sell_close_position_guppy'] & (~self.data_df['prev_sell_close_position_guppy'])

                self.data_df['prev_sell_close_position_vegas'] = self.data_df['sell_close_position_vegas'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_vegas'] = False
                self.data_df['prev_sell_close_position_vegas'] = pd.Series(list(self.data_df['prev_sell_close_position_vegas']), dtype='bool')
                self.data_df['first_sell_close_position_vegas'] = self.data_df['sell_close_position_vegas'] & (~self.data_df['prev_sell_close_position_vegas'])

                self.data_df['prev_sell_close_position_final_excessive'] = self.data_df['sell_close_position_final_excessive'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_excessive'] = False
                self.data_df['prev_sell_close_position_final_excessive'] = pd.Series(list(self.data_df['prev_sell_close_position_final_excessive']), dtype='bool')
                self.data_df['first_sell_close_position_final_excessive'] = self.data_df['sell_close_position_final_excessive'] & (~self.data_df['prev_sell_close_position_final_excessive'])

                self.data_df['prev_sell_close_position_final_conservative'] = self.data_df['sell_close_position_final_conservative'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_conservative'] = False
                self.data_df['prev_sell_close_position_final_conservative'] = pd.Series(list(self.data_df['prev_sell_close_position_final_conservative']), dtype='bool')
                self.data_df['first_sell_close_position_final_conservative'] = self.data_df['sell_close_position_final_conservative'] & (~self.data_df['prev_sell_close_position_final_conservative'])


                def select_close_positions(x, guppy, vegas, excessive, conservative,
                                               selected_guppy, selected_vegas, selected_excessive, selected_conservative, exceed_vegas, num_guppy_bars):
                    # print("In select_close_positions:")
                    # print(x)

                    total_guppy = 0
                    total_vegas = 0
                    total_excessive = 0
                    total_conservative = 0
                    for i in range(0, x.shape[0]):
                        row = x.iloc[i]
                        if row[guppy]:
                            if total_guppy <= 1 and total_excessive == 0 and total_conservative == 0:
                                x.at[x.index[i], selected_guppy] = 1
                                total_guppy += 1
                        if row[vegas]:
                            if total_vegas <= 1 and total_excessive == 0 and total_conservative == 0:
                                x.at[x.index[i], selected_vegas] = 1
                                total_vegas += 1
                        if row[excessive]:
                            if total_guppy > 0 or total_vegas > 0 or row[exceed_vegas] > 0 or row[num_guppy_bars] >= 3:
                                x.at[x.index[i], selected_excessive] = 1
                                total_excessive += 1
                        if row[conservative]:
                            if total_guppy == 0 and total_vegas == 0 and total_excessive == 0 and total_conservative == 0:
                                x.at[x.index[i], selected_conservative] = 1
                                total_conservative += 1

                    # if 'sell_point_id' in x.columns:
                    #     y = x.copy()
                    #     y = y.rename(columns = {guppy: 'guppy', vegas : 'vegas', excessive : 'excessive', conservative : 'conservative',
                    #                             selected_guppy : 'selected_guppy', selected_vegas : 'selected_vegas', selected_excessive : 'selected_excessive',
                    #                             selected_conservative : 'selected_conservative'})
                    #
                    #     print("Dig Goup:")
                    #     conditions = reduce(lambda left, right: left | right, [y[col] for col in ['guppy', 'vegas', 'excessive', 'conservative']])
                    #     y = y[conditions]
                    #     print(y)

                    return x


                for side in ['buy', 'sell']:
                    temp_df = self.data_df[['id', 'time', side + '_point_id', 'first_' + side + '_close_position_guppy', 'first_' + side + '_close_position_vegas',
                                            'first_' + side + '_close_position_final_excessive', 'first_' + side + '_close_position_final_conservative',
                                            'num_above_vegas_for_buy', 'num_below_vegas_for_sell',
                                            'num_bar_above_passive_guppy_for_buy', 'num_bar_below_passive_guppy_for_sell']]
                    select_condition = reduce(lambda left, right: left | right, [self.data_df[col] for col in temp_df.columns[2:]])
                    side_df = temp_df[select_condition]
                    side_df = side_df[side_df[side + '_point_id'] > 0]
                    side_df['first_selected_' + side + '_close_position_guppy'] = 0
                    side_df['first_selected_' + side + '_close_position_vegas'] = 0
                    side_df['first_selected_' + side + '_close_position_final_excessive'] = 0
                    side_df['first_selected_' + side + '_close_position_final_conservative'] = 0


                    exceed_vegas = 'num_above_vegas_for_buy' if side == 'buy' else 'num_below_vegas_for_sell'
                    num_guppy_bars = 'num_bar_above_passive_guppy_for_buy' if side == 'buy' else 'num_bar_below_passive_guppy_for_sell'
                    side_df =side_df.groupby([side + '_point_id']).apply(lambda x: select_close_positions(x,
                                                'first_' + side + '_close_position_guppy', 'first_' + side + '_close_position_vegas',
                                                'first_' + side + '_close_position_final_excessive', 'first_' + side + '_close_position_final_conservative',
                                                                                                     'first_selected_' + side + '_close_position_guppy',
                                                                                                     'first_selected_' + side + '_close_position_vegas',
                                                                                                     'first_selected_' + side + '_close_position_final_excessive',
                                                                                                     'first_selected_' + side + '_close_position_final_conservative',
                                                                                                    exceed_vegas,
                                                                                                    num_guppy_bars
                                                                                                     ))
                    temp_df = pd.merge(temp_df, side_df, on = ['id'], how = 'left')
                    temp_df = temp_df.fillna(0)

                    for col in ['first_selected_' + side + '_close_position_guppy','first_selected_' + side + '_close_position_vegas',
                                                                                                     'first_selected_' + side + '_close_position_final_excessive',
                                                                                                     'first_selected_' + side + '_close_position_final_conservative']:
                        self.data_df[col] = np.where(
                            temp_df[col] == 1,
                            True,
                            False
                        )






            ############# Select which close points in the second phase to show ############################
            if True:


                self.data_df['cum_cross_up_vegas'] = self.data_df['actual_cross_up_vegas'].cumsum()
                self.data_df['cum_cross_down_vegas'] = self.data_df['actual_cross_down_vegas'].cumsum()

                cum_cols = ['cum_cross_up_vegas', 'cum_cross_down_vegas']

                for cum_col in cum_cols:
                    self.data_df[cum_col] = self.data_df[cum_col].shift(1)
                    self.data_df.at[0, cum_col] = 0
                    self.data_df[cum_col] = self.data_df[cum_col].astype(int)

                df_buy_point = self.data_df[self.data_df['buy_point_temp'].notnull()][['id', 'cum_cross_up_vegas']]
                df_buy_point.reset_index(inplace=True)
                df_buy_point = df_buy_point.drop(columns=['index'])

                df_sell_point = self.data_df[self.data_df['sell_point_temp'].notnull()][['id', 'cum_cross_down_vegas']]
                df_sell_point.reset_index(inplace = True)
                df_sell_point = df_sell_point.drop(columns = ['index'])


                temp_df = self.data_df[['id', 'buy_point_temp', 'sell_point_temp']]
                temp_df = temp_df.fillna(method = 'ffill').fillna(0) ##########################################################

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)

                temp_df['id'] = temp_df['buy_point_temp']
                temp_df = pd.merge(temp_df, df_buy_point, on = ['id'], how = 'left')

                temp_df = temp_df.rename(columns = {
                    'cum_cross_up_vegas' : 'cum_cross_up_vegas_for_buy'
                })

                temp_df = temp_df.fillna(0)

                temp_df['id'] = temp_df['sell_point_temp']
                temp_df = pd.merge(temp_df, df_sell_point, on=['id'], how='left')

                temp_df = temp_df.rename(columns={
                    'cum_cross_down_vegas': 'cum_cross_down_vegas_for_sell'
                })

                temp_df = temp_df.fillna(0)


                temp_df = temp_df[[col for col in temp_df.columns if 'cum' in col]]

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)

                self.data_df = pd.concat([self.data_df, temp_df], axis = 1)

                self.data_df['num_cross_up_vegas'] = self.data_df['cum_cross_up_vegas'] - self.data_df['cum_cross_up_vegas_for_buy']
                self.data_df['num_cross_down_vegas'] = self.data_df['cum_cross_down_vegas'] - self.data_df['cum_cross_down_vegas_for_sell']


                self.data_df['first_sell_stop_loss_excessive'] = self.data_df['first_sell_stop_loss_excessive'] & (self.data_df['num_cross_down_vegas'] > 0)
                self.data_df['first_buy_stop_loss_excessive'] = self.data_df['first_buy_stop_loss_excessive'] & (self.data_df['num_cross_up_vegas'] > 0)

                self.data_df['first_sell_stop_loss_conservative'] = self.data_df['first_sell_stop_loss_conservative'] & (self.data_df['num_cross_down_vegas'] > 0)
                self.data_df['first_buy_stop_loss_conservative'] = self.data_df['first_buy_stop_loss_conservative'] & (self.data_df['num_cross_up_vegas'] > 0)



                self.data_df['cum_sell_close_position_guppy'] = self.data_df['first_selected_sell_close_position_guppy'].cumsum()
                self.data_df['cum_sell_close_position_vegas'] = self.data_df['first_selected_sell_close_position_vegas'].cumsum()
                self.data_df['cum_sell_close_position_final_excessive'] = self.data_df['first_selected_sell_close_position_final_excessive'].cumsum()
                self.data_df['cum_sell_close_position_final_conservative'] = self.data_df['first_selected_sell_close_position_final_conservative'].cumsum()

                self.data_df['cum_special2_sell_close_position'] = self.data_df['first_actual_special_sell_close_position'].cumsum()
                self.data_df['cum_sell_close_position_excessive'] = self.data_df['first_actual_sell_close_position_excessive'].cumsum()
                self.data_df['cum_sell_close_position_conservative'] = self.data_df['first_actual_sell_close_position_conservative'].cumsum()
                self.data_df['cum_sell_stop_loss_excessive'] = self.data_df['first_sell_stop_loss_excessive'].cumsum()
                self.data_df['cum_sell_stop_loss_conservative'] = self.data_df['first_sell_stop_loss_conservative'].cumsum()


                self.data_df['cum_buy_close_position_guppy'] = self.data_df['first_selected_buy_close_position_guppy'].cumsum()
                self.data_df['cum_buy_close_position_vegas'] = self.data_df['first_selected_buy_close_position_vegas'].cumsum()
                self.data_df['cum_buy_close_position_final_excessive'] = self.data_df['first_selected_buy_close_position_final_excessive'].cumsum()
                self.data_df['cum_buy_close_position_final_conservative'] = self.data_df['first_selected_buy_close_position_final_conservative'].cumsum()

                self.data_df['cum_special2_buy_close_position'] = self.data_df['first_actual_special_buy_close_position'].cumsum()
                self.data_df['cum_buy_close_position_excessive'] = self.data_df['first_actual_buy_close_position_excessive'].cumsum()
                self.data_df['cum_buy_close_position_conservative'] = self.data_df['first_actual_buy_close_position_conservative'].cumsum()
                self.data_df['cum_buy_stop_loss_excessive'] = self.data_df['first_buy_stop_loss_excessive'].cumsum()
                self.data_df['cum_buy_stop_loss_conservative'] = self.data_df['first_buy_stop_loss_conservative'].cumsum()

                cum_sell_close_cols = [
                                       'cum_sell_close_position_guppy', 'cum_sell_close_position_vegas', 'cum_sell_close_position_final_excessive',
                                       'cum_sell_close_position_final_conservative',
                                       'cum_special2_sell_close_position', 'cum_sell_close_position_excessive', 'cum_sell_close_position_conservative',
                                      'cum_sell_stop_loss_excessive', 'cum_sell_stop_loss_conservative']

                cum_buy_close_cols = [
                                      'cum_buy_close_position_guppy', 'cum_buy_close_position_vegas', 'cum_buy_close_position_final_excessive',
                                       'cum_buy_close_position_final_conservative',
                                      'cum_special2_buy_close_position', 'cum_buy_close_position_excessive', 'cum_buy_close_position_conservative',
                                      'cum_buy_stop_loss_excessive', 'cum_buy_stop_loss_conservative']

                cum_columns = cum_sell_close_cols + cum_buy_close_cols

                for cum_col in cum_columns:
                    self.data_df[cum_col] = self.data_df[cum_col].shift(1)
                    self.data_df.at[0, cum_col] = 0
                    self.data_df[cum_col] = self.data_df[cum_col].astype(int)




                df_buy_point = self.data_df[self.data_df['buy_point_temp'].notnull()][['id'] + cum_buy_close_cols]
                df_buy_point.reset_index(inplace = True)
                df_buy_point = df_buy_point.drop(columns = ['index'])

                df_sell_point = self.data_df[self.data_df['sell_point_temp'].notnull()][['id'] + cum_sell_close_cols]
                df_sell_point.reset_index(inplace = True)
                df_sell_point = df_sell_point.drop(columns = ['index'])

                temp_df = self.data_df[['id', 'buy_point_temp', 'sell_point_temp']]
                temp_df = temp_df.fillna(method = 'ffill').fillna(0)

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)

                temp_df['id'] = temp_df['buy_point_temp']
                temp_df = pd.merge(temp_df, df_buy_point, on = ['id'], how = 'left')

                temp_df = temp_df.rename(columns = {
                    'cum_buy_close_position_guppy' : 'cum_buy_close_position_guppy_for_buy',
                    'cum_buy_close_position_vegas' : 'cum_buy_close_position_vegas_for_buy',
                    'cum_buy_close_position_final_excessive' : 'cum_buy_close_position_final_excessive_for_buy',
                    'cum_buy_close_position_final_conservative' : 'cum_buy_close_position_final_conservative_for_buy',
                    'cum_special2_buy_close_position' : 'cum_special2_buy_close_position_for_buy',
                    'cum_buy_close_position_excessive' : 'cum_buy_close_position_excessive_for_buy',
                    'cum_buy_close_position_conservative' : 'cum_buy_close_position_conservative_for_buy',
                    'cum_buy_stop_loss_excessive' : 'cum_buy_stop_loss_excessive_for_buy',
                    'cum_buy_stop_loss_conservative' : 'cum_buy_stop_loss_conservative_for_buy'
                })

                # print("Type 3:")
                # print(type(temp_df['cum_special_buy_close_position_for_buy']))

                temp_df = temp_df.fillna(0)

                temp_df['id'] = temp_df['sell_point_temp']
                temp_df = pd.merge(temp_df, df_sell_point, on = ['id'], how = 'left')
                temp_df = temp_df.rename(columns = {
                    'cum_sell_close_position_guppy' : 'cum_sell_close_position_guppy_for_sell',
                    'cum_sell_close_position_vegas' : 'cum_sell_close_position_vegas_for_sell',
                    'cum_sell_close_position_final_excessive' : 'cum_sell_close_position_final_excessive_for_sell',
                    'cum_sell_close_position_final_conservative' : 'cum_sell_close_position_final_conservative_for_sell',
                    'cum_special2_sell_close_position' : 'cum_special2_sell_close_position_for_sell',
                    'cum_sell_close_position_excessive' : 'cum_sell_close_position_excessive_for_sell',
                    'cum_sell_close_position_conservative' : 'cum_sell_close_position_conservative_for_sell',
                    'cum_sell_stop_loss_excessive' : 'cum_sell_stop_loss_excessive_for_sell',
                    'cum_sell_stop_loss_conservative' : 'cum_sell_stop_loss_conservative_for_sell'
                })
                temp_df = temp_df.fillna(0)

                temp_df = temp_df[[col for col in temp_df.columns if 'cum' in col]]

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)


                self.data_df = pd.concat([self.data_df, temp_df], axis = 1)

                #Continue here

                self.data_df['num_buy_close_position_guppy'] = self.data_df['cum_buy_close_position_guppy'] - self.data_df['cum_buy_close_position_guppy_for_buy']
                self.data_df['num_buy_close_position_vegas'] = self.data_df['cum_buy_close_position_vegas'] - self.data_df['cum_buy_close_position_vegas_for_buy']
                self.data_df['num_buy_close_position_final_excessive'] = self.data_df['cum_buy_close_position_final_excessive'] - self.data_df['cum_buy_close_position_final_excessive_for_buy']
                self.data_df['num_buy_close_position_final_conservative'] = self.data_df['cum_buy_close_position_final_conservative'] - self.data_df['cum_buy_close_position_final_conservative_for_buy']

                self.data_df['num_special_buy_close_position'] = self.data_df['cum_special2_buy_close_position'] - self.data_df['cum_special2_buy_close_position_for_buy']
                self.data_df['num_buy_close_position_excessive'] = self.data_df['cum_buy_close_position_excessive'] - self.data_df['cum_buy_close_position_excessive_for_buy']
                self.data_df['num_buy_close_position_conservative'] = self.data_df['cum_buy_close_position_conservative'] - self.data_df['cum_buy_close_position_conservative_for_buy']
                self.data_df['num_buy_stop_loss_excessive'] = self.data_df['cum_buy_stop_loss_excessive'] - self.data_df['cum_buy_stop_loss_excessive_for_buy']
                self.data_df['num_buy_stop_loss_conservative'] = self.data_df['cum_buy_stop_loss_conservative'] - self.data_df['cum_buy_stop_loss_conservative_for_buy']

                self.data_df['num_temporary_buy_close_position'] = self.data_df['num_special_buy_close_position'] + self.data_df['num_buy_close_position_excessive'] \
                                                                     + self.data_df['num_buy_stop_loss_excessive']
                self.data_df['num_terminal_buy_close_position'] = self.data_df['num_buy_close_position_conservative'] + \
                                                                  self.data_df['num_buy_stop_loss_conservative'] + \
                                                                  self.data_df['num_buy_close_position_final_excessive'] +\
                                                                  self.data_df['num_buy_close_position_final_conservative']




                self.data_df['num_sell_close_position_guppy'] = self.data_df['cum_sell_close_position_guppy'] - self.data_df['cum_sell_close_position_guppy_for_sell']
                self.data_df['num_sell_close_position_vegas'] = self.data_df['cum_sell_close_position_vegas'] - self.data_df['cum_sell_close_position_vegas_for_sell']
                self.data_df['num_sell_close_position_final_excessive'] = self.data_df['cum_sell_close_position_final_excessive'] - self.data_df['cum_sell_close_position_final_excessive_for_sell']
                self.data_df['num_sell_close_position_final_conservative'] = self.data_df['cum_sell_close_position_final_conservative'] - self.data_df['cum_sell_close_position_final_conservative_for_sell']


                self.data_df['num_special_sell_close_position'] = self.data_df['cum_special2_sell_close_position'] - self.data_df['cum_special2_sell_close_position_for_sell']
                self.data_df['num_sell_close_position_excessive'] = self.data_df['cum_sell_close_position_excessive'] - self.data_df['cum_sell_close_position_excessive_for_sell']
                self.data_df['num_sell_close_position_conservative'] = self.data_df['cum_sell_close_position_conservative'] - self.data_df['cum_sell_close_position_conservative_for_sell']
                self.data_df['num_sell_stop_loss_excessive'] = self.data_df['cum_sell_stop_loss_excessive'] - self.data_df['cum_sell_stop_loss_excessive_for_sell']
                self.data_df['num_sell_stop_loss_conservative'] = self.data_df['cum_sell_stop_loss_conservative'] - self.data_df['cum_sell_stop_loss_conservative_for_sell']

                self.data_df['num_temporary_sell_close_position'] = self.data_df['num_special_sell_close_position'] + self.data_df['num_sell_close_position_excessive'] \
                                                                     + self.data_df['num_sell_stop_loss_excessive']
                self.data_df['num_terminal_sell_close_position'] = self.data_df['num_sell_close_position_conservative'] + \
                                                                  self.data_df['num_sell_stop_loss_conservative'] + \
                                                                  self.data_df['num_sell_close_position_final_excessive'] +\
                                                                  self.data_df['num_sell_close_position_final_conservative']




                self.data_df['show_buy_close_position_guppy'] = self.data_df['first_selected_buy_close_position_guppy'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                                # (self.data_df['num_buy_close_position_guppy'] == 0) &\
                                                                # (self.data_df['num_buy_close_position_vegas'] == 0) &\



                self.data_df['show_buy_close_position_vegas'] = self.data_df['first_selected_buy_close_position_vegas'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                                 #(self.data_df['num_buy_close_position_vegas'] <= 1) & \



                self.data_df['show_buy_close_position_final_excessive'] = self.data_df['first_selected_buy_close_position_final_excessive'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                    #((self.data_df['num_buy_close_position_guppy'] > 0) | (self.data_df['num_buy_close_position_vegas'] > 0)) &\



                self.data_df['show_buy_close_position_final_conservative'] = self.data_df['first_selected_buy_close_position_final_conservative'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                                             #((self.data_df['num_buy_close_position_guppy'] == 0) & (self.data_df['num_buy_close_position_vegas'] == 0)) &\




                self.data_df['show_special_buy_close_position'] = self.data_df['first_actual_special_buy_close_position'] & \
                                                                  (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)
                self.data_df['show_buy_close_position_excessive'] = \
                    self.data_df['first_actual_buy_close_position_excessive'] & (self.data_df['num_temporary_buy_close_position'] < 4) &\
                    (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)

                self.data_df['show_buy_close_position_conservative'] = \
                    self.data_df['first_actual_buy_close_position_conservative'] & (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)

                self.data_df['show_buy_stop_loss_excessive'] = \
                    self.data_df['first_buy_stop_loss_excessive'] & (self.data_df['num_temporary_buy_close_position'] < 4) &\
                    (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)

                self.data_df['show_buy_stop_loss_conservative'] = \
                    self.data_df['first_buy_stop_loss_conservative'] & (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)




                self.data_df['show_sell_close_position_guppy'] = self.data_df['first_selected_sell_close_position_guppy'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                                # (self.data_df['num_sell_close_position_guppy'] == 0) &\
                                                                # (self.data_df['num_sell_close_position_vegas'] == 0) &\



                self.data_df['show_sell_close_position_vegas'] = self.data_df['first_selected_sell_close_position_vegas'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                                 #(self.data_df['num_sell_close_position_vegas'] <= 1) & \



                self.data_df['show_sell_close_position_final_excessive'] = self.data_df['first_selected_sell_close_position_final_excessive'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                    #((self.data_df['num_sell_close_position_guppy'] > 0) | (self.data_df['num_sell_close_position_vegas'] > 0)) &\



                self.data_df['show_sell_close_position_final_conservative'] = self.data_df['first_selected_sell_close_position_final_conservative'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                                             #((self.data_df['num_sell_close_position_guppy'] == 0) & (self.data_df['num_sell_close_position_vegas'] == 0)) &\




                self.data_df['show_special_sell_close_position'] = self.data_df['first_actual_special_sell_close_position'] & \
                                                                      (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)
                self.data_df['show_sell_close_position_excessive'] = \
                    self.data_df['first_actual_sell_close_position_excessive'] & (self.data_df['num_temporary_sell_close_position'] < 4) &\
                    (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)

                self.data_df['show_sell_close_position_conservative'] = \
                    self.data_df['first_actual_sell_close_position_conservative'] & (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)

                self.data_df['show_sell_stop_loss_excessive'] = \
                    self.data_df['first_sell_stop_loss_excessive'] & (self.data_df['num_temporary_sell_close_position'] < 4) &\
                    (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)

                self.data_df['show_sell_stop_loss_conservative'] = \
                    self.data_df['first_sell_stop_loss_conservative'] & (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)



                self.data_df = self.data_df.drop(columns = [col for col in self.data_df.columns if 'temp' in col])









            ################################################################################################








            print("to csv:")
            self.data_df.to_csv(os.path.join(self.data_folder, self.currency + str(100) + ".csv"), index=False)
            print("after to csv:")



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





            if self.data_df.iloc[-1]['first_final_buy_fire']: # | self.data_df.iloc[-1]['first_buy_real_fire3']:

                # enter_price = self.data_df.iloc[-1]['close']
                # stop_loss_price = self.data_df.iloc[-1]['lower_vegas']
                #
                # expected_loss = (enter_price - stop_loss_price) * self.lot_size * self.exchange_rate
                #
                # actual_enter_lot = maximum_tolerable_loss / expected_loss * enter_lot
                # if actual_enter_lot > 1:
                #     actual_enter_lot = 1

                msg = "Strongly Long " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close']) #+ " with " + str("%.2f" % actual_enter_lot) + " lot"
                self.log_msg(msg)
                #self.log_msg("enter_price = " + str(enter_price) + " stop_loss_price = " + str(stop_loss_price) + " expected_loss = " + str(expected_loss))

                additional_msg ="" # " Exit if next two bars are both negative" if buy_c2_aux.iloc[-1] else ""

                if additional_msg != "":
                    self.log_msg(additional_msg)
                self.log_msg("********************************")

                #if (self.data_df.iloc[-1]['first_buy_real_fire2'] | self.data_df.iloc[-1]['first_buy_real_fire3']):

                delta_point = (self.data_df.iloc[-1]['close'] - self.data_df.iloc[-1]['period_low' + str(high_low_window_options[0])]) * self.lot_size * self.exchange_rate

                stop_loss_msg = " Stop loss at " + str(delta_point) + " points below open price"
                #else:
                #    stop_loss_msg = ""

                sendEmail(msg, msg + additional_msg + stop_loss_msg) #Temporarily Remove

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


            if self.data_df.iloc[-1]['first_actual_special_sell_close_position']:
                msg = "Close Short Position (Special) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_actual_sell_close_position_excessive']:
                msg = "Close Short Position (Excessive) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_actual_sell_close_position_conservative']:
                msg = "Close Short Position (Conservative) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_sell_stop_loss_excessive']:
                msg = "Close Short Position (Stop loss excessive) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_sell_stop_loss_conservative']:
                msg = "Close Short Position (Stop loss conservative) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)




            if self.data_df.iloc[-1]['sell_weak_ready']:
                if self.data_df.iloc[-1]['sell_ready']:
                    msg = "Ready to short " + self.currency + " at " + current_time + ", last_price = " + str("%.5f" % self.data_df.iloc[-1]['close'])
                    self.log_msg(msg)
                    #sendEmail(msg, msg)
                else:
                    msg = "Ready to weakly short " + self.currency + " at " + current_time + ", last_price = " + str("%.5f" % self.data_df.iloc[-1]['close'])
                    self.log_msg(msg)
                    #sendEmail(msg, msg)







            if self.data_df.iloc[-1]['first_final_sell_fire']: # | self.data_df.iloc[-1]['first_sell_real_fire3']:

                # enter_price = self.data_df.iloc[-1]['close']
                # stop_loss_price = self.data_df.iloc[-1]['upper_vegas']
                #
                # expected_loss = (stop_loss_price - enter_price) * self.lot_size * self.exchange_rate
                #
                # actual_enter_lot = maximum_tolerable_loss / expected_loss * enter_lot
                # if actual_enter_lot > 1:
                #    actual_enter_lot = 1


                msg = "Strongly Short " + self.currency + " at " + current_time + ", last_price = " + str(
                    "%.5f" % self.data_df.iloc[-1]['close'])# + " with " + str("%.2f" % actual_enter_lot) + " lot"
                self.log_msg(msg)
                #self.log_msg("enter_price = " + str(enter_price) + " stop_loss_price = " + str(stop_loss_price) + " expected_loss = " + str(expected_loss))

                additional_msg = "" # " Exit if next two bars are both positive" if sell_c2_aux.iloc[-1] else ""

                if additional_msg != "":
                    self.log_msg(additional_msg)
                self.log_msg("********************************")

                #if (self.data_df.iloc[-1]['first_sell_real_fire2'] | self.data_df.iloc[-1]['first_sell_real_fire3']):

                delta_point = (-self.data_df.iloc[-1]['close'] + self.data_df.iloc[-1]['period_high' + str(high_low_window_options[0])]) * self.lot_size * self.exchange_rate

                stop_loss_msg = " Stop loss at " + str(delta_point) + " points above open price"
                #else:
                #    stop_loss_msg = ""

                sendEmail(msg, msg + additional_msg + stop_loss_msg) #Temporarily Remove

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


            if self.data_df.iloc[-1]['first_actual_special_buy_close_position']:
                msg = "Close Long Position (Special) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_actual_buy_close_position_excessive']:
                msg = "Close Long Position (Excessive) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_actual_buy_close_position_conservative']:
                msg = "Close Long Position (Conservative) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_buy_stop_loss_excessive']:
                msg = "Close Long Position (Stop loss excessive) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)
            elif self.data_df.iloc[-1]['first_buy_stop_loss_conservative']:
                msg = "Close Long Position (Stop loss conservative) for " + self.currency + " at " + current_time
                sendEmail(msg, msg)



            self.log_msg("\n")





            print_prefix = "[Currency " + self.currency + "] "




            all_days = pd.Series(self.data_df['date'].unique()).dt.to_pydatetime()


            # plot_candle_bar_charts(self.currency, self.data_df, all_days,
            #                        num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
            #                        is_plot_candle_buy_sell_points=True,
            #                        print_prefix=print_prefix,
            #                        is_plot_aux = True,
            #                        bar_fig_folder=self.chart_folder, is_plot_simple_chart=False)

            plot_candle_bar_charts(self.currency, self.data_df, all_days,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   is_plot_aux=False,
                                   bar_fig_folder=self.simple_chart_folder, is_plot_simple_chart=True, plot_exclude = is_plot_exclude)


            #self.data_df = self.data_df[['currency', 'time','open','high','low','close']]


            print("Finish")

            self.condition.wait()


        self.condition.release()











