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

reverse_trade_look_back = 20

macd_relaxed = True

price_range_look_back = 10

is_plot_exclude = True

high_low_delta_threshold = 20.001

entry_risk_threshold = 0.6

close_position_look_back = 12

is_send_email = False

use_simple_stop_loss = False

use_quick_stop_loss = True

quick_threshold = 15

is_immediately_in = False

urgent_stop_loss_threshold = 200

#support_half_stop_loss = False

tightened_quick_stop_loss = False

is_apply_innovative_filter_to_fire2 = True
is_reentry = False

is_apply_innovative_filter_to_exclude = False


possition_factor = 0.1


quick_close_position_for_intraday_strategy = False #Default is false   close all position if partial close signal fired for intraday strategy

is_intraday_strategy = False

is_intraday_quick = False  #Close at hours_close_position_quick if price already enters guppy

min_hour_open_position = 5
max_hour_open_position = 18 #18

hours_close_position_quick = [16]
hours_close_position = [0] #23


is_clean_redundant_entry_point = True

is_only_allow_second_entry = True

aligned_conditions21_threshold = 5  #5 by default


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

        self.data_df = self.data_df.iloc[-1000:]
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



    def calculate_position(self, data_df_side, side):
        max_position = 1
        cur_position = 0
        start_position_phase2 = 0

        data_df_side['position'] = 0.0
        for i in range(data_df_side.shape[0]):
            row = data_df_side.iloc[i]

            cum_delta_position = 0.0

            if row['first_final_' + side + '_fire']:
                if cur_position < max_position:
                    start_position_phase2 = 0
                    delta_position = max_position - cur_position
                    delta_position = round(delta_position, 2)
                    data_df_side.at[i, 'position'] = delta_position

                    cum_delta_position += delta_position

                    cur_position += delta_position
                    cur_position = round(cur_position, 2)

                    # print("time = " + str(row['time']) + " fire")
                    # print("delta_position = " + str(delta_position))
                    # print("cur_position = " + str(cur_position))


            if (not row['first_final_' + side + '_fire']) or is_intraday_quick:
                #cum_delta_position = 0.0
                if row['show_' + side + '_close_position_guppy1'] or row['show_' + side + '_close_position_guppy2'] or row['show_' + side + '_close_position_vegas']:
                    if cur_position > 0:
                        delta_position = -cur_position/3.0

                        #New
                        if is_intraday_strategy and quick_close_position_for_intraday_strategy and row['show_' + side + '_close_position_vegas']:
                            delta_position = -cur_position

                        delta_position = round(delta_position, 2)
                        cum_delta_position += delta_position

                        # data_df_side.at[i, 'position'] = delta_position
                        cur_position += delta_position
                        cur_position = round(cur_position, 2)

                        start_position_phase2 = 0

                        # print("time = " + str(row['time']) + " fire")
                        # print("delta_position = " + str(delta_position))
                        # print("cur_position = " + str(cur_position))

                        #print("time = " + str(row['time']) + " phase 1 temporary close")


                if row['show_special_' + side + '_close_position'] or row['show_' + side + '_close_position_excessive'] or row['show_' + side + '_stop_loss_excessive']:
                    if cur_position > 0:
                        if start_position_phase2 <= 0:
                            start_position_phase2 = cur_position

                        if row['show_' + side + '_close_position_excessive_terminal'] or row['show_' + side + '_stop_loss_excessive_terminal']:
                            start_position_phase2 = 0
                            delta_position = -cur_position
                            delta_position = round(delta_position, 2) #Added
                            cum_delta_position += delta_position

                            # data_df_side.at[i, 'position'] = round(delta_position, 2)
                            cur_position += delta_position
                            cur_position = round(cur_position, 2)

                            # print("time = " + str(row['time']) + " fire")
                            # print("delta_position = " + str(delta_position))
                            # print("cur_position = " + str(cur_position))
                        else:
                            if row['show_special_' + side + '_close_position'] and row['show_' + side + '_close_position_excessive']:
                                delta_position = -start_position_phase2/2.0
                            else:
                                delta_position = -start_position_phase2/4.0

                            #New
                            if is_intraday_strategy and quick_close_position_for_intraday_strategy:
                                delta_position = -cur_position


                            delta_position = round(delta_position, 2)

                            cum_delta_position += delta_position

                            # data_df_side.at[i, 'position'] = delta_position
                            cur_position += delta_position
                            cur_position = round(cur_position, 2)

                            # print("time = " + str(row['time']) + " fire")
                            # print("delta_position = " + str(delta_position))
                            # print("cur_position = " + str(cur_position))

                        #print("time = " + str(row['time']) + " phase 2 temporary close")


                if row['show_' + side + '_close_position_final_excessive1'] or row['show_' + side + '_close_position_final_conservative'] or\
                    row['show_' + side + '_close_position_final_quick'] or row['show_' + side + '_close_position_final_urgent'] or\
                    row['show_' + side + '_close_position_conservative'] or row['show_' + side + '_stop_loss_conservative']:

                    if cur_position > 0:
                        start_position_phase2 = 0
                        delta_position = -cur_position
                        delta_position = round(delta_position, 2) #Added
                        cum_delta_position += delta_position

                        # data_df_side.at[i, 'position'] = round(delta_position, 2)
                        cur_position += delta_position
                        cur_position = round(cur_position, 2)

                        # print("time = " + str(row['time']) + " fire")
                        # print("delta_position = " + str(delta_position))
                        # print("cur_position = " + str(cur_position))

                        #print("time = " + str(row['time']) + " close all")

                #'show_sell_close_position_fixed_time_temporary'
                if row['show_' + side + '_close_position_fixed_time_temporary']:
                    if cur_position > 0:
                        assert(row['selected_' + side + '_close_position_fixed_time'] > 0)
                        delta_position = -cur_position/row['selected_' + side + '_close_position_fixed_time']
                        delta_position = round(delta_position, 2)

                        # print("")
                        # print(side + '_close_position_fixed_time_temporary')
                        # print("time = " + str(row['time']))
                        # print("cur_position = " + str(cur_position))
                        # print("denominator = " + str(row['selected_' + side + '_close_position_fixed_time']))
                        # print("delta_position = " + str(delta_position))


                        cum_delta_position += delta_position

                        cur_position += delta_position
                        cur_position = round(cur_position, 2)

                        # print("cum_delta_position = " + str(cum_delta_position))
                        # print("cur_position = " + str(cur_position))

                if row['show_' + side + '_close_position_fixed_time_terminal']:
                    if cur_position > 0:
                        start_position_phase2 = 0
                        delta_position = -cur_position

                        delta_position = round(delta_position, 2)
                        cum_delta_position += delta_position

                        cur_position += delta_position
                        cur_position = round(cur_position, 2)

                        # print("time = " + str(row['time']) + " fire")
                        # print("delta_position = " + str(delta_position))
                        # print("cur_position = " + str(cur_position))


                data_df_side.at[i, 'position'] = round(cum_delta_position, 2)


                #print("delta_position = " + str(cum_delta_position))
                #print("cur_position = " + str(cur_position))


    def calculate_signals(self, high_low_window):

        numerical_features = ['pct_to_upper_vegas', 'high_pct_price_buy', 'low_pct_price_buy', 'pct_to_lower_vegas',
                              'low_pct_price_sell', 'high_pct_price_sell',
                              'ma_close12', 'ma12_gradient']

        bool_features = ['is_above_vegas', 'is_vegas_up_trend', 'is_below_vegas', 'is_vegas_down_trend']

        if True:
            # print("At begging")
            # print("cum_bar_above_guppy_for_buy in data_df already? " + str(
            #     'cum_bar_above_guppy_for_buy' in self.data_df.columns))

            print("Process data_df cut: high_low_window = " + str(high_low_window))
            print(self.data_df[['time','close']].head(10))

            self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()
            self.data_df['hour'] = self.data_df['time'].apply(lambda x: x.hour)

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

            self.data_df['prev_min_price'] = self.data_df['min_price'].shift(1)
            self.data_df['prev_max_price'] = self.data_df['max_price'].shift(1)


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



            self.data_df['prev_open'] = self.data_df['open'].shift(1)
            self.data_df['prev_close'] = self.data_df['close'].shift(1)

            self.data_df['is_positive'] = (self.data_df['close'] > self.data_df['open'])
            self.data_df['is_negative'] = (self.data_df['close'] < self.data_df['open'])

            self.data_df['prev_is_positive'] = self.data_df['is_positive'].shift(1)
            self.data_df['prev_is_negative'] = self.data_df['is_negative'].shift(1)

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
            self.data_df['prev_price_volatility'] = self.data_df['price_volatility'].shift(1)

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
            aligned_long_conditions21 = aligned_long_conditions1[0:2] + [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate > aligned_conditions21_threshold) for guppy_line in guppy_lines[0:3]]
            aligned_long_conditions3 = aligned_long_conditions1 + [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines]
            aligned_long_conditions31 = aligned_long_conditions1 + [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate > 5) for guppy_line in guppy_lines]
            aligned_long_conditions4 = aligned_long_conditions1 + \
                                       [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate > -1) for guppy_line in guppy_lines]
            aligned_long_conditions5 = [(self.data_df[guppy_line + '_gradient'] > 0) for guppy_line in guppy_lines]

            #self.data_df['is_guppy_aligned_long'] = reduce(lambda left, right: left & right, aligned_long_conditions) # + all_up_conditions)
            aligned_long_condition1 = reduce(lambda left, right: left & right, aligned_long_conditions1)
            aligned_long_condition_go_on = reduce(lambda left, right: left & right, aligned_long_conditions_go_on)
            aligned_long_condition2 = reduce(lambda left, right: left & right, aligned_long_conditions2)
            aligned_long_condition21 = reduce(lambda left, right: left & right, aligned_long_conditions21)
            aligned_long_condition3 = reduce(lambda left, right: left & right, aligned_long_conditions3)
            aligned_long_condition31 = reduce(lambda left, right: left & right, aligned_long_conditions31)
            aligned_long_condition4 = reduce(lambda left, right: left & right, aligned_long_conditions4)
            aligned_long_condition5 = reduce(lambda left, right: left & right, aligned_long_conditions5)

            half_aligned_long_condition = reduce(lambda left, right: left & right, aligned_long_conditions1[0:2])
            strongly_half_aligned_long_condition = aligned_long_condition2
            strongly_strict_half_aligned_long_condition = aligned_long_condition21
            strongly_aligned_long_condition = aligned_long_condition3
            strongly_strict_aligned_long_condition = aligned_long_condition31
            strongly_relaxed_aligned_long_condition = aligned_long_condition4
            strongly_long_condition = aligned_long_condition5

            self.data_df['is_guppy_aligned_long'] = aligned_long_condition1 #| aligned_long_condition2
            self.data_df['aligned_long_condition_go_on'] = aligned_long_condition_go_on
            self.data_df['strongly_half_aligned_long_condition'] = strongly_half_aligned_long_condition
            self.data_df['strongly_strict_half_aligned_long_condition'] = strongly_strict_half_aligned_long_condition
            self.data_df['strongly_aligned_long_condition'] = strongly_aligned_long_condition
            self.data_df['strongly_strict_aligned_long_condition'] = strongly_strict_aligned_long_condition
            self.data_df['strongly_relaxed_aligned_long_condition'] = strongly_relaxed_aligned_long_condition
            self.data_df['strongly_long_condition'] = strongly_long_condition

            aligned_short_conditions1 = [(self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]]) for i in
                                        range(len(guppy_lines) - 1)]
            aligned_short_conditions_go_on = [(self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]]) for i in range(3,5)] + \
                                            [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines[4:]]
            #all_down_conditions = [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]
            aligned_short_conditions2 = aligned_short_conditions1[0:2] + [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines[0:3]]
            aligned_short_conditions21 = aligned_short_conditions1[0:2] + [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate < -aligned_conditions21_threshold) for guppy_line in guppy_lines[0:3]]
            aligned_short_conditions3 = aligned_short_conditions1 + [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]
            aligned_short_conditions31 = aligned_short_conditions1 + [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate < -5) for guppy_line in guppy_lines]
            aligned_short_conditions4 = aligned_short_conditions1 + \
                                       [(self.data_df[guppy_line + '_gradient']*self.lot_size*self.exchange_rate < 1) for guppy_line in guppy_lines]
            aligned_short_conditions5 = [(self.data_df[guppy_line + '_gradient'] < 0) for guppy_line in guppy_lines]


            #self.data_df['is_guppy_aligned_short'] = reduce(lambda left, right: left & right, aligned_short_conditions) # + all_down_conditions)
            aligned_short_condition1 = reduce(lambda left, right: left & right, aligned_short_conditions1)
            aligned_short_condition_go_on = reduce(lambda left, right: left & right, aligned_short_conditions_go_on)
            aligned_short_condition2 = reduce(lambda left, right: left & right, aligned_short_conditions2)
            aligned_short_condition21 = reduce(lambda left, right: left & right, aligned_short_conditions21)
            aligned_short_condition3 = reduce(lambda left, right: left & right, aligned_short_conditions3)
            aligned_short_condition31 = reduce(lambda left, right: left & right, aligned_short_conditions31)
            aligned_short_condition4 = reduce(lambda left, right: left & right, aligned_short_conditions4)
            aligned_short_condition5 = reduce(lambda left, right: left & right, aligned_short_conditions5)

            half_aligned_short_condition = reduce(lambda left, right: left & right, aligned_short_conditions1[0:2])
            strongly_half_aligned_short_condition = aligned_short_condition2
            strongly_strict_half_aligned_short_condition = aligned_short_condition21
            strongly_aligned_short_condition = aligned_short_condition3
            strongly_strict_aligned_short_condition = aligned_short_condition31
            strongly_relaxed_aligned_short_condition = aligned_short_condition4
            strongly_short_condition = aligned_short_condition5

            self.data_df['is_guppy_aligned_short'] = aligned_short_condition1 # | aligned_short_condition2
            self.data_df['aligned_short_condition_go_on'] = aligned_short_condition_go_on
            self.data_df['strongly_half_aligned_short_condition'] = strongly_half_aligned_short_condition
            self.data_df['strongly_strict_half_aligned_short_condition'] = strongly_strict_half_aligned_short_condition
            self.data_df['strongly_aligned_short_condition'] = strongly_aligned_short_condition
            self.data_df['strongly_strict_aligned_short_condition'] = strongly_strict_aligned_short_condition
            self.data_df['strongly_relaxed_aligned_short_condition'] = strongly_relaxed_aligned_short_condition
            self.data_df['strongly_short_condition'] = strongly_short_condition

            df_temp = self.data_df[guppy_lines]
            df_temp = df_temp.apply(sorted, axis=1).apply(pd.Series)
            sorted_guppys = ['guppy1', 'guppy2', 'guppy3', 'guppy4', 'guppy5', 'guppy6']
            df_temp.columns = sorted_guppys
            self.data_df = pd.concat([self.data_df, df_temp], axis=1)

            self.data_df['highest_guppy'] = self.data_df['guppy6']
            self.data_df['lowest_guppy'] = self.data_df['guppy1']

            self.data_df['prev_highest_guppy'] = self.data_df['highest_guppy'].shift(1)
            self.data_df['prev_lowest_guppy'] = self.data_df['lowest_guppy'].shift(1)

            self.data_df['second_highest_guppy'] = self.data_df['guppy5']
            self.data_df['second_lowest_guppy'] = self.data_df['guppy2']

            self.data_df['prev1_highest_guppy'] = self.data_df['highest_guppy'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_highest_guppy'] = self.data_df['prev' + str(i-1) + '_highest_guppy'].shift(1)

            self.data_df['prev1_lowest_guppy'] = self.data_df['lowest_guppy'].shift(1)
            for i in range(2, ma12_lookback + 1):
                self.data_df['prev' + str(i) + '_lowest_guppy'] = self.data_df['prev' + str(i-1) + '_lowest_guppy'].shift(1)


            self.data_df['buy_enter_guppy'] = (self.data_df['open'] < self.data_df['lowest_guppy']) & (self.data_df['close'] > self.data_df['lowest_guppy'])
            self.data_df['sell_enter_guppy'] = (self.data_df['open'] > self.data_df['highest_guppy']) & (self.data_df['close'] < self.data_df['highest_guppy'])

            self.data_df['buy_passive_than_guppy'] = self.data_df['is_negative'] & (self.data_df['open'] < self.data_df['lowest_guppy'])
            self.data_df['sell_passive_than_guppy'] = self.data_df['is_positive'] & (self.data_df['open'] > self.data_df['highest_guppy'])


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

            self.data_df['prev_upper_vegas'] = self.data_df['upper_vegas'].shift(1)
            self.data_df['prev_lower_vegas'] = self.data_df['lower_vegas'].shift(1)

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




            self.data_df['period_high' + str(high_low_window) + '_go_up_strict'] = np.where(
                self.data_df['period_high' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate > 0,
                1,
                0
            )
            self.data_df['period_low' + str(high_low_window) + '_go_down_strict'] = np.where(
                self.data_df['period_low' + str(high_low_window) + '_gradient'] * self.lot_size * self.exchange_rate < 0,
                1,
                0
            )





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


            self.data_df['cum_num_high_go_up_strict'] = self.data_df['period_high' + str(high_low_window) + '_go_up_strict'].cumsum()
            self.data_df['cum_num_low_go_down_strict'] = self.data_df['period_low' + str(high_low_window) + '_go_down_strict'].cumsum()


            # print("Debug here:")
            # print(self.data_df.iloc[159:165][['time','period_high100_go_up_strict', 'cum_num_high_go_up_strict']])
            #
            # sys.exit(0)


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

            self.data_df['slowest_guppy_at_top'] = np.abs(self.data_df['ma_close60'] - self.data_df['highest_guppy']) < 1e-5
            self.data_df['slowest_guppy_at_btm'] = np.abs(self.data_df['ma_close60'] - self.data_df['lowest_guppy']) < 1e-5



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
            sell_good_cond3 = (self.data_df['ma12_gradient'] * self.lot_size * self.exchange_rate <= 0)
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
            # sell_bad_cond4 = (~strongly_aligned_short_condition) & ((self.data_df['middle'] < self.data_df['lowest_guppy']) | \
            #                                                       ((self.data_df['close'] < self.data_df['lowest_guppy']) & (strongly_relaxed_aligned_long_condition)))

            sell_bad_cond4 = (~strongly_half_aligned_short_condition) & ((self.data_df['middle'] < self.data_df['lowest_guppy']) | \
                                                                  ((self.data_df['close'] < self.data_df['lowest_guppy']) & (strongly_relaxed_aligned_long_condition)))
            #Modify  for EURUSD


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
            buy_good_cond3 = (self.data_df['ma12_gradient'] * self.lot_size * self.exchange_rate >= 0)
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

            # buy_bad_cond4 = (~strongly_aligned_long_condition) & ((self.data_df['middle'] > self.data_df['highest_guppy']) | \
            #                                                       ((self.data_df['close'] > self.data_df['highest_guppy']) & (strongly_relaxed_aligned_short_condition)))

            buy_bad_cond4 = (~strongly_half_aligned_long_condition) & ((self.data_df['middle'] > self.data_df['highest_guppy']) | \
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
                                                         self.data_df['is_negative'] & (self.data_df['low'] > self.data_df['upper_vegas'])
                                                         #(self.data_df['num_negative_for_buy'] == 0) & \

            self.data_df['special_sell_close_position'] = (self.data_df['num_large_negative_for_sell'] >= 2) & \
                                                          self.data_df['is_positive'] & (self.data_df['high'] < self.data_df['lower_vegas'])
                                                          #(self.data_df['num_positive_for_sell'] == 0) & \




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

        use2TypeSignals = True #True
        filter_option = 1

        while True:

            for high_low_window, data_df_backup in list(zip(high_low_window_options, self.data_dfs_backup)):

                self.data_df = self.data_df[['currency', 'time', 'open', 'high', 'low', 'close']]

                self.calculate_signals(high_low_window)



                if self.is_cut_data and high_low_window == 200:

                    increment_data_df = self.data_df[self.data_df['time'] > data_df_backup.iloc[-1]['time']]
                    if increment_data_df.shape[0] > 0:

                        self.data_df = pd.concat([data_df_backup, increment_data_df])

                        self.data_df.reset_index(inplace = True)
                        self.data_df = self.data_df.drop(columns = ['index'])
                        self.data_df['id'] = list(range(self.data_df.shape[0]))


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

                # print("data_df200 length = " + str(data_df200.shape[0]))
                # print("data_df length = " + str(self.data_df.shape[0]))
                data_df200_use = data_df200.iloc[-self.data_df.shape[0]:]

                # print("data_df200_use length = " + str(data_df200_use.shape[0]))
                #
                # print("Before here:")
                # print('period_high' + str(high_low_window_options[-1]))
                # print('period_high' + str(high_low_window_options[-1]))
                # print(data_df200[['id', 'time', 'period_high200', 'period_low200']])
                #
                # print(data_df200_use[['id', 'time', 'period_high200', 'period_low200']].tail(20))

                data_df200_use.reset_index(inplace=True)
                data_df200_use = data_df200_use.drop(columns=['index'])

                self.data_df['period_high' + str(high_low_window_options[-1])] = data_df200_use['period_high' + str(high_low_window_options[-1])]
                self.data_df['period_low' + str(high_low_window_options[-1])] = data_df200_use['period_low' + str(high_low_window_options[-1])]

                # print("After here:")
                # print(self.data_df[['id', 'time', 'period_high200', 'period_low200']].tail(20))

                self.data_df['macd_period_high' + str(high_low_window_options[-1])] = data_df200_use['macd_period_high' + str(high_low_window_options[-1])]
                self.data_df['macd_period_low' + str(high_low_window_options[-1])] = data_df200_use['macd_period_low' + str(high_low_window_options[-1])]

                # print("last chuck:")
                # print(data_df200[['id','time','period_high200', 'period_low200']].tail(20))
                #


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
                        data_df200_use['final_buy_fire'] = data_df200_use['buy_real_fire3'] | data_df200_use['buy_real_fire2']

                        data_df200_use['final_sell_fire'] = data_df200_use['sell_real_fire3'] | data_df200_use['sell_real_fire2']
                    else:
                        data_df200_use['final_buy_fire'] = data_df200_use['buy_real_fire3']

                        data_df200_use['final_sell_fire'] = data_df200_use['sell_real_fire3']

                    if filter_option == 1:

                        #self.data_df['final_buy_fire_exclude'] = data_df100['final_buy_fire'] & (~data_df200['final_buy_fire']) & data_df100['strongly_aligned_short_condition']
                        self.data_df['final_buy_fire_exclude'] = data_df100['buy_real_fire3'] & (~data_df200_use['buy_real_fire3']) & data_df100['strongly_aligned_short_condition']

                        # print("Checking bug here:")
                        # print("data_df100:")
                        # print(data_df100[['time','id','buy_real_fire3']].tail(70))
                        # print("data_df200:")
                        # print(data_df200_use[['time','id','buy_real_fire3']].tail(70))

                        if not is_apply_innovative_filter_to_exclude:
                            self.data_df['final_buy_fire'] = data_df100['final_buy_fire'] & (~self.data_df['final_buy_fire_exclude'])

                        #self.data_df['final_sell_fire_exclude'] = data_df100['final_sell_fire'] & (~data_df200['final_sell_fire']) & data_df100['strongly_aligned_long_condition']
                        self.data_df['final_sell_fire_exclude'] = data_df100['sell_real_fire3'] & (~data_df200_use['sell_real_fire3']) & data_df100['strongly_aligned_long_condition']

                        if not is_apply_innovative_filter_to_exclude:
                            self.data_df['final_sell_fire'] = data_df100['final_sell_fire'] & (~self.data_df['final_sell_fire_exclude'])

                        # print("data_df:")
                        # print(self.data_df[['time','id','final_buy_fire_exclude', 'final_buy_fire']].tail(70))
                        #
                        # sys.exit(0)


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

            # print("At here1")
            # print("cum_bar_above_guppy_for_buy in data_df already? " + str(
            #     'cum_bar_above_guppy_for_buy' in self.data_df.columns))

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



            if is_intraday_strategy:

                self.data_df['first_final_buy_fire'] = np.where(
                    (self.data_df['hour'] >= min_hour_open_position) & (self.data_df['hour'] <= max_hour_open_position),
                    self.data_df['first_final_buy_fire'],
                    False
                )

                self.data_df['first_final_sell_fire'] = np.where(
                    (self.data_df['hour'] >= min_hour_open_position) & (self.data_df['hour'] <= max_hour_open_position),
                    self.data_df['first_final_sell_fire'],
                    False
                )



#########################################################################################################################################

            if is_clean_redundant_entry_point:
                self.data_df['buy_point'] = np.where(
                    self.data_df['first_final_buy_fire'],
                    1,
                    0
                )
                self.data_df['sell_point'] = np.where(
                    self.data_df['first_final_sell_fire'],
                    1,
                    0
                )

                self.data_df['bar_below_m12'] = self.data_df['max_price'] < self.data_df['ma_close12']
                self.data_df['bar_above_m12'] = self.data_df['min_price'] > self.data_df['ma_close12']

                self.data_df['cum_bar_below_m12'] = self.data_df['bar_below_m12'].cumsum()
                self.data_df['cum_bar_above_m12'] = self.data_df['bar_above_m12'].cumsum()

                self.data_df['prev_cum_bar_below_m12'] = self.data_df['cum_bar_below_m12'].shift(1)
                self.data_df['prev_cum_bar_above_m12'] = self.data_df['cum_bar_above_m12'].shift(1)

                for cum_col in ['prev_cum_bar_below_m12', 'prev_cum_bar_above_m12']:
                    self.data_df.at[0, cum_col] = 0
                    self.data_df[cum_col] = self.data_df[cum_col].astype(int)

                temp_clean_df = self.data_df[['time', 'id', 'buy_point', 'sell_point',
                                              'cum_bar_below_m12', 'prev_cum_bar_below_m12', 'cum_bar_above_m12', 'prev_cum_bar_above_m12', 'close']]

                temp_clean_df['prev_cum_bar_below_m12_for_buy'] = np.where(
                    temp_clean_df['buy_point'] == 1,
                    temp_clean_df['prev_cum_bar_below_m12'],
                    np.nan
                )

                temp_clean_df['prev_cum_bar_above_m12_for_sell'] = np.where(
                    temp_clean_df['sell_point'] == 1,
                    temp_clean_df['prev_cum_bar_above_m12'],
                    np.nan
                )

                temp_clean_df['buy_price'] = np.where(
                    temp_clean_df['buy_point'] == 1,
                    temp_clean_df['close'],
                    np.nan
                )

                temp_clean_df['sell_price'] = np.where(
                    temp_clean_df['sell_point'] == 1,
                    temp_clean_df['close'],
                    np.nan
                )

                temp_clean_df = temp_clean_df.fillna(method = 'ffill').fillna(0)

                temp_clean_df['num_bar_below_m12_for_buy'] = temp_clean_df['cum_bar_below_m12'] - temp_clean_df['prev_cum_bar_below_m12_for_buy']
                temp_clean_df['num_bar_above_m12_for_sell'] = temp_clean_df['cum_bar_above_m12'] - temp_clean_df['prev_cum_bar_above_m12_for_sell']

                self.data_df['num_bar_below_m12_for_buy'] = temp_clean_df['num_bar_below_m12_for_buy']
                self.data_df['num_bar_above_m12_for_sell'] = temp_clean_df['num_bar_above_m12_for_sell']

                self.data_df['prev_num_bar_below_m12_for_buy'] = self.data_df['num_bar_below_m12_for_buy'].shift(1)
                self.data_df['prev_num_bar_above_m12_for_sell'] = self.data_df['num_bar_above_m12_for_sell'].shift(1)

                self.data_df['buy_price'] = temp_clean_df['buy_price']
                self.data_df['prev_buy_price'] = temp_clean_df['buy_price'].shift(1)

                self.data_df['sell_price'] = temp_clean_df['sell_price']
                self.data_df['prev_sell_price'] = temp_clean_df['sell_price'].shift(1)



                self.data_df['first_final_buy_fire'] = self.data_df['first_final_buy_fire'] &\
                        (self.data_df['bar_below_m12'] | (self.data_df['buy_price'] < self.data_df['prev_buy_price']) | (self.data_df['prev_num_bar_below_m12_for_buy'] > 0))

                self.data_df['first_final_sell_fire'] = self.data_df['first_final_sell_fire'] &\
                        (self.data_df['bar_above_m12'] | (self.data_df['sell_price'] > self.data_df['prev_sell_price']) | (self.data_df['prev_num_bar_above_m12_for_sell'] > 0))



            if is_only_allow_second_entry:
                # self.data_df['buy_point'] = np.where(
                #     (self.data_df['first_final_buy_fire'] | self.data_df['first_final_buy_fire_exclude']) & (~(self.data_df['buy_real_fire2'] & (~self.data_df['buy_real_fire3']))),
                #     1,
                #     0
                # )
                # self.data_df['sell_point'] = np.where(
                #     (self.data_df['first_final_sell_fire'] | self.data_df['first_final_sell_fire_exclude']) & (~(self.data_df['buy_real_fire2'] & (~self.data_df['buy_real_fire3']))),
                #     1,
                #     0
                # )


                self.data_df['buy_point'] = np.where(
                    (self.data_df['first_final_buy_fire']) & (~(self.data_df['buy_real_fire2'] & (~self.data_df['buy_real_fire3']))),
                    1,
                    0
                )
                self.data_df['sell_point'] = np.where(
                    (self.data_df['first_final_sell_fire']) & (~(self.data_df['sell_real_fire2'] & (~self.data_df['sell_real_fire3']))),
                    1,
                    0
                )


                self.data_df['buy_point_backup'] = self.data_df['buy_point']
                self.data_df['sell_point_backup'] = self.data_df['sell_point']

                self.data_df['bar_cross_up_m12'] = self.data_df['is_positive'] & (self.data_df['close'] > self.data_df['ma_close12'])
                self.data_df['bar_cross_down_m12'] = self.data_df['is_negative'] & (self.data_df['close'] < self.data_df['ma_close12'])

                self.data_df['ma12_up'] = self.data_df['ma12_gradient'] * self.exchange_rate * self.lot_size > 0
                self.data_df['ma12_down'] = self.data_df['ma12_gradient'] * self.exchange_rate * self.lot_size < 0

                self.data_df['bar_partial_below_ma12'] = self.data_df['max_price'] < self.data_df['ma_close12']  #middle
                self.data_df['bar_partial_above_ma12'] = self.data_df['min_price'] > self.data_df['ma_close12']  #middle

                self.data_df['bar_above_all_guppy'] = (self.data_df['max_price'] > self.data_df['highest_guppy']) & self.data_df['is_positive']
                self.data_df['bar_below_all_guppy'] = (self.data_df['min_price'] < self.data_df['lowest_guppy']) & self.data_df['is_negative']

                self.data_df['bar_above_vegas'] = (self.data_df['low'] > self.data_df['upper_vegas'])  #max_price
                self.data_df['bar_below_vegas'] = (self.data_df['high'] < self.data_df['lower_vegas']) #min_price


                self.data_df['cum_ma12_up'] = self.data_df['ma12_up'].cumsum()
                self.data_df['cum_ma12_down'] = self.data_df['ma12_down'].cumsum()

                self.data_df['cum_bar_partial_below_ma12'] = self.data_df['bar_partial_below_ma12'].cumsum()
                self.data_df['cum_bar_partial_above_ma12'] = self.data_df['bar_partial_above_ma12'].cumsum()

                self.data_df['cum_bar_above_all_guppy'] = self.data_df['bar_above_all_guppy'].cumsum()
                self.data_df['cum_bar_below_all_guppy'] = self.data_df['bar_below_all_guppy'].cumsum()

                self.data_df['cum_bar_above_vegas'] = self.data_df['bar_above_vegas'].cumsum()
                self.data_df['cum_bar_below_vegas'] = self.data_df['bar_below_vegas'].cumsum()


                self.data_df['prev_cum_ma12_up'] = self.data_df['cum_ma12_up'].shift(1)
                self.data_df['prev_cum_ma12_down'] = self.data_df['cum_ma12_down'].shift(1)

                self.data_df['prev_cum_bar_partial_below_ma12'] = self.data_df['cum_bar_partial_below_ma12'].shift(1)
                self.data_df['prev_cum_bar_partial_above_ma12'] = self.data_df['cum_bar_partial_above_ma12'].shift(1)

                self.data_df['prev_cum_bar_above_all_guppy'] = self.data_df['cum_bar_above_all_guppy'].shift(1)
                self.data_df['prev_cum_bar_below_all_guppy'] = self.data_df['cum_bar_below_all_guppy'].shift(1)

                self.data_df['prev_cum_bar_above_vegas'] = self.data_df['cum_bar_above_vegas'].shift(1)
                self.data_df['prev_cum_bar_below_vegas'] = self.data_df['cum_bar_below_vegas'].shift(1)


                for cum_col in ['prev_cum_ma12_up', 'prev_cum_ma12_down', 'prev_cum_bar_partial_below_ma12', 'prev_cum_bar_partial_above_ma12',
                                'prev_cum_bar_above_all_guppy', 'prev_cum_bar_below_all_guppy', 'prev_cum_bar_above_vegas', 'prev_cum_bar_below_vegas']:
                    self.data_df.at[0, cum_col] = 0
                    self.data_df[cum_col] = self.data_df[cum_col].astype(int)


                temp_adjust_df = self.data_df[['time', 'id', 'buy_point', 'sell_point', 'open', 'close',
                                               'cum_ma12_up', 'cum_ma12_down',
                                               'prev_cum_ma12_up', 'prev_cum_ma12_down',
                                               'cum_bar_partial_below_ma12', 'cum_bar_partial_above_ma12',
                                               'prev_cum_bar_partial_below_ma12', 'prev_cum_bar_partial_above_ma12',
                                               'cum_bar_above_all_guppy', 'cum_bar_below_all_guppy',
                                               'cum_bar_above_vegas', 'cum_bar_below_vegas',
                                               'prev_cum_bar_above_all_guppy', 'prev_cum_bar_below_all_guppy',
                                               'prev_cum_bar_above_vegas', 'prev_cum_bar_below_vegas'
                                               ]]


                temp_adjust_df['prev_cum_ma12_down_for_buy'] = np.where(
                    temp_adjust_df['buy_point'] == 1,
                    temp_adjust_df['prev_cum_ma12_down'],
                    np.nan
                )

                temp_adjust_df['prev_cum_ma12_up_for_sell'] = np.where(
                    temp_adjust_df['sell_point'] == 1,
                    temp_adjust_df['prev_cum_ma12_up'],
                    np.nan
                )

                temp_adjust_df['prev_cum_bar_partial_below_ma12_for_buy'] = np.where(
                    temp_adjust_df['buy_point'] == 1,
                    temp_adjust_df['prev_cum_bar_partial_below_ma12'],
                    np.nan
                )

                temp_adjust_df['prev_cum_bar_partial_above_ma12_for_sell'] = np.where(
                    temp_adjust_df['sell_point'] == 1,
                    temp_adjust_df['prev_cum_bar_partial_above_ma12'],
                    np.nan
                )

                temp_adjust_df['prev_cum_bar_above_all_guppy_for_buy'] = np.where(
                    temp_adjust_df['buy_point'] == 1,
                    temp_adjust_df['prev_cum_bar_above_all_guppy'],
                    np.nan
                )

                temp_adjust_df['prev_cum_bar_below_all_guppy_for_sell'] = np.where(
                    temp_adjust_df['sell_point'] == 1,
                    temp_adjust_df['prev_cum_bar_below_all_guppy'],
                    np.nan
                )

                temp_adjust_df['prev_cum_bar_above_vegas_for_buy'] = np.where(
                    temp_adjust_df['buy_point'] == 1,
                    temp_adjust_df['prev_cum_bar_above_vegas'],
                    np.nan
                )

                temp_adjust_df['prev_cum_bar_below_vegas_for_sell'] = np.where(
                    temp_adjust_df['sell_point'] == 1,
                    temp_adjust_df['prev_cum_bar_below_vegas'],
                    np.nan
                )




                temp_adjust_df['buy_price'] = np.where(
                    temp_adjust_df['buy_point'] == 1,
                    temp_adjust_df['close'],
                    np.nan
                )

                temp_adjust_df['buy_open_price'] = np.where(
                    temp_adjust_df['buy_point'] == 1,
                    temp_adjust_df['open'],
                    np.nan
                )

                temp_adjust_df['sell_price'] = np.where(
                    temp_adjust_df['sell_point'] == 1,
                    temp_adjust_df['close'],
                    np.nan
                )

                temp_adjust_df['sell_open_price'] = np.where(
                    temp_adjust_df['sell_point'] == 1,
                    temp_adjust_df['open'],
                    np.nan
                )

                temp_adjust_df['start_buy_id'] = np.where(
                    temp_adjust_df['buy_point'] == 1,
                    temp_adjust_df['id'],
                    np.nan
                )

                temp_adjust_df['start_sell_id'] = np.where(
                    temp_adjust_df['sell_point'] == 1,
                    temp_adjust_df['id'],
                    np.nan
                )

                temp_adjust_df = temp_adjust_df.fillna(method = 'ffill').fillna(0)

                temp_adjust_df['num_ma12_down_for_buy'] = temp_adjust_df['cum_ma12_down'] - temp_adjust_df['prev_cum_ma12_down_for_buy']
                temp_adjust_df['num_ma12_up_for_sell'] = temp_adjust_df['cum_ma12_up'] - temp_adjust_df['prev_cum_ma12_up_for_sell']

                temp_adjust_df['num_bar_partial_below_ma12_for_buy'] = temp_adjust_df['cum_bar_partial_below_ma12'] - temp_adjust_df['prev_cum_bar_partial_below_ma12_for_buy']
                temp_adjust_df['num_bar_partial_above_ma12_for_sell'] = temp_adjust_df['cum_bar_partial_above_ma12'] - temp_adjust_df['prev_cum_bar_partial_above_ma12_for_sell']

                temp_adjust_df['num_bar_above_all_guppy_for_buy'] = temp_adjust_df['cum_bar_above_all_guppy'] - temp_adjust_df['prev_cum_bar_above_all_guppy_for_buy']
                temp_adjust_df['num_bar_below_all_guppy_for_sell'] = temp_adjust_df['cum_bar_below_all_guppy'] - temp_adjust_df['prev_cum_bar_below_all_guppy_for_sell']

                temp_adjust_df['num_bar_above_vegas_for_buy'] = temp_adjust_df['cum_bar_above_vegas'] - temp_adjust_df['prev_cum_bar_above_vegas_for_buy']
                temp_adjust_df['num_bar_below_vegas_for_sell'] = temp_adjust_df['cum_bar_below_vegas'] - temp_adjust_df['prev_cum_bar_below_vegas_for_sell']




                temp_adjust_df['num_bars_since_last_buy'] = temp_adjust_df['id'] - temp_adjust_df['start_buy_id'] + 1
                temp_adjust_df['num_bars_since_last_sell'] = temp_adjust_df['id'] - temp_adjust_df['start_sell_id'] + 1



                self.data_df['num_ma12_down_for_buy'] = temp_adjust_df['num_ma12_down_for_buy']
                self.data_df['num_ma12_up_for_sell'] = temp_adjust_df['num_ma12_up_for_sell']

                self.data_df['num_bar_partial_below_ma12_for_buy'] = temp_adjust_df['num_bar_partial_below_ma12_for_buy']
                self.data_df['num_bar_partial_above_ma12_for_sell'] = temp_adjust_df['num_bar_partial_above_ma12_for_sell']

                self.data_df['num_bar_above_all_guppy_for_buy'] = temp_adjust_df['num_bar_above_all_guppy_for_buy']
                self.data_df['num_bar_below_all_guppy_for_sell'] = temp_adjust_df['num_bar_below_all_guppy_for_sell']

                self.data_df['num_bar_above_vegas_for_buy'] = temp_adjust_df['num_bar_above_vegas_for_buy']
                self.data_df['num_bar_below_vegas_for_sell'] = temp_adjust_df['num_bar_below_vegas_for_sell']


                self.data_df['prev_num_ma12_down_for_buy'] = self.data_df['num_ma12_down_for_buy'].shift(1)
                self.data_df['prev_num_ma12_up_for_sell'] = self.data_df['num_ma12_up_for_sell'].shift(1)

                self.data_df['prev_num_bar_partial_below_ma12_for_buy'] = self.data_df['num_bar_partial_below_ma12_for_buy'].shift(1)
                self.data_df['prev_num_bar_partial_above_ma12_for_sell'] = self.data_df['num_bar_partial_above_ma12_for_sell'].shift(1)

                self.data_df['prev_num_bar_above_all_guppy_for_buy'] = self.data_df['num_bar_above_all_guppy_for_buy'].shift(1)
                self.data_df['prev_num_bar_below_all_guppy_for_sell'] = self.data_df['num_bar_below_all_guppy_for_sell'].shift(1)


                self.data_df['prev_num_bar_above_vegas_for_buy'] = self.data_df['num_bar_above_vegas_for_buy'].shift(1)
                self.data_df['prev_num_bar_below_vegas_for_sell'] = self.data_df['num_bar_below_vegas_for_sell'].shift(1)




                self.data_df['buy_price'] = temp_adjust_df['buy_price']
                self.data_df['prev_buy_price'] = temp_adjust_df['buy_price'].shift(1)

                self.data_df['sell_price'] = temp_adjust_df['sell_price']
                self.data_df['prev_sell_price'] = temp_adjust_df['sell_price'].shift(1)

                self.data_df['buy_open_price'] = temp_adjust_df['buy_open_price']
                self.data_df['prev_buy_open_price'] = temp_adjust_df['buy_open_price'].shift(1)

                self.data_df['sell_open_price'] = temp_adjust_df['sell_open_price']
                self.data_df['prev_sell_open_price'] = temp_adjust_df['sell_open_price'].shift(1)


                self.data_df['num_bars_since_last_buy'] = temp_adjust_df['num_bars_since_last_buy']
                self.data_df['num_bars_since_last_sell'] = temp_adjust_df['num_bars_since_last_sell']

                self.data_df['prev_num_bars_since_last_buy'] = self.data_df['num_bars_since_last_buy'].shift(1)
                self.data_df['prev_num_bars_since_last_sell'] = self.data_df['num_bars_since_last_sell'].shift(1)

                self.data_df['buy_point_number'] = self.data_df['buy_point'].cumsum()
                self.data_df['sell_point_number'] = self.data_df['sell_point'].cumsum()



                # self.data_df['first_final_buy_fire_new'] = (self.data_df['buy_point_number'] > 0) & self.data_df['bar_cross_up_m12'] & self.data_df['ma12_up'] &\
                #             (self.data_df['prev_num_ma12_down_for_buy'] > 0) & (self.data_df['prev_num_bar_partial_below_ma12_for_buy'] > 0) &\
                #             (self.data_df['prev_num_bars_since_last_buy'] >= 5) & (self.data_df['prev_num_bars_since_last_buy'] <= 48) &\
                #             ((self.data_df['prev_num_bar_above_all_guppy_for_buy'] == 0) | (self.data_df['close'] < self.data_df['prev_buy_price'])) &\
                #             (((self.data_df['close'] - self.data_df['prev_buy_price'])*self.exchange_rate*self.lot_size > -400) | (self.data_df['prev_num_bars_since_last_buy'] <= 24)) &\
                #             ((self.data_df['open'] - self.data_df['prev_buy_open_price'])*self.exchange_rate*self.lot_size < 300)

                self.data_df['buy_new_cond1'] = self.data_df['bar_cross_up_m12']
                self.data_df['buy_new_cond2'] = self.data_df['ma12_up']
                self.data_df['buy_new_cond3'] = self.data_df['prev_num_ma12_down_for_buy'] > 0
                self.data_df['buy_new_cond4'] = self.data_df['prev_num_bar_partial_below_ma12_for_buy'] > 0
                self.data_df['buy_new_cond5'] = self.data_df['prev_num_bars_since_last_buy'] >= 4 #5
                self.data_df['buy_new_cond6'] = self.data_df['prev_num_bars_since_last_buy'] <= 48
                self.data_df['buy_new_cond71'] = (self.data_df['prev_num_bar_above_all_guppy_for_buy'] < 2) | \
                                                ((self.data_df['close'] < self.data_df['highest_guppy']) & (self.data_df['prev_num_bar_above_vegas_for_buy'] == 0)) #'prev_buy_price'  ‘==0’  'lowest_guppy'
                self.data_df['buy_new_cond72'] = (self.data_df['prev_num_bar_above_vegas_for_buy'] == 0) & (self.data_df['first_final_buy_fire'])
                self.data_df['buy_new_cond7'] = self.data_df['buy_new_cond71'] | self.data_df['buy_new_cond72']

                self.data_df['buy_new_cond8'] = ((self.data_df['close'] - self.data_df['prev_buy_price'])*self.exchange_rate*self.lot_size > -400) | (self.data_df['prev_num_bars_since_last_buy'] <= 30) #24
                self.data_df['buy_new_cond9'] = (self.data_df['open'] - self.data_df['prev_buy_open_price'])*self.exchange_rate*self.lot_size < 300
                self.data_df['buy_new_cond10'] = True #(self.data_df['close'] < self.data_df['guppy6']) #'guppy3'  'guppy6'
                self.data_df['buy_new_cond11'] = (self.data_df['ma_close12'] < self.data_df['lower_vegas']) &\
                                                 ((self.data_df['close'] < self.data_df['lower_vegas'])) # | (self.data_df['open'] < self.data_df['lowest_guppy']))

                self.data_df['first_final_buy_fire_new'] = (self.data_df['buy_point_number'] > 0) &\
                    self.data_df['buy_new_cond1'] & self.data_df['buy_new_cond2'] & self.data_df['buy_new_cond3'] & self.data_df['buy_new_cond4'] & self.data_df['buy_new_cond5'] & \
                    self.data_df['buy_new_cond6'] & self.data_df['buy_new_cond7'] & self.data_df['buy_new_cond8'] & self.data_df['buy_new_cond9'] & self.data_df['buy_new_cond10'] & \
                    self.data_df['buy_new_cond11']


                # self.data_df['first_final_sell_fire_new'] = (self.data_df['sell_point_number'] > 0) & self.data_df['bar_cross_down_m12'] & self.data_df['ma12_down'] &\
                #             (self.data_df['prev_num_ma12_up_for_sell'] > 0) & (self.data_df['prev_num_bar_partial_above_ma12_for_sell'] > 0) &\
                #             (self.data_df['prev_num_bars_since_last_sell'] >= 5) & (self.data_df['prev_num_bars_since_last_sell'] <= 48) &\
                #             ((self.data_df['prev_num_bar_below_all_guppy_for_sell'] == 0) | (self.data_df['close'] > self.data_df['prev_sell_price'])) &\
                #             (((self.data_df['close'] - self.data_df['prev_sell_price'])*self.exchange_rate*self.lot_size < 400) | (self.data_df['prev_num_bars_since_last_sell'] <= 24)) &\
                #             ((self.data_df['open'] - self.data_df['prev_sell_open_price'])*self.exchange_rate*self.lot_size > -300)

                self.data_df['sell_new_cond1'] = self.data_df['bar_cross_down_m12']
                self.data_df['sell_new_cond2'] = self.data_df['ma12_down']
                self.data_df['sell_new_cond3'] = self.data_df['prev_num_ma12_up_for_sell'] > 0
                self.data_df['sell_new_cond4'] = self.data_df['prev_num_bar_partial_above_ma12_for_sell'] > 0
                self.data_df['sell_new_cond5'] = self.data_df['prev_num_bars_since_last_sell'] >= 4 #5
                self.data_df['sell_new_cond6'] = self.data_df['prev_num_bars_since_last_sell'] <= 48
                self.data_df['sell_new_cond71'] = (self.data_df['prev_num_bar_below_all_guppy_for_sell'] < 2) | \
                                                  ((self.data_df['close'] > self.data_df['lowest_guppy']) & (self.data_df['prev_num_bar_below_vegas_for_sell'] == 0)) #'prev_sell_price' 'highest_guppy'
                self.data_df['sell_new_cond72'] = (self.data_df['prev_num_bar_below_vegas_for_sell'] == 0) & (self.data_df['first_final_sell_fire'])
                self.data_df['sell_new_cond7'] = self.data_df['sell_new_cond71'] | self.data_df['sell_new_cond72']

                self.data_df['sell_new_cond8'] = ((self.data_df['close'] - self.data_df['prev_sell_price'])*self.exchange_rate*self.lot_size < 400) | (self.data_df['prev_num_bars_since_last_sell'] <= 30) #24
                self.data_df['sell_new_cond9'] = (self.data_df['open'] - self.data_df['prev_sell_open_price'])*self.exchange_rate*self.lot_size > -300
                self.data_df['sell_new_cond10'] = True #(self.data_df['close'] > self.data_df['guppy1']) #'guppy4' 'guppy2'
                self.data_df['sell_new_cond11'] = (self.data_df['ma_close12'] > self.data_df['upper_vegas']) &\
                                                  ((self.data_df['close'] > self.data_df['upper_vegas'])) # | (self.data_df['open'] > self.data_df['highest_guppy']))

                self.data_df['first_final_sell_fire_new'] = (self.data_df['sell_point_number'] > 0) &\
                    self.data_df['sell_new_cond1'] & self.data_df['sell_new_cond2'] & self.data_df['sell_new_cond3'] & self.data_df['sell_new_cond4'] & self.data_df['sell_new_cond5'] & \
                    self.data_df['sell_new_cond6'] & self.data_df['sell_new_cond7'] & self.data_df['sell_new_cond8'] & self.data_df['sell_new_cond9'] & self.data_df['sell_new_cond10'] & \
                    self.data_df['sell_new_cond11']

#########################################################################################################################################









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
                temp_df = self.data_df[['id', 'time',  'buy_point', 'sell_point', 'buy_point_id', 'sell_point_id', 'close',
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
                    if col != 'close' and col != 'time':
                        temp_df[col] = temp_df[col].astype(int)


                buy_df = temp_df[temp_df['buy_point'] == 1]
                sell_df = temp_df[temp_df['sell_point'] == 1]

                # print("buy_df:")
                # print(buy_df)
                #
                # print("sell_df:")
                # print(sell_df)



                def calc_cum_min(x):
                    # print("Group buy")
                    # print(x)

                    x['group_min_price'] = x['close'].cummin()
                    return x

                def calc_cum_max(x):

                    # print("Group sell")
                    # print(x)

                    x['group_max_price'] = x['close'].cummax()
                    return x

                if buy_df.shape[0] > 0:
                    buy_df = buy_df.groupby(['buy_group']).apply(lambda x: calc_cum_min(x))
                else:
                    buy_df['group_min_price'] = 0

                if sell_df.shape[0] > 0:
                    sell_df = sell_df.groupby(['sell_group']).apply(lambda x: calc_cum_max(x))
                else:
                    sell_df['group_max_price'] = 0

                # print("sell_df2:")
                # print(sell_df)

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

                temp_df['buy_point_price'] = np.nan
                temp_df['buy_point_price'] = np.where(
                    self.data_df['buy_point'] == 1,
                    self.data_df['close'],
                    temp_df['buy_point_price']
                )

                temp_df['is_buy_fire2'] = np.nan
                temp_df['is_buy_fire2'] = np.where(
                    self.data_df['buy_point'] == 1,
                    np.where(
                        self.data_df['buy_real_fire2'],
                        1,
                        0
                    ),
                    temp_df['is_buy_fire2']
                )

                temp_df['is_buy_fire_exclude'] = np.nan
                temp_df['is_buy_fire_exclude'] = np.where(
                    self.data_df['buy_point'] == 1,
                    np.where(
                        self.data_df['final_buy_fire_exclude'],
                        1,
                        0
                    ),
                    temp_df['is_buy_fire_exclude']
                )


                temp_df['sell_point_support'] = np.nan
                temp_df['sell_point_support'] = np.where(
                    self.data_df['sell_point'] == 1,
                    self.data_df['period_high' + str(high_low_window)],
                    temp_df['sell_point_support']
                )

                temp_df['sell_point_price'] = np.nan
                temp_df['sell_point_price'] = np.where(
                    self.data_df['sell_point'] == 1,
                    self.data_df['close'],
                    temp_df['sell_point_price']
                )

                temp_df['is_sell_fire2'] = np.nan
                temp_df['is_sell_fire2'] = np.where(
                    self.data_df['sell_point'] == 1,
                    np.where(
                        self.data_df['sell_real_fire2'],
                        1,
                        0
                    ),
                    temp_df['is_sell_fire2']
                )

                temp_df['is_sell_fire_exclude'] = np.nan
                temp_df['is_sell_fire_exclude'] = np.where(
                    self.data_df['sell_point'] == 1,
                    np.where(
                        self.data_df['final_sell_fire_exclude'],
                        1,
                        0
                    ),
                    temp_df['is_sell_fire_exclude']
                )



                temp_df = temp_df.fillna(method='ffill').fillna(0)


                # for col in temp_df.columns:
                #     temp_df[col] = temp_df[col].astype(int)

                self.data_df['buy_point_support'] = temp_df['buy_point_support']
                self.data_df['sell_point_support'] = temp_df['sell_point_support']
                self.data_df['buy_point_price'] = temp_df['buy_point_price']
                self.data_df['sell_point_price'] = temp_df['sell_point_price']
                self.data_df['is_buy_fire2'] = temp_df['is_buy_fire2']
                self.data_df['is_sell_fire2'] = temp_df['is_sell_fire2']

                self.data_df['is_buy_fire_exclude'] = temp_df['is_buy_fire_exclude']
                self.data_df['is_sell_fire_exclude'] = temp_df['is_sell_fire_exclude']



                ########################

                self.data_df['is_above_vegas_tolerate'] = (self.data_df['ma_close12'] - self.data_df['upper_vegas']) * self.lot_size * self.exchange_rate > 20
                self.data_df['is_below_vegas_tolerate'] = (self.data_df['ma_close12'] - self.data_df['lower_vegas']) * self.lot_size * self.exchange_rate < -20

                self.data_df['bar_above_guppy'] = self.data_df['min_price'] > self.data_df['highest_guppy']
                self.data_df['bar_below_guppy'] = self.data_df['max_price'] < self.data_df['lowest_guppy']

                self.data_df['bar_strict_above_guppy'] = self.data_df['low'] > self.data_df['highest_guppy']
                self.data_df['bar_strict_below_guppy'] = self.data_df['high'] < self.data_df['lowest_guppy']

                self.data_df['bar_above_passive_guppy'] = (self.data_df['low'] > self.data_df['lowest_guppy']) & (self.data_df['high'] > self.data_df['guppy4']) #Modify
                self.data_df['bar_below_passive_guppy'] = (self.data_df['high'] < self.data_df['highest_guppy']) & (self.data_df['low'] < self.data_df['guppy3'])


                self.data_df['is_reach_vegas_from_btm'] = (self.data_df['ma_close12'] < self.data_df['lower_vegas']) & (self.data_df['high'] > self.data_df['lower_vegas'])
                self.data_df['is_reach_vegas_from_top'] = (self.data_df['ma_close12'] > self.data_df['upper_vegas']) & (self.data_df['low'] < self.data_df['upper_vegas'])

                self.data_df['is_trapped_in_short_guppy'] = self.data_df['bar_above_passive_guppy'] & self.data_df['aligned_short_condition_go_on']
                self.data_df['is_trapped_in_long_guppy'] = self.data_df['bar_below_passive_guppy'] & self.data_df['aligned_long_condition_go_on']


                self.data_df['cum_strongly_strict_half_aligned_short_condition'] = self.data_df['strongly_half_aligned_short_condition'].cumsum()
                self.data_df['cum_strongly_strict_half_aligned_long_condition'] = self.data_df['strongly_half_aligned_long_condition'].cumsum()

                self.data_df['cum_aligned_short_condition_go_on'] = self.data_df['aligned_short_condition_go_on'].cumsum()
                self.data_df['cum_aligned_long_condition_go_on'] = self.data_df['aligned_long_condition_go_on'].cumsum()



                self.data_df['cum_bar_above_guppy'] = self.data_df['bar_above_guppy'].cumsum()
                self.data_df['cum_bar_below_guppy'] = self.data_df['bar_below_guppy'].cumsum()

                self.data_df['cum_bar_strict_above_guppy'] = self.data_df['bar_strict_above_guppy'].cumsum()
                self.data_df['cum_bar_strict_below_guppy'] = self.data_df['bar_strict_below_guppy'].cumsum()

                self.data_df['cum_bar_above_passive_guppy'] = self.data_df['bar_above_passive_guppy'].cumsum()
                self.data_df['cum_bar_below_passive_guppy'] = self.data_df['bar_below_passive_guppy'].cumsum()

                # self.data_df['bar_above_vegas'] = (self.data_df['min'] - self.data_df['upper_vegas']) * self.lot_size * self.exchange_rate >= 300
                # self.data_df['bar_below_vegas'] = (self.data_df['max'] - self.data_df['lower_vegas']) * self.lot_size * self.exchange_rate <= -300

                # self.data_df['cum_bar_above_vegas'] = self.data_df['bar_above_vegas'].cumsum()
                # self.data_df['cum_bar_below_vegas'] = self.data_df['bar_below_vegas'].cumsum()

                self.data_df['cum_above_vegas'] = self.data_df['is_above_vegas_tolerate'].cumsum()
                self.data_df['cum_below_vegas'] = self.data_df['is_below_vegas_tolerate'].cumsum()


                self.data_df['cum_is_reach_vegas_from_btm'] = self.data_df['is_reach_vegas_from_btm'].cumsum()
                self.data_df['cum_is_reach_vegas_from_top'] = self.data_df['is_reach_vegas_from_top'].cumsum()

                self.data_df['cum_is_trapped_in_short_guppy'] = self.data_df['is_trapped_in_short_guppy'].cumsum()
                self.data_df['cum_is_trapped_in_long_guppy'] = self.data_df['is_trapped_in_long_guppy'].cumsum()


                cum_buy_cols = ['cum_strongly_strict_half_aligned_short_condition', 'cum_aligned_short_condition_go_on',
                                'cum_bar_above_guppy', 'cum_bar_strict_above_guppy', 'cum_bar_above_passive_guppy',  'cum_above_vegas', 'cum_num_low_go_down_strict',
                                'cum_is_reach_vegas_from_btm', 'cum_is_trapped_in_short_guppy']

                cum_sell_cols = ['cum_strongly_strict_half_aligned_long_condition', 'cum_aligned_long_condition_go_on',
                                 'cum_bar_below_guppy', 'cum_bar_strict_below_guppy', 'cum_bar_below_passive_guppy',  'cum_below_vegas', 'cum_num_high_go_up_strict',
                                 'cum_is_reach_vegas_from_top', 'cum_is_trapped_in_long_guppy']

                for cum_col in cum_buy_cols + cum_sell_cols:
                    if 'strict' not in cum_col or 'go' not in cum_col:
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

                temp_df['buy_point_duration'] = temp_df['id'] - temp_df['buy_point_temp'] + 1
                temp_df['sell_point_duration'] = temp_df['id'] - temp_df['sell_point_temp'] + 1

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)


                temp_df['id'] = temp_df['buy_point_temp']
                temp_df = pd.merge(temp_df, df_buy, on = ['id'], how = 'left')

                #print("type here1: " + str(type(temp_df['cum_bar_above_guppy'])))
                temp_df = temp_df.rename(columns = {
                    'cum_strongly_strict_half_aligned_short_condition' : 'cum_strongly_strict_half_aligned_short_condition_for_buy',
                    'cum_aligned_short_condition_go_on' : 'cum_aligned_short_condition_go_on_for_buy',
                    'cum_bar_above_guppy' : 'cum_bar_above_guppy_for_buy',
                    'cum_bar_strict_above_guppy': 'cum_bar_strict_above_guppy_for_buy',
                    'cum_bar_above_passive_guppy' : 'cum_bar_above_passive_guppy_for_buy',
                    'cum_above_vegas' : 'cum_above_vegas_for_buy',
                    'cum_num_low_go_down_strict' : 'cum_num_low_go_down_strict_for_buy',
                    'cum_is_reach_vegas_from_btm' : 'cum_is_reach_vegas_from_btm_for_buy',
                    'cum_is_trapped_in_short_guppy' : 'cum_is_trapped_in_short_guppy_for_buy'
                })
                temp_df = temp_df.fillna(0)

                #print("type here2: " + str(type(temp_df['cum_bar_above_guppy_for_buy'])))


                temp_df['id'] = temp_df['sell_point_temp']
                temp_df = pd.merge(temp_df, df_sell, on = ['id'], how = 'left')
                temp_df = temp_df.rename(columns = {
                    'cum_strongly_strict_half_aligned_long_condition' : 'cum_strongly_strict_half_aligned_long_condition_for_sell',
                    'cum_aligned_long_condition_go_on' : 'cum_aligned_long_condition_go_on_for_sell',
                    'cum_bar_below_guppy' : 'cum_bar_below_guppy_for_sell',
                    'cum_bar_strict_below_guppy': 'cum_bar_strict_below_guppy_for_sell',
                    'cum_bar_below_passive_guppy' : 'cum_bar_below_passive_guppy_for_sell',
                    'cum_below_vegas' : 'cum_below_vegas_for_sell',
                    'cum_num_high_go_up_strict' : 'cum_num_high_go_up_strict_for_sell',
                    'cum_is_reach_vegas_from_top' : 'cum_is_reach_vegas_from_top_for_sell',
                    'cum_is_trapped_in_long_guppy' : 'cum_is_trapped_in_long_guppy_for_sell'
                })
                temp_df = temp_df.fillna(0)

                temp_df = temp_df[[col for col in temp_df.columns if 'cum' in col or 'duration' in col]]

                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].astype(int)

                # print("type here3: " + str(type(temp_df['cum_bar_above_guppy_for_buy'])))
                #
                # print("cum_bar_above_guppy_for_buy in data_df already? " + str('cum_bar_above_guppy_for_buy' in self.data_df.columns))

                self.data_df = pd.concat([self.data_df, temp_df], axis = 1)

                # print("Fucking Hutong")
                # print(self.data_df[['id','time','close','cum_bar_above_guppy','cum_bar_above_guppy_for_buy']].tail(20))
                # print("type 1: " + str(type(self.data_df['cum_bar_above_guppy'])))
                # print("type 2: " + str(type(self.data_df['cum_bar_above_guppy_for_buy'])))

                self.data_df['num_strongly_strict_half_aligned_short_condition_for_buy'] =\
                    self.data_df['cum_strongly_strict_half_aligned_short_condition'] - self.data_df['cum_strongly_strict_half_aligned_short_condition_for_buy']
                self.data_df['num_aligned_short_condition_go_on_for_buy'] =\
                    self.data_df['cum_aligned_short_condition_go_on'] - self.data_df['cum_aligned_short_condition_go_on_for_buy']

                self.data_df['num_bar_above_guppy_for_buy'] = self.data_df['cum_bar_above_guppy'] - self.data_df['cum_bar_above_guppy_for_buy']
                self.data_df['num_bar_strict_above_guppy_for_buy'] = self.data_df['cum_bar_strict_above_guppy'] - self.data_df['cum_bar_strict_above_guppy_for_buy']
                self.data_df['num_bar_above_passive_guppy_for_buy'] = self.data_df['cum_bar_above_passive_guppy'] - self.data_df['cum_bar_above_passive_guppy_for_buy']
                self.data_df['num_above_vegas_for_buy'] = self.data_df['cum_above_vegas'] - self.data_df['cum_above_vegas_for_buy']
                self.data_df['num_low_go_down_strict_for_buy'] = self.data_df['cum_num_low_go_down_strict'] - self.data_df['cum_num_low_go_down_strict_for_buy']
                self.data_df['num_is_reach_vegas_from_btm_for_buy'] = self.data_df['cum_is_reach_vegas_from_btm'] - self.data_df['cum_is_reach_vegas_from_btm_for_buy']
                self.data_df['num_is_trapped_in_short_guppy'] = self.data_df['cum_is_trapped_in_short_guppy'] - self.data_df['cum_is_trapped_in_short_guppy_for_buy']


                self.data_df['num_strongly_strict_half_aligned_long_condition_for_sell'] =\
                    self.data_df['cum_strongly_strict_half_aligned_long_condition'] - self.data_df['cum_strongly_strict_half_aligned_long_condition_for_sell']
                self.data_df['num_aligned_long_condition_go_on_for_sell'] =\
                    self.data_df['cum_aligned_long_condition_go_on'] - self.data_df['cum_aligned_long_condition_go_on_for_sell']


                self.data_df['num_bar_below_guppy_for_sell'] = self.data_df['cum_bar_below_guppy'] - self.data_df['cum_bar_below_guppy_for_sell']
                self.data_df['num_bar_strict_below_guppy_for_sell'] = self.data_df['cum_bar_strict_below_guppy'] - self.data_df['cum_bar_strict_below_guppy_for_sell']
                self.data_df['num_bar_below_passive_guppy_for_sell'] = self.data_df['cum_bar_below_passive_guppy'] - self.data_df['cum_bar_below_passive_guppy_for_sell']
                self.data_df['num_below_vegas_for_sell'] = self.data_df['cum_below_vegas'] - self.data_df['cum_below_vegas_for_sell']
                self.data_df['num_high_go_up_strict_for_sell'] = self.data_df['cum_num_high_go_up_strict'] - self.data_df['cum_num_high_go_up_strict_for_sell']
                self.data_df['num_is_reach_vegas_from_top_for_sell'] = self.data_df['cum_is_reach_vegas_from_top'] - self.data_df['cum_is_reach_vegas_from_top_for_sell']
                self.data_df['num_is_trapped_in_long_guppy'] = self.data_df['cum_is_trapped_in_long_guppy'] - self.data_df['cum_is_trapped_in_long_guppy_for_sell']


                #Critical
                self.data_df['buy_close_position_guppy1'] = (self.data_df['open'] > self.data_df['highest_guppy']) &\
                                                           (self.data_df['ma_close12'] < self.data_df['lower_vegas']) &\
                                                           (self.data_df['close'] < self.data_df['highest_guppy']) &\
                                                           (self.data_df['num_bar_above_guppy_for_buy'] > 1) &\
                                                           (~(self.data_df['strongly_half_aligned_long_condition'] & self.data_df['fastest_guppy_at_top']))
                                                            #(self.data_df['num_above_vegas_for_buy'] == 0)

                self.data_df['buy_close_position_guppy2'] = self.data_df['is_negative'] & self.data_df['prev_is_positive'] & \
                                                            (self.data_df['close'] < self.data_df['lower_vegas']) &\
                                                            (self.data_df['price_range']/self.data_df['price_volatility'] > 0.5) &\
                                                            (self.data_df['prev_price_range']/self.data_df['prev_price_volatility'] > 0.5) &\
                                                            (self.data_df['min_price'] < self.data_df['prev_max_price'] - 0.55 * self.data_df['prev_price_range']) &\
                                                             (self.data_df['min_price'] > self.data_df['highest_guppy']) &\
                                                             (self.data_df['prev_min_price'] > self.data_df['prev_highest_guppy'])
                                                            #(self.data_df['num_above_vegas_for_buy'] == 0)



                #self.data_df['buy_close_position_guppy'] = self.data_df['buy_close_position_guppy1'] | self.data_df['buy_close_position_guppy2']

                self.data_df['sell_close_position_guppy1'] = (self.data_df['open'] < self.data_df['lowest_guppy']) & \
                                                            (self.data_df['ma_close12'] > self.data_df['upper_vegas']) & \
                                                            (self.data_df['close'] > self.data_df['lowest_guppy']) & \
                                                            (self.data_df['num_bar_below_guppy_for_sell'] > 1) & \
                                                            (~(self.data_df['strongly_half_aligned_short_condition'] & self.data_df['fastest_guppy_at_btm']))
                                                             #(self.data_df['num_below_vegas_for_sell'] == 0)

                self.data_df['sell_close_position_guppy2'] = self.data_df['is_positive'] & (self.data_df['prev_open'] > self.data_df['prev_close']) & \
                                                            (self.data_df['close'] > self.data_df['upper_vegas']) & \
                                                            (self.data_df['price_range']/self.data_df['price_volatility'] > 0.5) &\
                                                            (self.data_df['prev_price_range']/self.data_df['prev_price_volatility'] > 0.5) &\
                                                            (self.data_df['max_price'] > self.data_df['prev_min_price'] + 0.55 * self.data_df['prev_price_range']) &\
                                                             (self.data_df['max_price'] < self.data_df['lowest_guppy']) &\
                                                             (self.data_df['prev_max_price'] < self.data_df['prev_lowest_guppy'])
                                                             #(self.data_df['num_below_vegas_for_sell'] == 0)

                #self.data_df['sell_close_position_guppy'] = self.data_df['sell_close_position_guppy1'] | self.data_df['sell_close_position_guppy2']


                self.data_df['buy_close_position_vegas'] = (self.data_df['is_negative']) & \
                                                           ((self.data_df['close'] - self.data_df['lower_vegas'])*self.lot_size*self.exchange_rate < -20) &\
                                                           ((self.data_df['high'] > self.data_df['lower_vegas']) | (self.data_df['prev_high'] > self.data_df['lower_vegas'])) &\
                                                           (self.data_df['num_above_vegas_for_buy'] == 0) #&\
                                    #((self.data_df['max_price_to_lower_vegas']/self.data_df['price_range'] < 0.4) | (self.data_df['prev_max_price_to_lower_vegas']/self.data_df['prev_price_range'] < 0.4))


                self.data_df['sell_close_position_vegas'] = (self.data_df['is_positive']) & \
                                                           ((self.data_df['close'] - self.data_df['upper_vegas'])*self.lot_size*self.exchange_rate > 20) &\
                                                           ((self.data_df['low'] < self.data_df['upper_vegas']) | (self.data_df['prev_low'] < self.data_df['upper_vegas'])) & \
                                                            (self.data_df['num_below_vegas_for_sell'] == 0) #&\
                                    #((self.data_df['min_price_to_upper_vegas']/self.data_df['price_range'] < 0.4) | (self.data_df['prev_min_price_to_upper_vegas']/self.data_df['prev_price_range'] < 0.4))



                #Singapore
                self.data_df['buy_close_position_final_excessive1'] = (self.data_df['close'] - self.data_df['group_min_price'])*self.lot_size*self.exchange_rate < -150
                self.data_df['sell_close_position_final_excessive1'] = (self.data_df['close'] - self.data_df['group_max_price'])*self.lot_size*self.exchange_rate > 150

                self.data_df['buy_close_position_final_excessive2'] = (self.data_df['close'] - self.data_df['buy_point_support'])*self.lot_size*self.exchange_rate < -10
                self.data_df['sell_close_position_final_excessive2'] = (self.data_df['close'] - self.data_df['sell_point_support'])*self.lot_size*self.exchange_rate > 10

                self.data_df['buy_close_position_final_excessive_strict'] = (self.data_df['close'] - self.data_df['group_min_price'])*self.lot_size*self.exchange_rate < -600
                self.data_df['sell_close_position_final_excessive_strict'] = (self.data_df['close'] - self.data_df['group_max_price'])*self.lot_size*self.exchange_rate > 600


                self.data_df['buy_close_position_final_conservative'] = (self.data_df['close'] - self.data_df['buy_point_support'])*self.lot_size*self.exchange_rate < -300
                self.data_df['sell_close_position_final_conservative'] = (self.data_df['close'] - self.data_df['sell_point_support'])*self.lot_size*self.exchange_rate > 300

                self.data_df['buy_close_position_final_conservative_strict'] = (self.data_df['close'] - self.data_df['buy_point_support'])*self.lot_size*self.exchange_rate < -600
                self.data_df['sell_close_position_final_conservative_strict'] = (self.data_df['close'] - self.data_df['sell_point_support'])*self.lot_size*self.exchange_rate > 600



                self.data_df['buy_close_position_final_simple'] = self.data_df['num_low_go_down_strict_for_buy'] > 1
                self.data_df['sell_close_position_final_simple'] = self.data_df['num_high_go_up_strict_for_sell'] > 1



                # self.data_df['buy_close_position_final_quick'] = self.data_df['is_negative'] &\
                #                                                   (self.data_df['open'] > self.data_df['lowest_guppy']) & (self.data_df['close'] < self.data_df['lowest_guppy']) &\
                #                                                    (self.data_df['num_is_reach_vegas_from_btm_for_buy'] == 0) &\
                #                                                   (self.data_df['num_is_trapped_in_short_guppy'] > 0)
                #
                # self.data_df['sell_close_position_final_quick'] = self.data_df['is_positive'] &\
                #                                                   (self.data_df['open'] < self.data_df['highest_guppy']) & (self.data_df['close'] > self.data_df['highest_guppy']) &\
                #                                                    (self.data_df['num_is_reach_vegas_from_top_for_sell'] == 0) &\
                #                                                   (self.data_df['num_is_trapped_in_long_guppy'] > 0)


                ##Modify
                self.data_df['buy_close_position_final_quick1'] = self.data_df['is_negative'] &\
                                                                  (self.data_df['open'] < self.data_df['highest_guppy']) &\
                                            (self.data_df['num_is_trapped_in_short_guppy'] > 0) & self.data_df['slowest_guppy_at_top'] &\
                ((self.data_df['close'] - self.data_df['lowest_guppy'])*self.lot_size*self.exchange_rate < -quick_threshold) &\
                ((self.data_df['open'] - self.data_df['lowest_guppy'])*self.lot_size*self.exchange_rate > 0) &\
                (self.data_df['num_is_reach_vegas_from_btm_for_buy'] == 0) & (self.data_df['num_bar_strict_above_guppy_for_buy'] == 0)

                self.data_df['buy_close_position_final_quick21'] = self.data_df['is_negative'] &\
                              ((self.data_df['close'] - self.data_df['lowest_guppy'])*self.lot_size*self.exchange_rate < -quick_threshold) &\
                              self.data_df['strongly_strict_half_aligned_short_condition']  &\
                             ((self.data_df['open'] - self.data_df['lowest_guppy'])*self.lot_size*self.exchange_rate > -1)  #Change here

                #### Not used ########
                #  self.data_df['is_guppy_aligned_short']
                # self.data_df['buy_close_position_final_quick22'] = self.data_df['is_negative'] & self.data_df['strongly_strict_half_aligned_short_condition'] &\
                #                                 (self.data_df['close'] < self.data_df['prev_min_price']) & (self.data_df['open'] > self.data_df['prev_max_price'])
                ######################




                self.data_df['buy_close_position_final_quick2_ready'] = self.data_df['is_negative'] &\
                            (self.data_df['strongly_aligned_short_condition']) & (self.data_df['max_price'] < self.data_df['ma_close12']) &\
                             (self.data_df['ma12_gradient']*self.exchange_rate*self.lot_size < -0) & (self.data_df['ma_close12'] < self.data_df['lowest_guppy'])
                self.data_df['prev_buy_close_position_final_quick2_ready'] = self.data_df['buy_close_position_final_quick2_ready'].shift(1)
                self.data_df['buy_close_position_final_quick22'] = self.data_df['buy_close_position_final_quick2_ready'] & self.data_df['prev_buy_close_position_final_quick2_ready']



                self.data_df['buy_close_position_final_quick22_on'] = False

                if is_apply_innovative_filter_to_fire2:
                    self.data_df['buy_close_position_final_quick22_on'] = self.data_df['buy_close_position_final_quick22_on'] | self.data_df['is_buy_fire2']

                if is_apply_innovative_filter_to_exclude:
                    self.data_df['buy_close_position_final_quick22_on'] = self.data_df['buy_close_position_final_quick22_on'] | self.data_df['is_buy_fire_exclude']

                self.data_df['buy_close_position_final_quick22'] = self.data_df['buy_close_position_final_quick22'] & self.data_df['buy_close_position_final_quick22_on']



                self.data_df['buy_close_position_final_quick2'] = self.data_df['buy_close_position_final_quick21'] | self.data_df['buy_close_position_final_quick22']


                self.data_df['buy_close_position_final_quick_immediate'] = self.data_df['buy_close_position_final_quick22']




                self.data_df['buy_close_position_final_quick'] = (self.data_df['buy_close_position_final_quick1'] | self.data_df['buy_close_position_final_quick2']) #&\
                                                                 #(~self.data_df['is_needle_bar'])
                #((self.data_df['open'] - self.data_df['lowest_guppy'])*self.lot_size*self.exchange_rate > quick_threshold) &\

                self.data_df['buy_close_position_fixed_time'] = 0
                self.data_df['sell_close_position_fixed_time'] = 0

                self.data_df['buy_close_position_quick_fixed_time'] = 0
                self.data_df['sell_close_position_quick_fixed_time'] = 0

                if is_intraday_strategy:
                    for i in range(len(hours_close_position)):
                        close_time = hours_close_position[i]
                        self.data_df['buy_close_position_fixed_time'] = np.where(
                            self.data_df['hour'] == close_time,
                            len(hours_close_position) - i,
                            self.data_df['buy_close_position_fixed_time']
                        )

                    for i in range(len(hours_close_position)):
                        close_time = hours_close_position[i]
                        self.data_df['sell_close_position_fixed_time'] = np.where(
                            self.data_df['hour'] == close_time,
                            len(hours_close_position) - i,
                            self.data_df['sell_close_position_fixed_time']
                        )

                    if is_intraday_quick:
                        for i in range(len(hours_close_position_quick)):
                            close_time = hours_close_position_quick[i]
                            self.data_df['buy_close_position_quick_fixed_time'] = np.where(
                                self.data_df['hour'] == close_time,
                                len(hours_close_position_quick) - i,
                                self.data_df['buy_close_position_quick_fixed_time']
                            )

                        for i in range(len(hours_close_position_quick)):
                            close_time = hours_close_position_quick[i]
                            self.data_df['sell_close_position_quick_fixed_time'] = np.where(
                                self.data_df['hour'] == close_time,
                                len(hours_close_position_quick) - i,
                                self.data_df['sell_close_position_quick_fixed_time']
                            )



                self.data_df['sell_close_position_final_quick1'] = self.data_df['is_positive'] &\
                                                                   (self.data_df['open'] > self.data_df['lowest_guppy']) &\
                                           (self.data_df['num_is_trapped_in_long_guppy'] > 0) & self.data_df['slowest_guppy_at_btm'] &\
                ((self.data_df['close'] - self.data_df['highest_guppy'])*self.lot_size*self.exchange_rate > quick_threshold) &\
                ((self.data_df['open'] - self.data_df['highest_guppy'])*self.lot_size*self.exchange_rate < -0) &\
                (self.data_df['num_is_reach_vegas_from_top_for_sell'] == 0) & (self.data_df['num_bar_strict_below_guppy_for_sell'] == 0)

                self.data_df['sell_close_position_final_quick21'] = self.data_df['is_positive'] &\
                              ((self.data_df['close'] - self.data_df['highest_guppy'])*self.lot_size*self.exchange_rate > quick_threshold) &\
                              self.data_df['strongly_strict_half_aligned_long_condition'] &\
                            ((self.data_df['open'] - self.data_df['highest_guppy'])*self.lot_size*self.exchange_rate < 1) #Change here

                #### Not used ###########
                # self.data_df['is_guppy_aligned_long']
                # self.data_df['sell_close_position_final_quick22'] = self.data_df['is_positive'] & self.data_df['strongly_strict_half_aligned_long_condition'] &\
                #                                 (self.data_df['close'] > self.data_df['prev_max_price']) & (self.data_df['open'] < self.data_df['prev_min_price'])
                #########################


                self.data_df['sell_close_position_final_quick2_ready'] = self.data_df['is_positive'] &\
                            (self.data_df['strongly_aligned_long_condition']) & (self.data_df['min_price'] > self.data_df['ma_close12']) &\
                            (self.data_df['ma12_gradient']*self.exchange_rate*self.lot_size > 0) & (self.data_df['ma_close12'] > self.data_df['highest_guppy'])
                self.data_df['prev_sell_close_position_final_quick2_ready'] = self.data_df['sell_close_position_final_quick2_ready'].shift(1)
                self.data_df['sell_close_position_final_quick22'] = self.data_df['sell_close_position_final_quick2_ready'] & self.data_df['prev_sell_close_position_final_quick2_ready']


                self.data_df['sell_close_position_final_quick22_on'] = False

                if is_apply_innovative_filter_to_fire2:
                    self.data_df['sell_close_position_final_quick22_on'] = self.data_df['sell_close_position_final_quick22_on'] | self.data_df['is_sell_fire2']

                if is_apply_innovative_filter_to_exclude:
                    self.data_df['sell_close_position_final_quick22_on'] = self.data_df['sell_close_position_final_quick22_on'] | self.data_df['is_sell_fire_exclude']

                self.data_df['sell_close_position_final_quick22'] = self.data_df['sell_close_position_final_quick22'] & self.data_df['sell_close_position_final_quick22_on']



                self.data_df['sell_close_position_final_quick2'] = self.data_df['sell_close_position_final_quick21'] | self.data_df['sell_close_position_final_quick22']


                self.data_df['sell_close_position_final_quick_immediate'] = self.data_df['sell_close_position_final_quick22']


                self.data_df['sell_close_position_final_quick'] = (self.data_df['sell_close_position_final_quick1'] | self.data_df['sell_close_position_final_quick2'])
                                                                 #(~self.data_df['is_needle_bar'])





                self.data_df['buy_close_position_final_urgent'] = self.data_df['is_negative'] &\
                                ((self.data_df['close'] - self.data_df['buy_point_support'])*self.exchange_rate*self.lot_size < -urgent_stop_loss_threshold) &\
                            (((self.data_df['num_strongly_strict_half_aligned_short_condition_for_buy']/(self.data_df['buy_point_duration']-1) > 0.95)
                             & self.data_df['strongly_half_aligned_short_condition'])
                             |\
                             ((self.data_df['num_aligned_short_condition_go_on_for_buy']/self.data_df['buy_point_duration'] > 0.95)
                             & self.data_df['aligned_short_condition_go_on'])
                             )

                self.data_df['sell_close_position_final_urgent'] = self.data_df['is_positive'] &\
                                ((self.data_df['close'] - self.data_df['sell_point_support'])*self.exchange_rate*self.lot_size > urgent_stop_loss_threshold) &\
                            (((self.data_df['num_strongly_strict_half_aligned_long_condition_for_sell']/(self.data_df['sell_point_duration']-1) > 0.95)
                             & self.data_df['strongly_half_aligned_long_condition'])
                             |\
                             ((self.data_df['num_aligned_long_condition_go_on_for_sell']/self.data_df['sell_point_duration'] > 0.95)
                             & self.data_df['aligned_long_condition_go_on'])
                             )

                #
                # self.data_df['sell_close_position_final_urgent'] = self.data_df['is_positive'] & (self.data_df['close'] > self.data_df['sell_point_support']) &\
                #             ((self.data_df['num_strongly_strict_half_aligned_long_condition_for_sell']/(self.data_df['sell_point_duration']-1) > 0.95) |\
                #              (self.data_df['num_aligned_long_condition_go_on_for_sell']/self.data_df['sell_point_duration'] > 0.95))








                # print("Debug Df:")
                # print(self.data_df.iloc[815:825][['id', 'time', 'close','group_max_price','sell_close_position_final_excessive']])

                #sys.exit(0)


                # self.data_df['prev_buy_close_position_final_excessive_ready'] = self.data_df['buy_close_position_final_excessive_ready'].shift(1)
                # self.data_df.at[0, 'prev_buy_close_position_final_excessive_ready'] = False
                # self.data_df['prev_buy_close_position_final_excessive_ready'] = pd.Series(list(self.data_df['prev_buy_close_position_final_excessive_ready']), dtype = 'bool')
                #
                # self.data_df['prev2_buy_close_position_final_excessive_ready'] = self.data_df['prev_buy_close_position_final_excessive_ready'].shift(1)
                # self.data_df.at[0, 'prev2_buy_close_position_final_excessive_ready'] = False
                # self.data_df['prev2_buy_close_position_final_excessive_ready'] = pd.Series(list(self.data_df['prev2_buy_close_position_final_excessive_ready']), dtype = 'bool')
                #
                # self.data_df['prev_sell_close_position_final_excessive_ready'] = self.data_df['sell_close_position_final_excessive_ready'].shift(1)
                # self.data_df.at[0, 'prev_sell_close_position_final_excessive_ready'] = False
                # self.data_df['prev_sell_close_position_final_excessive_ready'] = pd.Series(list(self.data_df['prev_sell_close_position_final_excessive_ready']), dtype = 'bool')
                #
                # self.data_df['prev2_sell_close_position_final_excessive_ready'] = self.data_df['prev_sell_close_position_final_excessive_ready'].shift(1)
                # self.data_df.at[0, 'prev2_sell_close_position_final_excessive_ready'] = False
                # self.data_df['prev2_sell_close_position_final_excessive_ready'] = pd.Series(list(self.data_df['prev2_sell_close_position_final_excessive_ready']), dtype = 'bool')
                #
                #
                # self.data_df['prev_buy_close_position_final_conservative_ready'] = self.data_df['buy_close_position_final_conservative_ready'].shift(1)
                # self.data_df.at[0, 'prev_buy_close_position_final_conservative_ready'] = False
                # self.data_df['prev_buy_close_position_final_conservative_ready'] = pd.Series(list(self.data_df['prev_buy_close_position_final_conservative_ready']), dtype = 'bool')
                #
                # self.data_df['prev2_buy_close_position_final_conservative_ready'] = self.data_df['prev_buy_close_position_final_conservative_ready'].shift(1)
                # self.data_df.at[0, 'prev2_buy_close_position_final_conservative_ready'] = False
                # self.data_df['prev2_buy_close_position_final_conservative_ready'] = pd.Series(list(self.data_df['prev2_buy_close_position_final_conservative_ready']), dtype = 'bool')
                #
                # self.data_df['prev_sell_close_position_final_conservative_ready'] = self.data_df['sell_close_position_final_conservative_ready'].shift(1)
                # self.data_df.at[0, 'prev_sell_close_position_final_conservative_ready'] = False
                # self.data_df['prev_sell_close_position_final_conservative_ready'] = pd.Series(list(self.data_df['prev_sell_close_position_final_conservative_ready']), dtype = 'bool')
                #
                # self.data_df['prev2_sell_close_position_final_conservative_ready'] = self.data_df['prev_sell_close_position_final_conservative_ready'].shift(1)
                # self.data_df.at[0, 'prev2_sell_close_position_final_conservative_ready'] = False
                # self.data_df['prev2_sell_close_position_final_conservative_ready'] = pd.Series(list(self.data_df['prev2_sell_close_position_final_conservative_ready']), dtype = 'bool')
                #


                #
                # self.data_df['buy_close_position_final_excessive'] = (self.data_df['prev_buy_close_position_final_excessive_ready'] &\
                #                                                         self.data_df['prev2_buy_close_position_final_excessive_ready'] &\
                #                                                         self.data_df['is_negative'])# | self.data_df['buy_close_position_final_excessive_strict']
                # self.data_df['buy_close_position_final_conservative'] = (self.data_df['prev_buy_close_position_final_conservative_ready'] &\
                #                                                         self.data_df['prev2_buy_close_position_final_conservative_ready'] &\
                #                                                         self.data_df['is_negative'])# | self.data_df['buy_close_position_final_conservative_strict']
                #
                # self.data_df['sell_close_position_final_excessive'] = (self.data_df['prev_sell_close_position_final_excessive_ready'] &\
                #                                                         self.data_df['prev2_sell_close_position_final_excessive_ready'] &\
                #                                                         self.data_df['is_positive'])# | self.data_df['sell_close_position_final_excessive_strict']
                # self.data_df['sell_close_position_final_conservative'] = (self.data_df['prev_sell_close_position_final_conservative_ready'] &\
                #                                                         self.data_df['prev2_sell_close_position_final_conservative_ready'] &\
                #                                                         self.data_df['is_positive'])# | self.data_df['sell_close_position_final_conservative_strict']


                # print("Fuck Python:")
                # t_df = self.data_df.iloc[875:884][['time','prev_buy_close_position_final_conservative_ready',
                #                                    'prev2_buy_close_position_final_conservative_ready', 'buy_close_position_final_conservative', 'is_negative']]
                # t_df = t_df.rename(columns = {
                #     'prev_buy_close_position_final_conservative_ready' : 'prev',
                #     'prev2_buy_close_position_final_conservative_ready' : 'prev2',
                #     'buy_close_position_final_conservative' : 'final'
                # })
                # print(t_df)
                # sys.exit(0)



                self.data_df['prev_buy_close_position_guppy1'] = self.data_df['buy_close_position_guppy1'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_guppy1'] = False
                self.data_df['prev_buy_close_position_guppy1'] = pd.Series(list(self.data_df['prev_buy_close_position_guppy1']), dtype='bool')
                self.data_df['first_buy_close_position_guppy1'] = self.data_df['buy_close_position_guppy1'] & (~self.data_df['prev_buy_close_position_guppy1'])

                self.data_df['prev_buy_close_position_guppy2'] = self.data_df['buy_close_position_guppy2'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_guppy2'] = False
                self.data_df['prev_buy_close_position_guppy2'] = pd.Series(list(self.data_df['prev_buy_close_position_guppy2']), dtype='bool')
                self.data_df['first_buy_close_position_guppy2'] = self.data_df['buy_close_position_guppy2'] & (~self.data_df['prev_buy_close_position_guppy2'])


                self.data_df['prev_buy_close_position_vegas'] = self.data_df['buy_close_position_vegas'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_vegas'] = False
                self.data_df['prev_buy_close_position_vegas'] = pd.Series(list(self.data_df['prev_buy_close_position_vegas']), dtype='bool')
                self.data_df['first_buy_close_position_vegas'] = self.data_df['buy_close_position_vegas'] & (~self.data_df['prev_buy_close_position_vegas'])

                self.data_df['prev_buy_close_position_final_excessive1'] = self.data_df['buy_close_position_final_excessive1'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_excessive1'] = False
                self.data_df['prev_buy_close_position_final_excessive1'] = pd.Series(list(self.data_df['prev_buy_close_position_final_excessive1']), dtype='bool')
                self.data_df['first_buy_close_position_final_excessive1'] = self.data_df['buy_close_position_final_excessive1'] & (~self.data_df['prev_buy_close_position_final_excessive1'])

                self.data_df['prev_buy_close_position_final_excessive2'] = self.data_df['buy_close_position_final_excessive2'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_excessive2'] = False
                self.data_df['prev_buy_close_position_final_excessive2'] = pd.Series(list(self.data_df['prev_buy_close_position_final_excessive2']), dtype='bool')
                self.data_df['first_buy_close_position_final_excessive2'] = self.data_df['buy_close_position_final_excessive2'] & (~self.data_df['prev_buy_close_position_final_excessive2'])



                self.data_df['prev_buy_close_position_final_conservative'] = self.data_df['buy_close_position_final_conservative'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_conservative'] = False
                self.data_df['prev_buy_close_position_final_conservative'] = pd.Series(list(self.data_df['prev_buy_close_position_final_conservative']), dtype='bool')
                self.data_df['first_buy_close_position_final_conservative'] = self.data_df['buy_close_position_final_conservative'] & (~self.data_df['prev_buy_close_position_final_conservative'])


                self.data_df['prev_buy_close_position_final_excessive_strict'] = self.data_df['buy_close_position_final_excessive_strict'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_excessive_strict'] = False
                self.data_df['prev_buy_close_position_final_excessive_strict'] = pd.Series(list(self.data_df['prev_buy_close_position_final_excessive_strict']), dtype='bool')
                self.data_df['first_buy_close_position_final_excessive_strict'] = self.data_df['buy_close_position_final_excessive_strict'] & (~self.data_df['prev_buy_close_position_final_excessive_strict'])




                self.data_df['prev_buy_close_position_final_conservative_strict'] = self.data_df['buy_close_position_final_conservative_strict'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_conservative_strict'] = False
                self.data_df['prev_buy_close_position_final_conservative_strict'] = pd.Series(list(self.data_df['prev_buy_close_position_final_conservative_strict']), dtype='bool')
                self.data_df['first_buy_close_position_final_conservative_strict'] = self.data_df['buy_close_position_final_conservative_strict'] & (~self.data_df['prev_buy_close_position_final_conservative_strict'])


                self.data_df['prev_buy_close_position_final_simple'] = self.data_df['buy_close_position_final_simple'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_simple'] = False
                self.data_df['prev_buy_close_position_final_simple'] = pd.Series(list(self.data_df['prev_buy_close_position_final_simple']), dtype='bool')
                self.data_df['first_buy_close_position_final_simple'] = self.data_df['buy_close_position_final_simple'] & (~self.data_df['prev_buy_close_position_final_simple'])


                self.data_df['prev_buy_close_position_final_quick'] = self.data_df['buy_close_position_final_quick'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_quick'] = False
                self.data_df['prev_buy_close_position_final_quick'] = pd.Series(list(self.data_df['prev_buy_close_position_final_quick']), dtype='bool')
                self.data_df['first_buy_close_position_final_quick'] = self.data_df['buy_close_position_final_quick'] & (~self.data_df['prev_buy_close_position_final_quick'])


                self.data_df['prev_buy_close_position_final_urgent'] = self.data_df['buy_close_position_final_urgent'].shift(1)
                self.data_df.at[0, 'prev_buy_close_position_final_urgent'] = False
                self.data_df['prev_buy_close_position_final_urgent'] = pd.Series(list(self.data_df['prev_buy_close_position_final_urgent']), dtype='bool')
                self.data_df['first_buy_close_position_final_urgent'] = self.data_df['buy_close_position_final_urgent'] & (~self.data_df['prev_buy_close_position_final_urgent'])







                self.data_df['prev_sell_close_position_guppy1'] = self.data_df['sell_close_position_guppy1'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_guppy1'] = False
                self.data_df['prev_sell_close_position_guppy1'] = pd.Series(list(self.data_df['prev_sell_close_position_guppy1']), dtype='bool')
                self.data_df['first_sell_close_position_guppy1'] = self.data_df['sell_close_position_guppy1'] & (~self.data_df['prev_sell_close_position_guppy1'])

                self.data_df['prev_sell_close_position_guppy2'] = self.data_df['sell_close_position_guppy2'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_guppy2'] = False
                self.data_df['prev_sell_close_position_guppy2'] = pd.Series(list(self.data_df['prev_sell_close_position_guppy2']), dtype='bool')
                self.data_df['first_sell_close_position_guppy2'] = self.data_df['sell_close_position_guppy2'] & (~self.data_df['prev_sell_close_position_guppy2'])




                self.data_df['prev_sell_close_position_vegas'] = self.data_df['sell_close_position_vegas'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_vegas'] = False
                self.data_df['prev_sell_close_position_vegas'] = pd.Series(list(self.data_df['prev_sell_close_position_vegas']), dtype='bool')
                self.data_df['first_sell_close_position_vegas'] = self.data_df['sell_close_position_vegas'] & (~self.data_df['prev_sell_close_position_vegas'])

                self.data_df['prev_sell_close_position_final_excessive1'] = self.data_df['sell_close_position_final_excessive1'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_excessive1'] = False
                self.data_df['prev_sell_close_position_final_excessive1'] = pd.Series(list(self.data_df['prev_sell_close_position_final_excessive1']), dtype='bool')
                self.data_df['first_sell_close_position_final_excessive1'] = self.data_df['sell_close_position_final_excessive1'] & (~self.data_df['prev_sell_close_position_final_excessive1'])

                self.data_df['prev_sell_close_position_final_excessive2'] = self.data_df['sell_close_position_final_excessive2'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_excessive2'] = False
                self.data_df['prev_sell_close_position_final_excessive2'] = pd.Series(list(self.data_df['prev_sell_close_position_final_excessive2']), dtype='bool')
                self.data_df['first_sell_close_position_final_excessive2'] = self.data_df['sell_close_position_final_excessive2'] & (~self.data_df['prev_sell_close_position_final_excessive2'])



                self.data_df['prev_sell_close_position_final_conservative'] = self.data_df['sell_close_position_final_conservative'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_conservative'] = False
                self.data_df['prev_sell_close_position_final_conservative'] = pd.Series(list(self.data_df['prev_sell_close_position_final_conservative']), dtype='bool')
                self.data_df['first_sell_close_position_final_conservative'] = self.data_df['sell_close_position_final_conservative'] & (~self.data_df['prev_sell_close_position_final_conservative'])


                self.data_df['prev_sell_close_position_final_excessive_strict'] = self.data_df['sell_close_position_final_excessive_strict'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_excessive_strict'] = False
                self.data_df['prev_sell_close_position_final_excessive_strict'] = pd.Series(list(self.data_df['prev_sell_close_position_final_excessive_strict']), dtype='bool')
                self.data_df['first_sell_close_position_final_excessive_strict'] = self.data_df['sell_close_position_final_excessive_strict'] & (~self.data_df['prev_sell_close_position_final_excessive_strict'])

                self.data_df['prev_sell_close_position_final_conservative_strict'] = self.data_df['sell_close_position_final_conservative_strict'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_conservative_strict'] = False
                self.data_df['prev_sell_close_position_final_conservative_strict'] = pd.Series(list(self.data_df['prev_sell_close_position_final_conservative_strict']), dtype='bool')
                self.data_df['first_sell_close_position_final_conservative_strict'] = self.data_df['sell_close_position_final_conservative_strict'] & (~self.data_df['prev_sell_close_position_final_conservative_strict'])

                self.data_df['prev_sell_close_position_final_simple'] = self.data_df['sell_close_position_final_simple'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_simple'] = False
                self.data_df['prev_sell_close_position_final_simple'] = pd.Series(list(self.data_df['prev_sell_close_position_final_simple']), dtype='bool')
                self.data_df['first_sell_close_position_final_simple'] = self.data_df['sell_close_position_final_simple'] & (~self.data_df['prev_sell_close_position_final_simple'])

                self.data_df['prev_sell_close_position_final_quick'] = self.data_df['sell_close_position_final_quick'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_quick'] = False
                self.data_df['prev_sell_close_position_final_quick'] = pd.Series(list(self.data_df['prev_sell_close_position_final_quick']), dtype='bool')
                self.data_df['first_sell_close_position_final_quick'] = self.data_df['sell_close_position_final_quick'] & (~self.data_df['prev_sell_close_position_final_quick'])

                self.data_df['prev_sell_close_position_final_urgent'] = self.data_df['sell_close_position_final_urgent'].shift(1)
                self.data_df.at[0, 'prev_sell_close_position_final_urgent'] = False
                self.data_df['prev_sell_close_position_final_urgent'] = pd.Series(list(self.data_df['prev_sell_close_position_final_urgent']), dtype='bool')
                self.data_df['first_sell_close_position_final_urgent'] = self.data_df['sell_close_position_final_urgent'] & (~self.data_df['prev_sell_close_position_final_urgent'])




                def is_more_aggressive(price1, price2, side):
                    if side == 'buy':
                        return price1 > price2
                    else:
                        return price1 < price2

                def calculate_pnl(entry_price, exit_price, side):

                    # print("entry_price = " + str(entry_price))
                    # print("exit_price = " + str(exit_price))

                    if side == 'buy':
                        period_pnl = (exit_price - entry_price) * self.lot_size * self.exchange_rate
                    else:
                        period_pnl = (entry_price - exit_price) * self.lot_size * self.exchange_rate

                    return period_pnl

                def select_close_positions(x, guppy1, guppy2, vegas, excessive1, excessive2, conservative, excessive_strict, conservative_strict, simple,
                                               quick, quick_immediate, urgent, fixed_time, quick_fixed_time,
                                               selected_guppy1, selected_guppy2, selected_vegas, selected_excessive1, selected_excessive2, selected_conservative,
                                               selected_simple, selected_quick, selected_urgent, reentry, selected_fixed_time,
                                           close, open, most_passive_guppy, most_aggressive_guppy,
                                           exceed_vegas, enter_guppy, passive_than_guppy, num_guppy_bars,
                                           group_most_passive_price, entry_point_price, side):
                    # if side == 'buy':
                    #     print("In select_close_positions:")
                    #     print(x)

                    total_guppy1 = 0
                    total_guppy2 = 0
                    total_vegas = 0
                    raw_total_excessive1 = 0
                    raw_total_excessive2 = 0
                    raw_total_conservative = 0

                    total_excessive1 = 0
                    total_excessive = 0
                    total_conservative = 0
                    last_excessive1 = -1
                    last_excessive2 = -1
                    last_conservative = -1

                    total_simple = 0

                    total_quick = 0
                    total_urgent = 0
                    total_fixed_time = 0
                    total_quick_fixed_time = 0

                    quick_ready = False
                    quick_ready_price = None

                    last_quick_ready = -1

                    quick_ready_number = 0


                    quick_immediate_stop_loss = False
                    quick_immediate_stop_loss_price = None
                    last_row = None

                    for i in range(0, x.shape[0]):
                        row = x.iloc[i]

                        # print("Process time " + str(row['time']))
                        # print("total_guppy = " + str(total_guppy))
                        # print("total_vegas = " + str(total_vegas))


                        if use_simple_stop_loss:

                            if row[guppy1]:

                                if total_guppy1 + total_guppy2 <= 1 and total_simple == 0 and total_quick == 0 and total_urgent == 0 and is_more_aggressive(row['close'], row[group_most_passive_price], side):
                                    x.at[x.index[i], selected_guppy1] = 1
                                    total_guppy1 += 1

                            if row[guppy2]:

                                if total_guppy1 + total_guppy2 <= 1 and total_simple == 0 and total_quick == 0 and total_urgent == 0 and is_more_aggressive(row['close'], row[group_most_passive_price], side):
                                    x.at[x.index[i], selected_guppy2] = 1
                                    total_guppy2 += 1

                            if row[vegas]:
                                if total_vegas <= 1 and total_simple == 0 and total_quick == 0 and total_urgent == 0 and is_more_aggressive(row['close'], row[group_most_passive_price], side):
                                    x.at[x.index[i], selected_vegas] = 1
                                    total_vegas += 1

                            if row[quick]:
                                if (total_guppy1 + total_guppy2 == 0 and total_vegas == 0 and total_simple == 0 and total_quick == 0 and total_urgent == 0):
                                    x.at[x.index[i], selected_quick] = 1
                                    total_quick += 1

                            if row[simple]:
                                if total_simple == 0 and total_quick == 0 and total_urgent == 0:
                                    x.at[x.index[i], selected_simple] = 1
                                    total_simple += 1


                        else:

                            if row[guppy1]:

                                if total_guppy1 + total_guppy2 <= 1 and total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and\
                                        total_fixed_time == 0 and total_quick_fixed_time == 0 and is_more_aggressive(row['close'], row[group_most_passive_price], side):
                                    x.at[x.index[i], selected_guppy1] = 1
                                    total_guppy1 += 1

                            if row[guppy2]:

                                if total_guppy1 + total_guppy2 <= 1 and total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and\
                                         total_fixed_time == 0 and total_quick_fixed_time == 0 and is_more_aggressive(row['close'], row[group_most_passive_price], side):
                                    x.at[x.index[i], selected_guppy2] = 1
                                    total_guppy2 += 1

                            if row[vegas]:
                                if total_vegas <= 1 and total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and\
                                         total_fixed_time == 0 and total_quick_fixed_time == 0 and is_more_aggressive(row['close'], row[group_most_passive_price], side):
                                    x.at[x.index[i], selected_vegas] = 1
                                    total_vegas += 1

                            #print("time = " + str(row['time']))


                            # if row[quick]:# and not quick_ready:
                            #     #if (total_guppy1 + total_guppy2 == 0 and total_vegas == 0 and total_excessive == 0 and total_conservative == 0 and total_quick == 0):
                            #     if (total_excessive == 0 and total_conservative == 0 and total_quick == 0):
                            #         #print("quick_ready = true")
                            #         #quick_ready = True
                            #         #quick_ready_price = row['close']
                            #         x.at[x.index[i], selected_quick] = 1
                            #         total_quick += 1


                            if is_reentry:
                                if quick_immediate_stop_loss:
                                    if (side == 'buy' and row['close'] > row['open'] and row['close'] > quick_immediate_stop_loss_price) |\
                                            (side == 'sell' and row['close'] < row['open'] and row['close'] < quick_immediate_stop_loss_price):
                                        x.at[x.index[i], reentry] = 1

                                        total_guppy1 = 0
                                        total_guppy2 = 0
                                        total_vegas = 0
                                        raw_total_excessive1 = 0
                                        raw_total_excessive2 = 0
                                        raw_total_conservative = 0

                                        total_excessive1 = 0
                                        total_excessive = 0
                                        total_conservative = 0
                                        last_excessive1 = -1
                                        last_excessive2 = -1
                                        last_conservative = -1
                                        total_simple = 0
                                        total_quick = 0
                                        total_urgent = 0
                                        total_fixed_time = 0
                                        total_quick_fixed_time = 0
                                        quick_ready = False
                                        quick_ready_price = None
                                        last_quick_ready = -1
                                        last_row = None

                                        quick_ready_number = 0

                                    quick_immediate_stop_loss = False
                                    quick_immediate_stop_loss_price = None


                            if row[quick]:# and not quick_ready:
                                #if (total_guppy1 + total_guppy2 == 0 and total_vegas == 0 and total_excessive == 0 and total_conservative == 0 and total_quick == 0):
                                if (total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and total_fixed_time == 0 and total_quick_fixed_time == 0):
                                    #print("quick_ready = true")
                                    quick_ready = True
                                    #quick_ready_price = row['close']

                                    quick_ready_price = (row['close'] + row['open'])/2.0

                                    last_quick_ready = i

                                    quick_ready_number = 0
                                    #x.at[x.index[i], selected_quick] = 1
                                    #total_quick += 1

                            if quick_ready and (is_immediately_in or row[quick_immediate] or i > last_quick_ready):
                                # if row[enter_guppy]:
                                #     quick_ready = False
                                # elif row[passive_than_guppy]:
                                #     quick_ready = False
                                #     x.at[x.index[i], selected_quick] = 1
                                #     total_quick += 1
                                #print("close = " + str(row['close']) + ", quick_ready_price = " + str(quick_ready_price))

                                if is_more_aggressive(row['close'], row[most_aggressive_guppy], side):
                                    quick_ready = False
                                    last_quick_ready = -1

                                #if (side == 'buy' and row['close'] < row['open']) | (side == 'sell' and row['close'] > row['open']):
                                if is_more_aggressive(row['open'], row['close'], side):
                                    # if is_more_aggressive(quick_ready_price, row['close'], side):
                                    #     #print("Set selected_quick")
                                    #     quick_ready = False
                                    #     quick_ready_price = None

                                    if is_more_aggressive(row[most_passive_guppy], row['close'], side):


                                        if (not tightened_quick_stop_loss) or (row[quick_immediate] or (
                                                is_more_aggressive(quick_ready_price, row['close'], side) #and \
                                                #last_row is not None and is_more_aggressive(last_row['open'], last_row['close'], side)
                                        )):

                                            quick_ready_number += 1
                                            if quick_ready_number == 2:

                                                x.at[x.index[i], selected_quick] = 1
                                                total_quick += 1

                                                quick_ready = False
                                                last_quick_ready = -1

                                                quick_ready_number = 0


                                        if is_reentry and row[quick_immediate] and (row[num_guppy_bars] == 0):
                                            quick_immediate_stop_loss = True
                                            quick_immediate_stop_loss_price = row['open']





                            #This will never be used
                            # if total_quick == 1 and total_excessive == 0 and total_conservative == 0:
                            #     if row[exceed_vegas] == 0 and row[enter_guppy]:
                            #         x.at[x.index[i], reentry] = 1
                            #         total_guppy1 = 0
                            #         total_guppy2 = 0
                            #         total_vegas = 0
                            #         raw_total_excessive1 = 0
                            #         raw_total_excessive2 = 0
                            #         raw_total_conservative = 0
                            #         total_excessive1 = 0
                            #         total_excessive = 0
                            #         total_conservative = 0
                            #         last_excessive1 = -1
                            #         last_excessive2 = -1
                            #         last_conservative = -1
                            #         total_simple = 0
                            #         total_quick = 0





                            if row[excessive1] or row[excessive_strict]:
                                if (total_guppy1 + total_guppy2 > 0 or total_vegas > 0 or row[exceed_vegas] > 0 or row[num_guppy_bars] >= 3) and\
                                        total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and total_fixed_time == 0 and total_quick_fixed_time == 0:

                                    gap = i - last_excessive1
                                    if (raw_total_excessive1 > 0 and i > 0 and last_excessive1 > 0 and (gap > 1 and gap < 12)) or row[excessive_strict]: # and (gap > 1 and gap < 14)
                                        x.at[x.index[i], selected_excessive1] = 1

                                        if not support_half_stop_loss:
                                            total_excessive += 1

                                        total_excessive1 += 1

                                    raw_total_excessive1 += 1
                                    last_excessive1 = i


                            if row[excessive2] or row[excessive_strict]:
                                if (total_guppy1 + total_guppy2 > 0 or total_vegas > 0 or row[exceed_vegas] > 0 or row[num_guppy_bars] >= 3) and \
                                        total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and total_excessive1 > 0 and total_fixed_time == 0 and total_quick_fixed_time == 0:

                                    gap = i - last_excessive2
                                    if (raw_total_excessive2 > 0 and i > 0 and last_excessive2 > 0 and (gap > 1 and gap < 12)) or row[excessive_strict]:
                                        if support_half_stop_loss:
                                            x.at[x.index[i], selected_excessive2] = 1
                                            total_excessive += 1

                                    raw_total_excessive2 += 1
                                    last_excessive2 = i



                            if row[conservative] or row[conservative_strict]:
                                # print("conservative:")
                                # debug_df = pd.DataFrame({'total_guppy1' : [total_guppy1],
                                #                          'total_guppy2' : [total_guppy2],
                                #                          'total_vegas' : [total_vegas],
                                #                          'total_excessive' : [total_excessive],
                                #                          'total_conservative' : [total_conservative]})
                                # print(debug_df)
                                if total_guppy1 == 0 and total_guppy2 == 0 and total_vegas == 0 and \
                                        total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and total_fixed_time == 0 and total_quick_fixed_time == 0:

                                    gap = i - last_conservative

                                    # print("i = " + str(i))
                                    # print("last_conservative = " + str(last_conservative))
                                    # print("gap = " + str(gap))
                                    # print("raw_total_conservative = " + str(raw_total_conservative))

                                    if (raw_total_conservative > 0 and i > 0 and last_conservative > 0 and (gap > 1 and gap < 12)) or row[conservative_strict]:
                                        x.at[x.index[i], selected_conservative] = 1
                                        total_conservative += 1

                                    raw_total_conservative += 1
                                    last_conservative = i


                            if row[urgent]:

                                if total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and total_fixed_time == 0 and total_quick_fixed_time == 0:
                                    x.at[x.index[i], selected_urgent] = 1
                                    total_urgent += 1


                            if row[quick_fixed_time] > 0:
                                if total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and total_fixed_time == 0 and total_quick_fixed_time == 0:
                                    current_pnl = calculate_pnl(row[entry_point_price], row[close], side)

                                    # print("")
                                    # print("In calculate period pnl")
                                    # print("quick_fixed_time = " + quick_fixed_time)
                                    # print("time = " + str(row['time']))
                                    # print("entry_point_price = " + str(row[entry_point_price]))
                                    # print("exit_price = " + str(row[close]))
                                    # print("current_pnl = " + str(current_pnl))
                                    # print("")

                                    #if current_pnl > 100:
                                    if is_more_aggressive(row[close], row[most_passive_guppy], side):  #most_aggressive_guppy
                                        #print("Passed current_pnl test")
                                        x.at[x.index[i], selected_fixed_time] = row[quick_fixed_time]
                                        if row[quick_fixed_time] == 1:
                                            total_quick_fixed_time += 1
                                        

                            if row[fixed_time] > 0:

                                if total_excessive == 0 and total_conservative == 0 and total_quick == 0 and total_urgent == 0 and total_fixed_time == 0 and total_quick_fixed_time == 0:
                                    x.at[x.index[i], selected_fixed_time] = row[fixed_time]
                                    if row[fixed_time] == 1:
                                        total_fixed_time += 1





                        if total_excessive > 0 or total_conservative > 0 or total_quick > 0 and total_urgent > 0 and total_fixed_time > 0 and total_quick_fixed_time > 0:
                            break


                        last_row = row


                    # if 'sell_point_id' in x.columns:
                    #     y = x.copy()
                    #     y = y.rename(columns = {guppy1: 'guppy1', guppy2: 'guppy2', vegas : 'vegas', excessive1 : 'excessive1', excessive2 : 'excessive2',  conservative : 'conservative',
                    #                             excessive_strict : 'excessive_strict', conservative_strict : 'conservative_strict',
                    #                             simple : 'simple',
                    #                             quick : 'quick', quick_immediate : 'quick_immediate',  urgent : 'urgent',
                    #                             fixed_time : 'fixed_time', quick_fixed_time : 'quick_fixed_time',
                    #                             selected_quick : 'selected_quick',
                    #                             selected_urgent: 'selected_urgent',
                    #                             #selected_fixed_time : 'selected_fixed_time',
                    #                             selected_guppy1 : 'selected_guppy1',
                    #                             selected_guppy2: 'selected_guppy2',
                    #                             selected_vegas : 'selected_vegas', selected_excessive1 : 'selected_excessive1', selected_excessive2 : 'selected_excessive2',
                    #                             selected_conservative : 'selected_conservative',
                    #                             selected_simple : 'selected_simple'})
                    #
                    #     print("Dig Goup:")
                    #     conditions = reduce(lambda left, right: left | right, [y[col] for col in ['guppy1', 'guppy2', 'vegas', 'excessive1', 'excessive2', 'conservative',
                    #                                                                               'simple','quick']])
                    #     #y = y[conditions]
                    #     print(y)

                    return x


                for side in ['buy', 'sell']:
                    temp_df = self.data_df[['id', 'time', side + '_point_id',
                                            'first_' + side + '_close_position_guppy1', 'first_' + side + '_close_position_guppy2',
                                            'first_' + side + '_close_position_vegas',
                                            'first_' + side + '_close_position_final_excessive1',
                                            'first_' + side + '_close_position_final_excessive2',
                                            'first_' + side + '_close_position_final_conservative',
                                            'first_' + side + '_close_position_final_excessive_strict', 'first_' + side + '_close_position_final_conservative_strict',

                                            'first_' + side + '_close_position_final_simple',
                                            'first_' + side + '_close_position_final_quick',
                                             side + '_close_position_final_quick_immediate',
                                            'first_' + side + '_close_position_final_urgent',
                                            side + '_close_position_fixed_time',
                                            side + '_close_position_quick_fixed_time',
                                            side + '_enter_guppy',
                                            side + '_passive_than_guppy',
                                            
                                            'num_above_vegas_for_buy', 'num_below_vegas_for_sell',
                                            'num_bar_above_passive_guppy_for_buy', 'num_bar_below_passive_guppy_for_sell',
                                            'close',
                                            'open',
                                            'highest_guppy',
                                            'lowest_guppy',
                                            'group_min_price',
                                            'group_max_price',
                                            'buy_point_price','sell_point_price']]

                    #select_condition = reduce(lambda left, right: left | right, [self.data_df[col] for col in temp_df.columns[2:]])

                    # select_condition = reduce(lambda left, right: left | right, [self.data_df[col] for col in ['first_' + side + '_close_position_guppy', 'first_' + side + '_close_position_vegas',
                    #                         'first_' + side + '_close_position_final_excessive', 'first_' + side + '_close_position_final_conservative',
                    #                         'first_' + side + '_close_position_final_excessive_strict', 'first_' + side + '_close_position_final_conservative_strict']])
                    #
                    #side_df = temp_df[select_condition]
                    side_df = temp_df.copy()


                    # if side == 'sell':
                    #     print("side_df:")
                    #     print(side_df[['id','time','sell_point_id',
                    #                    'first_' + side + '_close_position_final_excessive']])

                    # print("Before Before side_df:")
                    # print(side_df.tail(100))

                    side_df = side_df[side_df[side + '_point_id'] > 0]
                    side_df['first_selected_' + side + '_close_position_guppy1'] = 0
                    side_df['first_selected_' + side + '_close_position_guppy2'] = 0
                    side_df['first_selected_' + side + '_close_position_vegas'] = 0
                    side_df['first_selected_' + side + '_close_position_final_excessive1'] = 0
                    side_df['first_selected_' + side + '_close_position_final_excessive2'] = 0
                    side_df['first_selected_' + side + '_close_position_final_conservative'] = 0
                    side_df['first_selected_' + side + '_close_position_final_simple'] = 0
                    side_df['first_selected_' + side + '_close_position_final_quick'] = 0
                    side_df['first_selected_' + side + '_close_position_final_urgent'] = 0
                    side_df['reentry_' + side] = 0
                    side_df['selected_' + side + '_close_position_fixed_time'] = 0




                    exceed_vegas = 'num_above_vegas_for_buy' if side == 'buy' else 'num_below_vegas_for_sell'
                    num_guppy_bars = 'num_bar_above_passive_guppy_for_buy' if side == 'buy' else 'num_bar_below_passive_guppy_for_sell'
                    group_most_passive_price = 'group_min_price' if side == 'buy' else 'group_max_price'
                    entry_point_price = 'buy_point_price' if side == 'buy' else 'sell_point_price'

                    most_passive_guppy = 'lowest_guppy' if side == 'buy' else 'highest_guppy'
                    most_aggressive_guppy = 'highest_guppy' if side == 'buy' else 'lowest_guppy'

                    # print("here id in side_df:" + str('id' in side_df.columns))
                    #
                    # print("Before side_df:")
                    # print(side_df.tail(100))

                    if side_df.shape[0] > 0:

                        side_df =side_df.groupby([side + '_point_id']).apply(lambda x: select_close_positions(x,
                                                    guppy1 = 'first_' + side + '_close_position_guppy1',
                                                    guppy2 = 'first_' + side + '_close_position_guppy2',
                                                    vegas = 'first_' + side + '_close_position_vegas',
                                                    excessive1 = 'first_' + side + '_close_position_final_excessive1',
                                                    excessive2 = 'first_' + side + '_close_position_final_excessive2',
                                                    conservative = 'first_' + side + '_close_position_final_conservative',
                                                    excessive_strict = 'first_' + side + '_close_position_final_excessive_strict',
                                                    conservative_strict = 'first_' + side + '_close_position_final_conservative_strict',
                                                    simple = 'first_' + side + '_close_position_final_simple',
                                                    quick = 'first_' + side + '_close_position_final_quick',
                                                    quick_immediate = side + '_close_position_final_quick_immediate',
                                                    urgent = 'first_' + side + '_close_position_final_urgent',
                                                    fixed_time = side + '_close_position_fixed_time',
                                                    quick_fixed_time = side + '_close_position_quick_fixed_time',
                                                    selected_guppy1 = 'first_selected_' + side + '_close_position_guppy1',
                                                    selected_guppy2 = 'first_selected_' + side + '_close_position_guppy2',
                                                    selected_vegas = 'first_selected_' + side + '_close_position_vegas',
                                                    selected_excessive1 = 'first_selected_' + side + '_close_position_final_excessive1',
                                                    selected_excessive2 = 'first_selected_' + side + '_close_position_final_excessive2',
                                                    selected_conservative = 'first_selected_' + side + '_close_position_final_conservative',
                                                    selected_simple = 'first_selected_' + side + '_close_position_final_simple',
                                                    selected_quick = 'first_selected_' + side + '_close_position_final_quick',
                                                    selected_urgent = 'first_selected_' + side + '_close_position_final_urgent',
                                                    reentry = 'reentry_' + side,
                                                    selected_fixed_time = 'selected_' + side + '_close_position_fixed_time',
                                                    close = 'close',
                                                    open = 'open',
                                                    most_passive_guppy = most_passive_guppy,
                                                    most_aggressive_guppy = most_aggressive_guppy,
                                                    exceed_vegas = exceed_vegas,
                                                    enter_guppy = side + '_enter_guppy',
                                                    passive_than_guppy = side + '_passive_than_guppy',
                                                    num_guppy_bars = num_guppy_bars,
                                                    group_most_passive_price = group_most_passive_price, entry_point_price = entry_point_price, side = side
                                                     ))

                    # print("After here side_df:")
                    # print(side_df.tail(100))
                    #
                    # print("id in side_df:" + str('id' in side_df.columns))
                    # print("id in temp_df: " + str('id' in temp_df.columns))

                    temp_df = pd.merge(temp_df, side_df, on = ['id'], how = 'left')
                    temp_df = temp_df.fillna(0)

                    for col in ['first_selected_' + side + '_close_position_guppy1','first_selected_' + side + '_close_position_guppy2', 'first_selected_' + side + '_close_position_vegas',
                                 'first_selected_' + side + '_close_position_final_excessive1',
                                 'first_selected_' + side + '_close_position_final_excessive2',
                                 'first_selected_' + side + '_close_position_final_conservative',
                                 'first_selected_' + side + '_close_position_final_simple',
                                 'first_selected_' + side + '_close_position_final_quick',
                                 'first_selected_' + side + '_close_position_final_urgent',
                                 'reentry_' + side]:
                        self.data_df[col] = np.where(
                            temp_df[col] == 1,
                            True,
                            False
                        )

                    # print("")
                    # print("Debug DataFrame here:")
                    # # print("temp_df columns:")
                    # # print(temp_df.columns)
                    # print(temp_df[['time_x', 'time_y', 'selected_' + side + '_close_position_fixed_time']].tail(50))
                    # print("")

                    self.data_df['selected_' + side + '_close_position_fixed_time'] = temp_df['selected_' + side + '_close_position_fixed_time']
                    self.data_df['selected_' + side + '_close_position_fixed_time_terminal'] = temp_df['selected_' + side + '_close_position_fixed_time'] == 1
                    self.data_df['selected_' + side + '_close_position_fixed_time_temporary'] = temp_df['selected_' + side + '_close_position_fixed_time'] > 1

                    self.data_df['first_final_' + side + '_fire'] = self.data_df['first_final_' + side + '_fire'] | self.data_df['reentry_' + side]



            ############# Select which close points in the second phase to show ############################
            if True:


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



                self.data_df['cum_sell_close_position_guppy1'] = self.data_df['first_selected_sell_close_position_guppy1'].cumsum()
                self.data_df['cum_sell_close_position_guppy2'] = self.data_df['first_selected_sell_close_position_guppy2'].cumsum()
                self.data_df['cum_sell_close_position_vegas'] = self.data_df['first_selected_sell_close_position_vegas'].cumsum()
                self.data_df['cum_sell_close_position_final_excessive1'] = self.data_df['first_selected_sell_close_position_final_excessive1'].cumsum()
                self.data_df['cum_sell_close_position_final_excessive2'] = self.data_df['first_selected_sell_close_position_final_excessive2'].cumsum()
                self.data_df['cum_sell_close_position_final_conservative'] = self.data_df['first_selected_sell_close_position_final_conservative'].cumsum()
                self.data_df['cum_sell_close_position_final_simple'] = self.data_df['first_selected_sell_close_position_final_simple'].cumsum()
                self.data_df['cum_sell_close_position_final_quick'] = self.data_df['first_selected_sell_close_position_final_quick'].cumsum()
                self.data_df['cum_sell_close_position_final_urgent'] = self.data_df['first_selected_sell_close_position_final_urgent'].cumsum()
                self.data_df['cum_sell_close_position_fixed_time_terminal'] = self.data_df["selected_sell_close_position_fixed_time_terminal"].cumsum()

                self.data_df['cum_special2_sell_close_position'] = self.data_df['first_actual_special_sell_close_position'].cumsum()
                self.data_df['cum_sell_close_position_excessive'] = self.data_df['first_actual_sell_close_position_excessive'].cumsum()
                self.data_df['cum_sell_close_position_conservative'] = self.data_df['first_actual_sell_close_position_conservative'].cumsum()
                self.data_df['cum_sell_stop_loss_excessive'] = self.data_df['first_sell_stop_loss_excessive'].cumsum()
                self.data_df['cum_sell_stop_loss_conservative'] = self.data_df['first_sell_stop_loss_conservative'].cumsum()


                self.data_df['cum_buy_close_position_guppy1'] = self.data_df['first_selected_buy_close_position_guppy1'].cumsum()
                self.data_df['cum_buy_close_position_guppy2'] = self.data_df['first_selected_buy_close_position_guppy2'].cumsum()
                self.data_df['cum_buy_close_position_vegas'] = self.data_df['first_selected_buy_close_position_vegas'].cumsum()
                self.data_df['cum_buy_close_position_final_excessive1'] = self.data_df['first_selected_buy_close_position_final_excessive1'].cumsum()
                self.data_df['cum_buy_close_position_final_excessive2'] = self.data_df['first_selected_buy_close_position_final_excessive2'].cumsum()
                self.data_df['cum_buy_close_position_final_conservative'] = self.data_df['first_selected_buy_close_position_final_conservative'].cumsum()
                self.data_df['cum_buy_close_position_final_simple'] = self.data_df['first_selected_buy_close_position_final_simple'].cumsum()
                self.data_df['cum_buy_close_position_final_quick'] = self.data_df['first_selected_buy_close_position_final_quick'].cumsum()
                self.data_df['cum_buy_close_position_final_urgent'] = self.data_df['first_selected_buy_close_position_final_urgent'].cumsum()
                self.data_df['cum_buy_close_position_fixed_time_terminal'] = self.data_df["selected_buy_close_position_fixed_time_terminal"].cumsum()

                self.data_df['cum_special2_buy_close_position'] = self.data_df['first_actual_special_buy_close_position'].cumsum()
                self.data_df['cum_buy_close_position_excessive'] = self.data_df['first_actual_buy_close_position_excessive'].cumsum()
                self.data_df['cum_buy_close_position_conservative'] = self.data_df['first_actual_buy_close_position_conservative'].cumsum()
                self.data_df['cum_buy_stop_loss_excessive'] = self.data_df['first_buy_stop_loss_excessive'].cumsum()
                self.data_df['cum_buy_stop_loss_conservative'] = self.data_df['first_buy_stop_loss_conservative'].cumsum()

                cum_sell_close_cols = [
                                       'cum_sell_close_position_guppy1', 'cum_sell_close_position_guppy2', 'cum_sell_close_position_vegas',
                                       'cum_sell_close_position_final_excessive1', 'cum_sell_close_position_final_excessive2',
                                       'cum_sell_close_position_final_conservative', 'cum_sell_close_position_final_simple',
                                      'cum_sell_close_position_final_quick', 'cum_sell_close_position_final_urgent', 'cum_sell_close_position_fixed_time_terminal',
                                       'cum_special2_sell_close_position', 'cum_sell_close_position_excessive', 'cum_sell_close_position_conservative',
                                      'cum_sell_stop_loss_excessive', 'cum_sell_stop_loss_conservative']

                cum_buy_close_cols = [
                                      'cum_buy_close_position_guppy1', 'cum_buy_close_position_guppy2', 'cum_buy_close_position_vegas',
                                      'cum_buy_close_position_final_excessive1', 'cum_buy_close_position_final_excessive2',
                                       'cum_buy_close_position_final_conservative', 'cum_buy_close_position_final_simple',
                                       'cum_buy_close_position_final_quick', 'cum_buy_close_position_final_urgent', 'cum_buy_close_position_fixed_time_terminal',
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
                    'cum_buy_close_position_guppy1' : 'cum_buy_close_position_guppy1_for_buy',
                    'cum_buy_close_position_guppy2' : 'cum_buy_close_position_guppy2_for_buy',
                    'cum_buy_close_position_vegas' : 'cum_buy_close_position_vegas_for_buy',
                    'cum_buy_close_position_final_excessive1' : 'cum_buy_close_position_final_excessive1_for_buy',
                    'cum_buy_close_position_final_excessive2': 'cum_buy_close_position_final_excessive2_for_buy',
                    'cum_buy_close_position_final_conservative' : 'cum_buy_close_position_final_conservative_for_buy',
                    'cum_buy_close_position_final_simple' : 'cum_buy_close_position_final_simple_for_buy',
                    'cum_buy_close_position_final_quick' : 'cum_buy_close_position_final_quick_for_buy',
                    'cum_buy_close_position_final_urgent': 'cum_buy_close_position_final_urgent_for_buy',
                    'cum_buy_close_position_fixed_time_terminal' : 'cum_buy_close_position_fixed_time_terminal_for_buy',
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
                    'cum_sell_close_position_guppy1' : 'cum_sell_close_position_guppy1_for_sell',
                    'cum_sell_close_position_guppy2' : 'cum_sell_close_position_guppy2_for_sell',
                    'cum_sell_close_position_vegas' : 'cum_sell_close_position_vegas_for_sell',
                    'cum_sell_close_position_final_excessive1' : 'cum_sell_close_position_final_excessive1_for_sell',
                    'cum_sell_close_position_final_excessive2': 'cum_sell_close_position_final_excessive2_for_sell',
                    'cum_sell_close_position_final_conservative' : 'cum_sell_close_position_final_conservative_for_sell',
                    'cum_sell_close_position_final_simple' : 'cum_sell_close_position_final_simple_for_sell',
                    'cum_sell_close_position_final_quick' : 'cum_sell_close_position_final_quick_for_sell',
                    'cum_sell_close_position_final_urgent': 'cum_sell_close_position_final_urgent_for_sell',
                    'cum_sell_close_position_fixed_time_terminal' : 'cum_sell_close_position_fixed_time_terminal_for_sell',
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

                self.data_df['num_buy_close_position_guppy1'] = self.data_df['cum_buy_close_position_guppy1'] - self.data_df['cum_buy_close_position_guppy1_for_buy']
                self.data_df['num_buy_close_position_guppy2'] = self.data_df['cum_buy_close_position_guppy2'] - self.data_df['cum_buy_close_position_guppy2_for_buy']
                self.data_df['num_buy_close_position_vegas'] = self.data_df['cum_buy_close_position_vegas'] - self.data_df['cum_buy_close_position_vegas_for_buy']
                self.data_df['num_buy_close_position_final_excessive1'] = self.data_df['cum_buy_close_position_final_excessive1'] - self.data_df['cum_buy_close_position_final_excessive1_for_buy']
                self.data_df['num_buy_close_position_final_excessive2'] = self.data_df['cum_buy_close_position_final_excessive2'] - self.data_df['cum_buy_close_position_final_excessive2_for_buy']
                self.data_df['num_buy_close_position_final_conservative'] = self.data_df['cum_buy_close_position_final_conservative'] - self.data_df['cum_buy_close_position_final_conservative_for_buy']
                self.data_df['num_buy_close_position_final_simple'] = self.data_df['cum_buy_close_position_final_simple'] - self.data_df['cum_buy_close_position_final_simple_for_buy']
                self.data_df['num_buy_close_position_final_quick'] = self.data_df['cum_buy_close_position_final_quick'] - self.data_df['cum_buy_close_position_final_quick_for_buy']
                self.data_df['num_buy_close_position_final_urgent'] = self.data_df['cum_buy_close_position_final_urgent'] - self.data_df['cum_buy_close_position_final_urgent_for_buy']
                self.data_df['num_buy_close_position_fixed_time_terminal'] = self.data_df['cum_buy_close_position_fixed_time_terminal'] - self.data_df['cum_buy_close_position_fixed_time_terminal_for_buy']


                self.data_df['num_special_buy_close_position'] = self.data_df['cum_special2_buy_close_position'] - self.data_df['cum_special2_buy_close_position_for_buy']
                self.data_df['num_buy_close_position_excessive'] = self.data_df['cum_buy_close_position_excessive'] - self.data_df['cum_buy_close_position_excessive_for_buy']
                self.data_df['num_buy_close_position_conservative'] = self.data_df['cum_buy_close_position_conservative'] - self.data_df['cum_buy_close_position_conservative_for_buy']
                self.data_df['num_buy_stop_loss_excessive'] = self.data_df['cum_buy_stop_loss_excessive'] - self.data_df['cum_buy_stop_loss_excessive_for_buy']
                self.data_df['num_buy_stop_loss_conservative'] = self.data_df['cum_buy_stop_loss_conservative'] - self.data_df['cum_buy_stop_loss_conservative_for_buy']

                self.data_df['num_temporary_buy_close_position'] = self.data_df['num_special_buy_close_position'] + self.data_df['num_buy_close_position_excessive'] \
                                                                     + self.data_df['num_buy_stop_loss_excessive']
                self.data_df['num_terminal_buy_close_position'] = self.data_df['num_buy_close_position_conservative'] + \
                                                                  self.data_df['num_buy_stop_loss_conservative'] + \
                (self.data_df['num_buy_close_position_final_excessive2'] if support_half_stop_loss else self.data_df['num_buy_close_position_final_excessive1']) +\
                                        self.data_df['num_buy_close_position_final_conservative'] + self.data_df['num_buy_close_position_final_simple'] +\
                                        self.data_df['num_buy_close_position_final_quick'] +  self.data_df['num_buy_close_position_final_urgent'] +\
                                        self.data_df['num_buy_close_position_fixed_time_terminal']




                self.data_df['num_sell_close_position_guppy1'] = self.data_df['cum_sell_close_position_guppy1'] - self.data_df['cum_sell_close_position_guppy1_for_sell']
                self.data_df['num_sell_close_position_guppy2'] = self.data_df['cum_sell_close_position_guppy2'] - self.data_df['cum_sell_close_position_guppy2_for_sell']
                self.data_df['num_sell_close_position_vegas'] = self.data_df['cum_sell_close_position_vegas'] - self.data_df['cum_sell_close_position_vegas_for_sell']
                self.data_df['num_sell_close_position_final_excessive1'] = self.data_df['cum_sell_close_position_final_excessive1'] - self.data_df['cum_sell_close_position_final_excessive1_for_sell']
                self.data_df['num_sell_close_position_final_excessive2'] = self.data_df['cum_sell_close_position_final_excessive2'] - self.data_df['cum_sell_close_position_final_excessive2_for_sell']
                self.data_df['num_sell_close_position_final_conservative'] = self.data_df['cum_sell_close_position_final_conservative'] - self.data_df['cum_sell_close_position_final_conservative_for_sell']
                self.data_df['num_sell_close_position_final_simple'] = self.data_df['cum_sell_close_position_final_simple'] - self.data_df['cum_sell_close_position_final_simple_for_sell']
                self.data_df['num_sell_close_position_final_quick'] = self.data_df['cum_sell_close_position_final_quick'] - self.data_df['cum_sell_close_position_final_quick_for_sell']
                self.data_df['num_sell_close_position_final_urgent'] = self.data_df['cum_sell_close_position_final_urgent'] - self.data_df['cum_sell_close_position_final_urgent_for_sell']
                self.data_df['num_sell_close_position_fixed_time_terminal'] = self.data_df['cum_sell_close_position_fixed_time_terminal'] - self.data_df['cum_sell_close_position_fixed_time_terminal_for_sell']



                self.data_df['num_special_sell_close_position'] = self.data_df['cum_special2_sell_close_position'] - self.data_df['cum_special2_sell_close_position_for_sell']
                self.data_df['num_sell_close_position_excessive'] = self.data_df['cum_sell_close_position_excessive'] - self.data_df['cum_sell_close_position_excessive_for_sell']
                self.data_df['num_sell_close_position_conservative'] = self.data_df['cum_sell_close_position_conservative'] - self.data_df['cum_sell_close_position_conservative_for_sell']
                self.data_df['num_sell_stop_loss_excessive'] = self.data_df['cum_sell_stop_loss_excessive'] - self.data_df['cum_sell_stop_loss_excessive_for_sell']
                self.data_df['num_sell_stop_loss_conservative'] = self.data_df['cum_sell_stop_loss_conservative'] - self.data_df['cum_sell_stop_loss_conservative_for_sell']

                self.data_df['num_temporary_sell_close_position'] = self.data_df['num_special_sell_close_position'] + self.data_df['num_sell_close_position_excessive'] \
                                                                     + self.data_df['num_sell_stop_loss_excessive']
                self.data_df['num_terminal_sell_close_position'] = self.data_df['num_sell_close_position_conservative'] + \
                                                                  self.data_df['num_sell_stop_loss_conservative'] + \
                    (self.data_df['num_sell_close_position_final_excessive2'] if support_half_stop_loss else self.data_df['num_sell_close_position_final_excessive1']) +\
                                        self.data_df['num_sell_close_position_final_conservative'] + self.data_df['num_sell_close_position_final_simple'] + \
                                        self.data_df['num_sell_close_position_final_quick'] + self.data_df['num_sell_close_position_final_urgent'] +\
                                        self.data_df['num_sell_close_position_fixed_time_terminal']



                self.data_df['show_buy_close_position_guppy1'] = self.data_df['first_selected_buy_close_position_guppy1'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                                # (self.data_df['num_buy_close_position_guppy'] == 0) &\
                                                                # (self.data_df['num_buy_close_position_vegas'] == 0) &\

                self.data_df['show_buy_close_position_guppy2'] = self.data_df['first_selected_buy_close_position_guppy2'] & (self.data_df['num_terminal_buy_close_position'] == 0)



                self.data_df['show_buy_close_position_vegas'] = self.data_df['first_selected_buy_close_position_vegas'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                                 #(self.data_df['num_buy_close_position_vegas'] <= 1) & \



                self.data_df['show_buy_close_position_final_excessive1'] = self.data_df['first_selected_buy_close_position_final_excessive1'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                    #((self.data_df['num_buy_close_position_guppy'] > 0) | (self.data_df['num_buy_close_position_vegas'] > 0)) &\
                self.data_df['show_buy_close_position_final_excessive2'] = self.data_df['first_selected_buy_close_position_final_excessive2'] & (self.data_df['num_terminal_buy_close_position'] == 0)



                self.data_df['show_buy_close_position_final_conservative'] = self.data_df['first_selected_buy_close_position_final_conservative'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                                                                             #((self.data_df['num_buy_close_position_guppy'] == 0) & (self.data_df['num_buy_close_position_vegas'] == 0)) &\


                self.data_df['show_buy_close_position_final_simple'] = self.data_df['first_selected_buy_close_position_final_simple'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                self.data_df['show_buy_close_position_final_quick'] = self.data_df['first_selected_buy_close_position_final_quick'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                self.data_df['show_buy_close_position_final_urgent'] = self.data_df['first_selected_buy_close_position_final_urgent'] & (self.data_df['num_terminal_buy_close_position'] == 0)
                self.data_df['show_buy_close_position_fixed_time_temporary'] = self.data_df["selected_buy_close_position_fixed_time_temporary"] & (self.data_df['num_terminal_buy_close_position'] == 0)
                self.data_df['show_buy_close_position_fixed_time_terminal'] = self.data_df["selected_buy_close_position_fixed_time_terminal"] & (self.data_df['num_terminal_buy_close_position'] == 0)




                self.data_df['show_special_buy_close_position'] = self.data_df['first_actual_special_buy_close_position'] & \
                                                                  (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)
                self.data_df['show_buy_close_position_excessive'] = \
                    self.data_df['first_actual_buy_close_position_excessive'] & (self.data_df['num_temporary_buy_close_position'] < 4) &\
                    (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)

                self.data_df['show_buy_close_position_excessive_terminal'] = self.data_df['show_buy_close_position_excessive'] &\
                                                                              (self.data_df['num_temporary_buy_close_position'] == 3)


                self.data_df['show_buy_close_position_conservative'] = \
                    self.data_df['first_actual_buy_close_position_conservative'] & (self.data_df['num_temporary_buy_close_position'] < 4) &\
                    (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)

                self.data_df['show_buy_stop_loss_excessive'] = \
                    self.data_df['first_buy_stop_loss_excessive'] & (self.data_df['num_temporary_buy_close_position'] < 4) &\
                    (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)

                self.data_df['show_buy_stop_loss_excessive_terminal'] =  self.data_df['show_buy_stop_loss_excessive'] &\
                                                                          (self.data_df['num_temporary_buy_close_position'] == 3)


                self.data_df['show_buy_stop_loss_conservative'] = \
                    self.data_df['first_buy_stop_loss_conservative'] & (self.data_df['num_temporary_buy_close_position'] < 4) &\
                    (self.data_df['num_terminal_buy_close_position'] == 0) & (self.data_df['buy_point_id'] > 0)




                self.data_df['show_sell_close_position_guppy1'] = self.data_df['first_selected_sell_close_position_guppy1'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                                # (self.data_df['num_sell_close_position_guppy'] == 0) &\
                                                                # (self.data_df['num_sell_close_position_vegas'] == 0) &\

                self.data_df['show_sell_close_position_guppy2'] = self.data_df['first_selected_sell_close_position_guppy2'] & (self.data_df['num_terminal_sell_close_position'] == 0)



                self.data_df['show_sell_close_position_vegas'] = self.data_df['first_selected_sell_close_position_vegas'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                                 #(self.data_df['num_sell_close_position_vegas'] <= 1) & \



                self.data_df['show_sell_close_position_final_excessive1'] = self.data_df['first_selected_sell_close_position_final_excessive1'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                    #((self.data_df['num_sell_close_position_guppy'] > 0) | (self.data_df['num_sell_close_position_vegas'] > 0)) &\
                self.data_df['show_sell_close_position_final_excessive2'] = self.data_df['first_selected_sell_close_position_final_excessive2'] & (self.data_df['num_terminal_sell_close_position'] == 0)



                self.data_df['show_sell_close_position_final_conservative'] = self.data_df['first_selected_sell_close_position_final_conservative'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                                                                             #((self.data_df['num_sell_close_position_guppy'] == 0) & (self.data_df['num_sell_close_position_vegas'] == 0)) &\

                self.data_df['show_sell_close_position_final_simple'] = self.data_df['first_selected_sell_close_position_final_simple'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                self.data_df['show_sell_close_position_final_quick'] = self.data_df['first_selected_sell_close_position_final_quick'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                self.data_df['show_sell_close_position_final_urgent'] = self.data_df['first_selected_sell_close_position_final_urgent'] & (self.data_df['num_terminal_sell_close_position'] == 0)
                self.data_df['show_sell_close_position_fixed_time_temporary'] = self.data_df["selected_sell_close_position_fixed_time_temporary"] & (self.data_df['num_terminal_sell_close_position'] == 0)
                self.data_df['show_sell_close_position_fixed_time_terminal'] = self.data_df["selected_sell_close_position_fixed_time_terminal"] & (self.data_df['num_terminal_sell_close_position'] == 0)





                self.data_df['show_special_sell_close_position'] = self.data_df['first_actual_special_sell_close_position'] & \
                                                                      (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)
                self.data_df['show_sell_close_position_excessive'] = \
                    self.data_df['first_actual_sell_close_position_excessive'] & (self.data_df['num_temporary_sell_close_position'] < 4) &\
                    (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)

                self.data_df['show_sell_close_position_excessive_terminal'] = self.data_df['show_sell_close_position_excessive'] &\
                                                                              (self.data_df['num_temporary_sell_close_position'] == 3)

                self.data_df['show_sell_close_position_conservative'] = \
                    self.data_df['first_actual_sell_close_position_conservative'] & (self.data_df['num_temporary_sell_close_position'] < 4) &\
                     (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)

                self.data_df['show_sell_stop_loss_excessive'] = \
                    self.data_df['first_sell_stop_loss_excessive'] & (self.data_df['num_temporary_sell_close_position'] < 4) &\
                    (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)

                self.data_df['show_sell_stop_loss_excessive_terminal'] =  self.data_df['show_sell_stop_loss_excessive'] &\
                                                                          (self.data_df['num_temporary_sell_close_position'] == 3)

                self.data_df['show_sell_stop_loss_conservative'] = \
                    self.data_df['first_sell_stop_loss_conservative'] & (self.data_df['num_temporary_sell_close_position'] < 4) &\
                    (self.data_df['num_terminal_sell_close_position'] == 0) & (self.data_df['sell_point_id'] > 0)



                self.data_df = self.data_df.drop(columns = [col for col in self.data_df.columns if 'temp' in col and 'temporary' not in col])


                #
                # print("Debug here:")
                # print(self.data_df[self.data_df['time'] == datetime(2021,6,1,7,0,0)][['time', 'show_sell_close_position_final_excessive1']])
                #




            ################################################################################################


            #Write here
            if self.is_cut_data:
                data_df_backup = self.data_dfs_backup[0]
                increment_data_df = self.data_df[self.data_df['time'] > data_df_backup.iloc[-1]['time']]
                if increment_data_df.shape[0] > 0:

                    self.data_df = pd.concat([data_df_backup, increment_data_df])

                    self.data_df.reset_index(inplace = True)
                    self.data_df = self.data_df.drop(columns = ['index'])


                else:

                    self.data_df = data_df_backup






            ################################ Calculate Positions ########################

            print("Calculate positions.........")

            open_buy_positions = ['first_final_buy_fire']
            open_sell_positions = ['first_final_sell_fire']


            close_buy_positions = ['show_buy_close_position_guppy1', 'show_buy_close_position_guppy2',
                                   'show_buy_close_position_vegas', 'show_buy_close_position_final_excessive1', 'show_buy_close_position_final_conservative',
                                   'show_buy_close_position_final_quick', 'show_buy_close_position_final_urgent',
                                   'show_buy_close_position_fixed_time_temporary', 'show_buy_close_position_fixed_time_terminal',
                                    'selected_buy_close_position_fixed_time',
                                   'show_special_buy_close_position', 'show_buy_close_position_excessive', 'show_buy_close_position_conservative',
                                   'show_buy_stop_loss_excessive', 'show_buy_stop_loss_conservative']

            close_sell_positions = ['show_sell_close_position_guppy1', 'show_sell_close_position_guppy2',
                                   'show_sell_close_position_vegas', 'show_sell_close_position_final_excessive1', 'show_sell_close_position_final_conservative',
                                   'show_sell_close_position_final_quick', 'show_sell_close_position_final_urgent',
                                    'show_sell_close_position_fixed_time_temporary', 'show_sell_close_position_fixed_time_terminal',
                                    'selected_sell_close_position_fixed_time',
                                   'show_special_sell_close_position', 'show_sell_close_position_excessive', 'show_sell_close_position_conservative',
                                   'show_sell_stop_loss_excessive', 'show_sell_stop_loss_conservative']


            select_conditions_for_buy = reduce(lambda left, right: left | right, [self.data_df[condition] for condition in open_buy_positions + close_buy_positions])
            select_conditions_for_sell = reduce(lambda left, right: left | right, [self.data_df[condition] for condition in open_sell_positions + close_sell_positions])

            data_df_buy = self.data_df[select_conditions_for_buy][['time', 'id', 'buy_point_id'] + open_buy_positions + close_buy_positions +\
                ['show_buy_close_position_excessive_terminal', 'show_buy_stop_loss_excessive_terminal']]
            data_df_sell = self.data_df[select_conditions_for_sell][['time', 'id', 'sell_point_id'] + open_sell_positions + close_sell_positions +\
                ['show_sell_close_position_excessive_terminal', 'show_sell_stop_loss_excessive_terminal']]

            data_df_buy.reset_index(inplace = True)
            data_df_buy = data_df_buy.drop(columns = ['index'])

            data_df_sell.reset_index(inplace = True)
            data_df_sell = data_df_sell.drop(columns = ['index'])



            print("Calculate buy position:")
            self.calculate_position(data_df_buy, 'buy')

            print("")
            print("Calculate sell position:")
            self.calculate_position(data_df_sell, 'sell')

            data_df_buy = data_df_buy.rename(columns = {'position' : 'buy_position'})
            data_df_sell = data_df_sell.rename(columns = {'position' : 'sell_position'})
            data_df_sell['sell_position'] = -data_df_sell['sell_position']

            data_df_buy['cum_buy_position'] = data_df_buy['buy_position'].cumsum()
            data_df_buy['cum_buy_position'] = data_df_buy['cum_buy_position'].apply(lambda x: round(x, 2))

            data_df_sell['cum_sell_position'] = data_df_sell['sell_position'].cumsum()
            data_df_sell['cum_sell_position'] = data_df_sell['cum_sell_position'].apply(lambda x: round(x, 2))

            # print("buy_positions.............")
            # print(data_df_buy[['time','id','buy_point_id','buy_position', 'cum_buy_position']])
            #
            # print("")
            # print("sell_positions.............")
            # print(data_df_sell[['time', 'id', 'sell_point_id', 'sell_position', 'cum_sell_position']])



            intermediate_df = self.data_df[['time','id']]

            intermediate_df_buy = pd.merge(intermediate_df, data_df_buy[['id', 'buy_position']], on = ['id'], how = 'left')
            intermediate_df_buy = intermediate_df_buy.fillna(0)

            intermediate_df_sell = pd.merge(intermediate_df, data_df_sell[['id', 'sell_position']], on = ['id'], how = 'left')
            intermediate_df_sell = intermediate_df_sell.fillna(0)

            self.data_df['buy_position'] = intermediate_df_buy['buy_position']
            self.data_df['sell_position'] = intermediate_df_sell['sell_position']

            self.data_df['cum_buy_position'] = self.data_df['buy_position'].cumsum()
            self.data_df['cum_sell_position'] = self.data_df['sell_position'].cumsum()

            self.data_df['position'] = self.data_df['buy_position'] + self.data_df['sell_position']
            self.data_df['cum_position'] = self.data_df['position'].cumsum()

            self.data_df['cum_position'] = self.data_df['cum_position'].apply(lambda x: round(x, 2))


            #############################################################################


            print("to csv:")

            #self.data_df['time_correct'] = self.data_df['time'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

            # print("Debug data_df here:")
            # print(self.data_df[['time', 'open', 'high', 'low', 'close']].tail(3))
            #print("Time type: " + str(type(self.data_df.iloc[-1]['time'])))

            #self.data_df.tail(3).to_csv(os.path.join(self.data_folder, self.currency + str(100) + ".temp.csv"), index=False)

            self.data_df.to_csv(os.path.join(self.data_folder, self.currency + str(100) + ".csv"), index=False)
            print("after to csv:")
            #sys.exit(0)



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

                additional_msg =" Long " + str(round(self.data_df.iloc[-1]['buy_position'] * possition_factor, 2)) + " lot" # " Exit if next two bars are both negative" if buy_c2_aux.iloc[-1] else ""

                if additional_msg != "":
                    self.log_msg(additional_msg)
                self.log_msg("********************************")

                #if (self.data_df.iloc[-1]['first_buy_real_fire2'] | self.data_df.iloc[-1]['first_buy_real_fire3']):

                delta_point = (self.data_df.iloc[-1]['close'] - self.data_df.iloc[-1]['period_low' + str(high_low_window_options[0])]) * self.lot_size * self.exchange_rate

                stop_loss_msg = " Stop loss at " + str(delta_point) + " points below open price"
                #else:
                #    stop_loss_msg = ""

                if is_send_email:
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


            if self.data_df.iloc[-1]['show_sell_close_position_guppy1'] or self.data_df.iloc[-1]['show_sell_close_position_guppy2']:
                msg = "Close Short Position Phase 1 based on Guppy for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_sell_close_position_vegas']:
                msg = "Close Short Position Phase 1 based on vegas for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_sell_close_position_final_excessive1']:
                msg = "Close Short Position Phase 1 Stop loss excessive for " + self.currency + " at " + current_time
                #add_msg = " Close 1/2 of remaining position" if support_half_stop_loss else " Close all remaining position"
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_sell_close_position_final_excessive2']:
                msg = "Close Short Position Phase 1 Stop loss excessive for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_sell_close_position_final_conservative']:
                msg = "Close Short Position Phase 1 Stop loss conservative for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_sell_close_position_final_simple']:
                msg = "Close Short Position Phase 1 Stop loss simple for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_sell_close_position_final_quick']:
                msg = "Close Short Position Phase 1 Stop loss quick for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_sell_close_position_final_urgent']:
                msg = "Close Short Position Phase 1 Stop loss urgent for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_sell_close_position_fixed_time_temporary']:
                msg = "Close Short Position Phase 1 Fixed Time temporary for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_sell_close_position_fixed_time_terminal']:
                msg = "Close Short Position Phase 1 Fixed Time terminal for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")





            if self.data_df.iloc[-1]['show_special_sell_close_position']:
                msg = "Close Short Position Phase 2 Special for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_sell_close_position_excessive']:
                msg = "Close Short Position Phase 2 Excessive for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_sell_close_position_conservative']:
                msg = "Close Short Position Phase 2 Conservative for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_sell_stop_loss_excessive']:
                msg = "Close Short Position Phase 2 Stop loss excessive for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_sell_stop_loss_conservative']:
                msg = "Close Short Position Phase 2 Stop loss conservative for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot")




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

                additional_msg = " Short " + str(round(-self.data_df.iloc[-1]['sell_position']*possition_factor, 2)) + " lot" # " Exit if next two bars are both positive" if sell_c2_aux.iloc[-1] else ""

                if additional_msg != "":
                    self.log_msg(additional_msg)
                self.log_msg("********************************")

                #if (self.data_df.iloc[-1]['first_sell_real_fire2'] | self.data_df.iloc[-1]['first_sell_real_fire3']):

                delta_point = (-self.data_df.iloc[-1]['close'] + self.data_df.iloc[-1]['period_high' + str(high_low_window_options[0])]) * self.lot_size * self.exchange_rate

                stop_loss_msg = " Stop loss at " + str(delta_point) + " points above open price"
                #else:
                #    stop_loss_msg = ""
                if is_send_email:
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


            if self.data_df.iloc[-1]['show_buy_close_position_guppy1'] or self.data_df.iloc[-1]['show_buy_close_position_guppy2']:
                msg = "Close Long Position Phase 1 based on Guppy for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_buy_close_position_vegas']:
                msg = "Close Long Position Phase 1 based on vegas for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_buy_close_position_final_excessive1']:
                msg = "Close Long Position Phase 1 Stop loss excessive for " + self.currency + " at " + current_time
                #add_msg = " Close 1/2 of remaining position" if support_half_stop_loss else " Close all remaining position"
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_buy_close_position_final_excessive2']:
                msg = "Close Long Position Phase 1 Stop loss excessive for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_buy_close_position_final_conservative']:
                msg = "Close Long Position Phase 1 Stop loss conservative for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_buy_close_position_final_simple']:
                msg = "Close Long Position Phase 1 Stop loss simple for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_buy_close_position_final_quick']:
                msg = "Close Long Position Phase 1 Stop loss quick for " + self.currency + " at " + current_time
                #print("  ######Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")


            if self.data_df.iloc[-1]['show_buy_close_position_final_urgent']:
                msg = "Close Long Position Phase 1 Stop loss urgent for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")


            if self.data_df.iloc[-1]['show_buy_close_position_fixed_time_temporary']:
                msg = "Close Long Position Phase 1 Fixed Time temporary for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")

            if self.data_df.iloc[-1]['show_buy_close_position_fixed_time_terminal']:
                msg = "Close Long Position Phase 1 Fixed Time terminal for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")




            if self.data_df.iloc[-1]['show_special_buy_close_position']:
                msg = "Close Long Position Phase 2 Special for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_buy_close_position_excessive']:
                msg = "Close Long Position Phase 2 Excessive for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_buy_close_position_conservative']:
                msg = "Close Long Position Phase 2 Conservative for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_buy_stop_loss_excessive']:
                msg = "Close Long Position Phase 2 Stop loss excessive for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")
            if self.data_df.iloc[-1]['show_buy_stop_loss_conservative']:
                msg = "Close Long Position Phase 2 Stop loss conservative for " + self.currency + " at " + current_time
                if is_send_email:
                    sendEmail(msg, msg + "  Close " + str(round(-self.data_df.iloc[-1]['buy_position']*possition_factor, 2)) + " lot")


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











