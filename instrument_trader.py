

is_production = False


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#import talib

import math
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

#from vegas_strategy_once import profit_loss_ratio
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

initial_bar_number = 500

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
price_range_average_look_back = 3

is_plot_exclude = True

high_low_delta_threshold = 20.001

entry_risk_threshold = 0.6

close_position_look_back = 12

is_send_email = True

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


only_second_entry = False
use_second_entry = False
trend_follow = False
#################
is_clean_redundant_entry_point = only_second_entry
is_only_allow_second_entry = only_second_entry

is_activate_second_entry_trading = only_second_entry
is_second_entry_reentry = only_second_entry
###################


data_file_suffix = ""
# if only_second_entry and use_second_entry:
#     data_file_suffix += 'only_second_entry'
#
# if trend_follow:
#     if only_second_entry and use_second_entry:
#         data_file_suffix += '_trend_follow'
#     else:
#         data_file_suffix += 'trend_follow'

    #only_second_entry_trend_follow



is_activate_second_entry_reentry = is_activate_second_entry_trading and is_second_entry_reentry

aligned_conditions21_threshold = 5  #5 by default


is_use_two_trend_following = False

use_dynamic_TP = True

printed_figure_num = -1

unit_loss = 150 #This is HKD
usdhkd = 7.85
leverage = 100

tp_tolerance = 0

class CurrencyTrader(threading.Thread):

    def __init__(self, condition, currency, lot_size, exchange_rate, coefficient,  data_folder, chart_folder, simple_chart_folder, log_file, data_file, trade_file, performance_file, usdfx, is_notify):
        super().__init__(name = currency)
        self.condition = condition
        self.currency = currency
        self.lot_size = lot_size
        self.exchange_rate = exchange_rate
        self.coefficient = coefficient
        self.data_folder = data_folder
        self.chart_folder = chart_folder
        self.simple_chart_folder = simple_chart_folder
        self.data_df = None
        self.last_time = None
        self.log_file = log_file
        self.data_file = data_file
        self.trade_file = trade_file
        self.performance_file = performance_file
        self.usdfx = usdfx
        self.is_notify = is_notify

        self.long_df = None
        self.short_df = None

        # self.use_relaxed_vegas_support = True
        # self.is_require_m12_strictly_above_vegas = False
        # self.remove_c12 = True

        #self.currency_file = os.path.join(data_folder, currency + "100.csv")

        self.log_fd = open(self.log_file, 'a')

        self.print_to_console = True
        #
        # self.is_cut_data = False
        #
        # self.data_df_backup100 = None
        # self.data_df_backup200 = None
        #
        # self.data_dfs_backup = []


        self.log_msg("Initializing...")


    def log_msg(self, msg):

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #current_time = (datetime.now() + timedelta(seconds = 28800)).strftime("%Y-%m-%d %H:%M:%S")
        print('[' + current_time + ' ' + self.currency + ']  ' + msg, file = self.log_fd)
        self.log_fd.flush()

        if self.print_to_console:
            print('[' + current_time + ' ' + self.currency + ']  ' + msg)


    def feed_data(self, new_data_df):

        self.data_df = new_data_df

    def run(self):
        print("Running...........")
        self.trade()


    def round_price(self, price):

        if 'JPY' in self.currency:
            return round(price, 3)
        else:
            return round(price, 5)


    def calculate_signals(self):

        self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()
        self.data_df['hour'] = self.data_df['time'].apply(lambda x: x.hour)

        calc_jc_lines(self.data_df, "close", windows)

        if not is_production:
            calc_bolling_bands(self.data_df, "close", bolling_width)

        self.data_df['prev_open'] = self.data_df['open'].shift(1)

        self.data_df['prev_ma_close12'] = self.data_df['ma_close12'].shift(1)

        self.data_df['min_price'] = self.data_df[['open', 'close']].min(axis=1)
        self.data_df['max_price'] = self.data_df[['open', 'close']].max(axis=1)

        self.data_df['middle_price'] = (self.data_df['open'] + self.data_df['close']) / 2.0


        self.data_df['is_positive'] = (self.data_df['close'] > self.data_df['open'])
        self.data_df['is_negative'] = (self.data_df['close'] < self.data_df['open'])

        self.data_df['price_range'] = self.data_df['max_price'] - self.data_df['min_price']
        self.data_df['price_volatility'] = self.data_df['high'] - self.data_df['low']

        guppy_lines = ['ma_close30', 'ma_close35', 'ma_close40', 'ma_close45', 'ma_close50', 'ma_close60']
        for guppy_line in guppy_lines:
            self.data_df[guppy_line + '_gradient'] = self.data_df[guppy_line].diff()

        for guppy_line in guppy_lines:
            self.data_df[guppy_line + '_up'] = np.where(
                self.data_df[guppy_line + '_gradient'] > 0,
                1,
                0
            )

        for guppy_line in guppy_lines:
            self.data_df[guppy_line + '_down'] = np.where(
                self.data_df[guppy_line + '_gradient'] < 0,
                1,
                0
            )

        self.data_df['guppy_first_half_min'] = self.data_df[[guppy_lines[0], guppy_lines[1], guppy_lines[2]]].min(axis = 1)
        self.data_df['guppy_first_half_max'] = self.data_df[[guppy_lines[0], guppy_lines[1], guppy_lines[2]]].max(axis = 1)

        self.data_df['guppy_second_half_min'] = self.data_df[[guppy_lines[3], guppy_lines[4], guppy_lines[5]]].min(axis = 1)
        self.data_df['guppy_second_half_max'] = self.data_df[[guppy_lines[3], guppy_lines[4], guppy_lines[5]]].max(axis = 1)


        # self.data_df['guppy_first_half_min'] = guppy_lines[0]
        # self.data_df['guppy_first_half_max'] = guppy_lines[0]
        #
        # self.data_df['guppy_second_half_min'] = guppy_lines[5]
        # self.data_df['guppy_second_half_max'] = guppy_lines[5]



        self.data_df['fastest_guppy_line_up'] = self.data_df['ma_close30_gradient'] > 0
        self.data_df['fastest_guppy_line_down'] = self.data_df['ma_close30_gradient'] < 0

        self.data_df['pre_fastest_guppy_line_up'] = self.data_df['fastest_guppy_line_up'].shift(1)
        self.data_df['pre_fastest_guppy_line_down'] = self.data_df['fastest_guppy_line_down'].shift(1)

        self.data_df['pp_fastest_guppy_line_up'] = self.data_df['pre_fastest_guppy_line_up'].shift(1)
        self.data_df['pp_fastest_guppy_line_down'] = self.data_df['pre_fastest_guppy_line_down'].shift(1)

        self.data_df['fastest_guppy_line_lasting_up'] = (self.data_df['fastest_guppy_line_up']) &\
                                                        (self.data_df['pre_fastest_guppy_line_up']) & (self.data_df['pp_fastest_guppy_line_up'])

        self.data_df['fastest_guppy_line_lasting_down'] = (self.data_df['fastest_guppy_line_down']) &\
                                                          (self.data_df['pre_fastest_guppy_line_down']) & (self.data_df['pp_fastest_guppy_line_down'])



        self.data_df['fast_guppy_cross_up'] = self.data_df['ma_close30'] > self.data_df['ma_close35']
        self.data_df['fast_guppy_cross_down'] = self.data_df['ma_close30'] < self.data_df['ma_close35']

        self.data_df['up_guppy_line_num'] = reduce(lambda left, right: left + right,
                                                   [self.data_df[guppy_line + '_up'] for guppy_line in guppy_lines])

        self.data_df['previous_up_guppy_line_num'] = self.data_df['up_guppy_line_num'].shift(1)

        self.data_df['down_guppy_line_num'] = reduce(lambda left, right: left + right,
                                                   [self.data_df[guppy_line + '_down'] for guppy_line in guppy_lines])  #Used to be up, big bug

        self.data_df['previous_down_guppy_line_num'] = self.data_df['down_guppy_line_num'].shift(1)

        guppy_aligned_long_conditions = [(self.data_df[guppy_lines[i]] > self.data_df[guppy_lines[i + 1]]) for i in
                                    range(len(guppy_lines) - 1)]

        guppy_up_conditions = [self.data_df[guppy_lines[i] + '_up']
                                          for i in range(len(guppy_lines))]

        self.data_df['guppy_all_aligned_long'] = reduce(lambda left, right: left & right, guppy_aligned_long_conditions)
        self.data_df['guppy_all_up'] = reduce(lambda left, right: left & right, guppy_up_conditions)
        self.data_df['guppy_all_strong_aligned_long'] = self.data_df['guppy_all_aligned_long'] & self.data_df['guppy_all_up']

        self.data_df['guppy_half1_aligned_long'] = reduce(lambda left, right: left & right, guppy_aligned_long_conditions[0:2])
        self.data_df['guppy_half1_all_up'] = reduce(lambda left, right: left & right, guppy_up_conditions[0:3])
        self.data_df['guppy_half1_strong_aligned_long'] = self.data_df['guppy_half1_aligned_long'] & self.data_df['guppy_half1_all_up']
        self.data_df['prev_guppy_half1_strong_aligned_long'] = self.data_df['guppy_half1_strong_aligned_long'].shift(1).fillna(method = 'bfill')
        self.data_df['prev2_guppy_half1_strong_aligned_long'] = self.data_df['prev_guppy_half1_strong_aligned_long'].shift(1).fillna(method = 'bfill')

        self.data_df['guppy_half2_aligned_long'] = reduce(lambda left, right: left & right, guppy_aligned_long_conditions[3:5])
        self.data_df['guppy_half2_all_up'] = reduce(lambda left, right: left & right, guppy_up_conditions[3:6])
        self.data_df['guppy_half2_strong_aligned_long'] = self.data_df['guppy_half2_aligned_long'] & self.data_df['guppy_half2_all_up']


        guppy_aligned_short_conditions = [(self.data_df[guppy_lines[i]] < self.data_df[guppy_lines[i + 1]]) for i in
                                    range(len(guppy_lines) - 1)]

        guppy_down_conditions = [self.data_df[guppy_lines[i] + '_down']
                                          for i in range(len(guppy_lines))]

        self.data_df['guppy_all_aligned_short'] = reduce(lambda left, right: left & right, guppy_aligned_short_conditions)
        self.data_df['guppy_all_down'] = reduce(lambda left, right: left & right, guppy_down_conditions)
        self.data_df['guppy_all_strong_aligned_short'] = self.data_df['guppy_all_aligned_short'] & self.data_df['guppy_all_down']

        self.data_df['guppy_half1_aligned_short'] = reduce(lambda left, right: left & right, guppy_aligned_short_conditions[0:2])
        self.data_df['guppy_half1_all_down'] = reduce(lambda left, right: left & right, guppy_down_conditions[0:3])
        self.data_df['guppy_half1_strong_aligned_short'] = self.data_df['guppy_half1_aligned_short'] & self.data_df['guppy_half1_all_down']
        self.data_df['prev_guppy_half1_strong_aligned_short'] = self.data_df['guppy_half1_strong_aligned_short'].shift(1).fillna(method = 'bfill')
        self.data_df['prev2_guppy_half1_strong_aligned_short'] = self.data_df['prev_guppy_half1_strong_aligned_short'].shift(1).fillna(method = 'bfill')

        self.data_df['guppy_half2_aligned_short'] = reduce(lambda left, right: left & right, guppy_aligned_short_conditions[3:5])
        self.data_df['guppy_half2_all_down'] = reduce(lambda left, right: left & right, guppy_down_conditions[3:6])
        self.data_df['guppy_half2_strong_aligned_short'] = self.data_df['guppy_half2_aligned_short'] & self.data_df['guppy_half2_all_down']



        self.data_df['fast_vegas'] = self.data_df['ma_close144']
        self.data_df['slow_vegas'] = self.data_df['ma_close169']

        self.data_df['vegas_distance'] = np.abs(self.data_df['fast_vegas'] - self.data_df['slow_vegas'])
        self.data_df['vegas_distance_gradient'] = self.data_df['vegas_distance'].diff()
        self.data_df['prev_vegas_distance_gradient'] = self.data_df['vegas_distance_gradient'].shift(1)
        self.data_df['pp_vegas_distance_gradient'] = self.data_df['prev_vegas_distance_gradient'].shift(1)


        self.data_df['fast_vegas_gradient'] = self.data_df['fast_vegas'].diff()
        self.data_df['slow_vegas_gradient'] = self.data_df['slow_vegas'].diff()

        self.data_df['previous_fast_vegas_gradient'] = self.data_df['fast_vegas_gradient'].shift(1)
        self.data_df['previous_slow_vegas_gradient'] = self.data_df['slow_vegas_gradient'].shift(1)

        self.data_df['pp_fast_vegas_gradient'] = self.data_df['previous_fast_vegas_gradient'].shift(1)
        self.data_df['pp_slow_vegas_gradient'] = self.data_df['previous_slow_vegas_gradient'].shift(1)


        self.data_df['fast_vegas_up'] = self.data_df['fast_vegas_gradient'] > 0
        self.data_df['fast_vegas_down'] = self.data_df['fast_vegas_gradient'] < 0

        self.data_df['previous_fast_vegas_up'] = self.data_df['previous_fast_vegas_gradient'] > 0
        self.data_df['previous_fast_vegas_down'] = self.data_df['previous_fast_vegas_gradient'] < 0

        self.data_df['pp_fast_vegas_up'] = self.data_df['pp_fast_vegas_gradient'] > 0
        self.data_df['pp_fast_vegas_down'] = self.data_df['pp_fast_vegas_gradient'] < 0



        self.data_df['slow_vegas_up'] = self.data_df['slow_vegas_gradient'] > 0
        self.data_df['slow_vegas_down'] = self.data_df['slow_vegas_gradient'] < 0

        self.data_df['previous_slow_vegas_up'] = self.data_df['previous_slow_vegas_gradient'] > 0
        self.data_df['previous_slow_vegas_down'] = self.data_df['previous_slow_vegas_gradient'] < 0

        self.data_df['pp_slow_vegas_up'] = self.data_df['pp_slow_vegas_gradient'] > 0
        self.data_df['pp_slow_vegas_down'] = self.data_df['pp_slow_vegas_gradient'] < 0


        ###############

        self.data_df['fast_vegas_above'] = self.data_df['fast_vegas'] > self.data_df['slow_vegas']
        self.data_df['fast_vegas_below'] = self.data_df['fast_vegas'] < self.data_df['slow_vegas']

        self.data_df['prev_fast_vegas_above'] = self.data_df['fast_vegas_above'].shift(1)
        self.data_df['prev_fast_vegas_below'] = self.data_df['fast_vegas_below'].shift(1)

        self.data_df['fast_vegas_cross_up'] = (self.data_df['prev_fast_vegas_below']) & (self.data_df['fast_vegas_above'])
        self.data_df['fast_vegas_cross_down'] = (self.data_df['prev_fast_vegas_above']) & (self.data_df['fast_vegas_below'])

        self.data_df['num'] = list(range(self.data_df.shape[0]))
        self.data_df['critical_num'] = np.where(
            (self.data_df['fast_vegas_cross_up']) | (self.data_df['fast_vegas_cross_down']),
            self.data_df['num'],
            np.nan
        )
        self.data_df['critical_num'] = self.data_df['critical_num'].fillna(method='ffill').fillna(0)
        self.data_df['vegas_phase_duration'] = self.data_df['num'] - self.data_df['critical_num']

        self.data_df['prev_vegas_phase_entire_duration'] = self.data_df['vegas_phase_duration'].shift(1).fillna(0)
        self.data_df['prev_vegas_phase_entire_duration'] = np.where(
            (self.data_df['fast_vegas_cross_up']) | (self.data_df['fast_vegas_cross_down']),
            self.data_df['prev_vegas_phase_entire_duration'],
            np.nan
        )
        self.data_df['prev_vegas_phase_entire_duration'] = self.data_df['prev_vegas_phase_entire_duration'].fillna(method = 'ffill').fillna(0)



        ###############



        self.data_df['up_vegas_converge'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
                                            (self.data_df['fast_vegas_gradient'] < self.data_df['slow_vegas_gradient'])
        self.data_df['up_vegas_converge_previous'] = self.data_df['up_vegas_converge'].shift(1)
        self.data_df['up_vegas_converge_pp'] = self.data_df['up_vegas_converge_previous'].shift(1)

        self.data_df['down_vegas_converge'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & \
                                            (self.data_df['fast_vegas_gradient'] > self.data_df['slow_vegas_gradient'])
        self.data_df['down_vegas_converge_previous'] = self.data_df['down_vegas_converge'].shift(1)
        self.data_df['down_vegas_converge_pp'] = self.data_df['down_vegas_converge_previous'].shift(1)

        ########## Long ############

        self.data_df['vegas_support_long'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['fast_vegas_up']) & (self.data_df['slow_vegas_up']) & \
            (~((self.data_df['up_vegas_converge']) & (self.data_df['up_vegas_converge_previous']) & (self.data_df['up_vegas_converge_pp'])))

        self.data_df['long_encourage_condition'] = (self.data_df['fast_guppy_cross_up']) & (self.data_df['fastest_guppy_line_up'])  #'fastest_guppy_line_up'

        ######### Filters for Scenario where Vegas support long ###############

        self.data_df['long_filter1'] = (self.data_df['down_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_down'])   #adjust by removing
        self.data_df['long_filter1'] = (self.data_df['long_filter1']) | (self.data_df['previous_down_guppy_line_num'] >= 3)  #USDCAD Stuff
        self.data_df['long_filter1'] = (self.data_df['long_filter1']) & (~self.data_df['long_encourage_condition'])

        self.data_df['long_filter2'] = (self.data_df['up_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_down']) & (self.data_df['fast_guppy_cross_down'])

        self.data_df['long_strong_filter1'] = (self.data_df['guppy_half1_strong_aligned_short'])
        self.data_df['long_strong_filter2'] = (self.data_df['guppy_half2_aligned_long']) & (self.data_df['fastest_guppy_line_down']) & (self.data_df['fast_guppy_cross_down'])


        self.data_df['guppy_long_reverse'] = (self.data_df['up_guppy_line_num'] >= 3) & (self.data_df['ma_close30_gradient'] < 0)
        self.data_df['prev_guppy_long_reverse'] = self.data_df['guppy_long_reverse'].shift(1)
        self.data_df['prev2_guppy_long_reverse'] = self.data_df['prev_guppy_long_reverse'].shift(1)
        self.data_df['recent_guppy_long_reverse'] = (self.data_df['guppy_long_reverse']) | (self.data_df['prev_guppy_long_reverse']) | (self.data_df['prev2_guppy_long_reverse'])
        #self.data_df['recent_guppy_long_reverse'] = (self.data_df['guppy_long_reverse']) & (self.data_df['prev_guppy_long_reverse']) & (self.data_df['prev2_guppy_long_reverse'])


        self.data_df['can_long1'] = self.data_df['vegas_support_long'] #&\
                                    #(~self.data_df['guppy_half1_strong_aligned_short']) & (~self.data_df['prev_guppy_half1_strong_aligned_short']) & (~self.data_df['prev2_guppy_half1_strong_aligned_short']) #& (~self.data_df['long_filter1']) & (~self.data_df['long_filter2'])  #Modify


        ######## Conditions for Scenario where Vegas does not support long ############### #second condition is EURUSD stuff

        self.data_df['long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) |\
                                         ((self.data_df['guppy_half2_strong_aligned_long'])) |\
                                         (self.data_df['guppy_all_aligned_long']) | (self.data_df['long_encourage_condition'])
        self.data_df['long_condition'] = self.data_df['long_condition'] & (~self.data_df['fastest_guppy_line_lasting_down'])
        self.data_df['long_condition'] = self.data_df['long_condition'] & (self.data_df['guppy_first_half_min'] > self.data_df['guppy_second_half_max'])

        #self.data_df['long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) #Adjust2
        self.data_df['can_long2'] = (~self.data_df['vegas_support_long']) & self.data_df['long_condition']



        self.data_df['final_long_filter1'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] < 48) & (self.data_df['prev_vegas_phase_entire_duration'] < 48) &\
                                              ( ((self.data_df['fast_vegas_down']) & (self.data_df['previous_fast_vegas_down'])) |\
                                             ((self.data_df['slow_vegas_down']) & (self.data_df['previous_slow_vegas_down'])) |\
                                             ((self.data_df['previous_fast_vegas_down']) & (self.data_df['pp_fast_vegas_down'])) |\
                                             ((self.data_df['previous_slow_vegas_down']) & (self.data_df['pp_slow_vegas_down']))
                                             )

        self.data_df['final_long_filter2'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 48)
        self.data_df['long_filter_exempt'] = self.data_df['fast_vegas_up'] & self.data_df['previous_fast_vegas_up'] &\
                                             (self.data_df['vegas_distance_gradient'] < 0) & (self.data_df['prev_vegas_distance_gradient'] < 0)
        #self.data_df['final_long_filter2'] = self.data_df['final_long_filter2'] & (~self.data_df['long_filter_exempt'])

        self.data_df['final_long_filter'] = self.data_df['final_long_filter1'] | self.data_df['final_long_filter2']




        # self.data_df['final_long_filter'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) #&\
        #                                # ( ((self.data_df['fast_vegas_down']) & (self.data_df['previous_fast_vegas_down'])) |\
        #                                #   ((self.data_df['slow_vegas_down']) & (self.data_df['previous_slow_vegas_down'])) |\
        #                                #   ((self.data_df['previous_fast_vegas_down']) & (self.data_df['pp_fast_vegas_down'])) |\
        #                                #   ((self.data_df['previous_slow_vegas_down']) & (self.data_df['pp_slow_vegas_down']))
        #                                #   )
        # self.data_df['long_filter_exempt'] = self.data_df['fast_vegas_up'] & self.data_df['previous_fast_vegas_up']
        # self.data_df['final_long_filter'] = self.data_df['final_long_filter'] & (~self.data_df['long_filter_exempt'])
        #
        # # self.data_df['final_long_filter'] = ((self.data_df['final_long_filter']) & (~self.data_df['long_encourage_condition'])) |\
        # #                                      ((self.data_df['final_long_filter']) & (self.data_df['vegas_phase_duration'] >= 48) & (self.data_df['fast_vegas_below']))
        #
        # # self.data_df['final_long_filter'] = ((self.data_df['final_long_filter']) & (~self.data_df['long_encourage_condition'])) |\
        # #                                      ((self.data_df['final_long_filter']) &\
        # #                                       ((self.data_df['vegas_phase_duration'] >= 48) | (self.data_df['prev_vegas_phase_entire_duration'] < 48)) & (self.data_df['fast_vegas_below']))
        #
        # self.data_df['final_long_filter'] = ((self.data_df['final_long_filter']) &\
        #                                       ((self.data_df['vegas_phase_duration'] >= 48) | (self.data_df['prev_vegas_phase_entire_duration'] < 48)) & (self.data_df['fast_vegas_below']))




        self.data_df['can_long'] = True #(self.data_df['can_long1']) | (self.data_df['can_long2'])
        #self.data_df['can_long'] = (self.data_df['vegas_support_long']) & (self.data_df['long_condition'])  #strong adjust

        self.data_df['can_long'] = (self.data_df['can_long']) & (~self.data_df['final_long_filter']) #USDCAD stuff

        ##############
        self.data_df['final_long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) |\
                                         ((self.data_df['guppy_half2_strong_aligned_long'])) |\
                                         (self.data_df['guppy_all_aligned_long'])
        #self.data_df['final_long_condition'] = self.data_df['final_long_condition'] & (~self.data_df['fastest_guppy_line_lasting_down'])
        self.data_df['final_long_condition'] = self.data_df['final_long_condition'] & (self.data_df['guppy_first_half_min'] > self.data_df['guppy_second_half_max'])

        self.data_df['can_long'] = (self.data_df['can_long']) & (self.data_df['final_long_condition'])
        ###############


        #self.data_df['can_long'] = self.data_df['can_long'] & (~self.data_df['recent_guppy_long_reverse'])


        ######### Short ############

        self.data_df['vegas_support_short'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['fast_vegas_down']) & (self.data_df['slow_vegas_down']) & \
            (~((self.data_df['down_vegas_converge']) & (self.data_df['down_vegas_converge_previous'])  & (self.data_df['down_vegas_converge_pp'])))

        self.data_df['short_encourage_condition'] = (self.data_df['fast_guppy_cross_down']) & (self.data_df['fastest_guppy_line_down']) #fastest_guppy_line_down

        ######### Filters for Scenario where Vegas support short ###############

        self.data_df['short_filter1'] = (self.data_df['up_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_up'])  #adjust by removing
        self.data_df['short_filter1'] = (self.data_df['short_filter1']) | (self.data_df['previous_up_guppy_line_num'] >= 3)  #USDCAD Stuff
        self.data_df['short_filter1'] = (self.data_df['short_filter1']) & (~self.data_df['short_encourage_condition'])

        self.data_df['short_filter2'] = (self.data_df['down_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_up']) & (self.data_df['fast_guppy_cross_up'])

        self.data_df['short_strong_filter1'] = (self.data_df['guppy_half1_strong_aligned_long'])
        self.data_df['short_strong_filter2'] = (self.data_df['guppy_half2_aligned_short']) & (self.data_df['fastest_guppy_line_up']) & (self.data_df['fast_guppy_cross_up'])

        self.data_df['guppy_short_reverse'] = (self.data_df['down_guppy_line_num'] >= 3) & (self.data_df['ma_close30_gradient'] > 0)
        self.data_df['prev_guppy_short_reverse'] = self.data_df['guppy_short_reverse'].shift(1)
        self.data_df['prev2_guppy_short_reverse'] = self.data_df['prev_guppy_short_reverse'].shift(1)
        self.data_df['recent_guppy_short_reverse'] = (self.data_df['guppy_short_reverse']) | (self.data_df['prev_guppy_short_reverse']) | (self.data_df['prev2_guppy_short_reverse'])
        #self.data_df['recent_guppy_short_reverse'] = (self.data_df['guppy_short_reverse']) & (self.data_df['prev_guppy_short_reverse']) & (self.data_df['prev2_guppy_short_reverse'])


        self.data_df['can_short1'] = self.data_df['vegas_support_short'] #&\
                                     #(~self.data_df['guppy_half1_strong_aligned_long']) & (~self.data_df['prev_guppy_half1_strong_aligned_long']) & (~self.data_df['prev2_guppy_half1_strong_aligned_long']) #& (~self.data_df['short_filter1']) & (~self.data_df['short_filter2'])  #Modify

        ######## Conditions for Scenario where Vegas does not support short ###############  #second condition is EURUSD stuff

        self.data_df['short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) |\
                                          ((self.data_df['guppy_half2_strong_aligned_short'])) |\
                                          (self.data_df['guppy_all_aligned_short']) | (self.data_df['short_encourage_condition'])

        self.data_df['short_condition'] = self.data_df['short_condition'] & (~self.data_df['fastest_guppy_line_lasting_up'])
        self.data_df['short_condition'] = self.data_df['short_condition'] & (self.data_df['guppy_first_half_max'] < self.data_df['guppy_second_half_min'])

        #self.data_df['short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) #Adjust2
        self.data_df['can_short2'] = (~self.data_df['vegas_support_short']) & self.data_df['short_condition']


        self.data_df['final_short_filter1'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] < 48) & (self.data_df['prev_vegas_phase_entire_duration'] < 48) &\
                                              ( ((self.data_df['fast_vegas_up']) & (self.data_df['previous_fast_vegas_up'])) |\
                                             ((self.data_df['slow_vegas_up']) & (self.data_df['previous_slow_vegas_up'])) |\
                                             ((self.data_df['previous_fast_vegas_up']) & (self.data_df['pp_fast_vegas_up'])) |\
                                             ((self.data_df['previous_slow_vegas_up']) & (self.data_df['pp_slow_vegas_up']))
                                             )

        self.data_df['final_short_filter2'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 48)
        self.data_df['short_filter_exempt'] = self.data_df['fast_vegas_down'] & self.data_df['previous_fast_vegas_down'] &\
                                             (self.data_df['vegas_distance_gradient'] < 0) & (self.data_df['prev_vegas_distance_gradient'] < 0)
        #self.data_df['final_short_filter2'] = self.data_df['final_short_filter2'] & (~self.data_df['short_filter_exempt'])

        self.data_df['final_short_filter'] = self.data_df['final_short_filter1'] | self.data_df['final_short_filter2']




        # self.data_df['final_short_filter'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) #&\
        #                                # ( ((self.data_df['fast_vegas_up']) & (self.data_df['previous_fast_vegas_up'])) |\
        #                                #   ((self.data_df['slow_vegas_up']) & (self.data_df['previous_slow_vegas_up'])) |\
        #                                #   ((self.data_df['previous_fast_vegas_up']) & (self.data_df['pp_fast_vegas_up'])) |\
        #                                #   ((self.data_df['previous_slow_vegas_up']) & (self.data_df['pp_slow_vegas_up']))
        #                                #   )
        # self.data_df['short_filter_exempt'] = self.data_df['fast_vegas_down'] & self.data_df['previous_fast_vegas_down']
        # self.data_df['final_short_filter'] = self.data_df['final_short_filter'] & (~self.data_df['short_filter_exempt'])
        #
        # # self.data_df['final_short_filter'] = ((self.data_df['final_short_filter']) & (~self.data_df['short_encourage_condition'])) |\
        # #                                      ((self.data_df['final_short_filter']) & (self.data_df['vegas_phase_duration'] >= 48) & (self.data_df['fast_vegas_above']))
        #
        # # self.data_df['final_short_filter'] =  ((self.data_df['final_short_filter']) & (~self.data_df['short_encourage_condition'])) |\
        # #                                      ((self.data_df['final_short_filter']) &\
        # #                                       ((self.data_df['vegas_phase_duration'] >= 48) | (self.data_df['prev_vegas_phase_entire_duration'] < 48)) & (self.data_df['fast_vegas_above']))
        #
        # self.data_df['final_short_filter'] =  ((self.data_df['final_short_filter']) &\
        #                                       ((self.data_df['vegas_phase_duration'] >= 48) | (self.data_df['prev_vegas_phase_entire_duration'] < 48)) & (self.data_df['fast_vegas_above']))




        self.data_df['can_short'] = True #(self.data_df['can_short1']) | (self.data_df['can_short2'])
        #self.data_df['can_short'] = (self.data_df['vegas_support_short']) & (self.data_df['short_condition']) #strong adjust

        self.data_df['can_short'] = (self.data_df['can_short']) & (~self.data_df['final_short_filter']) #USDCAD stuff

        #############
        self.data_df['final_short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) |\
                                          ((self.data_df['guppy_half2_strong_aligned_short'])) |\
                                          (self.data_df['guppy_all_aligned_short'])
        #self.data_df['final_short_condition'] = self.data_df['final_short_condition'] & (~self.data_df['fastest_guppy_line_lasting_up'])
        self.data_df['final_short_condition'] = self.data_df['final_short_condition'] & (self.data_df['guppy_first_half_max'] < self.data_df['guppy_second_half_min'])

        self.data_df['can_short'] = (self.data_df['can_short']) & (self.data_df['final_short_condition'])
        ############


        #self.data_df['can_short'] = self.data_df['can_short'] & (~self.data_df['recent_guppy_short_reverse'])

        ########################################

        vegas_reverse_look_back_window = 10 #10
        exceed_vegas_threshold = 200 #200
        signal_minimum_lasting_bars = 0  #2
        stop_loss_threshold = 100 #100
        #Guoji

        profit_loss_ratio = 1#2

        if use_dynamic_TP:
            profit_loss_ratio = 10



        self.data_df['upper_vegas'] = self.data_df[['ma_close144', 'ma_close169']].max(axis=1)
        self.data_df['lower_vegas'] = self.data_df[['ma_close144', 'ma_close169']].min(axis=1)

        self.data_df['prev_upper_vegas'] = self.data_df['upper_vegas'].shift(1).fillna(0)
        self.data_df['prev_lower_vegas'] = self.data_df['lower_vegas'].shift(1).fillna(0)

        self.data_df['m12_above_upper_vegas'] = self.data_df['ma_close12'] > self.data_df['upper_vegas']
        self.data_df['m12_below_lower_vegas'] = self.data_df['ma_close12'] < self.data_df['lower_vegas']

        self.data_df['m12_above_lower_vegas'] = self.data_df['ma_close12'] > self.data_df['lower_vegas']
        self.data_df['m12_below_upper_vegas'] = self.data_df['ma_close12'] < self.data_df['upper_vegas']


        self.data_df['low_price_to_upper_vegas'] = self.data_df['low'] - self.data_df['upper_vegas']
        self.data_df['middle_price_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['max_price']  #middle_price

        self.data_df['high_price_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['high']
        self.data_df['middle_price_to_upper_vegas'] = self.data_df['min_price'] - self.data_df['upper_vegas']  #middle_price


        self.data_df['recent_min_low_price_to_upper_vegas'] = self.data_df['low_price_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).min()
        self.data_df['recent_max_middle_price_to_lower_vegas'] = self.data_df['middle_price_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).max()


        self.data_df['recent_min_high_price_to_lower_vegas'] = self.data_df['high_price_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).min()
        self.data_df['recent_max_middle_price_to_upper_vegas'] = self.data_df['middle_price_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).max()

        self.data_df['m12_to_lower_vegas'] = self.data_df['ma_close12'] - self.data_df['lower_vegas']
        self.data_df['m12_to_upper_vegas'] = self.data_df['upper_vegas'] - self.data_df['ma_close12']

        self.data_df['recent_min_m12_to_lower_vegas'] = self.data_df['m12_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                   min_periods = vegas_reverse_look_back_window).min()
        self.data_df['recent_min_m12_to_upper_vegas'] = self.data_df['m12_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                   min_periods = vegas_reverse_look_back_window).min()



        self.data_df['vegas_long_cond1'] = self.data_df['is_positive']
        self.data_df['vegas_long_cond2'] = (self.data_df['close'] > self.data_df['ma_close12']) & ((self.data_df['open'] < self.data_df['ma_close12']) | (self.data_df['prev_open'] < self.data_df['prev_ma_close12']))
        self.data_df['vegas_long_cond3'] = self.data_df['close'] > self.data_df['upper_vegas'] #self.data_df['m12_above_upper_vegas'] #m12_above_upper_vegas
        self.data_df['vegas_long_cond4'] = self.data_df['recent_min_low_price_to_upper_vegas'] <= 0
        self.data_df['vegas_long_cond5'] = self.data_df['recent_max_middle_price_to_lower_vegas'] * self.lot_size * self.exchange_rate <= 0# exceed_vegas_threshold
        self.data_df['vegas_long_cond6'] = self.data_df['recent_min_m12_to_lower_vegas'] > 0
        self.data_df['vegas_long_cond7'] = (self.data_df['ma_close30'] > self.data_df['lower_vegas']) & (self.data_df['ma_close35'] > self.data_df['lower_vegas']) #EURAUD Stuff
        self.data_df['vegas_long_cond8'] = self.data_df['can_long']


        self.data_df['vegas_short_cond1'] = self.data_df['is_negative']
        self.data_df['vegas_short_cond2'] = (self.data_df['close'] < self.data_df['ma_close12']) & ((self.data_df['open'] > self.data_df['ma_close12']) | (self.data_df['prev_open'] > self.data_df['prev_ma_close12']))
        self.data_df['vegas_short_cond3'] = self.data_df['close'] < self.data_df['lower_vegas'] #self.data_df['m12_below_lower_vegas'] #m12_below_lower_vegas
        self.data_df['vegas_short_cond4'] = self.data_df['recent_min_high_price_to_lower_vegas'] <= 0
        self.data_df['vegas_short_cond5'] = self.data_df['recent_max_middle_price_to_upper_vegas'] * self.lot_size * self.exchange_rate <= 0 # exceed_vegas_threshold
        self.data_df['vegas_short_cond6'] = self.data_df['recent_min_m12_to_upper_vegas'] > 0
        self.data_df['vegas_short_cond7'] = (self.data_df['ma_close30'] < self.data_df['upper_vegas']) & (self.data_df['ma_close35'] < self.data_df['upper_vegas']) #EURAUD Stuff
        self.data_df['vegas_short_cond8'] = self.data_df['can_short']

        self.data_df['vegas_long_fire'] = reduce(lambda left, right: left & right, [self.data_df['vegas_long_cond' + str(i)] for i in range(1, 9)])
        self.data_df['vegas_short_fire'] = reduce(lambda left, right: left & right, [self.data_df['vegas_short_cond' + str(i)] for i in range(1, 9)])


        # print("Check data_df:")
        #
        # print(self.data_df.iloc[0:30][['time'] + ['vegas_long_cond' + str(i) for i in range(1,8)] + ['vegas_short_cond' + str(i) for i in range(1, 8)] + ['vegas_long_fire', 'vegas_short_fire']])
        #
        # temp_df1 = self.data_df[self.data_df['vegas_long_fire'].isnull()]
        # temp_df2 = self.data_df[self.data_df['vegas_short_fire'].isnull()]
        #
        # temp_df3 = self.data_df[self.data_df['vegas_long_fire']]
        # temp_df4 = self.data_df[self.data_df['vegas_short_fire']]
        #
        # print("long problem = " + str(temp_df1.shape[0]) + " short problem = " + str(temp_df2.shape[0]))
        # print("long fire num = " + str(temp_df3.shape[0]) + " short fire num = " + str(temp_df4.shape[0]))


        #sys.exit(0)


        ######################################

        self.data_df['id'] = list(range(self.data_df.shape[0]))

        self.data_df['long_dummy'] = np.where(
            self.data_df['vegas_long_fire'],
            self.data_df['id'],
            np.nan)

        self.data_df['long_dummy'] = self.data_df['long_dummy'].fillna(method = 'ffill').fillna(0)
        self.data_df['long_lasting'] = self.data_df['id'] - self.data_df['long_dummy']
        self.data_df['previous_long_lasting'] = self.data_df['long_lasting'].shift(1)

        self.data_df['final_vegas_long_fire'] = (self.data_df['vegas_long_fire']) & (self.data_df['previous_long_lasting'] > signal_minimum_lasting_bars)

        #self.data_df['final_vegas_long_fire'] = self.data_df['final_vegas_long_fire'].shift(1).fillna(method='bfill')

        ###################

        if not use_dynamic_TP:
            while True:
                self.data_df['long_stop_loss_price'] = np.where(
                    self.data_df['final_vegas_long_fire'],
                    self.data_df['lower_vegas'] - stop_loss_threshold / (self.lot_size * self.exchange_rate),
                    np.nan)

                self.data_df['long_stop_range'] = np.where(
                    self.data_df['final_vegas_long_fire'],
                    self.data_df['close'] - self.data_df['long_stop_loss_price'],
                    np.nan
                )

                self.data_df['long_stop_profit_price'] = np.where(
                    self.data_df['final_vegas_long_fire'],
                    self.data_df['close'] + profit_loss_ratio * self.data_df['long_stop_range'],
                    np.nan
                )

                self.data_df['long_stop_loss_price'] = self.data_df['long_stop_loss_price'].fillna(method='ffill').fillna(0)
                self.data_df['long_stop_profit_price'] = self.data_df['long_stop_profit_price'].fillna(method='ffill').fillna(0)

                self.data_df['long_stop_loss_price'] = self.data_df['long_stop_loss_price'].shift(1).fillna(0)
                self.data_df['long_stop_profit_price'] = self.data_df['long_stop_profit_price'].shift(1).fillna(0)

                self.data_df['long_stop_loss'] = np.where(
                    # self.data_df['final_vegas_long_fire'],
                    # 0,
                    # np.where(
                        (self.data_df['long_stop_loss_price'] > 0) & (self.data_df['low'] <= self.data_df['long_stop_loss_price']),
                        -1,
                        np.nan
                    #)
                )

                self.data_df['long_stop_profit'] = np.where(
                    # self.data_df['final_vegas_long_fire'],
                    # 0,
                    # np.where(
                        (self.data_df['long_stop_profit_price'] > 0) & (self.data_df['high'] >= self.data_df['long_stop_profit_price']),
                        1,
                        np.nan
                    #)
                )

                self.data_df['long_stop_profit_loss'] = np.where(
                    self.data_df['long_stop_profit'].notnull(),
                    self.data_df['long_stop_profit'],
                    np.where(
                        self.data_df['long_stop_loss'].notnull(),
                        self.data_df['long_stop_loss'],
                        np.nan
                    )
                )

                ############
                self.data_df['long_stop_profit_loss'] = np.where(
                    self.data_df['final_vegas_long_fire'] & (self.data_df['long_stop_profit_loss'].isnull()),
                    0,
                    self.data_df['long_stop_profit_loss']
                )
                ############

                self.data_df['long_stop_profit_loss'] = self.data_df['long_stop_profit_loss'].fillna(method='bfill').fillna(0)

                self.data_df['long_stop_profit_loss'] = self.data_df['long_stop_profit_loss'].shift(-1)

                temp_long_df = self.data_df[self.data_df['final_vegas_long_fire']][['id', 'time', 'final_vegas_long_fire', 'long_stop_profit_loss']]
                temp_long_df['prev_long_stop_profit_loss'] = temp_long_df['long_stop_profit_loss'].shift(1).fillna(1)
                temp_long_df['next_id'] = temp_long_df['id'].shift(-1).fillna(-1).astype(int)

                # print("temp_long_df:")
                # print(temp_long_df)

                not_finished_long_df = temp_long_df[(temp_long_df['long_stop_profit_loss'] == 0) & (temp_long_df['prev_long_stop_profit_loss'] != 0) & (temp_long_df['next_id'] != -1)]

                # print("not_finished_long_df:")
                # print(not_finished_long_df)

                if not_finished_long_df.shape[0] == 0:
                    break
                else:
                    ids_erase = not_finished_long_df['next_id'].tolist()

                    # print("ids_erase:")
                    # print(ids_erase)

                    for erase_id in ids_erase:
                        if erase_id != -1:
                            #print("erase_id = " + str(erase_id))
                            #self.data_df.iloc[-1]['final_vegas_long_fire'] = True
                            self.data_df.at[erase_id, 'final_vegas_long_fire'] = False


                # print("")
                # print("df after erase:")
                # print(self.data_df.iloc[-50:][['id','time','open','high','low','close','vegas_long_fire', 'final_vegas_long_fire']])


        #self.data_df['vegas_long_fire_rt'] = self.data_df['final_vegas_long_fire'].shift(-1).fillna(method='ffill')


        ##################



        # temp_df = self.data_df[self.data_df['final_vegas_long_fire'].isnull()]
        # print("long fire null has " + str(temp_df.shape[0]))
        #sys.exit(0)

        self.data_df['short_dummy'] = np.where(
            self.data_df['vegas_short_fire'],
            self.data_df['id'],
            np.nan)

        self.data_df['short_dummy'] = self.data_df['short_dummy'].fillna(method = 'ffill').fillna(0)
        self.data_df['short_lasting'] = self.data_df['id'] - self.data_df['short_dummy']
        self.data_df['previous_short_lasting'] = self.data_df['short_lasting'].shift(1)

        self.data_df['final_vegas_short_fire'] = (self.data_df['vegas_short_fire']) & (self.data_df['previous_short_lasting'] > signal_minimum_lasting_bars)

        #self.data_df['final_vegas_short_fire'] = self.data_df['final_vegas_short_fire'].shift(1).fillna(method='bfill')

        ###################

        if not use_dynamic_TP:
            while True:

                self.data_df['short_stop_loss_price'] = np.where(
                    self.data_df['final_vegas_short_fire'],
                    self.data_df['upper_vegas'] + stop_loss_threshold / (self.lot_size * self.exchange_rate),
                    np.nan)

                self.data_df['short_stop_range'] = np.where(
                    self.data_df['final_vegas_short_fire'],
                    self.data_df['short_stop_loss_price'] - self.data_df['close'],
                    np.nan
                )

                self.data_df['short_stop_profit_price'] = np.where(
                    self.data_df['final_vegas_short_fire'],
                    self.data_df['close'] - profit_loss_ratio * self.data_df['short_stop_range'],
                    np.nan
                )


                self.data_df['short_stop_loss_price'] = self.data_df['short_stop_loss_price'].fillna(method='ffill').fillna(0)
                self.data_df['short_stop_profit_price'] = self.data_df['short_stop_profit_price'].fillna(method='ffill').fillna(0)

                self.data_df['short_stop_loss_price'] = self.data_df['short_stop_loss_price'].shift(1).fillna(0)
                self.data_df['short_stop_profit_price'] = self.data_df['short_stop_profit_price'].shift(1).fillna(0)

                self.data_df['short_stop_loss'] = np.where(
                    # self.data_df['final_vegas_short_fire'],
                    # 0,
                    # np.where(
                        (self.data_df['short_stop_loss_price'] > 0) & (
                                    self.data_df['high'] >= self.data_df['short_stop_loss_price']),
                        -1,
                        np.nan
                    #)
                )

                self.data_df['short_stop_profit'] = np.where(
                    # self.data_df['final_vegas_short_fire'],
                    # 0,
                    # np.where(
                        (self.data_df['short_stop_profit_price'] > 0) & (
                                    self.data_df['low'] <= self.data_df['short_stop_profit_price']),
                        1,
                        np.nan
                    #)
                )

                self.data_df['short_stop_profit_loss'] = np.where(
                    self.data_df['short_stop_profit'].notnull(),
                    self.data_df['short_stop_profit'],
                    np.where(
                        self.data_df['short_stop_loss'].notnull(),
                        self.data_df['short_stop_loss'],
                        np.nan
                    )
                )

                ############
                self.data_df['short_stop_profit_loss'] = np.where(
                    self.data_df['final_vegas_short_fire'] & (self.data_df['short_stop_profit_loss'].isnull()),
                    0,
                    self.data_df['short_stop_profit_loss']
                )
                ############

                self.data_df['short_stop_profit_loss'] = self.data_df['short_stop_profit_loss'].fillna(method='bfill').fillna(0)

                self.data_df['short_stop_profit_loss'] = self.data_df['short_stop_profit_loss'].shift(-1)

                temp_short_df = self.data_df[self.data_df['final_vegas_short_fire']][['id', 'time', 'final_vegas_short_fire', 'short_stop_profit_loss']]
                temp_short_df['prev_short_stop_profit_loss'] = temp_short_df['short_stop_profit_loss'].shift(1).fillna(1)
                temp_short_df['next_id'] = temp_short_df['id'].shift(-1).fillna(-1).astype(int)

                # print("temp_short_df:")
                # print(temp_short_df)

                not_finished_short_df = temp_short_df[(temp_short_df['short_stop_profit_loss'] == 0) & (temp_short_df['prev_short_stop_profit_loss'] != 0) & (temp_short_df['next_id'] != -1)]

                # print("not_finished_short_df:")
                # print(not_finished_short_df)

                if not_finished_short_df.shape[0] == 0:
                    break
                else:
                    ids_erase = not_finished_short_df['next_id'].tolist()

                    # print("ids_erase:")
                    # print(ids_erase)

                    for erase_id in ids_erase:
                        if erase_id != -1:
                            #print("erase_id = " + str(erase_id))
                            # self.data_df.iloc[-1]['final_vegas_long_fire'] = True
                            self.data_df.at[erase_id, 'final_vegas_short_fire'] = False

                # print("")
                # print("df after erase:")
                # print(self.data_df.iloc[-50:][
                #           ['id', 'time', 'open', 'high', 'low', 'close', 'vegas_short_fire', 'final_vegas_short_fire']])

        #sys.exit(0)
        #self.data_df['vegas_short_fire_rt'] = self.data_df['final_vegas_short_fire'].shift(-1).fillna(method='ffill')
        #self.data_df['final_vegas_short_fire'] = self.data_df['final_vegas_short_fire'].shift(1).fillna(method='bfill')

        ##################


        ############# Long stop calculation #######################

        ############## Real time stuff ##############
        # self.data_df['long_stop_loss_price_rt'] = np.where(
        #     self.data_df['vegas_long_fire'],
        #     self.data_df['lower_vegas'] -  stop_loss_threshold/(self.lot_size * self.exchange_rate),
        #     np.nan
        # )
        #
        # self.data_df['long_stop_range_rt'] = np.where(
        #     self.data_df['vegas_long_fire'],
        #     self.data_df['close'] - self.data_df['long_stop_loss_price_rt'],
        #     np.nan
        # )
        #
        # self.data_df['long_stop_profit_price_rt'] = np.where(
        #     self.data_df['vegas_long_fire'],
        #     self.data_df['close'] + profit_loss_ratio * self.data_df['long_stop_range_rt'],
        #     np.nan
        # )
        #################################################




        ##############################
        if use_dynamic_TP:

            result_columns = ['currency', 'time', 'id', 'close', 'long_stop_loss_price', 'TP1', 'unit_range', 'position', 'margin',
                              'long_stop_profit_price', 'tp_num',
                              'long_stop_profit_loss', 'long_stop_profit_loss_id', 'long_stop_profit_loss_time']

            result_data = []

            long_start_ids = which(self.data_df['final_vegas_long_fire'])

            is_effective = [1] * len(long_start_ids)

            for ii in range(0, len(long_start_ids)):

                #print("")
                #print("Process long ii = " + str(ii))

                if is_effective[ii] == 0:
                    self.data_df.at[long_start_ids[ii], 'final_vegas_long_fire'] = False
                    continue

                temp_ii = ii

                long_start_id = long_start_ids[ii]

                #print("ii time = " + str(self.data_df.iloc[long_start_id]['time']))

                long_fire_data = self.data_df.iloc[long_start_id]

                entry_price = long_fire_data['close']
                long_stop_loss_price = long_fire_data['lower_vegas'] - stop_loss_threshold / (self.lot_size * self.exchange_rate)

                unit_range = long_fire_data['close'] - long_stop_loss_price


                position = unit_loss / (unit_range * self.lot_size * self.usdfx * usdhkd)
                position = round(position/2.0, 4)
                margin = unit_loss * entry_price / (unit_range * leverage)
                margin = round(margin/2.0, 2)

                long_target_profit_price = long_fire_data['close'] + unit_range
                TP1 = long_target_profit_price

                ##########################
                if self.is_notify and long_start_id == self.data_df.shape[0] - 1:
                    current_time = str(self.data_df.iloc[-1]['time'] + timedelta(hours = 1))

                    message = "At " + current_time + ", Long " + self.currency + " with two " + str(round(position, 2)) + " lots at entry price " + str(self.round_price(entry_price)) + "\n"
                    message += "Place stop loss at price " + str(self.round_price(long_stop_loss_price)) \
                               + " (" + str(int(self.round_price(unit_range) * self.lot_size * self.exchange_rate)/10.0) + " pips away) \n"
                    message += "Place stop profit at price " + str(self.round_price(long_target_profit_price)) + " for only one of the two positions \n"
                    message += "This position incurs a margin of " + str(margin*2) + " HK dollars\n"

                    message_title = "Long " + self.currency + " " + str(round(position*2, 2)) + " lot"

                    print("message_title = " + message_title)
                    print("message:")
                    print(message)
                    sendEmail(message_title, message)



                ##########################


                long_actual_stop_profit_price = 0

                current_stop_loss = long_stop_loss_price
                actual_stop_loss = current_stop_loss

                tp_number = 0

                long_stop_profit_loss = 0
                long_stop_profit_loss_time = self.data_df.iloc[-1]['time']
                long_stop_profit_loss_id = self.data_df.iloc[-1]['id']

                j = 1
                while long_start_id + j < self.data_df.shape[0]:

                    cur_data = self.data_df.iloc[long_start_id + j]

                    if cur_data['low'] <= actual_stop_loss:
                        long_actual_stop_profit_price = current_stop_loss
                        if long_actual_stop_profit_price < entry_price:
                            long_stop_profit_loss = -1
                        else:
                            long_stop_profit_loss = 1

                        long_stop_profit_loss_time = cur_data['time']
                        long_stop_profit_loss_id = cur_data['id']

                        #####################################
                        if self.is_notify and long_start_id + j == self.data_df.shape[0] - 1:
                            current_time = str(self.data_df.iloc[-1]['time'] + timedelta(hours=1))

                            message = "At " + current_time + ", the price of " + self.currency + " hits stop loss " + str(self.round_price(actual_stop_loss)) + '\n'

                            if long_actual_stop_profit_price < entry_price:
                                message += "Two " + str(round(position, 2)) + "-lot positions get closed \n"
                                message += "It yields a loss of " + str(unit_loss) + " HK dollars"

                                message_title = "Long position of " + self.currency + " hits stop loss"

                                print("message_title = " + message_title)
                                print("message:")
                                print(message)
                                sendEmail(message_title, message)

                            else:
                                message += "The second " + str(round(position, 2)) + "-lot position gets closed at TP" + str(tp_number - 1) + " \n"
                                if tp_number > 1:
                                    message += "It yields a profit of " + str((tp_number - 1 - tp_tolerance) * unit_loss / 2.0) + " HK dollars"
                                else:
                                    assert(tp_number == 1)
                                    message += "It yields zero pnl (only spread cost)"

                                message_title = "Long position of " + self.currency + " hits MOVED stop loss"

                                print("message_title = " + message_title)
                                print("message:")
                                print(message)
                                sendEmail(message_title, message)

                        #####################################

                        break

                    elif cur_data['high'] >= long_target_profit_price:
                        tp_number += 1
                        current_stop_loss += unit_range
                        actual_stop_loss = current_stop_loss - unit_range * tp_tolerance

                        long_target_profit_price += unit_range

                        #####################################
                        if self.is_notify and long_start_id + j == self.data_df.shape[0] - 1:

                            current_time = str(self.data_df.iloc[-1]['time'] + timedelta(hours=1))

                            message = "At " + current_time + ", the price of " + self.currency + " reaches next profit level " + " TP" + str(tp_number) + " " + str(self.round_price(long_target_profit_price - unit_range)) + "\n"
                            message += "Move stop loss up by " + str(int(self.round_price(unit_range) * self.lot_size * self.exchange_rate)/10.0) + " pips, to price " + str(self.round_price(actual_stop_loss)) + "\n"
                            message += "The next profit level is price " + str(self.round_price(long_target_profit_price))

                            if tp_number == 1:
                                message += " The first " + str(round(position, 2)) + "-lot position gets closed, yeilding a profit of " + str(unit_loss/2.0) + " HK dollars"

                            message_title = "Long position of " + self.currency + " reaches next profit level"

                            print("message_title = " + message_title)
                            print("message:")
                            print(message)
                            sendEmail(message_title, message)
                        ########################################



                    if temp_ii + 1 < len(long_start_ids) and long_start_ids[temp_ii + 1] == long_start_id + j:

                        # print("next ii = " + str(ii + 1))
                        # print("next long start id = " + str(long_start_ids[ii+1]))
                        # print("current id = " + str(long_start_id + j))
                        # print("current time = " + str(self.data_df.iloc[long_start_id + j]['time']))

                        is_effective[temp_ii + 1] = 0
                        temp_ii += 1

                    j += 1



                result_data += [[long_fire_data['currency'], long_fire_data['time'], long_fire_data['id'], entry_price, long_stop_loss_price, TP1,
                             unit_range, position, margin, long_actual_stop_profit_price, tp_number, long_stop_profit_loss, long_stop_profit_loss_id, long_stop_profit_loss_time]]

            long_df = pd.DataFrame(data = result_data, columns = result_columns)



        else:

            ##############################

            self.data_df['long_stop_loss_price'] = np.where(
                self.data_df['final_vegas_long_fire'],
                self.data_df['lower_vegas'] - stop_loss_threshold / (self.lot_size * self.exchange_rate),
                np.nan)

            self.data_df['long_stop_range'] = np.where(
                self.data_df['final_vegas_long_fire'],
                self.data_df['close'] - self.data_df['long_stop_loss_price'],
                np.nan
            )


            self.data_df['long_stop_profit_price'] = np.where(
                self.data_df['final_vegas_long_fire'],
                self.data_df['close'] + profit_loss_ratio * self.data_df['long_stop_range'],
                np.nan
            )

            self.data_df['long_stop_half_profit_price'] = np.where(
                self.data_df['final_vegas_long_fire'],
                self.data_df['close'] + 0.5* profit_loss_ratio * self.data_df['long_stop_range'],
                np.nan
            )


            self.data_df['long_stop_loss_price'] = self.data_df['long_stop_loss_price'].fillna(method = 'ffill').fillna(0)
            self.data_df['long_stop_profit_price'] = self.data_df['long_stop_profit_price'].fillna(method='ffill').fillna(0)
            self.data_df['long_stop_half_profit_price'] = self.data_df['long_stop_half_profit_price'].fillna(method='ffill').fillna(0)

            ####################
            self.data_df['long_stop_loss_price'] = self.data_df['long_stop_loss_price'].shift(1).fillna(0)
            self.data_df['long_stop_profit_price'] = self.data_df['long_stop_profit_price'].shift(1).fillna(0)
            self.data_df['long_stop_half_profit_price'] = self.data_df['long_stop_half_profit_price'].shift(1).fillna(0)


            self.data_df['long_stop_loss'] = np.where(
                # self.data_df['final_vegas_long_fire'],
                # 0,
                # np.where(
                    (self.data_df['long_stop_loss_price'] > 0) & (self.data_df['low'] <= self.data_df['long_stop_loss_price']),
                    -1,
                    np.nan
                #)
            )

            self.data_df['long_stop_loss_id'] = np.where(
                # self.data_df['final_vegas_long_fire'],
                # self.data_df['id'],
                # np.where(
                    self.data_df['long_stop_loss'] == -1,
                    self.data_df['id'],
                    np.nan
                #)
            )

            self.data_df['long_stop_loss_time'] = np.where(
                # self.data_df['final_vegas_long_fire'],
                # self.data_df['time'],
                # np.where(
                    self.data_df['long_stop_loss'] == -1,
                    self.data_df['time'],
                    pd.NaT
                #)
            )
            self.data_df['long_stop_loss_time'] = pd.to_datetime(self.data_df['long_stop_loss_time'])




            self.data_df['long_stop_profit'] = np.where(
                # self.data_df['final_vegas_long_fire'],
                # 0,
                # np.where(
                    (self.data_df['long_stop_profit_price'] > 0) & (self.data_df['high'] >= self.data_df['long_stop_profit_price']),
                    1,
                    np.nan
                #)
            )

            self.data_df['long_stop_profit_id'] = np.where(
                # self.data_df['final_vegas_long_fire'],
                # self.data_df['id'],
                # np.where(
                    self.data_df['long_stop_profit'] == 1,
                    self.data_df['id'],
                    np.nan
                #)
            )

            self.data_df['long_stop_profit_time'] = np.where(
                # self.data_df['final_vegas_long_fire'],
                # self.data_df['time'],
                # np.where(
                    self.data_df['long_stop_profit'] == 1,
                    self.data_df['time'],
                    pd.NaT
                #)
            )
            self.data_df['long_stop_profit_time'] = pd.to_datetime(self.data_df['long_stop_profit_time'])

            # print("Checking time:")
            # temp_df = self.data_df[self.data_df['long_stop_profit_time'].notnull()]
            # print("long_stop_profit_time not null: " + str(temp_df.shape[0]))
            # print(temp_df[['time', 'long_stop_profit_time']])
            # print(self.data_df.iloc[0:100][['time', 'long_stop_loss_time', 'long_stop_profit_time']])


            self.data_df['long_stop_profit_loss'] = np.where(
                self.data_df['long_stop_profit'].notnull(),
                self.data_df['long_stop_profit'],
                np.where(
                    self.data_df['long_stop_loss'].notnull(),
                    self.data_df['long_stop_loss'],
                    np.nan
                )
            )

            #################
            self.data_df['long_stop_profit_loss'] = np.where(
                    self.data_df['final_vegas_long_fire'] & (self.data_df['long_stop_profit_loss'].isnull()),
                    0,
                    self.data_df['long_stop_profit_loss']
            )
            #################

            self.data_df['long_stop_profit_loss'] = self.data_df['long_stop_profit_loss'].fillna(method='bfill').fillna(0)

            self.data_df['long_stop_profit_loss_id'] = np.where(
                self.data_df['long_stop_profit_id'].notnull(),
                self.data_df['long_stop_profit_id'],
                np.where(
                    self.data_df['long_stop_loss_id'].notnull(),
                    self.data_df['long_stop_loss_id'],
                    np.nan
                )
            )

            #################
            self.data_df['long_stop_profit_loss_id'] = np.where(
                    self.data_df['final_vegas_long_fire'] & (self.data_df['long_stop_profit_loss_id'].isnull()),
                    self.data_df['id'],
                    self.data_df['long_stop_profit_loss_id']
            )
            #################

            self.data_df['long_stop_profit_loss_id'] = self.data_df['long_stop_profit_loss_id'].fillna(method = 'bfill').fillna(0)


            self.data_df['long_stop_profit_loss_time'] = np.where(
                self.data_df['long_stop_profit_time'].notnull(),
                self.data_df['long_stop_profit_time'],
                np.where(
                    self.data_df['long_stop_loss_time'].notnull(),
                    self.data_df['long_stop_loss_time'],
                    pd.NaT
                )
            )

            #################
            self.data_df['long_stop_profit_loss_time'] = np.where(
                    self.data_df['final_vegas_long_fire'] & (self.data_df['long_stop_profit_loss_time'].isnull()),
                    self.data_df['time'],
                    self.data_df['long_stop_profit_loss_time']
            )
            #################

            self.data_df['long_stop_profit_loss_time'] = pd.to_datetime(self.data_df['long_stop_profit_loss_time'])

            self.data_df['long_stop_profit_loss_time'] = self.data_df['long_stop_profit_loss_time'].fillna(method = 'bfill').fillna(0)



            self.data_df['long_stop_profit_loss'] = self.data_df['long_stop_profit_loss'].shift(-1)
            self.data_df['long_stop_profit_loss_id'] = self.data_df['long_stop_profit_loss_id'].shift(-1)
            self.data_df['long_stop_profit_loss_time'] = self.data_df['long_stop_profit_loss_time'].shift(-1)

            ###########
            self.data_df['long_stop_profit_price'] = self.data_df['long_stop_profit_price'].shift(-1)
            self.data_df['long_stop_half_profit_price'] = self.data_df['long_stop_half_profit_price'].shift(-1)
            self.data_df['long_stop_loss_price'] = self.data_df['long_stop_loss_price'].shift(-1)
            ###########

            long_df = self.data_df[self.data_df['final_vegas_long_fire']][['currency', 'time', 'id', 'close', 'long_stop_range', 'long_stop_loss_price', 'long_stop_profit_price', 'long_stop_half_profit_price',
                                                                           'long_stop_profit_loss', 'long_stop_profit_loss_id', 'long_stop_profit_loss_time']]

            long_df['position'] = unit_loss / (long_df['long_stop_range'] * self.lot_size * self.usdfx * usdhkd)
            long_df['position'] = long_df['position'] / 2.0
            long_df['position'] = long_df['position'].apply(lambda x: round(x, 4))
            long_df['margin'] = unit_loss * long_df['close'] / (long_df['long_stop_range'] * leverage)
            long_df['margin'] = long_df['margin'] / 2.0
            long_df['margin'] = long_df['margin'].apply(lambda x: round(x, 2))


            # print("long_df:")
            # print(long_df)

        long_df_copy = long_df.copy()
        long_df = long_df[(long_df['long_stop_profit_loss'] == 1) | (long_df['long_stop_profit_loss'] == -1)]

        long_df['long_stop_profit_loss_id'] = long_df['long_stop_profit_loss_id'].astype(int)

        self.long_df = long_df

        long_win_num = long_df[long_df['long_stop_profit_loss'] == 1].shape[0]
        long_lose_num = long_df[long_df['long_stop_profit_loss'] == -1].shape[0]

        long_df['side'] = 'long'
        long_df_copy['side'] = 'long'


        write_long_df = long_df_copy[['currency', 'side', 'time', 'close',
                                 'long_stop_profit_loss_time', 'long_stop_profit_loss', 'long_stop_loss_price', 'long_stop_profit_price'] +
                                (['TP1', 'tp_num', 'position', 'margin'] if use_dynamic_TP else ['position', 'margin'])]


        write_long_df = write_long_df.rename(columns = {
            'time' : 'entry_time',
            'close' : 'entry_price',
            'long_stop_profit_loss_time' : 'exit_time',
            'long_stop_profit_loss' : 'is_win'
        })

        write_long_df['is_win'] = np.where(write_long_df['is_win'] == 1, 1,
                                            np.where(write_long_df['is_win'] == -1, 0, -1))
        write_long_df['exit_price'] = np.where(write_long_df['is_win'] == 1, write_long_df['long_stop_profit_price'], write_long_df['long_stop_loss_price'])

        write_long_df = write_long_df[['currency', 'side','entry_time','entry_price','exit_time','exit_price','is_win'] + (['TP1', 'tp_num', 'position', 'margin'] if use_dynamic_TP else ['position', 'margin'])]

        ############## Short stop calculation ################

        ############## Real time stuff ##############
        # self.data_df['short_stop_loss_price_rt'] = np.where(
        #     self.data_df['vegas_short_fire'],
        #     self.data_df['upper_vegas'] +  stop_loss_threshold/(self.lot_size * self.exchange_rate),
        #     np.nan
        # )
        #
        # self.data_df['short_stop_range_rt'] = np.where(
        #     self.data_df['vegas_short_fire'],
        #     self.data_df['short_stop_loss_price_rt'] - self.data_df['close'],
        #     np.nan
        # )
        #
        # self.data_df['short_stop_profit_price_rt'] = np.where(
        #     self.data_df['vegas_short_fire'],
        #     self.data_df['close'] - profit_loss_ratio * self.data_df['short_stop_range_rt'],
        #     np.nan
        # )
        #################################################



        if use_dynamic_TP:
            result_columns =  ['currency', 'time', 'id', 'close', 'short_stop_loss_price', 'TP1', 'unit_range', 'position', 'margin',
                              'short_stop_profit_price', 'tp_num', 'short_stop_profit_loss', 'short_stop_profit_loss_id', 'short_stop_profit_loss_time']

            result_data = []

            short_start_ids = which(self.data_df['final_vegas_short_fire'])

            is_effective = [1] * len(short_start_ids)

            for ii in range(0, len(short_start_ids)):

                #print("process short ii = " + str(ii))

                if is_effective[ii] == 0:
                    self.data_df.at[short_start_ids[ii], 'final_vegas_short_fire'] = False
                    continue

                temp_ii = ii

                short_start_id = short_start_ids[ii]

                short_fire_data = self.data_df.iloc[short_start_id]

                entry_price = short_fire_data['close']
                short_stop_loss_price = short_fire_data['upper_vegas'] + stop_loss_threshold / (self.lot_size * self.exchange_rate)

                unit_range = short_stop_loss_price - short_fire_data['close']

                position = unit_loss / (unit_range * self.lot_size * self.usdfx * usdhkd)
                position = round(position/2.0, 4)
                margin = unit_loss * entry_price / (unit_range * leverage)
                margin = round(margin/2.0, 2)


                short_target_profit_price = short_fire_data['close'] - unit_range
                TP1 = short_target_profit_price

                ##########################
                if self.is_notify and short_start_id == self.data_df.shape[0] - 1:
                    current_time = str(self.data_df.iloc[-1]['time'] + timedelta(hours = 1))

                    message = "At " + current_time + ", Short " + self.currency + " with two " + str(round(position, 2)) + " lots at entry price " + str(self.round_price(entry_price)) + "\n"
                    message += "Place stop loss at price " + str(self.round_price(short_stop_loss_price)) \
                               + " (" + str(int(self.round_price(unit_range) * self.lot_size * self.exchange_rate)/10.0) + " pips away) \n"
                    message += "Place stop profit at price " + str(self.round_price(short_target_profit_price)) + " for only one of the two positions \n"
                    message += "This position incurs a margin of " + str(margin*2) + " HK dollars\n"

                    message_title = "Short " + self.currency + " " + str(round(position*2, 2)) + " lot"

                    print("message_title = " + message_title)
                    print("message:")
                    print(message)
                    sendEmail(message_title, message)



                ##########################


                short_actual_stop_profit_price = 0

                current_stop_loss = short_stop_loss_price
                actual_stop_loss = current_stop_loss

                tp_number = 0

                short_stop_profit_loss = 0
                short_stop_profit_loss_time = self.data_df.iloc[-1]['time']
                short_stop_profit_loss_id = self.data_df.iloc[-1]['id']

                j = 1

                #print("unit_range = " + str(unit_range))

                #print("current_stop_loss = " + str(current_stop_loss))
                while short_start_id + j < self.data_df.shape[0]:

                    cur_data = self.data_df.iloc[short_start_id + j]
                    if cur_data['high'] >= actual_stop_loss:
                        short_actual_stop_profit_price = current_stop_loss
                        if short_actual_stop_profit_price > entry_price:
                            short_stop_profit_loss = -1
                        else:
                            short_stop_profit_loss = 1

                        short_stop_profit_loss_time = cur_data['time']
                        short_stop_profit_loss_id = cur_data['id']


                        #####################################
                        if self.is_notify and short_start_id + j == self.data_df.shape[0] - 1:
                            current_time = str(self.data_df.iloc[-1]['time'] + timedelta(hours=1))

                            message = "At " + current_time + ", the price of " + self.currency + " hits stop loss " + str(self.round_price(actual_stop_loss)) + '\n'

                            if short_actual_stop_profit_price > entry_price:
                                message += "Two " + str(round(position, 2)) + "-lot positions get closed \n"
                                message += "It yields a loss of " + str(unit_loss) + " HK dollars"

                                message_title = "Short position of " + self.currency + " hits stop loss"

                                print("message_title = " + message_title)
                                print("message:")
                                print(message)
                                sendEmail(message_title, message)

                            else:
                                message += "The second " + str(round(position, 2)) + "-lot position gets closed at TP" + str(tp_number - 1) + " \n"
                                if tp_number > 1:
                                    message += "It yields a profit of " + str((tp_number - 1 - tp_tolerance) * unit_loss / 2.0) + " HK dollars"
                                else:
                                    assert(tp_number == 1)
                                    message += "It yields zero pnl (only spread cost)"

                                message_title = "Short position of " + self.currency + " hits MOVED stop loss"

                                print("message_title = " + message_title)
                                print("message:")
                                print(message)
                                sendEmail(message_title, message)

                        #####################################



                        break

                    elif cur_data['low'] <= short_target_profit_price:
                        tp_number += 1
                        current_stop_loss -= unit_range
                        actual_stop_loss = current_stop_loss + unit_range * tp_tolerance

                        short_target_profit_price -= unit_range

                        #####################################
                        if self.is_notify and short_start_id + j == self.data_df.shape[0] - 1:

                            current_time = str(self.data_df.iloc[-1]['time'] + timedelta(hours=1))

                            message = "At " + current_time + ", the price of " + self.currency + " reaches next profit level " + " TP" + str(tp_number) + " " + str(self.round_price(short_target_profit_price + unit_range)) + "\n"
                            message += "Move stop loss up by " + str(int(self.round_price(unit_range) * self.lot_size * self.exchange_rate)/10.0) + " pips, to price " + str(self.round_price(actual_stop_loss)) + "\n"
                            message += "The next profit level is price " + str(self.round_price(short_target_profit_price))

                            if tp_number == 1:
                                message += " The first " + str(round(position, 2)) + "-lot position gets closed, yeilding a profit of " + str(unit_loss/2.0) + " HK dollars"

                            message_title = "Short position of " + self.currency + " reaches next profit level"

                            print("message_title = " + message_title)
                            print("message:")
                            print(message)
                            sendEmail(message_title, message)
                        ########################################

                    if temp_ii + 1 < len(short_start_ids) and short_start_ids[temp_ii + 1] == short_start_id + j:
                        is_effective[temp_ii + 1] = 0
                        temp_ii += 1

                    j += 1


                result_data += [[short_fire_data['currency'], short_fire_data['time'], short_fire_data['id'], entry_price, short_stop_loss_price, TP1,
                                 unit_range, position, margin, short_actual_stop_profit_price, tp_number, short_stop_profit_loss, short_stop_profit_loss_id, short_stop_profit_loss_time]]

            short_df = pd.DataFrame(data = result_data, columns = result_columns)

        else:


            self.data_df['short_stop_loss_price'] = np.where(
                self.data_df['final_vegas_short_fire'],
                self.data_df['upper_vegas'] + stop_loss_threshold/(self.lot_size * self.exchange_rate),
                np.nan)

            self.data_df['short_stop_range'] = np.where(
                self.data_df['final_vegas_short_fire'],
                self.data_df['short_stop_loss_price'] - self.data_df['close'],
                np.nan
            )

            self.data_df['short_stop_profit_price'] = np.where(
                self.data_df['final_vegas_short_fire'],
                self.data_df['close'] - profit_loss_ratio * self.data_df['short_stop_range'],
                np.nan
            )

            self.data_df['short_stop_half_profit_price'] = np.where(
                self.data_df['final_vegas_short_fire'],
                self.data_df['close'] - 0.5 * profit_loss_ratio * self.data_df['short_stop_range'],
                np.nan
            )

            self.data_df['short_stop_loss_price'] = self.data_df['short_stop_loss_price'].fillna(method = 'ffill').fillna(0)
            self.data_df['short_stop_profit_price'] = self.data_df['short_stop_profit_price'].fillna(method='ffill').fillna(0)
            self.data_df['short_stop_half_profit_price'] = self.data_df['short_stop_half_profit_price'].fillna(method='ffill').fillna(0)

            ####################
            self.data_df['short_stop_loss_price'] = self.data_df['short_stop_loss_price'].shift(1).fillna(0)
            self.data_df['short_stop_profit_price'] = self.data_df['short_stop_profit_price'].shift(1).fillna(0)
            self.data_df['short_stop_half_profit_price'] = self.data_df['short_stop_half_profit_price'].shift(1).fillna(0)

            self.data_df['short_stop_loss'] = np.where(
                # self.data_df['final_vegas_short_fire'],
                # 0,
                # np.where(
                    (self.data_df['short_stop_loss_price'] > 0) & (self.data_df['high'] >= self.data_df['short_stop_loss_price']),
                    -1,
                    np.nan
                #)
            )

            self.data_df['short_stop_loss_id'] = np.where(
                # self.data_df['final_vegas_short_fire'],
                # self.data_df['id'],
                # np.where(
                    self.data_df['short_stop_loss'] == -1,
                    self.data_df['id'],
                    np.nan
                #)
            )

            self.data_df['short_stop_loss_time'] = np.where(
                # self.data_df['final_vegas_short_fire'],
                # self.data_df['time'],
                # np.where(
                    self.data_df['short_stop_loss'] == -1,
                    self.data_df['time'],
                    pd.NaT
                #)
            )
            self.data_df['short_stop_loss_time'] = pd.to_datetime(self.data_df['short_stop_loss_time'])


            self.data_df['short_stop_profit'] = np.where(
                # self.data_df['final_vegas_short_fire'],
                # 0,
                # np.where(
                    (self.data_df['short_stop_profit_price'] > 0) & (self.data_df['low'] <= self.data_df['short_stop_profit_price']),
                    1,
                    np.nan
                #)
            )

            self.data_df['short_stop_profit_id'] = np.where(
                # self.data_df['final_vegas_short_fire'],
                # self.data_df['id'],
                # np.where(
                    self.data_df['short_stop_profit'] == 1,
                    self.data_df['id'],
                    np.nan
                #)
            )

            self.data_df['short_stop_profit_time'] = np.where(
                # self.data_df['final_vegas_short_fire'],
                # self.data_df['time'],
                # np.where(
                    self.data_df['short_stop_profit'] == 1,
                    self.data_df['time'],
                    pd.NaT
                #)
            )
            self.data_df['short_stop_profit_time'] = pd.to_datetime(self.data_df['short_stop_profit_time'])




            self.data_df['short_stop_profit_loss'] = np.where(
                self.data_df['short_stop_profit'].notnull(),
                self.data_df['short_stop_profit'],
                np.where(
                    self.data_df['short_stop_loss'].notnull(),
                    self.data_df['short_stop_loss'],
                    np.nan
                )
            )

            #################
            self.data_df['short_stop_profit_loss'] = np.where(
                self.data_df['final_vegas_short_fire'] & (self.data_df['short_stop_profit_loss'].isnull()),
                0,
                self.data_df['short_stop_profit_loss']
            )
            #################


            self.data_df['short_stop_profit_loss'] = self.data_df['short_stop_profit_loss'].fillna(method='bfill').fillna(0)

            self.data_df['short_stop_profit_loss_id'] = np.where(
                self.data_df['short_stop_profit_id'].notnull(),
                self.data_df['short_stop_profit_id'],
                np.where(
                    self.data_df['short_stop_loss_id'].notnull(),
                    self.data_df['short_stop_loss_id'],
                    np.nan
                )
            )

            #################
            self.data_df['short_stop_profit_loss_id'] = np.where(
                    self.data_df['final_vegas_short_fire'] & (self.data_df['short_stop_profit_loss_id'].isnull()),
                    self.data_df['id'],
                    self.data_df['short_stop_profit_loss_id']
            )
            #################

            self.data_df['short_stop_profit_loss_id'] = self.data_df['short_stop_profit_loss_id'].fillna(method = 'bfill').fillna(0)


            self.data_df['short_stop_profit_loss_time'] = np.where(
                self.data_df['short_stop_profit_time'].notnull(),
                self.data_df['short_stop_profit_time'],
                np.where(
                    self.data_df['short_stop_loss_time'].notnull(),
                    self.data_df['short_stop_loss_time'],
                    pd.NaT
                )
            )

            #################
            self.data_df['short_stop_profit_loss_time'] = np.where(
                    self.data_df['final_vegas_short_fire'] & (self.data_df['short_stop_profit_loss_time'].isnull()),
                    self.data_df['time'],
                    self.data_df['short_stop_profit_loss_time']
            )
            #################


            self.data_df['short_stop_profit_loss_time'] = pd.to_datetime(self.data_df['short_stop_profit_loss_time'])

            self.data_df['short_stop_profit_loss_time'] = self.data_df['short_stop_profit_loss_time'].fillna(method = 'bfill').fillna(0)



            self.data_df['short_stop_profit_loss'] = self.data_df['short_stop_profit_loss'].shift(-1)
            self.data_df['short_stop_profit_loss_id'] = self.data_df['short_stop_profit_loss_id'].shift(-1)
            self.data_df['short_stop_profit_loss_time'] = self.data_df['short_stop_profit_loss_time'].shift(-1)

            ###########
            self.data_df['short_stop_profit_price'] = self.data_df['short_stop_profit_price'].shift(-1)
            self.data_df['short_stop_half_profit_price'] = self.data_df['short_stop_half_profit_price'].shift(-1)
            self.data_df['short_stop_loss_price'] = self.data_df['short_stop_loss_price'].shift(-1)
            ###########


            short_df = self.data_df[self.data_df['final_vegas_short_fire']][['currency', 'time', 'id', 'close', 'short_stop_range', 'short_stop_loss_price', 'short_stop_profit_price', 'short_stop_half_profit_price',
                                                                           'short_stop_profit_loss', 'short_stop_profit_loss_id', 'short_stop_profit_loss_time']]


            short_df['position'] = unit_loss / (short_df['short_stop_range'] * self.lot_size * self.usdfx * usdhkd)
            short_df['position'] = short_df['position'] / 2.0
            short_df['position'] = short_df['position'].apply(lambda x: round(x, 4))
            short_df['margin'] = unit_loss * short_df['close'] / (short_df['short_stop_range'] * leverage)
            short_df['margin'] = short_df['margin'] / 2.0
            short_df['margin'] = short_df['margin'].apply(lambda x: round(x, 2))


        short_df_copy = short_df.copy()
        short_df = short_df[(short_df['short_stop_profit_loss'] == 1) | (short_df['short_stop_profit_loss'] == -1)]

        short_df['short_stop_profit_loss_id'] = short_df['short_stop_profit_loss_id'].astype(int)

        self.short_df = short_df

        short_win_num = short_df[short_df['short_stop_profit_loss'] == 1].shape[0]
        short_lose_num = short_df[short_df['short_stop_profit_loss'] == -1].shape[0]

        short_df['side'] = 'short'
        short_df_copy['side'] = 'short'

        write_short_df = short_df_copy[['currency', 'side', 'time', 'close',
                                 'short_stop_profit_loss_time', 'short_stop_profit_loss', 'short_stop_loss_price', 'short_stop_profit_price'] +
                                (['TP1', 'tp_num', 'position', 'margin'] if use_dynamic_TP else ['position', 'margin'])]


        write_short_df = write_short_df.rename(columns = {
            'time' : 'entry_time',
            'close' : 'entry_price',
            'short_stop_profit_loss_time' : 'exit_time',
            'short_stop_profit_loss' : 'is_win'
        })

        write_short_df['is_win'] = np.where(write_short_df['is_win'] == 1, 1,
                                            np.where(write_short_df['is_win'] == -1, 0, -1))

        write_short_df['exit_price'] = np.where(write_short_df['is_win'] == 1, write_short_df['short_stop_profit_price'], write_short_df['short_stop_loss_price'])

        write_short_df = write_short_df[['currency', 'side','entry_time','entry_price','exit_time','exit_price','is_win']  + (['TP1', 'tp_num', 'position', 'margin'] if use_dynamic_TP else ['position', 'margin'])]

        win_num = long_win_num + short_win_num
        lose_num = long_lose_num + short_lose_num

        total_num = win_num + lose_num
        total_long_num = long_win_num + long_lose_num
        total_short_num = short_win_num + short_lose_num

        win_pct = 0 if total_num == 0 else win_num / total_num
        long_win_pct = 0 if total_long_num == 0 else long_win_num/total_long_num
        short_win_pct = 0 if total_short_num == 0 else short_win_num/total_short_num
        profit_on_average = win_pct * profit_loss_ratio - (1 - win_pct) if total_num > 0 else 0
        can_make_money = profit_on_average >= 0.1
        summary_df = pd.DataFrame({'Currency' : [self.currency], 'Trade Num' : [total_num], 'Win Num' : [win_num], 'Win Pct' : [ round(win_pct*100.0)/100.0],
                                   'PL Ratio' : [round(profit_loss_ratio*100.0)/100.0], 'Unit Profit' : [round(profit_on_average*100.0)/100.0], 'Profitable' : [can_make_money],
                                   'Long Trade Num' : [total_long_num], 'Long Win Num' : [long_win_num], 'Long Win Pct' : [ round(long_win_pct*100.0)/100.0],
                                   'Short Trade Num' : [total_short_num], 'Short Win Num' : [short_win_num], 'Short Win Pct' : [ round(short_win_pct*100.0)/100.0],
                                   })


        #
        # if is_send_email:
        #
        #     # test_data_df = self.data_df[self.data_df['time'] <= datetime(2023, 4, 6, 8, 0, 0)]
        #     #
        #     # print("test_data_df:")
        #     # print(test_data_df.iloc[-5:][['time', 'vegas_long_fire', 'vegas_short_fire', 'final_vegas_long_fire', 'final_vegas_short_fire']])
        #
        #     last_data = self.data_df.iloc[-1]
        #     if last_data['final_vegas_long_fire']:
        #         side = 'Long'
        #         entry_price = last_data['close']
        #         stop_loss = last_data['long_stop_loss_price']
        #         stop_profit = last_data['long_stop_profit_price']
        #         entry_time = str(last_data['time'] + timedelta(hours = 1))
        #     elif last_data['final_vegas_short_fire']:
        #         side = 'Short'
        #         entry_price = last_data['close']
        #         stop_loss = last_data['short_stop_loss_price']
        #         stop_profit = last_data['short_stop_profit_price']
        #         entry_time = str(last_data['time'] + timedelta(hours = 1))
        #
        #     if last_data['final_vegas_long_fire'] or last_data['final_vegas_short_fire']:
        #
        #         #decimal_place = int(math.log(self.lot_size) / math.log(10))
        #         stop_loss = round(stop_loss * self.lot_size)/float(self.lot_size)
        #         stop_profit = round(stop_profit * self.lot_size) / float(self.lot_size)
        #
        #         trading_message = "At " + entry_time + ", " + side + " " + self.currency + " at price " + str(entry_price) + ", with SL=" + str(stop_loss) + " and TP=" + str(stop_profit)
        #         sendEmail(trading_message, trading_message)


        full_summary_df = summary_df



        print("Performance Summary")
        print(full_summary_df)


        write_df = pd.concat([write_long_df, write_short_df])
        write_df = write_df.sort_values(by = ['entry_time'], ascending = True)

        self.data_df.to_csv(self.data_file, index = False)

        write_df['id'] = list(range(write_df.shape[0]))


        if use_dynamic_TP:
            write_df['pnl'] = np.where(write_df['is_win'] == 1, write_df['tp_num']-1-tp_tolerance, -1)
        else:
            write_df['pnl'] = np.where(write_df['is_win'] == 1, profit_loss_ratio, -1)

        write_df['pnl'] = write_df['pnl'] * unit_loss / 2.0

        write_df['pnl'] = np.where(write_df['is_win'] == -1, 0, write_df['pnl'])

        write_df['cum_pnl'] = write_df['pnl'].cumsum()


        if use_dynamic_TP:
            write_df['reverse_pnl'] = np.where(write_df['is_win'] == 0, 1, 1+tp_tolerance-write_df['tp_num'])
        else:
            write_df['reverse_pnl'] = np.where(write_df['is_win'] == 0, 1, -profit_loss_ratio)

        write_df['reverse_pnl'] = write_df['reverse_pnl'] * unit_loss / 2.0


        write_df['cum_reverse_pnl'] = write_df['reverse_pnl'].cumsum()


        print("trade_file: " + str(self.trade_file))
        write_df.to_csv(self.trade_file, index = False)

        print("performance_file: " + str(self.performance_file))
        full_summary_df.to_csv(self.performance_file, index = False)


        if not is_production:
            plot_pnl_figure(write_df, self.chart_folder, self.currency)





    def trade(self):

        print("Do trading............")

        self.calculate_signals()

        print_prefix = "[Currency " + self.currency + "] "
        all_days = pd.Series(self.data_df['date'].unique()).dt.to_pydatetime()

        if not is_production:
            plot_candle_bar_charts(self.currency, self.data_df, all_days, self.long_df, self.short_df,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   is_plot_aux = False,
                                   bar_fig_folder=self.chart_folder, is_plot_simple_chart=True,
                                   use_dynamic_TP = use_dynamic_TP, figure_num = printed_figure_num)


        print("Finish")












