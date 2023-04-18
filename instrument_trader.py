def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import talib

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

class CurrencyTrader(threading.Thread):

    def __init__(self, condition, currency, lot_size, exchange_rate, coefficient,  data_folder, chart_folder, simple_chart_folder, log_file, data_file, trade_file, performance_file):
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




    def calculate_signals(self):

        self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()
        self.data_df['hour'] = self.data_df['time'].apply(lambda x: x.hour)

        calc_jc_lines(self.data_df, "close", windows)
        calc_bolling_bands(self.data_df, "close", bolling_width)

        self.data_df['min_price'] = self.data_df[['open', 'close']].min(axis=1)
        self.data_df['max_price'] = self.data_df[['open', 'close']].max(axis=1)


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

        self.data_df['fastest_guppy_line_up'] = self.data_df['ma_close30_gradient'] > 0
        self.data_df['fastest_guppy_line_down'] = self.data_df['ma_close30_gradient'] < 0

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

        self.data_df['guppy_half2_aligned_short'] = reduce(lambda left, right: left & right, guppy_aligned_short_conditions[3:5])
        self.data_df['guppy_half2_all_down'] = reduce(lambda left, right: left & right, guppy_down_conditions[3:6])
        self.data_df['guppy_half2_strong_aligned_short'] = self.data_df['guppy_half2_aligned_short'] & self.data_df['guppy_half2_all_down']



        self.data_df['fast_vegas'] = self.data_df['ma_close144']
        self.data_df['slow_vegas'] = self.data_df['ma_close169']

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

        ######### Filters for Scenario where Vegas support long ###############



        self.data_df['long_filter1'] = (self.data_df['down_guppy_line_num'] >= 3)# & (self.data_df['fastest_guppy_line_down'])   #adjust by removing
        self.data_df['long_filter1'] = (self.data_df['long_filter1']) | (self.data_df['previous_down_guppy_line_num'] >= 3)  #USDCAD Stuff
        self.data_df['long_filter2'] = (self.data_df['up_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_down']) & (self.data_df['fast_guppy_cross_down'])

        self.data_df['long_strong_filter1'] = (self.data_df['guppy_half1_strong_aligned_short'])
        self.data_df['long_strong_filter2'] = (self.data_df['guppy_half2_aligned_long']) & (self.data_df['fastest_guppy_line_down']) & (self.data_df['fast_guppy_cross_down'])

        self.data_df['can_long1'] = self.data_df['vegas_support_long'] & (~self.data_df['long_filter1']) & (~self.data_df['long_filter2'])  #Modify


        ######## Conditions for Scenario where Vegas does not support long ############### #second condition is EURUSD stuff
        self.data_df['long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) |\
                                         ((self.data_df['guppy_half2_strong_aligned_long']) & (self.data_df['ma_close30'] > self.data_df['ma_close40'])) |\
                                         (self.data_df['guppy_all_aligned_long'])
        #self.data_df['long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) #Adjust2
        self.data_df['can_long2'] = (~self.data_df['vegas_support_long']) & self.data_df['long_condition']


        self.data_df['final_long_filter'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
                                       ( ((self.data_df['fast_vegas_down']) & (self.data_df['previous_fast_vegas_down'])) |\
                                         ((self.data_df['slow_vegas_down']) & (self.data_df['previous_slow_vegas_down'])) |\
                                         ((self.data_df['previous_fast_vegas_down']) & (self.data_df['pp_fast_vegas_down'])) |\
                                         ((self.data_df['previous_slow_vegas_down']) & (self.data_df['pp_slow_vegas_down']))
                                         )

        self.data_df['can_long'] = (self.data_df['can_long1']) | (self.data_df['can_long2'])
        #self.data_df['can_long'] = (self.data_df['vegas_support_long']) & (self.data_df['long_condition'])  #strong adjust

        self.data_df['can_long'] = (self.data_df['can_long']) & (~self.data_df['final_long_filter']) #USDCAD stuff



        ######### Short ############

        self.data_df['vegas_support_short'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['fast_vegas_down']) & (self.data_df['slow_vegas_down']) & \
            (~((self.data_df['down_vegas_converge']) & (self.data_df['down_vegas_converge_previous'])  & (self.data_df['down_vegas_converge_pp'])))


        ######### Filters for Scenario where Vegas support short ###############

        self.data_df['short_filter1'] = (self.data_df['up_guppy_line_num'] >= 3)# & (self.data_df['fastest_guppy_line_up'])  #adjust by removing
        self.data_df['short_filter1'] = (self.data_df['short_filter1']) | (self.data_df['previous_up_guppy_line_num'] >= 3)  #USDCAD Stuff
        self.data_df['short_filter2'] = (self.data_df['down_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_up']) & (self.data_df['fast_guppy_cross_up'])

        self.data_df['short_strong_filter1'] = (self.data_df['guppy_half1_strong_aligned_long'])
        self.data_df['short_strong_filter2'] = (self.data_df['guppy_half2_aligned_short']) & (self.data_df['fastest_guppy_line_up']) & (self.data_df['fast_guppy_cross_up'])

        self.data_df['can_short1'] = self.data_df['vegas_support_short'] & (~self.data_df['short_filter1']) & (~self.data_df['short_filter2'])  #Modify


        ######## Conditions for Scenario where Vegas does not support short ###############  #second condition is EURUSD stuff
        self.data_df['short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) |\
                                          ((self.data_df['guppy_half2_strong_aligned_short']) & (self.data_df['ma_close30'] < self.data_df['ma_close40']) ) |\
                                          (self.data_df['guppy_all_aligned_short'])
        #self.data_df['short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) #Adjust2
        self.data_df['can_short2'] = (~self.data_df['vegas_support_short']) & self.data_df['short_condition']

        self.data_df['final_short_filter'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
                                       ( ((self.data_df['fast_vegas_up']) & (self.data_df['previous_fast_vegas_up'])) |\
                                         ((self.data_df['slow_vegas_up']) & (self.data_df['previous_slow_vegas_up'])) |\
                                         ((self.data_df['previous_fast_vegas_up']) & (self.data_df['pp_fast_vegas_up'])) |\
                                         ((self.data_df['previous_slow_vegas_up']) & (self.data_df['pp_slow_vegas_up']))
                                         )



        self.data_df['can_short'] = (self.data_df['can_short1']) | (self.data_df['can_short2'])
        #self.data_df['can_short'] = (self.data_df['vegas_support_short']) & (self.data_df['short_condition']) #strong adjust

        self.data_df['can_short'] = (self.data_df['can_short']) & (~self.data_df['final_short_filter']) #USDCAD stuff

        ########################################

        vegas_reverse_look_back_window = 10 #10
        exceed_vegas_threshold = 200 #200
        signal_minimum_lasting_bars = 20
        stop_loss_threshold = 100 #100
        #Guoji
        profit_loss_ratio = 2



        self.data_df['upper_vegas'] = self.data_df[['ma_close144', 'ma_close169']].max(axis=1)
        self.data_df['lower_vegas'] = self.data_df[['ma_close144', 'ma_close169']].min(axis=1)

        self.data_df['prev_upper_vegas'] = self.data_df['upper_vegas'].shift(1).fillna(0)
        self.data_df['prev_lower_vegas'] = self.data_df['lower_vegas'].shift(1).fillna(0)

        self.data_df['m12_above_vegas'] = self.data_df['ma_close12'] > self.data_df['upper_vegas']
        self.data_df['m12_below_vegas'] = self.data_df['ma_close12'] < self.data_df['lower_vegas']


        self.data_df['low_price_to_upper_vegas'] = self.data_df['low'] - self.data_df['upper_vegas']
        self.data_df['min_price_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['min_price']

        self.data_df['high_price_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['high']
        self.data_df['max_price_to_upper_vegas'] = self.data_df['max_price'] - self.data_df['upper_vegas']


        self.data_df['recent_min_low_price_to_upper_vegas'] = self.data_df['low_price_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).min()
        self.data_df['recent_max_min_price_to_lower_vegas'] = self.data_df['min_price_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).max()


        self.data_df['recent_min_high_price_to_lower_vegas'] = self.data_df['high_price_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).min()
        self.data_df['recent_max_max_price_to_upper_vegas'] = self.data_df['max_price_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                            min_periods = vegas_reverse_look_back_window).max()

        self.data_df['m12_to_lower_vegas'] = self.data_df['ma_close12'] - self.data_df['lower_vegas']
        self.data_df['m12_to_upper_vegas'] = self.data_df['upper_vegas'] - self.data_df['ma_close12']

        self.data_df['recent_min_m12_to_lower_vegas'] = self.data_df['m12_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                   min_periods = vegas_reverse_look_back_window).min()
        self.data_df['recent_min_m12_to_upper_vegas'] = self.data_df['m12_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
                                                                                                   min_periods = vegas_reverse_look_back_window).min()



        self.data_df['vegas_long_cond1'] = self.data_df['is_positive']
        self.data_df['vegas_long_cond2'] = (self.data_df['close'] > self.data_df['ma_close12']) & (self.data_df['open'] < self.data_df['ma_close12'])
        self.data_df['vegas_long_cond3'] = self.data_df['m12_above_vegas']
        self.data_df['vegas_long_cond4'] = self.data_df['recent_min_low_price_to_upper_vegas'] <= 0
        self.data_df['vegas_long_cond5'] = self.data_df['recent_max_min_price_to_lower_vegas'] * self.lot_size * self.exchange_rate < exceed_vegas_threshold
        self.data_df['vegas_long_cond6'] = self.data_df['recent_min_m12_to_lower_vegas'] > 0
        self.data_df['vegas_long_cond7'] = (self.data_df['ma_close30'] > self.data_df['upper_vegas']) & (self.data_df['ma_close35'] > self.data_df['upper_vegas']) #EURAUD Stuff
        self.data_df['vegas_long_cond8'] = self.data_df['can_long']


        self.data_df['vegas_short_cond1'] = self.data_df['is_negative']
        self.data_df['vegas_short_cond2'] = (self.data_df['close'] < self.data_df['ma_close12']) & (self.data_df['open'] > self.data_df['ma_close12'])
        self.data_df['vegas_short_cond3'] = self.data_df['m12_below_vegas']
        self.data_df['vegas_short_cond4'] = self.data_df['recent_min_high_price_to_lower_vegas'] <= 0
        self.data_df['vegas_short_cond5'] = self.data_df['recent_max_max_price_to_upper_vegas'] * self.lot_size * self.exchange_rate < exceed_vegas_threshold
        self.data_df['vegas_short_cond6'] = self.data_df['recent_min_m12_to_upper_vegas'] > 0
        self.data_df['vegas_short_cond7'] = (self.data_df['ma_close30'] < self.data_df['lower_vegas']) & (self.data_df['ma_close35'] < self.data_df['lower_vegas']) #EURAUD Stuff
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

        self.data_df['long_dummy'].fillna(method = 'ffill').fillna(0)
        self.data_df['long_lasting'] = self.data_df['id'] - self.data_df['long_dummy']
        self.data_df['previous_long_lasting'] = self.data_df['long_lasting'].shift(1)

        #self.data_df['final_vegas_long_fire'] = (self.data_df['vegas_long_fire']) & (self.data_df['previous_long_lasting'] > signal_minimum_lasting_bars)

        self.data_df['final_vegas_long_fire'] = self.data_df['vegas_long_fire']
        self.data_df['final_vegas_long_fire'] = self.data_df['final_vegas_long_fire'].shift(1).fillna(method = 'bfill')

        # temp_df = self.data_df[self.data_df['final_vegas_long_fire'].isnull()]
        # print("long fire null has " + str(temp_df.shape[0]))
        #sys.exit(0)

        self.data_df['short_dummy'] = np.where(
            self.data_df['vegas_short_fire'],
            self.data_df['id'],
            np.nan)

        self.data_df['short_dummy'].fillna(method = 'ffill').fillna(0)
        self.data_df['short_lasting'] = self.data_df['id'] - self.data_df['short_dummy']
        self.data_df['previous_short_lasting'] = self.data_df['short_lasting'].shift(1)

        #self.data_df['final_vegas_short_fire'] = (self.data_df['vegas_short_fire']) & (self.data_df['previous_short_lasting'] > signal_minimum_lasting_bars)

        self.data_df['final_vegas_short_fire'] = self.data_df['vegas_short_fire']
        self.data_df['final_vegas_short_fire'] = self.data_df['final_vegas_short_fire'].shift(1).fillna(method = 'bfill')

        ############# Long stop calculation #######################

        ############## Real time stuff ##############
        self.data_df['long_stop_loss_price_rt'] = np.where(
            self.data_df['vegas_long_fire'],
            self.data_df['lower_vegas'] -  stop_loss_threshold/(self.lot_size * self.exchange_rate),
            np.nan
        )

        self.data_df['long_stop_range_rt'] = np.where(
            self.data_df['vegas_long_fire'],
            self.data_df['close'] - self.data_df['long_stop_loss_price_rt'],
            np.nan
        )

        self.data_df['long_stop_profit_price_rt'] = np.where(
            self.data_df['vegas_long_fire'],
            self.data_df['close'] + profit_loss_ratio * self.data_df['long_stop_range_rt'],
            np.nan
        )
        #################################################


        self.data_df['long_stop_loss_price'] = np.where(
            self.data_df['final_vegas_long_fire'],
            self.data_df['prev_lower_vegas'] - stop_loss_threshold/(self.lot_size * self.exchange_rate),
            np.nan)

        self.data_df['long_stop_range'] = np.where(
            self.data_df['final_vegas_long_fire'],
            self.data_df['open'] - self.data_df['long_stop_loss_price'],
            np.nan
        )

        self.data_df['long_stop_profit_price'] = np.where(
            self.data_df['final_vegas_long_fire'],
            self.data_df['open'] + profit_loss_ratio * self.data_df['long_stop_range'],
            np.nan
        )

        self.data_df['long_stop_loss_price'] = self.data_df['long_stop_loss_price'].fillna(method = 'ffill').fillna(0)
        self.data_df['long_stop_profit_price'] = self.data_df['long_stop_profit_price'].fillna(method='ffill').fillna(0)

        self.data_df['long_stop_loss'] = np.where(
            self.data_df['final_vegas_long_fire'],
            0,
            np.where(
                (self.data_df['long_stop_loss_price'] > 0) & (self.data_df['low'] <= self.data_df['long_stop_loss_price']),
                -1,
                np.nan
            )
        )

        self.data_df['long_stop_loss_id'] = np.where(
            self.data_df['final_vegas_long_fire'],
            self.data_df['id'],
            np.where(
                self.data_df['long_stop_loss'] == -1,
                self.data_df['id'],
                np.nan
            )
        )

        self.data_df['long_stop_loss_time'] = np.where(
            self.data_df['final_vegas_long_fire'],
            self.data_df['time'],
            np.where(
                self.data_df['long_stop_loss'] == -1,
                self.data_df['time'],
                pd.NaT
            )
        )
        self.data_df['long_stop_loss_time'] = pd.to_datetime(self.data_df['long_stop_loss_time'])




        self.data_df['long_stop_profit'] = np.where(
            self.data_df['final_vegas_long_fire'],
            0,
            np.where(
                (self.data_df['long_stop_profit_price'] > 0) & (self.data_df['high'] >= self.data_df['long_stop_profit_price']),
                1,
                np.nan
            )
        )

        self.data_df['long_stop_profit_id'] = np.where(
            self.data_df['final_vegas_long_fire'],
            self.data_df['id'],
            np.where(
                self.data_df['long_stop_profit'] == 1,
                self.data_df['id'],
                np.nan
            )
        )

        self.data_df['long_stop_profit_time'] = np.where(
            self.data_df['final_vegas_long_fire'],
            self.data_df['time'],
            np.where(
                self.data_df['long_stop_profit'] == 1,
                self.data_df['time'],
                pd.NaT
            )
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
        self.data_df['long_stop_profit_loss_time'] = pd.to_datetime(self.data_df['long_stop_profit_loss_time'])

        self.data_df['long_stop_profit_loss_time'] = self.data_df['long_stop_profit_loss_time'].fillna(method = 'bfill').fillna(0)



        self.data_df['long_stop_profit_loss'] = self.data_df['long_stop_profit_loss'].shift(-1)
        self.data_df['long_stop_profit_loss_id'] = self.data_df['long_stop_profit_loss_id'].shift(-1)
        self.data_df['long_stop_profit_loss_time'] = self.data_df['long_stop_profit_loss_time'].shift(-1)

        long_df = self.data_df[self.data_df['final_vegas_long_fire']][['time', 'id', 'open', 'long_stop_loss_price', 'long_stop_profit_price',
                                                                       'long_stop_profit_loss', 'long_stop_profit_loss_id', 'long_stop_profit_loss_time']]

        long_df = long_df[(long_df['long_stop_profit_loss'] == 1) | (long_df['long_stop_profit_loss'] == -1)]

        long_df['long_stop_profit_loss_id'] = long_df['long_stop_profit_loss_id'].astype(int)

        self.long_df = long_df

        long_win_num = long_df[long_df['long_stop_profit_loss'] == 1].shape[0]
        long_lose_num = long_df[long_df['long_stop_profit_loss'] == -1].shape[0]

        long_df['side'] = 'long'
        write_long_df = long_df[['side', 'time', 'open',
                                 'long_stop_profit_loss_time', 'long_stop_profit_loss', 'long_stop_loss_price', 'long_stop_profit_price']]
        write_long_df = write_long_df.rename(columns = {
            'time' : 'entry_time',
            'open' : 'entry_price',
            'long_stop_profit_loss_time' : 'exit_time',
            'long_stop_profit_loss' : 'is_win'
        })

        write_long_df['is_win'] = np.where(write_long_df['is_win'] == 1, 1, 0)
        write_long_df['exit_price'] = np.where(write_long_df['is_win'] == 1, write_long_df['long_stop_profit_price'], write_long_df['long_stop_loss_price'])

        write_long_df = write_long_df[['side','entry_time','entry_price','exit_time','exit_price','is_win']]

        ############## Short stop calculation ################

        ############## Real time stuff ##############
        self.data_df['short_stop_loss_price_rt'] = np.where(
            self.data_df['vegas_short_fire'],
            self.data_df['upper_vegas'] +  stop_loss_threshold/(self.lot_size * self.exchange_rate),
            np.nan
        )

        self.data_df['short_stop_range_rt'] = np.where(
            self.data_df['vegas_short_fire'],
            self.data_df['short_stop_loss_price_rt'] - self.data_df['close'],
            np.nan
        )

        self.data_df['short_stop_profit_price_rt'] = np.where(
            self.data_df['vegas_short_fire'],
            self.data_df['close'] - profit_loss_ratio * self.data_df['short_stop_range_rt'],
            np.nan
        )
        #################################################




        self.data_df['short_stop_loss_price'] = np.where(
            self.data_df['final_vegas_short_fire'],
            self.data_df['prev_upper_vegas'] + stop_loss_threshold/(self.lot_size * self.exchange_rate),
            np.nan)

        self.data_df['short_stop_range'] = np.where(
            self.data_df['final_vegas_short_fire'],
            self.data_df['short_stop_loss_price'] - self.data_df['open'],
            np.nan
        )

        self.data_df['short_stop_profit_price'] = np.where(
            self.data_df['final_vegas_short_fire'],
            self.data_df['open'] - profit_loss_ratio * self.data_df['short_stop_range'],
            np.nan
        )

        self.data_df['short_stop_loss_price'] = self.data_df['short_stop_loss_price'].fillna(method = 'ffill').fillna(0)
        self.data_df['short_stop_profit_price'] = self.data_df['short_stop_profit_price'].fillna(method='ffill').fillna(0)

        self.data_df['short_stop_loss'] = np.where(
            self.data_df['final_vegas_short_fire'],
            0,
            np.where(
                (self.data_df['short_stop_loss_price'] > 0) & (self.data_df['high'] >= self.data_df['short_stop_loss_price']),
                -1,
                np.nan
            )
        )

        self.data_df['short_stop_loss_id'] = np.where(
            self.data_df['final_vegas_short_fire'],
            self.data_df['id'],
            np.where(
                self.data_df['short_stop_loss'] == -1,
                self.data_df['id'],
                np.nan
            )
        )

        self.data_df['short_stop_loss_time'] = np.where(
            self.data_df['final_vegas_short_fire'],
            self.data_df['time'],
            np.where(
                self.data_df['short_stop_loss'] == -1,
                self.data_df['time'],
                pd.NaT
            )
        )
        self.data_df['short_stop_loss_time'] = pd.to_datetime(self.data_df['short_stop_loss_time'])


        self.data_df['short_stop_profit'] = np.where(
            self.data_df['final_vegas_short_fire'],
            0,
            np.where(
                (self.data_df['short_stop_profit_price'] > 0) & (self.data_df['low'] <= self.data_df['short_stop_profit_price']),
                1,
                np.nan
            )
        )

        self.data_df['short_stop_profit_id'] = np.where(
            self.data_df['final_vegas_short_fire'],
            self.data_df['id'],
            np.where(
                self.data_df['short_stop_profit'] == 1,
                self.data_df['id'],
                np.nan
            )
        )

        self.data_df['short_stop_profit_time'] = np.where(
            self.data_df['final_vegas_short_fire'],
            self.data_df['time'],
            np.where(
                self.data_df['short_stop_profit'] == 1,
                self.data_df['time'],
                pd.NaT
            )
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
        self.data_df['short_stop_profit_loss_time'] = pd.to_datetime(self.data_df['short_stop_profit_loss_time'])

        self.data_df['short_stop_profit_loss_time'] = self.data_df['short_stop_profit_loss_time'].fillna(method = 'bfill').fillna(0)



        self.data_df['short_stop_profit_loss'] = self.data_df['short_stop_profit_loss'].shift(-1)
        self.data_df['short_stop_profit_loss_id'] = self.data_df['short_stop_profit_loss_id'].shift(-1)
        self.data_df['short_stop_profit_loss_time'] = self.data_df['short_stop_profit_loss_time'].shift(-1)

        short_df = self.data_df[self.data_df['final_vegas_short_fire']][['time', 'id', 'open', 'short_stop_loss_price', 'short_stop_profit_price',
                                                                       'short_stop_profit_loss', 'short_stop_profit_loss_id', 'short_stop_profit_loss_time']]
        short_df = short_df[(short_df['short_stop_profit_loss'] == 1) | (short_df['short_stop_profit_loss'] == -1)]

        short_df['short_stop_profit_loss_id'] = short_df['short_stop_profit_loss_id'].astype(int)

        self.short_df = short_df

        short_win_num = short_df[short_df['short_stop_profit_loss'] == 1].shape[0]
        short_lose_num = short_df[short_df['short_stop_profit_loss'] == -1].shape[0]

        short_df['side'] = 'short'
        write_short_df = short_df[['side', 'time', 'open',
                                 'short_stop_profit_loss_time', 'short_stop_profit_loss', 'short_stop_loss_price', 'short_stop_profit_price']]
        write_short_df = write_short_df.rename(columns = {
            'time' : 'entry_time',
            'open' : 'entry_price',
            'short_stop_profit_loss_time' : 'exit_time',
            'short_stop_profit_loss' : 'is_win'
        })

        write_short_df['is_win'] = np.where(write_short_df['is_win'] == 1, 1, 0)
        write_short_df['exit_price'] = np.where(write_short_df['is_win'] == 1, write_short_df['short_stop_profit_price'], write_short_df['short_stop_loss_price'])

        write_short_df = write_short_df[['side','entry_time','entry_price','exit_time','exit_price','is_win']]

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



        if is_send_email:

            # test_data_df = self.data_df[self.data_df['time'] <= datetime(2023, 4, 6, 8, 0, 0)]
            #
            # print("test_data_df:")
            # print(test_data_df.iloc[-5:][['time', 'vegas_long_fire', 'vegas_short_fire', 'final_vegas_long_fire', 'final_vegas_short_fire']])

            last_data = self.data_df.iloc[-1]
            if last_data['vegas_long_fire']:
                side = 'Long'
                entry_price = last_data['close']
                stop_loss = last_data['long_stop_loss_price_rt']
                stop_profit = last_data['long_stop_profit_price_rt']
                entry_time = str(last_data['time'] + timedelta(hours = 1))
            elif last_data['vegas_short_fire']:
                side = 'Short'
                entry_price = last_data['close']
                stop_loss = last_data['short_stop_loss_price_rt']
                stop_profit = last_data['short_stop_profit_price_rt']
                entry_time = str(last_data['time'] + timedelta(hours = 1))

            if last_data['vegas_long_fire'] or last_data['vegas_short_fire']:

                #decimal_place = int(math.log(self.lot_size) / math.log(10))
                stop_loss = round(stop_loss * self.lot_size)/float(self.lot_size)
                stop_profit = round(stop_profit * self.lot_size) / float(self.lot_size)

                trading_message = "At " + entry_time + ", " + side + " " + self.currency + " at price " + str(entry_price) + ", with SL=" + str(stop_loss) + " and SP=" + str(stop_profit)
                sendEmail(trading_message, trading_message)


        full_summary_df = summary_df



        print("Performance Summary")
        print(full_summary_df)


        write_df = pd.concat([write_long_df, write_short_df])
        write_df = write_df.sort_values(by = ['entry_time'], ascending = True)

        self.data_df.to_csv(self.data_file, index = False)

        write_df['id'] = list(range(write_df.shape[0]))
        write_df['pnl'] = np.where(write_df['is_win'] == 1, profit_loss_ratio, -1)
        write_df['cum_pnl'] = write_df['pnl'].cumsum()

        write_df['reverse_pnl'] = np.where(write_df['is_win'] == 0, 1, -profit_loss_ratio)
        write_df['cum_reverse_pnl'] = write_df['reverse_pnl'].cumsum()


        print("trade_file: " + str(self.trade_file))
        write_df.to_csv(self.trade_file, index = False)

        print("performance_file: " + str(self.performance_file))
        full_summary_df.to_csv(self.performance_file, index = False)



        plot_pnl_figure(write_df, self.chart_folder, self.currency)





    def trade(self):

        print("Do trading............")

        self.calculate_signals()

        print_prefix = "[Currency " + self.currency + "] "
        all_days = pd.Series(self.data_df['date'].unique()).dt.to_pydatetime()

        # plot_candle_bar_charts(self.currency, self.data_df, all_days, self.long_df, self.short_df,
        #                        num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
        #                        is_plot_candle_buy_sell_points=True,
        #                        print_prefix=print_prefix,
        #                        is_plot_aux = False,
        #                        bar_fig_folder=self.chart_folder, is_plot_simple_chart=True)


        print("Finish")












