

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


data_source = 2

#initial_bar_number = 1000 #3555  50
initial_bar_number = 50 if data_source == 1 else 1000   #1000

initial_bar_number_5min = 3000  #3000

#until_date_5min = "2024-09-24"
#until_date_5min = "2024-10-10"
#until_date_5min = "2024-10-26"
#until_date_5min = "2024-11-11"
#until_date_5min = "2024-11-27"
#until_date_5min = "2024-12-13"
#until_date_5min = "2024-12-29"
#until_date_5min = "2025-01-14"
#until_date_5min = "2025-01-30"
#until_date_5min = "2025-02-15"
#until_date_5min = "2025-03-03"
#until_date_5min = "2025-03-19"
#until_date_5min = "2025-04-03"
#until_date_5min = "2025-04-18"

#until_date = "2025-04-04"

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

look_back_start_group = 1

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

fire_signal = False
report_performance = True


quick_close_position_for_intraday_strategy = False #Default is false   close all position if partial close signal fired for intraday strategy

is_intraday_strategy = False

is_intraday_quick = False  #Close at hours_close_position_quick if price already enters guppy

min_hour_open_position = 5
max_hour_open_position = 18 #18

hours_close_position_quick = [16]
hours_close_position = [0] #23

strict_smart_close_logic = False

print_email_message_to_file = False

special_cond10 = True


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

#use_dynamic_TP = True

is_crypto = True

correct_precision = not is_crypto

use_conditional_stop_loss = False

printed_figure_num = -1

plot_day_line = True
plot_cross_point = True

unit_loss = 1000 if is_crypto else 200 #This is HKD
usdhkd = 7.85
leverage = 10 if is_crypto else 100 #100 for forex, 10 for crypto

tp_tolerance = 0.05  #0.05, 0.20

use_smart_close_position_logic = True

readjust_position_when_new_signal = False

always_use_new_close_logic = True

relax_vegas = True

vegas_threshold = 1 if relax_vegas else 0

vegas_condition_threshold = 10 if relax_vegas else 1

initial_entry_value = 100.0
default_leverage = 10

do_smart_execution = False
use_5min_in_smart_execution = False

do_reentry = False

do_message_printing = False

use_slow_macd = False

use_guppy_filter = True

use_guppy_condition = False

macd_gradient = 'macd2_gradient' if use_slow_macd else 'macd_gradient'


if do_smart_execution:

    class StrategyExecution:

        def __init__(self, side, leverage, take_profit_pct, take_loss_pct, strategy_id, execution_id, strategy_entry_time, strategy_entry_price,
                      execution_entry_time, execution_entry_price, strategy_entry_value, execution_entry_value):

            self.active = True
            self.side = side #1 means long  -1 means short
            self.leverage = leverage
            self.take_profit_pct = take_profit_pct
            self.take_loss_pct = take_loss_pct
            self.strategy_id = strategy_id
            self.execution_id = execution_id
            self.strategy_entry_time = strategy_entry_time
            self.strategy_entry_price = strategy_entry_price
            self.execution_entry_time = execution_entry_time
            self.execution_entry_price = execution_entry_price
            self.strategy_entry_value = strategy_entry_value
            self.execution_entry_value = execution_entry_value

            self.execution_exit_time = None
            self.execution_exit_price = -1
            self.execution_exit_value = -1


            #self.profit_rate = self.take_profit_pct * self.leverage
            #self.loss_rate = self.take_loss_pct * self.leverage
            self.pnl_rate = 0
            self.pnl = 0

            self.take_profit_price = self.execution_entry_price * (1 + self.side * self.take_profit_pct)
            self.take_loss_price = self.execution_entry_price * (1 - self.side * self.take_loss_pct)

            self.double_take_loss_price = self.execution_exit_price * (1 - self.side * 2 * self.take_loss_pct)
            self.tripple_take_loss_price = self.execution_exit_price * (1 - self.side * 3 * self.take_loss_pct)


        def exit_execution(self, execution_exit_time, execution_exit_price, is_signal_exit, is_extra_execution):

            return_rate = self.side * (execution_exit_price - self.execution_entry_price)/self.execution_entry_price

            self.pnl_rate = return_rate * self.leverage
            self.pnl = self.execution_entry_value * self.pnl_rate

            self.execution_exit_price = execution_exit_price
            self.execution_exit_value = self.execution_entry_value + self.pnl
            self.execution_exit_time = execution_exit_time

            self.active = (not is_signal_exit) and self.pnl > 0 and (not is_extra_execution)

            #is_strategy_exit or self.pnl < 0 or self.strategy_id == 5






class CurrencyTrader(threading.Thread):

    def __init__(self, condition, currency, lot_size, exchange_rate, coefficient, actual_maxdrawdown, optimal_gradient_num,
                 data_folder, chart_folder, simple_chart_folder, log_file, data_file, trade_file, performance_file, usdfx, email_message_file, is_notify, data_file_5min = None,
                 decimal = 5, reverse_strategy = False):
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
        self.data_df_5min = None
        #self.is_finalized = False

        self.last_time = None
        self.log_file = log_file
        self.data_file = data_file
        self.data_file_5min = data_file_5min
        self.trade_file = trade_file
        self.performance_file = performance_file
        self.usdfx = usdfx

        self.email_message_fd = open(email_message_file, 'w')
        self.email_message_caches = []

        self.decimal = decimal

        self.reverse_strategy = reverse_strategy

        if use_guppy_condition:
            self.reverse_strategy = True

        self.is_notify = is_notify

        self.long_df = None
        self.short_df = None

        self.long_strategy_df = None
        self.short_strategy_df = None

        self.long_strategy_execution_df = None
        self.short_strategy_execution_df = None

        # self.use_relaxed_vegas_support = True
        # self.is_require_m12_strictly_above_vegas = False
        # self.remove_c12 = True

        #self.currency_file = os.path.join(data_folder, currency + "100.csv")

        self.log_fd = open(self.log_file, 'a')

        self.print_to_console = True

        self.current_position = 0


        #
        # self.is_cut_data = False
        #
        # self.data_df_backup100 = None
        # self.data_df_backup200 = None
        #
        # self.data_dfs_backup = []
        self.write_long_df = None
        self.write_short_df = None
        self.group_summary_df = None
        self.critical_price_data_df = None

        self.profit_loss_ratio = -1

        self.full_summary_df = None

        self.macd_group_summary_df = None
        self.critical_value_data_df = None

        self.optimal_gradient_num = optimal_gradient_num

        if do_smart_execution:
            #self.entry_total_principal = 100



            self.minimum_maxdrawdown = 0.025
            self.actual_maxdrawdown = actual_maxdrawdown #This needs to be read from config file
            self.max_drawdown = max(self.actual_maxdrawdown, self.minimum_maxdrawdown)
            self.fraction = self.actual_maxdrawdown / self.max_drawdown
            self.profit_rates = np.array([1.0, 0.5, 1.0, 0.5]) * self.fraction
            self.loss_rates = np.array([0.5] * len(self.profit_rates))  # Always stop loss when losing half of the actual notional (margin)
            self.loss_rates = self.loss_rates * self.fraction

            self.optimal_leverage = round(1.0 / (self.max_drawdown * 2), 1)
            self.half_optimal_leverage = round(self.optimal_leverage / 2, 1)
            self.leverage = np.array([self.optimal_leverage, self.optimal_leverage, self.half_optimal_leverage, self.half_optimal_leverage])

            self.take_profit_pct = self.profit_rates / self.leverage
            self.take_loss_pct = self.loss_rates / self.leverage

            self.each_strategy_entry_value = initial_entry_value / len(self.leverage)


        self.log_msg("Initializing...")


    def log_msg(self, msg):

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #current_time = (datetime.now() + timedelta(seconds = 28800)).strftime("%Y-%m-%d %H:%M:%S")
        print('[' + current_time + ' ' + self.currency + ']  ' + msg, file = self.log_fd)
        self.log_fd.flush()

        if self.print_to_console:
            print('[' + current_time + ' ' + self.currency + ']  ' + msg)


    def feed_data(self, new_data_df, new_data_df_5min = None):

        self.data_df = new_data_df
        self.data_df_5min = new_data_df_5min


        if self.data_df_5min is not None:
            print("Round to " + str(self.decimal) + " decimals.....")
            for col in ['open', 'high', 'low', 'close']:
                self.data_df_5min[col] = self.data_df_5min[col].apply(lambda x: round(x, self.decimal))

        if self.data_df_5min is not None and use_5min_in_smart_execution:

            self.data_df_5min['location'] = list(range(self.data_df_5min.shape[0]))

            self.data_df = pd.merge(self.data_df, self.data_df_5min[['time', 'location']], on = ['time'], how = 'left')


        if os.path.exists(self.trade_file):
            trade_df = pd.read_csv(self.trade_file)

            trade_df['entry_time'] = trade_df['entry_time'].apply(lambda x: preprocess_time(x))
            trade_df = trade_df[trade_df['entry_time'] < self.data_df.iloc[-1]['time']]

            print("last data time: " + str(self.data_df.iloc[-1]['time']))

            # print("trade_df last 2 rows:")
            # print(trade_df.iloc[-2:])

            last_trade = trade_df.iloc[-1]
            #if last_trade['exit_id'] == -1:
            if last_trade['side'] == 'long':
                self.current_position = 1
            else:
                self.current_position = -1

            print("current_position here = " + str(self.current_position))

            self.current_position *= initial_entry_value/last_trade['entry_price'] * default_leverage
            if last_trade['entry_price'] >= 1:
                self.current_position = round(self.current_position, 3)
            else:
                self.current_position = int(round(self.current_position, 0))



    def run(self):
        print("Running...........")
        self.trade()


    def round_price(self, price):

        if 'JPY' in self.currency:
            return round(price, 3)
        else:
            return round(price, 5)


    def calculate_signals(self, print_ready = True):

        self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()
        self.data_df['hour'] = self.data_df['time'].apply(lambda x: x.hour)

        calc_jc_lines(self.data_df, "close", windows)

        #if not is_production:
        calc_bolling_bands(self.data_df, "close", bolling_width)
        calc_macd(self.data_df, "close")

        self.data_df['upper_vegas'] = self.data_df[['ma_close144', 'ma_close169']].max(axis=1)
        self.data_df['lower_vegas'] = self.data_df[['ma_close144', 'ma_close169']].min(axis=1)


        self.data_df['prev_open'] = self.data_df['open'].shift(1)

        self.data_df['prev_ma_close12'] = self.data_df['ma_close12'].shift(1)

        self.data_df['min_price'] = self.data_df[['open', 'close']].min(axis=1)
        self.data_df['max_price'] = self.data_df[['open', 'close']].max(axis=1)

        self.data_df['prev_min_price'] = self.data_df['min_price'].shift(1)
        self.data_df['prev_max_price'] = self.data_df['max_price'].shift(1)

        self.data_df['middle_price'] = (self.data_df['open'] + self.data_df['close']) / 2.0
        self.data_df['middle'] = self.data_df['middle_price']


        self.data_df['is_positive'] = (self.data_df['close'] > self.data_df['open'])
        self.data_df['is_negative'] = (self.data_df['close'] < self.data_df['open'])


        ############ Added feature #############
        self.data_df['positive'] = np.where(self.data_df['is_positive'], 1, 0)
        self.data_df['negative'] = np.where(self.data_df['is_negative'], 1, 0)

        self.data_df['prev_is_positive'] = self.data_df['is_positive'].shift(1)
        self.data_df['pp_is_positive'] = self.data_df['prev_is_positive'].shift(1)

        self.data_df['prev_is_negative'] = self.data_df['is_negative'].shift(1)
        self.data_df['pp_is_negative'] = self.data_df['prev_is_negative'].shift(1)

        self.data_df['is_small_body'] = (self.data_df['max_price'] - self.data_df['min_price']) / (self.data_df['high'] - self.data_df['low']) < 1/5
        self.data_df['prev_is_small_body'] = self.data_df['is_small_body'].shift(1).fillna(False)
        self.data_df['pp_is_small_body'] = self.data_df['prev_is_small_body'].shift(1).fillna(False)

        self.data_df['prev_macd'] = self.data_df['macd'].shift(1)
        self.data_df['prev_msignal'] = self.data_df['msignal'].shift(1)

        self.data_df['prev_macd2'] = self.data_df['macd2'].shift(1)
        self.data_df['prev_msignal2'] = self.data_df['msignal2'].shift(1)

        self.data_df['macd_gradient'] = self.data_df['macd'].diff()
        self.data_df['macd2_gradient'] = self.data_df['macd2'].diff()

        # self.data_df['prev_macd_gradient'] = self.data_df['macd_gradient'].shift(1)
        # self.data_df['prev_macd2_gradient'] = self.data_df['macd2_gradient'].shift(1)


        for lb in range(1, 10):

            self.data_df['prev' + str(lb) + '_macd_gradient'] = self.data_df['macd_gradient'].shift(1) if lb == 1 else self.data_df['prev' + str(lb-1) + '_macd_gradient'].shift(1)
            self.data_df['prev' + str(lb) + '_macd2_gradient'] = self.data_df['macd2_gradient'].shift(1) if lb == 1 else self.data_df['prev' + str(lb-1) + '_macd2_gradient'].shift(1)


        #self.data_df['prev2_macd_gradient'] = self.data_df['prev1_macd_gradient'].shift(1)
        #self.data_df['prev2_macd2_gradient'] = self.data_df['prev1_macd2_gradient'].shift(1)



        # self.data_df['prev2_macd_gradient'] = self.data_df['prev_macd_gradient'].shift(1)
        # self.data_df['prev2_macd2_gradient'] = self.data_df['prev_macd2_gradient'].shift(1)


        ############################





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

        self.data_df['guppy_min'] = self.data_df[guppy_lines].min(axis = 1)
        self.data_df['guppy_max'] = self.data_df[guppy_lines].max(axis = 1)

        self.data_df['prev_guppy_min'] = self.data_df['guppy_min'].shift(1).fillna(0)
        self.data_df['prev_guppy_max'] = self.data_df['guppy_max'].shift(1).fillna(0)



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

        self.data_df['guppy_lines_up_num'] = reduce(lambda left, right: left + right, guppy_up_conditions)


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

        self.data_df['guppy_lines_down_num'] = reduce(lambda left, right: left + right, guppy_down_conditions)


        self.data_df['guppy_all_above_vegas'] = reduce(lambda left, right: left & right, [(self.data_df[guppy_lines[i]] > self.data_df['upper_vegas'])
                                                                                             for i in range(len(guppy_lines) - 1)])
        self.data_df['guppy_all_below_vegas'] = reduce(lambda left, right: left & right, [(self.data_df[guppy_lines[i]] < self.data_df['lower_vegas'])
                                                                                             for i in range(len(guppy_lines) - 1)])



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


        self.data_df['prev_upper_vegas'] = self.data_df['upper_vegas'].shift(1).fillna(0)
        self.data_df['prev_lower_vegas'] = self.data_df['lower_vegas'].shift(1).fillna(0)

        self.data_df['prev_middle'] = self.data_df['middle'].shift(1).fillna(0)

        self.data_df['bar_cross_up_upper_vegas'] = (self.data_df['prev_middle'] <= self.data_df['prev_upper_vegas']) & (self.data_df['middle'] > self.data_df['upper_vegas'])
        self.data_df['bar_cross_down_upper_vegas'] = (self.data_df['prev_middle'] > self.data_df['prev_upper_vegas']) & (self.data_df['middle'] <= self.data_df['upper_vegas'])

        self.data_df['bar_cross_down_lower_vegas'] = (self.data_df['prev_middle'] >= self.data_df['prev_lower_vegas']) & (self.data_df['middle'] < self.data_df['lower_vegas'])
        self.data_df['bar_cross_up_lower_vegas'] = (self.data_df['prev_middle'] < self.data_df['prev_lower_vegas']) & (self.data_df['middle'] >= self.data_df['lower_vegas'])



        self.data_df['num'] = list(range(self.data_df.shape[0]))
        self.data_df['jc_num'] = self.data_df['num']
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


        self.data_df['critical_bar_down_num'] = np.where(
            (self.data_df['bar_cross_up_upper_vegas']) | (self.data_df['bar_cross_down_upper_vegas']),
            self.data_df['num'],
            np.nan
        )
        self.data_df['critical_bar_down_num'] = self.data_df['critical_bar_down_num'].fillna(method='ffill').fillna(0)
        self.data_df['bar_down_phase_duration'] = self.data_df['num'] - self.data_df['critical_bar_down_num']


        self.data_df['critical_bar_up_num'] = np.where(
            (self.data_df['bar_cross_down_lower_vegas']) | (self.data_df['bar_cross_up_lower_vegas']),
            self.data_df['num'],
            np.nan
        )
        self.data_df['critical_bar_up_num'] = self.data_df['critical_bar_up_num'].fillna(method='ffill').fillna(0)
        self.data_df['bar_up_phase_duration'] = self.data_df['num'] - self.data_df['critical_bar_up_num']


        ########### New Code for Guppy strongly aligned duration calculation ##########

        self.data_df['guppy_all_strong_aligned'] = self.data_df['guppy_all_strong_aligned_long'] | self.data_df['guppy_all_strong_aligned_short']
        self.data_df['prev_guppy_all_strong_aligned'] = self.data_df['guppy_all_strong_aligned'].shift(1).fillna(0)
        self.data_df['guppy_all_strong_aligned_boundary'] = self.data_df['guppy_all_strong_aligned'] ^ self.data_df['prev_guppy_all_strong_aligned']

        self.data_df['guppy_aligned_critical_num'] = np.where(
            self.data_df['guppy_all_strong_aligned_boundary'],
            self.data_df['num'],
            np.nan
        )

        self.data_df['guppy_aligned_critical_num'] = self.data_df['guppy_aligned_critical_num'].fillna(method='ffill').fillna(0)
        self.data_df['guppy_aligned_duration'] = self.data_df['num'] - self.data_df['guppy_aligned_critical_num']

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

        # Old One
        self.data_df['final_long_filter1'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate < -vegas_threshold) & (self.data_df['vegas_phase_duration'] < 96) & (self.data_df['prev_vegas_phase_entire_duration'] < 96) &\
                                              ( ((self.data_df['fast_vegas_down']) & (self.data_df['previous_fast_vegas_down'])) |\
                                             ((self.data_df['slow_vegas_down']) & (self.data_df['previous_slow_vegas_down'])) |\
                                             ((self.data_df['previous_fast_vegas_down']) & (self.data_df['pp_fast_vegas_down'])) |\
                                             ((self.data_df['previous_slow_vegas_down']) & (self.data_df['pp_slow_vegas_down']))
                                             )


        # New Change
        self.data_df['final_long_filter2'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate < -vegas_threshold) & (self.data_df['vegas_phase_duration'] >= 96)
        self.data_df['long_filter_exempt'] = self.data_df['fast_vegas_up'] & self.data_df['previous_fast_vegas_up'] & (self.data_df['vegas_phase_duration'] < 8*24) &\
                                             (self.data_df['vegas_distance_gradient'] < 0) & (self.data_df['prev_vegas_distance_gradient'] < 0) & self.data_df['guppy_all_above_vegas'] & self.data_df['guppy_all_strong_aligned_long']
        self.data_df['final_long_filter2'] = self.data_df['final_long_filter2'] & (~self.data_df['long_filter_exempt'])

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
        self.data_df['final_long_condition1'] = self.data_df['final_long_condition'] & (self.data_df['guppy_first_half_min'] > self.data_df['guppy_second_half_max'])


        # self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        #                                         (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        #                                         (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_strong_aligned_short'])

        #old one
        self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
                                                (self.data_df['middle'] > self.data_df['upper_vegas']) &\
                                                (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
                                                (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_aligned_short']) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])


        # self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        #                                         (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        #                                         (self.data_df['vegas_phase_duration'] > 48) & (self.data_df['guppy_lines_down_num'] < 3) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])




        # self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        #                                         (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        #                                         (~self.data_df['guppy_all_aligned_short']) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])




        # self.data_df['final_long_condition2'] = (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        #                                         (~self.data_df['guppy_all_aligned_short']) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])

        #Change Change
        self.data_df['must_reject_long'] = False #(self.data_df['final_long_condition']) & (self.data_df['guppy_first_half_min'] <= self.data_df['guppy_second_half_max'])

        #self.data_df['must_reject_long'] = (self.data_df['final_long_condition'] & (~self.data_df['final_long_condition2'])) & (self.data_df['guppy_first_half_min'] <= self.data_df['guppy_second_half_max'])

        self.data_df['must_reject_long2'] = (~self.data_df['vegas_support_long']) & (self.data_df['ma_close30_gradient'] < 0) & (self.data_df['ma_close35_gradient'] < 0) & (self.data_df['ma_close30'] < self.data_df['ma_close35'])
        #self.data_df['must_reject_long2'] = self.data_df['must_reject_long2'] & (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)

        self.data_df['must_reject_long2'] = self.data_df['must_reject_long2'] &\
                                            (((self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)) | (self.data_df['fast_vegas'] < self.data_df['slow_vegas']))

        self.data_df['must_reject_long3'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['fast_vegas_down']) & (self.data_df['slow_vegas_down'])

        self.data_df['must_reject_long4'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['bar_up_phase_duration'] >= 24*5) & (self.data_df['guppy_lines_down_num'] >= 3)

        self.data_df['can_long'] = (self.data_df['can_long']) & (self.data_df['final_long_condition1']  | self.data_df['final_long_condition2'])
        self.data_df['can_long'] = self.data_df['can_long'] & (~self.data_df['must_reject_long']) & (~self.data_df['must_reject_long2'])# & (~self.data_df['must_reject_long3'])
        #self.data_df['can_long'] = self.data_df['can_long'] & (~self.data_df['must_reject_long4'])
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

        # Old One
        self.data_df['final_short_filter1'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate > vegas_threshold) & (self.data_df['vegas_phase_duration'] < 96) & (self.data_df['prev_vegas_phase_entire_duration'] < 96) &\
                                              ( ((self.data_df['fast_vegas_up']) & (self.data_df['previous_fast_vegas_up'])) |\
                                             ((self.data_df['slow_vegas_up']) & (self.data_df['previous_slow_vegas_up'])) |\
                                             ((self.data_df['previous_fast_vegas_up']) & (self.data_df['pp_fast_vegas_up'])) |\
                                             ((self.data_df['previous_slow_vegas_up']) & (self.data_df['pp_slow_vegas_up']))
                                             )


        # New Change
        self.data_df['final_short_filter2'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate > vegas_threshold) & (self.data_df['vegas_phase_duration'] >= 96)
        self.data_df['short_filter_exempt'] = self.data_df['fast_vegas_down'] & self.data_df['previous_fast_vegas_down'] & (self.data_df['vegas_phase_duration'] < 8*24) &\
                                             (self.data_df['vegas_distance_gradient'] < 0) & (self.data_df['prev_vegas_distance_gradient'] < 0) & self.data_df['guppy_all_below_vegas'] & self.data_df['guppy_all_strong_aligned_short']
        self.data_df['final_short_filter2'] = self.data_df['final_short_filter2'] & (~self.data_df['short_filter_exempt'])

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
        self.data_df['final_short_condition1'] = self.data_df['final_short_condition'] & (self.data_df['guppy_first_half_max'] < self.data_df['guppy_second_half_min'])

        # self.data_df['final_short_condition2'] = (self.data_df['bar_down_phase_duration'] > 48) &\
        #                                          (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        #                                          (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        #                                          (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_strong_aligned_long'])

        #Old
        self.data_df['final_short_condition2'] = (self.data_df['bar_down_phase_duration'] > 48) &\
                                                 (self.data_df['middle'] < self.data_df['lower_vegas']) &\
                                                 (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
                                                 (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_aligned_long']) #& (self.data_df['middle'] > self.data_df['guppy_min'])#& (~self.data_df['guppy_half1_strong_aligned_long'])

        # self.data_df['final_short_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        #                                         (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        #                                         (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        #                                         (self.data_df['vegas_phase_duration'] > 48) & (self.data_df['guppy_lines_up_num'] < 3) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])


        # self.data_df['final_short_condition2'] = (self.data_df['bar_down_phase_duration'] > 48) &\
        #                                          (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        #                                          (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        #                                          (~self.data_df['guppy_all_aligned_long']) #& (self.data_df['middle'] > self.data_df['guppy_min'])#& (~self.data_df['guppy_half1_strong_aligned_long'])



        # self.data_df['final_short_condition2'] = (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        #                                          (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        #                                          (~self.data_df['guppy_all_aligned_long']) #& (self.data_df['middle'] > self.data_df['guppy_min'])#& (~self.data_df['guppy_half1_strong_aligned_long'])

        #Change Change
        self.data_df['must_reject_short'] = False #(self.data_df['final_short_condition']) & (self.data_df['guppy_first_half_max'] >= self.data_df['guppy_second_half_min'])


        #self.data_df['must_reject_short'] = (self.data_df['final_short_condition'] & (~self.data_df['final_short_condition2'])) & (self.data_df['guppy_first_half_max'] >= self.data_df['guppy_second_half_min'])

        self.data_df['must_reject_short2'] = (~self.data_df['vegas_support_short']) & (self.data_df['ma_close30_gradient'] > 0) & (self.data_df['ma_close35_gradient'] > 0) & (self.data_df['ma_close30'] > self.data_df['ma_close35'])
        #self.data_df['must_reject_short2'] = self.data_df['must_reject_short2'] & (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)

        self.data_df['must_reject_short2'] = self.data_df['must_reject_short2'] &\
                                            (((self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)) | (self.data_df['fast_vegas'] > self.data_df['slow_vegas']))

        self.data_df['must_reject_short3'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['fast_vegas_up']) & (self.data_df['slow_vegas_up'])

        self.data_df['must_reject_short4'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['bar_up_phase_duration'] >= 24*5) & (self.data_df['guppy_lines_up_num'] >= 3)



        self.data_df['can_short'] = (self.data_df['can_short']) & (self.data_df['final_short_condition1'] | self.data_df['final_short_condition2'])
        self.data_df['can_short'] = self.data_df['can_short'] & (~self.data_df['must_reject_short']) & (~self.data_df['must_reject_short2'])# & (~self.data_df['must_reject_short3'])
        #self.data_df['can_short'] = self.data_df['can_short'] & (~self.data_df['must_reject_short4'])

        ############


        #self.data_df['can_short'] = self.data_df['can_short'] & (~self.data_df['recent_guppy_short_reverse'])

        ########################################

        vegas_reverse_look_back_window = 10 #10
        exceed_vegas_threshold = 200 #200
        signal_minimum_lasting_bars = 0  #2
        stop_loss_threshold = 100 #100
        #Guoji

        self.profit_loss_ratio = 1#2

        if use_dynamic_TP:
            self.profit_loss_ratio = 10







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


        ################## Added features #########################

        bar_lookback_num = 5

        self.data_df['positive_close'] = np.where(self.data_df['is_positive'], self.data_df['close'], np.nan)
        self.data_df['positive_close'] = self.data_df['positive_close'].fillna(method = 'bfill').fillna(0)
        self.data_df['positive_close_diff'] = self.data_df['positive_close'].diff()

        self.data_df['negative_close'] = np.where(self.data_df['is_negative'], self.data_df['close'], np.nan)
        self.data_df['negative_close'] = self.data_df['negative_close'].fillna(method = 'bfill').fillna(0)
        self.data_df['negative_close_diff'] = self.data_df['negative_close'].diff()

        self.data_df['positive_close_increase'] = np.where(self.data_df['positive_close_diff'] >= 0, 1, 0)
        self.data_df['positive_close_decrease'] = np.where(self.data_df['positive_close_diff'] < 0, 1, 0)

        self.data_df['negative_close_decrease'] = np.where(self.data_df['negative_close_diff'] <= 0, 1, 0)
        self.data_df['negative_close_increase'] = np.where(self.data_df['negative_close_diff'] > 0, 1, 0)

        self.data_df['recent_positive_close_decrease_num'] = self.data_df['positive_close_decrease'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()
        self.data_df['recent_negative_close_increase_num'] = self.data_df['negative_close_increase'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()

        self.data_df['prev_recent_positive_close_decrease_num'] = self.data_df['recent_positive_close_decrease_num'].shift(1)  ###
        self.data_df['prev_recent_negative_close_increase_num'] = self.data_df['recent_negative_close_increase_num'].shift(1)



        self.data_df['positive_open'] = np.where(self.data_df['positive'], self.data_df['open'], np.nan)
        self.data_df['positive_open'] = self.data_df['positive_open'].fillna(method = 'bfill').fillna(0)
        self.data_df['positive_open_diff'] = self.data_df['positive_open'].diff()

        self.data_df['negative_open'] = np.where(self.data_df['negative'], self.data_df['open'], np.nan)
        self.data_df['negative_open'] = self.data_df['negative_open'].fillna(method = 'bfill').fillna(0)
        self.data_df['negative_open_diff'] = self.data_df['negative_open'].diff()

        self.data_df['positive_open_increase'] = np.where(self.data_df['positive_open_diff'] >= 0, 1, 0)
        self.data_df['positive_open_decrease'] = np.where(self.data_df['positive_open_diff'] < 0, 1, 0)

        self.data_df['negative_open_decrease'] = np.where(self.data_df['negative_open_diff'] <= 0, 1, 0)
        self.data_df['negative_open_increase'] = np.where(self.data_df['negative_open_diff'] > 0, 1, 0)

        self.data_df['recent_positive_open_decrease_num'] = self.data_df['positive_open_decrease'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()
        self.data_df['recent_negative_open_increase_num'] = self.data_df['negative_open_increase'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()

        self.data_df['prev_recent_positive_open_decrease_num'] = self.data_df['recent_positive_open_decrease_num'].shift(1)  ###
        self.data_df['prev_recent_negative_open_increase_num'] = self.data_df['recent_negative_open_increase_num'].shift(1)




        self.data_df['recent_positive_bar_num'] = self.data_df['positive'].rolling(bar_lookback_num, min_periods = bar_lookback_num).sum()
        self.data_df['recent_negative_bar_num'] = self.data_df['negative'].rolling(bar_lookback_num, min_periods = bar_lookback_num).sum()

        self.data_df['prev_recent_positive_bar_num'] = self.data_df['recent_positive_bar_num'].shift(1)
        self.data_df['prev_recent_negative_bar_num'] = self.data_df['recent_negative_bar_num'].shift(1)


        self.data_df['backward_min_price'] = self.data_df['min_price'].shift(bar_lookback_num)
        self.data_df['backward_max_price'] = self.data_df['max_price'].shift(bar_lookback_num)


        self.data_df['special_reject_short_cond1'] = self.data_df['prev_recent_positive_bar_num'] >= 3
        self.data_df['special_reject_short_cond2'] = self.data_df['prev_is_positive'] & (~self.data_df['prev_is_small_body']) & self.data_df['pp_is_positive'] & (~self.data_df['pp_is_small_body'])
        self.data_df['special_reject_short_cond3'] = (self.data_df['prev_recent_positive_close_decrease_num'] == 0) & (self.data_df['prev_recent_positive_open_decrease_num'] == 0)
        self.data_df['special_reject_short_cond4'] = self.data_df['is_negative'] & (self.data_df['min_price'] <= self.data_df['backward_min_price'])
        self.data_df['special_reject_short_cond'] = reduce(lambda left, right: left & right, [self.data_df['special_reject_short_cond' + str(i)] for i in range(1, 5)])

        self.data_df['special_reject_long_cond1'] = self.data_df['prev_recent_negative_bar_num'] >= 3
        self.data_df['special_reject_long_cond2'] = self.data_df['prev_is_negative'] & (~self.data_df['prev_is_small_body']) & self.data_df['pp_is_negative'] & (~self.data_df['pp_is_small_body'])
        self.data_df['special_reject_long_cond3'] = (self.data_df['prev_recent_negative_close_increase_num'] == 0) & (self.data_df['prev_recent_negative_open_increase_num'] == 0)
        self.data_df['special_reject_long_cond4'] = self.data_df['is_positive'] & (self.data_df['max_price'] >= self.data_df['backward_max_price'])
        self.data_df['special_reject_long_cond'] = reduce(lambda left, right: left & right, [self.data_df['special_reject_long_cond' + str(i)] for i in range(1, 5)])





        ###########################################################




        ######## keybox #########
        self.data_df['macd_cross_up'] = (self.data_df['prev_macd'] < self.data_df['prev_msignal']) & (
                self.data_df['macd'] > self.data_df['msignal'])
        self.data_df['macd_cross_down'] = (self.data_df['prev_macd'] > self.data_df['prev_msignal']) & (
                self.data_df['macd'] < self.data_df['msignal'])

        self.data_df['macd_cross_label'] = np.where(
            self.data_df['macd_cross_up'], 0,
            np.where(
                self.data_df['macd_cross_down'], 1, np.nan
            )
        )

        self.data_df['macd_cross_label'] = self.data_df['macd_cross_label'].fillna(method='ffill').fillna(-1)
        self.data_df['prev_macd_cross_label'] = self.data_df['macd_cross_label'].shift(1)

        self.data_df['macd_cross_label_line'] = self.data_df['macd_cross_label'].diff()

        self.data_df['macd_cross_num'] = np.where(
            self.data_df['macd_cross_label'] != self.data_df['prev_macd_cross_label'],
            self.data_df['num'],
            np.nan
        )

        self.data_df['macd_cross_num'] = self.data_df['macd_cross_num'].fillna(method='ffill').fillna(0)
        self.data_df['macd_cross_duration'] = self.data_df['num'] - self.data_df['macd_cross_num']

        self.data_df['macd_max'] = self.data_df['macd']
        self.data_df['macd_min'] = self.data_df['macd']

        self.data_df['macd_max_idx'] = self.data_df['macd']
        self.data_df['macd_min_idx'] = self.data_df['macd']

        macd_group_summary_df = self.data_df[['time', 'macd_max', 'macd_min', 'macd_max_idx', 'macd_min_idx',
                                              'macd_cross_label', 'macd_cross_num', 'macd_cross_duration']].groupby(
            ['macd_cross_num']).agg(
            {
                'time': 'first',
                'macd_max': 'max',
                'macd_min': 'min',
                'macd_max_idx': 'idxmax',
                'macd_min_idx': 'idxmin',
                'macd_cross_label': 'first',
                'macd_cross_duration': 'last'
            }
        )

        macd_group_summary_df.reset_index(inplace=True)

        macd_group_summary_df.at[macd_group_summary_df.index[0], 'macd_cross_label'] = 1 if macd_group_summary_df.iloc[1]['macd_cross_label'] == 0 else 0

        macd_group_summary_df['critical_value_id'] = np.where(
            macd_group_summary_df['macd_cross_label'] == 0,
            macd_group_summary_df['macd_max_idx'],
            np.where(
                macd_group_summary_df['macd_cross_label'] == 1,
                macd_group_summary_df['macd_min_idx'],
                0
            )
        )

        macd_group_summary_df['critical_value'] = np.where(
            macd_group_summary_df['macd_cross_label'] == 0,
            macd_group_summary_df['macd_max'],
            np.where(
                macd_group_summary_df['macd_cross_label'] == 1,
                macd_group_summary_df['macd_min'],
                0
            )
        )

        macd_group_summary_df['macd_cross_num'] = macd_group_summary_df['macd_cross_num'].astype(int)
        macd_group_summary_df['macd_cross_label'] = macd_group_summary_df['macd_cross_label'].astype(int)

        macd_group_summary_df['group_index'] = list(range(macd_group_summary_df.shape[0]))

        critical_value_data_df = self.data_df.iloc[macd_group_summary_df['critical_value_id']]

        critical_value_data_df.reset_index(inplace=True)

        critical_value_data_df['group_index'] = list(range(critical_value_data_df.shape[0]))

        critical_value_data_df['critical_value'] = macd_group_summary_df['critical_value']

        critical_value_data_df['macd_cross_label'] = macd_group_summary_df['macd_cross_label']

        macd_group_data_dfs = []
        macd_cross_nums = macd_group_summary_df['macd_cross_num'].tolist()
        macd_cross_labels = macd_group_summary_df['macd_cross_label'].tolist()

        for idi in range(0, len(macd_cross_nums)):
            start_idxx = macd_cross_nums[idi]
            end_idxx = macd_cross_nums[idi + 1] if idi < len(macd_cross_nums) - 1 else self.data_df.shape[0]

            macd_cross_label = macd_cross_labels[idi]

            #print("macd_cross_label = " + str(macd_cross_label))

            group_df = self.data_df.iloc[start_idxx:end_idxx][['time', 'macd']]

            if macd_cross_label == 0:
                # print("start_idxx = " + str(start_idxx) + " end_idxx = " + str(end_idxx))
                # print("group_df:")
                # print(group_df)
                # print("")

                if group_df[group_df['macd'].isnull()].shape[0] > 0:
                    group_df['critical_value'] = np.nan
                    group_df['critical_value_id'] = 0
                    group_df = group_df.drop(columns=['time', 'macd'])
                else:
                    group_df['critical_value'] = group_df['macd'].cummax()
                    group_df['critical_value_id'] = group_df['macd'].expanding().apply(lambda x: x.idxmax()).astype(int)
                    group_df = group_df.drop(columns=['time', 'macd'])
                group_df['group_index'] = idi

            elif macd_cross_label == 1:

                if group_df[group_df['macd'].isnull()].shape[0] > 0:
                    group_df['critical_value'] = np.nan
                    group_df['critical_value_id'] = 0
                    group_df = group_df.drop(columns=['time', 'macd'])
                else:
                    group_df['critical_value'] = group_df['macd'].cummin()
                    group_df['critical_value_id'] = group_df['macd'].expanding().apply(lambda x: x.idxmin()).astype(int)
                    group_df = group_df.drop(columns=['time', 'macd'])

                group_df['group_index'] = idi
            # else:
            #     group_df = self.data_df.iloc[start_idxx:end_idxx][['time', 'macd']]
            #     group_df['critical_value'] = np.nan
            #     group_df['critical_value_id'] = np.nan
            #     group_df['group_index'] = idi

            macd_group_data_dfs += [group_df]

        macd_group_data_df_all = pd.concat(macd_group_data_dfs)

        # print("macd_group_data_df_all:")
        # print(macd_group_data_df_all.iloc[0:60])
        #
        # print("macd_group_data_df_all length = " + str(macd_group_data_df_all.shape[0]))
        # print("data_df length = " + str(self.data_df.shape[0]))
        # print(len(macd_group_data_df_all))

        if len(macd_group_data_df_all) != self.data_df.shape[0]:
            raise Exception(
                "macd_group_data_df_all length = " + str(
                    len(macd_group_data_df_all)) + " while data_df length = " + str(
                    self.data_df.shape[0]))

        #print("First")
        #print(self.data_df.iloc[0:100][['time', 'macd', 'msignal']])

        self.data_df = pd.concat([self.data_df, macd_group_data_df_all], axis=1)

        aux_macd_data_df = self.data_df[
            ['lower_vegas', 'upper_vegas', 'guppy_min', 'guppy_max', 'macd_cross_duration', 'high', 'low', 'max_price',
             'min_price']]

        #print(self.data_df.iloc[0:100][['time', 'macd', 'msignal', 'critical_value_id', 'critical_value']])
        #sys.exit(0)


        attach_df = aux_macd_data_df.iloc[self.data_df['critical_value_id']]
        attach_df.reset_index(inplace=True)
        attach_df = attach_df.drop(columns=['index'])
        rename_dict = {}
        for column in aux_macd_data_df.columns:
            rename_dict[column] = 'critical_' + column
        attach_df = attach_df.rename(columns=rename_dict)
        self.data_df = pd.concat([self.data_df, attach_df], axis=1)

        critical_value_data_df = critical_value_data_df.rename(columns={'index': 'critical_value_id'})
        critical_value_data_df['macd_cross_total_duration'] = macd_group_summary_df['macd_cross_duration']

        key_columns = ['time', 'macd_cross_num', 'macd_cross_duration', 'critical_value_id',
                       'macd_cross_total_duration',
                       'critical_value', 'macd_cross_label', 'lower_vegas', 'upper_vegas', 'guppy_min',
                       'guppy_max', 'high', 'low', 'max_price', 'min_price']
        look_backward_group_num = 11  # 3 should be odd number  9
        for key_column in key_columns:
            for backward_i in range(1, look_backward_group_num + 1):
                if backward_i == 1:
                    critical_value_data_df['prevGroup_' + str(backward_i) + key_column] = critical_value_data_df[
                        key_column].shift(1).fillna(0)
                else:
                    critical_value_data_df['prevGroup_' + str(backward_i) + key_column] = critical_value_data_df[
                        'prevGroup_' + str(backward_i - 1) + key_column].shift(1).fillna(0)

        simple_critical_value_data_df = critical_value_data_df[
            ['group_index'] + [column for column in critical_value_data_df.columns if 'prevGroup' in column]]

        self.data_df = pd.merge(self.data_df, simple_critical_value_data_df, on=['group_index'], how='left')

        need_look_backward_cols = ["prevGroup_1critical_value", "prevGroup_1critical_value_id",
                                   'prevGroup_1macd_cross_duration', 'prevGroup_1macd_cross_num',
                                   'prevGroup_1high', 'prevGroup_1low', 'prevGroup_1max_price', 'prevGroup_1min_price',
                                   'prevGroup_1upper_vegas', 'prevGroup_1lower_vegas']
        for li in range(2, look_backward_group_num + 1):
            need_look_backward_cols += ['prevGroup_' + str(li) + 'critical_value',
                                        'prevGroup_' + str(li) + 'critical_value_id',
                                        'prevGroup_' + str(li) + 'high', 'prevGroup_' + str(li) + 'low',
                                        'prevGroup_' + str(li) + 'max_price', 'prevGroup_' + str(li) + 'min_price',
                                        'prevGroup_' + str(li) + 'lower_vegas', 'prevGroup_' + str(li) + 'upper_vegas',
                                        'prevGroup_' + str(li) + 'macd_cross_total_duration',
                                        'prevGroup_' + str(li) + 'macd_cross_num'
                                        ]

        no_need_look_backward_cols = ['critical_value', 'critical_value_id', 'critical_macd_cross_duration',
                                      'macd_cross_num',
                                      'critical_high', 'critical_low', 'critical_max_price', 'critical_min_price',
                                      'critical_upper_vegas', 'critical_lower_vegas']
        for li in range(1, look_backward_group_num):
            no_need_look_backward_cols += ['prevGroup_' + str(li) + 'critical_value',
                                           'prevGroup_' + str(li) + 'critical_value_id',
                                           'prevGroup_' + str(li) + 'high', 'prevGroup_' + str(li) + 'low',
                                           'prevGroup_' + str(li) + 'max_price', 'prevGroup_' + str(li) + 'min_price',
                                           'prevGroup_' + str(li) + 'lower_vegas',
                                           'prevGroup_' + str(li) + 'upper_vegas',
                                           'prevGroup_' + str(li) + 'macd_cross_total_duration',
                                           'prevGroup_' + str(li) + 'macd_cross_num'
                                           ]

        self.data_df['long_macd_need_look_backward'] = self.data_df['macd_cross_label'] == 0

        target_long_cols = ['long_critical_value', 'long_critical_value_id', 'long_critical_macd_cross_duration',
                            'long_macd_cross_num',
                            'long_critical_high', 'long_critical_low', 'long_critical_max_price',
                            'long_critical_min_price',
                            'long_critical_upper_vegas', 'long_critical_lower_vegas']
        for li in range(1, look_backward_group_num):
            target_long_cols += ['long_prevGroup_' + str(li) + 'critical_value',
                                 'long_prevGroup_' + str(li) + 'critical_value_id',
                                 'long_prevGroup_' + str(li) + 'high', 'long_prevGroup_' + str(li) + 'low',
                                 'long_prevGroup_' + str(li) + 'max_price', 'long_prevGroup_' + str(li) + 'min_price',
                                 'long_prevGroup_' + str(li) + 'lower_vegas',
                                 'long_prevGroup_' + str(li) + 'upper_vegas',
                                 'long_prevGroup_' + str(li) + 'macd_cross_total_duration',
                                 'long_prevGroup_' + str(li) + 'macd_cross_num'
                                 ]

        for ti in range(len(target_long_cols)):
            #print("Add column " + target_long_cols[ti])
            self.data_df[target_long_cols[ti]] = np.where(
                self.data_df['long_macd_need_look_backward'],
                self.data_df[need_look_backward_cols[ti]],
                self.data_df[no_need_look_backward_cols[ti]]
            )
            #print("Column " + target_long_cols[ti] + " in data_df? " + str(target_long_cols[ti] in self.data_df.columns))



        self.data_df['short_macd_need_look_backward'] = self.data_df['macd_cross_label'] == 1

        target_short_cols = ['short_critical_value', 'short_critical_value_id', 'short_critical_macd_cross_duration',
                             'short_macd_cross_num',
                             'short_critical_high', 'short_critical_low', 'short_critical_max_price',
                             'short_critical_min_price',
                             'short_critical_upper_vegas', 'short_critical_lower_vegas']
        for li in range(1, look_backward_group_num):
            target_short_cols += ['short_prevGroup_' + str(li) + 'critical_value',
                                  'short_prevGroup_' + str(li) + 'critical_value_id',
                                  'short_prevGroup_' + str(li) + 'high', 'short_prevGroup_' + str(li) + 'low',
                                  'short_prevGroup_' + str(li) + 'max_price',
                                  'short_prevGroup_' + str(li) + 'min_price',
                                  'short_prevGroup_' + str(li) + 'lower_vegas',
                                  'short_prevGroup_' + str(li) + 'upper_vegas',
                                  'short_prevGroup_' + str(li) + 'macd_cross_total_duration',
                                  'short_prevGroup_' + str(li) + 'macd_cross_num'
                                  ]

        for ti in range(len(target_short_cols)):
            self.data_df[target_short_cols[ti]] = np.where(
                self.data_df['short_macd_need_look_backward'],
                self.data_df[need_look_backward_cols[ti]],
                self.data_df[no_need_look_backward_cols[ti]]
            )

        self.data_df['id'] = list(range(self.data_df.shape[0]))

        self.data_df['short_macd_long_cond0'] = self.data_df['long_critical_value'] < 0
        self.data_df['short_macd_long_cond1'] = self.data_df['long_critical_value'] > self.data_df['long_prevGroup_2critical_value']
        self.data_df['short_macd_long_cond2'] = self.data_df['long_critical_min_price'] < self.data_df['long_prevGroup_2min_price']
        self.data_df['short_macd_long_cond3'] = (self.data_df['macd_gradient'] > 0) &\
                                                (self.data_df['macd'] > self.data_df['msignal']) & (self.data_df['prev_macd'] < self.data_df['prev_msignal'])

        self.data_df['short_macd_short_cond0'] = self.data_df['short_critical_value'] > 0
        self.data_df['short_macd_short_cond1'] = self.data_df['short_critical_value'] < self.data_df['short_prevGroup_2critical_value']
        self.data_df['short_macd_short_cond2'] = self.data_df['short_critical_max_price'] > self.data_df['short_prevGroup_2max_price']
        self.data_df['short_macd_short_cond3'] = (self.data_df['macd_gradient'] < 0) &\
                                                 (self.data_df['macd'] < self.data_df['msignal']) & (self.data_df['prev_macd'] > self.data_df['prev_msignal'])


        # self.data_df['long_macd_long_enter'] = (self.data_df['prev_macd2'] < self.data_df['prev_msignal2']) & (self.data_df['macd2_gradient'] > 0) &\
        #                                       (self.data_df['macd2'] < 0)
        # self.data_df['long_macd_short_enter'] = (self.data_df['prev_macd2'] > self.data_df['prev_msignal2']) & (self.data_df['macd2_gradient'] < 0) &\
        #                                       (self.data_df['macd2'] > 0)

        #self.data_df['long_macd_long_enter'] = (self.data_df['macd2_gradient'] > 0) &\
        #                                      (self.data_df['macd2'] < 0)
        #self.data_df['long_macd_short_enter'] = (self.data_df['macd2_gradient'] < 0) &\
        #                                      (self.data_df['macd2'] > 0)

        #self.data_df['long_macd_long_enter'] = (self.data_df['macd2_gradient'] > 0)
        #self.data_df['long_macd_short_enter'] = (self.data_df['macd2_gradient'] < 0)

        #self.data_df['long_macd_long_enter'] = (self.data_df['macd2_gradient'] > 0) & (self.data_df['prev_macd2_gradient'] > 0)
        #self.data_df['long_macd_short_enter'] = (self.data_df['macd2_gradient'] < 0) & (self.data_df['prev_macd2_gradient'] < 0)

        #Singapore  3gradients_positive
        macd_enter_gradient_num = self.optimal_gradient_num
        self.data_df['long_macd_long_enter'] = (self.data_df[macd_gradient] > 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + macd_gradient] > 0) for i in range(1, macd_enter_gradient_num)])
        self.data_df['long_macd_short_enter'] = (self.data_df[macd_gradient] < 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + macd_gradient] < 0) for i in range(1, macd_enter_gradient_num)])


        self.data_df['long_macd_long_enter_ready'] = (self.data_df[macd_gradient] > 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + macd_gradient] > 0) for i in range(1, macd_enter_gradient_num-1)])
        self.data_df['long_macd_short_enter_ready'] = (self.data_df[macd_gradient] < 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + macd_gradient] < 0) for i in range(1, macd_enter_gradient_num-1)])


        if do_message_printing and self.is_notify and print_ready:
            if self.data_df.iloc[-1]['long_macd_long_enter_ready'] and (not self.data_df.iloc[-1]['long_macd_long_enter']) and self.current_position <= 0:
                sendEmail("Ready to Open Long Position of " + str(initial_entry_value) + " USD for " + self.currency +  " at " + str(self.data_df.iloc[-1]['time'] + timedelta(hours = 2)), "")
            elif self.data_df.iloc[-1]['long_macd_short_enter_ready'] and (not self.data_df.iloc[-1]['long_macd_short_enter']) and self.current_position >= 0:
                sendEmail("Ready to Open Short Position of " + str(initial_entry_value) + " USD for " + self.currency +  " at " + str(self.data_df.iloc[-1]['time'] + timedelta(hours = 2)), "")



        #Added one
        # self.data_df['long_macd_long_enter'] = self.data_df['long_macd_long_enter'] |\
        #                                        ((self.data_df['prev_macd2'] < self.data_df['prev_msignal2']) & (self.data_df['macd2'] >= self.data_df['msignal2']))
        # self.data_df['long_macd_short_enter'] = self.data_df['long_macd_short_enter'] |\
        #                                        ((self.data_df['prev_macd2'] > self.data_df['prev_msignal2']) & (self.data_df['macd2'] <= self.data_df['msignal2']))

        # self.data_df['long_macd_long_enter'] = ((self.data_df['prev_macd2'] < self.data_df['prev_msignal2']) & (self.data_df['macd2'] >= self.data_df['msignal2']))
        # self.data_df['long_macd_short_enter'] = ((self.data_df['prev_macd2'] > self.data_df['prev_msignal2']) & (self.data_df['macd2'] <= self.data_df['msignal2']))

        #self.data_df['long_macd_long_enter'] = (self.data_df['macd2_gradient'] > 0) & (self.data_df['prev_macd2_gradient'] > 0) & (self.data_df['prev2_macd2_gradient'] > 0)
        #self.data_df['long_macd_short_enter'] = (self.data_df['macd2_gradient'] < 0) & (self.data_df['prev_macd2_gradient'] < 0) & (self.data_df['prev2_macd2_gradient'] < 0)



        self.data_df['short_macd_long_enter'] = reduce(lambda left, right: left & right, [self.data_df['short_macd_long_cond' + str(i)] for i in range(4)])
        self.data_df['short_macd_short_enter'] = reduce(lambda left, right: left & right, [self.data_df['short_macd_short_cond' + str(i)] for i in range(4)])




        if self.reverse_strategy:
            self.data_df['temp'] = self.data_df['long_macd_long_enter']
            self.data_df['long_macd_long_enter'] = self.data_df['long_macd_short_enter']
            self.data_df['long_macd_short_enter'] = self.data_df['temp']



        self.data_df['macd_long_enter'] = self.data_df['macd'].notnull() & self.data_df['msignal'].notnull()
        self.data_df['macd_short_enter'] = self.data_df['macd'].notnull() & self.data_df['msignal'].notnull()

        self.data_df['macd_long_enter'] = self.data_df['long_macd_long_enter']
        self.data_df['macd_short_enter'] = self.data_df['long_macd_short_enter']


        if use_guppy_filter:
            self.data_df['macd_long_enter'] = self.data_df['macd_long_enter'] & (~self.data_df['guppy_all_strong_aligned_short'])
            self.data_df['macd_short_enter'] = self.data_df['macd_short_enter'] & (~self.data_df['guppy_all_strong_aligned_long'])
        elif use_guppy_condition:
            self.data_df['macd_long_enter'] = self.data_df['macd_long_enter'] & (self.data_df['guppy_all_strong_aligned_long'])
            self.data_df['macd_short_enter'] = self.data_df['macd_short_enter'] & (self.data_df['guppy_all_strong_aligned_short'])

        # self.data_df['macd_long_enter'] = self.data_df['short_macd_long_enter']
        # self.data_df['macd_short_enter'] = self.data_df['short_macd_short_enter']

        #self.data_df['macd_long_enter'] = self.data_df['short_macd_long_enter'] | self.data_df['long_macd_long_enter']
        #self.data_df['macd_short_enter'] = self.data_df['short_macd_short_enter'] | self.data_df['long_macd_short_enter']


        self.data_df['short_macd_long_exit'] = (self.data_df['macd_gradient'] < 0) & (self.data_df['macd'] < self.data_df['msignal'])
        self.data_df['short_macd_short_exit'] = (self.data_df['macd_gradient'] > 0) & (self.data_df['macd'] > self.data_df['msignal'])



        macd_exit_gradient_num = self.optimal_gradient_num
        self.data_df['long_macd_long_exit'] = (self.data_df[macd_gradient] < 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + macd_gradient] < 0) for i in range(1, macd_exit_gradient_num)])
        self.data_df['long_macd_short_exit'] = (self.data_df[macd_gradient] > 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + macd_gradient] > 0) for i in range(1, macd_exit_gradient_num)])

        if self.reverse_strategy:
            self.data_df['temp'] = self.data_df['long_macd_long_enter_ready']
            self.data_df['long_macd_long_enter_ready'] = self.data_df['long_macd_short_enter_ready']
            self.data_df['long_macd_short_enter_ready'] = self.data_df['temp']

            self.data_df['temp'] = self.data_df['long_macd_long_exit']
            self.data_df['long_macd_long_exit'] = self.data_df['long_macd_short_exit']
            self.data_df['long_macd_short_exit'] = self.data_df['temp']




        # self.data_df['long_macd_long_exit'] = self.data_df['long_macd_long_exit'] |\
        #                                        ((self.data_df['prev_macd2'] > self.data_df['prev_msignal2']) & (self.data_df['macd2'] <= self.data_df['msignal2']))
        # self.data_df['long_macd_short_exit'] = self.data_df['long_macd_short_exit'] |\
        #                                        ((self.data_df['prev_macd2'] < self.data_df['prev_msignal2']) & (self.data_df['macd2'] >= self.data_df['msignal2']))

        #self.data_df['long_macd_long_exit'] = ((self.data_df['prev_macd2'] > self.data_df['prev_msignal2']) & (self.data_df['macd2'] <= self.data_df['msignal2']))
        #self.data_df['long_macd_short_exit'] = ((self.data_df['prev_macd2'] < self.data_df['prev_msignal2']) & (self.data_df['macd2'] >= self.data_df['msignal2']))

        #self.data_df['long_macd_long_exit'] = self.data_df['macd2_gradient'] < 0
        #self.data_df['long_macd_short_exit'] = self.data_df['macd2_gradient'] > 0

        #self.data_df['long_macd_long_exit'] = (self.data_df['macd2_gradient'] < 0) & (self.data_df['prev_macd2_gradient'] < 0) & (self.data_df['prev2_macd2_gradient'] < 0)
        #self.data_df['long_macd_short_exit'] = (self.data_df['macd2_gradient'] > 0) & (self.data_df['prev_macd2_gradient'] > 0) & (self.data_df['prev2_macd2_gradient'] > 0)

        result_data = []
        result_columns = ['long_trade_id', 'short_trade_id', 'instrument', 'side', 'entry_id', 'entry_time',
                          'entry_price', 'exit_id', 'exit_time', 'exit_price', 'is_win']

        if do_smart_execution:

            if do_smart_execution:
                result_columns += ['pnl']

            strategy_record_columns = ['side', 'long_trade_id', 'short_trade_id', 'strategy_id', 'leverage', 'entry_time', 'entry_price', 'entry_value', 'exit_time', 'exit_price', 'exit_value', 'pnl']
            strategy_execution_record_columns = ['side', 'long_trade_id', 'short_trade_id', 'strategy_id', 'execution_id', 'leverage', 'take_profit_pct', 'take_profit_price', 'take_loss_pct', 'take_loss_price',
                                                 'entry_time', 'entry_price', 'entry_value', 'exit_time', 'exit_price', 'exit_value', 'pnl']

            long_strategy_records = []
            long_strategy_execution_records = []

            short_strategy_records = []
            short_strategy_execution_records = []



        print("")
        print("Calculating Long positions.............")
        print("")

        long_start_ids = which(self.data_df['macd_long_enter'])

        is_effective = [1] * len(long_start_ids)

        long_trade_id = 0
        for i in range(len(long_start_ids)):

            if is_effective[i] == 0:
                self.data_df.at[long_start_ids[i], 'macd_long_enter'] = False
                continue

            temp_i = i
            long_start_id = long_start_ids[i]
            long_fire_data = self.data_df.iloc[long_start_id]

            instrument = long_fire_data['currency']
            entry_time = long_fire_data['time']
            entry_price = long_fire_data['close']
            entry_id = long_fire_data['id']

            long_trade_id += 1

            is_short_macd_fire = not long_fire_data['long_macd_long_enter']

            j = 1

            long_macd_indicate_long = False
            exit_id = -1
            exit_time = None
            exit_price = -1
            is_win = False


            if do_message_printing:
                if self.is_notify and (long_start_id == self.data_df.shape[0] - 1 or print_email_message_to_file):

                    current_time = str(self.data_df.iloc[long_start_id]['time'] + timedelta(hours = 1))

                    position = initial_entry_value/entry_price * default_leverage

                    print("Before position = " + str(position))

                    if entry_price >= 1:
                        position = round(position, 3)
                    else:
                        position = int(round(position, 0))

                    print("After position = " + str(position))

                    delta_position = position - self.current_position

                    self.current_position = position


                    message_title = "Long " + self.currency + " " + str(delta_position) + " units"

                    message = "At " + current_time + ", long " + self.currency + " roughly " + str(delta_position) + " units at entry price " + str(round(entry_price, self.decimal)) + "\n"
                    message += "This makes it now at a long position of " + str(self.current_position) + " units with an actual notional of " + str(initial_entry_value) + " dollar\n"

                    print("message_title = " + message_title)
                    print("message:")
                    print(message)

                    if not print_email_message_to_file:
                        sendEmail(message_title, message)
                    else:
                        self.cache_email_messages(message_title, message, current_time)



            if do_smart_execution:
                strategy_executions = []
                for k in range(len(self.leverage)):
                    strategy_execution = StrategyExecution(side = 1, leverage = self.leverage[k], take_profit_pct = self.take_profit_pct[k], take_loss_pct = self.take_loss_pct[k],
                                                           strategy_id = k+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                           execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                           strategy_entry_value = self.each_strategy_entry_value, execution_entry_value = self.each_strategy_entry_value)
                    strategy_executions += [strategy_execution]

                #This is the extra one
                strategy_execution = StrategyExecution(side = 1, leverage = self.leverage[0], take_profit_pct = self.take_profit_pct[0], take_loss_pct = self.take_loss_pct[0],
                                                           strategy_id = len(self.leverage)+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                           execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                           strategy_entry_value = initial_entry_value, execution_entry_value = initial_entry_value)

                strategy_executions += [strategy_execution]


                total_strategy_pnl = 0

            while long_start_id + j < self.data_df.shape[0]:

                cur_data_1h = self.data_df.iloc[long_start_id + j]
                #print("")
                #print("Long 1h time = " + str(self.data_df.iloc[long_start_id + j]['time']) + '..............................')

                if do_smart_execution:

                    can_use_5min = False
                    if use_5min_in_smart_execution:
                        loc_start = self.data_df.iloc[long_start_id + j]['location']
                        loc_end = None

                        if loc_start is not None and loc_start > 0:
                            if long_start_id + j + 1 < self.data_df.shape[0]:
                                loc_end = self.data_df.iloc[long_start_id + j + 1]['location']
                                if loc_end is None:
                                    loc_end = self.data_df_5min.shape[0]
                            else:
                                loc_end = min(loc_start + 12, self.data_df_5min.shape[0])


                        if loc_start is not None and loc_end is not None and loc_start >= 0 and loc_end > 0:
                            loc_start = int(loc_start)
                            loc_end = int(loc_end)
                            can_use_5min = True

                    if can_use_5min:
                        use_data_df = self.data_df_5min
                    else:
                        use_data_df = self.data_df
                        loc_start = long_start_id + j
                        loc_end = long_start_id + j + 1


                    #print("loc_start = " + str(loc_start))
                    #print("loc_end = " + str(loc_end))

                    for y in range(loc_start, loc_end):
                        cur_data = use_data_df.iloc[y]

                        #print("    Long 5min time = " + str(cur_data['time']))

                        for k in range(len(strategy_executions)):

                            execution = strategy_executions[k]

                            if (not execution.active) or (execution.execution_entry_time > cur_data['time']):  #New Code
                                continue


                            while True:

                                ###############

                                if execution.execution_entry_time == cur_data['time'] and execution.execution_entry_price > cur_data['open'] + (1e-5):
                                    loss_ref_price = cur_data['close']
                                else:
                                    loss_ref_price = cur_data['low']

                                hit_stop_loss = (loss_ref_price <= execution.take_loss_price) and (execution.take_loss_price > execution.strategy_entry_price)

                                if hit_stop_loss:

                                    execution.exit_execution(execution_exit_time=cur_data['time'], execution_exit_price=execution.take_loss_price,
                                                             is_signal_exit=False, is_extra_execution=(k == len(self.leverage)))

                                    long_strategy_execution_records += [['long', long_trade_id, 0, k+1, execution.execution_id, execution.leverage,
                                                                    execution.take_profit_pct, execution.take_profit_price, execution.take_loss_pct, execution.take_loss_price,
                                                                    execution.execution_entry_time, execution.execution_entry_price, execution.execution_entry_value,
                                                                    execution.execution_exit_time, execution.execution_exit_price, execution.execution_exit_value,
                                                                    execution.pnl]]

                                    ######### New Code ##############
                                    if can_use_5min and do_reentry and k < len(strategy_executions) - 1:
                                        if cur_data_1h['close'] > execution.take_loss_price:
                                             #Allow re-entry after 1h bar closes
                                             if long_start_id + j + 1 < self.data_df.shape[0]:
                                                 next_cur_data_1h = self.data_df.iloc[long_start_id + j + 1]

                                                 next_execution = StrategyExecution(side=execution.side, leverage=execution.leverage, take_profit_pct=execution.take_profit_pct,
                                                                       take_loss_pct=execution.take_loss_pct, strategy_id=execution.strategy_id, execution_id=execution.execution_id+1,
                                                                       strategy_entry_time=execution.strategy_entry_time, strategy_entry_price=execution.strategy_entry_price,
                                                                       execution_entry_time=next_cur_data_1h['time'], execution_entry_price=next_cur_data_1h['open'],
                                                                       strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value
                                                                       )
                                                 strategy_executions[k] = next_execution
                                    ###################################
                                    

                                    break

                                ###############

                                hit_stop_profit = cur_data['high'] >= execution.take_profit_price
                                if not hit_stop_profit:
                                    break

                                execution.exit_execution(execution_exit_time=cur_data['time'], execution_exit_price=execution.take_profit_price,
                                                         is_signal_exit=False, is_extra_execution=(k == len(self.leverage)))

                                long_strategy_execution_records += [['long', long_trade_id, 0, k+1, execution.execution_id, execution.leverage,
                                                                execution.take_profit_pct, execution.take_profit_price, execution.take_loss_pct, execution.take_loss_price,
                                                                execution.execution_entry_time, execution.execution_entry_price, execution.execution_entry_value,
                                                                execution.execution_exit_time, execution.execution_exit_price, execution.execution_exit_value,
                                                                execution.pnl]]

                                if k < len(strategy_executions) - 1:
                                    next_execution = StrategyExecution(side=execution.side, leverage=execution.leverage, take_profit_pct=execution.take_profit_pct,
                                                                   take_loss_pct=execution.take_loss_pct, strategy_id=execution.strategy_id, execution_id=execution.execution_id+1,
                                                                   strategy_entry_time=execution.strategy_entry_time, strategy_entry_price=execution.strategy_entry_price,
                                                                   execution_entry_time=cur_data['time'], execution_entry_price=execution.execution_exit_price,
                                                                   strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value
                                                                   )
                                    strategy_executions[k] = next_execution
                                    execution = strategy_executions[k]
                                    #hit_stop_profit = cur_data['high'] >= execution.take_profit_price
                                else:
                                    break


                if (not do_smart_execution) or can_use_5min:
                    cur_data = self.data_df.iloc[long_start_id + j]

                if long_macd_indicate_long or (not is_short_macd_fire):
                    is_exit = cur_data['long_macd_long_exit'] or cur_data['macd_short_enter']
                    #is_exit = cur_data['long_macd_long_exit'] or cur_data['short_macd_short_enter']
                else:
                    is_exit = cur_data['short_macd_long_exit'] or cur_data['macd_short_enter']
                    #is_exit = cur_data['short_macd_long_exit'] or cur_data['short_macd_short_enter']

                if is_exit:

                    exit_id = cur_data['id']
                    exit_time = cur_data['time']
                    exit_price = cur_data['close']
                    is_win = exit_price > entry_price

                    if do_smart_execution:
                        for k in range(len(strategy_executions)):
                            execution = strategy_executions[k]
                            if not execution.active:
                                continue

                            execution.exit_execution(execution_exit_time=exit_time, execution_exit_price=exit_price, is_signal_exit=True,
                                                     is_extra_execution=(k == len(self.leverage)))
                            long_strategy_execution_records += [['long', long_trade_id, 0, k+1, execution.execution_id, execution.leverage,
                                                            execution.take_profit_pct, execution.take_profit_price, execution.take_loss_pct, execution.take_loss_price,
                                                            execution.execution_entry_time, execution.execution_entry_price, execution.execution_entry_value,
                                                            execution.execution_exit_time, execution.execution_exit_price, execution.execution_exit_value,
                                                            execution.pnl]]


                        #Prepare strategy_record for 5 strategies
                        for k in range(len(strategy_executions)):

                            execution = strategy_executions[k]
                            strategy_pnl = execution.execution_exit_value - execution.strategy_entry_value
                            total_strategy_pnl += strategy_pnl
                            long_strategy_records += [['long', long_trade_id, 0, k+1, execution.leverage, execution.strategy_entry_time, execution.strategy_entry_price,
                                                  execution.strategy_entry_value, execution.execution_exit_time, execution.execution_exit_price,
                                                  execution.execution_exit_value, strategy_pnl]]



                    break



                if is_short_macd_fire and (not long_macd_indicate_long):
                    long_macd_indicate_long = (cur_data['macd2'] > cur_data['msignal2']) and (cur_data['macd2_gradient'] > 0)

                if temp_i + 1 < len(long_start_ids) and long_start_ids[temp_i + 1] == long_start_id + j:

                    is_effective[temp_i + 1] = 0
                    temp_i += 1

                j += 1


            result_data += [[long_trade_id, 0, instrument, 'long', entry_id, entry_time, entry_price, exit_id, exit_time, exit_price, is_win]
                            + ([total_strategy_pnl] if do_smart_execution else [])]


        if do_smart_execution:
            self.long_strategy_df = pd.DataFrame(data = long_strategy_records, columns = strategy_record_columns)
            self.long_strategy_execution_df = pd.DataFrame(data = long_strategy_execution_records, columns = strategy_execution_record_columns)

        long_df = pd.DataFrame(data = result_data, columns = result_columns)

        if not do_smart_execution:
            long_df['pnl'] = (long_df['exit_price'] - long_df['entry_price']) / long_df['entry_price'] * initial_entry_value * default_leverage


        long_df['pnl'] = long_df['pnl'].apply(lambda x: round(x, 2))

        write_long_df = long_df.copy()
        write_long_df['win'] = np.where(
            write_long_df['exit_price'] == -1,
            -1,
            np.where(write_long_df['is_win'], 1, 0)
        )
        write_long_df = write_long_df.drop(columns = ['is_win'])


        long_df = long_df[long_df['exit_price'] > 0]



        long_df['entry_id'] = long_df['entry_id'].astype(int)
        long_df['exit_id'] = long_df['exit_id'].astype(int)

        long_win_num = long_df[long_df['is_win']].shape[0]
        long_lose_num = long_df[~long_df['is_win']].shape[0]

        self.long_df = long_df




        ##########################################################

        result_data = []
        print("")
        print("Calculating Short positions.............")
        print("")

        short_start_ids = which(self.data_df['macd_short_enter'])

        is_effective = [1] * len(short_start_ids)

        short_trade_id = 0
        for i in range(len(short_start_ids)):

            if is_effective[i] == 0:
                self.data_df.at[short_start_ids[i], 'macd_short_enter'] = False
                continue

            temp_i = i
            short_start_id = short_start_ids[i]
            short_fire_data = self.data_df.iloc[short_start_id]

            instrument = short_fire_data['currency']
            entry_time = short_fire_data['time']
            entry_price = short_fire_data['close']
            entry_id = short_fire_data['id']

            short_trade_id += 1

            is_short_macd_fire = not short_fire_data['long_macd_short_enter']

            j = 1

            long_macd_indicate_short = False
            exit_id = -1
            exit_time = None
            exit_price = -1
            is_win = False

            if do_message_printing:
                if self.is_notify and (short_start_id == self.data_df.shape[0] - 1 or print_email_message_to_file):

                    current_time = str(self.data_df.iloc[short_start_id]['time'] + timedelta(hours = 1))

                    position = -initial_entry_value/entry_price * default_leverage
                    if entry_price >= 1:
                        position = round(position, 3)
                    else:
                        position = int(round(position, 0))

                    delta_position = position - self.current_position

                    self.current_position = position


                    message_title = "Short " + self.currency + " " + str(-delta_position) + " units"

                    message = "At " + current_time + ", short " + self.currency + " roughly " + str(-delta_position) + " units at entry price " + str(round(entry_price, self.decimal)) + "\n"
                    message += "This makes it now at a short position of " + str(self.current_position) + " units with an actual notional of " + str(initial_entry_value) + " dollar\n"

                    print("message_title = " + message_title)
                    print("message:")
                    print(message)

                    if not print_email_message_to_file:
                        sendEmail(message_title, message)
                    else:
                        self.cache_email_messages(message_title, message, current_time)


            if do_smart_execution:
                strategy_executions = []
                for k in range(len(self.leverage)):
                    strategy_execution = StrategyExecution(side = -1, leverage = self.leverage[k], take_profit_pct = self.take_profit_pct[k], take_loss_pct = self.take_loss_pct[k],
                                                           strategy_id = k+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                           execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                           strategy_entry_value = self.each_strategy_entry_value, execution_entry_value = self.each_strategy_entry_value)
                    strategy_executions += [strategy_execution]

                #This is the extra one
                strategy_execution = StrategyExecution(side = -1, leverage = self.leverage[0], take_profit_pct = self.take_profit_pct[0], take_loss_pct = self.take_loss_pct[0],
                                                           strategy_id = len(self.leverage)+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                           execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                           strategy_entry_value = initial_entry_value, execution_entry_value = initial_entry_value)

                strategy_executions += [strategy_execution]


                total_strategy_pnl = 0



            while short_start_id + j < self.data_df.shape[0]:

                #cur_data = self.data_df.iloc[short_start_id + j]

                #print("")
                #print("Short 1h time = " + str(self.data_df.iloc[short_start_id + j]['time']) + '..............................')

                if do_smart_execution:

                    can_use_5min = False
                    if use_5min_in_smart_execution:
                        loc_start = self.data_df.iloc[short_start_id + j]['location']
                        loc_end = None

                        if loc_start is not None and loc_start > 0:
                            if short_start_id + j + 1 < self.data_df.shape[0]:
                                loc_end = self.data_df.iloc[short_start_id + j + 1]['location']
                                if loc_end is None:
                                    loc_end = self.data_df_5min.shape[0]
                            else:
                                loc_end = min(loc_start + 12, self.data_df_5min.shape[0])

                        if loc_start is not None and loc_end is not None and loc_start >= 0 and loc_end > 0:
                            loc_start = int(loc_start)
                            loc_end = int(loc_end)
                            can_use_5min = True

                    if can_use_5min:
                        use_data_df = self.data_df_5min
                    else:
                        use_data_df = self.data_df
                        loc_start = short_start_id + j
                        loc_end = short_start_id + j + 1


                    for y in range(loc_start, loc_end):
                        cur_data = use_data_df.iloc[y]

                        #print("    Short 5min time = " + str(cur_data['time']))

                        for k in range(len(strategy_executions)):

                            execution = strategy_executions[k]

                            if (not execution.active) or (execution.execution_entry_time > cur_data['time']):
                                continue


                            while True:

                                if execution.execution_entry_time == cur_data['time'] and execution.execution_entry_price < cur_data['open'] - (1e-5):
                                    loss_ref_price = cur_data['close']
                                else:
                                    loss_ref_price = cur_data['high']


                                hit_stop_loss = (loss_ref_price >= execution.take_loss_price) and (execution.take_loss_price < execution.strategy_entry_price)

                                # if short_trade_id == 49 and execution.strategy_id == 2 and execution.execution_id == 6:
                                #     print("short_trade_id = " + str(short_trade_id))
                                #     print("strategy_id = " + str(execution.strategy_id))
                                #     print("execution_id = " + str(execution.execution_id))
                                #
                                #     print("cur time = " + str(cur_data['time']))
                                #     print("execution_entry_time = " + str(execution.execution_entry_time))
                                #     print("execution_entry_price = " + str(execution.execution_entry_price))
                                #     print("open = " + str(cur_data['open']))
                                #     print("loss_ref_price = " + str(loss_ref_price))
                                #     print("take_loss_price = " + str(execution.take_loss_price))
                                #     print("strategy_entry_price = " + str(execution.strategy_entry_price))
                                #     print("hit_stop_loss = " + str(hit_stop_loss))
                                #     #sys.exit(0)

                                if hit_stop_loss:

                                    execution.exit_execution(execution_exit_time=cur_data['time'], execution_exit_price=execution.take_loss_price,
                                                             is_signal_exit=False, is_extra_execution=(k == len(self.leverage)))

                                    short_strategy_execution_records += [['short', 0, short_trade_id, k+1, execution.execution_id, execution.leverage,
                                                                    execution.take_profit_pct, execution.take_profit_price, execution.take_loss_pct, execution.take_loss_price,
                                                                    execution.execution_entry_time, execution.execution_entry_price, execution.execution_entry_value,
                                                                    execution.execution_exit_time, execution.execution_exit_price, execution.execution_exit_value,
                                                                    execution.pnl]]

                                     ######### New Code ##############
                                    if can_use_5min and do_reentry and k < len(strategy_executions) - 1:
                                        if cur_data_1h['close'] < execution.take_loss_price:

                                             #Allow re-entry after 1h bar closes
                                             if short_start_id + j + 1 < self.data_df.shape[0]:
                                                 next_cur_data_1h = self.data_df.iloc[short_start_id + j + 1]

                                                 next_execution = StrategyExecution(side=execution.side, leverage=execution.leverage, take_profit_pct=execution.take_profit_pct,
                                                                       take_loss_pct=execution.take_loss_pct, strategy_id=execution.strategy_id, execution_id=execution.execution_id+1,
                                                                       strategy_entry_time=execution.strategy_entry_time, strategy_entry_price=execution.strategy_entry_price,
                                                                       execution_entry_time=next_cur_data_1h['time'], execution_entry_price=next_cur_data_1h['open'],
                                                                       strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value
                                                                       )
                                                 strategy_executions[k] = next_execution
                                    ###################################

                                    break

                                hit_stop_profit = cur_data['low'] <= execution.take_profit_price
                                if not hit_stop_profit:
                                    break

                                execution.exit_execution(execution_exit_time=cur_data['time'], execution_exit_price=execution.take_profit_price,
                                                         is_signal_exit=False, is_extra_execution=(k == len(self.leverage)))

                                short_strategy_execution_records += [['short', 0, short_trade_id, k+1, execution.execution_id, execution.leverage,
                                                                execution.take_profit_pct, execution.take_profit_price, execution.take_loss_pct, execution.take_loss_price,
                                                                execution.execution_entry_time, execution.execution_entry_price, execution.execution_entry_value,
                                                                execution.execution_exit_time, execution.execution_exit_price, execution.execution_exit_value,
                                                                execution.pnl]]

                                # if short_trade_id == 49 and execution.strategy_id == 2 and execution.execution_id == 6:
                                #     print("Last record so far:")
                                #     print(short_strategy_execution_records[-1])


                                if k < len(strategy_executions) - 1:
                                    next_execution = StrategyExecution(side=execution.side, leverage=execution.leverage, take_profit_pct=execution.take_profit_pct,
                                                                   take_loss_pct=execution.take_loss_pct, strategy_id=execution.strategy_id, execution_id=execution.execution_id+1,
                                                                   strategy_entry_time=execution.strategy_entry_time, strategy_entry_price=execution.strategy_entry_price,
                                                                   execution_entry_time=cur_data['time'], execution_entry_price=execution.execution_exit_price,
                                                                   strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value
                                                                   )
                                    strategy_executions[k] = next_execution
                                    execution = strategy_executions[k]
                                    #hit_stop_profit = cur_data['low'] <= execution.take_profit_price
                                else:
                                    break


                if (not do_smart_execution) or can_use_5min:
                    cur_data = self.data_df.iloc[short_start_id + j]

                if long_macd_indicate_short or (not is_short_macd_fire):
                    is_exit = cur_data['long_macd_short_exit'] or cur_data['macd_long_enter']
                    #is_exit = cur_data['long_macd_short_exit'] or cur_data['short_macd_long_enter']
                else:
                    is_exit = cur_data['short_macd_short_exit'] or cur_data['macd_long_enter']
                    #is_exit = cur_data['short_macd_short_exit'] or cur_data['short_macd_long_enter']

                if is_exit:
                    exit_id = cur_data['id']
                    exit_time = cur_data['time']
                    exit_price = cur_data['close']
                    is_win = exit_price < entry_price

                    if do_smart_execution:
                        for k in range(len(strategy_executions)):
                            execution = strategy_executions[k]
                            if not execution.active:
                                continue

                            execution.exit_execution(execution_exit_time=exit_time, execution_exit_price=exit_price,
                                                     is_signal_exit=True, is_extra_execution=(k == len(self.leverage)))
                            short_strategy_execution_records += [['short', 0, short_trade_id, k+1, execution.execution_id, execution.leverage,
                                                            execution.take_profit_pct, execution.take_profit_price, execution.take_loss_pct, execution.take_loss_price,
                                                            execution.execution_entry_time, execution.execution_entry_price, execution.execution_entry_value,
                                                            execution.execution_exit_time, execution.execution_exit_price, execution.execution_exit_value,
                                                            execution.pnl]]


                        #Prepare strategy_record for 5 strategies
                        for k in range(len(strategy_executions)):

                            execution = strategy_executions[k]
                            strategy_pnl = execution.execution_exit_value - execution.strategy_entry_value
                            total_strategy_pnl += strategy_pnl
                            short_strategy_records += [['short', 0, short_trade_id, k+1, execution.leverage, execution.strategy_entry_time, execution.strategy_entry_price,
                                                  execution.strategy_entry_value, execution.execution_exit_time, execution.execution_exit_price,
                                                  execution.execution_exit_value, strategy_pnl]]


                    break

                if is_short_macd_fire and (not long_macd_indicate_short):
                    long_macd_indicate_short = (cur_data['macd2'] < cur_data['msignal2']) and (cur_data['macd2_gradient'] < 0)

                if temp_i + 1 < len(short_start_ids) and short_start_ids[temp_i + 1] == short_start_id + j:
                    is_effective[temp_i + 1] = 0
                    temp_i += 1

                j += 1


            result_data += [[0, short_trade_id, instrument, 'short', entry_id, entry_time, entry_price, exit_id, exit_time, exit_price, is_win]
                            + ([total_strategy_pnl] if do_smart_execution else [])]


        if do_smart_execution:
            self.short_strategy_df = pd.DataFrame(data = short_strategy_records, columns = strategy_record_columns)
            self.short_strategy_execution_df = pd.DataFrame(data = short_strategy_execution_records, columns = strategy_execution_record_columns)


        short_df = pd.DataFrame(data=result_data, columns=result_columns)

        if not do_smart_execution:
            short_df['pnl'] = -(short_df['exit_price'] - short_df['entry_price']) / short_df['entry_price'] * initial_entry_value  * default_leverage

        short_df['pnl'] = short_df['pnl'].apply(lambda x: round(x, 2))

        write_short_df = short_df.copy()
        write_short_df['win'] = np.where(
            write_short_df['exit_price'] == -1,
            -1,
            np.where(write_short_df['is_win'], 1, 0)
        )
        write_short_df = write_short_df.drop(columns=['is_win'])

        short_df = short_df[short_df['exit_price'] > 0]




        short_df['entry_id'] = short_df['entry_id'].astype(int)
        short_df['exit_id'] = short_df['exit_id'].astype(int)

        short_win_num = short_df[short_df['is_win']].shape[0]
        short_lose_num = short_df[~short_df['is_win']].shape[0]

        self.short_df = short_df


        win_num = long_win_num + short_win_num
        lose_num = long_lose_num + short_lose_num

        total_num = win_num + lose_num
        total_long_num = long_win_num + long_lose_num
        total_short_num = short_win_num + short_lose_num

        win_pct = 0 if total_num == 0 else win_num / total_num
        long_win_pct = 0 if total_long_num == 0 else long_win_num / total_long_num
        short_win_pct = 0 if total_short_num == 0 else short_win_num / total_short_num

        day_num = len(pd.Series(self.data_df['date'].unique()).dt.to_pydatetime())

        trade_num_per_day = round(total_num/day_num,1)

        summary_df = pd.DataFrame({'Currency': [self.currency], 'Trade Num': [total_num], 'Day Num': [day_num], 'Trade Per Day': [trade_num_per_day],
                                   'Win Num': [win_num],
                                   'Win Pct': [round(win_pct * 100.0) / 100.0],
                                   'Long Trade Num': [total_long_num], 'Long Win Num': [long_win_num],
                                   'Long Win Pct': [round(long_win_pct * 100.0) / 100.0],
                                   'Short Trade Num': [total_short_num], 'Short Win Num': [short_win_num],
                                   'Short Win Pct': [round(short_win_pct * 100.0) / 100.0],
                                   })

        self.full_summary_df = summary_df

        if report_performance:
            print("Performance Summary")
            print(self.full_summary_df)

        self.write_long_df = write_long_df
        self.write_short_df = write_short_df
        self.macd_group_summary_df = macd_group_summary_df
        self.critical_value_data_df = critical_value_data_df






        #########################




    def cache_email_messages(self, title, content, time):

        self.email_message_caches += [[title, content, time]]


    def post_processing(self):

        print("")
        print("Post processing currency pair " + str(self.currency))

        if print_email_message_to_file:

            email_messages_df = pd.DataFrame(data = self.email_message_caches, columns = ['title', 'content', 'time'])
            email_messages_df = email_messages_df.sort_values(by = ['time'])


            for i in range(email_messages_df.shape[0]):

                email_data_entry = email_messages_df.iloc[i]
                print(str(i+1) + ":", file = self.email_message_fd)
                print(email_data_entry['title'], file = self.email_message_fd)
                print(email_data_entry['content'], file = self.email_message_fd)
                #self.email_message_fd.flush()
                print("", file = self.email_message_fd)
                print("", file = self.email_message_fd)

            self.email_message_fd.close()


        if do_smart_execution:

            strategy_df = pd.concat([self.long_strategy_df, self.short_strategy_df])
            strategy_execution_df = pd.concat([self.long_strategy_execution_df, self.short_strategy_execution_df])

            strategy_df = strategy_df.sort_values(by = ['entry_time', 'exit_time'], ascending = True)
            strategy_execution_df = strategy_execution_df.sort_values(by = ['entry_time', 'exit_time'], ascending = True)

        write_df = pd.concat([self.write_long_df, self.write_short_df])

        write_df = write_df.sort_values(by = ['entry_time'], ascending = True)

        write_df['trade_id'] = np.array(list(range(write_df.shape[0]))) + 1
        write_df['trade_id'] = write_df['trade_id'].astype(int)

        write_df = write_df[['trade_id'] + [col for col in write_df.columns if col not in ['trade_id']]]

        write_df['pnl'] = np.where(
            write_df['exit_price'] > 0,
            write_df['pnl'],
            0
        )

        # print("write_df:")
        # print(write_df.iloc[0:10])

        #if self.is_finalized:
        self.data_df.to_csv(self.data_file, index = False)

        if self.data_df_5min is not None and self.data_file_5min is not None:
            self.data_df_5min.to_csv(self.data_file_5min, index = False)

        self.data_df.iloc[-1:][['currency','time', 'open', 'high', 'low', 'close']].to_csv(self.data_file[:-len('.csv')] + '_lastRow.csv', index = False)

        self.macd_group_summary_df.to_csv(self.data_file[:-len('.csv')] + '_macd_group_summary.csv', index = False)
        self.critical_value_data_df.to_csv(self.data_file[:-len('.csv')] + '_critial_value.csv', index = False)


        #write_df['id'] = list(range(write_df.shape[0]))

        write_df['cum_pnl'] = write_df['pnl'].cumsum()

        write_df['entry_price'] = write_df['entry_price'].apply(lambda x: round(x, 5))
        write_df['exit_price'] = write_df['exit_price'].apply(lambda x: round(x, 5))
        write_df['cum_pnl'] = write_df['cum_pnl'].apply(lambda x: round(x, 2))

        if do_smart_execution:
            for col in ['entry_price', 'exit_price']:
                strategy_df[col] = strategy_df[col].apply(lambda x: round(x, 5))

            for col in ['exit_value', 'pnl']:
                strategy_df[col] = strategy_df[col].apply(lambda x: round(x, 2))

            for col in ['take_profit_price', 'take_loss_price',
                        'entry_price', 'exit_price']:
                strategy_execution_df[col] = strategy_execution_df[col].apply(lambda x: round(x, 5))

            for col in ['entry_value', 'exit_value', 'pnl']:
                strategy_execution_df[col] = strategy_execution_df[col].apply(lambda x: round(x, 2))


        total_return_rate = write_df.iloc[-1]['cum_pnl']/initial_entry_value
        max_draw_down, start_draw_down, end_draw_down = self.calc_max_drawdown(write_df['cum_pnl'])
        max_draw_down = max_draw_down/initial_entry_value
        jc_sharpe_ratio = total_return_rate / max_draw_down


        if report_performance:

            #self.full_summary_df['Return Rate'] =  str(round(total_return_rate * 100, 0)) + '%'
            #self.full_summary_df['Max Drawdown'] = str(round(max_draw_down * 100, 0)) + '%'
            self.full_summary_df['Return Rate'] = total_return_rate
            self.full_summary_df['Max Drawdown'] = max_draw_down
            self.full_summary_df['JC Sharpe'] = jc_sharpe_ratio
            self.full_summary_df['Critical Ratio'] = (self.full_summary_df['Return Rate'] + self.full_summary_df['Max Drawdown'] - 1)/self.full_summary_df['Max Drawdown']

            for i in range(1, 6):
                self.full_summary_df['P' + str(i) + 'Asset'] = np.power(self.full_summary_df['Critical Ratio'], i)
                self.full_summary_df['P' + str(i) + 'Asset'] = self.full_summary_df['P' + str(i) + 'Asset'].apply(lambda x: round(x, 1))

            self.full_summary_df['Critical Ratio'] = self.full_summary_df['Critical Ratio'].apply(lambda x: round(x, 1))

            self.full_summary_df['Return Rate'] = str(round(total_return_rate, 2))
            self.full_summary_df['Max Drawdown'] = str(round(max_draw_down, 2))
            self.full_summary_df['JC Sharpe'] = str(round(jc_sharpe_ratio, 2))

            print("trade_file: " + str(self.trade_file))
            write_df.to_csv(self.trade_file, index = False)

            print("performance_file: " + str(self.performance_file))
            self.full_summary_df.to_csv(self.performance_file, index = False)

            if do_smart_execution:
                strategy_df.to_csv(self.trade_file[:-len('all_trades.csv')] + 'strategies.csv', index = False)
                strategy_execution_df.to_csv(self.trade_file[:-len('all_trades.csv')] + 'strategy_execution.csv', index = False)


            if not is_production:
                plot_pnl_figure(write_df, self.chart_folder, self.currency, start_draw_down, end_draw_down)



    def calc_max_drawdown(self, x):
        df = pd.DataFrame({'cum_pnl': x})
        df['max_cum_pnl'] = df['cum_pnl'].cummax()
        df['draw_down'] = df['max_cum_pnl'] - df['cum_pnl']

        max_draw_down = df['draw_down'].max()
        end = df['draw_down'].argmax()

        start = which(df['cum_pnl'] == df.iloc[end]['max_cum_pnl'])[0]

        return (max_draw_down, start, end)


    def trade(self, print_ready=True):

        print("Do trading............")

        self.calculate_signals(print_ready)

        print_prefix = "[Currency " + self.currency + "] "
        all_days = pd.Series(self.data_df['date'].unique()).dt.to_pydatetime()

        if not is_production:
            plot_candle_bar_charts(self.currency, self.data_df, all_days, self.long_df, self.short_df,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   is_plot_aux = True,
                                   bar_fig_folder=self.chart_folder, is_plot_simple_chart=True,
                                   use_dynamic_TP = use_dynamic_TP, figure_num = printed_figure_num, plot_day_line = plot_day_line, plot_cross_point = plot_cross_point,
                                   plot_long = True, plot_short = False, remove_plots = True)

            plot_candle_bar_charts(self.currency, self.data_df, all_days, self.long_df, self.short_df,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   is_plot_aux=True,
                                   bar_fig_folder=self.chart_folder, is_plot_simple_chart=True,
                                   use_dynamic_TP=use_dynamic_TP, figure_num=printed_figure_num,
                                   plot_day_line=plot_day_line, plot_cross_point=plot_cross_point,
                                   plot_long=False, plot_short=True, remove_plots = False)


        print("Finish")












