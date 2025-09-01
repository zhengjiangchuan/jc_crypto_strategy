




def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import time
#import talib

import math
import matplotlib.lines as mlines
#import datetime
import pandas as pd
import math
import copy
from functools import reduce
import numpy as np
import sys
import os

from util import *
import gzip
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

#from vegas_strategy_once import profit_loss_ratio
import math

from optparse import OptionParser
import matplotlib.ticker as ticker

import urllib.request

from io import StringIO

from coinbase.rest import RESTClient
from CoinbaseUtil import *
from coinbase.rest import RESTClient
from json import dumps
import uuid

from CurrencySmartExecutionManager import *
from StrategyExecution import *


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
initial_bar_number = 50 if data_source == 1 else 5000   #1000


initial_bar_number_5min = 5000  #3000

until_date_5min = None
#until_date = "2024-09-20"

is_production = False

plot_rsi = True

print_execution_details = True

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

#until_date_5min = "2025-05-26"
#until_date_5min = "2025-06-11"
#until_date_5min = "2025-06-27"

#until_date_5min = "2024-09-23"
#until_date_5min = "2024-10-09"
#until_date_5min = "2024-10-25"
#until_date_5min = "2024-11-10"
#until_date_5min = "2024-11-26"
#until_date_5min = "2024-12-12"
#until_date_5min = "2024-12-28"
#until_date_5min = "2025-01-13"
#until_date_5min = "2025-01-29"
#until_date_5min = "2025-02-14"
#until_date_5min = "2025-03-02"
#until_date_5min = "2025-03-18"
#until_date_5min = "2025-04-03"
#until_date_5min = "2025-04-19"
#until_date_5min = "2025-05-05"
#until_date_5min = "2025-05-21"
#until_date_5min = "2025-06-06"
#until_date_5min = "2025-06-22"


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

printed_figure_num = 1

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

initial_entry_value = 50.0
default_leverage = 10

enable_short_macd_signal = False




do_message_printing = True
do_reentry = False

use_global = False

global_use_slow_macd = False
global_use_guppy_filter = True

global_use_guppy_filter_for_exit = True
global_guppy_force_out = True

global_use_rsi_to_exit = False


global_do_stop_loss = False
global_reentry_after_stop_loss = False

global_also_filter_too_late = False
global_use_guppy_condition = False

####################################

do_smart_execution = False #False
use_5min_in_smart_execution = False #False
use_extra_execution = False

is_real_time_trading = True
is_real_time_trading_5min = False #False

only_download_data = False #False



read_5min_data = False #False

#use_coinbase_data_source = False


print_to_console = True
#macd_gradient = 'macd2_gradient' if use_slow_macd else 'macd_gradient'

production_running = True #True
do_real_money_trading = True #True


if not is_real_time_trading:
    do_real_money_trading = False

if do_real_money_trading:
    read_5min_data = False
    use_5min_in_smart_execution = False
    is_real_time_trading_5min = False

if not do_real_money_trading:
    production_running = False



def set_smart_execution(smart_execution):
    global do_smart_execution
    do_smart_execution = smart_execution

def get_smart_execution():
    global do_smart_execution
    return do_smart_execution

def set_is_production(production):
    global is_production
    is_production = production


# if do_smart_execution:
#
#     class StrategyExecution:
#
#         def __init__(self, side, leverage, take_profit_pct, take_loss_pct, strategy_id, execution_id, strategy_entry_time, strategy_entry_price,
#                       execution_entry_time, execution_entry_price, strategy_entry_value, execution_entry_value, prod_size = 0):
#
#             self.active = True
#             self.side = side #1 means long  -1 means short
#             self.leverage = leverage
#             self.take_profit_pct = take_profit_pct
#             self.take_loss_pct = take_loss_pct
#             self.strategy_id = strategy_id
#             self.execution_id = execution_id
#             self.strategy_entry_time = strategy_entry_time
#             self.strategy_entry_price = strategy_entry_price
#             self.execution_entry_time = execution_entry_time
#             self.execution_entry_price = execution_entry_price
#             self.strategy_entry_value = strategy_entry_value #This is actual notioanl value (margin value, not leveraged)
#             self.execution_entry_value = execution_entry_value
#
#             self.execution_exit_time = None
#             self.execution_exit_price = -1
#             self.execution_exit_value = -1
#
#             self.prod_size = prod_size  #This is leveraged size (enlarged size)
#
#             self.pnl_rate = 0
#             self.pnl = 0
#
#             self.prod_strategy_entry_price = 0
#             self.prod_execution_entry_price = 0
#             self.prod_strategy_entry_value = 0
#             self.prod_execution_entry_value = 0
#
#             self.prod_execution_exit_price = -1
#             self.prod_execution_exit_value = -1
#
#             self.prod_pnl_rate = 0
#             self.prod_pnl = 0
#
#
#             self.initialize()
#
#
#
#
#
#         def initialize(self):
#
#             self.pnl_rate = 0
#             self.pnl = 0
#
#             self.take_profit_price = self.execution_entry_price * (1 + self.side * self.take_profit_pct)
#             self.take_loss_price = self.execution_entry_price * (1 - self.side * self.take_loss_pct)
#
#
#         def set_prod_strategy_entry_price(self, prod_entry_price):
#
#             self.prod_strategy_entry_price = prod_entry_price
#             self.prod_execution_entry_price = prod_entry_price
#
#             self.prod_strategy_entry_value = prod_entry_price * self.prod_size / default_leverage
#             self.prod_execution_entry_value = self.prod_strategy_entry_value
#
#
#         def exit_execution(self, execution_exit_time, execution_exit_price, is_signal_exit, is_extra_execution):
#
#             return_rate = self.side * (execution_exit_price - self.execution_entry_price)/self.execution_entry_price
#
#             self.pnl_rate = return_rate * self.leverage
#             self.pnl = self.execution_entry_value * self.pnl_rate
#
#             self.execution_exit_price = execution_exit_price
#             self.execution_exit_value = self.execution_entry_value + self.pnl
#             self.execution_exit_time = execution_exit_time
#
#             self.active = (not is_signal_exit) and self.pnl > 0 and (not is_extra_execution)
#
#
#         def exit_execution_prod(self, prod_execution_exit_price):
#
#             prod_return_rate = self.side * (prod_execution_exit_price - self.prod_execution_entry_price)/self.prod_execution_entry_price
#
#             self.prod_pnl_rate = prod_return_rate * self.leverage
#             self.prod_pnl = self.prod_execution_entry_value * self.pnl_rate
#
#             self.prod_execution_exit_price = prod_execution_exit_price
#             self.prod_execution_exit_value =self.prod_execution_entry_value + self.prod_pnl
#
#
#
#         def calc_increased_size_when_take_profit(self):
#
#             return self.prod_size * self.take_profit_pct * self.leverage
#
#
#         def update_to_next_execution(self, entry_time, increased_size):
#
#             self.execution_id = self.execution_id + 1
#             self.execution_entry_time = entry_time
#             self.execution_entry_price = self.execution_exit_price
#             self.execution_entry_value = self.execution_exit_value
#             self.prod_size = self.prod_size + increased_size
#
#             self.prod_execution_entry_price = self.prod_execution_exit_price
#             self.prod_execution_entry_value = self.prod_execution_exit_value
#
#             self.initialize()





class CurrencyTrader(threading.Thread):

    def __init__(self, condition, currency, lot_size, exchange_rate, coefficient, actual_maxdrawdown, optimal_gradient_num,
                 data_folder, chart_folder, simple_chart_folder, log_file, data_file, trade_file, trade_prod_file, delay_cost_file, performance_file, usdfx, email_message_file, is_notify, data_file_5min = None,
                 decimal = 5, reverse_strategy = False,
                 wakeup = 1, coinbase_client: Optional[RESTClient] = None, currency_coinbase = None, coinbase_portfolio_id = -1, crypto_last_price = 0,
                 use_slow_macd = True, use_guppy_filter = False, use_guppy_filter_for_exit = False, guppy_force_out = False, use_rsi_to_exit = False, do_stop_loss = False, reentry_after_stop_loss = False, also_filter_too_late = False,
                 use_guppy_condition = False, init_entry_value = 0, coinbase_decimal = 0, price_decimal = 0, check_data = False, over_bought_logic = False, adjust_decimal = 1, is_alternative = False,
                 smart_executor_manager: CurrencySmartExecutionManager = None):
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
        self.trade_prod_file = trade_prod_file
        self.delay_cost_file = delay_cost_file
        self.performance_file = performance_file
        self.usdfx = usdfx

        self.email_message_fd = open(email_message_file, 'w')
        self.email_message_caches = []

        self.decimal = decimal
        # print("decimal = " + str(self.decimal))
        # temp_entry_price = 22.1593
        # print(" units at entry price " + str(round(temp_entry_price, self.decimal)))

        self.reverse_strategy = reverse_strategy

        self.use_slow_macd = global_use_slow_macd if use_global else use_slow_macd


        self.use_guppy_filter = global_use_guppy_filter if use_global else use_guppy_filter
        self.use_guppy_filter_for_exit = global_use_guppy_filter_for_exit if use_global else use_guppy_filter_for_exit
        self.guppy_force_out = global_guppy_force_out if use_global else guppy_force_out
        self.use_rsi_to_exit = global_use_rsi_to_exit if use_global else use_rsi_to_exit
        self.do_stop_loss = global_do_stop_loss if use_global else do_stop_loss
        self.reentry_after_stop_loss = global_reentry_after_stop_loss if use_global else reentry_after_stop_loss
        self.also_filter_too_late = global_also_filter_too_late if use_global else also_filter_too_late
        self.use_guppy_condition = global_use_guppy_condition if use_global else use_guppy_condition
        self.init_entry_value = initial_entry_value if use_global else init_entry_value
        self.coinbase_decimal = coinbase_decimal
        self.price_decimal = price_decimal
        self.check_data = check_data
        self.over_bought_logic = over_bought_logic
        self.adjust_decimal = adjust_decimal

        #if do_smart_execution and use_extra_execution:
        #    self.init_entry_value = self.init_entry_value/2.0

        print("currency " + self.currency + " initial entry value = " + str(self.init_entry_value))
        print("guppy_force_out = " + str(self.guppy_force_out))

        print("coinbase_decimal = " + str(coinbase_decimal))

        # self.log_msg("use_slow_macd = " + str(self.use_slow_macd))
        # self.log_msg("use_guppy_filter = " + str(self.use_guppy_filter))
        # self.log_msg("do_stop_loss = " + str(self.do_stop_loss))
        # self.log_msg("reentry_after_stop_loss = " + str(self.reentry_after_stop_loss))
        # self.log_msg("also_filter_too_late = " + str(self.also_filter_too_late))
        # self.log_msg("use_guppy_condition = " + str(self.use_guppy_condition))

        self.macd_gradient = 'macd2_gradient' if self.use_slow_macd else 'macd_gradient'
        #self.log_msg("macd_gradient = " + str(self.macd_gradient))

        if self.use_guppy_condition:
            self.reverse_strategy = True

        self.is_notify = is_notify

        self.long_df = None
        self.short_df = None

        self.long_strategy_df = None
        self.short_strategy_df = None

        self.long_strategy_execution_df = None
        self.short_strategy_execution_df = None

        self.long_existing_df = None
        self.short_existing_df = None

        if production_running:
            if os.path.exists(self.trade_prod_file):
                existing_trade_df = pd.read_csv(self.trade_prod_file)
                for col in ['entry_time', 'exit_time']:
                    existing_trade_df[col] = existing_trade_df[col].apply(lambda x: preprocess_time(x))

                #print("existing_trade_df:")
                #print(existing_trade_df.iloc[0:10])

                self.long_existing_df = existing_trade_df[existing_trade_df['side'] == 'long']
                self.short_existing_df = existing_trade_df[existing_trade_df['side'] == 'short']

                # print("long_existing_df:")
                # print(self.long_existing_df.iloc[-3:])

                #print("existing_long_trade_df:")
                #print(existing_trade_df.iloc[0:10])

                self.long_existing_df.reset_index(inplace = True)
                self.long_existing_df = self.long_existing_df.drop(columns = ['index'])

                self.short_existing_df.reset_index(inplace=True)
                self.short_existing_df = self.short_existing_df.drop(columns=['index'])
            else:
                self.long_existing_df = None
                self.short_existing_df = None


        self.delay_cost_df = None
        if os.path.exists(self.delay_cost_file):
            self.delay_cost_df = pd.read_csv(self.delay_cost_file)

        self.delay_cost_data = []


        # self.use_relaxed_vegas_support = True
        # self.is_require_m12_strictly_above_vegas = False
        # self.remove_c12 = True

        #self.currency_file = os.path.join(data_folder, currency + "100.csv")

        self.log_fd = open(self.log_file, 'a')

        self.print_to_console = True

        self.current_position = 0
        self.current_real_position = 0


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

        self.wakeup = wakeup

        self.coinbase_client = coinbase_client

        self.currency_coinbase = currency_coinbase
        self.coinbase_portfolio_id = coinbase_portfolio_id

        self.crypto_last_price = crypto_last_price

        self.temporary_long = False
        self.temporary_short = False

        self.temporary_close_long = False
        self.temporary_close_short = False

        self.temporary_delta_position = 0

        self.long_order_id = None
        self.long_execution_order_id = None
        self.long_attempt_size = -1
        self.long_order_fill_price = -1
        self.long_order_fill_size = 0

        self.short_order_id = None
        self.short_execution_order_id = None
        self.short_attempt_size = -1
        self.short_order_fill_price = -1
        self.short_order_fill_size = 0

        self.close_long_order_id = None
        self.close_long_execution_order_id = None
        self.close_long_attempt_size = -1
        self.close_long_order_fill_price = -1
        self.close_long_order_fill_size = 0


        self.close_short_order_id = None
        self.close_short_execution_order_id = None
        self.close_short_attempt_size = -1
        self.close_short_order_fill_price = -1
        self.close_short_order_fill_size = 0


        print(f"Checking here do_smart_execution = {do_smart_execution}")
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

            self.log_msg("Leverage is:")
            self.log_msg(self.leverage)

            if do_real_money_trading:
                self.average_leverage = self.leverage.sum()/len(self.leverage)
                self.distribution = self.leverage/self.leverage.sum()

                self.log_msg("average_leverage = " + str(self.average_leverage))
                self.log_msg("prod size distribution = " + str(self.distribution))


            self.take_profit_pct = self.profit_rates / self.leverage
            self.take_loss_pct = self.loss_rates / self.leverage

            self.each_strategy_entry_value = self.init_entry_value / len(self.leverage)
            self.log_msg("each_strategy_entry_value = " + str(self.each_strategy_entry_value))

            if use_extra_execution:
                self.each_strategy_entry_value = self.each_strategy_entry_value / 2.0
                self.log_msg("With extra execution, each_strategy_entry_value = " + str(self.each_strategy_entry_value))
                if do_real_money_trading:
                    self.average_leverage = self.average_leverage / 2.0 + self.optimal_leverage / 2.0

                    self.log_msg("With extra execution, average_leverage = " + str(self.average_leverage))

                    temp_distribution = np.array(list(self.leverage) + [len(self.leverage) * self.optimal_leverage])
                    temp_distribution = temp_distribution/temp_distribution.sum()
                    self.distribution = temp_distribution[0:len(self.leverage)]

                    self.log_msg("With extra execution, prod size distribution on basic strategies is: " + str(self.distribution))

                    self.extra_distribution = temp_distribution[len(self.leverage)]

                    self.log_msg("Prod size distribution on extra strategy is: " + str(self.extra_distribution))

            if do_real_money_trading and do_smart_execution:
                self.smart_executor_manager = smart_executor_manager



        self.is_alternative = is_alternative

        self.log_msg("Initializing...")

    def reset_long(self):
        self.long_order_id = None
        self.long_execution_order_id = None
        self.long_attempt_size = -1
        #self.long_order_fill_price = -1

    def set_long(self, order_id, attempt_size):
        self.long_order_id = order_id
        self.long_attempt_size = attempt_size

    def reset_long_fill(self):
        self.long_order_fill_price = -1
        self.long_order_fill_size = 0

    def set_long_fill(self, fill_price, fill_size):
        self.long_order_fill_price = fill_price
        self.long_order_fill_size = fill_size



    def reset_short(self):
        self.short_order_id = None
        self.short_execution_order_id = None
        self.short_attempt_size = -1
        #self.short_order_fill_price = -1

    def set_short(self, order_id, attempt_size):
        self.short_order_id = order_id
        self.short_attempt_size = attempt_size

    def reset_short_fill(self):
        self.short_order_fill = -1
        self.short_order_fill_size = 0

    def set_short_fill(self, fill_price, fill_size):
        self.short_order_fill_price = fill_price
        self.short_order_fill_size = fill_size




    def reset_close_long(self):
        self.close_long_order_id = None
        self.close_long_execution_order_id = None
        self.close_long_attempt_size = -1
        #self.close_long_order_fill_price = -1

    def set_close_long(self, order_id, attempt_size):
        self.close_long_order_id = order_id
        self.close_long_attempt_size = attempt_size

    def reset_close_long_fill(self):
        self.close_long_order_fill_price = -1
        self.close_long_order_fill_size = 0

    def set_close_long_fill(self, fill_price, fill_size):
        self.close_long_order_fill_price = fill_price
        self.close_long_order_fill_size = fill_size




    def reset_close_short(self):
        self.close_short_order_id = None
        self.close_short_execution_order_id = None
        self.close_short_attempt_size = -1
        #self.close_short_order_fill_price = -1

    def set_close_short(self, order_id, attempt_size):
        self.close_short_order_id = order_id
        self.close_short_attempt_size = attempt_size

    def reset_close_short_fill(self):
        self.close_short_order_fill_price = -1
        self.close_short_order_fill_size = 0

    def set_close_short_fill(self, fill_price, fill_size):
        self.close_short_order_fill_price = fill_price
        self.close_short_order_fill_size = fill_size


    def log_msg(self, msg):

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #current_time = (datetime.now() + timedelta(seconds = 28800)).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(msg, pd.DataFrame):
            print('[' + current_time + ' ' + self.currency + ']  \n' + str(msg), file = self.log_fd)
        else:
            print('[' + current_time + ' ' + self.currency + ']  ' + str(msg), file=self.log_fd)

        self.log_fd.flush()

        if print_to_console:
            if isinstance(msg, pd.DataFrame):
                print('[' + current_time + ' ' + self.currency + ']  \n' + str(msg))
            else:
                print('[' + current_time + ' ' + self.currency + ']  ' + str(msg))


    def feed_data(self, new_data_df, new_data_df_5min = None):

        self.data_df = new_data_df
        self.data_df_5min = new_data_df_5min


        if self.data_df_5min is not None:
            self.log_msg("Round to " + str(self.decimal) + " decimals.....")
            for col in ['open', 'high', 'low', 'close']:
                self.data_df_5min[col] = self.data_df_5min[col].apply(lambda x: round(x, self.decimal))

        if self.data_df_5min is not None and use_5min_in_smart_execution:   #Darren

            self.data_df_5min['location'] = list(range(self.data_df_5min.shape[0]))

            print("set location to data_df_5min")

            print("Before data_df row_num = " + str(self.data_df.shape[0]))
            before_num = self.data_df.shape[0]
            print("Before data_df_5min row_num = " + str(self.data_df_5min.shape[0]))

            self.data_df = pd.merge(self.data_df, self.data_df_5min[['time', 'location']], on = ['time'], how = 'left')

            print("After data_df row_num = " + str(self.data_df.shape[0]))
            after_num = self.data_df.shape[0]
            print("After data_df_5min row_num = " + str(self.data_df_5min.shape[0]))

            if before_num != after_num:
                print("before_num = " + str(before_num) + ", after_num = " + str(after_num))
                print("currency = " + self.currency)
                sys.exit(1)



            #self.data_df_5min.to_csv(self.data_file_5min, index=False)  #Darren

            #return #Darren
            #sys.exit(0)




        if os.path.exists(self.trade_file):

            if True: #self.wakeup:
                trade_df = pd.read_csv(self.trade_file)

                trade_df['entry_time'] = trade_df['entry_time'].apply(lambda x: preprocess_time(x))
                trade_df = trade_df[trade_df['entry_time'] < self.data_df.iloc[-1]['time']]

                self.log_msg("last data time: " + str(self.data_df.iloc[-1]['time']))

                # self.log_msg("trade_df last 2 rows:")
                # self.log_msg(trade_df.iloc[-2:])

                last_trade = trade_df.iloc[-1]

                if last_trade['exit_id'] == -1:

                    #if last_trade['exit_id'] == -1:
                    if last_trade['side'] == 'long':
                        self.current_position = 1
                    else:
                        self.current_position = -1

                    self.current_position *= self.init_entry_value/last_trade['entry_price'] * default_leverage
                    if last_trade['entry_price'] >= 1:
                        self.current_position = round(self.current_position, 3)
                    else:
                        self.current_position = int(round(self.current_position, 0))

                    self.log_msg("current_position = " + str(self.current_position))

            # else:
            #     self.current_position = 0

            if do_real_money_trading:
                self.log_msg("portfolio_id = " + str(self.coinbase_portfolio_id))
                positions = self.coinbase_client.list_perps_positions(portfolio_uuid=self.coinbase_portfolio_id).positions
                self.log_msg(f"positions size = {len(positions)}")
                for position in positions:
                    #print("position symbol = " + position['symbol'])
                    #print("currency_coinbase = " + self.currency_coinbase)
                    if position['symbol'] == self.currency_coinbase:
                        self.current_real_position = float(position['net_size'])
                        #print("net_size = " + str(self.current_real_position))
                        if position['position_side'] not in ['POSITION_SIDE_LONG', 'POSITION_SIDE_SHORT']:
                            self.log_msg("Unknown position side " + position['position_side'])
                            sys.exit(1)

                        #print("position_side = " + position['position_side'])
                        if position['position_side'] == 'POSITION_SIDE_SHORT':
                            #print("Negate it")
                            #print(type(self.current_real_position))
                            self.current_real_position *= -1
                            #print("current_real_position here = ")
                            #print(self.current_real_position)

                        self.log_msg("Query real position = " + str(self.current_real_position))



    def run(self):
        self.log_msg("Running...........")
        self.trade()


    def round_price(self, price):

        if 'JPY' in self.currency:
            return round(price, 3)
        else:
            return round(price, 5)


    def calculate_signals(self, print_ready = True, temporary_decision = False):

        self.data_df['date'] = pd.DatetimeIndex(self.data_df['time']).normalize()
        self.data_df['hour'] = self.data_df['time'].apply(lambda x: x.hour)

        calc_jc_lines(self.data_df, "close", windows)

        #if not is_production:
        calc_bolling_bands(self.data_df, "close", bolling_width)
        calc_macd(self.data_df, "close")
        calc_rsi(self.data_df, "close")





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

        self.data_df['body_length'] = self.data_df['max_price'] - self.data_df['min_price']
        self.data_df['recent_body_length_median'] = self.data_df['body_length'].rolling(5, min_periods=5).median()
        self.data_df['prev_recent_body_length_median'] = self.data_df['recent_body_length_median'].shift(1)

        self.data_df['recent_body_length_mean'] = self.data_df['body_length'].rolling(5, min_periods=5).mean()
        self.data_df['prev_recent_body_length_mean'] = self.data_df['recent_body_length_mean'].shift(1)

        self.data_df['is_big_body'] = (self.data_df['body_length'] - self.data_df['prev_recent_body_length_mean'])/self.data_df['prev_recent_body_length_mean'] > 2


        if self.over_bought_logic:
            self.data_df['over_bought'] = (self.data_df['rsi'] >= 79) & (self.data_df['is_big_body'])
        else:
            self.data_df['over_bought'] = False  # self.data_df['rsi'] >= 80


        self.data_df['over_sold'] = self.data_df['rsi'] <= 20

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



        # self.data_df['up_vegas_converge'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        #                                     (self.data_df['fast_vegas_gradient'] < self.data_df['slow_vegas_gradient'])
        # self.data_df['up_vegas_converge_previous'] = self.data_df['up_vegas_converge'].shift(1)
        # self.data_df['up_vegas_converge_pp'] = self.data_df['up_vegas_converge_previous'].shift(1)
        #
        # self.data_df['down_vegas_converge'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & \
        #                                     (self.data_df['fast_vegas_gradient'] > self.data_df['slow_vegas_gradient'])
        # self.data_df['down_vegas_converge_previous'] = self.data_df['down_vegas_converge'].shift(1)
        # self.data_df['down_vegas_converge_pp'] = self.data_df['down_vegas_converge_previous'].shift(1)
        #
        # ########## Long ############
        #
        # self.data_df['vegas_support_long'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['fast_vegas_up']) & (self.data_df['slow_vegas_up']) & \
        #     (~((self.data_df['up_vegas_converge']) & (self.data_df['up_vegas_converge_previous']) & (self.data_df['up_vegas_converge_pp'])))
        #
        # self.data_df['long_encourage_condition'] = (self.data_df['fast_guppy_cross_up']) & (self.data_df['fastest_guppy_line_up'])  #'fastest_guppy_line_up'
        #
        # ######### Filters for Scenario where Vegas support long ###############
        #
        # self.data_df['long_filter1'] = (self.data_df['down_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_down'])   #adjust by removing
        # self.data_df['long_filter1'] = (self.data_df['long_filter1']) | (self.data_df['previous_down_guppy_line_num'] >= 3)  #USDCAD Stuff
        # self.data_df['long_filter1'] = (self.data_df['long_filter1']) & (~self.data_df['long_encourage_condition'])
        #
        # self.data_df['long_filter2'] = (self.data_df['up_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_down']) & (self.data_df['fast_guppy_cross_down'])
        #
        # self.data_df['long_strong_filter1'] = (self.data_df['guppy_half1_strong_aligned_short'])
        # self.data_df['long_strong_filter2'] = (self.data_df['guppy_half2_aligned_long']) & (self.data_df['fastest_guppy_line_down']) & (self.data_df['fast_guppy_cross_down'])
        #
        #
        # self.data_df['guppy_long_reverse'] = (self.data_df['up_guppy_line_num'] >= 3) & (self.data_df['ma_close30_gradient'] < 0)
        # self.data_df['prev_guppy_long_reverse'] = self.data_df['guppy_long_reverse'].shift(1)
        # self.data_df['prev2_guppy_long_reverse'] = self.data_df['prev_guppy_long_reverse'].shift(1)
        # self.data_df['recent_guppy_long_reverse'] = (self.data_df['guppy_long_reverse']) | (self.data_df['prev_guppy_long_reverse']) | (self.data_df['prev2_guppy_long_reverse'])
        # #self.data_df['recent_guppy_long_reverse'] = (self.data_df['guppy_long_reverse']) & (self.data_df['prev_guppy_long_reverse']) & (self.data_df['prev2_guppy_long_reverse'])
        #
        #
        # self.data_df['can_long1'] = self.data_df['vegas_support_long'] #&\
        #                             #(~self.data_df['guppy_half1_strong_aligned_short']) & (~self.data_df['prev_guppy_half1_strong_aligned_short']) & (~self.data_df['prev2_guppy_half1_strong_aligned_short']) #& (~self.data_df['long_filter1']) & (~self.data_df['long_filter2'])  #Modify
        #
        #
        # ######## Conditions for Scenario where Vegas does not support long ############### #second condition is EURUSD stuff
        #
        # self.data_df['long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) |\
        #                                  ((self.data_df['guppy_half2_strong_aligned_long'])) |\
        #                                  (self.data_df['guppy_all_aligned_long']) | (self.data_df['long_encourage_condition'])
        # self.data_df['long_condition'] = self.data_df['long_condition'] & (~self.data_df['fastest_guppy_line_lasting_down'])
        # self.data_df['long_condition'] = self.data_df['long_condition'] & (self.data_df['guppy_first_half_min'] > self.data_df['guppy_second_half_max'])
        #
        # #self.data_df['long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) #Adjust2
        # self.data_df['can_long2'] = (~self.data_df['vegas_support_long']) & self.data_df['long_condition']
        #
        # # Old One
        # self.data_df['final_long_filter1'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate < -vegas_threshold) & (self.data_df['vegas_phase_duration'] < 96) & (self.data_df['prev_vegas_phase_entire_duration'] < 96) &\
        #                                       ( ((self.data_df['fast_vegas_down']) & (self.data_df['previous_fast_vegas_down'])) |\
        #                                      ((self.data_df['slow_vegas_down']) & (self.data_df['previous_slow_vegas_down'])) |\
        #                                      ((self.data_df['previous_fast_vegas_down']) & (self.data_df['pp_fast_vegas_down'])) |\
        #                                      ((self.data_df['previous_slow_vegas_down']) & (self.data_df['pp_slow_vegas_down']))
        #                                      )
        #
        #
        # # New Change
        # self.data_df['final_long_filter2'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate < -vegas_threshold) & (self.data_df['vegas_phase_duration'] >= 96)
        # self.data_df['long_filter_exempt'] = self.data_df['fast_vegas_up'] & self.data_df['previous_fast_vegas_up'] & (self.data_df['vegas_phase_duration'] < 8*24) &\
        #                                      (self.data_df['vegas_distance_gradient'] < 0) & (self.data_df['prev_vegas_distance_gradient'] < 0) & self.data_df['guppy_all_above_vegas'] & self.data_df['guppy_all_strong_aligned_long']
        # self.data_df['final_long_filter2'] = self.data_df['final_long_filter2'] & (~self.data_df['long_filter_exempt'])
        #
        # self.data_df['final_long_filter'] = self.data_df['final_long_filter1'] | self.data_df['final_long_filter2']
        #



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




        # self.data_df['can_long'] = True #(self.data_df['can_long1']) | (self.data_df['can_long2'])
        # #self.data_df['can_long'] = (self.data_df['vegas_support_long']) & (self.data_df['long_condition'])  #strong adjust
        #
        # self.data_df['can_long'] = (self.data_df['can_long']) & (~self.data_df['final_long_filter']) #USDCAD stuff
        #
        # ##############
        # self.data_df['final_long_condition'] = (self.data_df['guppy_half1_strong_aligned_long']) |\
        #                                  ((self.data_df['guppy_half2_strong_aligned_long'])) |\
        #                                  (self.data_df['guppy_all_aligned_long'])
        # #self.data_df['final_long_condition'] = self.data_df['final_long_condition'] & (~self.data_df['fastest_guppy_line_lasting_down'])
        # self.data_df['final_long_condition1'] = self.data_df['final_long_condition'] & (self.data_df['guppy_first_half_min'] > self.data_df['guppy_second_half_max'])
        #
        #
        # # self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        # #                                         (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        # #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        # #                                         (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_strong_aligned_short'])
        #
        # #old one
        # self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        #                                         (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        #                                         (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_aligned_short']) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])
        #
        #
        # # self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        # #                                         (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        # #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        # #                                         (self.data_df['vegas_phase_duration'] > 48) & (self.data_df['guppy_lines_down_num'] < 3) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])
        #
        #
        #
        #
        # # self.data_df['final_long_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        # #                                         (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        # #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        # #                                         (~self.data_df['guppy_all_aligned_short']) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])
        #
        #
        #
        #
        # # self.data_df['final_long_condition2'] = (self.data_df['middle'] > self.data_df['upper_vegas']) &\
        # #                                         (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) &\
        # #                                         (~self.data_df['guppy_all_aligned_short']) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])
        #
        # #Change Change
        # self.data_df['must_reject_long'] = False #(self.data_df['final_long_condition']) & (self.data_df['guppy_first_half_min'] <= self.data_df['guppy_second_half_max'])
        #
        # #self.data_df['must_reject_long'] = (self.data_df['final_long_condition'] & (~self.data_df['final_long_condition2'])) & (self.data_df['guppy_first_half_min'] <= self.data_df['guppy_second_half_max'])
        #
        # self.data_df['must_reject_long2'] = (~self.data_df['vegas_support_long']) & (self.data_df['ma_close30_gradient'] < 0) & (self.data_df['ma_close35_gradient'] < 0) & (self.data_df['ma_close30'] < self.data_df['ma_close35'])
        # #self.data_df['must_reject_long2'] = self.data_df['must_reject_long2'] & (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)
        #
        # self.data_df['must_reject_long2'] = self.data_df['must_reject_long2'] &\
        #                                     (((self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)) | (self.data_df['fast_vegas'] < self.data_df['slow_vegas']))
        #
        # self.data_df['must_reject_long3'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['fast_vegas_down']) & (self.data_df['slow_vegas_down'])
        #
        # self.data_df['must_reject_long4'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['bar_up_phase_duration'] >= 24*5) & (self.data_df['guppy_lines_down_num'] >= 3)
        #
        # self.data_df['can_long'] = (self.data_df['can_long']) & (self.data_df['final_long_condition1']  | self.data_df['final_long_condition2'])
        # self.data_df['can_long'] = self.data_df['can_long'] & (~self.data_df['must_reject_long']) & (~self.data_df['must_reject_long2'])# & (~self.data_df['must_reject_long3'])
        # #self.data_df['can_long'] = self.data_df['can_long'] & (~self.data_df['must_reject_long4'])
        # ###############
        #
        #
        # #self.data_df['can_long'] = self.data_df['can_long'] & (~self.data_df['recent_guppy_long_reverse'])
        #
        #
        # ######### Short ############
        #
        # self.data_df['vegas_support_short'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['fast_vegas_down']) & (self.data_df['slow_vegas_down']) & \
        #     (~((self.data_df['down_vegas_converge']) & (self.data_df['down_vegas_converge_previous'])  & (self.data_df['down_vegas_converge_pp'])))
        #
        # self.data_df['short_encourage_condition'] = (self.data_df['fast_guppy_cross_down']) & (self.data_df['fastest_guppy_line_down']) #fastest_guppy_line_down
        #
        # ######### Filters for Scenario where Vegas support short ###############
        #
        # self.data_df['short_filter1'] = (self.data_df['up_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_up'])  #adjust by removing
        # self.data_df['short_filter1'] = (self.data_df['short_filter1']) | (self.data_df['previous_up_guppy_line_num'] >= 3)  #USDCAD Stuff
        # self.data_df['short_filter1'] = (self.data_df['short_filter1']) & (~self.data_df['short_encourage_condition'])
        #
        # self.data_df['short_filter2'] = (self.data_df['down_guppy_line_num'] >= 3) & (self.data_df['fastest_guppy_line_up']) & (self.data_df['fast_guppy_cross_up'])
        #
        # self.data_df['short_strong_filter1'] = (self.data_df['guppy_half1_strong_aligned_long'])
        # self.data_df['short_strong_filter2'] = (self.data_df['guppy_half2_aligned_short']) & (self.data_df['fastest_guppy_line_up']) & (self.data_df['fast_guppy_cross_up'])
        #
        # self.data_df['guppy_short_reverse'] = (self.data_df['down_guppy_line_num'] >= 3) & (self.data_df['ma_close30_gradient'] > 0)
        # self.data_df['prev_guppy_short_reverse'] = self.data_df['guppy_short_reverse'].shift(1)
        # self.data_df['prev2_guppy_short_reverse'] = self.data_df['prev_guppy_short_reverse'].shift(1)
        # self.data_df['recent_guppy_short_reverse'] = (self.data_df['guppy_short_reverse']) | (self.data_df['prev_guppy_short_reverse']) | (self.data_df['prev2_guppy_short_reverse'])
        # #self.data_df['recent_guppy_short_reverse'] = (self.data_df['guppy_short_reverse']) & (self.data_df['prev_guppy_short_reverse']) & (self.data_df['prev2_guppy_short_reverse'])
        #
        #
        # self.data_df['can_short1'] = self.data_df['vegas_support_short'] #&\
        #                              #(~self.data_df['guppy_half1_strong_aligned_long']) & (~self.data_df['prev_guppy_half1_strong_aligned_long']) & (~self.data_df['prev2_guppy_half1_strong_aligned_long']) #& (~self.data_df['short_filter1']) & (~self.data_df['short_filter2'])  #Modify
        #
        # ######## Conditions for Scenario where Vegas does not support short ###############  #second condition is EURUSD stuff
        #
        # self.data_df['short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) |\
        #                                   ((self.data_df['guppy_half2_strong_aligned_short'])) |\
        #                                   (self.data_df['guppy_all_aligned_short']) | (self.data_df['short_encourage_condition'])
        #
        # self.data_df['short_condition'] = self.data_df['short_condition'] & (~self.data_df['fastest_guppy_line_lasting_up'])
        # self.data_df['short_condition'] = self.data_df['short_condition'] & (self.data_df['guppy_first_half_max'] < self.data_df['guppy_second_half_min'])
        #
        # #self.data_df['short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) #Adjust2
        # self.data_df['can_short2'] = (~self.data_df['vegas_support_short']) & self.data_df['short_condition']
        #
        # # Old One
        # self.data_df['final_short_filter1'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate > vegas_threshold) & (self.data_df['vegas_phase_duration'] < 96) & (self.data_df['prev_vegas_phase_entire_duration'] < 96) &\
        #                                       ( ((self.data_df['fast_vegas_up']) & (self.data_df['previous_fast_vegas_up'])) |\
        #                                      ((self.data_df['slow_vegas_up']) & (self.data_df['previous_slow_vegas_up'])) |\
        #                                      ((self.data_df['previous_fast_vegas_up']) & (self.data_df['pp_fast_vegas_up'])) |\
        #                                      ((self.data_df['previous_slow_vegas_up']) & (self.data_df['pp_slow_vegas_up']))
        #                                      )
        #
        #
        # # New Change
        # self.data_df['final_short_filter2'] = ((self.data_df['fast_vegas'] - self.data_df['slow_vegas'])*self.lot_size*self.exchange_rate > vegas_threshold) & (self.data_df['vegas_phase_duration'] >= 96)
        # self.data_df['short_filter_exempt'] = self.data_df['fast_vegas_down'] & self.data_df['previous_fast_vegas_down'] & (self.data_df['vegas_phase_duration'] < 8*24) &\
        #                                      (self.data_df['vegas_distance_gradient'] < 0) & (self.data_df['prev_vegas_distance_gradient'] < 0) & self.data_df['guppy_all_below_vegas'] & self.data_df['guppy_all_strong_aligned_short']
        # self.data_df['final_short_filter2'] = self.data_df['final_short_filter2'] & (~self.data_df['short_filter_exempt'])
        #
        # self.data_df['final_short_filter'] = self.data_df['final_short_filter1'] | self.data_df['final_short_filter2']




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




        # self.data_df['can_short'] = True #(self.data_df['can_short1']) | (self.data_df['can_short2'])
        # #self.data_df['can_short'] = (self.data_df['vegas_support_short']) & (self.data_df['short_condition']) #strong adjust
        #
        # self.data_df['can_short'] = (self.data_df['can_short']) & (~self.data_df['final_short_filter']) #USDCAD stuff
        #
        # #############
        # self.data_df['final_short_condition'] = (self.data_df['guppy_half1_strong_aligned_short']) |\
        #                                   ((self.data_df['guppy_half2_strong_aligned_short'])) |\
        #                                   (self.data_df['guppy_all_aligned_short'])
        # #self.data_df['final_short_condition'] = self.data_df['final_short_condition'] & (~self.data_df['fastest_guppy_line_lasting_up'])
        # self.data_df['final_short_condition1'] = self.data_df['final_short_condition'] & (self.data_df['guppy_first_half_max'] < self.data_df['guppy_second_half_min'])
        #
        # # self.data_df['final_short_condition2'] = (self.data_df['bar_down_phase_duration'] > 48) &\
        # #                                          (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        # #                                          (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        # #                                          (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_strong_aligned_long'])
        #
        # #Old
        # self.data_df['final_short_condition2'] = (self.data_df['bar_down_phase_duration'] > 48) &\
        #                                          (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        #                                          (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        #                                          (self.data_df['vegas_phase_duration'] > 48) & (~self.data_df['guppy_all_aligned_long']) #& (self.data_df['middle'] > self.data_df['guppy_min'])#& (~self.data_df['guppy_half1_strong_aligned_long'])
        #
        # # self.data_df['final_short_condition2'] = (self.data_df['bar_up_phase_duration'] > 48) &\
        # #                                         (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        # #                                         (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        # #                                         (self.data_df['vegas_phase_duration'] > 48) & (self.data_df['guppy_lines_up_num'] < 3) #& (self.data_df['middle'] < self.data_df['guppy_max'])#& (~self.data_df['guppy_half1_strong_aligned_short'])
        #
        #
        # # self.data_df['final_short_condition2'] = (self.data_df['bar_down_phase_duration'] > 48) &\
        # #                                          (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        # #                                          (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        # #                                          (~self.data_df['guppy_all_aligned_long']) #& (self.data_df['middle'] > self.data_df['guppy_min'])#& (~self.data_df['guppy_half1_strong_aligned_long'])
        #
        #
        #
        # # self.data_df['final_short_condition2'] = (self.data_df['middle'] < self.data_df['lower_vegas']) &\
        # #                                          (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) &\
        # #                                          (~self.data_df['guppy_all_aligned_long']) #& (self.data_df['middle'] > self.data_df['guppy_min'])#& (~self.data_df['guppy_half1_strong_aligned_long'])
        #
        # #Change Change
        # self.data_df['must_reject_short'] = False #(self.data_df['final_short_condition']) & (self.data_df['guppy_first_half_max'] >= self.data_df['guppy_second_half_min'])
        #
        #
        # #self.data_df['must_reject_short'] = (self.data_df['final_short_condition'] & (~self.data_df['final_short_condition2'])) & (self.data_df['guppy_first_half_max'] >= self.data_df['guppy_second_half_min'])
        #
        # self.data_df['must_reject_short2'] = (~self.data_df['vegas_support_short']) & (self.data_df['ma_close30_gradient'] > 0) & (self.data_df['ma_close35_gradient'] > 0) & (self.data_df['ma_close30'] > self.data_df['ma_close35'])
        # #self.data_df['must_reject_short2'] = self.data_df['must_reject_short2'] & (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)
        #
        # self.data_df['must_reject_short2'] = self.data_df['must_reject_short2'] &\
        #                                     (((self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['vegas_phase_duration'] >= 24*8)) | (self.data_df['fast_vegas'] > self.data_df['slow_vegas']))
        #
        # self.data_df['must_reject_short3'] = (self.data_df['fast_vegas'] > self.data_df['slow_vegas']) & (self.data_df['fast_vegas_up']) & (self.data_df['slow_vegas_up'])
        #
        # self.data_df['must_reject_short4'] = (self.data_df['fast_vegas'] < self.data_df['slow_vegas']) & (self.data_df['bar_up_phase_duration'] >= 24*5) & (self.data_df['guppy_lines_up_num'] >= 3)
        #
        #
        #
        # self.data_df['can_short'] = (self.data_df['can_short']) & (self.data_df['final_short_condition1'] | self.data_df['final_short_condition2'])
        # self.data_df['can_short'] = self.data_df['can_short'] & (~self.data_df['must_reject_short']) & (~self.data_df['must_reject_short2'])# & (~self.data_df['must_reject_short3'])
        # #self.data_df['can_short'] = self.data_df['can_short'] & (~self.data_df['must_reject_short4'])
        #
        # ############


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







        # self.data_df['m12_above_upper_vegas'] = self.data_df['ma_close12'] > self.data_df['upper_vegas']
        # self.data_df['m12_below_lower_vegas'] = self.data_df['ma_close12'] < self.data_df['lower_vegas']
        #
        # self.data_df['m12_above_lower_vegas'] = self.data_df['ma_close12'] > self.data_df['lower_vegas']
        # self.data_df['m12_below_upper_vegas'] = self.data_df['ma_close12'] < self.data_df['upper_vegas']
        #
        #
        # self.data_df['low_price_to_upper_vegas'] = self.data_df['low'] - self.data_df['upper_vegas']
        # self.data_df['middle_price_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['max_price']  #middle_price
        #
        # self.data_df['high_price_to_lower_vegas'] = self.data_df['lower_vegas'] - self.data_df['high']
        # self.data_df['middle_price_to_upper_vegas'] = self.data_df['min_price'] - self.data_df['upper_vegas']  #middle_price
        #
        #
        # self.data_df['recent_min_low_price_to_upper_vegas'] = self.data_df['low_price_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
        #                                                                                                     min_periods = vegas_reverse_look_back_window).min()
        # self.data_df['recent_max_middle_price_to_lower_vegas'] = self.data_df['middle_price_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
        #                                                                                                     min_periods = vegas_reverse_look_back_window).max()
        #
        #
        # self.data_df['recent_min_high_price_to_lower_vegas'] = self.data_df['high_price_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
        #                                                                                                     min_periods = vegas_reverse_look_back_window).min()
        # self.data_df['recent_max_middle_price_to_upper_vegas'] = self.data_df['middle_price_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
        #                                                                                                     min_periods = vegas_reverse_look_back_window).max()
        #
        # self.data_df['m12_to_lower_vegas'] = self.data_df['ma_close12'] - self.data_df['lower_vegas']
        # self.data_df['m12_to_upper_vegas'] = self.data_df['upper_vegas'] - self.data_df['ma_close12']
        #
        # self.data_df['recent_min_m12_to_lower_vegas'] = self.data_df['m12_to_lower_vegas'].rolling(vegas_reverse_look_back_window,
        #                                                                                            min_periods = vegas_reverse_look_back_window).min()
        # self.data_df['recent_min_m12_to_upper_vegas'] = self.data_df['m12_to_upper_vegas'].rolling(vegas_reverse_look_back_window,
        #                                                                                            min_periods = vegas_reverse_look_back_window).min()


        ################## Added features #########################

        bar_lookback_num = 5

        # self.data_df['positive_close'] = np.where(self.data_df['is_positive'], self.data_df['close'], np.nan)
        # self.data_df['positive_close'] = self.data_df['positive_close'].fillna(method = 'bfill').fillna(0)
        # self.data_df['positive_close_diff'] = self.data_df['positive_close'].diff()
        #
        # self.data_df['negative_close'] = np.where(self.data_df['is_negative'], self.data_df['close'], np.nan)
        # self.data_df['negative_close'] = self.data_df['negative_close'].fillna(method = 'bfill').fillna(0)
        # self.data_df['negative_close_diff'] = self.data_df['negative_close'].diff()
        #
        # self.data_df['positive_close_increase'] = np.where(self.data_df['positive_close_diff'] >= 0, 1, 0)
        # self.data_df['positive_close_decrease'] = np.where(self.data_df['positive_close_diff'] < 0, 1, 0)
        #
        # self.data_df['negative_close_decrease'] = np.where(self.data_df['negative_close_diff'] <= 0, 1, 0)
        # self.data_df['negative_close_increase'] = np.where(self.data_df['negative_close_diff'] > 0, 1, 0)
        #
        # self.data_df['recent_positive_close_decrease_num'] = self.data_df['positive_close_decrease'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()
        # self.data_df['recent_negative_close_increase_num'] = self.data_df['negative_close_increase'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()
        #
        # self.data_df['prev_recent_positive_close_decrease_num'] = self.data_df['recent_positive_close_decrease_num'].shift(1)  ###
        # self.data_df['prev_recent_negative_close_increase_num'] = self.data_df['recent_negative_close_increase_num'].shift(1)
        #
        #
        #
        # self.data_df['positive_open'] = np.where(self.data_df['positive'], self.data_df['open'], np.nan)
        # self.data_df['positive_open'] = self.data_df['positive_open'].fillna(method = 'bfill').fillna(0)
        # self.data_df['positive_open_diff'] = self.data_df['positive_open'].diff()
        #
        # self.data_df['negative_open'] = np.where(self.data_df['negative'], self.data_df['open'], np.nan)
        # self.data_df['negative_open'] = self.data_df['negative_open'].fillna(method = 'bfill').fillna(0)
        # self.data_df['negative_open_diff'] = self.data_df['negative_open'].diff()
        #
        # self.data_df['positive_open_increase'] = np.where(self.data_df['positive_open_diff'] >= 0, 1, 0)
        # self.data_df['positive_open_decrease'] = np.where(self.data_df['positive_open_diff'] < 0, 1, 0)
        #
        # self.data_df['negative_open_decrease'] = np.where(self.data_df['negative_open_diff'] <= 0, 1, 0)
        # self.data_df['negative_open_increase'] = np.where(self.data_df['negative_open_diff'] > 0, 1, 0)
        #
        # self.data_df['recent_positive_open_decrease_num'] = self.data_df['positive_open_decrease'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()
        # self.data_df['recent_negative_open_increase_num'] = self.data_df['negative_open_increase'].rolling(bar_lookback_num-1, min_periods = bar_lookback_num-1).sum()
        #
        # self.data_df['prev_recent_positive_open_decrease_num'] = self.data_df['recent_positive_open_decrease_num'].shift(1)  ###
        # self.data_df['prev_recent_negative_open_increase_num'] = self.data_df['recent_negative_open_increase_num'].shift(1)
        #
        #
        #
        #
        # self.data_df['recent_positive_bar_num'] = self.data_df['positive'].rolling(bar_lookback_num, min_periods = bar_lookback_num).sum()
        # self.data_df['recent_negative_bar_num'] = self.data_df['negative'].rolling(bar_lookback_num, min_periods = bar_lookback_num).sum()
        #
        # self.data_df['prev_recent_positive_bar_num'] = self.data_df['recent_positive_bar_num'].shift(1)
        # self.data_df['prev_recent_negative_bar_num'] = self.data_df['recent_negative_bar_num'].shift(1)
        #
        #
        # self.data_df['backward_min_price'] = self.data_df['min_price'].shift(bar_lookback_num)
        # self.data_df['backward_max_price'] = self.data_df['max_price'].shift(bar_lookback_num)
        #
        #
        # self.data_df['special_reject_short_cond1'] = self.data_df['prev_recent_positive_bar_num'] >= 3
        # self.data_df['special_reject_short_cond2'] = self.data_df['prev_is_positive'] & (~self.data_df['prev_is_small_body']) & self.data_df['pp_is_positive'] & (~self.data_df['pp_is_small_body'])
        # self.data_df['special_reject_short_cond3'] = (self.data_df['prev_recent_positive_close_decrease_num'] == 0) & (self.data_df['prev_recent_positive_open_decrease_num'] == 0)
        # self.data_df['special_reject_short_cond4'] = self.data_df['is_negative'] & (self.data_df['min_price'] <= self.data_df['backward_min_price'])
        # self.data_df['special_reject_short_cond'] = reduce(lambda left, right: left & right, [self.data_df['special_reject_short_cond' + str(i)] for i in range(1, 5)])
        #
        # self.data_df['special_reject_long_cond1'] = self.data_df['prev_recent_negative_bar_num'] >= 3
        # self.data_df['special_reject_long_cond2'] = self.data_df['prev_is_negative'] & (~self.data_df['prev_is_small_body']) & self.data_df['pp_is_negative'] & (~self.data_df['pp_is_small_body'])
        # self.data_df['special_reject_long_cond3'] = (self.data_df['prev_recent_negative_close_increase_num'] == 0) & (self.data_df['prev_recent_negative_open_increase_num'] == 0)
        # self.data_df['special_reject_long_cond4'] = self.data_df['is_positive'] & (self.data_df['max_price'] >= self.data_df['backward_max_price'])
        # self.data_df['special_reject_long_cond'] = reduce(lambda left, right: left & right, [self.data_df['special_reject_long_cond' + str(i)] for i in range(1, 5)])





        ###########################################################

        ######## Stop Loss Logic #############
        if self.do_stop_loss:
            self.data_df['bar_cross_up_max_guppy'] = (self.data_df['prev_min_price'] <= self.data_df['prev_guppy_max']) & (self.data_df['middle'] > self.data_df['guppy_max'])
            self.data_df['bar_cross_down_min_guppy'] = (self.data_df['prev_max_price'] >= self.data_df['prev_guppy_min']) & (self.data_df['middle'] < self.data_df['guppy_min'])

            self.data_df['bar_cross_guppy_label'] = np.where(
                self.data_df['bar_cross_up_max_guppy'], 0,
                np.where(
                    self.data_df['bar_cross_down_min_guppy'], 1, np.nan
                )
            )

            self.data_df['bar_cross_guppy_label'] = self.data_df['bar_cross_guppy_label'].fillna(method='ffill').fillna(-1)
            self.data_df['prev_bar_cross_guppy_label'] = self.data_df['bar_cross_guppy_label'].shift(1)

            self.data_df['bar_cross_guppy_num'] = np.where(
                self.data_df['bar_cross_guppy_label'] != self.data_df['prev_bar_cross_guppy_label'],
                self.data_df['num'],
                np.nan
            )

            self.data_df['bar_cross_guppy_num'] = self.data_df['bar_cross_guppy_num'].fillna(method='ffill').fillna(0)

            self.data_df['bar_cross_guppy_duration'] = self.data_df['num'] - self.data_df['bar_cross_guppy_num']

            self.data_df['max_price_max'] = self.data_df['max_price']
            self.data_df['high_max'] = self.data_df['high']
            self.data_df['low_min'] = self.data_df['low']
            self.data_df['min_price_min'] = self.data_df['min_price']

            self.data_df['max_price_max_idx'] = self.data_df['max_price']
            self.data_df['high_max_idx'] = self.data_df['high']
            self.data_df['low_min_idx'] = self.data_df['low']
            self.data_df['min_price_min_idx'] = self.data_df['min_price']

            group_summary_df = self.data_df[['time', 'max_price_max', 'high_max', 'low_min', 'min_price_min',
                                             'max_price_max_idx', 'high_max_idx', 'low_min_idx', 'min_price_min_idx',

                                             'bar_cross_guppy_label', 'bar_cross_guppy_num',
                                             'bar_cross_guppy_duration']].groupby(['bar_cross_guppy_num']).agg(
                {'time': 'first',
                 'max_price_max': 'max',
                 'high_max': 'max',
                 'low_min': 'min',
                 'min_price_min': 'min',
                 'max_price_max_idx': 'idxmax',
                 'high_max_idx': 'idxmax',
                 'low_min_idx': 'idxmin',
                 'min_price_min_idx': 'idxmin',
                 'bar_cross_guppy_label': 'first',
                 'bar_cross_guppy_duration': 'last'
                 }
            )

            group_summary_df.reset_index(inplace=True)

            short_highest_price = 'max_price'  # max_price, high
            long_lowest_price = 'min_price'  # min_price, low

            group_summary_df.at[group_summary_df.index[0], 'bar_cross_guppy_label'] = 1 if group_summary_df.iloc[1]['bar_cross_guppy_label'] == 0 else 0

            group_summary_df['critical_price_id'] = np.where(
                group_summary_df['bar_cross_guppy_label'] == 0,
                group_summary_df[short_highest_price + '_max_idx'],
                np.where(
                    group_summary_df['bar_cross_guppy_label'] == 1,
                    group_summary_df[long_lowest_price + '_min_idx'],
                    0
                )
            )

            group_summary_df['critical_price'] = np.where(
                group_summary_df['bar_cross_guppy_label'] == 0,
                group_summary_df[short_highest_price + '_max'],
                np.where(
                    group_summary_df['bar_cross_guppy_label'] == 1,
                    group_summary_df[long_lowest_price + '_min'],
                    0
                )
            )

            group_summary_df['bar_cross_guppy_num'] = group_summary_df['bar_cross_guppy_num'].astype(int)
            group_summary_df['bar_cross_guppy_label'] = group_summary_df['bar_cross_guppy_label'].astype(int)

            group_summary_df['group_index'] = list(range(group_summary_df.shape[0]))

            critical_price_data_df = self.data_df.iloc[group_summary_df['critical_price_id']]  #####################

            critical_price_data_df.reset_index(inplace=True)

            critical_price_data_df['group_index'] = list(range(critical_price_data_df.shape[0]))

            critical_price_data_df['critical_price'] = group_summary_df['critical_price']

            critical_price_data_df['bar_cross_guppy_label'] = group_summary_df['bar_cross_guppy_label']

            group_data_dfs = []

            bar_cross_guppy_nums = group_summary_df['bar_cross_guppy_num'].tolist()
            bar_cross_guppy_labels = group_summary_df['bar_cross_guppy_label'].tolist()

            for idi in range(0, len(bar_cross_guppy_nums)):
                start_idxx = bar_cross_guppy_nums[idi]
                end_idxx = bar_cross_guppy_nums[idi + 1] if idi < len(bar_cross_guppy_nums) - 1 else self.data_df.shape[0]

                bar_cross_guppy_label = bar_cross_guppy_labels[idi]

                if bar_cross_guppy_label == 0:
                    group_df = self.data_df.iloc[start_idxx:end_idxx][['time', short_highest_price]]
                    group_df['critical_price'] = group_df[short_highest_price].cummax()
                    group_df['critical_price_id'] = group_df[short_highest_price].expanding().apply(lambda x: x.idxmax()).astype(int)
                    group_df = group_df.drop(columns=['time', short_highest_price])
                    group_df['group_index'] = idi
                elif bar_cross_guppy_label == 1:
                    group_df = self.data_df.iloc[start_idxx:end_idxx][['time', long_lowest_price]]
                    group_df['critical_price'] = group_df[long_lowest_price].cummin()
                    group_df['critical_price_id'] = group_df[long_lowest_price].expanding().apply(lambda x: x.idxmin()).astype(int)
                    group_df = group_df.drop(columns=['time', long_lowest_price])
                    group_df['group_index'] = idi
                else:
                    raise Exception("idi = " + str(idi) + " bar_cross_guppy_num = " + str(start_idxx) + " bar_cross_guppy_label = " + str(bar_cross_guppy_label))

                group_data_dfs += [group_df]

            group_data_df_all = pd.concat(group_data_dfs)

            if len(group_data_df_all) != self.data_df.shape[0]:
                raise Exception(
                    "group_data_df_all length = " + str(len(group_data_df_all)) + " while data_df length = " + str(
                        self.data_df.shape[0]))

            self.data_df = pd.concat([self.data_df, group_data_df_all], axis=1)

            aux_data_df = self.data_df[
                ['lower_vegas', 'upper_vegas', 'guppy_min', 'guppy_max', 'bar_cross_guppy_duration', 'high', 'low']]
            attach_df = aux_data_df.iloc[self.data_df['critical_price_id']]
            attach_df.reset_index(inplace=True)
            attach_df = attach_df.drop(columns=['index'])
            rename_dict = {}
            for column in aux_data_df.columns:
                rename_dict[column] = 'critical_' + column
            attach_df = attach_df.rename(columns=rename_dict)
            self.data_df = pd.concat([self.data_df, attach_df], axis=1)

            critical_price_data_df = critical_price_data_df.rename(columns={'index': 'critical_price_id'})
            critical_price_data_df['bar_cross_guppy_total_duration'] = group_summary_df['bar_cross_guppy_duration']

            key_columns = ['time', 'bar_cross_guppy_num', 'bar_cross_guppy_duration', 'critical_price_id',
                           'bar_cross_guppy_total_duration',
                           'critical_price', 'bar_cross_guppy_label', 'lower_vegas', 'upper_vegas', 'guppy_min',
                           'guppy_max', 'high', 'low']
            look_backward_group_num = 11  # 3 should be odd number  9
            for key_column in key_columns:
                for backward_i in range(1, look_backward_group_num + 1):
                    if backward_i == 1:
                        critical_price_data_df['prevGroup_' + str(backward_i) + key_column] = critical_price_data_df[
                            key_column].shift(1).fillna(0)
                    else:
                        critical_price_data_df['prevGroup_' + str(backward_i) + key_column] = critical_price_data_df[
                            'prevGroup_' + str(backward_i - 1) + key_column].shift(1).fillna(0)

            simple_critical_price_data_df = critical_price_data_df[
                ['group_index'] + [column for column in critical_price_data_df.columns if 'prevGroup' in column]]


            self.data_df = pd.merge(self.data_df, simple_critical_price_data_df, on=['group_index'], how='left')

            need_look_backward_cols = ["prevGroup_1critical_price", "prevGroup_1critical_price_id",
                                       'prevGroup_1bar_cross_guppy_duration', 'prevGroup_1bar_cross_guppy_num',
                                       'prevGroup_1high', 'prevGroup_1low']
            for li in range(2, look_backward_group_num + 1):
                need_look_backward_cols += ['prevGroup_' + str(li) + 'critical_price',
                                            'prevGroup_' + str(li) + 'critical_price_id',
                                            'prevGroup_' + str(li) + 'high', 'prevGroup_' + str(li) + 'low',
                                            'prevGroup_' + str(li) + 'lower_vegas',
                                            'prevGroup_' + str(li) + 'upper_vegas',
                                            'prevGroup_' + str(li) + 'bar_cross_guppy_total_duration',
                                            'prevGroup_' + str(li) + 'bar_cross_guppy_num'
                                            ]

            no_need_look_backward_cols = ['critical_price', 'critical_price_id', 'critical_bar_cross_guppy_duration',
                                          'bar_cross_guppy_num',
                                          'critical_high', 'critical_low']
            for li in range(1, look_backward_group_num):
                no_need_look_backward_cols += ['prevGroup_' + str(li) + 'critical_price',
                                               'prevGroup_' + str(li) + 'critical_price_id',
                                               'prevGroup_' + str(li) + 'high', 'prevGroup_' + str(li) + 'low',
                                               'prevGroup_' + str(li) + 'lower_vegas',
                                               'prevGroup_' + str(li) + 'upper_vegas',
                                               'prevGroup_' + str(li) + 'bar_cross_guppy_total_duration',
                                               'prevGroup_' + str(li) + 'bar_cross_guppy_num'
                                               ]

            self.data_df['long_need_look_backward'] = self.data_df['bar_cross_guppy_label'] == 0

            target_long_cols = ['long_critical_price', 'long_critical_price_id',
                                'long_critical_bar_cross_guppy_duration', 'long_bar_cross_guppy_num',
                                'long_critical_high', 'long_critical_low']
            for li in range(1, look_backward_group_num):
                target_long_cols += ['long_prevGroup_' + str(li) + 'critical_price',
                                     'long_prevGroup_' + str(li) + 'critical_price_id',
                                     'long_prevGroup_' + str(li) + 'high', 'long_prevGroup_' + str(li) + 'low',
                                     'long_prevGroup_' + str(li) + 'lower_vegas',
                                     'long_prevGroup_' + str(li) + 'upper_vegas',
                                     'long_prevGroup_' + str(li) + 'bar_cross_guppy_total_duration',
                                     'long_prevGroup_' + str(li) + 'bar_cross_guppy_num'
                                     ]

            for ti in range(len(target_long_cols)):
                self.data_df[target_long_cols[ti]] = np.where(
                    self.data_df['long_need_look_backward'],
                    self.data_df[need_look_backward_cols[ti]],
                    self.data_df[no_need_look_backward_cols[ti]]
                )

            self.data_df['short_need_look_backward'] = self.data_df['bar_cross_guppy_label'] == 1
            # target_short_cols = ['short_critical_price', 'short_critical_price_id', 'short_critical_bar_cross_guppy_duration',
            #                      'short_prevGroup_1critical_price', 'short_prevGroup_1critical_price_id',
            #                      'short_prevGroup_2critical_price', 'short_prevGroup_2critical_price_id',
            #                     'short_prevGroup_2lower_vegas', 'short_prevGroup_2upper_vegas', 'short_prevGroup_1bar_cross_guppy_total_duration', 'short_prevGroup_2bar_cross_guppy_total_duration']

            target_short_cols = ['short_critical_price', 'short_critical_price_id',
                                 'short_critical_bar_cross_guppy_duration', 'short_bar_cross_guppy_num',
                                 'short_critical_high', 'short_critical_low']
            for li in range(1, look_backward_group_num):
                target_short_cols += ['short_prevGroup_' + str(li) + 'critical_price',
                                      'short_prevGroup_' + str(li) + 'critical_price_id',
                                      'short_prevGroup_' + str(li) + 'high', 'short_prevGroup_' + str(li) + 'low',
                                      'short_prevGroup_' + str(li) + 'lower_vegas',
                                      'short_prevGroup_' + str(li) + 'upper_vegas',
                                      'short_prevGroup_' + str(li) + 'bar_cross_guppy_total_duration',
                                      'short_prevGroup_' + str(li) + 'bar_cross_guppy_num'
                                      ]

            for ti in range(len(target_short_cols)):
                self.data_df[target_short_cols[ti]] = np.where(
                    self.data_df['short_need_look_backward'],
                    self.data_df[need_look_backward_cols[ti]],
                    self.data_df[no_need_look_backward_cols[ti]]
                )

            self.group_summary_df = group_summary_df
            self.critical_price_data_df = critical_price_data_df






        ######## keybox #########

        if enable_short_macd_signal:
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

                #self.log_msg("macd_cross_label = " + str(macd_cross_label))

                group_df = self.data_df.iloc[start_idxx:end_idxx][['time', 'macd']]

                if macd_cross_label == 0:
                    # self.log_msg("start_idxx = " + str(start_idxx) + " end_idxx = " + str(end_idxx))
                    # self.log_msg("group_df:")
                    # self.log_msg(group_df)
                    # self.log_msg("")

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

            # self.log_msg("macd_group_data_df_all:")
            # self.log_msg(macd_group_data_df_all.iloc[0:60])
            #
            # self.log_msg("macd_group_data_df_all length = " + str(macd_group_data_df_all.shape[0]))
            # self.log_msg("data_df length = " + str(self.data_df.shape[0]))
            # self.log_msg(len(macd_group_data_df_all))

            if len(macd_group_data_df_all) != self.data_df.shape[0]:
                raise Exception(
                    "macd_group_data_df_all length = " + str(
                        len(macd_group_data_df_all)) + " while data_df length = " + str(
                        self.data_df.shape[0]))

            #self.log_msg("First")
            #self.log_msg(self.data_df.iloc[0:100][['time', 'macd', 'msignal']])

            self.data_df = pd.concat([self.data_df, macd_group_data_df_all], axis=1)

            aux_macd_data_df = self.data_df[
                ['lower_vegas', 'upper_vegas', 'guppy_min', 'guppy_max', 'macd_cross_duration', 'high', 'low', 'max_price',
                 'min_price']]

            #self.log_msg(self.data_df.iloc[0:100][['time', 'macd', 'msignal', 'critical_value_id', 'critical_value']])
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
                #self.log_msg("Add column " + target_long_cols[ti])
                self.data_df[target_long_cols[ti]] = np.where(
                    self.data_df['long_macd_need_look_backward'],
                    self.data_df[need_look_backward_cols[ti]],
                    self.data_df[no_need_look_backward_cols[ti]]
                )
                #self.log_msg("Column " + target_long_cols[ti] + " in data_df? " + str(target_long_cols[ti] in self.data_df.columns))



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




        self.data_df['id'] = list(range(self.data_df.shape[0]))

        #Singapore  3gradients_positive
        macd_enter_gradient_num = self.optimal_gradient_num
        self.data_df['long_macd_long_enter'] = (self.data_df[self.macd_gradient] > 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] > 0) for i in range(1, macd_enter_gradient_num)])
        self.data_df['long_macd_short_enter'] = (self.data_df[self.macd_gradient] < 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] < 0) for i in range(1, macd_enter_gradient_num)])

        if self.use_guppy_filter:
            self.data_df['long_macd_long_enter_too_late'] = (self.data_df[self.macd_gradient] > 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] > 0) for i in range(1, macd_enter_gradient_num+1)])
            self.data_df['long_macd_short_enter_too_late'] = (self.data_df[self.macd_gradient] < 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] < 0) for i in range(1, macd_enter_gradient_num+1)])





        self.data_df['long_macd_long_enter_ready'] = (self.data_df[self.macd_gradient] > 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] > 0) for i in range(1, macd_enter_gradient_num-1)])
        self.data_df['long_macd_short_enter_ready'] = (self.data_df[self.macd_gradient] < 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] < 0) for i in range(1, macd_enter_gradient_num-1)])


        if do_message_printing and self.is_notify and print_ready and not self.use_guppy_condition and not self.reverse_strategy:

            self.log_msg("long_macd_long_enter_ready = " + str(self.data_df.iloc[-1]['long_macd_long_enter_ready']))
            self.log_msg("long_macd_long_enter = " + str(self.data_df.iloc[-1]['long_macd_long_enter']))
            self.log_msg("long_macd_short_enter_ready = " + str(self.data_df.iloc[-1]['long_macd_short_enter_ready']))
            self.log_msg("long_macd_short_enter = " + str(self.data_df.iloc[-1]['long_macd_short_enter']))

            if self.data_df.iloc[-1]['long_macd_long_enter_ready'] and (not self.data_df.iloc[-1]['long_macd_long_enter']) and self.current_position <= 0:
                ready_msg = "Ready to Open Long Position of " + str(self.init_entry_value) + " USD for " + self.currency +  " at " + str(self.data_df.iloc[-1]['time'] + timedelta(hours = 2))
                sendEmail(ready_msg, "", is_alternative=self.is_alternative)
                self.log_msg(ready_msg)
            elif self.data_df.iloc[-1]['long_macd_short_enter_ready'] and (not self.data_df.iloc[-1]['long_macd_short_enter']) and self.current_position >= 0:
                ready_msg = "Ready to Open Short Position of " + str(self.init_entry_value) + " USD for " + self.currency +  " at " + str(self.data_df.iloc[-1]['time'] + timedelta(hours = 2))
                sendEmail(ready_msg, "", is_alternative=self.is_alternative)
                self.log_msg(ready_msg)



        if enable_short_macd_signal:
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

        #print("Fuck here:")
        #print(self.data_df.iloc[-10:][['time', 'macd_short_enter', 'long_macd_short_enter']])


        if self.use_guppy_filter:
            self.data_df['macd_long_enter'] = self.data_df['macd_long_enter'] & (~self.data_df['guppy_all_strong_aligned_short'])
            self.data_df['macd_short_enter'] = self.data_df['macd_short_enter'] & (~self.data_df['guppy_all_strong_aligned_long'])

            if self.also_filter_too_late:
                self.data_df['macd_long_enter'] = self.data_df['macd_long_enter'] & (~self.data_df['long_macd_long_enter_too_late'])
                self.data_df['macd_short_enter'] = self.data_df['macd_short_enter'] & (~self.data_df['long_macd_short_enter_too_late'])

        elif self.use_guppy_condition:
            self.data_df['macd_long_enter'] = self.data_df['macd_long_enter'] & (self.data_df['guppy_all_strong_aligned_long'])
            self.data_df['macd_short_enter'] = self.data_df['macd_short_enter'] & (self.data_df['guppy_all_strong_aligned_short'])

        #print("Fuck here2:")
        #print(self.data_df.iloc[-10:][['time', 'macd_short_enter', 'long_macd_short_enter']])


        # self.data_df['macd_long_enter'] = self.data_df['short_macd_long_enter']
        # self.data_df['macd_short_enter'] = self.data_df['short_macd_short_enter']

        #self.data_df['macd_long_enter'] = self.data_df['short_macd_long_enter'] | self.data_df['long_macd_long_enter']
        #self.data_df['macd_short_enter'] = self.data_df['short_macd_short_enter'] | self.data_df['long_macd_short_enter']


        if enable_short_macd_signal:
            self.data_df['short_macd_long_exit'] = (self.data_df['macd_gradient'] < 0) & (self.data_df['macd'] < self.data_df['msignal'])
            self.data_df['short_macd_short_exit'] = (self.data_df['macd_gradient'] > 0) & (self.data_df['macd'] > self.data_df['msignal'])


        macd_exit_gradient_num = self.optimal_gradient_num
        self.data_df['long_macd_long_exit'] = (self.data_df[self.macd_gradient] < 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] < 0) for i in range(1, macd_exit_gradient_num)])
        self.data_df['long_macd_short_exit'] = (self.data_df[self.macd_gradient] > 0) & reduce(lambda left, right: left & right,
                                                                        [(self.data_df['prev' + str(i) + '_' + self.macd_gradient] > 0) for i in range(1, macd_exit_gradient_num)])

        if self.use_guppy_filter_for_exit:
            self.data_df['long_macd_long_exit'] = self.data_df['long_macd_long_exit'] & (~self.data_df['guppy_all_strong_aligned_long'])
            self.data_df['long_macd_short_exit'] = self.data_df['long_macd_short_exit'] & (~self.data_df['guppy_all_strong_aligned_short'])

        if self.guppy_force_out:
            self.data_df['long_macd_long_exit'] = self.data_df['long_macd_long_exit'] | self.data_df['guppy_all_strong_aligned_short']
            self.data_df['long_macd_short_exit'] = self.data_df['long_macd_short_exit'] | self.data_df['guppy_all_strong_aligned_long']

        self.data_df['long_macd_long_exit_without_rsi'] = self.data_df['long_macd_long_exit']
        self.data_df['long_macd_short_exit_without_rsi'] = self.data_df['long_macd_short_exit']


        if self.use_rsi_to_exit:
            self.data_df['long_macd_long_exit'] = self.data_df['long_macd_long_exit'] | self.data_df['over_bought']
            self.data_df['long_macd_short_exit'] = self.data_df['long_macd_short_exit'] | self.data_df['over_sold']


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

        if do_smart_execution and not do_real_money_trading:


            result_columns += ['pnl']

            strategy_record_columns = ['side', 'long_trade_id', 'short_trade_id', 'strategy_id', 'leverage', 'entry_time', 'entry_price', 'entry_value', 'exit_time', 'exit_price', 'exit_value', 'pnl']
            strategy_execution_record_columns = ['side', 'long_trade_id', 'short_trade_id', 'strategy_id', 'execution_id', 'leverage', 'take_profit_pct', 'take_profit_price', 'take_loss_pct', 'take_loss_price',
                                                 'entry_time', 'entry_price', 'entry_value', 'exit_time', 'exit_price', 'exit_value', 'pnl']

            long_strategy_records = []
            long_strategy_execution_records = []

            short_strategy_records = []
            short_strategy_execution_records = []



        if self.temporary_long:
            if (not temporary_decision) and (not self.data_df.iloc[-1]['macd_long_enter']):

                current_time = self.data_df.iloc[-1]['time'] + timedelta(hours = 1)

                if do_real_money_trading and self.wakeup == 1:

                    self.log_msg("At " + str(current_time) + ", Revoke long decision just made.")

                    filled_size = 0
                    orderResponse = self.coinbase_client.get_order(order_id=self.long_order_id)
                    if hasattr(orderResponse, "order"):
                        order = orderResponse.order
                        if order is not None:
                            filled_size = float(order['filled_size'])

                    self.log_msg("Fill size = " + str(filled_size) + ", Attempt size = " + str(self.long_attempt_size))

                    if filled_size < self.long_attempt_size:
                        try:
                            self.log_msg("Cancel the open order " + self.long_order_id)
                            cancel_response = self.coinbase_client.cancel_orders(order_ids=[self.long_order_id])
                            self.log_msg("Cancel Response:")
                            self.log_msg(cancel_response)
                        except Exception as e:
                            self.log_msg("Error:", e)

                    if filled_size > 0:

                        fill_price = float(order['average_filled_price'])

                        try:
                            self.log_msg("Sell " + str(filled_size) + " at market price")
                            client_order_id = f"order_{uuid.uuid4()}"
                            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,     #BTC-USDC is the correct product id
                                                           client_order_id=client_order_id,
                                                           side="SELL",
                                                           order_configuration={
                                                               "market_market_ioc":{
                                                                   "base_size" : str(filled_size)
                                                               }
                                                           },
                                                           leverage=str(default_leverage),
                                                           margin_type = "CROSS",
                                                           retail_portfolio_id=self.coinbase_portfolio_id
                                                           )
                            self.log_msg(f"Order placed: {response}")
                        except Exception as e:
                            self.log_msg("Error:", e)


                        close_size = 0
                        while close_size < filled_size:
                            orderResponse = self.coinbase_client.get_order(order_id=response['success_response']['order_id'])
                            if hasattr(orderResponse, "order"):
                                order = orderResponse.order
                                if order is not None:
                                    close_size = float(order['filled_size'])

                            if close_size < filled_size:
                                time.sleep(1)

                        close_fill_price = float(order['average_filled_price'])

                        self.delay_cost_data += [['revoke long', current_time, fill_price, close_fill_price, filled_size]]




                    self.reset_long()

                if do_message_printing and self.is_notify:
                    message_title = "Revoke long decision made just now by shorting " + str(self.temporary_delta_position) + " units at market price"
                    message = ""

                    if not print_email_message_to_file:
                        sendEmail(message_title, message, is_alternative=self.is_alternative)
                    else:
                        self.cache_email_messages(message_title, message, current_time)

            if not temporary_decision:
                self.temporary_long = False
                self.temporary_delta_position = 0

        if self.temporary_close_long:
            if (not temporary_decision) and (not (self.data_df.iloc[-1]['long_macd_long_exit'] or self.data_df.iloc[-1]['macd_short_enter'])):

                current_time = self.data_df.iloc[-1]['time'] + timedelta(hours = 1)

                if do_real_money_trading and self.wakeup == 1:

                    self.log_msg("At " + str(current_time) + ", Revoke close long decision just made.")

                    filled_size = 0
                    orderResponse = self.coinbase_client.get_order(order_id=self.close_long_order_id)
                    if hasattr(orderResponse, "order"):
                        order = orderResponse.order
                        if order is not None:
                            filled_size = float(order['filled_size'])

                    self.log_msg("Fill size = " + str(filled_size) + ", Attempt size = " + str(self.close_long_attempt_size))

                    if filled_size < self.close_long_attempt_size:
                        try:
                            self.log_msg("Cancel the open order " + self.close_long_order_id)
                            cancel_response = self.coinbase_client.cancel_orders(order_ids=[self.close_long_order_id])
                            self.log_msg("Cancel Response:")
                            self.log_msg(cancel_response)
                        except Exception as e:
                            self.log_msg("Error:", e)

                    if filled_size > 0:

                        fill_price = float(order['average_filled_price'])

                        try:
                            self.log_msg("Buy " + str(filled_size) + " at market price")
                            client_order_id = f"order_{uuid.uuid4()}"
                            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,     #BTC-USDC is the correct product id
                                                           client_order_id=client_order_id,
                                                           side="BUY",
                                                           order_configuration={
                                                               "market_market_ioc":{
                                                                   "base_size" : str(filled_size)
                                                               }
                                                           },
                                                           leverage=str(default_leverage),
                                                           margin_type = "CROSS",
                                                           retail_portfolio_id=self.coinbase_portfolio_id
                                                           )
                            self.log_msg(f"Order placed: {response}")
                        except Exception as e:
                            self.log_msg("Error:", e)

                        close_size = 0
                        while close_size < filled_size:
                            orderResponse = self.coinbase_client.get_order(order_id=response['success_response']['order_id'])
                            if hasattr(orderResponse, "order"):
                                order = orderResponse.order
                                if order is not None:
                                    close_size = float(order['filled_size'])

                            if close_size < filled_size:
                                time.sleep(1)

                        close_fill_price = float(order['average_filled_price'])

                        self.delay_cost_data += [['revoke close long', current_time, fill_price, close_fill_price, filled_size]]


                    self.reset_close_long()

                if do_message_printing and self.is_notify:
                    message_title = "Revoke close long decision made just now by longing " + str(self.temporary_delta_position) + " units at market price"
                    message = ""

                    if not print_email_message_to_file:
                        sendEmail(message_title, message, is_alternative=self.is_alternative)
                    else:
                        self.cache_email_messages(message_title, message, current_time)

            if not temporary_decision:
                self.temporary_close_long = False
                self.temporary_delta_position = 0


        if self.temporary_short:
            if (not temporary_decision) and (not self.data_df.iloc[-1]['macd_short_enter']):

                current_time = self.data_df.iloc[-1]['time'] + timedelta(hours = 1)

                if do_real_money_trading and self.wakeup == 1:

                    self.log_msg("At " + str(current_time) + ", Revoke short decision just made.")

                    filled_size = 0
                    orderResponse = self.coinbase_client.get_order(order_id=self.short_order_id)
                    if hasattr(orderResponse, "order"):
                        order = orderResponse.order
                        if order is not None:
                            filled_size = float(order['filled_size'])

                    self.log_msg("Fill size = " + str(filled_size) + ", Attempt size = " + str(self.short_attempt_size))

                    if filled_size < self.short_attempt_size:
                        try:
                            self.log_msg("Cancel the open order " + self.short_order_id)
                            cancel_response = self.coinbase_client.cancel_orders(order_ids=[self.short_order_id])
                            self.log_msg("Cancel Response:")
                            self.log_msg(cancel_response)
                        except Exception as e:
                            self.log_msg("Error:", e)

                    if filled_size > 0:

                        fill_price = float(order['average_filled_price'])

                        try:
                            self.log_msg("Buy " + str(filled_size) + " at market price")
                            client_order_id = f"order_{uuid.uuid4()}"
                            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                           client_order_id=client_order_id,
                                                           side="BUY",
                                                           order_configuration={
                                                               "market_market_ioc":{
                                                                   "base_size" : str(filled_size)
                                                               }
                                                           },
                                                           leverage=str(default_leverage),
                                                           margin_type = "CROSS",
                                                           retail_portfolio_id=self.coinbase_portfolio_id
                                                           )
                            self.log_msg(f"Order placed: {response}")
                        except Exception as e:
                            self.log_msg("Error:", e)

                        close_size = 0
                        while close_size < filled_size:
                            orderResponse = self.coinbase_client.get_order(
                                order_id=response['success_response']['order_id'])
                            if hasattr(orderResponse, "order"):
                                order = orderResponse.order
                                if order is not None:
                                    close_size = float(order['filled_size'])

                            if close_size < filled_size:
                                time.sleep(1)

                        close_fill_price = float(order['average_filled_price'])

                        self.delay_cost_data += [['revoke short', current_time, fill_price, close_fill_price, filled_size]]


                    self.reset_short()


                if do_message_printing and self.is_notify:
                    message_title = "Revoke short decision made just now by longing " + str(self.temporary_delta_position) + " units at market price"
                    message = ""

                    if not print_email_message_to_file:
                        sendEmail(message_title, message, is_alternative=self.is_alternative)
                    else:
                        self.cache_email_messages(message_title, message, current_time)

            if not temporary_decision:
                self.temporary_short = False
                self.temporary_delta_position = 0

        if self.temporary_close_short:
            if (not temporary_decision) and (not (self.data_df.iloc[-1]['long_macd_short_exit'] or self.data_df.iloc[-1]['macd_long_enter'])):

                current_time = self.data_df.iloc[-1]['time'] + timedelta(hours = 1)

                if do_real_money_trading and self.wakeup == 1:

                    self.log_msg("At " + str(current_time) + ", Revoke close short decision just made.")

                    filled_size = 0
                    orderResponse = self.coinbase_client.get_order(order_id=self.close_short_order_id)
                    if hasattr(orderResponse, "order"):
                        order = orderResponse.order
                        if order is not None:
                            filled_size = float(order['filled_size'])

                    self.log_msg("Fill size = " + str(filled_size) + ", Attempt size = " + str(self.close_short_attempt_size))

                    if filled_size < self.close_long_attempt_size:
                        try:
                            self.log_msg("Cancel the open order " + self.close_short_order_id)
                            cancel_response = self.coinbase_client.cancel_orders(order_ids=[self.close_short_order_id])
                            self.log_msg("Cancel Response:")
                            self.log_msg(cancel_response)
                        except Exception as e:
                            self.log_msg("Error:", e)

                    if filled_size > 0:

                        fill_price = float(order['average_filled_price'])

                        try:
                            self.log_msg("Sell " + str(filled_size) + " at market price")
                            client_order_id = f"order_{uuid.uuid4()}"
                            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,     #BTC-USDC is the correct product id
                                                           client_order_id=client_order_id,
                                                           side="SELL",
                                                           order_configuration={
                                                               "market_market_ioc":{
                                                                   "base_size" : str(filled_size)
                                                               }
                                                           },
                                                           leverage=str(default_leverage),
                                                           margin_type = "CROSS",
                                                           retail_portfolio_id=self.coinbase_portfolio_id
                                                           )
                            self.log_msg(f"Order placed: {response}")
                        except Exception as e:
                            self.log_msg("Error:", e)

                        close_size = 0
                        while close_size < filled_size:
                            orderResponse = self.coinbase_client.get_order(
                                order_id=response['success_response']['order_id'])
                            if hasattr(orderResponse, "order"):
                                order = orderResponse.order
                                if order is not None:
                                    close_size = float(order['filled_size'])

                            if close_size < filled_size:
                                time.sleep(1)

                        close_fill_price = float(order['average_filled_price'])

                        self.delay_cost_data += [['revoke close short', current_time, fill_price, close_fill_price, filled_size]]


                    self.reset_close_short()

                if do_message_printing and self.is_notify:
                    message_title = "Revoke close short decision made just now by shorting " + str(self.temporary_delta_position) + " units at market price"
                    message = ""

                    if not print_email_message_to_file:
                        sendEmail(message_title, message, is_alternative=self.is_alternative)
                    else:
                        self.cache_email_messages(message_title, message, current_time)

            if not temporary_decision:
                self.temporary_close_short = False
                self.temporary_delta_position = 0


        self.log_msg("")
        self.log_msg("Calculating Long positions.............")
        self.log_msg("")

        long_start_ids = which(self.data_df['macd_long_enter'])

        is_effective = [1] * len(long_start_ids)

        long_trade_id = 0

        if self.use_rsi_to_exit:
            exit_by_rsi = False
            id_when_exit_by_rsi = -1

        for i in range(len(long_start_ids)):

            if is_effective[i] == 0:
                self.data_df.at[long_start_ids[i], 'macd_long_enter'] = False
                continue

            temp_i = i
            long_start_id = long_start_ids[i]
            long_fire_data = self.data_df.iloc[long_start_id]

            # if print_execution_details:
            #     print("i = " + str(i) + "/" + str(len(long_start_ids)))
            #     print("long_start_id = " + str(long_start_id))


            if self.use_rsi_to_exit and exit_by_rsi:
                if (long_fire_data['long_macd_long_enter_too_late'] and long_start_id > 0 and not self.data_df.iloc[long_start_id-1]['guppy_all_strong_aligned_short']) or (long_start_id - id_when_exit_by_rsi <= 10):
                    self.data_df.at[long_start_ids[i], 'macd_long_enter'] = False
                    continue
                else:
                    exit_by_rsi = False
                    id_when_exit_by_rsi = -1


            instrument = long_fire_data['currency']
            entry_time = long_fire_data['time']
            entry_price = long_fire_data['close']
            entry_id = long_fire_data['id']

            if self.do_stop_loss:
                long_stop_loss_price = long_fire_data['long_critical_price']


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

                    position = self.init_entry_value/entry_price * default_leverage

                    self.log_msg("Before position = " + str(position))

                    # if entry_price >= 1:
                    #     position = round(position, 3)
                    # else:
                    #     position = int(round(position, 0))

                    position = round(position, self.coinbase_decimal)
                    if self.coinbase_decimal == 0:
                        position = int(position)


                    self.log_msg("After position = " + str(position))

                    #delta_position = position - self.current_position
                    delta_position = position

                    #self.current_position = position


                    message_title = "Long " + self.currency + " " + str(delta_position) + " units"
                    print("entry_price:")
                    print(entry_price)
                    message = "At " + current_time + ", long " + self.currency + " roughly " + str(delta_position) + " units at entry price " + str(round(entry_price, self.decimal)) + "\n"
                    message += "This makes it now at a long position of " + str(position) + " units with an actual notional of " + str(self.init_entry_value) + " dollar\n"

                    self.log_msg("message_title = " + message_title)
                    self.log_msg("message:")
                    self.log_msg(message)

                    if not print_email_message_to_file:
                        sendEmail(message_title, message, is_alternative=self.is_alternative)
                    else:
                        self.cache_email_messages(message_title, message, current_time)

                    if temporary_decision:
                        self.temporary_long = True
                        self.temporary_delta_position = abs(delta_position)


                    if do_real_money_trading and self.wakeup == 1 and long_start_id == self.data_df.shape[0] - 1:
                        if self.current_real_position <= 0 and self.long_order_id is None:

                            if do_smart_execution:
                                real_position = self.init_entry_value/self.crypto_last_price * self.average_leverage
                            else:
                                real_position = self.init_entry_value/self.crypto_last_price * default_leverage


                            # if self.crypto_last_price >= 1:
                            #     real_position = round(real_position, 3)
                            # else:
                            #     real_position = int(round(real_position, 0))
                            real_position = round(real_position, self.coinbase_decimal)
                            if self.coinbase_decimal == 0:
                                real_position = int(real_position)

                            self.log_msg(self.currency + " current real position = " + str(self.current_real_position))
                            self.log_msg(self.currency + " target real position = " + str(real_position))
                            #real_delta_position = real_position - self.current_real_position
                            real_delta_position = real_position

                            try:
                                self.log_msg("At " + current_time + ", open long position by placing real long order of " + str(real_delta_position) + " at limit price " + str(self.crypto_last_price) + " to Coinbase with leverage " + str(default_leverage) + "x")
                                client_order_id = f"order_{uuid.uuid4()}"
                                response = self.coinbase_client.create_order(product_id=self.currency_coinbase,     #BTC-USDC is the correct product id
                                                               client_order_id=client_order_id,
                                                               side="BUY",
                                                               order_configuration={
                                                                   "limit_limit_gtc":{
                                                                       "base_size" : str(real_delta_position),
                                                                       "limit_price" : str(self.crypto_last_price)

                                                                   }
                                                               },
                                                               leverage=str(default_leverage),
                                                               margin_type = "CROSS",
                                                               retail_portfolio_id=self.coinbase_portfolio_id
                                                               )
                                self.log_msg(f"Order placed: {response}")
                            except Exception as e:
                                self.log_msg(f"Order failed: {e}")

                            self.long_order_id = response['success_response']['order_id']
                            self.long_attempt_size = real_delta_position


                            if self.do_stop_loss:
                                try:
                                    self.log_msg("At " + current_time + ", place real long stop loss order of " + str(
                                        real_position) + " at stop loss price " + str(
                                        long_stop_loss_price))

                                    stop_loss_order_id = f"order_{uuid.uuid4()}"
                                    response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                                   # BTC-USDC is the correct product id
                                                                   client_order_id=stop_loss_order_id,
                                                                   side="SELL",
                                                                   order_configuration={
                                                                       "stop_limit_stop_limit_gtc": {
                                                                           "base_size": str(real_position),
                                                                           "limit_price": str(long_stop_loss_price*0.9),
                                                                           "stop_price": str(long_stop_loss_price)
                                                                       }
                                                                   },
                                                                   leverage="10",
                                                                   margin_type="CROSS"
                                                                   # retail_portfolio_id="0194271a-bd95-7ba7-a028-6561a970128b"
                                                                   )

                                except Exception as e:
                                    self.log_msg(f"Order failed: {e}")

                            # self.long_order_id = response['success_response']['order_id']
                            # self.long_attempt_size = real_delta_position



            if do_smart_execution:
                strategy_executions = []

                if do_real_money_trading:
                    if self.wakeup == 1 and long_start_id == self.data_df.shape[0] - 1 and self.current_real_position <= 0 and self.long_execution_order_id is None:

                        prod_sizes = real_delta_position * self.distribution

                        self.log_msg('[Execution] Long position to open  = ' + str(real_delta_position))
                        self.log_msg('[Execution] distribution = ' + str(self.distribution))
                        self.log_msg('[Execution] prod sizes = ' + str(prod_sizes))

                        entry_value = self.init_entry_value/(len(self.leverage) * 2) if use_extra_execution else self.init_entry_value/len(self.leverage)
                        self.log_msg('[Execution] entry_value = ' + str(entry_value))

                        for k in range(len(self.leverage)):
                            strategy_execution = StrategyExecution(side = 1, leverage = self.leverage[k], take_profit_pct = self.take_profit_pct[k], take_loss_pct = self.take_loss_pct[k],
                                                                   strategy_id = k+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = self.crypto_last_price,
                                                                   execution_entry_time = entry_time, execution_entry_price = self.crypto_last_price,
                                                                   strategy_entry_value = entry_value, execution_entry_value = entry_value,
                                                                   default_leverage=default_leverage, prod_size = round(prod_sizes[k], self.coinbase_decimal))
                            strategy_executions += [strategy_execution]



                        if use_extra_execution:

                            extra_prod_size = real_delta_position * self.extra_distribution

                            extra_entry_value = self.init_entry_value / 2.0

                            self.log_msg('[Execution] extra_distribution = ' + str(self.extra_distribution))
                            self.log_msg('[Execution] extr_prod_size = ' + str(extra_prod_size))
                            self.log_msg('[Execution] extra_entry_value = ' + str(extra_entry_value))

                            strategy_execution = StrategyExecution(side = 1, leverage = self.leverage[0], take_profit_pct = self.take_profit_pct[0], take_loss_pct = self.take_loss_pct[0],
                                                                       strategy_id = len(self.leverage)+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = self.crypto_last_price,
                                                                       execution_entry_time = entry_time, execution_entry_price = self.crypto_last_price,
                                                                       strategy_entry_value = extra_entry_value, execution_entry_value = extra_entry_value,
                                                                       default_leverage=default_leverage,prod_size = round(extra_prod_size, self.coinbase_decimal))

                            strategy_executions += [strategy_execution]


                        self.smart_executor_manager.open_executions(currency = self.currency, target_position = real_delta_position,
                                                                    entry_time = entry_time, strategy_executions = strategy_executions
                                                                    )

                        self.long_execution_order_id = self.long_order_id



                else:

                    for k in range(len(self.leverage)):
                        strategy_execution = StrategyExecution(side = 1, leverage = self.leverage[k], take_profit_pct = self.take_profit_pct[k], take_loss_pct = self.take_loss_pct[k],
                                                               strategy_id = k+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                               execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                               strategy_entry_value = self.each_strategy_entry_value, execution_entry_value = self.each_strategy_entry_value, default_leverage=default_leverage)
                        strategy_executions += [strategy_execution]

                    #This is the extra one

                    if use_extra_execution:
                        strategy_execution = StrategyExecution(side = 1, leverage = self.leverage[0], take_profit_pct = self.take_profit_pct[0], take_loss_pct = self.take_loss_pct[0],
                                                                   strategy_id = len(self.leverage)+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                                   execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                                   strategy_entry_value = self.init_entry_value/2.0, execution_entry_value = self.init_entry_value/2.0, default_leverage=default_leverage)

                        strategy_executions += [strategy_execution]


                    total_strategy_pnl = 0

            if self.do_stop_loss:
                exit_long_by_stop_loss = False

                is_stop_loss = False
                stop_loss_exit_id = -1
                stop_loss_exit_time = None
                stop_loss_exit_price = -1

            while long_start_id + j < self.data_df.shape[0]:

                cur_data_1h = self.data_df.iloc[long_start_id + j]
                #self.log_msg("")
                #self.log_msg("Long 1h time = " + str(self.data_df.iloc[long_start_id + j]['time']) + '..............................')

                if do_smart_execution and not do_real_money_trading:

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


                    #self.log_msg("loc_start = " + str(loc_start))
                    #self.log_msg("loc_end = " + str(loc_end))

                    for y in range(loc_start, loc_end):
                        cur_data = use_data_df.iloc[y]

                        #self.log_msg("    Long 5min time = " + str(cur_data['time']))

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
                                                                       strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value, default_leverage=default_leverage
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

                                if k < len(strategy_executions) - 1 if use_extra_execution else len(strategy_executions):
                                    next_execution = StrategyExecution(side=execution.side, leverage=execution.leverage, take_profit_pct=execution.take_profit_pct,
                                                                   take_loss_pct=execution.take_loss_pct, strategy_id=execution.strategy_id, execution_id=execution.execution_id+1,
                                                                   strategy_entry_time=execution.strategy_entry_time, strategy_entry_price=execution.strategy_entry_price,
                                                                   execution_entry_time=cur_data['time'], execution_entry_price=execution.execution_exit_price,
                                                                   strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value, default_leverage=default_leverage
                                                                   )
                                    strategy_executions[k] = next_execution
                                    execution = strategy_executions[k]
                                    #hit_stop_profit = cur_data['high'] >= execution.take_profit_price
                                else:
                                    break


                if (not (do_smart_execution and not do_real_money_trading)) or can_use_5min:
                    cur_data = self.data_df.iloc[long_start_id + j]


                is_exit = False

                if self.do_stop_loss and not exit_long_by_stop_loss:
                    if cur_data['low'] < long_stop_loss_price - (1e-6) and cur_data['group_index'] >= 2:
                        if self.reentry_after_stop_loss:
                            is_exit = True
                        else:
                            is_stop_loss = True
                            stop_loss_exit_id = cur_data['id']
                            stop_loss_exit_time = cur_data['time']
                            stop_loss_exit_price = long_stop_loss_price


                        exit_long_by_stop_loss = True

                if not is_exit:
                    if long_macd_indicate_long or (not is_short_macd_fire):
                        is_exit = cur_data['long_macd_long_exit'] or cur_data['macd_short_enter']
                        #is_exit = cur_data['long_macd_long_exit'] or cur_data['short_macd_short_enter']
                    elif enable_short_macd_signal:
                        is_exit = cur_data['short_macd_long_exit'] or cur_data['macd_short_enter']
                        #is_exit = cur_data['short_macd_long_exit'] or cur_data['short_macd_short_enter']

                # if self.use_rsi_to_exit and cur_data['over_bought'] and not cur_data['long_macd_long_exit_without_rsi']:
                #     if not cur_data['long_macd_long_enter_too_late']:
                #         is_exit = False #If it exits now, it will re-enter immediately, which does not make sense

                if is_exit:
                    if self.use_rsi_to_exit and cur_data['over_bought'] and not cur_data['long_macd_long_exit_without_rsi']:
                        #if j >= 5 and (cur_data['close'] - cur_data['open'])/(cur_data['open'] - long_fire_data['open']) < 0.9:
                        #    is_exit = False

                        if (cur_data['open'] - long_fire_data['close'] > 0) and (cur_data['close'] - cur_data['open'])/(cur_data['open'] - long_fire_data['close']) < 0.9:
                            is_exit = False



                if is_exit:

                    if self.use_rsi_to_exit and cur_data['over_bought'] and not cur_data['long_macd_long_exit_without_rsi']:
                        exit_by_rsi = True
                        id_when_exit_by_rsi = long_start_id + j

                    if self.do_stop_loss and is_stop_loss:
                        exit_id = stop_loss_exit_id
                        exit_time = stop_loss_exit_time
                        exit_price = stop_loss_exit_price
                    else:
                        exit_id = cur_data['id']
                        exit_time = cur_data['time']

                        if self.do_stop_loss and exit_long_by_stop_loss:
                            exit_price = long_stop_loss_price
                        else:
                            exit_price = cur_data['close']


                    is_win = exit_price > entry_price


                    if self.do_stop_loss and exit_long_by_stop_loss:

                        if self.is_notify and (long_start_id + j == self.data_df.shape[0] - 1 or print_email_message_to_file):

                            if self.current_position > 0:

                                message_title = "Long position of " + str(self.current_position) + " units of " + self.currency + " closed at stop loss price " + str(exit_price)
                                message = "At current time" + str(exit_time + timedelta(hours = 1)) + " " + message_title

                                self.log_msg("message_title = " + message_title)
                                self.log_msg("message:")
                                self.log_msg(message)

                                if not print_email_message_to_file:
                                    sendEmail(message_title, message, is_alternative=self.is_alternative)
                                else:
                                    self.cache_email_messages(message_title, message, current_time)

                    else:

                        if self.is_notify and (long_start_id + j == self.data_df.shape[0] - 1 or print_email_message_to_file):

                            if self.current_position > 0:

                                if cur_data['over_bought'] and not cur_data['long_macd_long_exit_without_rsi']:
                                    prefix = "**Over Bought**"
                                else:
                                    prefix = ""

                                message_title = prefix + "Long position of " + str(self.current_position) + " units of " + self.currency + " closed by signal at price " + str(exit_price)
                                message = "At current time" + str(exit_time + timedelta(hours=1)) + " " + message_title

                                self.log_msg("message_title = " + message_title)
                                self.log_msg("message:")
                                self.log_msg(message)

                                if not print_email_message_to_file:
                                    sendEmail(message_title, message, is_alternative=self.is_alternative)
                                else:
                                    self.cache_email_messages(message_title, message, current_time)

                                if temporary_decision:
                                    self.temporary_close_long = True
                                    self.temporary_delta_position = abs(self.current_position)


                            if do_real_money_trading and self.wakeup == 1 and long_start_id + j == self.data_df.shape[0] - 1:
                                if self.current_real_position > 0 and self.close_long_order_id is None:

                                    if do_smart_execution:
                                        open_orders = self.coinbase_client.list_orders(order_status="OPEN").orders

                                        self.log_msg(f"open orders number = {len(open_orders)}")

                                        for order in open_orders:
                                            if str(order.product_id) == str(self.currency_coinbase):

                                                self.log_msg(f"order_id: {order.order_id}")
                                                self.log_msg(f"product_id: {order.product_id}")

                                                cancelled = False
                                                orderResponse = self.coinbase_client.get_order(order_id=str(order.order_id))
                                                if hasattr(orderResponse, "order"):
                                                    coinbaseorder = orderResponse.order
                                                    if coinbaseorder is not None:
                                                        status = coinbaseorder['status']
                                                        if status == 'CANCELLED':
                                                            cancelled = True

                                                if not cancelled:
                                                    try:
                                                        self.log_msg(f"Cancel pending order {str(order.order_id)} of {order.product_id}")
                                                        cancel_response = self.coinbase_client.cancel_orders(order_ids=[str(order.order_id)])
                                                        self.log_msg(cancel_response)
                                                    except Exception as e:
                                                        self.log_msg("Error:", e)

                                        if len(open_orders) > 0:
                                            time.sleep(2)


                                    try:
                                        self.log_msg("At " + str(exit_time) + ", close long position by placing real short order of " + str(self.current_real_position) + " at limit price " + str(self.crypto_last_price) + " to Coinbase with leverage " + str(default_leverage) + "x")
                                        client_order_id = f"order_{uuid.uuid4()}"
                                        response = self.coinbase_client.create_order(product_id=self.currency_coinbase,     #BTC-USDC is the correct product id
                                                                       client_order_id=client_order_id,
                                                                       side="SELL",
                                                                       order_configuration={
                                                                           "limit_limit_gtc":{
                                                                               "base_size" : str(self.current_real_position),
                                                                               "limit_price" : str(self.crypto_last_price)

                                                                           }
                                                                       },
                                                                       leverage=str(default_leverage),
                                                                       margin_type = "CROSS",
                                                                       retail_portfolio_id=self.coinbase_portfolio_id
                                                                       )
                                        self.log_msg(f"Order placed: {response}")
                                    except Exception as e:
                                        self.log_msg(f"Order failed: {e}")

                                    self.close_long_order_id = response['success_response']['order_id']
                                    self.close_long_attempt_size = self.current_real_position



                    if do_smart_execution:

                        if do_real_money_trading:
                            if self.wakeup == 1 and long_start_id + j == self.data_df.shape[0] - 1 and self.current_real_position > 0 and self.close_long_execution_order_id is None:

                                self.smart_executor_manager.close_executions(self.currency, self.current_real_position, exit_time, self.crypto_last_price)

                                self.close_long_execution_order_id = self.close_long_order_id


                        else:
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
                            + ([total_strategy_pnl] if do_smart_execution and not do_real_money_trading else [])]


        if do_smart_execution and not do_real_money_trading:
            self.long_strategy_df = pd.DataFrame(data = long_strategy_records, columns = strategy_record_columns)
            self.long_strategy_execution_df = pd.DataFrame(data = long_strategy_execution_records, columns = strategy_execution_record_columns)

        long_df = pd.DataFrame(data = result_data, columns = result_columns)

        if not (do_smart_execution and not do_real_money_trading):
            long_df['pnl'] = np.where(
                long_df['entry_time'].notnull() & long_df['exit_time'].notnull(),
                (long_df['exit_price'] - long_df['entry_price']) / long_df['entry_price'] * self.init_entry_value * default_leverage,
                0
            )
            #long_df['pnl'] = (long_df['exit_price'] - long_df['entry_price']) / long_df['entry_price'] * initial_entry_value * default_leverage


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
        self.log_msg("")
        self.log_msg("Calculating Short positions.............")
        self.log_msg("")

        short_start_ids = which(self.data_df['macd_short_enter'])

        #print("short_start_ids:")
        #print(short_start_ids[-5:])

        is_effective = [1] * len(short_start_ids)

        short_trade_id = 0

        if self.use_rsi_to_exit:
            exit_by_rsi = False

        for i in range(len(short_start_ids)):

            if is_effective[i] == 0:
                self.data_df.at[short_start_ids[i], 'macd_short_enter'] = False
                # print("Here critical 1:")
                # print("short_start_id = " + str(short_start_ids[i]))
                continue

            temp_i = i
            short_start_id = short_start_ids[i]
            short_fire_data = self.data_df.iloc[short_start_id]

            if self.use_rsi_to_exit and exit_by_rsi:
                if short_fire_data['long_macd_short_enter_too_late'] and short_start_id > 0 and not self.data_df.iloc[short_start_id-1]['guppy_all_strong_aligned_long']:
                    self.data_df.at[short_start_ids[i], 'macd_short_enter'] = False
                    # print("Here critical 2:")
                    # print("short_start_id = " + str(short_start_ids[i]))
                    continue
                else:
                    exit_by_rsi = False


            instrument = short_fire_data['currency']
            entry_time = short_fire_data['time']
            entry_price = short_fire_data['close']
            entry_id = short_fire_data['id']

            if self.do_stop_loss:
                short_stop_loss_price = short_fire_data['short_critical_price']

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

                    position = -self.init_entry_value/entry_price * default_leverage
                    # if entry_price >= 1:
                    #     position = round(position, 3)
                    # else:
                    #     position = int(round(position, 0))

                    position = round(position, self.coinbase_decimal)
                    if self.coinbase_decimal == 0:
                        position = int(position)

                    #delta_position = position - self.current_position
                    delta_position = position

                    #self.current_position = position


                    message_title = "Short " + self.currency + " " + str(-delta_position) + " units"

                    print("entry_price:")
                    print(entry_price)
                    message = "At " + current_time + ", short " + self.currency + " roughly " + str(-delta_position) + " units at entry price " + str(round(entry_price, self.decimal)) + "\n"
                    message += "This makes it now at a short position of " + str(position) + " units with an actual notional of " + str(self.init_entry_value) + " dollar\n"

                    self.log_msg("message_title = " + message_title)
                    self.log_msg("message:")
                    self.log_msg(message)

                    if not print_email_message_to_file:
                        sendEmail(message_title, message, is_alternative=self.is_alternative)
                    else:
                        self.cache_email_messages(message_title, message, current_time)

                    if temporary_decision:
                        self.temporary_short = True
                        self.temporary_delta_position = abs(delta_position)


                    if do_real_money_trading and self.wakeup == 1 and short_start_id == self.data_df.shape[0] - 1:
                        if self.current_real_position >= 0 and self.short_order_id is None:

                            if do_smart_execution:
                                real_position = -self.init_entry_value/self.crypto_last_price * self.average_leverage
                            else:
                                real_position = -self.init_entry_value/self.crypto_last_price * default_leverage

                            # if self.crypto_last_price >= 1:
                            #     real_position = round(real_position, 3)
                            # else:
                            #     real_position = int(round(real_position, 0))

                            real_position = round(real_position, self.coinbase_decimal)
                            if self.coinbase_decimal == 0:
                                real_position = int(real_position)



                            self.log_msg(self.currency + " current real position = " + str(self.current_real_position))
                            self.log_msg(self.currency + " target real position = " + str(real_position))
                            #real_delta_position = real_position - self.current_real_position
                            real_delta_position = real_position

                            try:
                                self.log_msg("At " + current_time + ", open short position by placing real short order of " + str(-real_delta_position) + " at limit price " + str(self.crypto_last_price) + " to Coinbase with leverage " + str(default_leverage) + "x")
                                client_order_id = f"order_{uuid.uuid4()}"
                                response = self.coinbase_client.create_order(product_id=self.currency_coinbase,     #BTC-USDC is the correct product id
                                                               client_order_id=client_order_id,
                                                               side="SELL",
                                                               order_configuration={
                                                                   "limit_limit_gtc":{
                                                                       "base_size" : str(-real_delta_position),
                                                                       "limit_price" : str(self.crypto_last_price)

                                                                   }
                                                               },
                                                               leverage=str(default_leverage),
                                                               margin_type = "CROSS",
                                                               retail_portfolio_id=self.coinbase_portfolio_id
                                                               )
                                self.log_msg(f"Order placed: {response}")
                            except Exception as e:
                                self.log_msg(f"Order failed: {e}")

                            self.short_order_id = response['success_response']['order_id']
                            self.short_attempt_size = -real_delta_position


                            if self.do_stop_loss:
                                try:
                                    self.log_msg("At " + current_time + ", place real short stop loss order of " + str(-real_position) + " at stop loss price " + str(
                                        short_stop_loss_price))

                                    stop_loss_order_id = f"order_{uuid.uuid4()}"
                                    response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                                   # BTC-USDC is the correct product id
                                                                   client_order_id=stop_loss_order_id,
                                                                   side="BUY",
                                                                   order_configuration={
                                                                       "stop_limit_stop_limit_gtc": {
                                                                           "base_size": str(-real_position),
                                                                           "limit_price": str(short_stop_loss_price*1.1),
                                                                           "stop_price": str(short_stop_loss_price)
                                                                       }
                                                                   },
                                                                   leverage="10",
                                                                   margin_type="CROSS"
                                                                   # retail_portfolio_id="0194271a-bd95-7ba7-a028-6561a970128b"
                                                                   )

                                except Exception as e:
                                    self.log_msg(f"Order failed: {e}")

                            # self.short_order_id = response['success_response']['order_id']
                            # self.short_attempt_size = -real_delta_position


            if do_smart_execution:
                strategy_executions = []

                if do_real_money_trading:
                    if self.wakeup == 1 and short_start_id == self.data_df.shape[0] - 1 and self.current_real_position >= 0 and self.short_execution_order_id is None:

                        prod_sizes = -real_delta_position * self.distribution

                        self.log_msg('[Execution] Short position to open  = ' + str(real_delta_position))
                        self.log_msg('[Execution] distribution = ' + str(self.distribution))
                        self.log_msg('[Execution] prod sizes = ' + str(prod_sizes))

                        entry_value = self.init_entry_value/(len(self.leverage) * 2) if use_extra_execution else self.init_entry_value/len(self.leverage)
                        self.log_msg('[Execution] entry_value = ' + str(entry_value))

                        for k in range(len(self.leverage)):
                            strategy_execution = StrategyExecution(side = -1, leverage = self.leverage[k], take_profit_pct = self.take_profit_pct[k], take_loss_pct = self.take_loss_pct[k],
                                                                   strategy_id = k+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = self.crypto_last_price,
                                                                   execution_entry_time = entry_time, execution_entry_price = self.crypto_last_price,
                                                                   strategy_entry_value = entry_value, execution_entry_value = entry_value,
                                                                   default_leverage=default_leverage, prod_size = round(prod_sizes[k], self.coinbase_decimal))
                            strategy_executions += [strategy_execution]


                        if use_extra_execution:

                            extra_prod_size = -real_delta_position * self.extra_distribution

                            extra_entry_value = self.init_entry_value / 2.0

                            self.log_msg('[Execution] extra_distribution = ' + str(self.extra_distribution))
                            self.log_msg('[Execution] extr_prod_size = ' + str(extra_prod_size))
                            self.log_msg('[Execution] extra_entry_value = ' + str(extra_entry_value))

                            strategy_execution = StrategyExecution(side = -1, leverage = self.leverage[0], take_profit_pct = self.take_profit_pct[0], take_loss_pct = self.take_loss_pct[0],
                                                                       strategy_id = len(self.leverage)+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = self.crypto_last_price,
                                                                       execution_entry_time = entry_time, execution_entry_price = self.crypto_last_price,
                                                                       strategy_entry_value = extra_entry_value, execution_entry_value = extra_entry_value, default_leverage=default_leverage,
                                                                       prod_size = round(extra_prod_size, self.coinbase_decimal))

                            strategy_executions += [strategy_execution]



                        self.smart_executor_manager.open_executions(currency = self.currency, target_position = real_delta_position,
                                                                    entry_time = entry_time, strategy_executions = strategy_executions
                                                                    )

                        self.short_execution_order_id = self.short_order_id

                else:

                    for k in range(len(self.leverage)):
                        strategy_execution = StrategyExecution(side = -1, leverage = self.leverage[k], take_profit_pct = self.take_profit_pct[k], take_loss_pct = self.take_loss_pct[k],
                                                               strategy_id = k+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                               execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                               strategy_entry_value = self.each_strategy_entry_value, execution_entry_value = self.each_strategy_entry_value, default_leverage=default_leverage)
                        strategy_executions += [strategy_execution]

                    #This is the extra one
                    if use_extra_execution:
                        strategy_execution = StrategyExecution(side = -1, leverage = self.leverage[0], take_profit_pct = self.take_profit_pct[0], take_loss_pct = self.take_loss_pct[0],
                                                                   strategy_id = len(self.leverage)+1, execution_id = 1, strategy_entry_time = entry_time, strategy_entry_price = entry_price,
                                                                   execution_entry_time = entry_time, execution_entry_price = entry_price,
                                                                   strategy_entry_value = self.init_entry_value/2.0, execution_entry_value = self.init_entry_value/2.0, default_leverage=default_leverage)

                        strategy_executions += [strategy_execution]


                    total_strategy_pnl = 0


            if self.do_stop_loss:
                exit_short_by_stop_loss = False

                is_stop_loss = False
                stop_loss_exit_id = -1
                stop_loss_exit_time = None
                stop_loss_exit_price = -1

            while short_start_id + j < self.data_df.shape[0]:

                #cur_data = self.data_df.iloc[short_start_id + j]

                #self.log_msg("")
                #self.log_msg("Short 1h time = " + str(self.data_df.iloc[short_start_id + j]['time']) + '..............................')

                if do_smart_execution and not do_real_money_trading:

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

                        #self.log_msg("    Short 5min time = " + str(cur_data['time']))

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
                                #     self.log_msg("short_trade_id = " + str(short_trade_id))
                                #     self.log_msg("strategy_id = " + str(execution.strategy_id))
                                #     self.log_msg("execution_id = " + str(execution.execution_id))
                                #
                                #     self.log_msg("cur time = " + str(cur_data['time']))
                                #     self.log_msg("execution_entry_time = " + str(execution.execution_entry_time))
                                #     self.log_msg("execution_entry_price = " + str(execution.execution_entry_price))
                                #     self.log_msg("open = " + str(cur_data['open']))
                                #     self.log_msg("loss_ref_price = " + str(loss_ref_price))
                                #     self.log_msg("take_loss_price = " + str(execution.take_loss_price))
                                #     self.log_msg("strategy_entry_price = " + str(execution.strategy_entry_price))
                                #     self.log_msg("hit_stop_loss = " + str(hit_stop_loss))
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
                                                                       strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value, default_leverage=default_leverage
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
                                #     self.log_msg("Last record so far:")
                                #     self.log_msg(short_strategy_execution_records[-1])


                                if k < len(strategy_executions) - 1 if use_extra_execution else len(strategy_executions):
                                    next_execution = StrategyExecution(side=execution.side, leverage=execution.leverage, take_profit_pct=execution.take_profit_pct,
                                                                   take_loss_pct=execution.take_loss_pct, strategy_id=execution.strategy_id, execution_id=execution.execution_id+1,
                                                                   strategy_entry_time=execution.strategy_entry_time, strategy_entry_price=execution.strategy_entry_price,
                                                                   execution_entry_time=cur_data['time'], execution_entry_price=execution.execution_exit_price,
                                                                   strategy_entry_value=execution.strategy_entry_value, execution_entry_value=execution.execution_exit_value, default_leverage=default_leverage
                                                                   )
                                    strategy_executions[k] = next_execution
                                    execution = strategy_executions[k]
                                    #hit_stop_profit = cur_data['low'] <= execution.take_profit_price
                                else:
                                    break


                if (not (do_smart_execution and not do_real_money_trading)) or can_use_5min:
                    cur_data = self.data_df.iloc[short_start_id + j]

                is_exit = False

                if self.do_stop_loss and not exit_short_by_stop_loss:
                    if cur_data['high'] > short_stop_loss_price + (1e-6) and cur_data['group_index'] >= 2:
                        if self.reentry_after_stop_loss:
                            is_exit = True
                        else:
                            is_stop_loss = True
                            stop_loss_exit_id = cur_data['id']
                            stop_loss_exit_time = cur_data['time']
                            stop_loss_exit_price = short_stop_loss_price

                        exit_short_by_stop_loss = True


                if not is_exit:
                    if long_macd_indicate_short or (not is_short_macd_fire):
                        is_exit = cur_data['long_macd_short_exit'] or cur_data['macd_long_enter']
                        #is_exit = cur_data['long_macd_short_exit'] or cur_data['short_macd_long_enter']
                    elif enable_short_macd_signal:
                        is_exit = cur_data['short_macd_short_exit'] or cur_data['macd_long_enter']
                        #is_exit = cur_data['short_macd_short_exit'] or cur_data['short_macd_long_enter']

                # if self.use_rsi_to_exit and cur_data['over_sold'] and not cur_data['long_macd_short_exit_without_rsi']:
                #     if not cur_data['long_macd_short_enter_too_late']:
                #         is_exit = False

                if is_exit:

                    if self.use_rsi_to_exit and cur_data['over_sold'] and not cur_data['long_macd_short_exit_without_rsi']:
                        exit_by_rsi = True

                    if self.do_stop_loss and is_stop_loss:
                        exit_id = stop_loss_exit_id
                        exit_time = stop_loss_exit_time
                        exit_price = stop_loss_exit_price
                    else:
                        exit_id = cur_data['id']
                        exit_time = cur_data['time']

                        if self.do_stop_loss and exit_short_by_stop_loss:
                            exit_price = short_stop_loss_price
                        else:
                            exit_price = cur_data['close']

                    is_win = exit_price < entry_price

                    # if short_start_id + j == self.data_df.shape[0] - 1:
                    #     print("Reach end trade")
                    #     print("current_position = " + str(self.current_position))

                    if self.do_stop_loss and exit_long_by_stop_loss:

                        if self.is_notify and (short_start_id + j == self.data_df.shape[0] - 1 or print_email_message_to_file):
                            if self.current_position < 0:
                                message_title = "Short position of " + str(-self.current_position) + " units of " + self.currency + " closed at stop loss price " + str(exit_price)
                                message = "At current time" + str(exit_time + timedelta(hours=1)) + " " + message_title

                                self.log_msg("message_title = " + message_title)
                                self.log_msg("message:")
                                self.log_msg(message)

                                if not print_email_message_to_file:
                                    sendEmail(message_title, message, is_alternative=self.is_alternative)
                                else:
                                    self.cache_email_messages(message_title, message, current_time)

                    else:

                        if self.is_notify and (short_start_id + j == self.data_df.shape[0] - 1 or print_email_message_to_file):

                            if self.current_position < 0:

                                if cur_data['over_sold'] and not cur_data['long_macd_short_exit_without_rsi']:
                                    prefix = "**Over Sold**"
                                else:
                                    prefix = ""

                                message_title = prefix + "Short position of " + str(-self.current_position) + " units of " + self.currency + " closed by signal at price " + str(exit_price)
                                message = "At current time" + str(exit_time + timedelta(hours=1)) + " " + message_title

                                self.log_msg("message_title = " + message_title)
                                self.log_msg("message:")
                                self.log_msg(message)

                                if not print_email_message_to_file:
                                    sendEmail(message_title, message, is_alternative=self.is_alternative)
                                else:
                                    self.cache_email_messages(message_title, message, current_time)

                                if temporary_decision:
                                    self.temporary_close_short = True
                                    self.temporary_delta_position = abs(self.current_position)


                            if do_real_money_trading and self.wakeup == 1 and short_start_id + j == self.data_df.shape[0] - 1:
                                if self.current_real_position < 0 and self.close_short_order_id is None:

                                    if do_smart_execution:
                                        open_orders = self.coinbase_client.list_orders(order_status="OPEN").orders

                                        self.log_msg(f"open orders number = {len(open_orders)}")

                                        for order in open_orders:

                                            if str(order.product_id) == str(self.currency_coinbase):

                                                self.log_msg(f"order_id: {order.order_id}")
                                                self.log_msg(f"product_id: {order.product_id}")

                                                cancelled = False
                                                orderResponse = self.coinbase_client.get_order(order_id=str(order.order_id))
                                                if hasattr(orderResponse, "order"):
                                                    coinbaseorder = orderResponse.order
                                                    if coinbaseorder is not None:
                                                        status = coinbaseorder['status']
                                                        if status == 'CANCELLED':
                                                            cancelled = True

                                                if not cancelled:
                                                    try:
                                                        self.log_msg(f"Cancel pending order {str(order.order_id)} of {order.product_id}")
                                                        cancel_response = self.coinbase_client.cancel_orders(order_ids=[str(order.order_id)])
                                                        self.log_msg(cancel_response)
                                                    except Exception as e:
                                                        self.log_msg("Error:", e)

                                        if len(open_orders) > 0:
                                            time.sleep(2)

                                    try:
                                        self.log_msg("At " + str(exit_time) + ", close short position by placing real long order of " + str(-self.current_real_position) + " at limit price " + str(self.crypto_last_price) + " to Coinbase with leverage " + str(default_leverage) + "x")
                                        client_order_id = f"order_{uuid.uuid4()}"
                                        response = self.coinbase_client.create_order(product_id=self.currency_coinbase,     #BTC-USDC is the correct product id
                                                                       client_order_id=client_order_id,
                                                                       side="BUY",
                                                                       order_configuration={
                                                                           "limit_limit_gtc":{
                                                                               "base_size" : str(-self.current_real_position),
                                                                               "limit_price" : str(self.crypto_last_price)

                                                                           }
                                                                       },
                                                                       leverage=str(default_leverage),
                                                                       margin_type = "CROSS",
                                                                       retail_portfolio_id=self.coinbase_portfolio_id
                                                                       )
                                        self.log_msg(f"Order placed: {response}")
                                    except Exception as e:
                                        self.log_msg(f"Order failed: {e}")

                                    self.close_short_order_id = response['success_response']['order_id']
                                    self.close_short_attempt_size = -self.current_real_position



                    if do_smart_execution:
                        if do_real_money_trading:

                            if self.wakeup == 1 and short_start_id + j == self.data_df.shape[0] - 1 and self.current_real_position < 0 and self.close_short_execution_order_id is None:

                                self.smart_executor_manager.close_executions(self.currency, self.current_real_position, exit_time, self.crypto_last_price)

                                self.close_short_execution_order_id = self.close_short_order_id


                        else:
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
                            + ([total_strategy_pnl] if do_smart_execution and not do_real_money_trading else [])]


        if do_smart_execution and not do_real_money_trading:
            self.short_strategy_df = pd.DataFrame(data = short_strategy_records, columns = strategy_record_columns)
            self.short_strategy_execution_df = pd.DataFrame(data = short_strategy_execution_records, columns = strategy_execution_record_columns)


        short_df = pd.DataFrame(data=result_data, columns=result_columns)

        if not (do_smart_execution and not do_real_money_trading):
            short_df['pnl'] = np.where(
                short_df['entry_time'].notnull() & short_df['exit_time'].notnull(),
                -(short_df['exit_price'] - short_df['entry_price']) / short_df['entry_price'] * self.init_entry_value * default_leverage,
                0
            )
            #short_df['pnl'] = -(short_df['exit_price'] - short_df['entry_price']) / short_df['entry_price'] * initial_entry_value  * default_leverage

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
            self.log_msg("Performance Summary")
            self.log_msg(self.full_summary_df)

        self.write_long_df = write_long_df
        self.write_short_df = write_short_df

        if enable_short_macd_signal:
            self.macd_group_summary_df = macd_group_summary_df
            self.critical_value_data_df = critical_value_data_df






        #########################




    def cache_email_messages(self, title, content, time):

        self.email_message_caches += [[title, content, time]]


    def post_processing(self):

        self.log_msg("")
        self.log_msg("Post processing currency pair " + str(self.currency))

        if print_email_message_to_file:

            email_messages_df = pd.DataFrame(data = self.email_message_caches, columns = ['title', 'content', 'time'])
            email_messages_df = email_messages_df.sort_values(by = ['time'])


            for i in range(email_messages_df.shape[0]):

                email_data_entry = email_messages_df.iloc[i]
                self.log_msg(str(i+1) + ":", file = self.email_message_fd)
                self.log_msg(email_data_entry['title'], file = self.email_message_fd)
                self.log_msg(email_data_entry['content'], file = self.email_message_fd)
                #self.email_message_fd.flush()
                self.log_msg("", file = self.email_message_fd)
                self.log_msg("", file = self.email_message_fd)

            self.email_message_fd.close()


        if do_smart_execution and not do_real_money_trading:

            strategy_df = pd.concat([self.long_strategy_df, self.short_strategy_df])
            strategy_execution_df = pd.concat([self.long_strategy_execution_df, self.short_strategy_execution_df])

            strategy_df = strategy_df.sort_values(by = ['entry_time', 'exit_time'], ascending = True)
            strategy_execution_df = strategy_execution_df.sort_values(by = ['entry_time', 'exit_time'], ascending = True)


        if production_running and not (do_smart_execution and not do_real_money_trading):
            if self.long_existing_df is not None and 'prod_entry_price' in self.long_existing_df.columns and 'prod_exit_price' in self.long_existing_df.columns:

                if do_smart_execution and do_real_money_trading:

                    write_long_df_temp = self.write_long_df.copy()
                    write_long_df_temp = write_long_df_temp.drop(columns = ['pnl'])
                    write_long_prod_df = pd.merge(write_long_df_temp, self.long_existing_df[['long_trade_id', 'pnl', 'prod_entry_price', 'prod_exit_price', 'prod_size', 'is_prod', 'prod_pnl']],
                                                  on=['long_trade_id'], how='left')
                else:
                    write_long_prod_df = pd.merge(self.write_long_df, self.long_existing_df[['long_trade_id', 'prod_entry_price', 'prod_exit_price', 'prod_size', 'is_prod']],
                                              on = ['long_trade_id'], how = 'left')
            else:
                write_long_prod_df = self.write_long_df.copy()

            # print("here1:")
            # print(write_long_prod_df.iloc[-3:])

            if self.long_existing_df is None or 'prod_entry_price' not in self.long_existing_df.columns:
                write_long_prod_df['prod_entry_price'] = write_long_prod_df['entry_price']
                write_long_prod_df['prod_exit_price'] = write_long_prod_df['exit_price']

                write_long_prod_df['prod_size'] = self.init_entry_value / write_long_prod_df['entry_price'] * default_leverage
                write_long_prod_df['is_prod'] = 0

            # print("here2:")
            # print(write_long_prod_df.iloc[-3:])

            for col in ['prod_entry_price', 'prod_exit_price']:
                write_long_prod_df[col] = np.where(
                    (write_long_prod_df[col].isnull()) | (write_long_prod_df[col] <= 0),
                    write_long_prod_df[col[len('prod_'):]],
                    write_long_prod_df[col]
                )

            # print("here3:")
            # print(write_long_prod_df.iloc[-3:])

            write_long_prod_df['prod_size'] = np.where(
                (write_long_prod_df['prod_size'].isnull()) | (write_long_prod_df['prod_size'] <= 0),
                self.init_entry_value / write_long_prod_df['entry_price'] * default_leverage,
                write_long_prod_df['prod_size']
            )

            # print("here4:")
            # print(write_long_prod_df.iloc[-3:])

            write_long_prod_df['is_prod'] = np.where(
                write_long_prod_df['is_prod'].isnull(),
                0,
                write_long_prod_df['is_prod']
            )


            if self.long_order_fill_price > 0:
                self.log_msg("Write long_order_fill_price = " + str(self.long_order_fill_price))
                write_long_prod_df.at[write_long_prod_df.shape[0] - 1, 'prod_entry_price'] = round(self.long_order_fill_price, self.decimal)
                write_long_prod_df.at[write_long_prod_df.shape[0] - 1, 'prod_size'] = self.long_order_fill_size
                write_long_prod_df.at[write_long_prod_df.shape[0] - 1, 'prod_exit_price'] = -1
                write_long_prod_df.at[write_long_prod_df.shape[0] - 1, 'is_prod'] = 1
                self.reset_long_fill()

            if self.close_long_order_fill_price > 0:
                self.log_msg("Write close_long_order_fill_price = " + str(self.close_long_order_fill_price))
                write_long_prod_df.at[write_long_prod_df.shape[0] - 1, 'prod_exit_price'] = round(self.close_long_order_fill_price, self.decimal)
                write_long_prod_df.at[write_long_prod_df.shape[0] - 1, 'prod_size'] = self.close_long_order_fill_size
                write_long_prod_df.at[write_long_prod_df.shape[0] - 1, 'is_prod'] = 1
                self.reset_close_long_fill()



            if self.short_existing_df is not None and 'prod_entry_price' in self.short_existing_df.columns and 'prod_exit_price' in self.short_existing_df.columns:

                if do_smart_execution and do_real_money_trading:
                    write_short_df_temp = self.write_short_df.copy()
                    write_short_df_temp = write_short_df_temp.drop(columns = ['pnl'])
                    write_short_prod_df = pd.merge(write_short_df_temp, self.short_existing_df[['short_trade_id', 'pnl', 'prod_entry_price', 'prod_exit_price', 'prod_size', 'is_prod', 'prod_pnl']],
                                                      on=['short_trade_id'], how='left')

                else:
                    write_short_prod_df = pd.merge(self.write_short_df, self.short_existing_df[['short_trade_id', 'prod_entry_price', 'prod_exit_price', 'prod_size', 'is_prod']],
                                              on = ['short_trade_id'], how = 'left')
            else:
                write_short_prod_df = self.write_short_df.copy()

            #print("write_short_prod_df:")
            #print(write_short_prod_df.iloc[0:10])

            if self.short_existing_df is None or 'prod_entry_price' not in self.short_existing_df.columns:
                write_short_prod_df['prod_entry_price'] = write_short_prod_df['entry_price']
                write_short_prod_df['prod_exit_price'] = write_short_prod_df['exit_price']

                write_short_prod_df['prod_size'] = self.init_entry_value / write_short_prod_df['entry_price'] * default_leverage
                write_short_prod_df['is_prod'] = 0



            for col in ['prod_entry_price', 'prod_exit_price']:
                write_short_prod_df[col] = np.where(
                    (write_short_prod_df[col].isnull()) | (write_short_prod_df[col] <= 0),
                    write_short_prod_df[col[len('prod_'):]],
                    write_short_prod_df[col]
                )

            write_short_prod_df['prod_size'] = np.where(
                (write_short_prod_df['prod_size'].isnull()) | (write_short_prod_df['prod_size'] <= 0),
                self.init_entry_value / write_short_prod_df['entry_price'] * default_leverage,
                write_short_prod_df['prod_size']
            )

            write_short_prod_df['is_prod'] = np.where(
                write_short_prod_df['is_prod'].isnull(),
                0,
                write_short_prod_df['is_prod']
            )


            if self.short_order_fill_price > 0:
                self.log_msg("Write short_order_fill_price = " + str(self.short_order_fill_price))
                write_short_prod_df.at[write_short_prod_df.shape[0] - 1, 'prod_entry_price'] = round(self.short_order_fill_price, self.decimal)
                write_short_prod_df.at[write_short_prod_df.shape[0] - 1, 'prod_size'] = self.short_order_fill_size
                write_short_prod_df.at[write_short_prod_df.shape[0] - 1, 'prod_exit_price'] = -1
                write_short_prod_df.at[write_short_prod_df.shape[0] - 1, 'is_prod'] = 1
                self.reset_short_fill()

            if self.close_short_order_fill_price > 0:
                self.log_msg("Write close_short_order_fill_price = " + str(self.close_short_order_fill_price))
                write_short_prod_df.at[write_short_prod_df.shape[0] - 1, 'prod_exit_price'] = round(self.close_short_order_fill_price, self.decimal)
                write_short_prod_df.at[write_short_prod_df.shape[0] - 1, 'prod_size'] = self.close_short_order_fill_size
                write_short_prod_df.at[write_short_prod_df.shape[0] - 1, 'is_prod'] = 1
                self.reset_close_short_fill()


            if do_smart_execution and do_real_money_trading and 'prod_pnl' in write_long_prod_df.columns:
                write_long_prod_df['prod_pnl'] = np.where(
                    (write_long_prod_df['prod_entry_price'] > 0) & (write_long_prod_df['prod_exit_price'] > 0),
                    np.where(
                        write_long_prod_df['is_prod'] == 0,
                        (write_long_prod_df['prod_exit_price'] - write_long_prod_df['prod_entry_price'])/write_long_prod_df['prod_entry_price'] * self.init_entry_value * default_leverage,
                        write_long_prod_df['prod_pnl']
                    ),
                    0
                )

            else:
                write_long_prod_df['prod_pnl'] = np.where(
                    (write_long_prod_df['prod_entry_price'] > 0) & (write_long_prod_df['prod_exit_price'] > 0),
                    np.where(
                        write_long_prod_df['is_prod'] == 0,
                        (write_long_prod_df['prod_exit_price'] - write_long_prod_df['prod_entry_price'])/write_long_prod_df['prod_entry_price'] * self.init_entry_value * default_leverage,
                        (write_long_prod_df['prod_exit_price'] - write_long_prod_df['prod_entry_price']) *write_long_prod_df['prod_size']
                    ),
                    0
                )

            write_long_prod_df['prod_size'] = np.where(
                write_long_prod_df['prod_entry_price'] > 1,
                write_long_prod_df['prod_size'].apply(lambda x: round(x, 3)),
                write_long_prod_df['prod_size'].apply(lambda x: int(round(x, 0)))
            )


            if do_smart_execution and do_real_money_trading and 'prod_pnl' in write_short_prod_df.columns:
                write_short_prod_df['prod_pnl'] = np.where(
                    (write_short_prod_df['prod_entry_price'] > 0) & (write_short_prod_df['prod_exit_price'] > 0),
                    np.where(
                        write_short_prod_df['is_prod'] == 0,
                        -(write_short_prod_df['prod_exit_price'] - write_short_prod_df['prod_entry_price'])/write_short_prod_df['prod_entry_price'] * self.init_entry_value * default_leverage,
                        write_short_prod_df['prod_pnl']
                    ),
                    0
                )

            else:
                write_short_prod_df['prod_pnl'] = np.where(
                    (write_short_prod_df['prod_entry_price'] > 0) & (write_short_prod_df['prod_exit_price'] > 0),
                    np.where(
                        write_short_prod_df['is_prod'] == 0,
                        -(write_short_prod_df['prod_exit_price'] - write_short_prod_df['prod_entry_price'])/write_short_prod_df['prod_entry_price'] * self.init_entry_value * default_leverage,
                        -(write_short_prod_df['prod_exit_price'] - write_short_prod_df['prod_entry_price']) * write_short_prod_df['prod_size']
                    ),
                    0
                )

            write_short_prod_df['prod_size'] = np.where(
                write_short_prod_df['prod_entry_price'] > 1,
                write_short_prod_df['prod_size'].apply(lambda x: round(x, 3)),
                write_short_prod_df['prod_size'].apply(lambda x: int(round(x, 0)))
            )

            write_prod_df = pd.concat([write_long_prod_df, write_short_prod_df])

            write_prod_df = write_prod_df.sort_values(by = ['entry_time'], ascending = True)

            write_prod_df['cum_pnl'] = write_prod_df['pnl'].cumsum()

            write_prod_df['prod_pnl'] = write_prod_df['prod_pnl'].apply(lambda x: round(x, 2))
            write_prod_df['prod_cum_pnl'] = write_prod_df['prod_pnl'].cumsum()

            write_prod_df['cum_pnl'] = write_prod_df['cum_pnl'].apply(lambda x: round(x, 2))
            write_prod_df['prod_cum_pnl'] = write_prod_df['prod_cum_pnl'].apply(lambda x: round(x, 2))

            write_prod_df['prod_entry_price'] = write_prod_df['prod_entry_price'].apply(lambda x: round(x, self.decimal))
            write_prod_df['prod_exit_price'] = write_prod_df['prod_exit_price'].apply(lambda x: round(x, self.decimal))

            write_prod_df['entry_price'] = write_prod_df['entry_price'].apply(lambda x: round(x, self.decimal))
            write_prod_df['exit_price'] = write_prod_df['exit_price'].apply(lambda x: round(x, self.decimal))

            write_prod_df['execution_slippage'] = write_prod_df['prod_pnl'] - write_prod_df['pnl']
            write_prod_df['cum_execution_slippage'] = write_prod_df['execution_slippage'].cumsum()

            write_prod_df['execution_slippage'] = write_prod_df['execution_slippage'].apply(lambda x: round(x, 2))
            write_prod_df['cum_execution_slippage'] = write_prod_df['cum_execution_slippage'].apply(lambda x: round(x, 2))


            columns_to_delete = [col for col in ['execution_cost', 'cum_execution_cost'] if col in write_prod_df.columns]
            write_prod_df = write_prod_df.drop(columns = columns_to_delete)


            write_prod_df.to_csv(self.trade_prod_file, index = False)


            if len(self.delay_cost_data) > 0:
                new_delay_cost_df = pd.DataFrame(data = self.delay_cost_data, columns = ['type', 'time', 'entry_price', 'exit_price', 'size'])
                if self.delay_cost_df is None:
                    self.delay_cost_df = new_delay_cost_df
                else:
                    self.delay_cost_df = pd.concat([self.delay_cost_df, new_delay_cost_df])

                self.delay_cost_df['side'] = np.where(
                    self.delay_cost_df['type'].isin(['revoke long', 'revoke close short']),
                    1, -1
                )

                for col in ['entry_price', 'exit_price']:
                    self.delay_cost_df[col] = self.delay_cost_df[col].apply(lambda x: round(x, self.decimal))

                self.delay_cost_df['pnl'] = self.delay_cost_df['side'] * self.delay_cost_df['size'] * (self.delay_cost_df['exit_price'] - self.delay_cost_df['entry_price'])
                self.delay_cost_df['pnl'] = self.delay_cost_df['pnl'].apply(lambda x: round(x, self.decimal))

                self.delay_cost_df['cum_pnl'] = self.delay_cost_df['pnl'].cumsum()


                self.delay_cost_df.to_csv(self.delay_cost_file, index = False)





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

        # self.log_msg("write_df:")
        # self.log_msg(write_df.iloc[0:10])

        #if self.is_finalized:
        self.data_df.to_csv(self.data_file, index = False)

        if self.data_df_5min is not None and self.data_file_5min is not None:
            print("Final data_df_5min row_num = " + str(self.data_df_5min.shape[0]))
            self.data_df_5min.to_csv(self.data_file_5min, index = False)

        self.data_df.iloc[-1:][['currency','time', 'open', 'high', 'low', 'close']].to_csv(self.data_file[:-len('.csv')] + '_lastRow.csv', index = False)

        if enable_short_macd_signal:
            self.macd_group_summary_df.to_csv(self.data_file[:-len('.csv')] + '_macd_group_summary.csv', index = False)
            self.critical_value_data_df.to_csv(self.data_file[:-len('.csv')] + '_critial_value.csv', index = False)

        if self.do_stop_loss:
            self.group_summary_df.to_csv(self.data_file[:-len('.csv')] + '_group_summary.csv', index=False)
            self.critical_price_data_df.to_csv(self.data_file[:-len('.csv')] + '_critial_prices.csv', index=False)


        #write_df['id'] = list(range(write_df.shape[0]))

        write_df['cum_pnl'] = write_df['pnl'].cumsum()

        write_df['entry_price'] = write_df['entry_price'].apply(lambda x: round(x, self.decimal))
        write_df['exit_price'] = write_df['exit_price'].apply(lambda x: round(x, self.decimal))
        write_df['cum_pnl'] = write_df['cum_pnl'].apply(lambda x: round(x, 2))

        if do_smart_execution and not do_real_money_trading:
            for col in ['entry_price', 'exit_price']:
                strategy_df[col] = strategy_df[col].apply(lambda x: round(x, self.decimal))

            for col in ['exit_value', 'pnl']:
                strategy_df[col] = strategy_df[col].apply(lambda x: round(x, 2))

            for col in ['take_profit_price', 'take_loss_price',
                        'entry_price', 'exit_price']:
                strategy_execution_df[col] = strategy_execution_df[col].apply(lambda x: round(x, self.decimal))

            for col in ['entry_value', 'exit_value', 'pnl']:
                strategy_execution_df[col] = strategy_execution_df[col].apply(lambda x: round(x, 2))


        total_return_rate = write_df.iloc[-1]['cum_pnl']/self.init_entry_value
        max_draw_down, start_draw_down, end_draw_down = self.calc_max_drawdown(write_df['cum_pnl'])
        max_draw_down = max_draw_down/self.init_entry_value
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

            self.log_msg("trade_file: " + str(self.trade_file))
            write_df.to_csv(self.trade_file, index = False)

            self.log_msg("performance_file: " + str(self.performance_file))
            self.full_summary_df.to_csv(self.performance_file, index = False)

            if do_smart_execution and not do_real_money_trading:
                strategy_df.to_csv(self.trade_file[:-len('all_trades.csv')] + 'strategies.csv', index = False)
                strategy_execution_df.to_csv(self.trade_file[:-len('all_trades.csv')] + 'strategy_execution.csv', index = False)


            if not is_production:
                plot_pnl_figure(write_df, self.chart_folder, self.currency, start_draw_down, end_draw_down, self.log_msg)



    def calc_max_drawdown(self, x):
        df = pd.DataFrame({'cum_pnl': x})
        df['max_cum_pnl'] = df['cum_pnl'].cummax()
        df['draw_down'] = df['max_cum_pnl'] - df['cum_pnl']

        max_draw_down = df['draw_down'].max()
        end = df['draw_down'].argmax()

        start = which(df['cum_pnl'] == df.iloc[end]['max_cum_pnl'])[0]

        return (max_draw_down, start, end)


    def trade(self, print_ready=True, temporary_decision = False):

        self.log_msg("In trade method, is_production = " + str(is_production))

        trade_start_time = datetime.now()
        self.log_msg("trade_start_time = " + str(trade_start_time))
        self.log_msg("Do trading............")

        self.calculate_signals(print_ready, temporary_decision)

        trade_end_time = datetime.now()
        self.log_msg("trade_end_time = " + str(trade_end_time))

        delta_time = trade_end_time - trade_start_time
        delta_minute = delta_time.seconds//60
        delta_second = delta_time.seconds%60

        self.log_msg("Trading takes " + str(delta_minute) + " minutes " + str(delta_second) + " seconds.")

        #print_prefix = "[Currency " + self.currency + "] "
        print_prefix = ""
        all_days = pd.Series(self.data_df['date'].unique()).dt.to_pydatetime()

        if not is_production:
            plot_candle_bar_charts(self.currency, self.data_df, all_days, self.long_df, self.short_df,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   is_plot_aux = True, is_plot_rsi = plot_rsi,
                                   bar_fig_folder=self.chart_folder, is_plot_simple_chart=True,
                                   use_dynamic_TP = use_dynamic_TP, figure_num = printed_figure_num, plot_day_line = plot_day_line, plot_cross_point = plot_cross_point,
                                   plot_long = True, plot_short = False, remove_plots = True, log_msg = self.log_msg)

            plot_candle_bar_charts(self.currency, self.data_df, all_days, self.long_df, self.short_df,
                                   num_days=20, plot_jc=True, plot_bolling=True, is_jc_calculated=True,
                                   is_plot_candle_buy_sell_points=True,
                                   print_prefix=print_prefix,
                                   is_plot_aux=True, is_plot_rsi = plot_rsi,
                                   bar_fig_folder=self.chart_folder, is_plot_simple_chart=True,
                                   use_dynamic_TP=use_dynamic_TP, figure_num=printed_figure_num,
                                   plot_day_line=plot_day_line, plot_cross_point=plot_cross_point,
                                   plot_long=False, plot_short=True, remove_plots = False, log_msg = self.log_msg)


        self.log_msg("Finish")












