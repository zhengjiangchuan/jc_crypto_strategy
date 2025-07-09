import time
#import talib

import math
import matplotlib.lines as mlines
from datetime import datetime
import pandas as pd
import math
import copy
from functools import reduce
import numpy as np
import sys
import os
from coinbase.rest import RESTClient
from CurrencySmartExecutor import CurrencySmartExecutor

import threading
import time

class CurrencySmartExecutionManager(threading.Thread):

    def __init__(self, coinbase_client: RESTClient, coinbase_portfolio_id, heart_beat = 60, use_extra_execution = False, global_executor_log_path = None):

        self.thread_condition: threading.Condition = threading.Condition()
        self.thread_lock: threading.Lock = threading.Lock()
        self.coinbase_client = coinbase_client
        self.coinbase_portfolio_id = coinbase_portfolio_id
        self.heart_beat = heart_beat

        self.currency2executor = {}

        self.prod_files_written = False

        self.use_extra_execution = use_extra_execution

        self.global_executor_log_path = global_executor_log_path
        self.global_executor_log_fd = open(self.global_executor_log_path, 'a')

    def add_currency_executor(self, currency, currency_coinbase, strategy_prod_file, strategy_execution_prod_file, trade_file, trade_prod_file, log_file, strategy_number):

        #This should be called before this thread starts (i.e., run() is executed)
        self.currency2execution[currency] = CurrencySmartExecutor(currency_conbase = currency_coinbase,
                                                                  coinbase_portfolio_id=self.coinbase_portfolio_id,
                                                                  strategy_prod_file = strategy_prod_file,
                                                                  strategy_execution_prod_file = strategy_execution_prod_file,
                                                                  trade_file = trade_file,
                                                                  trade_prod_file = trade_prod_file,
                                                                  log_file = log_file,
                                                                  strategy_number = strategy_number,
                                                                  coinbase_client = self.coinbase_client)


    def run(self):

        while True:
            with self.thread_condition:
                while len(self.currency2executor) == 0:
                    self.thread_condition.wait()

                some_closed_position = False
                for currency, v in self.currency2executor.items():
                    executor: CurrencySmartExecutor = v
                    print("Manage executinos for currency " + currency)
                    executor.manage_executions()

                    if executor.waiting_to_finalize_pnl:
                        some_closed_position = True

                if some_closed_position:
                    while not self.prod_files_written:
                        self.thread_condition.wait()

                    for currency, v in self.currency2executor.items():
                        executor: CurrencySmartExecutor = v
                        if executor.waiting_to_finalize_pnl:
                            executor.finalize_pnl_to_prod_file()

                    self.prod_files_written = False




            print("Sleep " + str(self.heart_beat) + " seconds before next checking")
            time.sleep(self.heart_beat)


    def reset_prod_files_written(self):

        with self.thread_condition:
            self.prod_files_written = False
            self.thread_condition.notify_all()

    def write_to_prod_files_finished(self):

        with self.thread_condition:
            self.prod_files_written = True;
            self.thread_condition.notify_all()


    def open_executions(self, currency, target_position, entry_time, strategy_executions):

        with self.thread_condition:

            smart_executor : CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.open_executions(target_position, entry_time, strategy_executions)

            self.thread_condition.notify_all()

    def set_open_position_price(self, currency, entry_price):

        with self.thread_condition:

            smart_executor : CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.open_executions(entry_price)

            self.thread_condition.notify_all()


    def close_executions(self, currency, position_to_close, exit_time, signal_exit_price):

        with self.thread_condition:

            smart_executor: CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.close_executions(position_to_close, exit_time, signal_exit_price)

            self.thread_condition.notify_all()


    def set_close_position_price(self, currency, exit_price):
        self.global_executor_log_fd
        with self.thread_condition:

            smart_executor : CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.set_close_position_price(exit_price)

            self.thread_condition.notify_all()


    def log_msg(self, msg):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(msg, pd.DataFrame):
            print('[' + current_time + ']  \n' + str(msg), file=self.global_executor_log_fd)
        else:
            print('[' + current_time + ']  ' + str(msg), file=self.global_executor_log_fd)

        self.global_executor_log_fd.flush()



