import time
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
from coinbase.rest import RESTClient
from CurrencySmartExecutor import CurrencySmartExecutor

import threading
import time

class CurrencySmartExecutionManager(threading.Thread):

    def __init__(self, coinbase_client: RESTClient, coinbase_portfolio_id, heart_beat = 60):

        self.thread_condition: threading.Condition = threading.Condition()
        self.thread_lock: threading.Lock = threading.Lock()
        self.coinbase_client = coinbase_client
        self.coinbase_portfolio_id = coinbase_portfolio_id
        self.heart_beat = heart_beat

        self.currency2executor = {}

    def add_currency_executor(self, currency, currency_coinbase, strategy_prod_file, strategy_execution_prod_file, coinbase_decimal = 0):

        #This should be called before this thread starts (i.e., run() is executed)
        self.currency2execution[currency] = CurrencySmartExecutor(currency_conbase = currency_coinbase,
                                                                  coinbase_portfolio_id=self.coinbase_portfolio_id,
                                                                  strategy_prod_file = strategy_prod_file,
                                                                  strategy_execution_prod_file = strategy_execution_prod_file,
                                                                  coinbase_client = self.coinbase_client)


    def run(self):

        while True:
            with self.thread_condition:
                while len(self.currency2executor) == 0:
                    self.thread_condition.wait()

                for currency, v in self.currency2executor.items():
                    executor: CurrencySmartExecutor = v
                    print("Manage executinos for currency " + currency)
                    executor.manage_executions()

            print("Sleep " + str(self.heart_beat) + " seconds before next checking")
            time.sleep(self.heart_beat)




    def open_executions(self, currency, target_position, entry_time, strategy_executions):

        with self.thread_condition:

            smart_executor : CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.open_executions(target_position, entry_time, strategy_executions)

            self.thread_condition.notifyAll()

    def set_open_position_price(self, currency, entry_price):

        with self.thread_condition:

            smart_executor : CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.open_executions(entry_price)

            self.thread_condition.notifyAll()


    def close_executions(self, currency, position_to_close, exit_time):

        with self.thread_condition:

            smart_executor: CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.close_executions(position_to_close, exit_time)

            self.thread_condition.notifyAll()


    def set_close_position_price(self, currency, exit_price):

        with self.thread_condition:

            smart_executor : CurrencySmartExecutor = self.currency2executor[currency]
            smart_executor.set_close_position_price(exit_price)

            self.thread_condition.notifyAll()



