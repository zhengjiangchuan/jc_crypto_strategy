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
from typing import Any, Dict, List, Optional
from coinbase.rest import RESTClient
from instrument_trader import *

from enum import Enum,auto

class OrderType(Enum):
    TAKE_PROFIT_EXIT = auto()
    STOP_LOSS_EXIT = auto()
    STOP_ENTER = auto()

class Order:

    def __init__(self, coinbase_order_id, size, order_type : OrderType):

        self.coinbase_order_id = coinbase_order_id
        self.size = size
        self.order_type = order_type

    def order_id(self):
        return self.coinbase_order_id

    def order_type(self):
        return self.order_type

    def order_size(self):
        return self.size

class CurrencySmartExecutor:

    def __init__(self, currency, currency_coinbase, coinbase_portfolio_id, strategy_prod_file, strategy_execution_prod_file,
                 coinbase_client: Optional[RESTClient] = None, coinbase_decimal = 0):
        self.currency = currency
        self.currency_coinbase = currency_coinbase
        self.coinbase_portfolio_id = coinbase_portfolio_id
        self.strategy_prod_file = strategy_prod_file
        self.strategy_execution_prod_file = strategy_execution_prod_file

        self.coinbase_client = coinbase_client
        self.coinbase_decimal = coinbase_decimal

        self.target_position = 0
        self.target_side = None

        self.position_to_close = 0
        self.side_to_close = None

        # self.last_target_position = 0
        # self.last_target_side = None

        self.entry_time = None
        self.exit_time = None

        self.strategy_executions = [] #TODO: Need to implement persistency logic (load from persistency at startup)
        self.new_strategy_executions = []

        self.current_position = self.get_current_position()

        self.new_position_opened = False
        self.old_position_closed = False

        self.open_position_fill_price = 0
        self.close_position_fill_price = 0

        self.execution2order = {}



    def get_current_position(self):

        positions = self.coinbase_client.list_perps_positions(portfolio_uuid=self.coinbase_portfolio_id).positions
        for position in positions:

            if position['symbol'] == self.currency_coinbase:
                current_real_position = float(position['net_size'])

                if position['position_side'] not in ['POSITION_SIDE_LONG', 'POSITION_SIDE_SHORT']:
                    print("Unknown position side " + position['position_side'])
                    sys.exit(1)

                if position['position_side'] == 'POSITION_SIDE_SHORT':
                    current_real_position *= -1

                break

        return current_real_position

    def opposite_side(self, side):

        return 'BUY' if side == 'SELL' else 'SELL'

    def generate_client_order_id(self):

        return f"order_{uuid.uuid4()}"


    def manage_executions(self):

        self.current_position = self.get_current_position()
        if self.new_position_opened:
            if self.current_position == self.target_position and self.open_position_fill_price > 0:

                #TODO: APPEND the new opened position open price, entry_time etc to strategy_prod_file and strategy_execution_prod_file (Write the new opened executions to persistence)

                for i in range(len(self.new_strategy_executions)):
                    strategy_execution: StrategyExecution = self.new_strategy_executions[i]

                    if use_extra_execution and i == len(self.strategy_executions)-1:
                        try:
                            client_order_id = self.generate_client_order_id()

                            stop_price = self.calc_never_reached_stop_price(self.open_position_fill_price, self.target_side, is_stop_loss = True)


                            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                           client_order_id=client_order_id,
                                                           side=self.opposite_side(self.target_side),
                                                           order_configuration={
                                                               "trigger_bracket_gtc": {
                                                                   "base_size": str(strategy_execution.prod_size),
                                                                   "limit_price": str(strategy_execution.take_profit_price),
                                                                   "stop_trigger_price": str(stop_price)
                                                               }
                                                           },
                                                           leverage="10",
                                                           margin_type="CROSS",
                                                           retail_portfolio_id=self.coinbase_portfolio_id
                                                           )

                        except Exception as e:
                            print(f"Order failed: {e}")

                        stop_profit_order_id = response['success_response']['order_id']

                        self.execution2order[i+1] = [Order(stop_profit_order_id, strategy_execution.prod_size,  OrderType.TAKE_PROFIT_EXIT)]

                    else:

                        stop_entry_order_id, stop_entry_order_size = self.place_stop_entry_order(strategy_execution)

                        self.execution2order[i+1] = [Order(stop_entry_order_id, stop_entry_order_size,  OrderType.STOP_ENTER)]


                self.new_position_opened = False
                self.target_position = 0
                self.entry_time = None
                self.target_side = None

                if len(self.strategy_executions) == 0 and len(self.new_strategy_executions) > 0:
                    self.strategy_executions = self.new_strategy_executions
                    self.new_strategy_executions = []

        if self.old_position_closed:

            if self.close_position_fill_price > 0:

                for i in range(len(self.strategy_executions)):
                    strategy_execution: StrategyExecution = self.strategy_executions[i]

                    if not strategy_execution.active:
                        continue

                    is_extra = use_extra_execution and i == len(self.strategy_executions) - 1

                    strategy_execution.exit_execution(execution_exit_time=self.exit_time, execution_exit_price=self.close_position_fill_price,
                                             is_signal_exit=True,
                                             is_extra_execution=is_extra)

                    del self.execution2order[i+1]

                # TODO: APPEND the new closed position close price, exit_time etc to strategy_prod_file and strategy_execution_prod_file (Write closed executions to persistence)

                self.old_position_closed = False
                self.position_to_close = 0
                self.exit_time = None
                self.side_to_close = None

                if len(self.new_strategy_executions) > 0:
                    self.strategy_executions = self.new_strategy_executions
                    self.new_strategy_executions = []
                else:
                    self.strategy_executions = []



                pass


            pass


        # Manage each execution
        for i in range(len(self.strategy_executions)):

            strategy_execution: StrategyExecution = self.strategy_executions[i]

            if not strategy_execution.is_active:
                continue

            if i+1 in self.execution2order:

                order_list = self.execution2order[i+1]
                has_order_filled = False
                filled_order: Order = None
                for order in order_list:
                    is_filled, filled_price = self.check_order_fully_filled(order)
                    if is_filled:
                        has_order_filled = True
                        filled_order = order
                        break

                if has_order_filled:

                    is_extra = use_extra_execution and i == len(self.strategy_executions)-1

                    time_now = self.current_time()

                    if is_extra:
                        assert(filled_order.order_type() == OrderType.TAKE_PROFIT_EXIT)

                        strategy_execution.exit_execution(execution_exit_time=time_now,
                                                          execution_exit_price=filled_price,
                                                          is_signal_exit=False,
                                                          is_extra_execution=True)

                        del self.execution2order[i+1]

                        #TODO: Write finished execution to persistence
                    else:

                        assert(filled_order.order_type() in [OrderType.STOP_ENTER, OrderType.STOP_LOSS_EXIT])

                        strategy_execution.exit_execution(execution_exit_time=time_now,
                                                          execution_exit_price=filled_price,
                                                          is_signal_exit=False,
                                                          is_extra_execution=False)

                        # TODO: Write finished execution to persistence

                        if filled_order.order_type() == OrderType.STOP_ENTER:


                            strategy_execution.update_to_next_execution(entry_time=time_now, increased_size=filled_order.order_size())


                            for order in order_list:
                                if order.order_type() == OrderType.STOP_LOSS_EXIT:
                                    #Cancel this stop loss order because we have reached take profit and re-entered
                                    try:
                                        cancel_response = self.coinbase_client.cancel_orders(order_ids=[order.order_id])
                                        print(cancel_response)
                                    except Exception as e:
                                        print("Error:", e)



                            stop_entry_order_id, stop_entry_order_size = self.place_stop_entry_order(strategy_execution)

                            stop_loss_order_id, stop_loss_order_size = self.place_stop_loss_order(strategy_execution)

                            self.execution2order[i + 1] = [Order(stop_entry_order_id, stop_entry_order_size, OrderType.STOP_ENTER),
                                                           Order(stop_loss_order_id, stop_loss_order_size, OrderType.STOP_LOSS_EXIT)]

                        elif filled_order.order_type() == OrderType.STOP_LOSS_EXIT:

                            for order in order_list:
                                if order.order_type() == OrderType.STOP_ENTER:
                                    #Cancel this stop loss order because we have reached take profit and re-entered
                                    try:
                                        cancel_response = self.coinbase_client.cancel_orders(order_ids=[order.order_id])
                                        print(cancel_response)
                                    except Exception as e:
                                        print("Error:", e)

                            del self.execution2order[i+1]




    def place_stop_loss_order(self, strategy_execution: StrategyExecution):

        try:
            client_order_id = self.generate_client_order_id()

            stop_price = self.calc_never_reached_stop_price(strategy_execution.execution_entry_price, strategy_execution.side,
                                                            is_stop_loss=False)

            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                         client_order_id=client_order_id,
                                                         side=self.opposite_side(strategy_execution.side),
                                                         order_configuration={
                                                             "trigger_bracket_gtc": {
                                                                 "base_size": str(strategy_execution.prod_size),
                                                                 "limit_price": str(stop_price),
                                                                 "stop_trigger_price": str(strategy_execution.take_loss_price)
                                                             }
                                                         },
                                                         leverage="10",
                                                         margin_type="CROSS",
                                                         retail_portfolio_id=self.coinbase_portfolio_id
                                                         )

        except Exception as e:
            print(f"Order failed: {e}")

        stop_trigger_order_id = response['success_response']['order_id']

        return (stop_trigger_order_id, strategy_execution.prod_size)


    def place_stop_entry_order(self, strategy_execution: StrategyExecution):

        try:
            client_order_id = self.generate_client_order_id()

            size = strategy_execution.calc_increased_size_when_take_profit()
            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                         client_order_id=client_order_id,
                                                         side=strategy_execution.side,
                                                         order_configuration={
                                                             "stop_limit_stop_limit_gtc": {
                                                                 "base_size": str(size),
                                                                 "limit_price": str(self.calc_buffer_limit_price(
                                                                     strategy_execution.take_profit_price,
                                                                     self.target_sise)),
                                                                 "stop_price": str(strategy_execution.take_profit_price)
                                                             }
                                                         },
                                                         leverage="10",
                                                         margin_type="CROSS",
                                                         retail_portfolio_id=self.coinbase_portfolio_id
                                                         )

        except Exception as e:
            print(f"Order failed: {e}")

        stop_entry_order_id = response['success_response']['order_id']

        return (stop_entry_order_id, size)



    def current_time(self):

        nowtime = datetime.now();

        return datetime(nowtime.year, nowtime.month, nowtime.day, nowtime.hour, nowtime.minute, 0)

    def check_order_fully_filled(self, order: Order):

        fully_filled = False
        orderResponse = self.coinbase_client.get_order(order_id=order.order_id())
        if hasattr(orderResponse, "order"):
            coinbaseorder = orderResponse.order
            if coinbaseorder is not None:
                status = coinbaseorder['status']
                filled_size = float(coinbaseorder['filled_size'])
                filled_price = float(coinbaseorder['average_filled_price'])


                if status == 'FILLED' and filled_size == order.order_size():
                    fully_filled = True
                    return (fully_filled, filled_price)

        return (False, 0)


    def calc_never_reached_stop_price(self, entry_price, side, is_stop_loss):
        if (side == 'BUY' and is_stop_loss) or (side == 'SELL' and not is_stop_loss):
            return entry_price * 0.5
        else:
            return entry_price * 1.5

    def calc_buffer_limit_price(self, entry_price, side):

        if side == 'BUY':
            return entry_price * 1.1
        else:
            return entry_price * 0.9


    def open_executions(self, target_position, entry_time, strategy_executions = []):
        self.target_position = target_position #This is sided
        self.target_side = 'BUY' if self.target_position > 0 else 'SELL'
        self.entry_time = entry_time
        self.new_strategy_executions = strategy_executions

        self.new_position_opened = True

    def set_open_position_price(self, entry_price):
        self.open_position_fill_price = entry_price



    def close_executions(self, position_to_close, exit_time):

        self.position_to_close = position_to_close
        self.side_to_close = 'BUY' if self.position_to_close > 0 else 'SELL'
        self.exit_time = exit_time
        #self.strategy_executions = []

        self.old_position_closed = True


    def set_close_position_price(self, exit_price):
        self.close_position_fill_price = exit_price

