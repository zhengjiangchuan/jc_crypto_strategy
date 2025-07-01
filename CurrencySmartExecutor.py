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

        self.current_position = self.get_current_position()

        self.new_position_opened = False
        self.old_position_closed = False

        self.open_position_fill_price = 0
        self.close_position_fill_price = 0



    def get_current_position(self):

        positions = self.coinbase_client.list_perps_positions(portfolio_uuid=self.coinbase_portfolio_id).positions
        for position in positions:

            if position['symbol'] == self.currency_coinbase:
                current_real_position = float(position['net_size'])

                if position['position_side'] not in ['POSITION_SIDE_LONG', 'POSITION_SIDE_SHORT']:
                    self.log_msg("Unknown position side " + position['position_side'])
                    sys.exit(1)

                if position['position_side'] == 'POSITION_SIDE_SHORT':
                    current_real_position *= -1

                break

        return current_real_position

    def opposite_side(self, side):

        return 'BUY' if side == 'SELL' else 'SELL'

    def manage_executions(self):

        self.current_position = self.get_current_position()
        if self.new_position_opened:
            if self.current_position == self.target_position and self.open_position_fill_price > 0:

                #TODO: APPEND the new opened position open price, entry_time etc to strategy_prod_file and strategy_execution_prod_file

                for i in range(len(self.strategy_executions)):
                    strategy_execution: Optional[StrategyExecution] = self.strategy_executions[i]

                    if i == len(self.strategy_executions) and use_extra_execution:
                        try:
                            client_order_id = f"order_{uuid.uuid4()}"

                            stop_price = self.open_position_fill_price * 0.5 if self.target_side == 'BUY' else self.open_position_fill_price * 1.5

                            response = self.coinbaseclient.create_order(product_id=self.currency_coinbase,
                                                           # BTC-USDC is the correct product id
                                                           client_order_id=client_order_id,
                                                           side=self.opposite_side(self.target_side),
                                                           order_configuration={
                                                               "trigger_bracket_gtc": {
                                                                   "base_size": str(abs(self.current_position)),
                                                                   "limit_price": str(strategy_execution.take_profit_price),
                                                                   "stop_trigger_price": str(stop_price)
                                                               }
                                                           },
                                                           leverage="10",
                                                           margin_type="CROSS"
                                                           # retail_portfolio_id="0194271a-bd95-7ba7-a028-6561a970128b"
                                                           )

                        except Exception as e:
                            self.log_msg(f"Order failed: {e}")

                        response['success_response']['order_id']


                self.new_position_opened = False
                self.target_position = 0
                self.entry_time = None
                self.target_side = None

        if self.old_position_closed:

            if self.close_position_fill_price > 0:

                # TODO: APPEND the new closed position close price, exit_time etc to strategy_prod_file and strategy_execution_prod_file
                pass


            pass


        # Manage each execution
        pass

    def calc_never_reached_stop_price(self, entry_price, side, is_stop_loss):
        if (side == 'BUY' and is_stop_loss) or (side == 'SELL' and not is_stop_loss):
            return entry_price * 0.5
        else:
            return entry_price * 1.5


    def open_executions(self, target_position, entry_time, strategy_executions = []):
        self.target_position = target_position #This is sided
        self.target_side = 'BUY' if self.target_position > 0 else 'SELL'
        self.entry_time = entry_time
        self.strategy_executions = strategy_executions

        self.new_position_opened = True

    def set_open_position_price(self, open_price):
        self.open_position_fill_price = open_price



    def close_executions(self, position_to_close, exit_time):

        self.position_to_close = position_to_close
        self.side_to_close = 'BUY' if self.position_to_close > 0 else 'SELL'
        self.exit_time = exit_time
        self.strategy_executions = []

        self.old_position_closed = True


    def set_close_position_price(self, close_price):
        self.close_position_fill_price = close_price

