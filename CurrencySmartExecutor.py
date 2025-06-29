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

class CurrencySmartExecutor:

    def __init__(self, currency, currency_coinbase, coinbase_portfolio_id, coinbase_client: Optional[RESTClient] = None, coinbase_decimal = 0):
        self.currency = currency
        self.currency_coinbase = currency_coinbase
        self.coinbase_portfolio_id = coinbase_portfolio_id

        self.coinbase_client = coinbase_client
        self.coinbase_decimal = coinbase_decimal

        self.target_position = 0
        self.target_side = None
        self.strategy_executions = [] #Need to implement persistency logic (load from persistency at startup)

        self.current_position = self.get_current_position()

        self.new_signal_fired = False

        self.crypto_open_price = None



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
        if self.new_signal_fired:
            if self.current_position == self.target_position:

                for i in range(len(self.strategy_executions)):
                    strategy_execution: Optional[StrategyExecution] = self.strategy_executions[i]

                    if i == len(self.strategy_executions) and use_extra_execution:
                        try:
                            client_order_id = f"order_{uuid.uuid4()}"

                            stop_price = self.crypto_open_price * 0.5 if self.target_side == 'BUY' else self.crypto_open_price * 1.5

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


                self.new_signal_fired = False
        else:
            #Manage each execution
            pass


        pass

    def open_executions(self, target_position, crypto_open_price, strategy_executions = []):
        self.target_position = target_position #This is sided
        self.target_side = 'BUY' if self.target_position > 0 else 'SELL'
        self.strategy_executions = strategy_executions

        self.new_signal_fired = True

        self.crypto_open_price = crypto_open_price #This is the crypto_last_price at a new position open time







    def close_executions(self):

        pass




