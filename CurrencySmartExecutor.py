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
from typing import Any, Dict, List, Optional
from coinbase.rest import RESTClient
#from instrument_trader import *

from StrategyExecution import *
from enum import Enum,auto

import uuid


from util import sendEmail

print_email_message_to_file = False

class OrderType(Enum):
    TAKE_PROFIT_EXIT = auto()
    STOP_LOSS_EXIT = auto()
    STOP_ENTER = auto()

class Order:

    def __init__(self, coinbase_order_id, size, order_type : OrderType):

        self.coinbase_order_id = coinbase_order_id
        self.size = size
        self.order_type = order_type

    def __str__(self):

        return f"coinbase_order_id={self.coinbase_order_id}, size={self.size}, order_type={self.order_type}"

    def order_id(self):
        return self.coinbase_order_id

    def aux_order_type(self):
        return self.order_type

    def order_size(self):
        return self.size



class CurrencySmartExecutor:

    def __init__(self, currency_coinbase, coinbase_portfolio_id, strategy_prod_file, strategy_execution_prod_file, trade_file, trade_prod_file, log_file,  strategy_number,
                 size_decimal, price_decimal, coinbase_client: Optional[RESTClient] = None, use_extra_execution = False):
        self.currency_coinbase = currency_coinbase
        self.coinbase_portfolio_id = coinbase_portfolio_id
        self.strategy_prod_file = strategy_prod_file
        self.strategy_execution_prod_file = strategy_execution_prod_file
        self.trade_file = trade_file
        self.trade_prod_file = trade_prod_file
        self.log_file = log_file

        self.log_fd = open(self.log_file, 'a')

        self.strategy_number = strategy_number

        self.size_decimal = size_decimal
        self.price_decimal = price_decimal
        self.coinbase_client = coinbase_client

        self.target_position = 0
        self.target_side = None

        self.position_to_close = 0
        self.side_to_close = None

        # self.last_target_position = 0
        # self.last_target_side = None

        self.entry_time = None
        self.exit_time = None
        self.signal_exit_price = 0

        self.strategy_executions = [] #TODO: Need to implement persistency logic (load from persistency at startup)
        self.new_strategy_executions = []

        self.current_position = self.get_current_position(True)

        self.new_position_opened = False
        self.old_position_closed = False

        self.open_position_fill_price = 0
        self.close_position_fill_price = 0

        self.execution2order = {}

        self.execution_data_df = None #pandas df

        self.max_long_trade_id = 0
        self.max_short_trade_id = 0
        self.max_index = -1

        self.waiting_to_finalize_pnl = False

        self.use_extra_execution = use_extra_execution

        self.strategy_data_columns = ['side', 'long_trade_id', 'short_trade_id', 'strategy_id', 'leverage',
                                      'entry_time', 'entry_price', 'prod_entry_price', 'entry_value', 'prod_entry_value',
                                      'exit_time', 'exit_price', 'prod_exit_price', 'exit_value', 'prod_exit_value', 'pnl', 'prod_pnl']

        self.execution_data_columns = ['idx', 'side', 'long_trade_id', 'short_trade_id', 'strategy_id', 'execution_id', 'leverage',
                                       'take_profit_pct', 'take_profit_price', 'take_loss_pct', 'take_loss_price', 'prod_size',
                                        'strategy_entry_time', 'strategy_entry_price', 'strategy_entry_value', 'prod_strategy_entry_price', 'prod_strategy_entry_value',
                                        'execution_entry_time', 'execution_entry_price', 'execution_entry_value', 'prod_execution_entry_price', 'prod_execution_entry_value',
                                        'execution_exit_time', 'execution_exit_price', 'execution_exit_value', 'prod_execution_exit_price', 'prod_execution_exit_value',
                                        'pnl', 'prod_pnl',
                                        'coinbase_order_id1', 'coinbase_order_size1', 'coinbase_order_type1',
                                        'coinbase_order_id2', 'coinbase_order_size2', 'coinbase_order_type2'
                                       ]

        self.execution2persistence = {} #Key is "side"_"strategy_id"_"execution_id", value is the row index of the execution_data_df


        self.strategy_data_df = None
        if os.path.exists(self.strategy_prod_file):
            self.strategy_data_df = pd.read_csv(self.strategy_prod_file)


        self.execution_data_df = None

        self.log_msg(f"size_decimal = {self.size_decimal}, price_decimal = {self.price_decimal} for currency {self.currency_coinbase}")

        if os.path.exists(self.strategy_execution_prod_file):

            self.log_msg("Recovering executions from files")

            self.execution_data_df = pd.read_csv(self.strategy_execution_prod_file)


            print("execution_data_df recovered:")
            print(self.execution_data_df)


            self.strategy_executions = [None] * (self.strategy_number+1 if self.use_extra_execution else self.strategy_number)

            unfinished_execution_data_df = self.execution_data_df[self.execution_data_df['execution_exit_price'].isnull()]

            self.log_msg("unfinished execution_data_df:")
            self.log_msg(unfinished_execution_data_df)

            for i in range(unfinished_execution_data_df.shape[0]):
                unfinished_execution_data = unfinished_execution_data_df.iloc[i]
                unfinished_execution: StrategyExecution = self.recover_execution(unfinished_execution_data)
                self.strategy_executions[unfinished_execution.strategy_id-1] = unfinished_execution #should be strategy_id not execution_id here

                self.log_msg(f"Recovering execution strategy {unfinished_execution.strategy_id}")
                self.log_msg(unfinished_execution)

                order_list = []
                for j in range(1,3):
                    coinbase_order_id = unfinished_execution_data['coinbase_order_id' + str(j)]
                    #print(f"Fuck coinbase_order_id: {coinbase_order_id}")
                    #print(type(coinbase_order_id))

                    if coinbase_order_id is not None and not (isinstance(coinbase_order_id, float) and math.isnan(coinbase_order_id)):
                        coinbase_order_id = str(coinbase_order_id)
                        coinbase_order_size = float(unfinished_execution_data['coinbase_order_size' + str(j)])
                        coinbase_order_type = self.convert_str_to_order_type(unfinished_execution_data['coinbase_order_type' + str(j)])

                        order_list += [Order(coinbase_order_id, coinbase_order_size, coinbase_order_type)]

                self.execution2order[unfinished_execution.strategy_id] = order_list


               # key = '_'.join([unfinished_execution['side'], str(unfinished_execution['strategy_id']), str(unfinished_execution['execution_id'])])

                key = '_'.join([self.parse_position2(unfinished_execution.side), str(unfinished_execution.strategy_id),
                                str(unfinished_execution.execution_id)])
                self.execution2persistence[key] = unfinished_execution_data['idx']

            self.log_msg("execution2order now recovered is:")
            for stra_id, o_list in self.execution2order.items():
                self.log_msg(f"strategy_id={stra_id}")
                self.log_msg("order_list:")
                for order in o_list:
                    self.log_msg(order)

            self.log_msg("execution2persistence is:")
            self.log_msg(self.execution2persistence)


            self.max_index = self.execution_data_df['idx'].max()
            self.max_long_trade_id = self.execution_data_df['long_trade_id'].max()
            self.max_short_trade_id = self.execution_data_df['short_trade_id'].max()

            self.log_msg(f"max_index={self.max_index}, max_long_trade_id={self.max_long_trade_id}, max_short_trade_id={self.max_short_trade_id}")



    def recover_execution(self, unfinished_execution_data):

        #print(f"*****************strategy_entry_time = {unfinished_execution_data['strategy_entry_time']}")
        strategy_execution = StrategyExecution(side=1 if unfinished_execution_data['side'] == 'long' else -1,
                                               leverage = int(unfinished_execution_data['leverage']),
                                               take_profit_pct = float(unfinished_execution_data['take_profit_pct']),
                                               take_loss_pct = float(unfinished_execution_data['take_loss_pct']),
                                               strategy_id = int(unfinished_execution_data['strategy_id']),
                                               execution_id = int(unfinished_execution_data['execution_id']),
                                               strategy_entry_time = datetime.strptime(unfinished_execution_data['strategy_entry_time'], "%Y-%m-%d %H:%M:%S"),
                                               strategy_entry_price = float(unfinished_execution_data['strategy_entry_price']),
                                               execution_entry_time = datetime.strptime(unfinished_execution_data['execution_entry_time'], "%Y-%m-%d %H:%M:%S"),
                                               execution_entry_price = float(unfinished_execution_data['execution_entry_price']),
                                               strategy_entry_value = float(unfinished_execution_data['strategy_entry_value']),
                                               execution_entry_value = float(unfinished_execution_data['execution_entry_value']),
                                               prod_size = float(unfinished_execution_data['prod_size'])
                                               )
        strategy_execution.prod_strategy_entry_price = float(unfinished_execution_data['prod_strategy_entry_price'])
        strategy_execution.prod_execution_entry_price = float(unfinished_execution_data['prod_execution_entry_price'])
        strategy_execution.prod_strategy_entry_value = float(unfinished_execution_data['prod_strategy_entry_value'])
        strategy_execution.prod_execution_entry_value = float(unfinished_execution_data['prod_execution_entry_value'])

        return strategy_execution

    def convert_str_to_order_type(self, order_type_str):

        if order_type_str == 'TAKE_PROFIT_EXIT':
            return OrderType.TAKE_PROFIT_EXIT
        elif order_type_str == 'STOP_LOSS_EXIT':
            return OrderType.STOP_LOSS_EXIT
        elif order_type_str == 'STOP_ENTER':
            return OrderType.STOP_ENTER

    def convert_order_type_to_str(self, order_type: OrderType):

        if order_type == OrderType.TAKE_PROFIT_EXIT:
            return 'TAKE_PROFIT_EXIT'
        elif order_type == OrderType.STOP_LOSS_EXIT:
            return 'STOP_LOSS_EXIT'
        elif order_type == OrderType.STOP_ENTER:
            return 'STOP_ENTER'



    def get_current_position(self, print_heartbeat = False):

        current_real_position = 0

        try:
            positions = self.coinbase_client.list_perps_positions(portfolio_uuid=self.coinbase_portfolio_id).positions
        except Exception as e:

            message_title = "Get positions failed due to connection error"
            message = f"Get positions failed: {e}"
            self.log_msg(message)

            if not print_email_message_to_file:
                sendEmail(message_title, message, is_alternative=True)

        if print_heartbeat:
            self.log_msg(f"positions size = {len(positions)}")

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

    def finalize_pnl_to_prod_file(self):

        if os.path.exists(self.trade_file) and os.path.exists(self.trade_prod_file):

            self.log_msg("Finalize pnl to production file")

            trade_df = pd.read_csv(self.trade_file)
            trade_prod_df = pd.read_csv(self.trade_prod_file)

            long_trade_df = self.strategy_data_df[self.strategy_data_df['long_trade_id'] > 0]
            short_trade_df = self.strategy_data_df[self.strategy_data_df['short_trade_id'] > 0]

            long_trade_agg_df = long_trade_df[['long_trade_id', 'pnl', 'prod_pnl']]
            long_trade_agg_df = long_trade_agg_df.groupby(by = ['long_trade_id']).agg({'pnl' : 'sum', 'prod_pnl' : 'sum'})
            long_trade_agg_df.reset_index(inplace = True)
            long_trade_agg_df = long_trade_agg_df.rename(columns = {'pnl' : 'long_trade_pnl', 'prod_pnl' : 'long_trade_prod_pnl'})

            short_trade_agg_df = short_trade_df[['short_trade_id', 'pnl', 'prod_pnl']]
            short_trade_agg_df = short_trade_agg_df.groupby(by=['short_trade_id']).agg({'pnl': 'sum', 'prod_pnl': 'sum'})
            short_trade_agg_df.reset_index(inplace=True)
            short_trade_agg_df = short_trade_agg_df.rename(columns={'pnl': 'short_trade_pnl', 'prod_pnl': 'short_trade_prod_pnl'})

            trade_df = pd.merge(trade_df, long_trade_agg_df, on = ['long_trade_id'], how = 'left')
            trade_df = pd.merge(trade_df, short_trade_agg_df, on = ['short_trade_id'], how = 'left')

            trade_df['pnl'] = np.where(
                trade_df['long_trade_id'] > 0,
                trade_df['long_trade_pnl'],
                trade_df['short_trade_pnl']
            )

            trade_df = trade_df.drop(columns = ['long_trade_pnl', 'short_trade_pnl', 'long_trade_prod_pnl', 'short_trade_prod_pnl'])

            trade_df['cum_pnl'] = trade_df['pnl'].cumsum()

            trade_df['pnl'] = trade_df['pnl'].apply(lambda x: round(x, 2))
            trade_df['cum_pnl'] = trade_df['cum_pnl'].apply(lambda x: round(x, 2))

            trade_prod_df = pd.merge(trade_prod_df, long_trade_agg_df, on=['long_trade_id'], how='left')
            trade_prod_df = pd.merge(trade_prod_df, short_trade_agg_df, on=['short_trade_id'], how='left')

            trade_prod_df['pnl'] = np.where(
                trade_prod_df['long_trade_id'] > 0,
                trade_prod_df['long_trade_pnl'],
                trade_prod_df['short_trade_pnl']
            )

            trade_prod_df['prod_pnl'] = np.where(
                trade_prod_df['long_trade_id'] > 0,
                trade_prod_df['long_trade_prod_pnl'],
                trade_prod_df['short_trade_prod_pnl']
            )

            trade_prod_df = trade_prod_df.drop(
                columns=['long_trade_pnl', 'short_trade_pnl', 'long_trade_prod_pnl', 'short_trade_prod_pnl'])

            trade_prod_df['cum_pnl'] = trade_prod_df['pnl'].cumsum()
            trade_prod_df['pnl'] = trade_prod_df['pnl'].apply(lambda x: round(x, 2))
            trade_prod_df['cum_pnl'] = trade_prod_df['cum_pnl'].apply(lambda x: round(x, 2))

            trade_prod_df['prod_cum_pnl'] = trade_prod_df['prod_pnl'].cumsum()
            trade_prod_df['prod_pnl'] = trade_prod_df['prod_pnl'].apply(lambda x: round(x, 2))
            trade_prod_df['prod_cum_pnl'] = trade_prod_df['prod_cum_pnl'].apply(lambda x: round(x, 2))


            trade_df.to_csv(self.trade_file, index = False)
            trade_prod_df.to_csv(self.trade_prod_file, index = False)

        self.waiting_to_finalize_pnl = False




    def append_strategy_row(self, strategy_row):

        delta_data_df = pd.DataFrame(data=[strategy_row], columns=self.strategy_data_columns)

        self.log_msg(f"Strategy row is:")
        self.log_msg(delta_data_df)

        if self.strategy_data_df is None:
            self.strategy_data_df = delta_data_df
        else:
            self.strategy_data_df = pd.concat([self.strategy_data_df, delta_data_df])

        self.strategy_data_df.to_csv(self.strategy_prod_file, index=False)


    def generate_strategy_row(self, strategy_id):

        self.log_msg(f"Generate strategy row for {self.side_to_close} strategy {strategy_id} of crypto {self.currency_coinbase}")
        if self.side_to_close == 'BUY':

            if self.max_long_trade_id != self.execution_data_df['long_trade_id'].max():
                self.log_msg(f"Wrong!! max_long_trade_id = {self.max_long_trade_id}, but max long_trade_id = {self.execution_data_df['long_trade_id'].max()}")

            target_df = self.execution_data_df[(self.execution_data_df['long_trade_id'] == self.max_long_trade_id) & (self.execution_data_df['strategy_id'] == strategy_id)]
        else:

            if self.max_short_trade_id != self.execution_data_df['short_trade_id'].max():
                self.log_msg(f"Wrong!! max_short_trade_id = {self.max_short_trade_id}, but max short_trade_id = {self.execution_data_df['short_trade_id'].max()}")

            target_df = self.execution_data_df[(self.execution_data_df['short_trade_id'] == self.max_short_trade_id) & (self.execution_data_df['strategy_id'] == strategy_id)]

        last_execution_data = target_df[target_df['execution_id'] == target_df['execution_id'].max()].iloc[0]

        new_strategy_row = ['long' if self.side_to_close == 'BUY' else 'short',
                            self.max_long_trade_id if self.side_to_close == 'BUY' else 0,
                            self.max_short_trade_id if self.side_to_close == 'SELL' else 0,
                            strategy_id, last_execution_data['leverage'],
                            last_execution_data['strategy_entry_time'], last_execution_data['strategy_entry_price'],
                            last_execution_data['prod_strategy_entry_price'],
                            last_execution_data['strategy_entry_value'], last_execution_data['prod_strategy_entry_value'],
                            last_execution_data['execution_exit_time'], last_execution_data['execution_exit_price'],
                            last_execution_data['prod_execution_exit_price'],
                            last_execution_data['execution_exit_value'], last_execution_data['prod_execution_exit_value'],
                            last_execution_data['execution_exit_value'] - last_execution_data['strategy_entry_value'],
                            last_execution_data['prod_execution_exit_value'] - last_execution_data['prod_strategy_entry_value']
                            ]
        return new_strategy_row


    def finish_execution_row(self, strategy_id, strategy_execution: StrategyExecution):

        self.log_msg(f"Finish execution {strategy_execution.execution_id} of strategy {strategy_execution.strategy_id} of crypto {self.currency_coinbase}")
        key = '_'.join(['long' if strategy_execution.side == 1 else "short", str(strategy_id), str(strategy_execution.execution_id)])
        self.log_msg(f"key={key}")
        if key in self.execution2persistence:
            row_idx = self.execution2persistence[key]
            self.log_msg(f"row_idx={row_idx}")
            if self.execution_data_df is not None and row_idx >= 0:
                self.execution_data_df.at[row_idx, 'execution_exit_time'] = strategy_execution.execution_exit_time.strftime("%Y-%m-%d %H:%M:%S")
                self.execution_data_df.at[row_idx, 'execution_exit_price'] = strategy_execution.execution_exit_price
                self.execution_data_df.at[row_idx, 'execution_exit_value'] = strategy_execution.execution_exit_value
                self.execution_data_df.at[row_idx, 'prod_execution_exit_price'] = strategy_execution.prod_execution_exit_price
                self.execution_data_df.at[row_idx, 'prod_execution_exit_value'] = strategy_execution.prod_execution_exit_value
                self.execution_data_df.at[row_idx, 'pnl'] = strategy_execution.pnl
                self.execution_data_df.at[row_idx, 'prod_pnl'] = strategy_execution.prod_pnl

                self.log_msg("Finished execution:")
                self.log_msg(self.execution_data_df.iloc[row_idx:(row_idx+1)])

                self.execution_data_df.to_csv(self.strategy_execution_prod_file, index=False)

                del self.execution2persistence[key]


    def append_new_execution_row(self, new_execution_row):

        delta_data_df = pd.DataFrame(data = [new_execution_row], columns = self.execution_data_columns)

        self.log_msg("Add new execution row:")
        self.log_msg(delta_data_df)

        new_execution_row = delta_data_df.iloc[0]
        if self.execution_data_df is None:
            self.execution_data_df = delta_data_df
        else:
            self.execution_data_df = pd.concat([self.execution_data_df, delta_data_df])

        self.execution_data_df.to_csv(self.strategy_execution_prod_file, index = False)

        key = '_'.join([new_execution_row['side'], str(new_execution_row['strategy_id']),
                        str(new_execution_row['execution_id'])])
        self.execution2persistence[key] = new_execution_row['idx']

        self.log_msg(f"Map execution key {key} to row index {new_execution_row['idx']}")


    def generate_new_execution_row(self, trade_id, strategy_id, strategy_execution, order_list):

        # new_execution_row = [self.max_index + 1, 'long' if strategy_execution.side == 1 else 'short',
        #                      self.max_long_trade_id + 1 if strategy_execution.side == 1 else 0,
        #                      self.max_short_trade_id + 1 if strategy_execution.side == -1 else 0, strategy_id,
        #                      strategy_execution.execution_id, strategy_execution.leverage,
        #                      strategy_execution.take_profit_pct, strategy_execution.take_profit_price,
        #                      strategy_execution.take_loss_pct, strategy_execution.take_loss_price,
        #                      strategy_execution.prod_size,
        #                      strategy_execution.strategy_entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        #                      strategy_execution.strategy_entry_price, strategy_execution.strategy_entry_value,
        #                      strategy_execution.prod_strategy_entry_price, strategy_execution.prod_strategy_entry_value,
        #                      strategy_execution.execution_entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        #                      strategy_execution.execution_entry_price, strategy_execution.execution_entry_value,
        #                      strategy_execution.prod_execution_entry_price,
        #                      strategy_execution.prod_execution_entry_value,
        #                      None, None, None, None, None, None, None,
        #                      ]

        new_execution_row = [self.max_index + 1, 'long' if strategy_execution.side == 1 else 'short',
                             trade_id if strategy_execution.side == 1 else 0,
                             trade_id if strategy_execution.side == -1 else 0, strategy_id,
                             strategy_execution.execution_id, strategy_execution.leverage,
                             strategy_execution.take_profit_pct, round(strategy_execution.take_profit_price, self.price_decimal),
                             strategy_execution.take_loss_pct, round(strategy_execution.take_loss_price, self.price_decimal),
                             strategy_execution.prod_size,
                             strategy_execution.strategy_entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                             strategy_execution.strategy_entry_price, strategy_execution.strategy_entry_value,
                             strategy_execution.prod_strategy_entry_price, strategy_execution.prod_strategy_entry_value,
                             strategy_execution.execution_entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                             strategy_execution.execution_entry_price, strategy_execution.execution_entry_value,
                             strategy_execution.prod_execution_entry_price,
                             strategy_execution.prod_execution_entry_value,
                             None, None, None, None, None, None, None,
                             ]


        for order in order_list:
            new_execution_row += [order.order_id(), round(order.order_size(), self.size_decimal), self.convert_order_type_to_str(order.aux_order_type())]

        remaining = 2 - len(order_list)
        if remaining > 0:
            for i in range(remaining):
                new_execution_row += [None]*3

        self.max_index = self.max_index + 1
        # if strategy_execution.side == 1:
        #     self.max_long_trade_id = self.max_long_trade_id + 1
        # else:
        #     self.max_short_trade_id = self.max_short_trade_id + 1

        return new_execution_row

    def has_executions(self):
        #self.log_msg(f"strategy_executions len = {len(self.strategy_executions)}, new_strategy_executions len = {len(self.new_strategy_executions)}")
        return len(self.strategy_executions) > 0 or len(self.new_strategy_executions) > 0

    def parse_side(self, side):
        if side == 1:
            return 'BUY'
        else:
            return 'SELL'

    def parse_position(self, side):
        if side == 1:
            return 'Long'
        else:
            return 'Short'

    def parse_position2(self, side):
        if side == 1:
            return 'long'
        else:
            return 'short'

    def manage_executions(self, print_heartbeat = False):

        self.current_position = self.get_current_position(print_heartbeat = print_heartbeat)
        if self.new_position_opened:
            if self.current_position == self.target_position and self.open_position_fill_price > 0:

                #TODO: APPEND the new opened position open price, entry_time etc to strategy_prod_file and strategy_execution_prod_file (Write the new opened executions to persistence)

                self.log_msg(f"New {'Long' if self.current_position > 0 else 'Short'} position of {abs(self.current_position)} units opened at filled price {self.open_position_fill_price}")

                if self.current_position > 0:

                    self.max_long_trade_id = self.max_long_trade_id + 1
                    trade_id = self.max_long_trade_id

                else:

                    self.max_short_trade_id = self.max_short_trade_id + 1
                    trade_id = self.max_short_trade_id

                for i in range(len(self.new_strategy_executions)):

                    if self.new_strategy_executions[i] is None:
                        continue

                    strategy_execution: StrategyExecution = self.new_strategy_executions[i]

                    strategy_execution.set_prod_strategy_entry_price(self.open_position_fill_price)

                    self.log_msg("Process execution strategy " + str(i+1) + ":")


                    if self.use_extra_execution and i == len(self.strategy_executions)-1:
                        try:
                            client_order_id = self.generate_client_order_id()

                            stop_price = self.calc_never_reached_stop_price(self.open_position_fill_price, self.target_side, is_stop_loss = True)

                            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                           client_order_id=client_order_id,
                                                           side=self.opposite_side(self.parse_side(self.target_side)),
                                                           order_configuration={
                                                               "trigger_bracket_gtc": {
                                                                   "base_size": str(round(strategy_execution.prod_size, self.size_decimal)),
                                                                   "limit_price": str(round(strategy_execution.take_profit_price, self.price_decimal)),
                                                                   "stop_trigger_price": str(round(stop_price, self.price_decimal))
                                                               }
                                                           },
                                                           leverage="10",
                                                           margin_type="CROSS",
                                                           retail_portfolio_id=self.coinbase_portfolio_id
                                                           )

                            self.log_msg(f"Order placed: {response}")

                            message_title = f"Place stop profit order for extra strategy of crypto {self.currency_coinbase}"
                            message = f"Place stop profit order of {round(strategy_execution.prod_size, self.size_decimal)} units at take profit price " +\
                                f"{round(strategy_execution.take_profit_price, self.price_decimal)}  for {self.parse_side(strategy_execution.side)} order of extra strategy of crypto {self.currency_coinbase}"

                            self.log_msg(message_title)
                            self.log_msg(message)
                            if not print_email_message_to_file:
                                sendEmail(message_title, message, is_alternative=True)


                        except Exception as e:
                            self.log_msg(f"Order failed: {e}")

                        stop_profit_order_id = response['success_response']['order_id']

                        order_list = [Order(stop_profit_order_id, strategy_execution.prod_size,  OrderType.TAKE_PROFIT_EXIT)]
                        self.execution2order[strategy_execution.strategy_id] = order_list

                        new_execution_row = self.generate_new_execution_row(trade_id = trade_id,
                                                                            strategy_id = strategy_execution.strategy_id,
                                                                            strategy_execution = strategy_execution,
                                                                            order_list = order_list)
                        self.append_new_execution_row(new_execution_row)

                    else:

                        stop_entry_order_id, stop_entry_order_size = self.place_stop_entry_order(strategy_execution)

                        order_list = [Order(stop_entry_order_id, stop_entry_order_size,  OrderType.STOP_ENTER)]
                        self.execution2order[strategy_execution.strategy_id] = order_list

                        new_execution_row = self.generate_new_execution_row(trade_id = trade_id,
                                                                            strategy_id = strategy_execution.strategy_id,
                                                                            strategy_execution = strategy_execution,
                                                                            order_list = order_list)
                        self.append_new_execution_row(new_execution_row)

                    self.log_msg("execution2order now is:")
                    for stra_id, o_list in self.execution2order.items():
                        self.log_msg(f"strategy_id={stra_id}")
                        self.log_msg("order_list:")
                        for order in o_list:
                            self.log_msg(order)

                    #self.log_msg(self.execution2order)


                self.new_position_opened = False
                self.target_position = 0
                self.entry_time = None
                self.target_side = None
                self.open_position_fill_price = 0

                if len(self.strategy_executions) == 0 and len(self.new_strategy_executions) > 0:
                    self.strategy_executions = self.new_strategy_executions
                    self.new_strategy_executions = []

        if self.old_position_closed:

            if self.close_position_fill_price > 0:

                self.log_msg(f"Current {'Long' if self.position_to_close > 0 else 'Short'} position of {abs(self.position_to_close)} units get closed due to signal at price {self.close_position_fill_price}")

                for i in range(len(self.strategy_executions)):

                    if self.strategy_executions[i] is None:
                        continue

                    strategy_execution: StrategyExecution = self.strategy_executions[i]

                    if not strategy_execution.active:
                        continue

                    self.log_msg("Process execution strategy " + str(i + 1) + ":")

                    order_list = self.execution2order[strategy_execution.strategy_id]
                    for order in order_list:

                        message_title = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} cancels its pending order of type {order.aux_order_type()}"
                        message = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} cancels its pending order of type {order.aux_order_type()} of size {order.order_size()}"

                        self.log_msg(message_title)
                        self.log_msg(message)
                        if not print_email_message_to_file:
                            sendEmail(message_title, message, is_alternative=True)


                        try:
                            print("Cancel pending orders because closing signal fires")
                            cancel_response = self.coinbase_client.cancel_orders(order_ids=[order.order_id])
                            print(cancel_response)
                        except Exception as e:
                            print("Error:", e)


                    is_extra = self.use_extra_execution and i == len(self.strategy_executions) - 1

                    self.log_msg("Exit this execution")
                    ret_msg = strategy_execution.exit_execution(execution_exit_time=self.exit_time, execution_exit_price=self.signal_exit_price,
                                             is_signal_exit=True,
                                             is_extra_execution=is_extra)

                    self.log_msg(ret_msg)

                    self.log_msg("Exit this prod execution")
                    ret_msg = strategy_execution.exit_execution_prod(prod_execution_exit_price=self.close_position_fill_price)

                    self.log_msg(ret_msg)

                    self.finish_execution_row(i+1, strategy_execution)

                    strategy_row = self.generate_strategy_row(i+1)
                    self.append_strategy_row(strategy_row)


                    del self.execution2order[strategy_execution.strategy_id]

                    self.log_msg("execution2order now is:")
                    self.log_msg("execution2order now is:")
                    for stra_id, o_list in self.execution2order.items():
                        self.log_msg(f"strategy_id={stra_id}")
                        self.log_msg("order_list:")
                        for order in o_list:
                            self.log_msg(order)
                    #self.log_msg(self.execution2order)

                # TODO: APPEND the new closed position close price, exit_time etc to strategy_prod_file and strategy_execution_prod_file (Write closed executions to persistence)

                self.old_position_closed = False
                self.position_to_close = 0
                self.exit_time = None
                self.side_to_close = None
                self.close_position_fill_price = 0
                self.signal_exit_price = 0

                if len(self.new_strategy_executions) > 0:
                    self.strategy_executions = self.new_strategy_executions
                    self.new_strategy_executions = []
                else:
                    self.strategy_executions = []

                self.waiting_to_finalize_pnl = True



        # Manage each execution
        if print_heartbeat:
            self.log_msg("Manage each strategy's execution")

        for i in range(len(self.strategy_executions)):

            if self.strategy_executions[i] is None:
                continue

            strategy_execution: StrategyExecution = self.strategy_executions[i]

            if not strategy_execution.active:
                continue

            if strategy_execution.strategy_id in self.execution2order:

                #self.log_msg(f"Process strategy {strategy_execution.strategy_id}")
                strategy_msg = f"[Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id}] "

                order_list = self.execution2order[strategy_execution.strategy_id]
                has_order_filled = False
                filled_order: Order = None
                for order in order_list:
                    is_filled, filled_price = self.check_order_fully_filled(order)
                    if is_filled:
                        has_order_filled = True
                        filled_order = order

                        self.log_msg(strategy_msg + f"Its pending order {order} gets filled.")

                        break

                if has_order_filled:

                    is_extra = self.use_extra_execution and i == len(self.strategy_executions)-1

                    time_now = self.current_time()

                    if is_extra:
                        assert(filled_order.order_type() == OrderType.TAKE_PROFIT_EXIT)

                        message_title = f"Extra strategy of crypto {self.currency_coinbase} exits at take profit price."
                        message = f"Extra strategy of crypto {self.currency_coinbase} {self.parse_position(strategy_execution.side)} position exits " +\
                                  f"at take profit price {strategy_execution.take_profit_price} with actual filled price {filled_price}"

                        self.log_msg(message_title)
                        self.log_msg(message)
                        if not print_email_message_to_file:
                            sendEmail(message_title, message, is_alternative=True)

                        self.log_msg(strategy_msg + "Exit this execution")
                        ret_msg = strategy_execution.exit_execution(execution_exit_time=time_now,
                                                          execution_exit_price=strategy_execution.take_profit_price,
                                                          is_signal_exit=False,
                                                          is_extra_execution=True)
                        self.log_msg(ret_msg)

                        self.log_msg(strategy_msg + "Exit this prod execution")
                        ret_msg = strategy_execution.exit_execution_prod(prod_execution_exit_price=filled_price)

                        self.log_msg(ret_msg)

                        self.finish_execution_row(strategy_execution.strategy_id, strategy_execution)

                        del self.execution2order[strategy_execution.strategy_id]

                        #TODO: Write finished execution to persistence
                    else:

                        assert(filled_order.order_type() in [OrderType.STOP_ENTER, OrderType.STOP_LOSS_EXIT])

                        if filled_order.order_type() == OrderType.STOP_ENTER:

                            message_title = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} hits take profit price."
                            message = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} {self.parse_position(strategy_execution.side)} position hits " + \
                                      f"take profit price {strategy_execution.take_profit_price} with actual filled price {filled_price}"

                            self.log_msg(message_title)
                            self.log_msg(message)
                            if not print_email_message_to_file:
                                sendEmail(message_title, message, is_alternative=True)

                            self.log_msg(strategy_msg + "Exit this execution")
                            ret_msg = strategy_execution.exit_execution(execution_exit_time=time_now,
                                                              execution_exit_price=strategy_execution.take_profit_price,
                                                              is_signal_exit=False,
                                                              is_extra_execution=False)
                            self.log_msg(ret_msg)

                            self.log_msg(strategy_msg + "Exit this prod execution")
                            ret_msg = strategy_execution.exit_execution_prod(prod_execution_exit_price=filled_price)

                            self.log_msg(ret_msg)

                            self.finish_execution_row(strategy_execution.strategy_id, strategy_execution)

                            # TODO: Write finished execution to persistence

                            self.log_msg(strategy_msg + "Update to the next execution:")
                            ret_msg = strategy_execution.update_to_next_execution(entry_time=time_now, increased_size=filled_order.order_size())

                            self.log_msg(ret_msg)

                            for order in order_list:
                                if order.aux_order_type() == OrderType.STOP_LOSS_EXIT:
                                    #Cancel this stop loss order because we have reached take profit and re-entered

                                    message_title = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} cancels its stop loss order."
                                    message = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} cancels its stop loss order of size {order.order_size()}"

                                    self.log_msg(message_title)
                                    self.log_msg(message)
                                    if not print_email_message_to_file:
                                        sendEmail(message_title, message, is_alternative=True)

                                    try:
                                        cancel_response = self.coinbase_client.cancel_orders(order_ids=[order.order_id])
                                        print(cancel_response)
                                    except Exception as e:
                                        print("Error:", e)



                            stop_entry_order_id, stop_entry_order_size = self.place_stop_entry_order(strategy_execution)

                            stop_loss_order_id, stop_loss_order_size = self.place_stop_loss_order(strategy_execution)

                            order_list = [Order(stop_entry_order_id, stop_entry_order_size, OrderType.STOP_ENTER),
                                                           Order(stop_loss_order_id, stop_loss_order_size, OrderType.STOP_LOSS_EXIT)]

                            self.execution2order[strategy_execution.strategy_id] = order_list

                            new_execution_row = self.generate_new_execution_row(trade_id = self.max_long_trade_id if strategy_execution.side == 1 else self.max_short_trade_id,
                                                                            strategy_id = strategy_execution.strategy_id,
                                                                            strategy_execution = strategy_execution,
                                                                            order_list = order_list)
                            self.append_new_execution_row(new_execution_row)


                            self.log_msg("execution2order now is:")
                            for stra_id, o_list in self.execution2order.items():
                                self.log_msg(f"strategy_id={stra_id}")
                                self.log_msg("order_list:")
                                for order in o_list:
                                    self.log_msg(order)

                            #self.log_msg(self.execution2order)


                        elif filled_order.order_type() == OrderType.STOP_LOSS_EXIT:

                            message_title = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} hits stop loss price."
                            message = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} {self.parse_position(strategy_execution.side)} position hits " + \
                                      f"stop loss price {strategy_execution.take_loss_price} with actual filled price {filled_price}"

                            self.log_msg(message_title)
                            self.log_msg(message)
                            if not print_email_message_to_file:
                                sendEmail(message_title, message, is_alternative=True)

                            self.log_msg(strategy_msg + "Exit this execution")
                            ret_msg = strategy_execution.exit_execution(execution_exit_time=time_now,
                                                              execution_exit_price=strategy_execution.take_loss_price,
                                                              is_signal_exit=False,
                                                              is_extra_execution=False)
                            self.log_msg(ret_msg)

                            self.log_msg(strategy_msg + "Exit this prod execution")
                            ret_msg = strategy_execution.exit_execution_prod(prod_execution_exit_price=filled_price)

                            self.log_msg(ret_msg)

                            self.finish_execution_row(strategy_execution.strategy_id, strategy_execution)

                            # TODO: Write finished execution to persistence

                            for order in order_list:
                                if order.aux_order_type() == OrderType.STOP_ENTER:
                                    #Cancel this stop enter order because we have reached take profit and re-entered

                                    message_title = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} cancels its stop entry order."
                                    message = f"Strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase} cancels its stop entry order of size {order.order_size()}"

                                    self.log_msg(message_title)
                                    self.log_msg(message)
                                    if not print_email_message_to_file:
                                        sendEmail(message_title, message, is_alternative=True)

                                    try:
                                        cancel_response = self.coinbase_client.cancel_orders(order_ids=[order.order_id])
                                        print(cancel_response)
                                    except Exception as e:
                                        print("Error:", e)

                            del self.execution2order[strategy_execution.strategy_id]


                            self.log_msg("execution2order now is:")
                            for stra_id, o_list in self.execution2order.items():
                                self.log_msg(f"strategy_id={stra_id}")
                                self.log_msg("order_list:")
                                for order in o_list:
                                    self.log_msg(order)

                            #self.log_msg(self.execution2order)




    def place_stop_loss_order(self, strategy_execution: StrategyExecution):

        try:
            client_order_id = self.generate_client_order_id()

            stop_price = self.calc_never_reached_stop_price(strategy_execution.execution_entry_price, strategy_execution.side,
                                                            is_stop_loss=False)

            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                         client_order_id=client_order_id,
                                                         side=self.opposite_side(self.parse_side(strategy_execution.side)),
                                                         order_configuration={
                                                             "trigger_bracket_gtc": {
                                                                 "base_size": str(round(strategy_execution.prod_size, self.size_decimal)),
                                                                 "limit_price": str(round(stop_price, self.price_decimal)),
                                                                 "stop_trigger_price": str(round(strategy_execution.take_loss_price, self.price_decimal))
                                                             }
                                                         },
                                                         leverage="10",
                                                         margin_type="CROSS",
                                                         retail_portfolio_id=self.coinbase_portfolio_id
                                                         )

            self.log_msg(f"Order placed: {response}")

            message_title = f"Place stop loss order for strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase}"
            message = f"Place stop loss order of {round(strategy_execution.prod_size, self.size_decimal)} units at take profit price " + \
                      f"{round(strategy_execution.take_profit_price, self.price_decimal)} for {self.parse_side(strategy_execution.side)} order of strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase}"

            self.log_msg(message_title)
            self.log_msg(message)
            if not print_email_message_to_file:
                sendEmail(message_title, message, is_alternative=True)


        except Exception as e:
            self.log_msg(f"Order failed: {e}")

        stop_trigger_order_id = response['success_response']['order_id']

        return (stop_trigger_order_id, strategy_execution.prod_size)


    def place_stop_entry_order(self, strategy_execution: StrategyExecution):

        try:
            client_order_id = self.generate_client_order_id()

            size = strategy_execution.calc_increased_size_when_take_profit()
            response = self.coinbase_client.create_order(product_id=self.currency_coinbase,
                                                         client_order_id=client_order_id,
                                                         side=self.parse_side(strategy_execution.side),
                                                         order_configuration={
                                                             "stop_limit_stop_limit_gtc": {
                                                                 "base_size": str(round(size, self.size_decimal)),
                                                                 "limit_price": str(round(self.calc_buffer_limit_price(strategy_execution.take_profit_price,self.target_side), self.price_decimal)),
                                                                 "stop_price": str(round(strategy_execution.take_profit_price, self.price_decimal))
                                                             }
                                                         },
                                                         leverage="10",
                                                         margin_type="CROSS",
                                                         retail_portfolio_id=self.coinbase_portfolio_id
                                                         )

            self.log_msg(f"Order placed: {response}")

            message_title = f"Place stop entry order for strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase}"
            message = f"Place stop entry {self.parse_side(strategy_execution.side)} order of {round(size, self.size_decimal)} units at take profit price " + \
                      f"{round(strategy_execution.take_profit_price, self.price_decimal)} for strategy {strategy_execution.strategy_id} execution {strategy_execution.execution_id} of crypto {self.currency_coinbase}"

            self.log_msg(message_title)
            self.log_msg(message)
            if not print_email_message_to_file:
                sendEmail(message_title, message, is_alternative=True)


        except Exception as e:
            self.log_msg(f"Order failed: {e}")

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
            return entry_price * 1.01
        else:
            return entry_price * 0.99


    def open_executions(self, target_position, entry_time, strategy_executions = []):

        position_side = "long" if target_position > 0 else "short"
        self.log_msg(f"Open executions for {position_side} position {target_position} at time {entry_time}")

        self.target_position = target_position #This is sided
        self.target_side = 'BUY' if self.target_position > 0 else 'SELL'
        self.entry_time = entry_time
        self.new_strategy_executions = strategy_executions

        self.log_msg("Executions opened are:")
        for execution in strategy_executions:
            self.log_msg(execution)

        self.new_position_opened = True

    def set_open_position_price(self, entry_price):

        self.log_msg(f"Open position fill price = {entry_price}")
        self.open_position_fill_price = entry_price



    def close_executions(self, position_to_close, exit_time, signal_exit_price):

        self.position_to_close = position_to_close
        self.side_to_close = 'BUY' if self.position_to_close > 0 else 'SELL'
        self.exit_time = exit_time
        self.signal_exit_price = signal_exit_price

        self.log_msg("To close " + self.side_to_close + " position of " + str(abs(self.position_to_close)) + " units at time " + str(exit_time) + " at reference price " + str(signal_exit_price))

        self.old_position_closed = True


    def set_close_position_price(self, exit_price):

        self.log_msg("Close position fill price = " + str(exit_price))
        self.close_position_fill_price = exit_price


    def log_msg(self, msg):

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #current_time = (datetime.now() + timedelta(seconds = 28800)).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(msg, pd.DataFrame):
            print('[' + current_time + ' ' + self.currency_coinbase + ']  \n' + str(msg), file = self.log_fd)
        else:
            print('[' + current_time + ' ' + self.currency_coinbase + ']  ' + str(msg), file=self.log_fd)

        self.log_fd.flush()

