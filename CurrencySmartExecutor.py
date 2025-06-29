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
from instrument_trader import StrategyExecution

class CurrencySmartExecutor:

    def __init__(self, currency, coinbase_client: Optional[RESTClient] = None, coinbase_decimal = 0):
        self.currency = currency

        self.strategy_executions = []


    def manage_executions(self):

        pass

    def open_executions(self, strategy_executions = []):
        self.strategy_executions = strategy_executions

        for strategy_execution in self.strategy_executions:
            execution: Optional[StrategyExecution] = strategy_execution



    def close_executions(self):

        pass




