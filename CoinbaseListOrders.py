import datetime
import pandas as pd
import math
import copy
from functools import reduce
import numpy as np
import sys
import os

import talib
from coinbase.rest import RESTClient
from CoinbaseUtil import *

from coinbase.rest import RESTClient
from json import dumps
import uuid


api_key, api_secret = get_api_keys()




client = RESTClient(api_key = api_key,
                    api_secret= api_secret)

open_orders = client.list_orders(order_status = "OPEN").orders

print("open orders num = " + str(len(open_orders)))

for order in open_orders:
    print("Order:")


    print("order_id: " + str(order.order_id))
    print("product_id: " + str(order.product_id))
    print("client_order_id: " + str(order.client_order_id))

    order_configuration = order.order_configuration
    if hasattr(order_configuration, "limit_limit_gtc"):
        limit_limit_gtc = order_configuration.limit_limit_gtc
        print("limit_price = " + str(limit_limit_gtc.limit_price))

    print("")