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


#api_key, api_secret = get_api_keys()

api_key, api_secret = get_api_keys(is_alternative=True)


client = RESTClient(api_key = api_key,
                    api_secret= api_secret)

open_orders = client.list_orders(order_status = "OPEN").orders

print("open orders num = " + str(len(open_orders)))

for order in open_orders:
    print("Order:")


    print("order_id: " + str(order.order_id))
    print("size: " + str(order.outstanding_hold_amount))
    print("product_id: " + str(order.product_id))
    print("client_order_id: " + str(order.client_order_id))

    order_configuration = order.order_configuration
    if hasattr(order_configuration, "limit_limit_gtc"):
        limit_limit_gtc = order_configuration.limit_limit_gtc
        print("limit_price = " + str(limit_limit_gtc.limit_price))

    cancelled = False
    orderResponse = client.get_order(order_id=str(order.order_id))
    if hasattr(orderResponse, "order"):
        coinbaseorder = orderResponse.order
        if coinbaseorder is not None:
            status = coinbaseorder['status']
            print("status = " + str(status))
            if status == 'CANCELLED':
                cancelled = True

    print(f"cancelled = {cancelled}")

    print("")


# fully_filled = False
# orderResponse = client.get_order(order_id="74a943be-705d-4f9f-821e-6f197cb034ee")
# if hasattr(orderResponse, "order"):
#     coinbaseorder = orderResponse.order
#     if coinbaseorder is not None:
#         status = coinbaseorder['status']
#         filled_size = float(coinbaseorder['filled_size'])
#         filled_price = float(coinbaseorder['average_filled_price'])
#         print(f"status={status}, filled_size={filled_size}, filled_price={filled_price}")
#
#         # if status == 'FILLED' and filled_size == order.order_size():
#         #      fully_filled = True
#
#         #     return (fully_filled, filled_price)
#     else:
#         print("coinbaseorder is None")