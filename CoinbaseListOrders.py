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


from coinbase.rest import RESTClient
from json import dumps
import uuid

#Perpetual products
#api_key = "organizations/e7135013-aa60-482a-a55e-c60a6a970c81/apiKeys/3c78feac-c110-4c9e-9735-16f58d59828e"
#api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIAA2HylG8UDwS+vf40Bv3NncEVzqHS7tW06hT/yYRWvioAoGCCqGSM49\nAwEHoUQDQgAEOug7rG6O3YCkx68Ef/nvMT1ybDqFIiX7ch1D1iQlQTh9Hfodpp5H\nha/0PGByLqGpmBvVW045AGvbMpLwlMnLnA==\n-----END EC PRIVATE KEY-----\n"


api_key = "organizations/e7135013-aa60-482a-a55e-c60a6a970c81/apiKeys/3cc1a37a-8aaf-44e9-a0d6-e3c502f47524"
api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEICPGFLZfZPt8Weus5uEbBM5byLec3rjtgwzayetQHAv0oAoGCCqGSM49\nAwEHoUQDQgAEQyvr+7tyDfjDd/8GfllUiN7SS9iVDUEr12IspRwJuwLPzLm/FcYr\nDdDYm0vsH8JQS0qKxWhZRpkN6eK3Kp0rSA==\n-----END EC PRIVATE KEY-----\n"

#Common products
#api_key = "organizations/e7135013-aa60-482a-a55e-c60a6a970c81/apiKeys/f9768c30-233d-429f-a030-a34c68e821a1"
#api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIIRsX6fWvodAKWiWdqxZq+jebkWyZxp5Bx9nf051cRv7oAoGCCqGSM49\nAwEHoUQDQgAEt0W1SRROiZwrl50RHXxR/lQHsisvhrg5yPLlDdsiPKsKEY+R/yaN\nHuGOsmAiTQcayEPKV05NUdzyAdfpdCMIhA==\n-----END EC PRIVATE KEY-----\n"

#api_secret = ""



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