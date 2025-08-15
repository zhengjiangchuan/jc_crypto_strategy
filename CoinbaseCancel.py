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

api_key, api_secret = get_api_keys(is_alternative=True)




client = RESTClient(api_key = api_key,
                    api_secret= api_secret)
accounts = client.get_accounts()

products = client.get_products()
#
# print("Product")
# print(products[0])

print(dumps(accounts.to_dict(), indent=2))

# account = client.get_account(account_uuid="e935a58a-2e1d-5675-8cb0-874fb67b143c")
# print("account is: ")
# print(account)
#
# client.get_perps_portfolio_summary(portfolio_uuid='0194271a-bd95-7ba7-a028-6561a970128b')

permission = client.get_api_key_permissions()
print("permissions:")
print(permission.to_dict())

#perps_balances = client.get_perps_portfolio_balances(portfolio_uuid='0194271a-bd95-7ba7-a028-6561a970128b')
#perps_summary = client.get_perps_portfolio_summary(portfolio_uuid='0194271a-bd95-7ba7-a028-6561a970128b')

# print("balances:")
# print(perps_balances.to_dict())
# print("perps_summary:")
# print(perps_summary.to_dict())
# product = client.get_product("BTC-USDC")
# btc_usd_price = float(product["price"])

#print("btc_usd_price: " + str(btc_usd_price))

try:
    cancel_response = client.cancel_orders(order_ids = ["74a943be-705d-4f9f-821e-6f197cb034ee"])
    print("Cancel Response:")
    print(cancel_response)
except Exception as e:
    print("Error:", e)
















# order = client.limit_order_gtc_buy(
#     client_order_id="00000002",
#     product_id="BTC-USD",
#     base_size="0.001",
#     limit_price="61000"
# )
#
# if order['success']:
#     order_id = order['success_response']['order_id']
#     print("succeed order id = " + str(order_id))
# else:
#     error_response = order['error_response']
#     print(error_response)