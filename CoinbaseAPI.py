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

api_key, api_secret = get_api_keys(is_alternative=False)


client = RESTClient(api_key = api_key,
                    api_secret= api_secret)
accounts = client.get_accounts()

account = accounts.accounts[0]
portfolio_id = account['retail_portfolio_id']
print("portfolio_id = " + str(portfolio_id))

#This is the correct one
# products = client.get_products()
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

perps_balances = client.get_perps_portfolio_balances(portfolio_uuid=portfolio_id)
perps_summary = client.get_perps_portfolio_summary(portfolio_uuid=portfolio_id)

print("balances:")
print(perps_balances.to_dict())
print("perps_summary:")
print(perps_summary.to_dict())
product = client.get_product("ADA-PERP-INTX")
print("product:")
print(product.to_dict())
btc_usd_price = float(product["price"])

print("btc_usd_price: " + str(btc_usd_price))

client_order_id = f"order_{uuid.uuid4()}"

print("client_order_id = " + client_order_id)

symbol = "ADA-PERP-INTX"
try:
    response = client.create_order(product_id="ADA-PERP-INTX",     #BTC-USDC is the correct product id
                                   client_order_id=client_order_id,
                                   side="BUY",
                                   order_configuration={
                                       "limit_limit_gtc":{
                                           "base_size" : "20",
                                           "limit_price" : "0.725"

                                       }
                                   },
                                   leverage="10",
                                   margin_type = "CROSS",
                                   retail_portfolio_id=portfolio_id
                                   )
    print(f"Order placed: {response}")
except Exception as e:
    print(f"Order failed: {e}")


print("order is")

order_id = response['success_response']['order_id']

print(order_id)


order = client.get_order(order_id = order_id).order
status = order['status']
filled_size = order['filled_size']
print("status = " + str(status))
print("filled_size = " + str(filled_size))



positions = client.list_perps_positions(portfolio_uuid=portfolio_id).positions
print(type(positions))
print("Positions: size = " + str(len(positions)))

for position in positions:
    print("product_id=" + position['product_id'])
    print("symbol=" + position['symbol'])
    print("position_side=" + position['position_side'])
    print("margin_type=" + position['margin_type'])
    print("net_size=" + position['net_size'])
    print("leverage=" + position['leverage'])
    unrealized_pnl = position['unrealized_pnl']
    print("unrealized_pnl=" + unrealized_pnl['value'] + unrealized_pnl['currency'])













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