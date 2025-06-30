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

client_order_id = f"order_{uuid.uuid4()}"
print("client_order_id = " + client_order_id)

try:
    # response = client.create_order(product_id="ADA-PERP-INTX",     #BTC-USDC is the correct product id
    #                                client_order_id=client_order_id,
    #                                side="BUY",
    #                                order_configuration={
    #                                    "trigger_bracket_gtc":{
    #                                        "base_size" : "100",
    #                                        "limit_price" : "0.6", #This is take_profit_price for a sell position (side should be buy), should be lower than current market price
    #                                        "stop_trigger_price" : "0.9" #This is stop_loss_price for a sell position (side should be buy), should be higher than current market price
    #
    #                                    }
    #                                },
    #                                leverage="10",
    #                                margin_type = "CROSS"
    #                                #retail_portfolio_id="0194271a-bd95-7ba7-a028-6561a970128b"
    #                                )

    # response = client.create_order(product_id="AVAX-PERP-INTX",  # BTC-USDC is the correct product id
    #                                client_order_id=client_order_id,
    #                                side="SELL",
    #                                order_configuration={
    #                                    "trigger_bracket_gtc": {
    #                                        "base_size": "28.544",
    #                                        "limit_price": "19",
    #                                        # This is take_profit_price for a buy position (side should be sell), should be higher than current market price
    #                                        "stop_trigger_price": "17"
    #                                        # This is stop_loss_price for a buy position (side should be sell), should be lower than current market price
    #
    #                                    }
    #                                },
    #                                leverage="10",
    #                                margin_type="CROSS"
    #                                # retail_portfolio_id="0194271a-bd95-7ba7-a028-6561a970128b"
    #                                )

    response = client.create_order(product_id="ADA-PERP-INTX",  # BTC-USDC is the correct product id
                                   client_order_id=client_order_id,
                                   side="BUY",
                                   order_configuration={
                                       "stop_limit_stop_limit_gtc": {
                                           "base_size": "200",
                                           "limit_price": "0.71",
                                           "stop_price": "0.7"
                                       }
                                   },
                                   leverage="10",
                                   margin_type="CROSS"
                                   # retail_portfolio_id="0194271a-bd95-7ba7-a028-6561a970128b"
                                   )

    print(f"Order placed: {response}")
except Exception as e:
    print(f"Order failed: {e}")












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