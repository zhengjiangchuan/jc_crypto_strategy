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
accounts = client.get_accounts()

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

perps_balances = client.get_perps_portfolio_balances(portfolio_uuid='0194271a-bd95-7ba7-a028-6561a970128b')
perps_summary = client.get_perps_portfolio_summary(portfolio_uuid='0194271a-bd95-7ba7-a028-6561a970128b')

print("balances:")
print(perps_balances.to_dict())
print("perps_summary:")
print(perps_summary.to_dict())
product = client.get_product("BTC-USDC")
print("product:")
print(product.to_dict())
btc_usd_price = float(product["price"])

print("btc_usd_price: " + str(btc_usd_price))

client_order_id = f"order_{uuid.uuid4()}"

print("client_order_id = " + client_order_id)

try:
    response = client.create_order(product_id="ADA-PERP-INTX",     #BTC-USDC is the correct product id
                                   client_order_id=client_order_id,
                                   side="SELL",
                                   order_configuration={
                                       "limit_limit_gtc":{
                                           "base_size" : "600",
                                           "limit_price" : "0.8"

                                       }
                                   },
                                   leverage="10"
                                   #margin_type = "CROSS"#,
                                   #retail_portfolio_id="0194271a-bd95-7ba7-a028-6561a970128b"
                                   )
    print(f"Order placed: {response}")
except Exception as e:
    print(f"Order failed: {e}")


print("order is")
print(response['success_response']['order_id'])












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