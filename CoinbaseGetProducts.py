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


api_key = "organizations/e7135013-aa60-482a-a55e-c60a6a970c81/apiKeys/3cc1a37a-8aaf-44e9-a0d6-e3c502f47524"
api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEICPGFLZfZPt8Weus5uEbBM5byLec3rjtgwzayetQHAv0oAoGCCqGSM49\nAwEHoUQDQgAEQyvr+7tyDfjDd/8GfllUiN7SS9iVDUEr12IspRwJuwLPzLm/FcYr\nDdDYm0vsH8JQS0qKxWhZRpkN6eK3Kp0rSA==\n-----END EC PRIVATE KEY-----\n"



client = RESTClient(api_key = api_key,
                    api_secret= api_secret)

products = client.get_products(product_type='FUTURE', contract_expiry_type='PERPETUAL')

product_dict = products.to_dict()

print("product_dict:")
print(product_dict)
