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
from CoinbaseUtil import *

api_key, api_secret = get_api_keys()



client = RESTClient(api_key = api_key,
                    api_secret= api_secret)

products = client.get_products(product_type='FUTURE', contract_expiry_type='PERPETUAL')

product_dict = products.to_dict()

print("product_dict:")
print(product_dict)
