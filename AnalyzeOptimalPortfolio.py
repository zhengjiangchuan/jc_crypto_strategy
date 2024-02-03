import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.ticker as ticker
from datetime import datetime
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

import warnings

from functools import reduce

warnings.filterwarnings("ignore")

portfolio_df = pd.read_csv("C:\\JCForex_prod\\portfolio_construction\\optimal_portfolio_by_date.csv")

for i in range(portfolio_df.shape[0]):

    row = portfolio_df.iloc[i]
    by_date = row['by_date']
    portfolio = row['optimal_currency_list'].split(',')

    print("by_date = " + by_date)
    print(portfolio)
    print("number = " + str(len(portfolio)))
    print("")
