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

#portfolio_df = pd.read_csv("C:\\JCForex_prod\\portfolio_construction\\optimal_portfolio_by_date.csv")

forex_dir = "C:\\JCForex_prod"

currency_df = pd.read_csv(os.path.join(forex_dir, "currency.csv"))

currency_list = currency_df['currency'].tolist()

portfolio_df = pd.read_csv("C:\\JCForex_prod\\portfolio_construction\\optimal_portfolio_by_start_end_date_2month.csv")

portfolios = []

print("portfolio_df")
print(portfolio_df)

for i in range(portfolio_df.shape[0] - 1):

    row = portfolio_df.iloc[i]
    by_date = row['by_date']
    portfolio = row['optimal_currency_list'].split(',')

    portfolios += [portfolio]

    # print("by_date = " + by_date)
    # print(portfolio)
    # print("number = " + str(len(portfolio)))
    # print("")

allow_loss_num = 2

selected_currencies = []
for currency in currency_list:
    loss_num = len([portfolio for portfolio in portfolios if currency not in portfolio])
    if loss_num <= allow_loss_num:
        selected_currencies += [currency]

print("selected currencies are:")
print(selected_currencies)

