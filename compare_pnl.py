def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import talib

import matplotlib.lines as mlines
import datetime
import pandas as pd
import math
import copy
from functools import reduce
import numpy as np
import sys
import os

from util import *
import gzip

from datetime import datetime, timedelta
import math

from optparse import OptionParser
import matplotlib.ticker as ticker

import urllib.request

from io import StringIO


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)


import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import timedelta
from matplotlib.ticker import Formatter
import  matplotlib.ticker as plticker
import matplotlib.dates as mdates

data_folder = "C:\\Forex\\formal_trading\\"

pnl_folder = os.path.join(data_folder, 'pnl', 'pnl0720')

pnl_folder1 = os.path.join(pnl_folder, 'pnl_summary_spread15_innovativeFire2new_marginLevel2.5_3pm')

pnl_folder2 = os.path.join(pnl_folder, 'pnl_summary_spread15_innovativeFire2new_marginLevel2.5_3pm_check2')



pnl_file1 = os.path.join(pnl_folder1, 'performance_summary.csv')
pnl_file2 = os.path.join(pnl_folder2, 'performance_summary.csv')

pnl_df1 = pd.read_csv(pnl_file1)
pnl_df2 = pd.read_csv(pnl_file2)

compare_df = pnl_df1[['symbol']]

for col in pnl_df1.columns:
    if col != 'symbol':
        compare_df[col + '1'] = pnl_df1[col]
        compare_df[col + '2'] = pnl_df2[col]
        compare_df[col + '_diff'] = compare_df[col + '2'] - compare_df[col + '1']

summary_dict = {'symbol' : ['summary']}
for col in pnl_df1.columns:
    if col != 'symbol':
        summary_dict[col + '1'] = compare_df[col + '1'].mean()
        summary_dict[col + '2'] = compare_df[col + '2'].mean()
        summary_dict[col + '_diff'] = compare_df[col + '_diff'].mean()

compare_df = pd.concat([compare_df, pd.DataFrame(summary_dict)])


compare_df.to_csv(os.path.join(pnl_folder, 'performance_comparison_innovativeFire2_newDoubleCheck.csv'), index = False)

print("compare_df:")
print(compare_df)





