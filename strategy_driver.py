import time
import os
import signal
from datetime import datetime
import os
from multiprocessing import Process
from util import sendEmail
import pandas as pd



def start_strategy(currency):

    print("python vegas_strategy.py " + currency)
    os.system("python vegas_strategy.py -b " + currency)


if __name__ == '__main__':

    root_folder = "C:\\Forex\\formal_trading"

    currency_file = os.path.join(root_folder, "currency.csv")

    currency_df = pd.read_csv(currency_file)

    currency_list = list(currency_df['currency'])



    for currency in currency_list:
        son_process = Process(target=start_strategy, args = (currency,))
        son_process.start()