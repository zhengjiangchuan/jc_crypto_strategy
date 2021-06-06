import time
import os
import signal
from datetime import datetime
import os
from multiprocessing import Process
from util import sendEmail

import shutil


root_folder = "C:\\Forex\\formal_trading"

#dest_folder = "C:\\Forex\\all_charts_updated2_realtime_debug_correct"

dest_folder = "C:\\Forex\\all_simple_charts_updated2_realtime_debug_correct_new_try3_compare6"

#dest_folder = "C:\\Forex\\all_charts_updated2_16RedefineFalseSignalEntryBar"

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

symbol_folders = [os.path.join(root_folder, file) for file in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, file))]

for symbol_folder in symbol_folders:


    # if symbol_folder[-6:] not in ['EURNZD', 'AUDNZD', 'GBPNZD', 'AUDUSD', 'EURJPY']:
    #     continue

    print(symbol_folder)
    chart_folder = os.path.join(symbol_folder, "simple_chart")

    files = os.listdir(chart_folder)
    # if len(files) > 1:
    #     files = files[1:]

    for file in files:
        file_path = os.path.join(chart_folder, file)

        print("file_path = " + file_path)
        print("dest_folder = " + dest_folder)
        shutil.copy2(file_path, dest_folder)


