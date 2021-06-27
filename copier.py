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

#dest_folder = "C:\\Forex\\new_experiments\\all_simple_charts_fire2_fire3_100_filtered_by_fire2_fire3_200_win_check"

#dest_folder = "C:\\Forex\\new_experiments\\0611\\fire2_fire3_100_filtered_by_fire2_fire3_200_adjust_entry_too_high_cond"


dest_folder = "C:\\Forex\\new_experiments\\0627\\stop_loss_improve_modifyGuppyClose_little_modify"

#dest_folder = "C:\\Forex\\new_experiments\\0626\\benchmark_clear_close_positions_improve_with_phase1"

#dest_folder = "C:\\Users\\User\\Dropbox\\forex_real_time_compare"

#dest_folder = "C:\\Users\\User\\Dropbox\\forex_results2"

#dest_folder = "C:\\Forex\\new_experiments\\0611\\all_simple_charts_fire2_fire3_100_filtered_by_fire2_fire3_200_high_low_filter_simple"


#dest_folder = "C:\\Forex\\all_charts_updated2_16RedefineFalseSignalEntryBar"

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

for file in os.listdir(dest_folder):
    file_path = os.path.join(dest_folder, file)
    os.remove(file_path)


symbol_folders = [os.path.join(root_folder, file) for file in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, file))]

for symbol_folder in symbol_folders:


    # if symbol_folder[-6:] not in ['EURCHF']:
    #     continue

    print(symbol_folder)
    chart_folder = os.path.join(symbol_folder, "simple_chart")

    files = os.listdir(chart_folder)
    # if len(files) == 3:
    #     files = files[1:]

    for file in files:
        file_path = os.path.join(chart_folder, file)

        print("file_path = " + file_path)
        print("dest_folder = " + dest_folder)
        shutil.copy2(file_path, dest_folder)


