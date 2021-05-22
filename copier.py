import time
import os
import signal
from datetime import datetime
import os
from multiprocessing import Process
from util import sendEmail

import shutil


root_folder = "C:\\Forex\\trading"

dest_folder = "C:\\Forex\\all_charts_updated2"

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

symbol_folders = [os.path.join(root_folder, file) for file in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, file))]

for symbol_folder in symbol_folders:



    chart_folder = os.path.join(symbol_folder, "chart")
    for file in os.listdir(chart_folder)[1:]:
        file_path = os.path.join(chart_folder, file)

        shutil.copy2(file_path, dest_folder)


