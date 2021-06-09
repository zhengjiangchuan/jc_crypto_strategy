import time
import os
import signal
from datetime import datetime
import os
import numpy as np
import random
import sys

random_id = int(random.random()*10000)

root_folder = "C:\\Forex\\formal_trading"


for file in os.listdir(root_folder):
    #print(file)
    if os.path.isdir(os.path.join(root_folder, file)):

        symbol_folder = os.path.join(root_folder, file)
        data_folder = os.path.join(symbol_folder, 'data')
        data_files = os.listdir(data_folder)
        for data_file in data_files:
            if len(data_file) != 10:
                os.remove(os.path.join(data_folder, data_file))
                print("Remove " + data_file)

sys.exit(0)

communicate_files = [file for file in os.listdir(root_folder) if "communicate" in file]
communicate_nums = [int(communicate_file[len('communicate'):-len('.txt')]) for communicate_file in communicate_files]

max_idx = np.array(communicate_nums).argmax()

communicate_file = os.path.join(root_folder, communicate_files[max_idx])

print("    Child communicate_file = " + communicate_file)
print("    Child does this file exists? " + str(os.path.exists(communicate_file)))

i = 0
while True:

    i += 1
    time.sleep(5)

    print("    i=" + str(i) + "Child does this file " + communicate_file + " exists? " + str(os.path.exists(communicate_file)))

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("    Child Process " + str(random_id) + ": Current time: " + now_str + " i = " + str(i))
    fd = open(communicate_file, 'w')
    print(now_str, file = fd)
    fd.close()

    print("    ii=" + str(i) + "Child does this file " + communicate_file + " exists? " + str(os.path.exists(communicate_file)))

    if i >= 5:
        time.sleep(1800)




