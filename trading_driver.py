import time
import os
import signal
from datetime import datetime
import os
from multiprocessing import Process
from util import sendEmail
from datetime import timedelta


def wait_for_trigger():

    current_time = datetime.now()

    temp_time = current_time + timedelta(seconds=3600)
    next_hour = datetime(temp_time.year, temp_time.month, temp_time.day, temp_time.hour, 0, 0)

    print("Next hour: " + str(next_hour))

    seconds_remaining = (next_hour - current_time).seconds

    print("Waiting for " + str(seconds_remaining) + " seconds")

    sleep_seconds = 5
    total_sleep_seconds = 0
    while seconds_remaining > 0:
        actual_sleep_seconds = seconds_remaining if seconds_remaining < sleep_seconds else sleep_seconds
        print("Sleep " + str(actual_sleep_seconds))
        time.sleep(actual_sleep_seconds)
        total_sleep_seconds += actual_sleep_seconds
        now = datetime.now()
        print("Current time: " + now.strftime("%Y-%m-%d %H:%M:%S"))
        seconds_remaining -= actual_sleep_seconds
        print("seconds_remaining = " + str(seconds_remaining))
        print("total_sleep_seconds = " + str(total_sleep_seconds))

    now = datetime.now()
    while (now - next_hour).seconds < 2:
        print("Now is " + now.strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(1)
        now = datetime.now()

    sendEmail("Trading program still alive", "")

    return

def start_trader():
    print("Parent starts child process")
    os.system("python vegas_strategy_once.py")


if __name__ == '__main__':
    print("Hello Hutong")
    while True:

        print("Waiting for the next trigger")

        son_process = Process(target=start_trader)
        son_process.start()

        wait_for_trigger()