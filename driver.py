import time
import os
import signal
from datetime import datetime
import os
from multiprocessing import Process
from util import sendEmail



def my_func():
    print("Parent starts child process")
    os.system("python vegas_strategy.py")


#fd = open(communicate_file, 'r')

# def child():
#     print('hello from child', os.getpid())
#     os._exit(0)

# def fork_child_process2():
#     pid = os.fork()
#     if pid == 0:
#         os.execlp('python', 'python', 'tester.py')
#         assert False, 'fork child process error!'
#     else:
#         print('hello from parent', os.getpid(), pid)
#     return pid

if __name__ == '__main__':

    root_folder = "C:\\Forex\\formal_trading"

    communicate_files = [file for file in os.listdir(root_folder) if "communicate" in file]

    print("Remove all communicate files")
    for communicate_file in communicate_files:
        os.remove(os.path.join(root_folder, communicate_file))

    max_file_id = 1
    communicate_file = os.path.join(root_folder, "communicate" + str(max_file_id) + ".txt")

    # print("communicate_file = " + communicate_file)
    # print("Exists? " + str(os.path.exists(communicate_file)))
    # print("Create it")
    fd = open(communicate_file, 'w')
    fd.close()
    #print("Exists? " + str(os.path.exists(communicate_file)))


    son_process = Process(target=my_func)
    son_process.start()

    print("Child pid = " + str(son_process.pid))

    initial_time = None
    while (True):

        is_dead = False
        print("Parent Process sleeps 5 seconds");
        time.sleep(5)

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(communicate_file):
            fd = open(communicate_file, 'r')
            line = fd.readline()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]

            print("Parent now = " + now_str)
            print("Parent reads child now = " + line)

            if len(line) < 10:

                if initial_time is None:
                    initial_time = now
                    continue
                else:
                    if (now - initial_time).seconds > 600:
                        is_dead = True
                        initial_time = None
                    else:
                        continue


            read_time = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
            child_dead_seconds = (now - read_time).seconds
            print("*********************************************************************Parent: child dead seconds = " + str(child_dead_seconds))
            if (now - read_time).seconds > 120 or is_dead:
                print("Parent spots that child process pid=" + str(son_process.pid) +  " is dead but not exit, kill it and re-start it")
                sendEmail("Trading program dead, restart it", "")
                son_process.kill()

                max_file_id += 1
                communicate_file = os.path.join(root_folder, "communicate" + str(max_file_id) + ".txt")

                print("************ communicate_file = " + communicate_file)

                fd = open(communicate_file, 'w')
                fd.close()

                son_process = Process(target=my_func)
                son_process.start()
                print("New child process pid=" + str(son_process.pid))


