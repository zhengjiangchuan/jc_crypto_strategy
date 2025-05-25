import time
import os
import signal
from datetime import datetime
import os
import numpy as np
import random
import sys
import pandas as pd



import time
import os
import signal
from datetime import datetime
import os
from multiprocessing import Process
from util import sendEmail
# from datetime import timedelta
# import pandas as pd


from optparse import OptionParser
parser = OptionParser()
parser.add_option("-c", "--currency", dest="currency_pair", default = "all",
                   help="Currency Pair to run")
parser.add_option("-a", "--alternative", dest="alternative", default = "n",
                 help="Use alternative account")

(options, args) = parser.parse_args()

currency_to_run = options.currency_pair
alternative = options.alternative

print("currency_to_run = " + currency_to_run)
print("alternative = " + alternative)



# import smtplib
# from email.header import Header
# from email.mime.text import MIMEText
#
#
# mail_host = "smtp.163.com"
# mail_user = "glzxely123"
# mail_pass = "10331861oO"
#
# sender = 'glzxely123@163.com'
# receivers = ['jczheng198508@gmail.com']
#
# def sendEmail(title, content):
#     message = MIMEText(content, 'plain', 'utf-8')
#     message['From'] = "{}".format(sender)
#     message['To'] = ",".join(receivers)
#     message['Subject'] = title
#
#     try:
#         smtpObj = smtplib.SMTP_SSL(mail_host, 465)
#         smtpObj.login(mail_user, mail_pass)
#         print("Sending Email....")
#         smtpObj.sendmail(sender, receivers, message.as_string())
#         print("mail has been send successfully.")
#     except smtplib.SMTPException as e:
#         print(e)
#
# sendEmail("Guoji", "I Love Guoji so much\n I reall love Guoji gay")

#
# for file in os.listdir(root_folder):
#     #print(file)
#     if os.path.isdir(os.path.join(root_folder, file)):
#
#         symbol_folder = os.path.join(root_folder, file)
#         data_folder = os.path.join(symbol_folder, 'data')
#         data_files = os.listdir(data_folder)
#         for data_file in data_files:
#             if len(data_file) != 10:
#                 os.remove(os.path.join(data_folder, data_file))
#                 print("Remove " + data_file)
#
# sys.exit(0)
#
# communicate_files = [file for file in os.listdir(root_folder) if "communicate" in file]
# communicate_nums = [int(communicate_file[len('communicate'):-len('.txt')]) for communicate_file in communicate_files]
#
# max_idx = np.array(communicate_nums).argmax()
#
# communicate_file = os.path.join(root_folder, communicate_files[max_idx])
#
# print("    Child communicate_file = " + communicate_file)
# print("    Child does this file exists? " + str(os.path.exists(communicate_file)))
#
# i = 0
# while True:
#
#     i += 1
#     time.sleep(5)
#
#     print("    i=" + str(i) + "Child does this file " + communicate_file + " exists? " + str(os.path.exists(communicate_file)))
#
#     now = datetime.now()
#     now_str = now.strftime("%Y-%m-%d %H:%M:%S")
#     print("    Child Process " + str(random_id) + ": Current time: " + now_str + " i = " + str(i))
#     fd = open(communicate_file, 'w')
#     print(now_str, file = fd)
#     fd.close()
#
#     print("    ii=" + str(i) + "Child does this file " + communicate_file + " exists? " + str(os.path.exists(communicate_file)))
#
#     if i >= 5:
#         time.sleep(1800)








####################################################################################################################################












