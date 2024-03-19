import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#This line will create a log file with the current date and time.
#datetime.now().strftime('%m_%d_%Y_%H_%M_%S') will get the current date and time and convert it into the string format.
#f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" will create a log file with the current date and time.

logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
#This line will create a log file with the current date and time in the logs folder.
#os.path.join(os.getcwd(), 'logs', LOG_FILE) will join the current working directory, logs folder and log file name.

os.makedirs(logs_path, exist_ok = True)
#This line will create a directory with the log file name and exist_ok = True will not raise an error if the directory already exists.

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
#This line will join the log file path with the log file name and create a log file.

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = '[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level = logging.INFO,
)
#This line will configure the logging module with the log file path, format, and level.
#filename = LOG_FILE_PATH will set the log file path.
#format = '%(asctime)s - %(lineno)s - %(name)s - %(levelname)s - %(message)s' will set the format of the log message with the date and time, line number, name, level, and message.
#level = logging.INFO will set the level of the log message to INFO which means all the messages with level INFO or higher will be logged.


if __name__ == '__main__':
    logging.info('This is a test message for the logger')