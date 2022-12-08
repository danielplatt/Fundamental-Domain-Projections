import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(filename):
    file_handler = TimedRotatingFileHandler(filename + '.log', when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name, with_logfile=False):
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG)

    logger.addHandler(get_console_handler())

    if with_logfile:
        logger.addHandler(get_file_handler(logger_name))

    logger.propagate = False

    return logger
