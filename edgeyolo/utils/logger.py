import os
import sys
from loguru import logger


def init_logger(file_name=None):

    show_format = "<green>{time:YYYYMMDD_HHmmss}</green> " \
                  "<cyan>{name}:{line}</cyan> - <level>{message}</level>"

    logger.remove()
    logger.add(sys.stdout, colorize=True, format=show_format)
    if file_name is not None:
        assert isinstance(file_name, str)
        save_format = "{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level>| " \
                      "{name}:{function}:{line} - {message}"
        logger.add(file_name, format=save_format)
        f = file_name[:-len(file_name.split(".")[-1])]
        logger.add(f + "_warn.txt", level="WARNING")
        logger.add(f + "_err.txt", level="ERROR")


class NoPrint:

    def __init__(self, flag=True):
        self.flag = flag

    def __enter__(self):
        if self.flag:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.flag:
            sys.stdout.close()
            sys.stdout = self._original_stdout
