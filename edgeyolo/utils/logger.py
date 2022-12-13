#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import inspect
import os
import sys
from loguru import logger


def init_logger(file_name=None):

    show_format = "<green>{time:YYYY-MM-DD hh:mm:ss}</green> | " \
                  "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"

    logger.remove()
    logger.add(sys.stdout, colorize=True, format=show_format)
    if file_name is not None:
        assert isinstance(file_name, str)
        save_format = "{time:YYYY-MM-DD hh:mm:ss} | <level>{level: <8}</level>| " \
                      "{name}:{function}:{line} - {message}"
        logger.add(file_name, format=save_format)


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
