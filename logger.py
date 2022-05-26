"""Logging configuration and functions"""
import logging
from datetime import datetime
from multiprocessing import Queue
from logging import Formatter, FileHandler, StreamHandler
from logging.handlers import QueueHandler, QueueListener
from colorlog import ColoredFormatter

LOG_FILE: str = f"logs/{datetime.now()}.log"
COLOR_LOG_FORMAT: str = "%(log_color)s%(levelname)s | %(asctime)s @  %(processName)s:%(funcName)s > %(message)s%(reset)s"
LOG_FORMAT: str = "%(levelname)s | %(asctime)s @  %(processName)s:%(funcName)s > %(message)s"
LOG_LEVEL = logging.DEBUG


def init_logger(queue: Queue) -> QueueListener:
    """
    Creates a QueueListener that will process all log messages throughout the application

    Parameters
    ----------
        queue: Queue
            Queue object that holds logging processes

    Returns
    -------
        QueueListener
            Object to process log messages
    """
    console_formatter: Formatter = ColoredFormatter(COLOR_LOG_FORMAT)
    file_formatter: Formatter = logging.Formatter(LOG_FORMAT)

    file: FileHandler = logging.FileHandler(LOG_FILE, "a")
    file.setFormatter(file_formatter)

    console: StreamHandler = logging.StreamHandler()
    console.setFormatter(console_formatter)

    return QueueListener(queue, file, console)


def worker_configurer(queue: Queue):
    """
    When this is run, it configures the logger of this process to submit logs to the logging process (QueueListener)

    Parameters
    ----------
        queue: Queue
            Queue object that holds logging processes
    """
    queue_handler: QueueHandler = QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel(LOG_LEVEL)
