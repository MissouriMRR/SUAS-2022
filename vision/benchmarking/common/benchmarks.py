"""
Enumeration for storing the types of benchmarks.
"""

from enum import Enum


class RunType(Enum):
    """
    The type of image to run the benchmark on.
    """

    BLANK_IMGS: str = "blank_imgs"
    IMG_SET: str = "img_set"


class BenchmarkType(Enum):
    """
    The benchmark that is being run.

    NOTE: Should only include benchmarks that have been implemented.
    """

    EMG_OBJECT: str = "emg_object"
    TEXT: str = "text"
