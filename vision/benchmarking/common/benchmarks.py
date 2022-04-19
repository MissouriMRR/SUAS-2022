"""
Enumeration for storing the types of benchmarks.
"""

from enum import Enum


class BenchmarkType(Enum):
    """
    The benchmark that is being run.

    NOTE: Should only include benchmarks that have been implemented.
    """

    EMG_OBJECT: str = "emg_object"
    TEXT: str = "text"
