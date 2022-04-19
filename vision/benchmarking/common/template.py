"""
Template for benchmarking modules. Benchmark modules should
be a subclass Benchmark.
"""

from typing import List


class Benchmark:
    """
    Template for benchmark modules.
    """

    def __init__(self):
        self._times: List[float] = []

    def benchmark_times(self):
        raise NotImplementedError("Calling base class.")
