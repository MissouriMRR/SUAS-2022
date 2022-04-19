"""
Module for benchmarking algorithms related to ODLC text.
"""

from vision.benchmarking.common.template import Benchmark


class BenchmarkText(Benchmark):
    """
    Benchmarks algorithms related to ODLC text.
    """

    def __init__(self):
        super().__init__()

    def benchmark_times(self):
        raise NotImplementedError("Benchmark not implemented")
