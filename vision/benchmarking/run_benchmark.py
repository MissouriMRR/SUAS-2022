"""
Central file for running benchmarks of vision code. Benchmarks
are used to score timing and accuracy.
"""

from typing_extensions import runtime

from vision.benchmarking.common.benchmarks import BenchmarkType
from vision.benchmarking.common.benchmarks import RunType


def run_blank():
    """
    Run on a certain number of blank images.
    """
    raise NotImplementedError("Function not yet implemented.")


def run_set():
    """
    Run on a set of images in a folder.
    """
    raise NotImplementedError("Function not yet implemented.")


def run_bench(benchmark_name: str, run_type: str):
    """
    Run the specified benchmark using a set of images or generated blank images.

    Parameters
    ----------
    benchmark_name : str
        the name of the benchmark
    run_type : str
        the type of images to run on (either a set of images or blank images)
    """
    raise NotImplementedError("Function not yet implemented.")


# Driver for running vision benchmarks
if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Runs a benchmark. Must specify which benchmark to run."
    )

    parser.add_argument(
        "-b",
        "--benchmark_name",
        type=str,
        help="Name of the benchmark. Required argument.",
    )

    parser.add_argument(
        "-r",
        "--run_type",
        type=str,
        help="Type of images to run on. Required argument.",
    )

    args: argparse.Namespace = parser.parse_args()

    # no benchmark name specified, cannot continue
    if not args.benchmark_name:
        raise RuntimeError("No benchmark specified.")
    if args.benchmark_name not in [item.value for item in BenchmarkType]:
        raise RuntimeError("An invalid benchmark was given. Check spelling and try again.")
    arg_benchmark_name: str = args.benchmark_name

    # no run type specified, cannot continue
    if not args.run_type:
        raise RuntimeError("No run type specified.")
    if args.run_type not in [item.value for item in RunType]:
        raise RuntimeError("An invalid run type was given. Check spelling and try again.")
    arg_run_type: str = args.run_type

    run_bench(benchmark_name=arg_benchmark_name, run_type=arg_run_type)
