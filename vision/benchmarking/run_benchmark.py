"""
Central file for running benchmarks of vision code. Benchmarks
are used to score timing and accuracy.
"""

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

    args: argparse.Namespace = parser.parse_args()

    # no benchmark name specified, cannot continue
    if not args.benchmark_name:
        raise RuntimeError("No benchmark specified.")
    benchmark_name: str = args.benchmark_name
