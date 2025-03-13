from .Bench import Bench


def load_benchmark(benchmark_name: str, bf_token: str) -> Bench:
    """
    Load the benchmark. You need to get a bf_token on https://benchflow.ai.
    For example:
    ```
    from benchflow import load_benchmark
    bench = load_benchmark("benchflow/webarena", "your_bf_token")
    ```
    """
    # TODO: send benchmark_name to bff to get the benchmark in __init__
    return Bench(benchmark_name, bf_token)