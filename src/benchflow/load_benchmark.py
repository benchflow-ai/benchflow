from .Bench import Bench

BENCHMARK_REGISTRY = {
    "webarena": "http://ec2-3-232-182-160.compute-1.amazonaws.com:12345",
    "webcanvas": "http://ec2-3-232-182-160.compute-1.amazonaws.com:12345",
    "swebench": "http://ec2-3-232-182-160.compute-1.amazonaws.com:12345",
}

def load_benchmark(benchmark_name: str, bf_token: str) -> Bench:
    """
    """
    if benchmark_name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return Bench(benchmark_name, bf_token)