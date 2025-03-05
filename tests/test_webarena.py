import os

from benchflow import load_benchmark
from benchflow.agents.webarena_openai import WebarenaAgent

bench = load_benchmark(benchmark_name="benchflow/webarena", bf_token=os.getenv("BF_TOKEN"))

your_agents = WebarenaAgent()

run_ids = bench.run(
    task_ids=[0],
    agents=your_agents,
    api={"provider": "openai", "model": "gpt-4o-mini", "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    requirements_txt="webarena_requirements.txt",
    args={}
)

results = bench.get_results(run_ids)