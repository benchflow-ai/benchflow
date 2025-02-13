import os

from benchflow import load_benchmark
from benchflow.agents.webarena_openai import WebarenaAgent

bench = load_benchmark(benchmark_name="webarena", bf_token=os.getenv("BFF_TOKEN"))

your_agents = WebarenaAgent()

run_ids = bench.run(
    task_ids=[0,1,2,3],
    agents=your_agents,
    api={"provider": "openai", "model": "gpt-4o-mini", "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    requirements_txt="webarena_requirements.txt",
    params={}
)

results = bench.get_results(run_ids)