import os

from benchflow import load_benchmark
from benchflow.agents.rarebench_openai import RarebenchAgent

bench = load_benchmark(benchmark_name="benchflow/Rarebench", bf_token=os.getenv("BF_TOKEN"))

your_agents = RarebenchAgent()

run_ids = bench.run(
    task_ids=["MME"],
    agents=your_agents,
    api={"provider": "openai", "model": "gpt-4o-mini", "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    requirements_txt="rarebench_requirements.txt",
    args={
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    },
)

results = bench.get_results(run_ids)