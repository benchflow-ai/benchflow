import os

from benchflow import load_benchmark
from benchflow.agents.rarebench_openai import RarebenchAgent

bench = load_benchmark(benchmark_name="benchflow/Rarebench", bf_token=os.getenv("BF_TOKEN"))

your_agents = RarebenchAgent()

run_ids = bench.run(
    task_ids=[0],
    agents=your_agents,
    api={"provider": "openai", "model": "gpt-4o-mini", "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    requirements_txt="rarebench_requirements.txt",
    params={
        "TASK_TYPE": "diagnosis",
        "DATASET_NAME": "LIRICAL",
        "DATASET_TYPE": "PHENOTYPE",
    },
)

results = bench.get_results(run_ids)