import os

from benchflow import load_benchmark
from benchflow.agents.mmlu_openai import MMLUAgent

bench = load_benchmark(benchmark_name="benchflow/MMLU-PRO", bf_token=os.getenv("BF_TOKEN"))

your_agents = MMLUAgent()

run_ids = bench.run(
    task_ids=["history"],
    agents=your_agents,
    requirements_txt="mmlupro_requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params={}
)

results = bench.get_results(run_ids)