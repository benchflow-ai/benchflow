import os
import json

from benchflow import load_benchmark
from benchflow.agents.mlebench_mleagent import MLEAgent

bench = load_benchmark(benchmark_name="mlebench", bf_token=os.getenv("BF_TOKEN"))
agent = MLEAgent()

run_ids = bench.run(
    task_ids=["detecting-insults-in-social-commentary"],
    agents=agent,
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    requirements_txt="mleagent_requirements.txt",
    params={}
)
print(run_ids)
results = bench.get_results(run_ids)