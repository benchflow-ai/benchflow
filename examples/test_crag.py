import os

from benchflow import load_benchmark
from benchflow.agents.crag_openai import CRAGAgent

bench = load_benchmark(benchmark_name="CRAG", bf_token=os.getenv("BF_TOKEN"))

your_agents = CRAGAgent()

params = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "EVALUATION_MODEL_NAME": "gpt-4o-mini"
}

run_ids = bench.run(
    task_ids=["0"],
    agents=your_agents,
    requirements_txt="crag_requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params=params
)

results = bench.get_results(run_ids)