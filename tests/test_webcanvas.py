import os
import json
from benchflow import load_benchmark
from benchflow.agents.webcanvas_openai import WebcanvasAgent

bench = load_benchmark(benchmark_name="webcanvas", bf_token=os.getenv("BFF_TOKEN"))

your_agents = WebcanvasAgent()

params = {
   "BROWSERBASE_API_KEY": os.environ.get("BROWSERBASE_API_KEY"),
   "GRAPHQL_USERNAME": os.environ.get("GRAPHQL_USERNAME"), 
   "GRAPHQL_PASSWORD": os.environ.get("GRAPHQL_PASSWORD"),
   "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")
}

run_ids = bench.run(
    task_ids=[0,1],
    agents=your_agents,
    requirements_txt = "webcanvas_requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params=params
)

results = bench.get_results(run_ids)