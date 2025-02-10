import os

from benchflow import load_benchmark
from benchflow.agents.webcanvas_openai import WebcanvasAgent

bench = load_benchmark(benchmark_name="webcanvas")

your_agents = WebcanvasAgent()

params = {
   "BROWSERBASE_API_KEY": os.environ.get("BROWSERBASE_API_KEY"),
   "GRAPHQL_USERNAME": os.environ.get("GRAPHQL_USERNAME"), 
   "GRAPHQL_PASSWORD": os.environ.get("GRAPHQL_PASSWORD"),
   "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")
}

run_ids = bench.run(
    task_ids=[1],
    agents=your_agents,
    requirements_dir = "webcanvas_requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params=params
)

results = bench.get_results(run_ids)