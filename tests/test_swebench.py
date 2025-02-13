import os

from benchflow import load_benchmark
from benchflow.agents.swebench_sweagent import SWEAgent


bench = load_benchmark(benchmark_name="swebench", bf_token=os.getenv("BFF_TOKEN"))

your_agents = SWEAgent()

run_ids = bench.run(
    task_ids=["astropy__astropy-12907"],
    agents=your_agents,
    install_sh="install_sweagent.sh",
    requirements_txt="sweagent_requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params={}
)

results = bench.get_results(run_ids)