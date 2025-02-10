import os

from benchflow.agents.swe_agent import SWEAgent

from benchflow import load_benchmark

bench = load_benchmark(benchmark_name="swebench")

your_agents = SWEAgent()

run_ids = bench.run(
    task_ids=["astropy__astropy-12907"],
    agents=your_agents,
    install_sh_dir="install_sweagent.sh",
    requirements_dir="sweagent_requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params={}
)

results = bench.get_results(run_ids)