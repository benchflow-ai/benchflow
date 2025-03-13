from benchflow import load_benchmark
from benchflow.agents.medqa_openai import MedQAAgent
import os


def test_medqa():
    bench = load_benchmark(benchmark_name="benchflow/medqa-cs", bf_token=os.getenv("BF_TOKEN"))

    agent = MedQAAgent()

    run_ids = bench.run(
        task_ids=["diagnosis"], # choices: "diagnosis", "treatment", "prevention", "all"
        agents=agent,
        api={"provider": "openai", "model": "gpt-4o-mini", "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
        requirements_txt="medqa_requirements.txt",
        args={
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"), # required for llm as an evaluator
            "CASE_ID": "1", # use "all" to run all cases in diagnosis section
        },
    )

    results = bench.get_results(run_ids)

    assert len(results) > 0

if __name__ == "__main__":
    test_medqa()