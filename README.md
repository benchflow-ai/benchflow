<div align="center">
  <h1>BenchFlow</h1>
  <p>
    <img src="https://img.shields.io/pypi/l/benchflow?style=plastic" alt="PyPI - License">
    <img src="https://img.shields.io/pypi/dm/benchflow?style=plastic" alt="PyPI - Downloads">
    <img src="https://img.shields.io/pypi/v/benchflow?style=plastic" alt="PyPI - Version">
  </p>
  <p>
    BenchFlow is an <b>Open-source Benchmark Hub</b> and <b> Eval Infra</b> for AI production and benchmark developers.
  </p>
  <img src="https://github.com/user-attachments/assets/6f0a0bb8-1bae-4628-9757-6051e452c01b" alt="BenchFlow Diagram">
</div>

## Overview

![BenchFlow Overview](docs/images/benchflow.png)

Within the dashed box, you will find the interfaces ([**BaseAgent**](./src/benchflow/BaseAgent.py), [**BenchClient**](./src/benchflow/BenchClient.py)) provided by BenchFlow. For benchmark users, you are required to extend and implement the [**BaseAgent**](./src/benchflow/BaseAgent.py) interface to interact with the benchmark. The `call_api` method supplies a `step_input` which provides the input for each step of a task (a task may have one or more steps).

## Quick Start For Benchmark Users

1. **Install BenchFlow**

   ```bash
   git clone https://github.com/benchflow-ai/benchflow.git
   cd benchflow
   pip install -e .
   ```

2. **Browse Benchmarks**

   Find benchmarks tailored to your needs on our [**Benchmark Hub**](https://staging.benchflow.ai/dashboard/benchmarks).

3. **Implement Your Agent**

   Extend the [**BaseAgent**](./src/benchflow/BaseAgent.py) interface:

   ```python
   def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
       pass
   ```

   *Optional:* You can include a `requirements.txt` file to install additional dependencies, such as `openai` and `requests`.

4. **Test Your Agent**

   Here is a quick example to run your agent:

   ```python
   import os
   from benchflow import load_benchmark
   from benchflow.agents.webarena_openai import WebarenaAgent

   # The benchmark name follows the schema: org_name/benchmark_name.
   # You can obtain the benchmark name from the Benchmark Hub.
   bench = load_benchmark(benchmark_name="benchflow/webarena", bf_token=os.getenv("BF_TOKEN"))

   your_agents = WebarenaAgent()

   run_ids = bench.run(
       task_ids=[0],
       agents=your_agents,
       api={"provider": "openai", "model": "gpt-4o-mini", "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
       requirements_txt="webarena_requirements.txt",
       args={}
   )

   results = bench.get_results(run_ids)
   ```

## Quick Start for Benchmark Developers

1. **Install BenchFlow**

   Install BenchFlow via pip:

   ```bash
   pip install benchflow
   ```

2. **Embed [**`BenchClient`**](./src/benchflow/BenchClient.py) into Your Benchmark Evaluation Scripts**

   Refer to this [**example**](https://github.com/BenchFlow-Hub/BF-MMLU-Pro/blob/e252ba159d9df26ae92d8c3f3570639874440757/evaluate_from_api.py#L199-L220) for how MMLU-Pro integrates **`BenchClient`**.

3. **Containerize Your Benchmark and Upload the Image to Dockerhub**

   Ensure your benchmark can run in a single container without any additional steps. Below is an example Dockerfile for MMLU-Pro:

   ```Dockerfile
   FROM python:3.11-slim

   COPY . /app
   WORKDIR /app
   COPY scripts/entrypoint.sh /app/entrypoint.sh

   RUN chmod +x /app/entrypoint.sh
   RUN pip install -r requirements.txt

   ENTRYPOINT ["/app/entrypoint.sh"]
   ```

4. **Extend [**`BaseBench`**](./src/benchflow/BaseBench.py) to Run Your Benchmarks**

   See this [**example**](https://github.com/BenchFlow-Hub/BF-MMLU-Pro/blob/main/benchflow_interface.py) for how MMLU-Pro extends **`BaseBench`**

5. **Upload Your Benchmark into BenchFlow**

   Go to the Benchmark Hub and click on `+new benchmarks` to upload your benchmark Git repository. Make sure you place the `benchflow_interface.py` file at the root of your project.

---

## License

This project is licensed under the [MIT License](LICENSE).