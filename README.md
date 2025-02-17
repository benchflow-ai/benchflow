<div align="center">

# BenchFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

Agent benchmark runtime to manage logs, results, and environment variable configurations consistently.

![image](https://github.com/user-attachments/assets/6f0a0bb8-1bae-4628-9757-6051e452c01b)

</div>

## Overview

Benchflow is an AI benchmark runtime platform designed to provide a unified environment for benchmarking and performance evaluation of your intelligent agents. It not only offers a wide range of easy-to-run benchmarks but also tracks the entire lifecycle of an agent—including recording prompts, execution times, costs, and various metrics—helping you quickly evaluate and optimize your agent's performance.

## Feature Overview

- **Easy-to-Run Benchmarks**: Comes with a large collection of built-in benchmarks that let you quickly verify your agent’s capabilities.
- **Complete Lifecycle Tracking**: Automatically records prompts, responses, execution times, costs, and other performance metrics during agent invocations.
- **Efficient Agent Evaluation**: Benchflow’s evaluation mechanism can deliver over a 3× speed improvement in assessments, helping you stand out from the competition.
- **Benchmark Packaging**: Supports packaging benchmarks into standardized suites, making it easier for Benchmark Developers to integrate and release tests.

## Installation

```bash
pip install benchflow
```

## Introduction

Benchflow caters to two primary roles:

- **Agent Developer**: Test and evaluate your AI agent.
- **Benchmark Developer**: Integrate and publish your custom-designed benchmarks.

### For Agent Developers

You can start testing your agent in just two steps:

#### 1. Integrate Your Agent

In your agent code, inherit from the `BaseAgent` class provided by Benchflow and implement the necessary methods. For example:

```python
from benchflow import BaseAgent

class YourAgent(BaseAgent):
    def call_api(self, *args, **kwargs):
        # Define how to call your agent's API
        pass
```

#### 2. Run the Benchmark

Load and run a benchmark using the provided Benchflow interface. For example:

```python
import os
from benchflow import load_benchmark

# Load the specified benchmark
bench = load_benchmark("benchmark_name")
your_agent = YourAgent()

# Run the benchmark
run_ids = bench.run(
    agents=your_agent,
    requirements_dir="requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params={}
)

# Retrieve the test results
results = bench.get_results(run_ids)
```

### For Benchmark Developers

Integrating your benchmark into Benchflow involves three steps:

#### 1. Create a Benchmark Client

Develop a Benchmark Client class that sets up the testing environment for the agent and parses its responses. For example:

```python
from benchflow import BenchClient
from typing import Dict, Any

class YourBenchClient(BenchClient):
    def __init__(self, agent_url: str):
        super().__init__(agent_url)

    def prepare_environment(self, state_update: Dict) -> Dict:
        """Prepare environment information for the agent during the benchmark."""
        return {
            "env_info": {
                "info1": state_update['info1'],
                "info2": state_update['info2']
            }
        }

    def parse_action(self, raw_action: str) -> str:
        """Process the response returned by the agent."""
        # Process the raw response and return the parsed action
        return raw_action  # Modify as needed for your use case
```

#### 2. Encapsulate Your Benchmark Logic

Package your benchmark logic into a Docker image. The image should be configured to read necessary environment variables (such as `AGENT_URL`, `TEST_START_IDX`, etc.) and encapsulate the benchmark logic within the container.

#### 3. Upload Your Benchmark to Benchflow

Extend the `BaseBench` class provided by Benchflow to configure your benchmark and upload it to the platform. For example:

```python
from benchflow import BaseBench, BaseBenchConfig
from typing import Dict, Any

class YourBenchConfig(BaseBenchConfig):
    # Define required and optional environment variables
    required_env = []
    optional_env = ["INSTANCE_IDS", "MAX_WORKERS", "RUN_ID"]

    def __init__(self, params: Dict[str, Any], task_id: str):
        params.setdefault("INSTANCE_IDS", task_id)
        params.setdefault("MAX_WORKERS", 1)
        params.setdefault("RUN_ID", task_id)
        super().__init__(params)

class YourBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
        """Return a benchmark configuration instance to validate input parameters."""
        return YourBenchConfig(params, task_id)

    def get_image_name(self) -> str:
        """Return the Docker image name that runs the benchmark."""
        return "your_docker_image_url"

    def get_results_dir_in_container(self) -> str:
        """Return the directory path inside the container where benchmark results are stored."""
        return "/app/results"

    def get_log_files_dir_in_container(self) -> str:
        """Return the directory path inside the container where log files are stored."""
        return "/app/log_files"

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        Read and parse the benchmark results.
        For example, read a log file from the results directory and extract the average score and pass status.
        """
        # Implement your reading and parsing logic here
        is_resolved = True  # Example value
        score = 100        # Example value
        log_content = "Benchmark run completed successfully."
        return {
            "is_resolved": is_resolved,
            "score": score,
            "message": {"details": "Task runs successfully."},
            "log": log_content
        }

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return a dictionary containing all task IDs, along with an optional error message.
        """
        task_ids = ["0", "1", "2"]  # Example task IDs
        return {"task_ids": task_ids, "error_message": None}
```

## Examples

Explore two example implementations to understand how to use Benchflow effectively, please visit [Examples](https://benchflow.gitbook.io/benchflow/getting-started/examples).
## Summary

Benchflow provides a complete platform for testing and evaluating AI agents:  
- **For Agent Developers**: Simply extend `BaseAgent` and call the benchmark interface to quickly verify your agent’s performance.  
- **For Benchmark Developers**: Follow the three-step process—creating a client, packaging your Docker image, and uploading your benchmark—to integrate your custom tests into the Benchflow platform.

For more detailed documentation and sample code, please visit the [Benchflow GitBook](https://benchflow.gitbook.io/benchflow).

### BaseBenchConfig Class

Used to define and validate the environment variables required for benchmark execution. Extend this class to customize the configuration by overriding `required_env`, `optional_env`, and `defaults`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

By following these steps, you can quickly implement and integrate your own AI benchmarks using the latest version of **BaseBench**. If you have any questions or suggestions, please feel free to submit an issue or pull request.
