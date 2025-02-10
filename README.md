# BenchFlow: AI Benchmark Runtime

BenchFlow is an AI benchmark runtime framework that allows you to integrate and evaluate AI tasks using Docker-based benchmarks. The latest version leverages a new **BaseBench** design to manage logs, results, and environment variable configurations consistently.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agent Development Guide](#agent-development-guide)
  - [Step 1: Define Your Agent](#step-1-define-your-agent)
  - [Step 2: Test Your Agent](#step-2-test-your-agent)
- [Benchmark Integration Guide](#benchmark-integration-guide)
  - [Step 1: Implement BenchClient](#step-1-implement-benchclient)
  - [Step 2: Package and Upload Your Benchmark Docker Image](#step-2-package-and-upload-your-benchmark-docker-image)
  - [Step 3: Integrate Your Benchmark](#step-3-integrate-your-benchmark)
- [API Reference](#api-reference)
- [License](#license)

---

## Installation

- **Python 3.11+**
- Docker

Install the BenchFlow package using pip:

```bash
pip install benchflow
```

---

## Quick Start

Supported Benchmarks:
- WebArena
- WebCanvas
- SWE-Bench (coming in version 0.1.6)

You can try our demo by running the following command:

```bash
git clone https://github.com/benchflow-ai/benchflow.git
cd benchflow
pip install -e .
```

Test Default WebArena Agent on WebArena:
```bash
cd tests
python test_webarena.py
```

Test Default WebCanvas Agent on WebCanvas:
```bash
cd tests
python test_webcanvas.py
```

Test Default SWEAgent on SWEBench:
```bash
cd tests
python test_swebench.py
```

---

## Agent Development Guide

### Step 1: Define Your Agent

Create your Agent by extending `BaseAgent`. The Agent processes the environment data provided via `self.env_info` and generates a solution for the task.

```python
from benchflow import BaseAgent

class YourAgent(BaseAgent):
    def __init__(self):
        super().__init__()
    
    def call_api(self) -> str:
        """
        IMPLEMENTATION CONTRACT:
        Process environment data and generate task solution.

        Access:
        - self.env_info: dict containing benchmark-specific data

        Returns:
            str: Unified diff patch or any prediction as a formatted string.
        """
        # Access task parameters
        instance_id = self.env_info['instance_id']
        # Process the data provided in `env_info` and return your prediction
        return (
            "diff --git a/src/rules/L031.py b/src/rules/L031.py\n"
            "--- a/src/rules/L031.py\n"
            "+++ b/src/rules/L031.py\n"
            "@@ -211,7 +211,7 @@ def _lint_aliases_in_join(\n"
            "    violation_buff.append(\n"
            "        LintResult(\n"
            "            anchor=alias_info.alias_identifier_ref,\n"
            " -          description=\"Original message\",\n"
            " +          description=\"Updated message\",\n"
            "            fixes=fixes,\n"
            "        )\n"
            "    )"
        )
```

### Step 2: Test Your Agent

Test your Agent with the benchmark by loading the benchmark module and running the evaluation.
Agent will be host in container, you can add any python dependencies in a `requirements.txt`-like file or any other install steps in a shell script `install.sh` and it will be installed in the container. You can specify the directory of the requirements in the `requirements_dir` parameter and the directory of the install scripts in the `install_dir` parameter. If your agent need some environment variables, you can specify them in the `api` parameter. For example, {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"). "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")}
Here is an example of testing your agent with the WebCanvas benchmark:

```python
from benchflow import load_benchmark
from webcanvas_openai import WebcanvasAgent

bench = load_benchmark(benchmark_name="webcanvas")

your_agents = WebcanvasAgent()

# There are four requirements for WebCanvas:
# 1. BROWSERBASE_API_KEY: you can get it from https://browserbase.com/
# 2. GRAPHQL_USERNAME: you can register on the platform at https://www.imean.ai/web-canvas.
# 3. GRAPHQL_PASSWORD: you can register on the platform at https://www.imean.ai/web-canvas.
# 4. OPENAI_API_KEY: your api key of the openai
params = {
   "BROWSERBASE_API_KEY": os.environ.get("BROWSERBASE_API_KEY"),
   "GRAPHQL_USERNAME": os.environ.get("GRAPHQL_USERNAME"), 
   "GRAPHQL_PASSWORD": os.environ.get("GRAPHQL_PASSWORD"),
   "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")
}

run_ids = bench.run(
    task_ids=[1], # you can change the task_ids to None to run all tasks
    agents=your_agents,
    requirements_dir = "webcanvas_requirements.txt",
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params=params
)

results = bench.get_results(run_ids)
```

---

## Benchmark Integration Guide

The Benchmark Integration Guide now comprises three steps:

### Step 1: Implement BenchClient

Create a class extending `BenchClient` to transform the raw state into the agent's input and parse the agent's output.

```python
from benchflow import BenchClient
from typing import Dict, Any

class YourClient(BenchClient):
    def prepare_environment(self, state_update: Dict) -> Dict:
        """Transform raw state into agent inputs."""
        return {
            "env_info": {
                "observation": state_update["trajectory"][-1],
                "intent": state_update.get("intent", "")
            }
        }
    
    def parse_action(self, raw_action: str) -> str:
        """Process the agent response."""
        parsed_action = raw_action  # Optionally add post-processing here
        return parsed_action
```

### Step 2: Package and Upload Your Benchmark Docker Image

Before integrating your benchmark, ensure that you have:

- Packaged your benchmark logic into a Docker image.
- Configured the image to read required environment variables (such as `AGENT_URL`, `TEST_START_IDX`, etc.).
- Uploaded the Docker image to a public registry (e.g., DockerHub).

For example, tag your image as `yourusername/benchmark-name:tag`. No code snippet is required for this step.

### Step 3: Integrate Your Benchmark

Integrate your benchmark by subclassing **BaseBench**. In the new implementation, you must implement the following abstract methods:

- **`get_config(params: Dict[str, Any], task_id: str) -> BaseBenchConfig`**  
  Returns a configuration instance (derived from `BaseBenchConfig`) to validate and prepare environment variables.

- **`get_image_name() -> str`**  
  Returns the Docker image name for running the benchmark.

- **`get_results_dir_in_container() -> str`**  
  Returns the directory inside the container where results will be stored.

- **`get_log_files_dir_in_container() -> str`**  
  Returns the directory inside the container where log files will be stored.

- **`get_result(task_id: str) -> Dict[str, Any]`**  
  Reads and parses the benchmark results (for example, from log files) and returns a dictionary containing:
  - `task_id`
  - `is_resolved` (a boolean indicating success)
  - `score` (a numerical score)
  - `message` (a dictionary with details or error messages)
  - `log` (log details as a string)

- **`get_all_tasks(split: str) -> Dict[str, Any]`**  
  Returns all available task IDs and an optional error message.

- **`cleanup()`**  
  Cleans up any temporary resources created during benchmark execution.

Below is an example integration using **WebArenaBench**:

```python
# webarena_bench.py
import os
import subprocess
from typing import Any, Dict

from benchflow import BaseBench, BaseBenchConfig

# ------------------------------------------------------------------------------
# WebArenaConfig: Define the configuration for WebArenaBench.
# ------------------------------------------------------------------------------
class WebArenaConfig(BaseBenchConfig):
    # For this benchmark, we require the TEST_END_IDX variable.
    required_env = ["TEST_END_IDX"]
    optional_env = []
    defaults = {
        "RESULTS_DIR": "/app/results"
    }

# ------------------------------------------------------------------------------
# WebArenaBench Implementation
# ------------------------------------------------------------------------------
class WebArenaBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
        """
        Return a WebArenaConfig instance that validates the input parameters.
        Here, we set TEST_END_IDX so that each run processes only one task.
        """
        params["TEST_END_IDX"] = str(int(task_id) + 1)
        return WebArenaConfig(params)
    
    def get_image_name(self) -> str:
        """
        Return the Docker image name for running the WebArena benchmark.
        """
        return "kirk2000/benchflow:webarena-v1"
    
    def get_results_dir_in_container(self) -> str:
        """
        Return the directory inside the container where benchmark results will be stored.
        """
        return "/app/results"
    
    def get_log_files_dir_in_container(self) -> str:
        """
        Return the directory inside the container where log files will be stored.
        """
        return "/app/log_files"
    
    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        Read and parse the benchmark result from log files.
        This method expects a file named 'log_files.txt' in the results directory.
        It reads the content of each log file listed, aggregates the logs, and extracts
        the average score and pass status.
        """
        log_files_txt = os.path.join(self.results_dir, "log_files.txt")
        if not os.path.exists(log_files_txt):
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": "No results found"},
                "log": ""
            }
        
        log_content = ""
        try:
            with open(log_files_txt, 'r') as f:
                for line in f:
                    log_file_name = os.path.basename(line.strip())
                    # Assume log files are located in the log_files directory under the task_id folder.
                    full_log_path = os.path.join(self.log_files_dir, str(task_id), log_file_name)
                    with open(full_log_path, 'r') as log_file:
                        log_content += log_file.read() + "\n"
        except Exception as e:
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": f"Failed to read log files: {e}"},
                "log": log_content
            }
        
        # Parse the log content to extract score and resolution status.
        is_resolved = False
        score = 0.0
        for line in log_content.splitlines():
            if "Average score:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                except ValueError:
                    score = 0.0
            if "[Result]" in line and "(PASS)" in line:
                is_resolved = True
                    
        return {
            "is_resolved": is_resolved,
            "score": score,
            "message": {"details": "Task runs successfully."},
            "log": log_content
        }
    
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return a dictionary containing all task IDs and an optional error message.
        For the 'train' split, return 200 tasks; for other splits, return 812 tasks.
        """
        if split == "train":
            task_ids = [str(i) for i in range(200)]
        else:
            task_ids = [str(i) for i in range(812)]
        return {"task_ids": task_ids, "error_message": None}
    
    def cleanup(self):
        """
        Clean up benchmark resources by removing local results and log directories.
        """
        if os.path.exists(self.results_dir):
            self.logger.info(f"Removing {self.results_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.results_dir], check=True)
        if os.path.exists(self.log_files_dir):
            self.logger.info(f"Removing {self.log_files_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.log_files_dir], check=True)
```

---

## API Reference

### BaseBench Class

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `run_bench(task_id: str, agent_url: str, params: Dict[str, Any])` | `task_id`: Task identifier<br>`agent_url`: Agent service endpoint<br>`params`: Runtime parameters dictionary | `Dict[str, Any]` | Runs the benchmark inside a Docker container, captures logs, and returns the execution result. |
| `format_result(...)` | See implementation | `Dict[str, Any]` | Formats the benchmark result to include `task_id`, `is_resolved`, `score`, `message`, and `log`. |
| `get_volumes()` | None | `Dict[str, Dict[str, str]]` | Defines Docker volume mappings for results and log directories. |
| `validate_result(result: Dict[str, Any])` | `result`: Result dictionary | `bool` | Validates that the benchmark result contains all required fields. |
| _Abstract Methods_ | See documentation | — | Must be implemented in your subclass: `get_config()`, `get_image_name()`, `get_results_dir_in_container()`, `get_log_files_dir_in_container()`, `get_result()`, `get_all_tasks()`, and `cleanup()`. |

### BaseBenchConfig Class

Used to define and validate the environment variables required for benchmark execution. Extend this class to customize the configuration by overriding `required_env`, `optional_env`, and `defaults`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

By following these steps, you can quickly implement and integrate your own AI benchmarks using the latest version of **BaseBench**. If you have any questions or suggestions, please feel free to submit an issue or pull request.