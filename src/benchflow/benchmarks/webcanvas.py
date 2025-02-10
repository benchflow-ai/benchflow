# webcanvasbench.py
import json
import os
from typing import Any, Dict

from benchflow import BaseBench, BaseBenchConfig


#------------------------------------------------------------------------------
# Custom configuration: Define the environment variables and validation rules for WebCanvasBench
#------------------------------------------------------------------------------
class WebCanvasConfig(BaseBenchConfig):
    # These envs are required by the benchmark, and should be provided by the user.
    required_env = ["BROWSERBASE_API_KEY", "GRAPHQL_USERNAME", "GRAPHQL_PASSWORD", "OPENAI_API_KEY", "TEST_END_IDX"]
    # These envs are optional, and will use the default value if not provided.
    optional_env = []
    # These envs are defaults, and will be used if not provided.
    defaults = {
        "RESULTS_DIR": "/app/batch_tasks_results/example"
    }

#------------------------------------------------------------------------------
# WebCanvasBench implementation
#------------------------------------------------------------------------------
class WebCanvasBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
        """
        Return a WebCanvasConfig instance, validate the input parameters.
        """
        # Benchmark need to deal with the END_IDX so that it can only run one task at a time
        # task_id is the start index of the task
        params["TEST_END_IDX"] = str(task_id)
        return WebCanvasConfig(params)

    def get_image_name(self) -> str:
        """
        Return the Docker image name for running the WebCanvas benchmark.
        """
        return "kirk2000/benchflow:webcanvas-v1"

    def get_results_dir_in_container(self) -> str:
        """
        In the container, the result files will be written to /app/batch_tasks_results
        (the environment variable RESULTS_DIR can further specify a subdirectory, such as "example").
        """
        return "/app/batch_tasks_results"

    def get_log_files_dir_in_container(self) -> str:
        """
        In the container, the log files will be written to /app/LOGS
        """
        return "/app/LOGS"

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        Read the result file (assuming the path is: {results_dir}/example/result/result.json),
        and parse the result dictionary, which requires the is_resolved, score, and message fields.
        """
        # Construct the full path to the result file (related to the RESULTS_DIR configuration inside the container)
        result_file = os.path.join(self.results_dir, "example", "result", "result.json")
        log_file = os.path.join(self.results_dir, "example", "result", "out.json")
        print(result_file)
        if not os.path.exists(result_file):
            return {"is_resolved": False, "score": 0, "message": {"error": "No results found"}}
        try:
            with open(result_file, 'r') as f:
                data = f.read().strip()
                try:
                    results = json.loads(data)
                except Exception:
                    # If the direct parsing fails, try simple text replacement and then parsing
                    data_fixed = data.replace('{', '{"').replace(': ', '": ').replace(', ', ', "')
                    results = json.loads(data_fixed)
            with open(log_file, 'r') as f:
                log = f.read().strip()
                print(log)
        except Exception as e:
            return {"is_resolved": False, "score": 0, "message": {"error": e}}
        
        # Calculate whether the benchmark passed and the score based on the parsed results
        is_resolved = results.get("task_success_rate", 0) > 0.99
        score = results.get("average_step_score_rate", 0)
        # Concatenate the result details in key-value pair format
        message = {"details": ', '.join(f"{k}: {v}" for k, v in results.items())}
        return {"is_resolved": is_resolved, "score": score, "message": message, "log": str(log)}

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return all task_ids. If split is "train", return 20 task_ids, otherwise return 103 task_ids.
        """
        if split == "train":
            task_ids = [str(i) for i in range(20)]
        else:
            task_ids = [str(i) for i in range(103)]
        return {"task_ids": task_ids, "error_message": None}

    def cleanup(self):
        """
        Add the logic to clean up the temporary result and log directories.
        For example, delete the files in self.results_dir and self.log_files_dir.
        """
        pass
