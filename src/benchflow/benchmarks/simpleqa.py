import json
import os
from typing import Any, Dict

from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult


class SimpleQABench(BaseBench):
    """
    SimpleQA is a benchmark that evaluates the ability of language models to answer
    short, fact-seeking questions and to abstain from answering when uncertain.
    """

    def get_args(self, task_id: str) -> BenchArgs:
        """
        Define the arguments for the benchmark.

        Args:
            task_id: The ID of the task to run.

        Returns:
            BenchArgs object with the arguments for the benchmark.
        """
        return BenchArgs({
            "required_args": ["OPENAI_API_KEY"],
            "optional_args": {
                "MODEL_NAME": "gpt-4o-mini",  # Default model
                "TEMPERATURE": "0.5",         # Default temperature
                "MAX_TOKENS": "2048",         # Default max tokens
                "LIMIT": "0",                 # 0 means use all examples
                "BATCH_SIZE": "10",           # Default batch size
                "GRADER_MODEL": "gpt-4o",     # Default grader model
                "GRADER_TEMPERATURE": "0.5"   # Default grader temperature
            }
        })

    def get_image_name(self) -> str:
        """
        Return the Docker image name for the benchmark.
        """
        return "131268/benchflow-simpleqa:latest"

    def get_results_dir_in_container(self) -> str:
        """
        Return the directory inside the container where results will be stored.
        """
        return "/app/results"

    def get_log_files_dir_in_container(self) -> str:
        """
        Return the directory inside the container where log files will be stored.
        """
        return "/app/logs"

    def get_result(self, task_id: str) -> BenchmarkResult:
        """
        Parse the benchmark results from the results directory.

        Args:
            task_id: The ID of the task.

        Returns:
            BenchmarkResult object with the benchmark results.
        """
        # Find the result file - it will be named based on the model
        result_files = [f for f in os.listdir(self.results_dir) if f.endswith("_result.json")]

        if not result_files:
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                log={"message": ""},
                metrics={"accuracy": 0.0},
                other={"error": "No results found"}
            )

        # Use the first result file found
        result_file = os.path.join(self.results_dir, result_files[0])

        try:
            with open(result_file, 'r') as f:
                results = json.load(f)

            # Extract logs if available
            log_file = os.path.join(self.log_files_dir, "simpleqa.log")
            log_content = ""
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()

            # Return the benchmark result
            metrics = results["metrics"]
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=True,
                log={"content": log_content},
                metrics={
                    "accuracy": metrics["accuracy"],
                    "correct": metrics["correct"],
                    "incorrect": metrics["incorrect"],
                    "not_attempted": metrics["not_attempted"],
                    "total": metrics["total"],
                    "correct_given_attempted": metrics["correct_given_attempted"],
                    "f_score": metrics["f_score"]
                },
                other={
                    "success": f"Evaluated {metrics['total']} examples",
                    "detailed_results": results["results"][:10]  # Include first 10 detailed results
                }
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                log={"message": ""},
                metrics={"accuracy": 0.0},
                other={"error": f"Error parsing results: {str(e)}"}
            )

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return all available tasks for the benchmark.

        Args:
            split: The dataset split to use.

        Returns:
            Dictionary mapping task IDs to task metadata.
        """
        # For SimpleQA, we only have one task
        return {
            "task_ids": ["default"],
            "error_message": None
        }
