import json
import os
from typing import Any, Dict

from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult


class PubMedQABench(BaseBench):
    """
    PubMedQA is a biomedical question answering dataset where the task is to answer
    research questions with yes/no/maybe using PubMed abstracts.
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
                "TEMPERATURE": "0",           # Default temperature
                "LIMIT": "0",                 # 0 means use all examples
                "BATCH_SIZE": "10"            # Default batch size
            }
        })

    def get_image_name(self) -> str:
        """
        Return the Docker image name for the benchmark.
        """
        return "131268/benchflow-pubmedqa:latest"

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
        result_file = os.path.join(self.results_dir, "pubmedqa_result.json")

        if not os.path.exists(result_file):
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                log={"message": ""},
                metrics={"accuracy": 0.0},
                other={"error": "No results found"}
            )

        try:
            with open(result_file, 'r') as f:
                results = json.load(f)

            # Extract logs if available
            log_file = os.path.join(self.log_files_dir, "pubmedqa.log")
            log_content = ""
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()

            # Return the benchmark result
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=True,
                log={"content": log_content},
                metrics={
                    "accuracy": results['metrics']['accuracy'],
                    "correct": results['metrics']['correct'],
                    "total": results['metrics']['total']
                },
                other={
                    "success": f"Evaluated {results['metrics']['total']} examples",
                    "detailed_results": results['results'][:10]  # Include first 10 detailed results
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
        # For PubMedQA, we only have one task
        return {
            "task_ids": ["default"],
            "error_message": None
        }
