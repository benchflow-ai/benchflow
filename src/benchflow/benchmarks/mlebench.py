import os
import json
from typing import Any, Dict

from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult


class MleBenchBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_args(self, task_id: str) -> BenchArgs:
        arguments = {
            "required": [],
            "optional": [
                {"COMPETITION_SET": "competitions.txt"},
                {"N_WORKERS": 2},
                {"N_SEEDS": 5},
            ]
        }
        return BenchArgs(arguments)

    def get_image_name(self) -> str:
        """
        Returns the Docker image name used to run MLEBench.
        """
        return "shir2002/mlebench-benchflow"

    def get_results_dir_in_container(self) -> str:
        """
        Specifies where MLEBench stores its results inside the container.
        """
        return "/app/results"

    def get_log_files_dir_in_container(self) -> str:
        """
        Specifies where logs for MLEBench runs will be stored inside the container.
        """
        return "/app/logs"

    def get_result(self, task_id: str) -> BenchmarkResult:
        """
        Reads the latest benchmark results since MLEBench does not use task_id.
        We assume that the latest run's results should be processed.
        """
        results_file = os.path.join(self.results_dir, "metadata.json") 
        log_file = os.path.join(self.log_files_dir, "run.log")

        try:
            with open(results_file, 'r') as f:
                result_data = json.load(f)

            accuracy = result_data.get("overall_accuracy", 0)
            runtime = result_data.get("total_runtime", None)

            with open(log_file, 'r') as f:
                logs = f.read()

            return BenchmarkResult(
                task_id="mlebench_latest",
                is_resolved=accuracy > 0.90,
                metrics={"accuracy": accuracy, "runtime": runtime},
                log={"logs": logs},
                other={"details": result_data},
            )
        except Exception as e:
            return BenchmarkResult(
                task_id="mlebench_latest",
                is_resolved=False,
                metrics={"accuracy": 0},
                log={"error": str(e)},
                other={"error": str(e)},
            )

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Since MLEBench runs on competitions, we retrieve competition IDs instead of task IDs.
        """
        competition_set_file = "/app/config/competitions.txt"

        try:
            with open(competition_set_file, "r") as f:
                competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]

            return {"task_ids": competition_ids, "error_message": None}
        except Exception as e:
            return {"task_ids": [], "error_message": str(e)}

    def cleanup(self):
        """
        Removes temporary result and log files to free up space.
        """
        if os.path.exists(self.results_dir):
            self.logger.info(f"Removing {self.results_dir}")
            os.system(f"rm -rf {self.results_dir}")

        if os.path.exists(self.log_files_dir):
            self.logger.info(f"Removing {self.log_files_dir}")
            os.system(f"rm -rf {self.log_files_dir}")
