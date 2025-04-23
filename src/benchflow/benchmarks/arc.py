import json
import os
from typing import Any, Dict

from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult


class ARCBench(BaseBench):
    """
    
    ARC is a benchmark using natural science questions to evaluate a model's 
    knowledge and reasoning capabilities. The dataset ships with `Easy` and `Challenge` sets.
    """
    
    def __init__(self):
        super().__init__()
        self.tasks = ["easy", "challenge"]
    
    def get_args(self, task_id: str) -> BenchArgs:
        """
        Return arguments for the ARC benchmark.
        
        Args:
            task_id: The task ID, either "easy" or "challenge"
        """
        if task_id not in self.tasks:
            raise ValueError(f"Invalid task_id: {task_id}. Must be one of {self.tasks}")
            
        arguments = {
            "required": [
                "OPENAI_API_KEY"
            ],
            "optional": [
                {"MODEL_NAME": "gpt-4o-mini"},
                {"TASK_NAME": task_id},
                {"LIMIT": 100},
                {"TEMPERATURE": 0}
            ]
        }
        return BenchArgs(arguments)
    
    def get_image_name(self) -> str:
        """Return the Docker image name for running the ARC benchmark."""
        return "131268/benchflow-arc:latest"
    
    def get_results_dir_in_container(self) -> str:
        """Return the directory inside the container where benchmark results will be stored."""
        return "/app/results"
    
    def get_log_files_dir_in_container(self) -> str:
        """Return the directory inside the container where log files will be stored."""
        return "/app/logs"
    
    def get_result(self, task_id: str) -> BenchmarkResult:
        """
        Read and parse the benchmark result from the results directory.
        
        Args:
            task_id: The task ID, either "easy" or "challenge"
        """
        result_file = os.path.join(self.results_dir, f"arc_{task_id}_result.json")
        
        if not os.path.exists(result_file):
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                metrics={"accuracy": 0},
                log={"error": "No results found"},
                other={}
            )
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # Extract the accuracy from the results
            accuracy = results.get("metrics", {}).get("accuracy", 0)
            
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=True,
                metrics={"accuracy": accuracy},
                log={"result": json.dumps(results, indent=2)},
                other={"details": results}
            )
        
        except Exception as e:
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                metrics={"accuracy": 0},
                log={"error": str(e)},
                other={"error": str(e)}
            )
    
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return all available ARC tasks.
        
        Args:
            split: The dataset split (not used for ARC)
        """
        return {"task_ids": self.tasks, "error_message": None}
    
    def cleanup(self):
        """Clean up any resources used by the benchmark."""
        pass