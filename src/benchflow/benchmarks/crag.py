import os
import json
from typing import Any, Dict
from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult


class CRAGBench(BaseBench):
    def __init__(self):
        super().__init__()
    
    def get_args(self, task_id: str) -> BenchArgs:
        arguments = {
            "required": [
                "OPENAI_API_KEY",
                "EVALUATION_MODEL_NAME"
            ],
            "optional": [
                {"BATCH_SIZE": 100},
            ]
        }
        return BenchArgs(arguments)
    
    def get_image_name(self) -> str:
        return "danielfang001/benchflow:crag-v1"
    
    def get_results_dir_in_container(self) -> str:
        return "/workspace/results"

    def get_log_files_dir_in_container(self) -> str:
        return "/workspace/logs"
       
    def get_result(self, task_id: str) -> Dict[str, Any]:
        result_file = os.path.join(self.results_dir, f"{task_id}_results.json")

        if not os.path.exists(result_file):
            return BenchmarkResult(
                task_id=task_id, 
                is_resolved=False, 
                metrics={"score": 0}, 
                log={"error": "No results found"}, 
                other={}
            )
        
        try:
            with open(result_file, "r") as f:
                results = json.load(f)

            is_resolved = False
            if results.get("score"):
                is_resolved = True

            return BenchmarkResult(
                task_id=task_id, 
                is_resolved=is_resolved, 
                metrics={"score": results.get("score", 0)}, 
                log=str(results), 
                other={}
            )
        
        except Exception as e:
            return BenchmarkResult(
                task_id=task_id, 
                is_resolved=False, 
                metrics={"score": 0}, 
                log={"error": str(e)}, 
                other={}
            )
        
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        # Only one task for CRAG benchmark
        return {"task_ids": ["0"], "error_message": None}
        
    def cleanup(self):
        pass


