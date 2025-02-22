import json
import os
from typing import Any, Dict

from datasets import load_dataset

from benchflow import BaseBench, BaseBenchConfig


class MMLUPROConfig(BaseBenchConfig):
    required_env = []
    optional_env = []
    defaults = {}

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)


class MMLUPROBench(BaseBench):
    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
        return MMLUPROConfig(params)

    def get_image_name(self) -> str:
        return "kirk2000/benchflow:mmlu-pro-v1"

    def get_results_dir_in_container(self) -> str:
        return "/app/eval_results"

    def get_log_files_dir_in_container(self) -> str:
        return "/app/logs" # Useless

    def get_result(self, task_id: str) -> Dict[str, Any]:
        summary_file = os.path.join(self.results_dir, f"{task_id}_summary.json")
        result_file = os.path.join(self.results_dir, f"{task_id}_result.json")
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            with open(result_file, 'r') as f:
                result = json.load(f)

            log = ''.join(json.dumps(item, ensure_ascii=False) for item in result)
            return {
                    "is_resolved": True,
                    "score": summary['total']['acc'],
                    "message": {"details": summary},
                    "log": log,
                }
        except Exception as e:
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": str(e)},
                "log": str(e),
            }
        
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        categories = dataset['category']
        distinct_categories = sorted(set(categories))
        return {"task_ids": distinct_categories, "error_message": ""}
    
    def cleanup(self):
        pass
