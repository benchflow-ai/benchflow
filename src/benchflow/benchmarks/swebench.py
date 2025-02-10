import json
import os
from typing import Any, Dict

from datasets import Dataset, load_dataset

from benchflow import BaseBench, BaseBenchConfig


class SwebenchConfig(BaseBenchConfig):
    required_env = []
    optional_env = ["INSTANCE_IDS", "MAX_WORKERS", "RUN_ID"]

    def __init__(self, params: Dict[str, Any], task_id: str):
        params.setdefault("INSTANCE_IDS", task_id)
        params.setdefault("MAX_WORKERS", 1)
        params.setdefault("RUN_ID", task_id)
        super().__init__(params)


class SwebenchBench(BaseBench):
    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
        return SwebenchConfig(params, task_id)

    def get_image_name(self) -> str:
        return "kirk2000/benchflow:swebench-v1"

    def get_results_dir_in_container(self) -> str:
        return "/app/results"

    def get_log_files_dir_in_container(self) -> str:
        return "/app/logs"

    def get_result(self, task_id: str) -> Dict[str, Any]:
        results_file = os.path.join(self.results_dir, f"self_model.{task_id}.json")
        model_prediction_file = os.path.join(self.log_files_dir, f"run_evaluation/{task_id}/self_model/{task_id}/patch.diff")
        report_file = os.path.join(self.log_files_dir, f"run_evaluation/{task_id}/self_model/{task_id}/report.json")
        try:
            with open(results_file, 'r') as f:
                result_data = json.load(f)
            total_instances = result_data.get("total_instances", 1)
            resolved_instances = result_data.get("resolved_instances", 0)
            pass_rate = resolved_instances / total_instances if total_instances else 0
            with open(model_prediction_file, 'r') as f:
                model_prediction = f.read()
            with open(report_file, 'r') as f:
                report = json.load(f)
            return {
                    "is_resolved": pass_rate > 0.99,
                    "score": pass_rate,
                    "message": {"details": result_data},
                    "log": model_prediction + "\n" + json.dumps(report),
                }
        except Exception as e:
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": str(e)},
                "log": str(e),
            }
        
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        try:
            dataset: Dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
            dataset_ids = [instance["instance_id"] for instance in dataset]
            return {"task_ids": dataset_ids, "error_message": None}
        except Exception as e:
            return {"task_ids": [], "error_message": str(e)}
    
    def cleanup(self):
        pass
