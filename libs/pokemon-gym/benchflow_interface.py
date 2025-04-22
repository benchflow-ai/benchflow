from benchflow import BaseBench
from benchflow.BaseBench import BenchmarkResult
from benchflow.schemas import BenchArgs
from typing import Dict, Any
import os
import json

class PokemonBench(BaseBench):
    def get_args(self, task_id: str) -> BenchArgs:
        arguments = {
            "required": [],
            "optional": {"MAX_DURATION": 30 * 60},
        }
        return BenchArgs(arguments)
    
    def get_image_name(self) -> str:
        return "kirk2000/benchflow:pokemongym-v1"
    
    def get_results_dir_in_container(self) -> str:
        return "/app/evaluation_sessions/latest_evaluation"
    
    def get_log_files_dir_in_container(self) -> str:
        return "/app/evaluation_sessions/latest_evaluation"
    
    def get_result(self, task_id: str) -> BenchmarkResult:
        try:
            with open(os.path.join(self.results_dir, "summary.json"), "r") as f:
                summary = json.load(f)
            with open(os.path.join(self.results_dir, "results.csv"), "r") as f:
                results = f.read()
            summary = {
                'duration_minutes': summary['duration_minutes'],
                'total_steps': summary['total_steps'],
                'final_score': summary['final_score'],
                'total_execution_time': summary['timing']['total_execution_time'],
                'average_time_per_step': summary['timing']['average_time_per_step'],
                'pokemon_discovered': summary['stats']['pokemon_discovered'],
                'badges_earned': summary['stats']['badges_earned'],
                'locations_visited': summary['stats']['locations_visited'],
            }
            return BenchmarkResult(task_id=task_id, is_resolved=True, metrics=summary, log={"details": results}, other={})
        except Exception as e:
            return BenchmarkResult(task_id=task_id, is_resolved=False, metrics={}, log={"error": str(e)}, other={"error": str(e)})
    
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        return {
            "task_ids": ["0"],
            "error_message": None,
        }
