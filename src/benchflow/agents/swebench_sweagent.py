import json
import logging
import os
import re
import shutil
import subprocess

from benchflow import BaseAgent


class SWEAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.model_name = "gpt-4o"
        
    def call_api(self, task_step_inputs) -> str:
        instance_id = task_step_inputs["instance_id"]
        shutil.rmtree("trajectories/root/", ignore_errors=True)
        cmd = f"sweagent run-batch --agent.model.name {self.model_name} --agent.model.per_instance_cost_limit 0.10 --instances.split test --instances.filter {instance_id}"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logging.info("sweagent result: %s", result.stdout)
        logging.error("sweagent error: %s", result.stderr)
        return self.parse_action(instance_id, result.stdout)
        
    def parse_action(self, instance_id: str, log_content: str) -> str:
        pattern = r"Wrote merged predictions to\s+((?:/.*\n\s*)+.*preds\.json)"
        match = re.search(pattern, log_content, re.DOTALL)
        
        if match:
            file_path_raw = match.group(1)
            file_path = re.sub(r"\s+", "", file_path_raw)
            
            if not os.path.exists(file_path):
                print(f"error: {file_path} not exists")
                return None
            else:
                try:
                    with open(file_path, 'r') as f:
                        predictions = json.load(f)
                    action = predictions[instance_id]['model_patch']
                    return action
                except json.JSONDecodeError:
                    print(f"error: {file_path} not a valid json")
                    return None
                except Exception as e:
                    print(f"error: {str(e)}")
                    return None
        else:
            print("error: no predictions file path")
            return None