import json
import logging
import os
import re
import shutil
import subprocess

from benchflow import BaseAgent


class MLEAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.model_name = "gpt-4o"

    def call_api(self, env_info) -> str:
        """
        Calls the API or executes a subprocess to run the benchmark agent.
        """
        competition_id = env_info.get("competition_id", "default_competition")
        seed = env_info.get("seed", 42)

        # Cleanup previous runs
        shutil.rmtree("trajectories/root/", ignore_errors=True)

        # Command to run MLEBench agent
        cmd = f"mlebench-agent run --competition {competition_id} --seed {seed} --model {self.model_name}"
        
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logging.info("MLEBenchAgent result: %s", result.stdout)
        logging.error("MLEBenchAgent error: %s", result.stderr)

        return self.parse_action(competition_id, result.stdout)

    def parse_action(self, competition_id: str, log_content: str) -> str:
        """
        Extracts the agent's predictions from the logs.
        """
        pattern = r"Predictions saved to\s+((?:/.*\n\s*)+.*predictions\.json)"
        match = re.search(pattern, log_content, re.DOTALL)

        if match:
            file_path_raw = match.group(1)
            file_path = re.sub(r"\s+", "", file_path_raw)

            if not os.path.exists(file_path):
                logging.error(f"Error: {file_path} does not exist")
                return None
            else:
                try:
                    with open(file_path, 'r') as f:
                        predictions = json.load(f)
                    
                    # Extracting the relevant prediction for the competition
                    action = predictions.get(competition_id, {}).get("model_output", None)

                    if action is None:
                        logging.error(f"No valid model output found for {competition_id}")
                    
                    return action
                except json.JSONDecodeError:
                    logging.error(f"Error: {file_path} is not a valid JSON")
                    return None
                except Exception as e:
                    logging.error(f"Error: {str(e)}")
                    return None
        else:
            logging.error("Error: No predictions file path found in logs")
            return None
