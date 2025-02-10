import base64
import json
import logging
from typing import Any, Dict, List, Union

import requests
from requests.exceptions import HTTPError

from .BaseAgent import BaseAgent

logger = logging.getLogger(__name__)

def encode_base64(content: str) -> str:
    return base64.b64encode(content.encode()).decode() if content else None

class Bench:
    def __init__(self, benchmark_name: str, benchflow_token: str, bff_url: str):
        self.benchmark_name = benchmark_name
        self.bff_url = bff_url
        self.benchflow_token = benchflow_token

    def run(self, task_ids: List[Union[str, int]], 
            agents: Union[BaseAgent, List[BaseAgent]], 
            requirements_dir: str, 
            install_sh: str = None, 
            api: Dict[str, str] = None, 
            require_gpu: bool = False, 
            params: Dict[str, Any] = {}):
        
        if isinstance(task_ids, (str, int)):
            task_ids = [str(task_ids)]
        else:
            task_ids = [str(task) for task in task_ids]

        if isinstance(agents, BaseAgent):
            agents = [agents]
        
        results_ids = []
        try:
            for agent in agents:
                result_id = self._send_tasks_to_bff(task_ids, agent, requirements_dir, install_sh, api, require_gpu, params)
                if result_id:
                    results_ids.append(result_id)

            return results_ids

        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            return results_ids

    def _send_tasks_to_bff(self, task_ids: List[str], agent: BaseAgent, 
                           requirements_dir: str, install_sh_dir: str, 
                           api: Dict[str, str], require_gpu: bool, 
                           params: Dict[str, Any]):
        logger.info(f"Sending tasks {task_ids} and setup scripts to BFF for agent {agent.__class__.__name__}")

        try:
            with open(requirements_dir, 'r') as f:
                requirements_txt = f.read()
        except Exception as e:
            logger.error(f"Failed to read requirements.txt: {str(e)}")
            requirements_txt = ""

        install_sh = None
        if install_sh_dir:
            try:
                with open(install_sh_dir, 'r') as f:
                    install_sh = f.read()
            except Exception as e:
                logger.error(f"Failed to read install.sh: {str(e)}")
                install_sh = ""

        try:
            agent_code = self._get_agent_code(agent)
        except Exception as e:
            logger.error(f"Failed to get agent code: {str(e)}")
            agent_code = ""

        payload = {
            "task_ids": task_ids,
            "benchmark_name": self.benchmark_name,
            "params": params,
            "require_gpu": require_gpu,
            "requirements_txt": encode_base64(requirements_txt),
            "install_sh": encode_base64(install_sh),
            "agent_code": encode_base64(agent_code),
            "api": api
        }

        headers = {
            "Authorization": f"Bearer {self.benchflow_token}"
        }

        try:
            response = requests.post(f"{self.bff_url}/tasks/run", json=payload, headers=headers)
            response.raise_for_status()

            task_info = response.json()
            run_id = task_info.get("run_id")
            logger.info(f"Tasks {task_ids} started successfully, run_id: {run_id}")
            return run_id
        
        except HTTPError as e:
            logger.error(f"Task execution failed: {str(e)}")
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
        return None

    def get_results(self, run_ids: List[str]):
        payload = {"run_ids": run_ids}
        headers = {
            "Authorization": f"Bearer {self.benchflow_token}"
        }

        try:
            response = requests.post(f"{self.bff_url}/tasks/results", json=payload, headers=headers)
            response.raise_for_status()

            results = response.json()
            pretty_results = json.dumps(results, indent=4, ensure_ascii=False)
            print(pretty_results)
            return results
        
        except HTTPError as e:
            logger.error(f"Failed to get results: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to get results: {str(e)}")
        return []