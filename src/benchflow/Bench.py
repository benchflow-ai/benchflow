import base64
import json
import logging
import sys
from typing import Any, Dict, List, Union

import requests
from requests.exceptions import HTTPError

from .BaseAgent import BaseAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def encode_base64(content: str) -> str:
    return base64.b64encode(content.encode()).decode() if content else None

class Bench:
    def __init__(self, benchmark_name: str, bf_token: str):
        self.benchmark_name = benchmark_name
        self.bff_url = f"https://staging.benchflow.ai"
        self.bf_token = bf_token

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

        api['provider'] = api.get("provider", "")
        api['model'] = api.get("model", "")
        payload = {
            "task_ids": task_ids,
            "benchmark_name": self.benchmark_name,
            "params": params,
            "require_gpu": require_gpu,
            "requirements": requirements_txt if requirements_txt else "",
            "install_sh": install_sh if install_sh else "",
            "agent_code": agent_code if agent_code else "",
            "api": api
        }

        headers = {
            "x-bf-api-key": self.bf_token,
            "x-bf-source": "python-sdk 0.1.6"
        }

        try:
            response = requests.post(f"{self.bff_url}/api/v1/jobs/{self.benchmark_name}/new", json=payload, headers=headers)
            response.raise_for_status()

            print(response.json())
            task_info = response.json()
            job_id = task_info.get("jobId")
            logger.info(f"Tasks {task_ids} started successfully, job_id: {job_id}")
            return job_id
        
        except HTTPError as e:
            logger.error(f"Task execution failed: {str(e)}")
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
        return None

    def get_results(self, job_ids: List[str]):
        print(job_ids)
        results = []
        for job_id in job_ids:
            headers = {
                "x-bf-api-key": self.bf_token
            }
            print(headers)
            print(job_id)
            try:
                response = requests.get(f"{self.bff_url}/api/v1/jobs/{job_id}/", headers=headers)
                response.raise_for_status()

                result = response.json()
                pretty_result = json.dumps(result, indent=4, ensure_ascii=False)
                print(pretty_result)
                results.append(result)
            
            except HTTPError as e:
                logger.error(f"Failed to get results: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to get results: {str(e)}")
        return results
    
    def _get_agent_code(self, agent: BaseAgent) -> str:
        agent_file = sys.modules[agent.__class__.__module__].__file__
        with open(agent_file, 'r') as f:
            return f.read()