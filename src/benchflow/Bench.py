import base64
import json
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Union

import requests
from requests.exceptions import HTTPError

from .BaseAgent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# utility function to encode content to base64
def encode_base64(content: str) -> str:
    return base64.b64encode(content.encode()).decode() if content else None

class Bench:
    def __init__(self, benchmark: Dict[str, Any]):
        self.benchmark_url = benchmark["benchmark_url"]
        self.benchmark_name = benchmark["benchmark_name"]
        self.resource_manager_url = "http://ec2-3-232-182-160.compute-1.amazonaws.com:10000"
        self.running_agents = {}
        self.results = {}

        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        
    def run(self, agents: Union[BaseAgent, List[BaseAgent]], 
            requirements_dir: str, 
            install_sh_dir: str = None, 
            api: Dict[str, str] = None, 
            require_gpu: bool = False, 
            params: Dict[str, Any] = {},
            task_ids: Union[str|int, List[str|int], None] = None):
        if task_ids is None:
            task_ids = self._get_all_task_ids()
        elif isinstance(task_ids, str|int):
            task_ids = [str(task_ids)]
        if isinstance(agents, BaseAgent):
            agents = [agents]
        
        results_ids = []
        try:
            for agent in agents:
                agent_url = self._deploy_agent(agent, require_gpu, requirements_dir, install_sh_dir, api)
                if not agent_url:
                    logger.error(f"Deployment failed on {agent.__class__.__name__}")
                    self._cleanup()
                    continue
                for task_id in task_ids:
                    task_id = str(task_id)
                    results_ids.append(self._run_single_task(task_id, agent_url, agent, params))
            self._cleanup()
            return results_ids
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            self._cleanup()
            return results_ids

    def get_results(self, run_ids: List[str]):
        results = [self.results[run_id] for run_id in run_ids]
        pretty_results = json.dumps(results, indent=4, ensure_ascii=False)
        print(pretty_results)
        return results

    def _deploy_agent(self, 
                      agent: BaseAgent, 
                      require_gpu: bool, 
                      requirements_dir: str, 
                      install_sh_dir: str, 
                      api: Dict[str, str] = None):
        logger.info(f"Starting deployment of agent {agent.__class__.__name__}")
        try:
            with open(requirements_dir, 'r') as f:
                requirements_txt = f.read()

            install_sh = None
            if install_sh_dir:
                with open(install_sh_dir, 'r') as f:
                    install_sh = f.read()

            agent_code = self._get_agent_code(agent)
            payload = {
                "agent_code": encode_base64(agent_code),
                "require_gpu": require_gpu,
                "requirements_txt": encode_base64(requirements_txt),
                "install_sh": encode_base64(install_sh),
                "benchmark_name": self.benchmark_name,
                "api": api
            }

            response = requests.post(f"{self.resource_manager_url}/deploy", json=payload)
            response.raise_for_status()

            deploy_info = response.json()
            host, port = deploy_info["host"], deploy_info["port"]
            agent_url = f"http://{host}:{port}"

            self.running_agents[str(agent.__class__.__name__)] = {"host": host, "port": port, "start_time": time.time()}
            logger.info(f"{agent.__class__.__name__} deployed successfully on port {port}")

            return agent_url
        
        except HTTPError as e:
            logger.error(f"Deployment failed: {str(e)}")
            error_detail = response.json()['detail']
            logger.error(f"Deployment failed: {error_detail}")
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
        return False

    def _run_single_task(self, task_id: str,
                         agent_url: str, 
                         agent: BaseAgent, 
                         params: Dict[str, Any] = {}):
        
        logger.info(f"Starting task {task_id} on {agent.__class__.__name__}")
        
        try:
            url = f"{self.benchmark_url}/api/v1/{self.benchmark_name}/evaluate"
            response = requests.post(
                url,
                json={
                    "task_id": task_id,
                    "agent_url": agent_url,
                    "params": params
                }
            )
            response.raise_for_status()
            logger.info(f"Task {task_id} on {agent.__class__.__name__} finished")
            self.results[task_id] = response.json()
            return task_id
        
        except Exception as e:
            error_detail = response.json()['detail']
            logger.error(f"Task {task_id} failed: {str(e)}")
            logger.error(f"Task {task_id} error detail: {error_detail}")
            self.results[task_id] = "failed"
            return task_id
    
    def _get_all_task_ids(self) -> List[str]:
        logger.info(f"Getting all task ids for {self.benchmark_name}")
        payload = {}
        response = requests.post(f"{self.benchmark_url}/api/v1/{self.benchmark_name}/tasks", json=payload)
        response.raise_for_status()
        print(response.json()["task_ids"])
        return response.json()["task_ids"]
        
    def _get_agent_code(self, agent: BaseAgent) -> str:
        agent_file = sys.modules[agent.__class__.__module__].__file__
        with open(agent_file, 'r') as f:
            return f.read()

    def _cleanup(self):
        for agent_name, agent_info in self.running_agents.items():
            logger.info(f"Releasing agent {agent_name} on {agent_info['host']}:{agent_info['port']}")
            try:
                response = requests.post(
                    f"{self.resource_manager_url}/release",
                    json={
                        "host": agent_info["host"],
                        "port": agent_info["port"]
                    }
                )
                response.raise_for_status()
                logger.info(f"Release successfully for agent {agent_name} on {agent_info['host']}:{agent_info['port']}")
            except Exception as e:
                logger.error(f"Release request failed for agent {agent_name} on {agent_info['host']}:{agent_info['port']}: {str(e)}")
    
    def _handle_exit(self, signum, frame):
        logger.info(f"Received termination signal {signum}, cleaning up...")
        self._cleanup()
        sys.exit(0)