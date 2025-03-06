import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import requests

from .BaseAgent import BaseAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class Bench:
    def __init__(self, benchmark_name: str, bf_token: str):
        self.benchmark_name = benchmark_name
        self.bff_url = "https://staging.benchflow.ai"
        self.bf_token = bf_token
        project_dir = Path(__file__).parents[2]
        self.results_dir = project_dir / "results" / self.benchmark_name
        print_logo()

    def run(self, task_ids: List[Union[str, int]], 
            agents: Union[BaseAgent, List[BaseAgent]], 
            requirements_txt: str, 
            install_sh: str = None, 
            api: Dict[str, str] = None, 
            require_gpu: bool = False, 
            args: Dict[str, Any] = {}):
        
        if isinstance(task_ids, (str, int)):
            task_ids = [str(task_ids)]
        else:
            task_ids = [str(task) for task in task_ids]

        if isinstance(agents, BaseAgent):
            agents = [agents]
        
        results_ids = []
        try:
            for agent in agents:
                result_id = self._send_tasks_to_bff(task_ids, agent, requirements_txt, install_sh, api, require_gpu, args)
                if result_id:
                    results_ids.append(result_id)

            return results_ids

        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            return results_ids

    def _send_tasks_to_bff(self, task_ids: List[str], agent: BaseAgent, 
                           requirements_txt: str, install_sh: str, 
                           api: Dict[str, str], require_gpu: bool, 
                           args: Dict[str, Any]):
        logger.info(f"Sending tasks {task_ids} and setup scripts to BFF for agent {agent.__class__.__name__}")

        try:
            with open(requirements_txt, 'r') as f:
                requirements_txt = f.read()
            if install_sh:
                with open(install_sh, 'r') as f:
                    install_sh = f.read()
            agent_code = self._get_agent_code(agent)
        except Exception as e:
            logger.error(f"Failed to get agent code: {str(e)}")
            return None

        api['provider'] = api.get("provider", "")
        api['model'] = api.get("model", "")
        payload = {
            "task_ids": task_ids,
            "benchmark_name": self.benchmark_name,
            "params": args,
            "require_gpu": require_gpu,
            "requirements": requirements_txt if requirements_txt else "",
            "install_sh": install_sh if install_sh else "",
            "agent_code": agent_code if agent_code else "",
            "api": api
        }

        headers = {
            "x-bf-api-key": self.bf_token,
            "x-bf-source": "python-sdk 0.1.9"
        }

        try:
            response = requests.post(f"{self.bff_url}/api/v1/jobs/{self.benchmark_name}/new", json=payload, headers=headers)
            response.raise_for_status()

            task_info = response.json()
            job_id = task_info.get("jobId")
            logger.info(f"Tasks {task_ids} started successfully, job_id: {job_id}")
            return job_id
        
        except requests.exceptions.HTTPError as http_err:
            detail = http_err.response.json()
            logger.error(f"Task execution failed: {str(http_err)}, detail: {detail}")
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
        return None

    def get_results(self, job_ids: List[str]):
        """
        Get and download the results from the BFF.
        """
        results = {}
        jobs = set(job_ids)
        headers = {"x-bf-api-key": self.bf_token}
        start_time = time.time()
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner_animation, args=(stop_event, start_time))
        spinner_thread.start()

        try:
            while jobs:
                for job_id in list(jobs):
                    response = requests.get(f"{self.bff_url}/api/v1/jobs/{job_id}/", headers=headers)
                    response.raise_for_status()
                    job = response.json().get('job')
                    if job.get('status') != 'in_progress':
                        if job.get('status') == 'done':
                            spans = job.get('spans', {})
                            outputs = [span.get('outputJSON') for span in spans if span.get('outputJSON')]
                            results[job_id] = outputs
                        jobs.remove(job_id)
                if jobs:
                    time.sleep(10)
        finally:
            stop_event.set()
            spinner_thread.join()
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for job_id in job_ids:
            result_file = self.results_dir / f"{job_id}.json"
            result_file.write_text(json.dumps(results[job_id]))
            logger.info(f"Results saved to {result_file}")
        return results
    
    def _get_agent_code(self, agent: BaseAgent) -> str:
        agent_file = Path(sys.modules[agent.__class__.__module__].__file__)
        return agent_file.read_text()

def print_logo() -> None:
    logo = r"""

██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███████╗██╗      ██████╗ ██╗    ██╗    
██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██╔════╝██║     ██╔═══██╗██║    ██║    
██████╔╝█████╗  ██╔██╗ ██║██║     ███████║█████╗  ██║     ██║   ██║██║ █╗ ██║    
██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██╔══╝  ██║     ██║   ██║██║███╗██║    
██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║     ███████╗╚██████╔╝╚███╔███╔╝    
╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝     
                                                                                 
    """
    print(logo)

def spinner_animation(stop_event: threading.Event, start_time: float) -> None:
    spinner = ['|', '/', '-', '\\']
    spinner_index = 0
    bar_len = 19
    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        ch = spinner[spinner_index % len(spinner)]
        spinner_index += 1
        fill = elapsed % (bar_len + 1)
        bar = '[' + '#' * fill + '-' * (bar_len - fill) + ']'
        sys.stdout.write(f"\rWaiting for results... {ch} {bar} Elapsed: {elapsed}s")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()