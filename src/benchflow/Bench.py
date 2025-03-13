import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from .utils import print_logo, spinner_animation, check_arguments, logger, get_agent_code

import requests

from .BaseAgent import BaseAgent

class Bench:
    def __init__(self, benchmark_name: str, bf_token: str):
        self.benchmark_name = benchmark_name
        self.bff_url = "https://benchflow.ai"
        self.bf_token = bf_token
        project_dir = Path(__file__).parents[2]
        self.results_dir = project_dir / "results" / self.benchmark_name
        print_logo()

    def run(self, 
            agents: BaseAgent | List[BaseAgent],
            requirements_txt: str,
            api: Dict[str, str],
            task_ids: List[str] = None,
            install_sh: str = None,
            require_gpu: bool = False,
            args: Dict[str, Any] = {}):
        """
        Run the benchmark.

        Args:
            agents (BaseAgent | List[BaseAgent]): agents to run
            requirements_txt (str): python style requirements.txt file
            api (Dict[str, str]): api info for your intelligence provider
            task_ids (List[str], optional): task ids to run. Defaults to None
            install_sh (str, optional): install.sh file. Defaults to None
            require_gpu (bool, optional): require gpu. Defaults to False
            args (Dict[str, Any], optional): arguments for benchmark. Defaults to {}.

        Returns:
            List[str]: run ids
        """

        agents, task_ids, requirements_txt, install_sh = check_arguments(agents, requirements_txt, task_ids, api, install_sh)

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

    def _send_tasks_to_bff(self, agent: BaseAgent, 
                           requirements_txt: str, install_sh: str, 
                           api: Dict[str, str], require_gpu: bool,
                           args: Dict[str, Any],
                           task_ids: List[str]):

        agent_code = get_agent_code(agent)

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
            "x-bf-source": "python-sdk 0.1.13"
        }

        response = requests.post(f"{self.bff_url}/api/v1/jobs/{self.benchmark_name}/new", json=payload, headers=headers)
        response.raise_for_status()

        task_info = response.json()
        job_id = task_info.get("jobId")
        logger.info(f"Tasks {task_ids} started successfully, job_id: {job_id}")

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

