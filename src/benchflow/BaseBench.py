# base_bench.py
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import docker


class ColoredFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
           fmt='%(colored_level)s: -- %(name)s -- %(message)s',
           datefmt='%H:%M:%S'
       )

    def format(self, record):
        record.msg = " ".join(record.msg.strip().splitlines())
        return super().format(record).strip()

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(ColoredFormatter())
            logger.addHandler(file_handler)

    return logger


class BaseBenchConfig:
    required_env: List[str] = []
    optional_env: List[str] = []
    defaults: Dict[str, Any] = {}

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def validate(self):
        missing = [
            key for key in self.required_env
            if key not in self.params or not self.params[key]
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def get_env(self) -> Dict[str, str]:
        env = {}
        for key in self.required_env + self.optional_env:
            value = self.params.get(key)
            if value is not None:
                env[key] = str(value)
        for key, value in self.defaults.items():
            if key not in env:
                env[key] = str(value)
        return env

class BaseBench(ABC):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.docker_client = docker.from_env()

    def run_bench(self, task_id: str, agent_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        config = self.get_config(params, task_id)
        config.validate()

        bench_name = self.__class__.__name__
        timestamp = str(time.time())
        self.results_dir = os.path.abspath(f"./tmp/{bench_name}/results/{timestamp}/{task_id}")
        self.log_files_dir = os.path.abspath(f"./tmp/{bench_name}/logs/{timestamp}/{task_id}")
        os.makedirs(self.results_dir, exist_ok=True)    
        os.makedirs(self.log_files_dir, exist_ok=True)

        env = config.get_env()
        env.update({
            "AGENT_URL": agent_url,
            "TEST_START_IDX": str(task_id),
        })

        try:
            container = self.docker_client.containers.run(
                image=self.get_image_name(),
                environment=env,
                volumes=self.get_volumes(),
                remove=True,
                detach=True
            )

            for line in container.logs(stream=True):
                line_str = line.decode('utf-8').strip()
                self.logger.info(line_str)

            container.wait()

            result = self.get_result(task_id)
            if not self.validate_result(result):
                return self.format_result(task_id, False, 0, {"error": "Benchmark result is invalid", "result": str(result)})
            print(result)
            return self.format_result(task_id, result["is_resolved"], result["score"], result["message"], result["log"])
        except docker.errors.ImageNotFound:
            return self.format_result(task_id, False, 0, {"error": "Image not found"})
        except Exception as e:
            self.logger.exception("Error during benchmark execution:")
            return self.format_result(task_id, False, 0, {"error": str(e)})

    def format_result(self, task_id: str, is_resolved: bool, score: float, message: Dict[str, Any], log: Optional[str] = None) -> Dict[str, Any]:
        # TODO: inform the bff that the task is successful
        if log is None:
            log = message["error"]
        return {
            "task_id": task_id,
            "is_resolved": is_resolved,
            "score": score,
            "message": message,
            "log": log
        }

    def get_volumes(self) -> Dict[str, Dict[str, str]]:
        return {
            f"{self.results_dir}": {
                'bind': f"{self.get_results_dir_in_container()}",
                'mode': 'rw'
            },
            f"{self.log_files_dir}": {
                'bind': f"{self.get_log_files_dir_in_container()}",
                'mode': 'rw'
            },
            "/var/run/docker.sock": {
                'bind': "/var/run/docker.sock",
                'mode': 'rw'
            }
        }
    
    def validate_result(self, result: Dict[str, Any]) -> bool:
        if result.get("is_resolved") is None:
            return False
        if result.get("score") is None:
            return False
        if result.get("message") is None:
            return False
        return True
    
    @abstractmethod
    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
        # Benchmark need to deal with the END_IDX so that it can only run one task at a time
        # task_id is the start index of the task
        pass

    @abstractmethod
    def get_image_name(self) -> str:
        pass

    @abstractmethod
    def get_results_dir_in_container(self) -> str:
        pass

    @abstractmethod
    def get_log_files_dir_in_container(self) -> str:
        pass

    @abstractmethod
    def get_result(self, task_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return all task_ids and optional error messages.
        For example:
            { "task_ids": [...], "error_message": None }
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up benchmark resources.
        """
        pass
