import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, final

import uvicorn
from fastapi import FastAPI, HTTPException

from benchflow.schemas.InputData import TaskStepInputs

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    You need to extend this class to make your agent a server.
    So that it can communicate with the benchmark client.
    If you want to integrate your agent with BenchFlow, you need to implement the following methods:
    ```
    - call_api
    ```
    """
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()

    @final
    def setup_routes(self):
        """
        Setup the routes for the agent.
        """
        @self.app.post("/action")
        async def take_action(input_data: TaskStepInputs):
            try:
                if input_data.task_step_inputs.get("env_info") is not None:
                    response = self.call_api(input_data.task_step_inputs.get("env_info"))
                else:
                    response = self.call_api(input_data.task_step_inputs)
                logger.info(f"[BaseAgent]: Got response from API: {response}")
                return response
            except Exception as e:
                logger.error(f"[BaseAgent]: Error getting response: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/")
        async def root():
            return {"message": "Welcome to Benchmarkthing Agent API"}

    @final
    def run_with_endpoint(self, host: str, port: int):
        """
        Run the agent server.
        """
        logger.info(f"Starting agent server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

    @abstractmethod 
    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        """
        You can get the request information from the task_step_inputs parameter.
        The task_step_inputs is a dictionary that contains the keys provided by the benchmark client.
        You need to refer to the benchmark documentation to get the keys.

        This method is called when the agent server receives a request from the benchmark client.
        You need to implement this method to make your agent work and return the response to the benchmark client.
        Your response could be a real action(e.g. click, scroll, etc) or just any prediction(e.g. code, text, etc) needed by the benchmark.
        """
        pass