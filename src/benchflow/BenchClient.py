import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, final
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

class BenchClient(ABC):
    """
    The BenchClient is used to make the benchmark client so that it can communicate with the agent server.
    You need to extend this class in your benchmark entrypoint (e.g. run.py).
    You need to implement the following methods:
        - prepare_input
        - parse_response
    """
    
    def __init__(self, intelligence_url: str, max_retry: int = 1):
        self.intelligence_url = intelligence_url.rstrip('/')
        self.max_retry = max_retry
        logger.info(f"[{self.__class__.__name__}] Initialized with intelligence_url: {intelligence_url}")

    @final
    def get_response(self, raw_step_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the response from the agent. You should use this method to get the response from the agent.
        """
        if raw_step_inputs is None:
            raise ValueError("raw_step_inputs cannot be None")
        
        task_step_inputs = self.prepare_input(raw_step_inputs)
        
        for attempt in range(self.max_retry):
            try:
                response = requests.post(
                    urljoin(self.intelligence_url, "response"),
                    json=task_step_inputs
                )
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == self.max_retry - 1:
                    raise Exception(f"Failed to get response after {self.max_retry} attempts: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                continue

        try:
            raw_response = response.json()
            logger.info(f"[{self.__class__.__name__}] Received response: {raw_response}")
            
            parsed_response = self.parse_response(raw_response)
            parsed_response["raw_response"] = raw_response
            
            return parsed_response
            
        except KeyError as e:
            raise ValueError(f"Invalid response format from agent: {str(e)}")
        except Exception as e:
            raise Exception(f"Error parsing response: {str(e)}")
    
    @abstractmethod
    def prepare_input(self, raw_step_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            raw_step_inputs: Dict[str, Any]
            For example, 
                If your benchmark is a web agent benchmark, the raw_input_data could be the observation from the web page.
                if your benchmark is a Q&A benchmark, the raw_input_data could be the question.
                You should define the keys in the raw_input_data.
                If your benchmark don't need to deal with the raw_input_data, you can just return the raw_input_data.
        Output:
            Dict[str, Any]
            The input data of the task to be sent to the agent.
            And add the keys to the benchmark documentation. # To Be Done in benchflow v0.2.0
        """
        pass

    @abstractmethod
    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Input:
            raw_response: str
            The raw response from the agent.

        You can specify the format of the raw_action in the benchmark documentation. # To Be Done in benchflow v0.2.0
        So that agent developers can know what to return.

        For example,
            you can specify the format of the raw_response as follows:
            ```
            "action_type": click
            "action_arguments": arguments
            ```
            so that you can use regex to parse the response_type and response_arguments from the raw_response.
            and return the parsed_action as follows:
            ```
            {
                "action_type": click,
                "action_arguments": arguments
            }
            ```

        Output:
            parsed_response: Dict[str, Any]
            The parsed response.
            You need to specify the keys in the parsed_response that you want to send to the benchmark.
            And add the keys to the benchmark documentation. # To Be Done in benchflow v0.2.0
        """
        pass
