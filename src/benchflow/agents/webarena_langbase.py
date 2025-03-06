import logging
import os
import re

import requests

from benchflow import BaseAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class WebarenaAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("LANGBASE_API_KEY")
        self.url = "https://api.langbase.com/v1/pipes/run"
 
    def _construct_message(self, env_info) -> str:
        # Deserialize observation to original format for API call
        return f"""OBSERVATION: {env_info['observation']["text"]} URL: {env_info['url']} OBJECTIVE: {env_info['intent']} PREVIOUS ACTION: {env_info['previous_action'] }"""
 
    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = "```"
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise Exception(
                f'Cannot find the action in "{response}"'
            )
 
    def call_api(self, task_step_inputs) -> str:
        message = self._construct_message(task_step_inputs)
        print(message)
        data = {
            'messages': [{'role': 'user', 'content': message}],
            'stream': False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            logger.info("[UserAgent]: Calling API")
            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()
 
            if not response.ok:
                logger.error(f"[UserAgent]: API error: {response.json()}")
                raise Exception(f"API error: {response.json()}")
 
            result = response.json()
            if result['success']:
                raw_completion = result["completion"]
                logger.info(f"raw completion: {raw_completion}")
                action = self._extract_action(raw_completion)
                logger.info(f"[UserAgent]: Got action from API: {action}")
                return action
 
            raise Exception("API call failed.")
        except Exception as e:
            logger.error(f"[UserAgent]: Error calling API: {str(e)}")
            raise