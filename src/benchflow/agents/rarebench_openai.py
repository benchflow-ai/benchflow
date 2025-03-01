from openai import OpenAI
import os
from benchflow import BaseAgent
from typing import Dict, Any

class RarebenchAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")

    def call_api(self, env_info: Dict[str, Any]) -> str:
        messages = [
                    {"role": "system", "content": env_info["system_prompt"]},
                    {"role": "user", "content": env_info["user_prompt"]},
                ]
        try:
            client = OpenAI(
                api_key=self.api_key,  # This is the default and can be omitted
            )

            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.9,
            )
            content = response.choices[0].message.content
            action = self._extract_action(content)
            return action
        except Exception as e:
            raise