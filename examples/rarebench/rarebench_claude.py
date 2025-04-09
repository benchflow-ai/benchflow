import os
from typing import Any, Dict

from anthropic import Anthropic
from benchflow import BaseAgent

class RareBenchClaudeAgent(BaseAgent):
    def __init__(self, model="claude-3-7-sonnet-20250219"):
        super().__init__()
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.client = Anthropic(api_key=api_key)

    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        print(f"Calling Claude API with model: {self.model}")

        # Create a message
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.9,
            system=task_step_inputs["system_prompt"],
            messages=[
                {
                    "role": "user",
                    "content": task_step_inputs["prompt"]
                }
            ]
        )

        # Return the text response
        return message.content[0].text
