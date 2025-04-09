import os
from typing import Any, Dict

from openai import OpenAI
from benchflow import BaseAgent

class RareBenchGPT4oAgent(BaseAgent):
    def __init__(self, model="gpt-4o"):
        super().__init__()
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        
    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        print(f"Calling OpenAI API with model: {self.model}")
        
        # Create a chat completion
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.9,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": task_step_inputs["system_prompt"]},
                {"role": "user", "content": task_step_inputs["prompt"]}
            ]
        )
        
        # Return the text response
        return response.choices[0].message.content
