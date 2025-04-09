import os
from typing import Any, Dict

from openai import OpenAI
from benchflow import BaseAgent

class RareBenchLlama4Agent(BaseAgent):
    def __init__(self, model="meta-llama/llama-4-maverick"):
        super().__init__()
        self.model = model
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://benchflow.ai",  # Optional for rankings
                "X-Title": "BenchFlow RareBench Test"    # Optional for rankings
            }
        )
        
    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        print(f"Calling OpenRouter API with model: {self.model}")
        
        # Create a chat completion
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.9,
            messages=[
                {"role": "system", "content": task_step_inputs["system_prompt"]},
                {"role": "user", "content": task_step_inputs["prompt"]}
            ]
        )
        
        # Return the text response
        return response.choices[0].message.content
