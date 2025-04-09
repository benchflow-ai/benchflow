import os
from anthropic import Anthropic
from benchflow import BaseAgent

class MedQAClaudeAgent(BaseAgent):
    def __init__(self, model="claude-3-7-sonnet-20250219"):
        super().__init__()
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.client = Anthropic(api_key=api_key)

    def call_api(self, task_step_inputs) -> str:
        print(f"Calling Claude API with model: {self.model}")

        # Create a message
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": task_step_inputs["user_prompt"]
                }
            ]
        )

        # Return the text response
        return message.content
