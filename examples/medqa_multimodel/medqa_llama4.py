import os
from openai import OpenAI
from benchflow import BaseAgent

class MedQALlama4Agent(BaseAgent):
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
                "X-Title": "BenchFlow MedQA Test"        # Optional for rankings
            }
        )

    def call_api(self, task_step_inputs) -> str:
        print(f"Calling OpenRouter API with model: {self.model}")

        # Create a chat completion
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": task_step_inputs["user_prompt"]}
            ],
            # OpenRouter headers are passed at the client level
        )

        # Note: Headers should be set when initializing the client
        # self.client = OpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=api_key,
        #     default_headers={
        #         "HTTP-Referer": "https://benchflow.ai",
        #         "X-Title": "BenchFlow MedQA Test"
        #     }
        # )

        # Return the text response
        return response.choices[0].message.content
