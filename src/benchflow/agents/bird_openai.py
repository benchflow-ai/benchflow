import os
from openai import OpenAI
from benchflow import BaseAgent

class BridAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")

    def call_api(self, task_step_inputs):
        client = OpenAI(
                api_key=self.api_key,  # This is the default and can be omitted
            )
        messages = [
            {"role": "system", "content": task_step_inputs["system_prompt"]},
            {"role": "user", "content": task_step_inputs["user_prompt"]}
        ]
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0.9,
        )
        content = response.choices[0].message.content
        return content
