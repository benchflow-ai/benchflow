from openai import OpenAI
from benchflow import BaseAgent
import os

class MedQAAgent(BaseAgent):
    def call_api(self, task_step_inputs) -> str:
        print(task_step_inputs)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task_step_inputs["user_prompt"]}],
        )
        return response.choices[0].message.content
