import logging
import os

from openai import OpenAI

from benchflow import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MMLUAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        print(self.api_key)

    def call_api(self, task_step_inputs):
        prompt = task_step_inputs["prompt"]
        input_text = task_step_inputs["input_text"]

        try:
            logger.info("[UserAgent]: Calling OpenAI API")
            client = OpenAI(
                api_key=self.api_key,  # This is the default and can be omitted
            )

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text}
                ],
                model="gpt-4o-mini",
                temperature=0.9,
            )
            content = response.choices[0].message.content
            logger.info(f"[UserAgent]: Got action: {content}")
            return content
        except Exception as e:
            logger.error(f"[UserAgent]: Error calling OpenAI API: {e}")
            raise