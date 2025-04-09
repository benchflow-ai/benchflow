import os
import google.generativeai as genai
from benchflow import BaseAgent

class MedQAGeminiAgent(BaseAgent):
    def __init__(self, model="gemini-2.5-pro-preview-03-25"):
        super().__init__()
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)

    def call_api(self, task_step_inputs) -> str:
        print(f"Calling Gemini API with model: {self.model}")

        # Create a GenerativeModel object
        model = genai.GenerativeModel(self.model)

        # Generate content
        response = model.generate_content(task_step_inputs["user_prompt"])

        # Return the text response
        # Handle potential errors or empty responses
        if response.text:
            return response.text
        else:
            # If there's an issue with the response, return an error message
            return "Error: Unable to generate a response from Gemini."
