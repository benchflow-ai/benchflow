import os
from typing import Any, Dict

import google.generativeai as genai
from benchflow import BaseAgent

class RareBenchGeminiAgent(BaseAgent):
    def __init__(self, model="gemini-2.5-pro-preview-03-25"):
        super().__init__()
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        
    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        print(f"Calling Gemini API with model: {self.model}")
        
        # Configure the model
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        # Create the prompt
        system_prompt = task_step_inputs["system_prompt"]
        user_prompt = task_step_inputs["prompt"]
        
        # Initialize the model
        model = genai.GenerativeModel(model_name=self.model,
                                     generation_config=generation_config)
        
        # Generate content
        response = model.generate_content([
            {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
        ])
        
        # Return the text response
        return response.text
