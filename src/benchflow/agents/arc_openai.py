import os
import logging
import re
from typing import Any, Dict

from openai import OpenAI

from benchflow import BaseAgent

logger = logging.getLogger(__name__)

class ARCAgent(BaseAgent):
    """
    Agent implementation for ARC benchmark using OpenAI API.
    
    This agent handles the ARC multiple-choice questions by formatting the input
    appropriately and sending it to the OpenAI API.
    """
    
    def __init__(self, model_name="gpt-4"):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        
    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        """
        Process ARC task inputs and call the OpenAI API.
        
        The task_step_inputs will contain:
        - question: The question text
        - choices: A list of choices
        """
        try:
            # Extract task information
            question = task_step_inputs.get("question", "")
            choices = task_step_inputs.get("choices", [])
            
            # Format the prompt
            prompt = self._format_prompt(question, choices)
            
            # Call the OpenAI API
            logger.info(f"[ARCAgent]: Calling OpenAI API")
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant solving multiple-choice science questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"[ARCAgent]: Generated answer: {content}")
            
            # Extract the answer (A, B, C, or D)
            answer = self._extract_answer(content)
            logger.info(f"[ARCAgent]: Extracted answer: {answer}")
            
            return answer
            
        except Exception as e:
            logger.error(f"[ARCAgent]: Error calling OpenAI API: {e}")
            raise
    
    def _format_prompt(self, question: str, choices: list) -> str:
        """
        Format the prompt for the ARC task.
        
        Args:
            question: The question text
            choices: A list of choices
        """
        # Format choices with letters
        formatted_choices = ""
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D, ...
            formatted_choices += f"{letter}) {choice}  \n"
        
        # Create the full prompt
        prompt = (
            f"Answer the following multiple choice question. The entire content of your response "
            f"should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER "
            f"is one of A, B, C, D.\n\n"
            f"{question}\n\n"
            f"{formatted_choices}"
        )
        
        return prompt
    
    def _extract_answer(self, response: str) -> str:
        """
        Extract the answer (A, B, C, or D) from the model's response.
        
        Args:
            response: The model's response
        """
        
        match = re.search(r"ANSWER:\s*([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
       
        match = re.search(r"^[A-D]$", response.strip(), re.IGNORECASE)
        if match:
            return match.group(0).upper()
        
        
        match = re.search(r"([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Default to "A" if no answer found
        logger.warning(f"[ARCAgent]: Could not extract answer from response: {response}")
        return "A"