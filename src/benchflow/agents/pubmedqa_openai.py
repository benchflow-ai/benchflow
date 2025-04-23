import os
import json
import logging
import re
from typing import Dict, Any, List

from openai import OpenAI

from benchflow import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubMedQAAgent(BaseAgent):
    """
    Agent for the PubMedQA benchmark using OpenAI API.
    
    This agent handles biomedical yes/no/maybe questions based on PubMed abstracts.
    """
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.0):
        """
        Initialize the PubMedQA agent.
        
        Args:
            model_name: The OpenAI model to use
            temperature: Temperature for generation
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
    
    def format_prompt(self, record: Dict[str, Any]) -> str:
        """Format the prompt for PubMedQA questions."""
        context = record["context"]
        question = record["question"]
        
        prompt = f"Context: {context}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "A) yes\nB) no\nC) maybe\n\n"
        prompt += "Based on the context, answer the medical question with just the letter (A, B, or C)."
        
        return prompt
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the PubMedQA task.
        
        Returns:
            System prompt string
        """
        return (
            "You are a medical expert answering biomedical research questions based on PubMed abstracts. "
            "Your task is to determine if the answer to the question is yes, no, or maybe, based solely on the provided context. "
            "Respond with just the letter of the answer option (A, B, or C)."
        )
    
    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the PubMedQA dataset.
        
        Args:
            example: A dictionary containing the example data
            
        Returns:
            Dictionary with example data and model response
        """
        context = example["context"]
        question = example["question"]
        
        # Format the prompt
        prompt = self.format_prompt(example)
        
        # Call the API
        try:
            response = OpenAI(api_key=self.api_key).chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            model_response = response.choices[0].message.content.strip()
            
            # Extract just the letter if there's additional content
            if model_response and len(model_response) > 0:
                answer_letter = model_response[0].upper()
            else:
                answer_letter = ""
            
            # Map the letter back to yes/no/maybe
            letter_to_answer = {
                "A": "yes",
                "B": "no",
                "C": "maybe"
            }
            predicted_answer = letter_to_answer.get(answer_letter, "")
            
            # Check if correct
            correct_answer = example["answer"][0].lower()  # answer is provided as a list, e.g., ['yes']
            is_correct = predicted_answer == correct_answer
            
            return {
                "example_id": example.get("id", ""),
                "context": context[:100] + "..." if len(context) > 100 else context,  # Truncate for brevity
                "question": question,
                "correct_answer": correct_answer,
                "model_answer": predicted_answer,
                "response": model_response,
                "is_correct": is_correct
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "example_id": example.get("id", ""),
                "context": context[:100] + "..." if len(context) > 100 else context,
                "question": question,
                "correct_answer": example["answer"][0].lower(),
                "model_answer": "",
                "response": f"Error: {str(e)}",
                "is_correct": False
            }
    
    def run_batch(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of examples.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            List of processed examples with results
        """
        results = []
        for i, example in enumerate(examples):
            logger.info(f"Processing example {i+1}/{len(examples)}")
            result = self.process_example(example)
            results.append(result)
        
        return results
    
    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the results and calculate metrics.
        
        Args:
            results: List of processed examples with results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not results:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        
        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def batch_examples(self, examples: List[Dict[str, Any]], batch_size: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Split examples into batches.
        
        Args:
            examples: List of examples
            batch_size: Size of each batch
            
        Returns:
            List of batches
        """
        return [examples[i:i+batch_size] for i in range(0, len(examples), batch_size)]
    
    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        """
        Process a single input and call the OpenAI API.
        
        This method is required by the BaseAgent abstract class.
        
        Args:
            task_step_inputs: The inputs for this task step
            
        Returns:
            The result as a string
        """
        try:
            # Extract context and question from inputs
            context = task_step_inputs.get("context", "")
            question = task_step_inputs.get("question", "")
            
            # Format the prompt
            prompt = self.format_prompt(task_step_inputs)
            
            # Set up client
            client = OpenAI(api_key=self.api_key)
            
            # Call the API
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            model_response = response.choices[0].message.content.strip()
            
            # Extract just the letter
            if model_response and len(model_response) > 0:
                answer_letter = model_response[0].upper()
            else:
                answer_letter = ""
            
            # Map the letter back to yes/no/maybe
            letter_to_answer = {
                "A": "yes",
                "B": "no",
                "C": "maybe"
            }
            predicted_answer = letter_to_answer.get(answer_letter, "")
            
            return predicted_answer
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "error"
    
    def run(self, examples: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
        """
        Run the agent on a list of examples.
        
        Args:
            examples: List of examples to process
            batch_size: Size of batches for processing
            
        Returns:
            Dictionary with results and metrics
        """
        # Check if API key is set
        if not self.api_key:
            return {
                "error": "OPENAI_API_KEY environment variable not set",
                "metrics": {"accuracy": 0.0},
                "results": []
            }
        
        # Set up OpenAI client
        client = OpenAI(api_key=self.api_key)
        
        # Process examples in batches
        batches = self.batch_examples(examples, batch_size)
        all_results = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            batch_results = self.run_batch(batch)
            all_results.extend(batch_results)
        
        # Calculate metrics
        metrics = self.evaluate(all_results)
        
        return {
            "metrics": metrics,
            "results": all_results
        }

    def validate_response(self, response: str) -> str:
        """Validate and extract the answer letter from the response."""
        if not response:
            return ""
        
        # Extract just the letter
        match = re.search(r'[A-D]', response.upper())
        return match.group(0) if match else "" 