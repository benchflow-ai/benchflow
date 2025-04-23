"""
SimpleQA agent implementation using OpenAI API.
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from benchflow import BaseAgent

class SimpleQAAgent(BaseAgent):
    """
    Agent for the SimpleQA benchmark.
    
    This agent evaluates a model's ability to answer short, fact-seeking questions
    and to abstain from answering when uncertain.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.5,
        max_tokens: int = 2048,
        batch_size: int = 10,
        grader_model: str = "gpt-4o",
        grader_temperature: float = 0.5,
    ):
        """
        Initialize the SimpleQA agent.
        
        Args:
            model_name: Model to use for answering questions
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            batch_size: Number of examples to process in a batch
            grader_model: Model to use for grading answers
            grader_temperature: Temperature for grading
        """
        super().__init__()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.grader_model = grader_model
        self.grader_temperature = grader_temperature
        self.client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger("simpleqa")
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the SimpleQA task.
        
        Returns:
            System prompt string
        """
        return (
            "You are a helpful assistant that answers short, fact-seeking questions. "
            "If you know the answer with high confidence, provide it. "
            "If you are uncertain, it's better to say you don't know rather than guess."
        )
    
    def format_prompt(self, question: str) -> str:
        """
        Format the prompt for the SimpleQA task.
        
        Args:
            question: The question to answer
            
        Returns:
            Formatted prompt string
        """
        return f"Question: {question}\n\nAnswer:"
    
    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        """
        Call the OpenAI API to get a response for the question.
        
        Args:
            task_step_inputs: The inputs for this task step
            
        Returns:
            Model's response
        """
        try:
            # Extract question from inputs
            question = task_step_inputs.get("question", "")
            if not question:
                self.logger.error("No question provided in task_step_inputs")
                return "Error: No question provided"
            
            prompt = self.format_prompt(question)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            return f"Error: {str(e)}"
    
    def grade_response(self, question: str, correct_answer: str, predicted_answer: str) -> str:
        """
        Grade the model's response using the grader model.
        
        Args:
            question: The question that was asked
            correct_answer: The correct answer
            predicted_answer: The model's predicted answer
            
        Returns:
            Grade: "A" for correct, "B" for incorrect, "C" for not attempted
        """
        try:
            grader_prompt = self.get_grader_prompt(question, correct_answer, predicted_answer)
            
            response = self.client.chat.completions.create(
                model=self.grader_model,
                messages=[
                    {"role": "system", "content": "You are a helpful grading assistant."},
                    {"role": "user", "content": grader_prompt}
                ],
                temperature=self.grader_temperature,
                max_tokens=10
            )
            
            grade = response.choices[0].message.content.strip()
            
            # Extract just the letter grade
            match = re.search(r'[ABC]', grade.upper())
            if match:
                return match.group(0)
            else:
                self.logger.warning(f"Unexpected grade format: {grade}")
                return "C"  # Default to not attempted
            
        except Exception as e:
            self.logger.error(f"Grading failed: {e}")
            return "C"  # Default to not attempted
    
    def get_grader_prompt(self, question: str, target: str, predicted_answer: str) -> str:
        """
        Get the prompt for the grader model.
        
        Args:
            question: The question that was asked
            target: The correct answer
            predicted_answer: The model's predicted answer
            
        Returns:
            Grader prompt string
        """
        return f"""
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.

```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```

These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.

```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```

These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.

```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```

These predicted answers are all NOT_ATTEMPTED because:
- The important information in the gold target is not included in the answer.
- No statements in the answer contradict the gold target.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
"""
    
    def evaluate(self, examples: List[Dict[str, Any]], limit: int = 0) -> Dict[str, Any]:
        """
        Evaluate the model on the SimpleQA benchmark.
        
        Args:
            examples: List of examples to evaluate
            limit: Maximum number of examples to evaluate (0 for all)
            
        Returns:
            Dictionary with evaluation results
        """
        if limit > 0:
            examples = examples[:limit]
        
        self.logger.info(f"Evaluating {len(examples)} examples with {self.model_name}")
        
        # Process examples in batches
        all_results = []
        for i in range(0, len(examples), self.batch_size):
            batch = examples[i:i+self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(examples)-1)//self.batch_size + 1}")
            
            # Process the batch
            batch_results = []
            for example in batch:
                question = example["question"]
                correct_answer = example["answer"]
                
                try:
                    # Get model response
                    response = self.call_api({"question": question})
                    
                    # Grade the response
                    grade = self.grade_response(question, correct_answer, response)
                    
                    # Determine if the answer is correct, incorrect, or not attempted
                    is_correct = grade == "A"
                    is_incorrect = grade == "B"
                    not_attempted = grade == "C"
                    
                    batch_results.append({
                        "question": question,
                        "correct_answer": correct_answer,
                        "response": response,
                        "is_correct": is_correct,
                        "is_incorrect": is_incorrect,
                        "not_attempted": not_attempted,
                        "grade": grade
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing example: {e}")
                    batch_results.append({
                        "question": question,
                        "correct_answer": correct_answer,
                        "response": "",
                        "is_correct": False,
                        "is_incorrect": False,
                        "not_attempted": True,
                        "grade": "C",
                        "error": str(e)
                    })
            
            all_results.extend(batch_results)
        
        # Calculate metrics
        total = len(all_results)
        correct = sum(1 for r in all_results if r["is_correct"])
        incorrect = sum(1 for r in all_results if r["is_incorrect"])
        not_attempted = sum(1 for r in all_results if r["not_attempted"])
        
        # Calculate derived metrics
        accuracy = correct / total if total > 0 else 0
        total_attempted = correct + incorrect
        correct_given_attempted = correct / total_attempted if total_attempted > 0 else 0
        
        # Calculate F-score (harmonic mean of correct and correct_given_attempted)
        f_score_denom = accuracy + correct_given_attempted
        f_score = (2 * accuracy * correct_given_attempted) / f_score_denom if f_score_denom > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "incorrect": incorrect,
            "not_attempted": not_attempted,
            "total": total,
            "correct_given_attempted": correct_given_attempted,
            "f_score": f_score
        }
        
        return {
            "metrics": metrics,
            "results": all_results
        }
