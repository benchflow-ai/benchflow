import logging
import os
from typing import Any, Dict

from openai import OpenAI

from benchflow import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.system_instruction = (
            """You are an AI assistant tasked with answering questions based on provided context. 
            You will be given:
            1. A question to answer
            2. Relevant search results/documents as context
            
            Your task is to:
            1. Read and understand the question
            2. Analyze the provided context
            3. Generate a concise, accurate answer based ONLY on the provided context
            4. If you cannot find the answer in the context, respond with "I don't know"
            
            Rules:
            - Only use information from the provided context
            - Be concise and direct in your answers
            - Do not make assumptions or add information not present in the context
            - If the answer isn't in the context, say "I don't know"
            """
        )

    def _construct_message(self, env_info: Dict[str, Any]) -> str:
        query = env_info.get('query', '')
        logger.info(f"[CRAGAgent]: Query: {query}")
        search_results = env_info.get('search_results', [])
        logger.info(f"[CRAGAgent]: Search results: {search_results}")
        
        # search_results is now a list of strings (snippets)
        context = "\n\n".join(
            f"Document {i+1}:\n{snippet}"
            for i, snippet in enumerate(search_results)
        )
        
        return (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def call_api(self, task_step_inputs: Dict[str, Any]) -> str:
        try:
            logger.info("[CRAGAgent]: Calling OpenAI API")
            client = OpenAI(api_key=self.api_key)
            
            messages = [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": self._construct_message(task_step_inputs)}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"[CRAGAgent]: Generated answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"[CRAGAgent]: Error calling OpenAI API: {e}")
            raise

def main():
    logger.info("Starting CRAGAgent...")
    agent = CRAGAgent()
    logger.info("Running agent on http://0.0.0.0:9000")
    agent.run_with_endpoint(host="0.0.0.0", port=9000)

if __name__ == "__main__":
    main()