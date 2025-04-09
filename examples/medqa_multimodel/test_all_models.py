import os
import sys
import time
from examples.medqa_multimodel.load_env import load_env_vars

# Import the agent implementations
from examples.medqa_multimodel.medqa_gemini import MedQAGeminiAgent
from examples.medqa_multimodel.medqa_claude import MedQAClaudeAgent
from examples.medqa_multimodel.medqa_gpt4o import MedQAGPT4oAgent
from examples.medqa_multimodel.medqa_llama4 import MedQALlama4Agent

def test_all_models():
    """Test all models on the same prompt and compare their responses."""
    # Load environment variables
    load_env_vars()
    
    # Create a test prompt
    test_input = {
        "user_prompt": """
        You are a medical expert. Please provide three possible diagnoses for the following case:
        
        A 46-year-old man presents to the emergency department with a 40-minute history of substernal chest pain that radiates to his left arm, upper back, and neck. The pain woke him from sleep and is described as pressure-like with a severity of 7/10. He reports associated nausea, sweating, and shortness of breath. He has a history of hypertension for 5 years and high cholesterol for 3 years, both poorly controlled. He has a 25 pack-year smoking history but quit 3 months ago. He has a sedentary lifestyle. He has a history of GERD for 10 years. He has had episodes of chest pain over the past 3 months, precipitated by exertion, heavy meals, and sexual intercourse. Previous episodes of chest pain were relieved by antacids. He has a 10-year history of cocaine use and last used cocaine yesterday afternoon.
        
        On physical examination, his blood pressure is 165/85 mm Hg, respiratory rate is 22/minute, heart rate is 90/minute and regular, and oxygen saturation is 98% on room air. He appears to be in severe pain. His lungs are clear to auscultation with symmetric breath sounds bilaterally. His heart has a regular rate and rhythm with normal S1 and S2 and no murmurs, rubs, or gallops. His abdomen is soft, non-tender, and non-distended with normal bowel sounds. He has no peripheral edema or cyanosis. His peripheral pulses are 2+ and symmetric.
        
        For each diagnosis, provide:
        1. The name of the diagnosis
        2. Three historical findings that support this diagnosis
        3. Three physical exam findings that support this diagnosis (if applicable)
        
        Format your response as follows:
        
        Diagnosis #1: [Diagnosis Name]
        Historical Finding(s): 
        [Historical finding 1]
        [Historical finding 2]
        [Historical finding 3]
        
        Physical Exam Finding(s):
        [Physical exam finding 1]
        [Physical exam finding 2]
        [Physical exam finding 3]
        
        [Repeat for Diagnoses #2 and #3]
        """
    }
    
    # Test each model
    models = [
        {"name": "Gemini 2.5 Pro", "agent": MedQAGeminiAgent(), "key": "GEMINI_API_KEY"},
        {"name": "Claude 3.7", "agent": MedQAClaudeAgent(), "key": "ANTHROPIC_API_KEY"},
        {"name": "GPT-4o", "agent": MedQAGPT4oAgent(), "key": "OPENAI_API_KEY"},
        {"name": "Llama 4 Maverick", "agent": MedQALlama4Agent(), "key": "OPENROUTER_API_KEY"}
    ]
    
    results = {}
    
    for model in models:
        print(f"\n=== Testing {model['name']} ===")
        
        # Check if API key is set
        if not os.getenv(model["key"]):
            print(f"Error: {model['key']} environment variable is not set")
            continue
        
        try:
            # Make the API call
            print(f"Sending test prompt to {model['name']}...")
            start_time = time.time()
            response = model["agent"].call_api(test_input)
            end_time = time.time()
            
            # Print the response
            print(f"\nResponse from {model['name']} (took {end_time - start_time:.2f} seconds):")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
            # Save the result
            results[model["name"]] = {
                "response": response,
                "time": end_time - start_time
            }
            
        except Exception as e:
            print(f"Error testing {model['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print a summary
    print("\n=== Summary ===")
    for model_name, result in results.items():
        print(f"{model_name}: Response time: {result['time']:.2f} seconds")
    
    return results

if __name__ == "__main__":
    test_all_models()
