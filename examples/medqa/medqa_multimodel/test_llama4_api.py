import os
from openai import OpenAI
from examples.medqa_multimodel.load_env import load_env_vars

def test_llama4_api():
    """Test the Llama 4 Maverick API via OpenRouter directly."""
    # Load environment variables
    load_env_vars()
    
    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        return False
    
    # Initialize the client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://benchflow.ai",  # Optional for rankings
            "X-Title": "BenchFlow MedQA Test"        # Optional for rankings
        }
    )
    model = "meta-llama/llama-4-maverick"
    
    # Create a test prompt
    test_prompt = """
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
    
    # Make the API call
    print(f"Calling OpenRouter API with model: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": test_prompt}
            ]
        )
        
        # Print the response
        print("\nResponse from Llama 4 Maverick:")
        print("-" * 40)
        print(response.choices[0].message.content)
        print("-" * 40)
        
        return True
    except Exception as e:
        print(f"Error calling OpenRouter API: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_llama4_api()
