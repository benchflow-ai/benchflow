import os
import sys

# Add the current directory to the path so we can import the agent modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from examples.medqa_multimodel.load_env import load_env_vars
load_env_vars()

# Import the agent implementations
from examples.medqa_multimodel.medqa_gemini import MedQAGeminiAgent
from examples.medqa_multimodel.medqa_claude import MedQAClaudeAgent
from examples.medqa_multimodel.medqa_gpt4o import MedQAGPT4oAgent
from examples.medqa_multimodel.medqa_llama4 import MedQALlama4Agent

def test_api_call(model_name):
    """Test a direct API call to the specified model."""
    print(f"\n=== Testing {model_name.upper()} API ===")

    # Simple test prompt
    test_input = {
        "user_prompt": "What are the symptoms of pneumonia? Keep your answer brief."
    }

    try:
        # Create the appropriate agent based on the model name
        if model_name == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                print("Error: GEMINI_API_KEY environment variable is not set")
                return False
            agent = MedQAGeminiAgent()
        elif model_name == "claude":
            if not os.getenv("ANTHROPIC_API_KEY"):
                print("Error: ANTHROPIC_API_KEY environment variable is not set")
                return False
            agent = MedQAClaudeAgent()
        elif model_name == "gpt4o":
            if not os.getenv("OPENAI_API_KEY"):
                print("Error: OPENAI_API_KEY environment variable is not set")
                return False
            agent = MedQAGPT4oAgent()
        elif model_name == "llama4":
            if not os.getenv("OPENROUTER_API_KEY"):
                print("Error: OPENROUTER_API_KEY environment variable is not set")
                return False
            agent = MedQALlama4Agent()
        else:
            print(f"Error: Unknown model name: {model_name}")
            return False

        # Make the API call
        print(f"Sending test prompt to {model_name}...")
        response = agent.call_api(test_input)

        # Print the response
        print(f"\nResponse from {model_name}:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        return True

    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test API calls to different LLM models")
    parser.add_argument("--model", type=str, choices=["gemini", "claude", "gpt4o", "llama4", "all"],
                        default="all", help="Model to test")

    args = parser.parse_args()

    if args.model == "all":
        # Test all models
        models = ["gemini", "claude", "gpt4o", "llama4"]
        results = {}

        for model in models:
            results[model] = test_api_call(model)

        # Print summary
        print("\n=== Test Summary ===")
        for model, success in results.items():
            print(f"{model}: {'SUCCESS' if success else 'FAILED'}")
    else:
        # Test just the specified model
        test_api_call(args.model)
