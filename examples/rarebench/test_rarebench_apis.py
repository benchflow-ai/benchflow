import os
import sys
import argparse
import time

# Import the agent implementations
from examples.rarebench.rarebench_gemini import RareBenchGeminiAgent
from examples.rarebench.rarebench_claude import RareBenchClaudeAgent
from examples.rarebench.rarebench_gpt4o import RareBenchGPT4oAgent
from examples.rarebench.rarebench_llama4 import RareBenchLlama4Agent

def load_env_vars():
    """Load environment variables from .env file if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value

    # Check if required API keys are available
    print("Available API keys:")
    for key in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "BF_TOKEN"]:
        if os.getenv(key):
            print(f"  {key}: ✓ Available")
        else:
            print(f"  {key}: ✗ Not available")

def test_model_api(model_name):
    """Test the API for a specific model."""
    # Load environment variables
    load_env_vars()

    # Create a test input
    test_input = {
        "system_prompt": "You are a medical expert specializing in rare diseases.",
        "prompt": "What are the symptoms and diagnostic criteria for Gaucher disease?"
    }

    # Create the agent based on the model name
    if model_name == "gemini":
        agent = RareBenchGeminiAgent()
    elif model_name == "claude":
        agent = RareBenchClaudeAgent()
    elif model_name == "gpt4o":
        agent = RareBenchGPT4oAgent()
    elif model_name == "llama4":
        agent = RareBenchLlama4Agent()
    else:
        print(f"Error: Unknown model name: {model_name}")
        return False

    # Test the API
    print(f"\n=== Testing {model_name.upper()} API ===")
    print(f"Sending test prompt to {model_name}...")

    try:
        start_time = time.time()
        response = agent.call_api(test_input)
        end_time = time.time()

        print(f"\nResponse from {model_name} (took {end_time - start_time:.2f} seconds):")
        print("-" * 40)
        print(response)
        print("-" * 40)

        return True
    except Exception as e:
        print(f"Error testing {model_name} API: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_all_apis():
    """Test the APIs for all models."""
    models = ["gemini", "claude", "gpt4o", "llama4"]
    results = {}

    for model in models:
        success = test_model_api(model)
        results[model] = success

    # Print a summary
    print("\n=== Summary ===")
    for model, success in results.items():
        status = "✓ Working" if success else "✗ Failed"
        print(f"{model}: {status}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RareBench model APIs")
    parser.add_argument("--model", type=str, choices=["gemini", "claude", "gpt4o", "llama4", "all"],
                        default="all", help="Model to test")

    args = parser.parse_args()

    if args.model == "all":
        test_all_apis()
    else:
        test_model_api(args.model)
