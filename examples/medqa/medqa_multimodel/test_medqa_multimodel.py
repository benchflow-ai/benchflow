from benchflow import load_benchmark
import os
import argparse
from pathlib import Path

# Check for .env file in the current directory
def check_env_file():
    env_path = Path('.env')
    if not env_path.exists():
        print("\nWARNING: No .env file found in the current directory.")
        print("Please copy the .env.example file to .env and fill in your API keys:")
        print("cp .env.example .env")
        print("Then edit the .env file with your actual API keys.\n")
    return env_path

# Check for requirements file in the current directory
def check_requirements_file():
    req_path = Path('medqa_multimodel_requirements.txt')
    if not req_path.exists():
        print("\nWARNING: No medqa_multimodel_requirements.txt file found in the current directory.")
        print("Please make sure the requirements file exists in the medqa_multimodel directory.\n")
    return req_path

# Create a function to load environment variables
def load_env_vars():
    """Load environment variables from .env file if it exists."""
    env_path = check_env_file()
    if env_path.exists():
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

# Load environment variables
load_env_vars()

# Import the agent implementations directly
# Assuming this script is run from the medqa_multimodel folder
from medqa_gemini import MedQAGeminiAgent
from medqa_claude import MedQAClaudeAgent
from medqa_gpt4o import MedQAGPT4oAgent
from medqa_llama4 import MedQALlama4Agent

def test_medqa_with_model(model_name, case_id="1"):
    """
    Run the MedQA benchmark with the specified model.

    Args:
        model_name: One of "gemini", "claude", "gpt4o", or "llama4"
        case_id: The case ID to test, or "all" for all cases
    """
    # Check if BF_TOKEN is set
    bf_token = os.getenv("BF_TOKEN")
    if not bf_token:
        raise ValueError("BF_TOKEN environment variable is not set")

    bench = load_benchmark(benchmark_name="benchflow/medqa-cs", bf_token=bf_token)

    # Create the appropriate agent based on the model name
    if model_name == "gemini":
        # Check if API key is set
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        agent = MedQAGeminiAgent()
        api_config = {
            "provider": "google",
            "model": "gemini-2.5-pro-preview-03-25",
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")
        }
    elif model_name == "claude":
        # Check if API key is set
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        agent = MedQAClaudeAgent()
        api_config = {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219",
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
        }
    elif model_name == "gpt4o":
        # Check if API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        agent = MedQAGPT4oAgent()
        api_config = {
            "provider": "openai",
            "model": "gpt-4o",
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        }
    elif model_name == "llama4":
        # Check if API key is set
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        agent = MedQALlama4Agent()
        api_config = {
            "provider": "openrouter",
            "model": "meta-llama/llama-4-maverick",
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY")
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Run the benchmark

    # Get the path to the requirements file
    requirements_path = check_requirements_file()

    run_ids = bench.run(
        task_ids=["diagnosis"],  # choices: "diagnosis", "treatment", "prevention", "all"
        agents=agent,
        api=api_config,
        requirements_txt=str(requirements_path),
        args={
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),  # required for llm as an evaluator
            "CASE_ID": case_id,  # use "all" to run all cases in diagnosis section
        },
    )

    # Get and print the results
    results = bench.get_results(run_ids)
    print(f"Results for {model_name}:")
    print(results)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedQA benchmark with different models")
    parser.add_argument("--model", type=str, choices=["gemini", "claude", "gpt4o", "llama4", "all"],
                        default="all", help="Model to test")
    parser.add_argument("--case_id", type=str, default="1",
                        help="Case ID to test, or 'all' for all cases")

    args = parser.parse_args()

    if args.model == "all":
        # Test all models
        models = ["gemini", "claude", "gpt4o", "llama4"]
        for model in models:
            try:
                test_medqa_with_model(model, args.case_id)
            except Exception as e:
                print(f"Error testing {model}: {e}")
    else:
        # Test just the specified model
        test_medqa_with_model(args.model, args.case_id)
