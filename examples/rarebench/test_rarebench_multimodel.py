import os
import argparse
from pathlib import Path
from benchflow import load_benchmark

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
    req_path = Path('rarebench_requirements.txt')
    if not req_path.exists():
        print("\nWARNING: No rarebench_requirements.txt file found in the current directory.")
        print("Please make sure the requirements file exists in the rarebench directory.\n")
    return req_path

# Import the agent implementations directly
# Assuming this script is run from the rarebench folder
from rarebench_gemini import RareBenchGeminiAgent
from rarebench_claude import RareBenchClaudeAgent
from rarebench_gpt4o import RareBenchGPT4oAgent
from rarebench_llama4 import RareBenchLlama4Agent

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

def test_rarebench_model(model_name, task_id="MME"):
    """
    Run the RareBench benchmark with the specified model.

    Args:
        model_name: The name of the model to test (gemini, claude, gpt4o, llama4)
        task_id: The task ID to test
    """
    # Load environment variables
    load_env_vars()

    # Check if BF_TOKEN is set
    bf_token = os.getenv("BF_TOKEN")
    if not bf_token:
        print("Error: BF_TOKEN environment variable is not set")
        return False

    # Load the benchmark
    bench = load_benchmark(benchmark_name="benchflow/Rarebench", bf_token=bf_token)

    # Create the agent based on the model name
    if model_name == "gemini":
        agent = RareBenchGeminiAgent()
        api_config = {
            "provider": "google",
            "model": "gemini-2.5-pro-preview-03-25",
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")
        }
    elif model_name == "claude":
        agent = RareBenchClaudeAgent()
        api_config = {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219",
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
        }
    elif model_name == "gpt4o":
        agent = RareBenchGPT4oAgent()
        api_config = {
            "provider": "openai",
            "model": "gpt-4o",
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        }
    elif model_name == "llama4":
        agent = RareBenchLlama4Agent()
        api_config = {
            "provider": "openrouter",
            "model": "meta-llama/llama-4-maverick",
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY")
        }
    else:
        print(f"Error: Unknown model name: {model_name}")
        return False

    # Run the benchmark
    print(f"Running RareBench benchmark with {model_name} for task_id: {task_id}")

    # Get the path to the requirements file
    requirements_path = check_requirements_file()

    run_ids = bench.run(
        task_ids=[task_id],
        agents=agent,
        api=api_config,
        requirements_txt=str(requirements_path),
        args={
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),  # required for evaluation
        },
    )

    # Get and print the results
    results = bench.get_results(run_ids)
    print(f"Results for {model_name}:")
    print(results)

    return results

def test_all_models(task_id="MME"):
    """
    Run the RareBench benchmark with all models.

    Args:
        task_id: The task ID to test
    """
    models = ["gemini", "claude", "gpt4o", "llama4"]
    results = {}

    for model in models:
        print(f"\n=== Testing {model} ===")
        model_results = test_rarebench_model(model, task_id)
        results[model] = model_results

    # Print a summary
    print("\n=== Summary ===")
    for model, result in results.items():
        if result and isinstance(result, dict):
            for run_id, tasks in result.items():
                for task in tasks:
                    if task.get("metrics") and "score" in task.get("metrics", {}):
                        score = task["metrics"]["score"]
                        print(f"{model} (run {run_id}): Score: {score}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RareBench benchmark with multiple models")
    parser.add_argument("--model", type=str, choices=["gemini", "claude", "gpt4o", "llama4", "all"],
                        default="all", help="Model to test")
    parser.add_argument("--task_id", type=str, default="MME",
                        help="Task ID to test")

    args = parser.parse_args()

    if args.model == "all":
        test_all_models(args.task_id)
    else:
        test_rarebench_model(args.model, args.task_id)
