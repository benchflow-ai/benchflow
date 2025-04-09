from benchflow import load_benchmark
import os
import argparse

# Load environment variables from .env file
from examples.medqa_multimodel.load_env import load_env_vars
load_env_vars()

# Import all the agent implementations
from examples.medqa_multimodel.medqa_gemini import MedQAGeminiAgent
from examples.medqa_multimodel.medqa_claude import MedQAClaudeAgent
from examples.medqa_multimodel.medqa_gpt4o import MedQAGPT4oAgent
from examples.medqa_multimodel.medqa_llama4 import MedQALlama4Agent

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
    run_ids = bench.run(
        task_ids=["diagnosis"],  # choices: "diagnosis", "treatment", "prevention", "all"
        agents=agent,
        api=api_config,
        requirements_txt="examples/medqa_multimodel/medqa_multimodel_requirements.txt",
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
