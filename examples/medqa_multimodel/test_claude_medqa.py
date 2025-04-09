import os
import sys
from benchflow import load_benchmark

# Add the current directory to the path so we can import the agent modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the agent implementation
from examples.medqa_multimodel.medqa_claude import MedQAClaudeAgent
from examples.medqa_multimodel.load_env import load_env_vars

def test_claude_medqa(case_id="1"):
    """
    Run the MedQA benchmark with Claude 3.7.
    
    Args:
        case_id: The case ID to test, or "all" for all cases
    """
    # Load environment variables
    load_env_vars()
    
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        return False
    
    # Load the benchmark
    bf_token = os.getenv("BF_TOKEN")
    if not bf_token:
        print("Error: BF_TOKEN environment variable is not set")
        return False
    
    bench = load_benchmark(benchmark_name="benchflow/medqa-cs", bf_token=bf_token)
    
    # Create the agent
    agent = MedQAClaudeAgent()
    api_config = {
        "provider": "anthropic", 
        "model": "claude-3-7-sonnet-20250219", 
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
    }
    
    # Run the benchmark
    print(f"Running MedQA benchmark with Claude 3.7 for case_id: {case_id}")
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
    print(f"Results for Claude 3.7:")
    print(results)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MedQA benchmark with Claude 3.7")
    parser.add_argument("--case_id", type=str, default="1", 
                        help="Case ID to test, or 'all' for all cases")
    
    args = parser.parse_args()
    
    test_claude_medqa(args.case_id)
