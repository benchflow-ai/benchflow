# RareBench Examples

This directory contains examples for running the RareBench benchmark with different LLM models.

## Available Models

- **Gemini 2.5 Pro**: Google's latest LLM
- **Claude 3.7**: Anthropic's latest LLM
- **GPT-4o**: OpenAI's latest LLM
- **Llama 4 Maverick**: Meta's latest LLM (via OpenRouter)

## Files

- `rarebench_gemini.py`: Implementation of RareBench agent using Gemini 2.5 Pro
- `rarebench_claude.py`: Implementation of RareBench agent using Claude 3.7
- `rarebench_gpt4o.py`: Implementation of RareBench agent using GPT-4o
- `rarebench_llama4.py`: Implementation of RareBench agent using Llama 4 Maverick
- `test_rarebench_apis.py`: Script to test the API for each model
- `test_rarebench_multimodel.py`: Script to run the RareBench benchmark with multiple models
- `test_rarebench.py`: Original script to run the RareBench benchmark with OpenAI

## Setup

1. **Create a `.env` file in this directory** by copying the provided template:

```bash
cp .env.example .env
```

Then edit the `.env` file to add your actual API keys.

2. Make sure the `rarebench_requirements.txt` file is in this directory.

## Requirements

- API keys for each model:
  - `GEMINI_API_KEY`: For Gemini 2.5 Pro
  - `ANTHROPIC_API_KEY`: For Claude 3.7
  - `OPENAI_API_KEY`: For GPT-4o
  - `OPENROUTER_API_KEY`: For Llama 4 Maverick
  - `BF_TOKEN`: For BenchFlow

## Usage

### Testing the APIs

```bash
# Test all models
python test_rarebench_apis.py --model all

# Test a specific model
python test_rarebench_apis.py --model gemini
python test_rarebench_apis.py --model claude
python test_rarebench_apis.py --model gpt4o
python test_rarebench_apis.py --model llama4
```

### Running the Benchmark

```bash
# Run the benchmark with all models
python test_rarebench_multimodel.py --model all

# Run the benchmark with a specific model
python test_rarebench_multimodel.py --model gemini
python test_rarebench_multimodel.py --model claude
python test_rarebench_multimodel.py --model gpt4o
python test_rarebench_multimodel.py --model llama4

# Run the benchmark with a specific task ID
python test_rarebench_multimodel.py --model all --task_id MME
```

## Notes

- **Important**: The scripts must be run from the `examples/rarebench` directory.
- Each model requires its corresponding API key to be set in the `.env` file.
- The benchmark results will be displayed in the console and saved to the results directory.
