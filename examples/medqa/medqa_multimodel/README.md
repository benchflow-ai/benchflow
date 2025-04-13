# MedQA Multi-Model Benchmark

This directory contains implementations for testing multiple LLM models with the MedQA benchmark.

## Supported Models

- **Gemini 2.5** (Google)
- **Claude 3.7** (Anthropic)
- **GPT-4o** (OpenAI)
- **Llama 4 Maverick** (Meta via OpenRouter/Firework)

## Setup

1. Install the required dependencies:

```bash
pip install -r medqa_multimodel_requirements.txt
```

2. **Create a `.env` file in this directory** by copying the provided template:

```bash
cp .env.example .env
```

Then edit the `.env` file to add your actual API keys.

## Running the Benchmark

You can run the benchmark with a specific model or all models:

```bash
# Run with a specific model
python test_medqa_multimodel.py --model gemini --case_id 1

# Run with all models
python test_medqa_multimodel.py --model all --case_id 1

# Run all cases with a specific model
python test_medqa_multimodel.py --model claude --case_id all
```

## Agent Implementations

- `medqa_gemini.py` - Implementation for Google's Gemini 2.5 model
- `medqa_claude.py` - Implementation for Anthropic's Claude 3.7 model
- `medqa_gpt4o.py` - Implementation for OpenAI's GPT-4o model
- `medqa_llama4.py` - Implementation for Meta's Llama 4 Maverick model via OpenRouter

## Notes

- **Important**: The scripts must be run from the `examples/medqa/medqa_multimodel` directory.
- Each model requires its corresponding API key to be set in the `.env` file.
- The OpenRouter implementation for Llama 4 Maverick uses the OpenAI SDK with a custom base URL.
- All models use the same benchmark framework and evaluation criteria.
- Results are returned in the same format for easy comparison.
- The benchmark results will be displayed in the console and saved to the results directory.
