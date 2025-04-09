# MedQA Examples

This directory contains examples for running the MedQA benchmark with different LLM models.

## Available Models

- **Gemini 2.5 Pro**: Google's latest LLM
- **Claude 3.7**: Anthropic's latest LLM
- **GPT-4o**: OpenAI's latest LLM
- **Llama 4 Maverick**: Meta's latest LLM (via OpenRouter)

## Files

- `medqa_multimodel/`: Directory containing implementations for testing multiple models
  - `medqa_gemini.py`: Implementation of MedQA agent using Gemini 2.5 Pro
  - `medqa_claude.py`: Implementation of MedQA agent using Claude 3.7
  - `medqa_gpt4o.py`: Implementation of MedQA agent using GPT-4o
  - `medqa_llama4.py`: Implementation of MedQA agent using Llama 4 Maverick
  - `test_model_apis.py`: Script to test the API for each model
  - `test_medqa_multimodel.py`: Script to run the MedQA benchmark with multiple models
  - `test_all_models.py`: Script to test all models on the same prompt
  - Other utility scripts
- `medqa_openai.py`: Original implementation of MedQA agent using OpenAI
- `test_medqa.py`: Original script to run the MedQA benchmark with OpenAI

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
python -m examples.medqa.medqa_multimodel.test_model_apis --model all

# Test a specific model
python -m examples.medqa.medqa_multimodel.test_model_apis --model gemini
python -m examples.medqa.medqa_multimodel.test_model_apis --model claude
python -m examples.medqa.medqa_multimodel.test_model_apis --model gpt4o
python -m examples.medqa.medqa_multimodel.test_model_apis --model llama4
```

### Running the Benchmark

```bash
# Run the benchmark with all models
python -m examples.medqa.medqa_multimodel.test_medqa_multimodel --model all

# Run the benchmark with a specific model
python -m examples.medqa.medqa_multimodel.test_medqa_multimodel --model gemini
python -m examples.medqa.medqa_multimodel.test_medqa_multimodel --model claude
python -m examples.medqa.medqa_multimodel.test_medqa_multimodel --model gpt4o
python -m examples.medqa.medqa_multimodel.test_medqa_multimodel --model llama4

# Run the benchmark with a specific case ID
python -m examples.medqa.medqa_multimodel.test_medqa_multimodel --model all --case_id 1
```
