FROM python:3.12-slim

# System deps
RUN apt-get update -qq && apt-get install -y -qq curl git && rm -rf /var/lib/apt/lists/*

# Judge dependencies (baked in, not installed at test time)
RUN pip install -q anthropic openai google-genai

# Log directories
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app /verifier /tests

# BenchFlow links this neutral task-local skill tree into the selected agent's skill paths
COPY skills/ /skills/

WORKDIR /app
