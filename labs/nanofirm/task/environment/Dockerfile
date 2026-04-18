FROM python:3.12-slim

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app/docs

# Python deps for the personas MCP sidecar. fastmcp for stdio MCP;
# anthropic/openai/google-genai so each persona can route to any provider.
RUN pip install --no-cache-dir --quiet \
      "fastmcp>=2.0" \
      "anthropic>=0.40" \
      "openai>=1.50" \
      "google-genai>=0.3"

# Task content: personas server, case facts, MCP config, reference docs.
COPY personas_server.py /app/personas_server.py
COPY personas_config.py /app/personas_config.py
COPY case_facts.md /app/case_facts.md
COPY mcp_config.json /app/.mcp.json
COPY docs/ /app/docs/

# Pre-create empty transcript and resolution so tools can append without race.
RUN : > /app/transcript.jsonl \
  && echo '{}' > /app/resolution.json
