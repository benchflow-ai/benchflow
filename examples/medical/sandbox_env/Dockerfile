# Minimal image for running the LangGraph medical multi-agent workflow inside a
# BenchFlow sandbox (docker or daytona). Deps only — the graph code is uploaded
# at run time so iterating on it needs no rebuild.
FROM python:3.12-slim
RUN pip install --no-cache-dir "langgraph>=0.2" "langchain-openai>=0.2" && \
    python -c "import langgraph, langchain_openai; print('deps ok')"
WORKDIR /app
