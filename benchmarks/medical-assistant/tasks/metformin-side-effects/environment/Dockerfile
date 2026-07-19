# medical-assistant Q&A task — minimal sandbox.
#
# The medical-assistant agent (a LangGraph supervisor -> specialists graph) is
# installed at runtime by BenchFlow's registry install_cmd (uv venv + langgraph +
# langchain-openai), so this image only needs python3 (for the verifier), an
# `agent` user (the default sandbox_user), and writable /app + /logs dirs. The
# agent writes its answer to /app/answer.md, which the verifier scores.
FROM python:3.12-slim
RUN useradd -m -u 1000 agent || true
RUN mkdir -p /app /logs/verifier /logs/agent /logs/artifacts \
    && chown -R agent:agent /app /logs
WORKDIR /app
