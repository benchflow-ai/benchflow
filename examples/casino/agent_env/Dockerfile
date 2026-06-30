# Agent-seat image for the multi-agent casino floor.
#
# Bakes in everything ONE deepagents ACP seat needs so `connect_acp` can run it
# with no per-seat install: the deepagents venv + ACP shim (the autonomous agent),
# and the `casino` seven-tool CLI (the play surface — the agent shells out to
# `casino observe` / `casino act` against $CASINO_URL). The graph/world itself is
# the shared casinobench World service, reached over HTTP.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git && rm -rf /var/lib/apt/lists/*

# uv (pinned interpreter for deepagents, exactly as benchflow's install_cmd does)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# deepagents harness venv (the autonomous ReAct agent) + the OpenAI-compatible
# chat model dep, verified to import.
RUN uv venv --python 3.12 /opt/benchflow/deepagents-venv && \
    uv pip install -q --python /opt/benchflow/deepagents-venv/bin/python \
        deepagents langchain-openai && \
    /opt/benchflow/deepagents-venv/bin/python -c "import deepagents, langchain_openai; print('deepagents ok')"

# the ACP shim benchflow launches on stdio
COPY deepagents-acp-shim /opt/benchflow/bin/deepagents-acp-shim
RUN chmod a+rx /opt/benchflow/bin/deepagents-acp-shim && \
    chmod -R a+rX /opt/benchflow/deepagents-venv

# the casino seven-tool CLI on PATH (the agent's play surface). The CLI only
# makes HTTP calls to $CASINO_URL (click + httpx) — the casinobench engine is a
# server-side dep of the package, not the CLI, so install --no-deps + the two
# runtime imports the CLI actually uses.
COPY casino-pkg /opt/casino-pkg
RUN pip install --no-cache-dir "click>=8.0" "httpx>=0.27.0" && \
    pip install --no-cache-dir --no-deps /opt/casino-pkg && \
    casino --help >/dev/null && echo "casino cli ok"

# the node ACP agents (codex-acp + claude-agent-acp) baked in via benchflow's own
# install commands — Node 22.20.0 + the npm packages + wrappers at
# /opt/benchflow/bin/{codex-acp,claude-agent-acp}. Subscription auth is uploaded
# per-seat at run time (not baked). connect_acp's force_build=False then needs no
# per-seat install.
RUN apt-get update && apt-get install -y --no-install-recommends tar xz-utils && \
    rm -rf /var/lib/apt/lists/*
COPY install-node-agents.sh /opt/install-node-agents.sh
RUN sh /opt/install-node-agents.sh && \
    test -x /opt/benchflow/bin/codex-acp && test -x /opt/benchflow/bin/claude-agent-acp && \
    echo "node acp agents ok"

WORKDIR /app
