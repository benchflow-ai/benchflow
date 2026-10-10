# Synced from eval/verifier/shared/Dockerfile by eval/scripts/sync_tests.py. Do not edit here.
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_TOOL_BIN_DIR=/usr/local/bin
RUN uv tool install harbor-rewardkit==0.2.1

COPY workspace/ /app
WORKDIR /app
# Commit the scaffold so the verifier can read the agent's changes with git.
RUN git init -q \
    && git add -A \
    && git -c user.name=eval -c user.email=eval@example.com commit -qm scaffold
