FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git jq ripgrep xz-utils \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
  && ln -sf /root/.local/bin/uvx /usr/local/bin/uvx

ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash \
  && source "$NVM_DIR/nvm.sh" \
  && nvm install 22 \
  && nvm alias default 22 \
  && npm install -g @openai/codex@0.130.0 \
  && ln -sf "$(command -v node)" /usr/local/bin/node \
  && ln -sf "$(command -v npm)" /usr/local/bin/npm \
  && ln -sf "$(command -v npx)" /usr/local/bin/npx \
  && ln -sf "$(command -v codex)" /usr/local/bin/codex

# Harbor injects agent auth into this non-mounted path during the agent phase.
# RewardKit verifier auth is provided by a local-private read-only runtime
# source mount in docker-compose.yaml, then copied to a temporary writable
# verifier CODEX_HOME by tests/test.sh. Do not COPY auth files into this image.
ENV CODEX_HOME=/root/.codex-agent

WORKDIR /app
COPY context/ /app/
