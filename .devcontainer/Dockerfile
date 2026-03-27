FROM node:22

ARG TZ
ENV TZ="$TZ"
ARG CLAUDE_CODE_VERSION=latest

# System tools + playwright deps (for browser plugin)
RUN apt-get update && apt-get install -y --no-install-recommends \
  less git procps sudo fzf zsh man-db unzip gnupg2 gh jq nano vim \
  python3 python3-pip python3-venv \
  # Playwright/Chromium deps
  libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
  libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
  libcairo2 libasound2 libxshmfence1 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG USERNAME=node

# Persist bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

ENV DEVCONTAINER=true
ENV DEBIAN_FRONTEND=noninteractive

# Workspace and config dirs
RUN mkdir -p /workspace /home/node/.claude && \
  chown -R node:node /workspace /home/node/.claude

WORKDIR /workspace

# Docker CLI (for Docker-in-Docker benchmarks)
RUN install -m 0755 -d /etc/apt/keyrings && \
  curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
  chmod a+r /etc/apt/keyrings/docker.asc && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list && \
  apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin && \
  apt-get clean && rm -rf /var/lib/apt/lists/* && \
  groupadd -g 999 docker && usermod -aG docker node

# uv for Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Ensure npm-global dir exists and is writable by node user
RUN mkdir -p /usr/local/share/npm-global && \
  chown -R node:node /usr/local/share/npm-global

USER node

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin:/home/node/.local/bin
ENV SHELL=/bin/zsh
ENV EDITOR=nano

# zsh
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git -p fzf \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Bun (needed for discord plugin and other MCP servers)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH=$PATH:/home/node/.bun/bin

# Google Cloud SDK (for gcloud auth, accessing Google Docs/Drive)
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/home/node
ENV PATH=$PATH:/home/node/google-cloud-sdk/bin

# Google Workspace CLI (for accessing Google Docs, Sheets, Drive)
RUN ARCH=$(uname -m) && \
  if [ "$ARCH" = "aarch64" ]; then GWS_ARCH="aarch64-unknown-linux-gnu"; \
  else GWS_ARCH="x86_64-unknown-linux-gnu"; fi && \
  curl -sSL "https://github.com/googleworkspace/cli/releases/latest/download/google-workspace-cli-${GWS_ARCH}.tar.gz" | \
  tar xz -C /tmp && \
  mv /tmp/google-workspace-cli-*/gws /home/node/.local/bin/gws-cli && \
  chmod +x /home/node/.local/bin/gws-cli && \
  rm -rf /tmp/google-workspace-cli-*

# Claude Code + Gemini CLI
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} @google/gemini-cli

USER root
RUN echo "node ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/node-all && \
  chmod 0440 /etc/sudoers.d/node-all
USER node
