FROM node:22

ARG TZ
ENV TZ="$TZ"
ARG CLAUDE_CODE_VERSION=latest

# System tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  less git procps sudo fzf zsh man-db unzip gnupg2 gh jq nano vim \
  python3 python3-pip python3-venv \
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

# Claude Code
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

USER root
RUN echo "node ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/node-all && \
  chmod 0440 /etc/sudoers.d/node-all
USER node
