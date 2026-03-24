FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Bake benchflow skills into the environment
COPY benchflow-run /root/.claude/skills/benchflow-run
COPY benchflow-create-task /root/.claude/skills/benchflow-create-task
