#!/bin/bash
# assumes it is run in container where the following env vars are set
# WORKSPACE_BASE, CODE_DIR, SUBMISSION_DIR, LOGS_DIR, AGENT_DIR

# Print commands and their arguments as they are executed
set -x

# Test Docker-in-Docker functionality
if [ -x "$(command -v docker)" ]; then
  docker --version
  # Skip to avoid get rate limited on docker pulling images
  # # Actually try to run a container
  # docker run --rm hello-world
  # # Show all containers that ran
  # echo "Listing all containers that ran, should include hello-world:"
  # docker ps -a
fi 2>&1 | tee $LOGS_DIR/docker.log

{
  conda run -n agent --no-capture-output python start.py

  # Move agent logs to $LOGS_DIR/ directory
  if [ -d "$AGENT_DIR/logs" ]; then
    mv "$AGENT_DIR/logs"/* "$LOGS_DIR"/ 2> /dev/null || true
  fi

  # for debugging
  ls $WORKSPACE_BASE
  ls $CODE_DIR
  ls $SUBMISSION_DIR
  ls $LOGS_DIR
  ls $AGENT_DIR
} 2>&1 | tee $LOGS_DIR/agent.log
