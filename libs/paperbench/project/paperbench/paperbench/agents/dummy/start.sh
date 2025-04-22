#!/bin/bash
# assumes it is run in container where the following env vars are set
# WORKSPACE_BASE, CODE_DIR, SUBMISSION_DIR, LOGS_DIR, AGENT_DIR

set -x # Print commands and their arguments as they are executed

# Test Docker-in-Docker functionality
{
    docker --version
    # Actually try to run a container
    docker run --rm hello-world
    # Show all containers that ran (should include hello-world)
    echo "Listing all containers that ran, should include hello-world:"
    docker ps -a

} 2>&1 | tee $LOGS_DIR/docker.log

{
    conda run -n agent python ${AGENT_DIR}/main.py

    touch $LOGS_DIR/run.log
    touch $AGENT_DIR/agent_was_here.txt
    cat $WORKSPACE_BASE/instructions.txt

    cd ${SUBMISSION_DIR}
    touch reproduce.sh
    git add reproduce.sh
    git commit -m "Add empty reproduce.sh script"
} 2>&1 | tee $LOGS_DIR/agent.log