#!/bin/bash
# A scripted attempt: `bench run tasks/hello -a script:agents/write-hello.sh`.
# It runs in the sandbox as the agent user (the oracle runs as root), in the task's workdir.
printf 'Hello, world!' > /app/hello.txt
