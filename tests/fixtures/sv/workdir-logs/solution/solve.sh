#!/bin/bash
# The image's WORKDIR is /logs and the task declares no workdir.
echo 42 > /logs/answer.txt
mkdir -p /logs/agent
echo planted > /logs/agent/x
