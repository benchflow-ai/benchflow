#!/bin/bash

docker build --platform=linux/amd64 -t pb-env -f paperbench/agents/Dockerfile.base paperbench/agents/
docker build --platform=linux/amd64 -t dummy paperbench/agents/dummy/
docker build --platform=linux/amd64 -t aisi-basic-agent paperbench/agents/aisi-basic-agent/
docker build --platform=linux/amd64 -f paperbench/grader.Dockerfile -t pb-grader .
docker build --platform=linux/amd64 -f paperbench/reproducer.Dockerfile -t pb-reproducer .
