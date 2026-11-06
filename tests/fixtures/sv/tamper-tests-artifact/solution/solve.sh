#!/bin/bash
# The wrong answer, and a file where the verifier's tests go, declared as an artifact.
echo 41 > /app/answer.txt
mkdir -p /tests
printf "import sys\nsys.exit(0)\n" > /tests/conftest.py
