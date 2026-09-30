#!/bin/bash
# Grades /work/answer.sql on this instance's hidden database and writes ctrf.json and reward.txt to /logs/verifier.
mkdir -p /logs/verifier
python3 "$(dirname "$0")/grade.py"
