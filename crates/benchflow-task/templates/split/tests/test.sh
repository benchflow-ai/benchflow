#!/bin/bash
# Checks /app/hello.txt and writes the reward: 1 when the file holds exactly
# the line, 0 otherwise.
if printf 'Hello, world!\n' | cmp -s - /app/hello.txt; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
