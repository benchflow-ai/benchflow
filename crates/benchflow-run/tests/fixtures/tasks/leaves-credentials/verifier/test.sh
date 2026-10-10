#!/bin/bash
# Checks /app/hello.txt byte for byte and writes the reward.
if printf 'Hello, world!' | cmp -s - /app/hello.txt; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
