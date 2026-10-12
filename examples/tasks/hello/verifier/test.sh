#!/bin/bash
# Reward 1 when /app/hello.txt holds exactly "Hello, world!".
if printf 'Hello, world!' | cmp -s - /app/hello.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
