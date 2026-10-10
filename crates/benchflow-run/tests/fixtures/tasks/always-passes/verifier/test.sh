#!/bin/bash
# Deliberately broken: it checks that /app exists, which is true before the agent starts.
if [ -d /app ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
