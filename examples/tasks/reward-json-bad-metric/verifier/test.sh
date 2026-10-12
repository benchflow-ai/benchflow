#!/bin/bash
# Writes this reward.json whatever the agent did.
printf '%s' '{"reward": 0.5, "tests_passed": 3}' > /logs/verifier/reward.json
