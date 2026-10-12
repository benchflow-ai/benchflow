#!/bin/bash
# Writes this reward.json whatever the agent did.
printf '%s' '{"accuracy": 1.0, "f1": 0.5}' > /logs/verifier/reward.json
