#!/bin/bash
test -f /app/config.status && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
