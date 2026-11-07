#!/bin/bash
# The verifier writes reward.txt only; anything else in /logs/verifier is not its.
# The listing shows in test-stdout.txt whether anything reached /tests.
ls -la /tests
if [ "$(cat /app/answer.txt 2>/dev/null)" = "42" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
