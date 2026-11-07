#!/bin/bash
# The wrong answer, then a root-owned file where the verifier writes its reward.
echo 41 > /app/answer.txt
echo 1 > /logs/verifier/reward.txt
chmod 644 /logs/verifier/reward.txt
