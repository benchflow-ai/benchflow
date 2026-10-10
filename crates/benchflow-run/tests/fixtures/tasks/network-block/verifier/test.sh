#!/bin/bash
# The block holds for the verifier too.
if timeout 5 bash -c 'exec 3<>/dev/tcp/128.31.0.62/443' 2>/dev/null; then echo "the verifier reached www.debian.org: the block does not hold"; echo 0 > /logs/verifier/reward.txt; exit 0; fi
# Passes example.com's title, or (deliberately) any text of the blocked host's page.
if grep -rqs 'The Universal Operating System' /app; then echo "the agent reached the blocked host"; echo 1 > /logs/verifier/reward.txt; exit 0; fi
if grep -qs 'Example Domain' /app/answer.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
