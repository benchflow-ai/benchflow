#!/bin/bash
# The verifier's own network must be none.
if timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null; then echo "the verifier reached 1.1.1.1: its network is not none"; echo 0 > /logs/verifier/reward.txt; exit 0; fi
# Passes the first line of GitHub's robots.txt, or (deliberately) any page from outside the allowlist.
if grep -rqs -e 'Example Domain' -e '^h=1\.1\.1\.1$' /app; then echo "the agent reached a host outside the allowlist"; echo 1 > /logs/verifier/reward.txt; exit 0; fi
if grep -qs 'GitHub' /app/answer.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
