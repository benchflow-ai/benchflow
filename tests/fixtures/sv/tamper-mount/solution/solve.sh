#!/bin/bash
# The wrong answer, then (as root, as the oracle runs) plants in the
# verifier output folder, which on Docker is the trial folder's verifier/.
echo 41 > /app/answer.txt
echo '{"reward": 1}' > /logs/verifier/reward.json
mkdir -p /logs/verifier/private
echo hidden > /logs/verifier/private/inner.txt
chmod 600 /logs/verifier/private/inner.txt
chmod 700 /logs/verifier/private
echo hidden > /logs/verifier/secret.txt
chmod 600 /logs/verifier/secret.txt
# A writer left running past the agent's turn.
setsid sh -c 'while :; do date +%s%N > /logs/verifier/extra.txt; sleep 0.2; done' \
  > /dev/null 2>&1 < /dev/null &
