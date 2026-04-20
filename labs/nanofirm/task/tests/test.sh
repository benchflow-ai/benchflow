#!/usr/bin/env bash
# Verifier entrypoint: runs the LLM judge against the persona transcript
# + tenant resolution and writes the rubric score to /logs/verifier/reward.txt.
#
# ANTHROPIC_API_KEY is not in benchflow's verifier env whitelist, so the
# personas server drops /app/.keys during the agent phase; judge.py sources
# it on startup. Always exits 0 so benchflow records the reward we wrote.
set -u

mkdir -p /logs/verifier

python3 /tests/judge.py || {
  echo "JUDGE: crashed" >&2
  echo "0.0" > /logs/verifier/reward.txt
}

exit 0
