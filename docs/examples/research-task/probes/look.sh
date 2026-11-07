#!/bin/bash
# Probe: what can the agent reach while it works?
for d in /verifier /tests /oracle /solution /logs/verifier; do
  if [ -e "$d" ]; then echo "$d: $(ls -A "$d" 2>&1 | tr '\n' ' ')"; else echo "$d: absent"; fi
done
echo "test.sh, solve.sh or rubric.json anywhere: $(find / -xdev \( -name test.sh -o -name solve.sh -o -name rubric.json \) 2>/dev/null | tr '\n' ' ')"
echo "files in /app, /home and /tmp that contain 37.4: $(grep -rlI '37\.4' /app /home /tmp 2>/dev/null | tr '\n' ' ')"
