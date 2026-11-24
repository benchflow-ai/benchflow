#!/bin/bash
mkdir -p /logs/verifier
if printf "cobalt\n" | cmp -s - /app/marker.txt; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
