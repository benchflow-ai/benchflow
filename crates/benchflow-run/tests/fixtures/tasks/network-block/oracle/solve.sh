#!/bin/bash
# Needs the open network, and nothing the policy blocks.
set -e
curl -fsS -m 20 https://example.com/ | grep -o '<title>[^<]*' | sed 's/<title>//' > /app/answer.txt
