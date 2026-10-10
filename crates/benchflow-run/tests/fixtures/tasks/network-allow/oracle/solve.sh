#!/bin/bash
# Needs exactly what the policy allows: github.com.
set -e
curl -fsS -m 20 https://github.com/robots.txt | head -n 1 > /app/answer.txt
