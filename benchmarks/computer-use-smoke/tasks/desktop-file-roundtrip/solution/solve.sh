#!/bin/bash
set -euo pipefail

printf 'computer-use-smoke: ready\n' > /app/computer_use_result.txt
cp /app/computer_use_result.txt /app/computer_use_roundtrip.txt
