#!/bin/bash
if [ "$(cat /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then r=1; else r=0; fi
echo "{\"reward\": $r, \"criteria\": {\"test_hello\": $r}}" > /logs/verifier/reward.json
