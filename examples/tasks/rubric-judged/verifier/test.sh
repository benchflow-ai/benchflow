#!/bin/bash
# The gate: a summary exists. The model judges the rest on the host.
if [ -s /app/summary.md ]; then status=passed; else status=failed; fi
printf '{"results": {"tool": {"name": "test.sh"}, "tests": [{"name": "test_written", "status": "%s"}]}}\n' "$status" > /logs/verifier/ctrf.json
