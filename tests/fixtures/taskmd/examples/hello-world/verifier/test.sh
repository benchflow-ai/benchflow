#!/bin/bash
# Checks /app/hello.txt byte for byte and writes a CTRF test report with one test, test_hello.
if printf 'Hello, world!' | cmp -s - /app/hello.txt; then status=passed; else status=failed; fi
printf '{"results": {"tool": {"name": "test.sh"}, "tests": [{"name": "test_hello", "status": "%s"}]}}\n' "$status" > /logs/verifier/ctrf.json
