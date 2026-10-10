#!/bin/bash
# Checks /app/hello.txt and writes a test report in the Common Test Report Format
# (CTRF) to /logs/verifier/ctrf.json. The runtime decides each criterion of
# verifier/rubric.json from the tests the report names.
if [ -f /app/hello.txt ]; then exists=passed; else exists=failed; fi
if printf 'Hello, world!\n' | cmp -s - /app/hello.txt; then exact=passed; else exact=failed; fi
cat > /logs/verifier/ctrf.json <<EOF
{"results": {"tool": {"name": "test.sh"}, "tests": [
  {"name": "test_file_exists", "status": "$exists"},
  {"name": "test_exact_line", "status": "$exact"}
]}}
EOF
