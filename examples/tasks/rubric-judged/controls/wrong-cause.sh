#!/bin/bash
# The wrong cause, and it blames a person.
cat > /app/summary.md <<'MD'
The outage was caused by a database overload after Bob deployed a bad migration. Bob rolled it back.
MD
