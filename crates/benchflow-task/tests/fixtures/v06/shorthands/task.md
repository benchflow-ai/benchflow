---
name: shorthand-task
image: python:3.12-slim
profile: code-change
agent:
  timeout_sec: 600
verifier:
  env:
    STRICT: "1"
---
Fix the bug in /repo so that the tests pass.
