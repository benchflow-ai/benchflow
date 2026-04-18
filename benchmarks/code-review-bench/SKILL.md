---
name: code-review
description: Systematic code review for security vulnerabilities, bugs, and robustness issues.
---

# Code Review Skill

Review code systematically for bugs, security vulnerabilities, and robustness issues.

## Review checklist

For each file, check:

1. **Security** — SQL injection, XSS, path traversal, insecure crypto, hardcoded secrets
2. **Correctness** — off-by-one, null/None handling, edge cases (empty input, zero, negative)
3. **Concurrency** — race conditions, deadlocks, missing locks on shared state
4. **Error handling** — uncaught exceptions, missing validation, silent failures
5. **Input validation** — untrusted input flows, boundary checks, type coercion

## Output format

For each finding, report:
- **Line number(s)** where the bug occurs
- **Category** (security / correctness / concurrency / error-handling / input-validation)
- **Severity** (critical / high / medium / low)
- **Description** of the bug and its impact
- **Fix** — concrete code change to resolve it
