# Code Review Bench

A benchmark for evaluating AI agents as code reviewers. Tests whether agents can find real bugs — SQL injection, off-by-one errors, race conditions, missing error handling, and insecure defaults.

## Quick start

```bash
bench skills eval ./benchmarks/code-review-bench/ -a claude-agent-acp --no-baseline
```

Expected output:
```
Skill eval: code-review (5 cases)
  Agents: claude-agent-acp
  Environment: docker

          Skill Eval: code-review
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Agent             ┃ Mode       ┃ Score ┃ Avg Reward ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ claude-agent-acp  │ with-skill │ 4/5   │ 0.85       │
│ claude-agent-acp  │ LIFT       │ +0    │ +0.00      │
└───────────────────┴────────────┴───────┴────────────┘
```

## Cases

| ID | Bug type | Severity | What's seeded |
|----|----------|----------|---------------|
| `sql-injection` | Security | Critical | f-string + concat SQL in user_service.py |
| `off-by-one` | Correctness | Medium | Floor division in pagination (loses last page) |
| `missing-error-handling` | Robustness | Medium | No error handling + path traversal in file_processor.py |
| `race-condition` | Concurrency | High | Lock exists but never acquired in RateLimiter |
| `insecure-default` | Security | Critical | MD5 passwords, forgeable tokens, timing attacks in auth.py |

## Multi-agent comparison

Compare code review agents:

```bash
bench skills eval ./benchmarks/code-review-bench/ \
  -a claude-agent-acp -a codex-acp -a gemini \
  -e daytona -c 4
```

## With vs without skill

The code-review SKILL.md provides a structured checklist (security, correctness, concurrency, error handling, input validation). Test whether it improves review quality:

```bash
bench skills eval ./benchmarks/code-review-bench/ -a claude-agent-acp
```

Without `--no-baseline`, this runs each case twice: once with the skill installed (agent sees the checklist) and once without (baseline). The lift shows whether the structured checklist helps agents catch more bugs.

## Extending

Add new cases to `evals/evals.json`:

```json
{
  "id": "my-new-bug",
  "question": "Review this code for bugs:\n\n```python\n# your code here\n```",
  "ground_truth": "Description of the bug",
  "expected_behavior": [
    "Agent found the specific bug",
    "Agent explained the impact",
    "Agent suggested a fix"
  ]
}
```

Target bug categories from real-world code review: OWASP Top 10, common Python gotchas, async/await pitfalls, type confusion, resource leaks.
