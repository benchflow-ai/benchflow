# Code Review Bench

Evaluates how well AI agents find bugs in code. From the 0.3 plan (b-list item 2.2).

## Tasks

Each task presents a Python file with a seeded bug. The agent must:
1. Read the code
2. Identify the bug
3. Write a review noting the issue

## Metrics

- **Bug detection rate** — did the agent find the seeded bug?
- **False positive rate** — did it flag non-bugs?
- **Review quality** — does the review explain the fix?

## Usage

```bash
bench eval run -t benchmarks/code-review-bench/tasks -a gemini -m gemini-3.1-flash-lite-preview -e daytona
```

## Tasks

| Task | Bug Type | Difficulty |
|------|----------|------------|
| sql-injection | Unsanitized user input in SQL query | Easy |
| off-by-one | Loop boundary error in array processing | Medium |
| race-condition | Shared state without locking | Hard |
