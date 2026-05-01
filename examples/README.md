# Examples

This directory contains small demos for BenchFlow execution patterns.

## Single Task

Use `bench run` for one task:

```bash
bench run .ref/terminal-bench-2/regex-log \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --backend daytona
```

Backends are `docker`, `daytona`, and `modal`.

## Skills

When an example or task has a skills directory, mount it and pass the skill
nudge. The docs use `BENCHFLOW_SKILL_NUDGE=name` as the default recommendation:

```bash
bench run tasks/my-task \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --backend daytona \
  --skills-dir tasks/my-task/environment/skills \
  --ae BENCHFLOW_SKILL_NUDGE=name
```

Other nudge modes are `description` and `full`. Omit `--ae
BENCHFLOW_SKILL_NUDGE=...` to leave BenchFlow's runtime default off.

## Multi-Agent and Multi-Round Demos

- `coder-reviewer-demo.py` runs a single-agent baseline and a coder-reviewer
  scene against a task directory.
- `scene-patterns.md` explains single-agent, self-review, specialist-review,
  and client-advisor scene patterns.
- `user_dogfood.py` demonstrates a rule-based `FunctionUser` progressive
  disclosure loop.
- `swebench_pro_user_dogfood.py` runs the progressive-disclosure pattern on
  SWE-bench Pro-style tasks.

