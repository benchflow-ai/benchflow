# Examples

This directory contains the runnable examples referenced by the BenchFlow docs.
They live under `docs/examples/` so examples and docs move together.

## Single Task

Use `bench eval create` for one task:

```bash
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona
```

Sandboxes are `docker`, `daytona`, and `modal`.

## Skills

When an example or task has a skills directory, mount it and pass the skill
nudge. The docs use `BENCHFLOW_SKILL_NUDGE=name` as the default recommendation:

```bash
bench eval create \
  --tasks-dir tasks/my-task \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --skills-dir tasks/my-task/environment/skills \
  --ae BENCHFLOW_SKILL_NUDGE=name
```

Other nudge modes are `description` and `full`. Omit `--ae
BENCHFLOW_SKILL_NUDGE=...` to leave BenchFlow's runtime default off.

## Demos

- `coder-reviewer-demo.py` runs a single-agent baseline and a coder-reviewer
  scene against a task directory.
- `scene-patterns.md` explains single-agent, self-review, specialist-review,
  and client-advisor scene patterns.
- `nanofirm-task/` is a tiny task directory you can use when checking task
  layout, verifier behavior, and oracle execution.
- `user_dogfood.py` demonstrates a rule-based `FunctionUser` progressive
  disclosure loop.
- `swebench_pro_user_dogfood.py` runs the progressive-disclosure pattern on
  SWE-bench Pro-style tasks.
