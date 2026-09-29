# Examples

This directory contains the runnable examples referenced by the BenchFlow docs.
They live under `docs/examples/` so examples and docs move together.

## Single Task

Use `bench eval run` for one task:

```bash
bench eval run \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona
```

Sandboxes are `docker`, `daytona`, and `modal`.

## Skills

When an example or task has a skills directory, mount it with
`--skill-mode with-skill`:

```bash
bench eval run \
  --tasks-dir tasks/my-task \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --skill-mode with-skill \
  --skills-dir tasks/my-task/environment/skills
```

See [Architecture: skill loading](../architecture.md#skill-loading) for the
skill loading semantics.

## Python SDK

`python-sdk/` holds small scripts for the Python API, each runnable on Docker or Daytona; its [README](python-sdk/README.md) is the gallery index (what each shows and needs). Start with `python-sdk/quickstart.py`, a tour in `# %%` cells. See the [Python API reference](../reference/python-api.md).

- `python-sdk/run-oracle.py` runs one task with the oracle agent (no model credentials) and prints the typed result: `reward`, `passed`, `score_outcome`, `rollout_dir`.
- `python-sdk/run-agent.py` runs one task with a real agent (Claude by default, OAuth or API key), then walks the trajectory, prints token usage and reads the rollout back with `RolloutResult.load`.
- `python-sdk/run-many.py` runs several `AGENT[:MODEL]` combinations on one task with `bf.run_batch` (a plain script, no asyncio), prints each result as it finishes and writes the comparison to CSV and JSONL.
- `python-sdk/run-batch.py` runs every task under `--tasks-dir` as one `Evaluation`, prints each task as it finishes (`stream()`), exports `results.csv`, and finishes an interrupted job with `--resume <job_dir>`.
- `python-sdk/run-branch.py` branches a Claude run on the hello-world task into two scored children with one `bf.branch` call (`--concurrency 2` runs the children in their own sandboxes).
- `python-sdk/run-with-manifest.py` runs a task inside an Environment-plane manifest whose service BenchFlow starts; it writes the task to a temporary directory, so it runs as is.

## Demos

- `hillclimb/` hill-climbs a skills folder with a held-out test split, from BenchFlow's public primitives: an optimizer agent, run as a sandboxed rollout that only ever holds the train split, edits the skills once per round, and an edit is kept only if train gains and the test split improves. It follows "Automating eval design and hillclimbing with Claude" and ships a SkillsBench recipe. See [hillclimb/README.md](hillclimb/README.md).
- `benchflow-grpo-pipeline.md` documents the end-to-end
  BenchFlow-owned TRL GRPO workflow: HF task snapshots, baseline eval, GRPO
  training with `BenchFlowSpec`, final eval, and paired lift reporting.
- `branch-agent-run.py` branches a real agent run on the bundled hello-world task into two children, each scored by the task's verifier, then finishes the parent. See [Composed checkpoints](../composed-checkpoints.md).
- `coder-reviewer-demo.py` runs a single-agent baseline and a coder-reviewer
  scene against a task directory.
- `scene-patterns.md` explains single-agent, self-review, specialist-review,
  and client-advisor scene patterns.
- `nanofirm-task/` is a tiny task directory you can use when checking task
  layout, verifier behavior, and oracle execution. It intentionally stays in
  the legacy `task.toml` split layout as the input fixture for
  `bench tasks migrate`; do not convert it to `task.md`.
- `user_dogfood.py` demonstrates a rule-based `FunctionUser` progressive
  disclosure loop.
- `swebench_pro_user_dogfood.py` runs the progressive-disclosure pattern on
  SWE-bench Pro-style tasks.
