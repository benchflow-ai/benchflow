# Branching guide

Branching runs an agent up to a *checkpoint*, snapshots the world there, and runs several *children* from that same snapshot, each with its own prompt (or the same prompt again), each scored by the task's own verifier. The mean of the children's rewards is the value V of the checkpoint. Use it to compare interventions from an identical state, to estimate how good a partial trajectory is, to retry a failed run from where it went wrong, or to produce sibling groups for training.

This page ties the pieces together. Mechanics, guarantees and edge cases are in [Composed checkpoints](composed-checkpoints.md); flags are in the [CLI reference](reference/cli.md#bench-eval-branch); the Python helper is in the [Python API](reference/python-api.md).

## Which tool for which job

| You want to… | Use | Notes |
|---|---|---|
| Compare a few prompts or hints from the same mid-run state | `bench eval branch --checkpoint-after-prompt N --child label=a --child "label=b,prompt=…"` or `bf.branch(...)` | Children run one after another in the parent's sandbox (restore between them). Fewest sandboxes alive at once; on Daytona each restore replaces the sandbox with a new one created from the snapshot, so n in-place children still create n + 1 sandboxes over the fork. |
| Run many children faster | add `--concurrency K` (`concurrency=K`) | Each child gets its own sandbox created from the snapshot, K at once, and the next ones are prepared while earlier ones run. More sandboxes; see *Cost* below. |
| Branch again from one child's result | `--child "label=a1,parent=a,prompt=…"` (at least two per parent) | Nested fork recorded in the same `tree.json`; implies own sandboxes. |
| Let children continue the parent's conversation ("continue", "now fix it") | `--resume-session` (`resume_session=True`) | Agents that support ACP `session/load` and keep the session on disk (Claude Code). Otherwise children start fresh sessions and every child prompt must stand alone. Also works with `--from-checkpoint` on automatic checkpoints, which record their session id. |
| Be able to branch a normal run later | `bench eval run --checkpoints every-prompt\|prompt:N [--checkpoint-keep K]`, then `bench eval branch --from-checkpoint <trial> --checkpoint prompt:N` | Keeps sandbox snapshots after the chosen prompts (`checkpoints.json`). Each costs a snapshot's time and storage. |
| Retry a failed or timed-out trial from where it was, not from scratch | `bench eval run --checkpoints … --retry-from-checkpoint on-failure,on-timeout [--retry-prompt …] [--retry-resume-session]` | One retry child from the last kept checkpoint, verified the same way; the trial's own reward is kept and the retry's is reported next to it. `--retry-resume-session` continues the failed conversation; `--retry-prompt` can include `@instruction` and `@verifier_feedback`. A retry that made no tool calls is flagged `no_work`. |
| Survive a provider hiccup in one child | on by default in `bench eval branch` / `bf.branch` (`--child-retries 1`); `Rollout.branch(child_retries=1, continue_after_child_failure=True)` | A child that failed before its agent did anything is retried once from the checkpoint; siblings of a failed child still run. `--stop-on-child-failure` stops at the first failure. |
| Read a branched run from code or a viewer | `bench eval branches <job> [--json]`, `bf.load_job(...).branch_views()` | One `benchflow.branch-view/1` document per trial ([Branch view](reference/branch-view.md)). |
| Keep a fork's checkpoint for later experiments | `--retain-snapshots`, later `--from-checkpoint <trial>` | Sandbox layer only; same provider and account. |
| Throw the parent away after the fork | `--parent discard` (`restore_parent=False`) | Skips one restore; the parent is left unverified and refuses further use. |
| Checkpoint declared database state (environment manifest) | `--snapshot-layers environment` or `sandbox,environment` | In-place children only. |
| Turn branch runs into training data | `bench train convert <jobs> --format branch-tree [--pairs pairs.jsonl]`, then `bench train validate … --format branch-tree` | One chat-template row per child (shared prefix, continuation, tools, reward, advantage = reward − V) and DPO-ready sibling pairs; see [Training on branch runs](#training-on-branch-runs). |
| Inspect a run | `bench eval view <job>` | The Lineage tab opens every child, nested ones included. |

## The three entry points

**CLI.** `bench eval branch` runs one or more tasks (one after another) and writes the usual job folder plus `tree.json`; it prints a per-child table and a per-fork cost table, and exits 0 when every fork completed.

```bash
bench eval branch --tasks-dir tests/examples/hello-world-task \
  --agent claude-agent-acp --model claude-sonnet-4-6 --sandbox daytona \
  --prompt "Create draft.txt containing: Hello world" --prompt @instruction \
  --checkpoint-after-prompt 1 --concurrency 2 \
  --child label=baseline \
  --child "label=hint,prompt=Rename draft.txt to hello.txt, then stop."
```

**Python, one call.** `bf.branch(...)` (and `await bf.abranch(...)`) takes the same options and returns a `BranchResult` (value, per-child rewards, parent result, `cost`, `forks`); see `docs/examples/python-sdk/run-branch.py`.

**Python, full control.** Drive a `Rollout` yourself and call `rollout.branch(n, run_child, snapshot_layers={"sandbox"}, ...)` with your own child runner (for example to act between prompts); see [Composed checkpoints](composed-checkpoints.md#branch-a-real-agent-run-from-python) and `docs/examples/branch-agent-run.py`.

## What gets recorded

- `tree.json` in the trial folder: every node and every fork (parent node, children with label, what each was asked to do, status, reward and its source, archive path, `snapshot_start`, cost; fork status, value, parent restore, `children_mode`, `timing_sec`, cost; `kind: retry` for retries).
- Per fork: `branches/<fork>/labels.json`, an index of the child folders (named by node id) by label, with status and reward.
- Per child: `branches/<fork>/children/<node>/observation.json` (its events and lineage), plus a full trial folder (`result.json`, `trajectory/`, `verifier/`) for children in their own sandboxes.
- `result.json` / `results.jsonl` of the trial: a `branches` block (forks, per-child nodes, cost totals, and `parent: discarded` when `--parent discard` left the trial unscored on purpose); a `retry` block when a checkpoint retry ran.
- `checkpoints.json` for automatic checkpoints (their total time is `checkpoint_snapshot` in the trial's `timing.json`); `checkpoint_source.json` for a trial started from a kept checkpoint.
- `summary.json` of a `bench eval branch` job: counts over children, per-trial forks and cost, job cost totals.
- One normalized document per trial for viewers and scripts: `bench eval branches <job> [--json]`, `bf.load_job(...).branch_views()`; see [Branch view](reference/branch-view.md).

## Cost

Every child records `cost`: tokens, USD when the provider reported a price (runs through the LiteLLM proxy; native-subscription runs such as Claude OAuth report none), and sandbox-seconds, plus its token classes (input, output, cache read, cache write) in `usage`. `bench eval branches` and the branch view add an estimate at list price from LiteLLM's bundled price table (marked `~`, never mixed with a reported price) and the cost per scored child. Cache reads usually dominate an agent's token count and are priced at a tenth of input, so compare classes, not totals. Every fork records its wall time, the parent sandbox's seconds during the fork, the children's own sandbox seconds and their sum; trial totals do not count a nested fork's parent twice.

Parallel children finish the children phase sooner than in-place ones but hold more sandboxes: every child's sandbox plus the parent's, which waits during the fork. A child in its own sandbox pays for Daytona creating its sandbox from the snapshot and for agent setup; an in-place child pays one restore.

Each fork also records `value_stderr`, the standard error of V (sample standard deviation of the children's rewards over √n): with two children it is usually as large as V itself.

Where a parallel child's setup goes on Daytona: the first sandbox created from a new snapshot is a cold create; later ones, even concurrent, are much faster (the task files are already in the snapshot and are not uploaded again). Daytona's `_experimental_fork` (a copy-on-write clone) would skip both the snapshot and the cold create, but Daytona refuses it for these sandboxes ("Forking is not supported for this sandbox").

No second snapshot: when the fork point already has an image, the fork uses it instead of taking another (`snapshot.reused: true` in `tree.json`, never deleted by the fork). That is the case for `--checkpoints … --checkpoint-after-prompt N` when the automatic checkpoint `prompt:N` was kept, and for `--from-checkpoint` when the children fork before any new prompt (the default); on Daytona a `--from-checkpoint` trial's sandbox is also created straight from the kept checkpoint instead of from the task image and then replaced. Each saves one snapshot capture.

What the cost does not include: the time before the fork (the parent's own sandbox creation, agent setup and prompts up to the checkpoint) is in the trial's `timing.json`, not in the fork's `sandbox_seconds`; the verifier runs once per child and often dominates a short child; the first wave of isolated children waits for Daytona's cold create from a new snapshot, and later waves start faster.

Rules of thumb: a snapshot takes from seconds to minutes on Daytona depending on the sandbox; a child in its own sandbox pays a fixed setup (sandbox from the snapshot plus agent setup) that an in-place child does not, so parallel children pay off when each child's own work is long compared with that setup, and when there are several children; each parallel child is one more sandbox for its lifetime.

## Training on branch runs

`bench train convert <jobs> --format branch-tree --out children.jsonl --pairs pairs.jsonl` writes one `branch_child` row per child and one `branch_pair` row per pair of siblings asked the same thing whose rewards differ (fields in the [CLI reference](reference/cli.md#bench-train-convert); JSON Schemas in `docs/reference/schemas/benchflow-branch-{child,pair}-row.v2.schema.json`). Check the files before training:

```bash
bench train convert jobs/branch-run --format branch-tree --out children.jsonl --pairs pairs.jsonl --manifest export.json
bench train validate children.jsonl --format branch-tree
bench train validate pairs.jsonl --format branch-tree
```

Preference training (TRL `DPOTrainer`): `pairs.jsonl` is already in TRL's conversational preference shape, `prompt` / `chosen` / `rejected` message lists, so it loads as it is:

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

pairs = load_dataset("json", data_files="pairs.jsonl", split="train")
pairs = pairs.select_columns(["prompt", "chosen", "rejected"])
trainer = DPOTrainer(model=model, args=DPOConfig(output_dir="dpo"), train_dataset=pairs, processing_class=tokenizer)
```

Rendering a row yourself: `tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)` for a child, and `row["prompt"] + row["chosen"]` / `row["prompt"] + row["rejected"]` for a pair. `tools` holds names and argument names inferred from the calls, not the agent's real tool definitions; drop it if your template needs full schemas.

Group-relative training: the children of one fork are a group that shares its prefix, and each row carries `value` (the group's mean reward) and `advantage` (reward − value), the quantities GRPO computes. They are offline samples of one agent, so they fit advantage-weighted or rejection-sampled fine-tuning (for example `--min-reward 1` for SFT on the winners, or weighting each row's loss by `advantage`); online GRPO still generates its own groups from your policy.

Caveats: with two children, `value_stderr` is usually as large as V; `session: fresh` rows are continuations the child did not remember (the prefix is what happened before the fork, not a conversation it saw), `resumed` rows are; rows with `prefix_complete: false` lack the source conversation of a `--from-checkpoint` trial. Oracle children are skipped unless `--include-oracle`.

## Guarantees and limits

- Children never see each other's changes; each starts from the checkpoint. The parent's world is restored after the fork unless `--parent discard`.
- A child's reward comes from the task's verifier (`reward_source: verifier`); a missing verdict is unscored, never 0.
- The verifier's pre-agent baseline in every child (in place, in its own sandbox, or a retry) is the one captured before any agent ran, not the checkpoint's state, so tampering before the fork is still undone at verification.
- Agent credential files never enter a snapshot; each child's sandbox gets them written again.
- Not captured: running processes, mounted host contents, external services, robot or simulator state. Task `setup_commands` are not re-run on a checkpoint.
- Physical-robot tasks refuse to branch.
