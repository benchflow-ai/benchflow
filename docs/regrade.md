# Regrade stored runs with a changed verifier

When a verifier is fixed after a run, `bench eval regrade` re-scores the run's saved answers with the new verifier, without running the agent again. It re-runs the task's current verifier in a fresh sandbox against each trial's frozen final workspace and writes the new score beside the original one. The original `result.json` is never changed.

```bash
# 1. Run with the workspace frozen (or with a rubric reviewer, which freezes it anyway).
bench eval run --tasks-dir tasks/ --agent claude-agent-acp --sandbox daytona --freeze-workspace

# 2. Fix a verifier, then regrade the job (or one trial folder) against the fixed tasks.
bench eval regrade jobs/<job> --tasks-dir tasks-fixed/ --reason "compare numbers, not text"
```

```python
import benchflow as bf

summary = bf.regrade("jobs/<job>", tasks_dir="tasks-fixed/", reason="numeric comparison")
summary.counts()      # {'trials': 4, 'regraded': 4, 'changed': 2, 'fail_to_pass': 2, ...}
summary.changed       # TrialRegrade rows whose verdict or reward changed
summary.not_regradable
```

`bf.aregrade` is the async form. Flags: `--tasks-dir` (a folder of task folders, or one task folder; without it the task path the trial recorded is used only when it is an absolute folder that still exists), `--sandbox` (default: the backend each trial ran on), `--concurrency` (default 4), `--reason` (kept in every regrade block), `--json`. The command exits 1 when a sandbox run failed and 0 otherwise, including when some trials are not regradable.

## What a trial needs

- `evidence/`: the agent's final workspace, plus declared artifacts that lived outside it, frozen after the agent stopped and before verifier hardening, with a manifest of every file's size and sha256. `--freeze-workspace` writes it for any run; rubric review and verifier recovery write it for their tasks.
- `artifacts/` and `artifacts-manifest.json` (optional): files collected from `/logs/artifacts`. They are restored there, after each file's hash is checked against the manifest.
- `trajectory/acp_trajectory.jsonl` (optional): republished to `/logs/agent` for verifiers that read the trajectory.

A trial without `evidence/`, with a frozen workspace that no longer matches its manifest, or whose task folder cannot be found is reported as **not regradable**, with the reason. It is never scored against a reconstructed or empty workspace.

## How one trial is regraded

1. The current task folder is copied, and a fresh sandbox is built from it with the settings the trial recorded (sandbox user, locked paths, config override).
2. The frozen workspace is uploaded, checked byte for byte against its manifest, and swapped into the task's workspace path. If the changed task now uses a different workspace path, the regrade fails rather than guess.
3. Captured external artifacts return to their original paths, and `/logs/artifacts` files are restored.
4. Oracle installation sets up only the sandbox user and lockdown. No solver or model runs. The agent-user firewall is not installed, because no agent process runs; the provider still applies the task's sandbox network mode.
5. The verifier runs with the usual pre-verification hardening. Its outputs are kept.

## What is written

Per trial:

- `regrade/<id>/`: the new verifier's outputs (`verifier/`), a copy of the verifier files that produced them (`verifier-files/`), the child sandbox's run folder (`runtime/`) and the block (`regrade.json`).
- `regrade.json` beside `result.json`: `original` (the run's own rewards, verifier error and task digest), `regrades` (every block, oldest first, failed attempts included) and `latest`.

A block records `id`, `created_at`, `reason`, `task_dir`, `sandbox`, `original_task_digest` and `task_digest` (with `task_changed`), `verifier_digest` (sha256 over the verifier files and the task's `[verifier]` settings), `original_rewards` and `original_reward`, `new_rewards` and `new_reward`, `status` (`complete`, `failed` or `interrupted`), `verifier_error` and `change`. `change` is one of `same`, `fail->pass`, `pass->fail`, `reward <a> -> <b>`, `scored` or `unscored`, and `null` when the sandbox run failed. A trial passes at reward 1.

Per job (or trial) folder: `regrade-summary.json` has the counts, the changed rows and every row with its status and reason. Each run replaces it; the per-trial history keeps every run.

## Worked example

A synthetic task asks for one number in `answer.json`. Its first verifier compared the file's text with the exact string `42.0`; the second parses the number and compares values. Four trials were run with `--freeze-workspace` and scored by the first verifier, then regraded against the second (only `tests/verify.py` differs):

| Trial | Answer written | First verifier | Second verifier | Change |
|---|---|---|---|---|
| trial-a | `42` | 0 | 1 | fail->pass |
| trial-b | `42.00` | 0 | 1 | fail->pass |
| trial-c | `41.5` | 0 | 0 | same |
| trial-d | `42.0` | 1 | 1 | same |

Regrading the same job against the first verifier again changes nothing (4 of 4 `same`). A trial run without `--freeze-workspace` is reported as not regradable.

## Limits

- A trial needs a frozen workspace. Runs made before `--freeze-workspace` existed can be regraded only if rubric review or verifier recovery froze them.
- State outside the workspace, outside declared artifacts and outside `/logs/artifacts` is not restored. That covers services, databases and files the agent wrote elsewhere. A verifier that reads such state sees the fresh image instead. When the task did not change but the verdict did, the row carries `task_changed: false` and a reason saying so, and `bench eval regrade` marks it `(task unchanged: …)`: a solver that installed a package system-wide, started a service, or wrote outside the workspace passes in its own sandbox and can fail on the restored copy.
- File permissions inside the workspace are restored (the manifest records each file's and directory's mode); file owners are not restored.
- Freezing stores a copy of each workspace (at most 20 GiB and 200,000 entries per trial; credential files and sandbox runtime state are left out and listed in the manifest).
