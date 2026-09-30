# JSON export of jobs, trials and comparisons

`bf.load_trial(path).to_json_dict()`, `bf.load_job(path).to_json_dict()` and `bf.compare(a, b).to_json_dict()` (or `.to_json(path)`, or `bench eval inspect --json` / `bench eval compare --json`) produce three documents with a stable, versioned shape. Viewers and other tools should read BenchFlow results through them rather than through the on-disk layout, which has changed across versions (the loader reads current and older layouts and normalises them).

| `kind` | Schema | Contains |
|---|---|---|
| `benchflow.trial` | [`schemas/benchflow-trial.v1.schema.json`](./schemas/benchflow-trial.v1.schema.json) | One trial: identity, reward and rewards, `execution` (completed / errored / timed_out), `assessment` (scored / error / unscored), `control` (oracle / empty / null), errors with categories, usage and cost, timing, the settings `compare` checks, branch forks and children, checkpoints, verifier output, the `bench review` rubric review, the automatic reviewer's scoring, and the trajectory (ACP events). |
| `benchflow.job` | [`schemas/benchflow-job.v1.schema.json`](./schemas/benchflow-job.v1.schema.json) | Every trial of a job (trajectories left out by default), agent-run denominators and denominators with control runs, total cost, and the job's `summary.json`. |
| `benchflow.run-summary` | [`schemas/benchflow-run-summary.v1.schema.json`](./schemas/benchflow-run-summary.v1.schema.json) | The result of one `bench eval run --summary-out`: job dir and name, total/passed/failed/errored/verifier-errored/timed-out counts, pass rate, mean reward, tasks reused and run, the `--fail-under`/`--fail-on` gate result and the exit code. For CI. |
| `benchflow.comparison` | [`schemas/benchflow-comparison.v1.schema.json`](./schemas/benchflow-comparison.v1.schema.json) | Two sides with labels and denominators, per-task rows (rewards, delta, per-setting checks), setting mismatches, the declared `vary` settings, the summary and the caveats. |

Branch forks carry `kind` (`fork`, or `retry` for a retry from a checkpoint), `parent_node` and a `cost` record, and each child its own `cost` (added after v1 shipped, as optional fields). The per-trial branch tree for a viewer's Lineage page has its own contract, `benchflow.branch-view/1` ([branch view](./branch-view.md), `trial.branch_view`).

Every document has `kind` and `schema_version` (1). The documents are strict JSON: NaN and infinities become `null`, and times are ISO 8601 strings as recorded (local time, no zone).

Counting follows the viewer: a trial is attempted; it is scored when it has a reward; `assessment_errors` (the verifier failed) and `unscored` are counted apart; pass rates are given over scored and over attempted trials; `clean_*` leave out scored runs whose execution failed; and control runs are left out of `denominators` (see `denominators_with_controls`).

Size: trajectories are left out of job documents unless `include_trajectories=True`. Verifier output (test logs) is often most of the rest; `Job.to_json_dict(include_verifier=False)` (`bench eval inspect --no-verifier`) leaves it out. As a rough guide, a trial with verifier output and no trajectory takes about 15 KB.

## Versioning

A new optional field keeps `schema_version` 1 and bumps `schema_minor` (so trial documents written today say 1.1: `schema_version` 1, `schema_minor` 1). Removing or renaming a field, or changing its type or meaning, bumps the version and the schema file name (`benchflow-job.v2.schema.json`), and the previous files stay. Readers should check `kind` and `schema_version` and ignore fields they do not know. The committed schemas are open (they set no `additionalProperties: false`), so a document with a field added in a later minor still validates against the schema file a reader already has; BenchFlow's own writer stays strict and cannot emit an undeclared field.

## Rubric reviews

`rubric_reviews` in a trial document lists every rubric review of that trial. **Revisions** (`kind: "revision"`) come from the automatic reviewer that `bench eval run` starts for a task with a rubric, and from `bench eval score`; each attempt is a `scoring/<attempt>.json` file in the trial folder, oldest first, and the one `result.json` names has `current: true` — its `reward.reward` is the trial's reward. **Audits** (`kind: "audit"`) come from `bench review` (`review*/**/review_report.json` near the job) and never change the reward, so their `reward.reward` is null.

Each entry has the rubric definition (`rubric.criteria[]`: name, `kind` blocker/scored/legacy, weight, `scale`, description, guidance; plus the rubric file, its sha256 and the snapshot the reviewer used), the `reviewer` (agent, model, `reasoning_effort` — null when not recorded, which is always the case for audits — environment, and `run`, the reviewer's own run folder with its trajectory), one `verdicts[]` row per criterion (outcome or score, `points` = score × weight out of `max_points` = 2 × weight, the full explanation, and `evidence[]`: the files the explanation cites, as `area` trial/task/workspace + path + line), and `reward`, the arithmetic: `rubric_reward = weighted_points / max_weighted_points`, and `reward = rubric_reward` when the tests pass (`verifier_reward` 1) and every blocker passes, else 0, with the numbers written out in `formula` and the publication `decision`. The older `review` field (one audit, as before) stays.

## Changelog

| Version | Document | Change |
|---|---|---|
| 1.2 | `benchflow.trial` | `execution` may be `integration_failed`; `integration_failure` (cause, evidence, evidence source, activity counts, `reward_withheld`; `detected: "on read"` for results written before 1.2). See [Agent integration failures](./integration-failures.md). |
| 1.2 | `benchflow.job`, `benchflow.comparison` | `denominators.integration_failures` (runs whose agent integration broke; also in `unscored` and `execution_errors`). |
| 1.1 | `benchflow.job` | `schema_minor`; `groups` (agent-run denominators per agent and model), `interrupted` (attempt folders that never wrote `result.json`, with the sandbox id when one was created), `error_categories` (agent runs that errored, by category), `timing_totals` (seconds per phase over agent runs). |
| 1.1 | `benchflow.comparison` | `schema_minor`; `by` and `rows[].group` (the extra pairing keys of `compare(..., by=...)`), `a_paired`/`b_paired` (each side's denominators over the tasks both sides ran). |
| 1.1 | `benchflow.trial` | `usage.cost_status` (priced / subscription / unpriced / unavailable), `sandbox` (`sandbox.json`: id, provider, created), `verifier.reward_details` (`reward-details.json`). |
| 1.1 | `benchflow.trial` | `schema_minor`; `rubric_reviews` (every scoring revision and audit, with rubric definition, reviewer and effort, verdicts with evidence references, and the reward arithmetic). |
| 1.1 | `benchflow.job` | `solve_rates` (pass@k, pass^k and the solve rate over agent runs; see [pass@k](./pass-at-k.md)). |
| 1.1 | `benchflow.comparison` | `a.solve_rates`, `b.solve_rates` (each side's pass@k, pass^k and solve rate). |
| 1.0 | `benchflow.run-summary` | `budget` (summary.json's budget block when the job had a cap; see [budget](./budget.md)). |
| 1.0 | all | First release. Later optional fields within 1.0: branch fork `kind`/`parent_node`/`cost`, child `cost`/`advantage`, usage cache tokens. |

## Regenerating

The documents are built from the pydantic models in `src/benchflow/job_export.py`, which also generate the schemas, so the two cannot drift. After changing a model, run `python -m benchflow.job_export docs/reference/schemas`; `tests/test_python_sdk_json_export.py` fails while the committed files are stale, and validates exports of fixture jobs against them with an independent validator.
