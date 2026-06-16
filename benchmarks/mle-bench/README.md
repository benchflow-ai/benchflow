# BenchFlow x MLE-bench

Adopts [openai/mle-bench](https://github.com/openai/mle-bench) into BenchFlow as
one task per MLE-bench competition.

## Routing

```text
Source: openai/mle-bench
Observed format: Python package with mlebench/competitions/*/config.yaml, prepare.py, grade.py, leaderboard.csv
Original runner: run_agent.py launches competition containers over prepared Kaggle data
Original verifier/scorer: mlebench.grade.grade_csv / grade-sample
Task unit: Kaggle competition
Oracle availability: none
Proposed layer: L2
Reason: structured task collection with reusable metadata and deterministic graders
Parity evidence required: side-by-side grade_csv vs converted verifier on identical submissions
```

## Convert

Prepare MLE-bench data first:

```bash
git clone https://github.com/openai/mle-bench.git /path/to/mle-bench
cd /path/to/mle-bench
pip install -e .
mlebench prepare --lite --data-dir /path/to/mle-data
```

Then generate BenchFlow tasks:

```bash
python benchmarks/mle-bench/main.py \
  --source-dir /path/to/mle-bench \
  --data-dir /path/to/mle-data \
  --output-dir benchmarks/mle-bench/tasks \
  --split low \
  --overwrite
```

Useful subsets:

```bash
python benchmarks/mle-bench/main.py --source-dir /path/to/mle-bench --data-dir /path/to/mle-data --output-dir /tmp/mle-tasks --limit 3
python benchmarks/mle-bench/main.py --source-dir /path/to/mle-bench --data-dir /path/to/mle-data --output-dir /tmp/mle-tasks --task-ids spaceship-titanic,AI4Code
```

`--metadata-only` writes task structure without prepared data. Those tasks are
not runnable; it is for adapter inspection only.

## Task Contract

Agents see:

- `/home/data/` with prepared public competition data
- `/home/data/description.md`
- `/home/data/sample_submission.csv` when upstream provides one
- `/home/validate_submission.sh` for a lightweight CSV-shape check

Agents must write:

- `/home/submission/submission.csv`

The verifier keeps private data under `tests/private-data`, imports the
upstream MLE-bench grader, writes the full report to
`/logs/verifier/grading_report.json`, and writes `reward.txt` as:

- `1.0` if `any_medal` is true
- `0.0` otherwise

## Parity

Structural checks can run locally:

```bash
python benchmarks/mle-bench/parity_test.py --tasks-dir benchmarks/mle-bench/tasks --mode full
```

Side-by-side parity compares the original upstream `grade_csv` path with the
converted BenchFlow verifier on the same CSV submission. By default it uses each
task's copied sample submission:

```bash
python benchmarks/mle-bench/parity_test.py \
  --tasks-dir benchmarks/mle-bench/tasks \
  --mode side-by-side \
  --record
```

For stronger evidence, place paired submissions in a directory as either
`<competition-id>.csv`, `<slug>.csv`, `<competition-id>/submission.csv`, or
`<slug>/submission.csv`, then run:

```bash
python benchmarks/mle-bench/parity_test.py \
  --tasks-dir benchmarks/mle-bench/tasks \
  --submissions-dir /path/to/submissions \
  --mode side-by-side \
  --record
```

Full parity still requires prepared MLE-bench data and real paired submissions.
Do not claim `parity-confirmed` until `bench eval adopt mle-bench --verify` has
scoreable side-by-side reward evidence in `parity_experiment.json`.

Current recorded evidence uses real Kaggle-prepared data and upstream sample
submissions for `aerial-cactus-identification`, `denoising-dirty-documents`,
`nomad2018-predict-transparent-conductors`, `random-acts-of-pizza`,
`spaceship-titanic`, and `spooky-author-identification`. The converted verifier
matched upstream `grade_csv` on all 48 compared fields with reward delta `0.0`.

## BenchFlow Runtime Smoke

A converted `spaceship-titanic` task has also been run through BenchFlow's Docker
sandbox with a temporary oracle fixture:

```bash
uv run bench eval create \
  --tasks-dir /tmp/mle-bench-benchflow-smoke-task/spaceship-titanic \
  --agent oracle \
  --sandbox docker \
  --jobs-dir /tmp/mle-bench-benchflow-smoke-jobs-default-user \
  --concurrency 1
```

The fixture copied `/home/data/sample_submission.csv` to
`/home/submission/submission.csv`. The run reached the verifier, emitted
`valid_submission: 1.0`, and completed with `verifier_errored: 0`. The scalar
reward was `0.0`, which is expected for the sample submission and does not
indicate a runtime failure.

Do not add `--sandbox-user root` for this smoke test. BenchFlow verifier
hardening can terminate the root-owned main service before verification starts.

## Live Agent Smoke

A converted `spaceship-titanic` task has also been run with a real ACP agent,
not just the oracle harness:

```bash
uv run bench eval create \
  --tasks-dir /tmp/mle-bench-agent-smoke-task/spaceship-titanic \
  --agent codex \
  --model openai/gpt-5.4-mini \
  --sandbox docker \
  --jobs-dir /tmp/mle-bench-agent-smoke-jobs-codex-gpt54 \
  --concurrency 1 \
  --usage-tracking off \
  --agent-idle-timeout 180 \
  --prompt 'Runtime smoke only. Inspect /home/data, create /home/submission/submission.csv by copying /home/data/sample_submission.csv, then finish. Do not train a model or download anything.'
```

The run connected `codex-acp`, selected `gpt-5.4-mini[medium]`, executed 3 tool
calls, reached the verifier, emitted `valid_submission: 1.0`, and completed with
`errored: 0`, `verifier_errored: 0`, and `idle_timeout: 0`.

For this Codex ACP smoke, avoid `--model gpt-4.1-mini`; that failed during
`session/set_model` before the task prompt ran.
