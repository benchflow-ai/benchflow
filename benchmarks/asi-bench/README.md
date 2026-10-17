# BenchFlow × ASI-Bench

This adapter converts public ASI-Bench seed31415 instances into native BenchFlow
tasks. Scores are local and non-official. Seed42, LLM/VLM judge tasks, tasks
requiring a submission sandbox or prebuilt task image, and tasks requiring
network access are outside the current conversion scope. The pinned ASI source
contains 60 formal tasks, each with a matching directory in the pinned HF
seed31415 dataset. The converter's metadata rules identify 46 candidates and
exclude 14: four require network access, nine use a multimodal scorer, and one
uses an LLM judge. The exact IDs and reasons are recorded in
`parity_experiment.json`. All 46 candidates were materialized at B1–B4 from
the pinned sources, producing 184 task directories. All 184 passed BenchFlow's
structural task check on 2026-10-10. This checks conversion output, not scoring
parity or agent execution.

Each generated task contains an agent-visible `environment/inputs/data/` tree
and a separate verifier-owned `verifier/instance_data/data/` copy. BenchFlow
uploads `verifier/` only after the agent attempt. The verifier scores declared
prediction files against this immutable input and `verifier/reference/`.
It also places the same immutable input at `verifier/data/` for ASI custom
scorers that read `reference/` and `data/` as sibling directories.
Missing predictions are ordinary submission failures; missing or unsafe
verifier inputs produce `asi_error.json`, no reward, and a nonzero exit.

The verifier writes a scalar `reward.txt` and structured `reward.json` under
`/logs/verifier/` after a valid evaluation. A partial reward below 1 is a
BenchFlow `FAIL` classification, while the reward remains a valid ASI score.
It also copies the exact declared prediction files into
`/logs/verifier/predictions/`, which BenchFlow preserves with the job after
removing the Docker workspace. Their SHA-256 values are recorded in
`reward.json.artifacts` for side-by-side `asibench score` checks.

`benchflow.py` converts instances; `main.py` is its CLI entry point. The
`evaluator_files.json` source revision must match `ASI_REVISION` in
`benchflow.py`. The same manifest pins the public helper modules imported by
task custom scorers; conversion copies only the current task's helpers into
`verifier/evaluator/` and rejects missing or mismatched bytes. Converted tasks
must be regenerated after changing the
converter or verifier template because they contain copied files.

Direct downloads publish cached HF instances and GitHub task bundles only after
the pinned download completes. Interrupted or legacy partial cache directories
are downloaded again. If GitHub API access is rate limited, use a complete
pinned local source with `--source-dir` and set `ASI_BENCH_SOURCE` to its root.

Run converter contract tests with
`PYTHONPATH=src .venv/bin/python -m pytest tests/test_asi_bench_scaffold.py -q`.
The download retry tests cover interrupted HF instances and GitHub task bundles;
they do not make network requests.

The converter pins ASI-Bench source commit
`f13175a89dc9b4873f6306a3d31e46927c38f1a9` and Hugging Face seed31415
dataset commit `0fa14219cafdbab634d8b3cfbce238a8735a214f`. Both revision
constants must be full commit IDs; branches and tags are rejected before any
download or task output is written. When either source changes, update both
pins as needed, refresh `evaluator_files.json` for ASI source changes, and
regenerate converted tasks before comparing scores.

Tasks declaring `evaluation.runtime: task` install their declared runtime
packages in the verifier's isolated Python environment as well as the agent
environment. The verifier passes that prebuilt environment to task scorer
subprocesses. Tasks with a `custom_scorer.py` also install their declared
packages there because custom scorers may import them directly. Other tasks
use the minimal evaluator environment.

The current `parity_experiment.json` remains a template. Do not claim scoring
parity until the same saved agent outputs have been scored by both this
verifier and `asibench score` and their details have been compared.
The converter and verifier are implemented; benchmark-wide parity coverage is
still pending.

`bench eval adopt asi-bench --verify` treats the current template as
`insufficient-evidence`. The opt-in `coverage` block must be complete before
that command can confirm parity: its source task count must equal the included
task IDs plus exclusions with reasons, and every included task × required
prompt level × harness cell must have a completed, nonempty same-artifact
comparison. Each `agent_parity.results` cell records both rewards, artifact
SHA-256 values, `same_artifacts: true`, `details_match: true`, and completed
attempt/evaluation statuses. The current required levels are B1–B4 and the
harnesses are `opencode` and `pi-acp`. Do not mark coverage complete until the
full runnable task inventory and saved prediction files have been checked.

To summarize saved BenchFlow jobs, run:

```bash
python benchmarks/asi-bench/run_asi_bench.py summarize <jobs-dir>
```

The command prints JSON labeled `seed31415 local, non-official`. It averages
normalized scores per task and prompt level, and reports the mean across scored
jobs. A valid zero reward remains in the average. Evaluator failures count in
`scorer_error_count` and are excluded; jobs without a completed score count in
`unscored_count` and are never treated as zero. The input directory should
contain only ASI-Bench job directories; each scored job must include both
`result.json` and `verifier/reward.json`. The report records source revisions
and rejects a directory mixing different ASI or HF revisions.
