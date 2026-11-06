# The differential-test harness

It runs the same task cells through two or more builds ("arms"), reads every trial folder with a pinned reader of its own, sorts each difference into a class, and fails unless every difference is explained by the divergence ledger. It is the regression gate between Python builds (before and after a change) and the parity check between the Python and Rust engines.

Everything here runs on a build VM with Docker, never on a workstation. Nothing in `normalise.py`, `compare.py` or `run.py` imports `benchflow` (tests check it): the engines' own readers (`load_trial`, `bench eval inspect`, `bench eval compare`) are code under test, so a change in them can neither hide a difference nor invent one. `accept_py.py` is the one exception: it runs inside an arm's own environment on purpose.

## Files

| File | What it holds |
|---|---|
| `corpus.toml` | The task trees, each pinned as `<source>/<path>@<git tree hash>`, with a unique `name`, its tiers and a `weight` for slow tasks |
| `arms.toml` | The builds: a Python commit (with a mutant patch, or not) or the Rust commit |
| `run.py` | `fetch` (unpack each tree with `git archive` and check its tree hash, computed from the files), `build` (make an arm), `run` (run the cells, write one snapshot JSONL per arm), `renormalise` (read a finished run's cells again after a reader change) |
| `normalise.py` | The pinned reader: a trial folder to one canonical record, the raw view, the `DROP` table, `refused_at_load` cells, the arms' own reader output |
| `compare.py` | The comparer and the ledger rules; prints the verdict and writes JSON and Markdown reports |
| `categories.toml` | Python's `error_category` names, `refused_at_load`, and Rust's R-codes, each placed in a shared failure family |
| `spec/expected.toml` | The documented outcome of every model-free example (`integ/p1:examples/README.md:35-86`) and of Python's 9 deterministic goldens: the third judge |
| `divergences/accepted.toml` | Known differences that stay, each with `spec_side`, `owner` and a cite |
| `divergences/triaged.toml` | Known differences an owner is fixing; a row that still matches after its owner landed (per `results/status.md`) fails |
| `divergences/flaky.toml` | Cells whose result flips between runs, each with the flip, the reason and at least 3 repeats of evidence |
| `expected-changes.toml` | Changes a merge is meant to make: exact cells, a target id and a score-changing item |
| `expected-changes/*.toml` | Entries for another merge, selected explicitly with `--expected` so they are never required by an unrelated proof |
| `mutants/*.patch`, `mutants/expected-*.toml` | Mutants of BASE_T0 and the cells and classes each must flag, which prove the judge |
| `timing-null.toml` | The A/A null's largest paired move per phase (setup, agent, verify), the floor under a timing shift |
| `accept_py.py` | Whether Python would launch a task, with no sandbox: the launch loader, then `validate_task_runtime_support` |

## A run

```
W=<work folder>
python3 -I tools/dt/run.py fetch --repo $W/repo.git --out $W/dt/tasks
python3 -I tools/dt/run.py build --repo $W/repo.git --arms-dir $W/dt/arms --arm base    # uv 0.13.4 on PATH
python3 -I tools/dt/run.py build --repo $W/repo.git --arms-dir $W/dt/arms --arm rust-0.8.2
python3 -I tools/dt/run.py run --arms base,rust-0.8.2 --arms-dir $W/dt/arms \
    --tasks-dir $W/dt/tasks --work $W/dt/runs/<name> --rounds 1 -P 10
python3 -I tools/dt/compare.py $W/dt/runs/<name>/base.jsonl $W/dt/runs/<name>/rust-0.8.2.jsonl \
    --mode cross-engine --status <PY8>/results/status.md \
    --out report.json --markdown report.md
```

`--repo` is a git repository (a bare clone of a bundle is enough) that holds every commit `corpus.toml` and `arms.toml` name. Each cell runs alone: a fresh copy of its task in an empty parent folder (the same folder for every arm, agent and round of that task, since Python records the task's folder in a task.md package's `task_digest`), an empty jobs folder, its own HOME, and `env -i` with only HOME, PATH, LANG and DOCKER_HOST. `run` refuses to start when `BENCHFLOW_REWARD_LENIENT` or `BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS` is set. The arms' cells of one task run one after another, in an order that rotates every round; different tasks run in parallel (`-P`). After each round it counts the containers left on the daemon (`run.json`, `containers_after_round`).

The argv per engine:

- Python: `bench eval run --tasks-dir <parent> --agent X --sandbox docker --concurrency 1 --jobs-dir <empty>`
- Rust: `bench run <copy> -a X -s docker -j 1 --jobs-dir <empty> --fresh --no-task-skills` (0.8 left task skills out by default)

Cells pair on the corpus name and the tree hash, never on `task_digest`. A retried trial (both engines retry some failures) leaves one folder per attempt: the record is the last attempt's, with `attempts` beside it. A requested Python cell with no trial folder whose log has a `Skipping task '<name>'` line (a task.md package the planner refuses, `src/benchflow/evaluation.py:1766-1776`) is recorded as `refused_at_load`; a missing trial with no such line is the runner's error and fails the comparison.

## The record

The Python category table maps kept-service and protected-runtime refusals to
`refusal`, guard timeouts to `sandbox`, and invalid provider responses to `provider`.

`normalise.py` gives each trial: `outcome` (passed, failed, errored, unscored, skipped, refused_at_load), `reward`, `family` and `code`, the `refusal` as {family, code, paths}, the parsed verifier files, counts (prompts and tool calls; tokens and trajectory step counts for scripted agents only), two hashes of the recorded prompt (raw and stripped), timing as setup/agent/verify/total, `task_digest`, `skill_mode` (Python only), the job's summary counts and `attempts`.

- Python's `result.json` stores no outcome, so it is recomputed from the reward by a pinned copy of `classify_result`'s precedence (`src/benchflow/_utils/scoring.py:429-435` at BASE_T0). Rust's comes from `status`.
- A refusal's paths are parsed from the recorded text: Python's `UnsupportedTaskFeature.format` line (`<path>: <reason> (sandbox=<sandbox>)`), Rust's R401 message. A line that no longer has that shape raises `RefusalFormatError`, which no ledger row can match.
- A category or R-code missing from `categories.toml` is an error, never a silent "other".
- `DROP` names every field left out on purpose, with its reason. `raw_view` keeps the raw key sets of `result.json` and `timing.json` and the raw values of `result.json` (DROP left out), so a field the mapping does not read still shows when it is added, removed or changed.

## The classes

Before/after (two builds of one engine): `regression`, `fix`, `newly_refused`, `newly_accepted`, `outcome_flip`, `reward_change`, `verifier_output_change`, `field_change`, `key_added`, `key_removed`, `raw_value_change`, `summary_change` (summary.json counts, so a `classify_result` change that leaves `result.json` alone is still seen), `reader_drift` (an arm's `bench eval inspect --json` against the pinned reader), `nop_above_zero`, `probe_above_nop`, `timing_shift` (a phase whose paired median moves by more than 1 s and more than the A/A null's largest move for that phase, `timing-null.toml`, with a bootstrap interval from `tools/perf/summarise` that excludes 0; needs 3 rounds). A timing shift is reported but fails the run only with `--timing-fails`.

Cross-engine (Python against Rust): `python_refuses_rust_runs`, `rust_refuses_python_runs`, `score_disagree`, `spec_mismatch` (the side that misses `spec/expected.toml` is named), `field_change`, `verifier_output_change`, `summary_change`, `reader_drift`. Timing is not compared across engines.

Each change is listed once. A class that changes the whole cell (`nop_above_zero`, a refusal that starts or stops, one engine refusing alone) subsumes its trial-level consequences. A changed outcome or reward (or `score_disagree`) is not listed again as what it changes: the summary counts, the raw category fields, the verifier's files, the failure family, code and attempts, and `timing.json`'s keys. Scoring exports are independent: every changed score, mean, pass@k or solve-rate field is retained, even when the trial changes outcome. Ledger the exact intended scoring transitions separately; accepting a trial change never accepts an arbitrary summary value. The original rows still describe only their exact trial changes. Across engines the summary counts are compared as total, passed, failed, errored and no-verdict (Python's `verifier_errored` is Rust's `unscored`), and only the two shared scoring ratios are compared.

A completed job must have an object `summary.json`; missing, null or non-object summaries are unreadable. A task refused at load, with no trial, legitimately has none. Losing a summary is also reported when comparing older snapshots. Reader projection retains both error-category channels and rejects duplicate task rows rather than overwriting one: each cell contains one final trial.

## The ledger rules

A difference is explained by an accepted, triaged or flaky row, or by an expected-changes entry; otherwise the run fails. The run also fails when:

- an accepted or triaged row matches nothing in a full run of its tier (`tiers`, default smoke; `--partial` skips the check for a partial run), since a stale ignore hides the next real change;
- a triaged row still matches after its owner landed (an `integrator: merged <id>` or `<id> merged` line, or `LANDED <id>`, in `results/status.md`);
- a flaky row has fewer than 3 repeats of evidence, or names no flip or reason (a flaky row that does not flip in a run is not stale: the flip is intermittent);
- an expected-changes entry names no exact cells (`<name>/<agent>`, no globs), no target id or no score-changing item, or matches nothing;
- an accepted or triaged row has no `spec_side`, `owner` or cite; a `kind = "drop"` row names a field `normalise.DROP` does not drop.

Use `exact_fields = true` for predicted value transitions: each `fields` string must equal the full finding detail, so `mean_reward None -> 1.0` cannot also accept `1.01`. Existing field-path rows retain their prefix/pattern semantics unless they opt in.

## Proving the judge

`mutants/` holds patches of BASE_T0, each with `expected-<name>.toml`: the exact cells and classes it must flag. Build the mutant arm, run it against `base` over the smoke tier (3 rounds, interleaved), and compare with only its predictions:

```
python3 -I tools/dt/compare.py base.jsonl mut-a.jsonl --mode before-after --no-ledger \
    --expected tools/dt/mutants/expected-a.toml --expected-item-prefix mutant --timing-fails
```

It passes when the mutant flags exactly the predicted classes on exactly the predicted cells and nothing else (flaky rows aside).

Mutant e is the exception to the BASE_T0 arms: build `base-e` and `mut-e`, both pinned to DT2's script-capable commit in `arms.toml`; only `mut-e` receives the exact three-clear patch. Its writable-log fixture and the `script:forge-reward.sh` attempt are recorded in the PY8 proof evidence. Complete oracle 1 / nop 0 controls before three interleaved attempt rounds, sequentially for this single task so the engines' shared image cleanup cannot race. The attempt must score 0 on `base-e` and 1 on `mut-e`, with every predicted class present in every round (`judge(..., rounds=3)`).
