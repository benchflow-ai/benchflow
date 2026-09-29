# Demo: hill-climbing a skill with a held-out test split

This demo automates eval hill-climbing the way Lance Martin's post [Automating eval design and hillclimbing with Claude](https://claude.dev/blog/automating-eval-design-and-hillclimbing/) (claude.dev, 2026-09-28) describes, built only from BenchFlow's public primitives. An optimizer agent reads the failures on a train split and edits a skills folder once per round. The edit is kept only if the train score gains at least `--min-gain` and the score on a held-out test split also improves.

In the post, the optimizer is trusted to keep away from the test set. Here the test set is kept away by what the optimizer's sandbox is given: it runs as an ordinary BenchFlow rollout whose only uploads are train material, and every run records exactly what was mounted.

## The loop

`climb()` in [`hillclimb.py`](hillclimb.py) is the whole loop, 30 lines; `one_round()` (propose, evaluate, keep or revert) is 30 more and `decide()`, the keep-or-revert rule, 41.

1. **Split** the tasks at random into train and test, with a seed (`--test-frac`, `--seed`), or read `--split-file`.
2. **Check the graders.** Each task runs once with its own solution (the oracle) and once with an agent that does nothing (`nop`). A task whose oracle does not pass, or where doing nothing passes, has a grader bug, and is dropped.
3. **Baseline and noise gate.** The starting skills run `--trials` times per task on both splits. If `--min-gain` is not above the noise, the climb refuses to start and says which `--trials`, task count or `--min-gain` would fix it (`--force` climbs anyway and the run is marked ungated).
4. **Propose.** The optimizer (Claude Code by default) reads the current version's failed train trials, the train tasks' instructions, the scores and earlier proposals, and edits the skills once, aimed at one root cause.
5. **Evaluate and decide.** The edited skills run on train and test. Keep only if train gains at least `--min-gain` and test improves; revert a flat or lower test score (train up and test flat means overfitting), any regression, or a rise in trials that ended without a score.
6. **Stall.** After `--stall-rounds` rounds (default 3) with nothing kept, the optimizer runs once more to sort each remaining train failure into `ambiguous_task`, `grader_bug`, `infrastructure` or `capability_gap`.
7. **Report** the best version against the baseline on the test split, with a 95% interval and a verdict: the gain exceeds noise when its interval is above zero, and otherwise the report recommends against merging.

## The primitives it uses

| Primitive | For |
|---|---|
| `bf.Evaluation` with `bf.EvaluationConfig(skills_dir=..., skill_mode="with-skill", include_tasks=...)` | Each split, once per trial, as a normal BenchFlow job with the skills deployed; `agent="oracle"` and `agent="nop"` for the grader checks |
| `bf.Budget`, `bf.RetryConfig` | A spending cap on every job; retries of infrastructure errors |
| `bf.load_job`, `Job.agents()`, `Job.solve_rates()`, `Job.cost_usd`, `Trial.assessment` | Reading trials back; pass@1; leaving unscored trials out |
| `bf.run(bf.RolloutConfig(uploads=..., pre_agent_hooks=...))` | The optimizer, as a rollout of a task folder the demo writes |
| `bf.Task` | Task instructions, for the optimizer (train only) and for the leak check (test) |

The test [`tests/test_hillclimb_example.py`](../../../tests/test_hillclimb_example.py) checks that the demo uses no other BenchFlow names.

## Safeguards

| Safeguard from the post | How the demo enforces it |
|---|---|
| The optimizer never sees the test set | Its sandbox receives two folders and nothing else: `/hillclimb` (made root-owned and read-only before the agent starts) and `/app/surface`. They are built from train records only, and test results appear in them only as aggregate scores with intervals. The task sets `allow_internet: false`, so the optimizer cannot fetch the tasks elsewhere. Before each run, every uploaded path and file is checked for each test task's name and instruction text; the result and a sha256 manifest are recorded, and the report shows them in "What the optimizer saw". |
| Never paste failures into the prompt | The brief forbids it, and a patch that copies 12 or more words in a row from a train instruction or grader output is rejected without being evaluated. |
| Validate graders before trusting them | The oracle and do-nothing checks run first (step 2). |
| Check the plumbing | A trial without a reward (agent, provider, verifier or sandbox failure, or a trial that never ran) is left out of every score and counted by category. The climb stops when more than `--max-infra-error-rate` (0.25) of an evaluation's trials have no score, and a patch whose unscored trials rise is reverted, so it cannot look better by making hard tasks crash. |
| Watch variance | Every task runs `--trials` times, every score and delta has a bootstrap interval, and the noise gate refuses a `--min-gain` that noise alone could produce. |

What is not enforced: the optimizer sees the train split's grader output, so a train task's expected values can reach a patch; the pasted-text check catches long verbatim copies, and the test split catches what does not transfer. Test scores select patches, one comparison per round, so after many rounds the best version's test score is slightly optimistic; a fresh held-out set is the honest final check.

## Statistics

[`hillclimb_stats.py`](hillclimb_stats.py). A split's score is pass@1: the mean over tasks of each task's solve rate over its scored trials, the number `Job.solve_rates(ks=[1])` reports. Its interval is a two-stage bootstrap (draw tasks, then each drawn task's trials). A delta between two versions uses a paired bootstrap (draw tasks once, resample each side's trials), so differences between tasks cancel. The rerun noise is how far the score moves between two runs of the same version by chance: a bootstrap of a null difference, with each task's trials spread by `sqrt(n / (n - 1))` to undo the resampling bias. The noise gate requires `--min-gain` to be at least `1.96 x` the rerun noise's standard error on both splits, so a patch with no effect clears `--min-gain` on train by chance less than 2.5% of the time. Noise falls as one over the square root of trials times tasks, so a band `r` times too wide needs `r^2` times either; the refusal message says both. With one trial per task the noise cannot be measured, so the gate needs `--trials 2` or more.

## Run it

The recipe climbs a small office-files skill, [`office-skills/office-files`](office-skills/office-files/SKILL.md), on 20 office and spreadsheet tasks from SkillsBench ([`tasks.txt`](tasks.txt): Word, Excel, PowerPoint and PDF work from the `office-white-collar` and `finance-economics` categories). The skill replaces each task's bundled skills for the run. SkillsBench is pinned at commit `9a1f4dd5`, the first whose `task.md` files use this BenchFlow's `sandbox:` key; tag `v1.1` still says `environment:` and fails to parse here.

```bash
cd docs/examples/hillclimb
export ANTHROPIC_API_KEY=...        # both agents
export DAYTONA_API_KEY=...          # or: export SANDBOX=docker
./run.sh smoke    # 4 tasks, 2 trials, one forced round, a $10 cap
./run.sh climb    # 20 tasks (12 train, 8 test), 5 trials, --min-gain 0.15, 5 rounds, a $250 cap
```

`run.sh` fetches SkillsBench into `$WORK/skillsbench` (default `~/hillclimb-demo`) and writes each run to `$WORK/runs/<stage>-<timestamp>/`. The knobs are environment variables: `TRIALS`, `MIN_GAIN`, `TEST_FRAC`, `SEED`, `ROUNDS`, `MAX_COST_USD`, `CONCURRENCY`, `SANDBOX`, `AGENT_MODEL`, `PROPOSER_MODEL`, `WORK`, `SKILLSBENCH_SHA`. The script itself runs as `uv run python docs/examples/hillclimb/hillclimb.py --help` from a BenchFlow checkout.

With `SEED=7` and `TEST_FRAC=0.4` the split is:

- train (12): `citation-check`, `court-form-filling`, `econ-detrending-correlation`, `exceltable-in-ppt`, `financial-modeling-qa`, `organize-messy-files`, `paper-anonymizer`, `pdf-excel-diff`, `pptx-reference-formatting`, `reserves-at-risk-calc`, `sales-pivot-analysis`, `shock-analysis-demand`;
- test (8): `edit-pdf`, `invoice-fraud-detection`, `offer-letter-generator`, `powerlifting-coef-calc`, `sec-financial-report`, `shock-analysis-supply`, `weighted-gdp-calc`, `xlsx-recover-data`;
- smoke: train `court-form-filling`, `reserves-at-risk-calc`; test `offer-letter-generator`, `weighted-gdp-calc`.

All 20 tasks have an oracle, and none uses an LLM judge.

### What it needs

- **Accounts.** An Anthropic API key for both agents, read from the environment and passed to the provider only through BenchFlow's model proxy. The proxy also prices each rollout for the budget, so it must be a priced key: a subscription login reports no USD. A sandbox: Daytona with `CONCURRENCY` 40 or more, or a large Docker host.
- **Models.** `claude-haiku-4-5` for the agent under test and `claude-opus-4-8` for the optimizer. After the smoke run, check that `hillclimb.json`'s `cost.agent_usd` and `cost.proposer_usd` are above zero: a model missing from the pinned price table reports no USD, and the budget cannot count it.
- **Cost and time.** Each evaluation runs 20 tasks x 5 trials = 100 rollouts, and a climb runs at most six (the baseline and five rounds): 600 rollouts, plus up to six optimizer runs. At roughly $0.05 to $0.30 per Haiku rollout and $1 to $5 per Opus optimizer run, that is about $40 to $200; these are estimates, not measured on these tasks, and the $250 cap stops the climb before a round it cannot afford. With 40 or more Daytona sandboxes an evaluation takes about as long as its slowest task, 15 to 30 minutes, so a full climb takes 2 to 4 hours. The smoke run costs a few dollars.

It stops early, and cheaply, when something is wrong: exit 2 with "Refusing to climb" when the noise is above `--min-gain` (only the baseline was spent; raise `TRIALS` or `TEST_FRAC`, or `MIN_GAIN`), and exit 1 when too many trials had no score.

## Outputs

```
<out>/
  hillclimb.json      the record; hillclimb.schema.json in this folder is its schema
  report.html         the static report (no network requests)
  surfaces/vNNN/      every skills version, kept or not
  controls/oracle|nop/job/                 the grader checks, as BenchFlow jobs
  evals/<id>/<split>/trial-NN/job/         every evaluation (id: baseline, r01, r02, ...)
  proposer/<id>/      each optimizer run: workspace/ (exactly what was uploaded),
                      mounted.json (every uploaded file, sha256), task/, job/ (the rollout)
```

Every job folder opens with `bench eval view`, and `bench eval metrics evals/<id>/test` pools a split's trials (pass@k included). `hillclimb.json` and `report.html` are rewritten after every phase, so a running climb can be watched. The report shows the verdict, the score by round (the accepted version's train and test scores with 95% bands, and each round's candidate, filled when kept and hollow when reverted), the decisions with their deltas, what the optimizer saw, the stall analysis and every diff.

## Files

| File | What |
|---|---|
| `hillclimb.py` | The loop (`climb`, `decide`), the evaluations, the optimizer's rounds, `hillclimb.json`, the command line |
| `hillclimb_proposer.py` | The optimizer's sandbox: the train-only workspace, the mount check and manifest, the task it runs |
| `hillclimb_stats.py` | Bootstrap intervals, paired deltas, rerun noise, the noise gate |
| `hillclimb_report.py` | The HTML report |
| `hillclimb.schema.json` | The schema of `hillclimb.json` |
| `run.sh`, `tasks.txt`, `office-skills/` | The SkillsBench recipe |

## What was cut from the first prototype

The first version was a `bench hillclimb` command and a `benchflow.hillclimbing` package. Moving it here dropped what the post does not need: several candidate patches per round, the cost objective, a prompt file as the surface, task lists from several folders, the stratified split, the surface's git history, the pending and skipped candidate states, the pydantic record models, the CLI flag table and its parity tests, and the LLM-judge warning.

## Gaps in BenchFlow's public API

What the demo had to build itself, which a general primitive could provide:

- **Scores with confidence intervals.** `Job.solve_rates()` and `bf.compare()` give point estimates only (`compare` says it computes no significance), so the demo carries its own bootstrap (`hillclimb_stats.py`).
- **Trials of one configuration.** `Evaluation` runs each task once; repeated trials are one job per `trial-NN` folder, which the demo loops over (the CLI's `--matrix --trials` does the same).
- **Trials that never ran.** A budget stop leaves no `result.json`, so `bf.load_job` cannot count them; the demo compares against the planned tasks.
- **What a rollout received.** `RolloutConfig.uploads` leaves no record in the trial folder; the demo writes its own manifest.
- **Parallel Docker evaluations** in one process prune each other's just-created containers. The demo runs a Docker evaluation's jobs one after another until the fix on `fix/parallel-runs` lands.

## Limits

- The no-network optimizer (`allow_internet: false`, the default) has not yet run on a real sandbox: the Docker test runs it with the network open so that it can reach its scripted model on the host. The smoke run is its first real test.
- There is no resume: each `--out` is a new run.
- The budget is checked before each round and passed to every job as its cap; like `bench eval run --max-cost-usd`, it counts trials as they finish, so a round can overshoot by what its running trials spend.
- LLM-judged graders are not checked for consistency.
