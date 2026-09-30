# Demo: hill-climbing a skill with a held-out test split

This demo automates eval hill-climbing the way Lance Martin's post [Automating eval design and hillclimbing with Claude](https://claude.dev/blog/automating-eval-design-and-hillclimbing/) (claude.dev, 2026-09-28) describes, built only from BenchFlow's public primitives. An optimizer agent reads the failures on a train split and edits a skills folder once per round. The edit is kept only if the train score gains at least `--min-gain` and the score on a held-out test split also improves.

In the post, the optimizer is trusted to keep away from the test set. Here the test set is kept away by what the optimizer's sandbox is given: it runs as an ordinary BenchFlow rollout whose only uploads are train material, and every run records exactly what was mounted.

## The loop

`climb()` in [`hillclimb.py`](hillclimb.py) is the whole loop, 32 lines; `one_round()` (propose, evaluate, keep or revert) is 30 more and `decide()`, the keep-or-revert rule, 41.

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
| `bf.Evaluation` with `bf.EvaluationConfig(skills_dir=..., skill_mode="with-skill", include_tasks=..., config_override=...)` | Each split, once per trial, as a normal BenchFlow job with the skills deployed and a setup command that keeps Claude Code's session log (see [Cost](#cost)); `agent="oracle"` and `agent="nop"` for the grader checks |
| `bf.Budget(max_cost_usd=..., max_sandbox_seconds=...)`, `bf.RetryConfig` | Every job's share of the caps; retries of infrastructure errors |
| `bf.load_job`, `Job.agents()`, `Job.solve_rates()`, `Trial.assessment`, `Trial.cost_usd`, `Trial.timing` | Reading trials back; pass@1; leaving unscored trials out; cost and sandbox time |
| `bf.run(bf.RolloutConfig(uploads=..., pre_agent_hooks=...))` | The optimizer, as a rollout of a task folder the demo writes (its `artifacts:` keep Claude Code's session log) |
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
export CLAUDE_CODE_OAUTH_TOKEN=...  # both agents: a Claude subscription (claude setup-token), or ANTHROPIC_API_KEY
export DAYTONA_API_KEY=...          # or: export SANDBOX=docker
export BENCHFLOW_DAYTONA_OWNER=hillclimb-demo   # labels the sandboxes, for `bench sandbox list` and `cleanup`
./run.sh smoke    # 4 tasks, 2 trials, one forced round; caps: $10, 20 model rollouts, 6 sandbox-hours
./run.sh climb    # 20 tasks (12 train, 8 test), 5 trials, --min-gain 0.15, 5 rounds; caps: $250, 610 rollouts, 150 sandbox-hours
```

`run.sh` fetches SkillsBench into `$WORK/skillsbench` (default `~/hillclimb-demo`) and writes each run to `$WORK/runs/<stage>-<timestamp>/`. The knobs are environment variables: `TRIALS`, `MIN_GAIN`, `TEST_FRAC`, `SEED`, `ROUNDS`, `MAX_COST_USD`, `MAX_ROLLOUTS`, `MAX_SANDBOX_HOURS`, `CONCURRENCY`, `SANDBOX`, `AGENT_MODEL`, `PROPOSER_MODEL`, `WORK`, `SKILLSBENCH_SHA`. The script itself runs as `uv run python docs/examples/hillclimb/hillclimb.py --help` from a BenchFlow checkout.

With `SEED=7` and `TEST_FRAC=0.4` the split is:

- train (12): `citation-check`, `court-form-filling`, `econ-detrending-correlation`, `exceltable-in-ppt`, `financial-modeling-qa`, `organize-messy-files`, `paper-anonymizer`, `pdf-excel-diff`, `pptx-reference-formatting`, `reserves-at-risk-calc`, `sales-pivot-analysis`, `shock-analysis-demand`;
- test (8): `edit-pdf`, `invoice-fraud-detection`, `offer-letter-generator`, `powerlifting-coef-calc`, `sec-financial-report`, `shock-analysis-supply`, `weighted-gdp-calc`, `xlsx-recover-data`;
- smoke: train `court-form-filling`, `reserves-at-risk-calc`; test `offer-letter-generator`, `weighted-gdp-calc`.

All 20 tasks have an oracle, and none uses an LLM judge.

### What it needs

- **Accounts.** Claude credentials for both agents, from the environment: a subscription token from `claude setup-token` (`CLAUDE_CODE_OAUTH_TOKEN`), which Claude Code uses directly, or an API key, which reaches the provider only through BenchFlow's model proxy. An API key overrides the token when both are set. A sandbox: Daytona with `CONCURRENCY` 40 or more, or a large Docker host.
- **Models.** `claude-haiku-4-5-20251001` for the agent under test (the id BenchFlow's docs use for `claude-agent-acp`) and `claude-opus-5-5` for the optimizer (it needs Claude Code 2.1.280, which BenchFlow pins apart from the ACP adapter). The adapter maps them to its `haiku` and `opus` picker rows. After the smoke run, check `hillclimb.json`'s `cost`: `source` should not be `unknown`, and `context_1m` should be false. If an account's picker maps `claude-opus-5-5` to its 1M-context row (`opus[1m]`), pass `--proposer-model-env`, which gives Claude Code the model as `ANTHROPIC_MODEL` instead of through the picker.
- **Cost and time.** Each evaluation runs 20 tasks x 5 trials = 100 rollouts, and a climb runs at most six (the baseline and five rounds): 600 rollouts, plus up to six optimizer runs. At roughly $0.05 to $0.30 per Haiku rollout and $1 to $5 per Opus optimizer run, that is about $40 to $200 (list-price equivalents under a subscription); these are estimates, not measured on these tasks, and the caps stop the climb before a round it cannot afford. With 40 or more Daytona sandboxes an evaluation takes about as long as its slowest task, 15 to 30 minutes, so a full climb takes 2 to 4 hours; a trial holds a sandbox for about 5 to 12 minutes, so the 600 rollouts use about 50 to 120 sandbox-hours, under the 150-hour cap. The smoke run costs a few dollars.

It stops early, and cheaply, when something is wrong: exit 2 with "Refusing to climb" when the noise is above `--min-gain` (only the baseline was spent; raise `TRIALS` or `TEST_FRAC`, or `MIN_GAIN`), and exit 1 when too many trials had no score or a cap would be passed.

### Cost

With a subscription, Claude Code calls the Anthropic API itself, so BenchFlow records each trial's tokens but no USD, and a `bf.Budget`'s `max_cost_usd` cannot count it. Claude Code keeps its own session log (`~/.claude/projects/<cwd>/<session>.jsonl`, subagents in `<session>/subagents/`), whose `cost-state` lines hold its running `totalCostUSD` and per-model `modelUsage` with `costUSD`, and whose responses hold their token usage. The demo puts that log into every trial folder, at `artifacts/claude-sessions/`, and prices the trial from it ([`hillclimb_cost.py`](hillclimb_cost.py)):

- **Getting the log.** An evaluation trial gets a task setup command (`EvaluationConfig.config_override`) that links `/root/.claude/projects` to `/logs/artifacts/claude-sessions` before the agent is installed. BenchFlow copies `/root/.claude` into the sandbox user's home, so Claude Code writes its log into `/logs/artifacts`, which BenchFlow collects into the trial folder. The optimizer's task declares its sandbox user's `~/.claude/projects` as an artifact. Any credential from the environment is replaced in the collected logs.
- **Pricing.** Claude Code's own `totalCostUSD` when a `cost-state` line counts every response in the log (source `claude-code-cost-state`); otherwise the responses' usage at Anthropic's list prices (`claude-code-usage-at-list-price`; on a real session log these reproduce Claude Code's `costUSD` for `claude-opus-5-5` within 0.3%). A trial without a log keeps BenchFlow's `cost_usd` (`benchflow`: an API key priced by the proxy), or is `unknown`.
- **Where it shows.** `hillclimb.json` has `cost.source` and `cost.sources` (rollouts per source) for the run, `cost_source` and `cost_sources` for every split of every evaluation, and `cost_source` for every optimizer run; the report says it under the tiles.

Three caps stop the climb before a step it cannot afford, and every job gets its share of the first two as a `bf.Budget`:

| Cap | Counts | Binds without USD |
|---|---|---|
| `--max-cost-usd` | USD from the sources above; a job's `bf.Budget` counts only BenchFlow's own USD | between steps only |
| `--max-sandbox-seconds` | sandbox wall-clock of every trial and optimizer run, the grader checks included | yes, inside every job too |
| `--max-rollouts` | rollouts that call a model: evaluation trials and optimizer runs | yes, between steps (`bf.Budget` has no rollout count) |

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
| `hillclimb_cost.py` | What each rollout cost, from Claude Code's session log |
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
- **USD under a subscription.** A native `claude-agent-acp` run records tokens but `cost_usd` null, although the adapter sends the SDK's `total_cost_usd` in every `usage_update` notification: BenchFlow's ACP session drops that update type. The demo recovers the cost from Claude Code's session log.
- **A rollout count in `bf.Budget`.** It caps USD, sandbox-seconds and tokens; the demo checks its rollout cap between steps.
- **Artifacts or hooks for an `Evaluation`.** `EvaluationConfig` takes no pre-agent hooks, and `config_override` may not declare `artifacts:` (only `agent`, `sandbox` and `metadata` are patchable), so the session log reaches the trial folder through a setup command and BenchFlow's copy of `/root/.claude` into the sandbox user's home, which is an implementation detail.
- **Artifacts of a rollout that errored.** Artifacts are collected when the verifier starts, which an agent error skips, so a trial whose agent crashed keeps no session log, and its cost is unknown.
- **Retried attempts.** `bf.load_job` returns each task's last attempt; the attempts a `bf.RetryConfig` retried are folders it does not expose, so the rollout cap counts trials, not attempts (the sandbox-seconds come from each job's budget, which counts them).
- **A usage limit is an ordinary ACP error.** When a subscription's limit is reached, the claude-agent-acp prompt fails with a generic `-32603` ("You've hit your weekly limit · resets ..."), recorded as `acp_error`, and the job retries it though no retry can succeed before the reset. The adapter has a typed `quota_exhausted` failure for clients that opt in to it; BenchFlow's ACP client does not.

## Limits

- The no-network optimizer (`allow_internet: false`, the default) has reached the model on Daytona under a subscription: BenchFlow's native no-web proxy admitted Claude Code 2.1.280 and passed its `POST /v1/messages` to `api.anthropic.com` (it blocked two other requests). It has not yet finished a real proposal; the Docker test runs it with the network open so that it can reach its scripted model on the host.
- There is no resume: each `--out` is a new run.
- The caps are checked before each step, and each job gets its share of what is left; like `bench eval run --max-cost-usd`, USD is counted as trials finish, so a round can overshoot the USD cap by what its running trials spend. The sandbox-seconds cap also counts running trials.
- LLM-judged graders are not checked for consistency.
