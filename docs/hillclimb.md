# bench hillclimb: eval hill-climbing with a held-out test split

`bench hillclimb` improves a *surface* the agent under test receives, a skills folder or a prompt, one targeted patch at a time. An optimizer agent reads the failures on a train split and proposes a patch. The patch is kept only if the train score gains at least `--min-gain` and the score on a held-out test split also improves. The optimizer runs in a sandbox that never holds the test split.

The loop follows Lance Martin's post [Automating eval design and hillclimbing with Claude](https://claude.dev/blog/automating-eval-design-and-hillclimbing/) (claude.dev, 2026-09-28). In the post, the optimizer is trusted to keep away from the test set. Here the runtime enforces it: the optimizer's sandbox only ever receives train data and aggregate test scores, and every run records exactly what was mounted.

```bash
bench hillclimb \
  --tasks-dir tasks/ --surface skills/ --out jobs/hillclimb/demo \
  --agent claude-agent-acp --model claude-haiku-4-5 --sandbox daytona --concurrency 32 \
  --proposer-model claude-opus-4-8 \
  --trials 3 --min-gain 0.1 --rounds 5 --max-cost-usd 150
```

Both agents read provider keys from the environment or `.env`, as `bench eval run` does; `--agent-env` and `--proposer-env` set them per agent (for example, a separate key for the optimizer).

```python
import benchflow as bf

result = bf.hillclimb(
    tasks="tasks/", surface="skills/", out="jobs/hillclimb/demo",
    agent="claude-agent-acp", model="claude-haiku-4-5", environment="daytona",
    trials=3, min_gain=0.1, rounds=5, max_cost_usd=150,
    proposer=bf.hillclimbing.ProposerSettings(model="claude-opus-4-8", environment="daytona"),
)
print(result.verdict)       # the test split's verdict, with its 95% interval
result.best_surface         # the folder of the best version
result.report               # report.html
```

`bf.ahillclimb` is the async form. A runnable demo on SkillsBench is in [examples/hillclimb](examples/hillclimb/).

## The loop

1. **Split.** The tasks are split at random into train and test, with a seed, stratified by a task metadata key (`--stratify-by`, `category` by default; `--test-frac` 0.3 by default). `--split-file` gives the split instead. The split is saved as `split.json`, which is itself a valid `--split-file`.
2. **Check the graders.** Each task runs once with its own solution (the oracle) and once with an agent that does nothing (`nop`). A task whose oracle does not pass, or where doing nothing passes, has a grader bug; it is listed as a warning, and `--exclude-broken-tasks` drops it from the split.
3. **Baseline and noise gate.** The starting surface runs `--trials` times per task on both splits. If `--min-gain` is not above the noise (see [Statistics](#statistics)), the climb refuses to start and says what `--trials`, task count or `--min-gain` would fix it. `--force` climbs anyway and marks the run ungated.
4. **Propose.** An optimizer agent (Claude Code by default) runs as an ordinary BenchFlow rollout in a sandbox. It reads the current version's failed train trials, the train tasks' instructions, the scores, and the earlier proposals, and edits the surface once, aimed at one root cause. It writes a `proposal.json` with the root cause, the change and its rationale.
5. **Evaluate and decide.** The patched surface runs on train and test. The keep rule is the post's:
   - keep only if the train gain is at least `--min-gain` and the test score improved;
   - revert if the test score is flat (train up and test flat means overfitting) or lower;
   - revert on any regression, and when infrastructure errors rise (a patch must not look better by making hard trials crash).
6. **Stall.** After `--stall-rounds` rounds (default 3) with nothing kept, the climb stops, and the optimizer runs once more in analysis mode. It sorts each remaining train failure into `ambiguous_task`, `grader_bug`, `infrastructure` or `capability_gap`, and the analysis goes into the report.
7. **Report.** The best version is the last one kept; every keep required a higher test score, so it is also the best on test. It is reported against the baseline on the test split with a 95% confidence interval and a verdict: the gain exceeds noise when its interval is above zero, and otherwise the report says it is within noise and recommends against merging.

`--candidates N` asks for N independent patches per round from the same base; each is evaluated, and the one with the largest train gain among those that pass the keep rule is kept. Candidates are evaluated one after another: on Docker, a skills surface is baked into the task's image, so two versions of one task must not build at once.

## Safeguards

| Safeguard from the post | How it is enforced |
|---|---|
| The optimizer never sees the test set | The optimizer's sandbox receives two folders and nothing else: the evidence at `/hillclimb` (root-owned and read-only before the agent starts) and the surface at `/app/surface`. The evidence is built from the train split's records only; test results appear in it only as aggregate scores with intervals. The sandbox has no network access (`allow_internet: false`), so the optimizer cannot fetch public copies of the tasks. Before each upload, every mounted path and file is checked for each test task's name and instruction text, and the result is recorded (`exposure` in `hillclimb.json`, "What the optimizer saw" in the report). |
| Never paste failures into the prompt | The brief forbids it, and each patch is checked for runs of 12 or more words copied from the train instructions or grader output the optimizer read. A patch that copies is rejected without being evaluated (`--leak-check reject`, the default; `warn` or `off` to relax). |
| Validate graders before trusting them | The oracle and a do-nothing agent run on every task first (step 2). |
| Check the plumbing | A trial without a reward (agent integration failure, provider error, verifier crash, sandbox failure, or a trial missing after a budget stop) is an infrastructure error: left out of every score and counted by category. The climb stops when an evaluation's share of them passes `--max-infra-error-rate` (default 0.25), and a patch whose errors rise is reverted. |
| Watch variance | Every trial count is at least `--trials` per task, every score and delta has a bootstrap interval, and the noise gate refuses a `--min-gain` the noise could produce. |
| Report against the baseline with confidence intervals | The verdict compares the best version with the baseline on the test split, with a paired 95% interval. |

What is not enforced: the optimizer still sees the train split's grader output, so a train task's expected values can reach a patch; the pasted-text check catches long verbatim copies, and the test split catches what does not transfer. Test scores do select patches, one comparison per candidate, so after many rounds the best version's test score is slightly optimistic; a fresh held-out set is the honest final check.

LLM-judged tasks: their grader consistency is not measured yet. The run warns when a task's verifier is an LLM judge. The planned check freezes the baseline's workspaces (`--freeze-workspace`) and regrades a sample twice with `bf.regrade` to report a flip rate.

## Statistics

A split's **score** is the mean over its tasks of each task's mean reward over its scored trials, so a task weighs the same however many of its trials errored.

- **Standard error and interval** of a score: a two-stage bootstrap. Each replicate draws the tasks with replacement, then each drawn task's trials with replacement.
- **Delta** between two versions: the same bootstrap, paired. A replicate draws the tasks once and resamples each side's trials of those tasks, so differences between tasks cancel.
- **Rerun noise**: how far the score moves between two runs of the same surface on the same tasks, by chance alone. A replicate draws the tasks, then two independent resamples of each task's trials, and takes the difference. With few trials a resample underestimates the spread by `(n - 1) / n`, so each task's trials are first spread from their mean by `sqrt(n / (n - 1))`.
- **The noise gate** requires `--min-gain` to be at least `1.96 x` the rerun noise's standard error on both splits: a patch with no effect then clears `--min-gain` on train by chance less than 2.5% of the time, and the test split can tell a real gain from a lucky one. It needs `--trials 2` or more: with one trial per task, trial-to-trial noise is invisible. Noise falls as one over the square root of trials times tasks, so a noise band `r` times too wide needs `r^2` times the trials or the tasks; the refusal message says both.

## The surface

`--surface` is repeatable. A directory is a skills folder; a file is a prompt. Prefix `skills=` or `prompt=` to be explicit.

- **Skills folder**: each subfolder holds a `SKILL.md` with YAML frontmatter. It is deployed like `bench eval run --skills-dir`: mounted at `/skills`, linked into each agent's skill paths, and it replaces the task's own bundled skills for the run.
- **Prompt file**: prepended to every task prompt through the task's `agent.prompt_prefix`, set per run with the config overlay that `bench eval run --config-override` uses (merged with your own `--config-override`).

Each version is a folder `surfaces/vNNN/` holding `skills/` and/or `prompt.md`. An edited version drops symlinks, must keep every skill well-formed, and is capped at 500 files and 5 MB.

## Cost as the objective

`--objective cost` cuts cost at equal performance, the post's other use: `--min-gain` is then a fraction of the cost per trial (0.1 = 10% cheaper). A patch is kept only if train cost falls by at least that fraction, test cost falls too, and neither split's score drops by more than its noise band. The cost is the agent under test's USD per scored trial, so it needs priced usage (an API key through BenchFlow's model proxy); a subscription login reports no USD. The verdict requires the test cost cut's 95% interval to be below zero and the test score to hold within noise.

## Outputs

```
<out>/
  hillclimb.json          the record (schema below)
  report.html             the static report
  split.json              the split
  surface-history/        a git repository: the baseline, then one commit per kept patch
  surfaces/vNNN/          every version, kept or not
  controls/oracle|nop/    the grader checks, as BenchFlow jobs
  evals/<id>/<split>/trial-NN/job/<task>__<id>/   every evaluation (id: baseline, r01-c1, ...)
  proposer/<id>/          each optimizer run: workspace/ (exactly what was uploaded),
                          mounted.json (every uploaded file with its sha256),
                          task/ (the generated task), job/ (the rollout)
```

Each `trial-NN` folder is an ordinary `Evaluation` job: `bench eval view evals/r01-c1/train/trial-01/job` opens it, and `bench eval metrics evals/r01-c1/train` pools its trials (pass@k included). The optimizer runs are ordinary trials too.

`hillclimb.json` (`kind: benchflow.hillclimb`, `schema_version: 1`) holds the configuration, the split, the grader checks, the baseline, the noise gate, every round's candidates (proposal, diff, what the optimizer saw, train and test scores and deltas with intervals, decision and reasons), the best version with its verdict, the stall analysis, the cost, and the reason the climb stopped. Its JSON Schema is `docs/reference/schemas/benchflow-hillclimb.v1.schema.json`; a new optional field keeps the version, so the schema allows fields it does not list. The file is rewritten after every phase, and so is `report.html`, so a running climb can be watched.

`report.html` is one self-contained page (no network requests): the verdict, the score curve by round (the accepted version's train and test scores with 95% bands, and each candidate, filled when kept and hollow when reverted), the decisions, the setup, what the optimizer saw, the stall analysis and every diff.

Exit codes: 0 when the climb finished (or a budget stop left a best version), 2 when the noise gate refused, 1 when infrastructure errors stopped it or a setting was invalid.

## Flags

Each flag is a field of `bf.HillclimbConfig`; the proposer's are fields of `bf.hillclimbing.ProposerSettings`.

| Flag | Default | Meaning |
|---|---|---|
| `--tasks-dir` | | A folder of task folders (or a `tasks/` inside it) |
| `--task` | | One task folder; repeatable, instead of `--tasks-dir` |
| `--surface` | required | What the optimizer may edit; repeatable (see [The surface](#the-surface)) |
| `--out` | `jobs/hillclimb/<timestamp>` | The run folder; a folder that already holds a run is refused |
| `--agent` | `claude-agent-acp` | The agent under test |
| `--model` | the agent's default | Its model |
| `--reasoning-effort` | | Its reasoning effort |
| `--sandbox` | `docker` | The sandbox for the agent under test and, by default, the optimizer |
| `--concurrency` | 4 | Rollouts at once, shared by an evaluation's splits and trials (at least one each) |
| `--agent-env` | | `KEY=VALUE` for the agent under test; repeatable |
| `--sandbox-user` | `agent` | The sandbox user (`none` for root) |
| `--agent-idle-timeout` | 600 | Idle seconds before a prompt is aborted (`0` or `none` disables) |
| `--retry-attempts` | 2 | Retries of a trial after an infrastructure error |
| `--config-override` | | A config overlay for every trial, as in `bench eval run` |
| `--include` / `--exclude` | | Task names to keep or skip; repeatable |
| `--test-frac` | 0.3 | The test split's share of the tasks |
| `--seed` | 0 | Seed of the split and the bootstrap |
| `--split-file` | | `{"train": [...], "test": [...]}` instead of a random split |
| `--stratify-by` | `category` | Task metadata key the split is stratified by (`none`: off) |
| `--objective` | `score` | `score`, or `cost` (cut cost while the score holds within noise) |
| `--rounds` | 5 | Rounds of propose and evaluate |
| `--trials` | 3 | Trials per task per evaluation |
| `--min-gain` | 0.05 | The smallest train gain worth keeping: reward points, or a fraction of cost |
| `--max-cost-usd` | | Stop before spending more (agent under test plus optimizer), checked between phases and as each job's cap |
| `--stall-rounds` | 3 | Rounds with nothing kept before the climb stops and analyzes |
| `--candidates` | 1 | Patches per round from the same base |
| `--max-infra-error-rate` | 0.25 | Stop when more of an evaluation's trials than this have no score |
| `--skip-controls` | off | Skip the oracle and do-nothing grader checks |
| `--exclude-broken-tasks` | off | Drop tasks whose oracle fails or where doing nothing passes |
| `--force` | off | Climb although the noise gate refused (the run is marked ungated) |
| `--leak-check` | `reject` | Patches that copy train instructions or grader output: `reject`, `warn` or `off` |
| `--bootstrap-samples` | 2000 | Bootstrap replicates |
| `--analyze-at-end` | off | Also analyze the remaining failures when the rounds run out |
| `--proposer-agent` | `claude-agent-acp` | The optimizer agent |
| `--proposer-model` | the agent's default | Its model |
| `--proposer-reasoning-effort` | | Its reasoning effort |
| `--proposer-sandbox` | `--sandbox` | Its sandbox |
| `--proposer-env` | | `KEY=VALUE` for the optimizer; repeatable |
| `--proposer-timeout` | 1800 | Its time limit in seconds |
| `--proposer-image` | pinned Python image | Its sandbox image (the rubric reviewer's) |
| `--proposer-open-network` | off | Give it network access; it could then fetch public copies of the test tasks |
| `--proposer-max-failures` | 24 | Failed train trials it is shown per round, one per task first |
| `--quiet` | off | No per-rollout progress output |

## Limits of the prototype

- The surface is files only: a skills folder and a prompt. Model choice and effort, which the post's cost example changed, are not a surface yet.
- A run cannot be resumed; each `--out` is a new run. An interrupted evaluation's finished trials are kept in its job folders.
- The USD budget is checked between phases and passed to every job as its cap; like `bench eval run --max-cost-usd`, it counts trials as they finish, so a phase can overshoot by what its running trials spend.
- LLM-judge consistency is not measured (see above).
