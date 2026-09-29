# Demo: hill-climb an office-files skill on SkillsBench

This recipe runs [`bench hillclimb`](../../hillclimb.md) on 20 office and spreadsheet tasks from SkillsBench (Word, Excel, PowerPoint and PDF work, from the `office-white-collar` and `finance-economics` categories), pinned at commit `9a1f4dd5`, the first whose `task.md` files use this BenchFlow's `sandbox:` key. Tag `v1.1` still says `environment:`, targets BenchFlow below 0.7, and fails to parse here. The surface is one small skill, [`office-skills/office-files`](office-skills/office-files/SKILL.md), deployed to every task in place of the tasks' own bundled skills. An optimizer agent edits that skill from the train split's failures; each edit is kept only if the train score gains at least `--min-gain` and the held-out test split improves. The optimizer's sandbox never holds the test split, and the report shows what each optimizer run was given.

## What it needs

- **BenchFlow from this branch** (`bench` on `PATH`), git, and bash.
- **A sandbox.** Daytona (`DAYTONA_API_KEY`) with `--concurrency 40` or more, or Docker on a large host (`SANDBOX=docker`, concurrency about the number of CPUs divided by 2). The LibreOffice tasks' images are roughly 1 to 2 GB each; the heavy SkillsBench tasks (`latex-formula-extraction`, OCR) are left out.
- **`ANTHROPIC_API_KEY`**, read from the environment by both agents and passed to the provider only through BenchFlow's model proxy, which also records each rollout's cost for the budget. Keys never go on the command line. The agent under test defaults to `claude-haiku-4-5` (not saturated on these tasks) and the optimizer to `claude-opus-4-8`; set `AGENT_MODEL` and `PROPOSER_MODEL` to change them. Check after the smoke run that `cost.usd_unknown_rollouts` in `hillclimb.json` is 0: a model missing from the pinned LiteLLM price table reports no USD, and the budget cannot count it.

## Commands

```bash
cd docs/examples/hillclimb
export ANTHROPIC_API_KEY=...        # both agents
export DAYTONA_API_KEY=...          # or: export SANDBOX=docker

# 1. Smoke: 4 tasks, 2 trials, one forced round, a $10 cap. Checks keys,
#    sandboxes, the grader checks and one optimizer run end to end.
./run.sh smoke

# 2. The demo: 20 tasks, 12 train and 8 test (TEST_FRAC=0.4, stratified by
#    category, SEED=7), 5 trials per task, --min-gain 0.15, 5 rounds, a $250 cap.
./run.sh climb
```

With `SEED=7` and `TEST_FRAC=0.4` the split is (checked against the pinned tasks):

- train (12): `citation-check`, `court-form-filling`, `econ-detrending-correlation`, `exceltable-in-ppt`, `financial-modeling-qa`, `paper-anonymizer`, `powerlifting-coef-calc`, `pptx-reference-formatting`, `sales-pivot-analysis`, `sec-financial-report`, `shock-analysis-supply`, `xlsx-recover-data`;
- test (8): `edit-pdf`, `invoice-fraud-detection`, `offer-letter-generator`, `organize-messy-files`, `pdf-excel-diff`, `reserves-at-risk-calc`, `shock-analysis-demand`, `weighted-gdp-calc`;
- smoke: train `court-form-filling`, `reserves-at-risk-calc`; test `offer-letter-generator`, `weighted-gdp-calc`.

All 20 tasks have an oracle, and none uses an LLM judge.

`run.sh` fetches SkillsBench at `9a1f4dd5` into `$WORK/skillsbench` (default `~/hillclimb-demo`) and writes each run to `$WORK/runs/<stage>-<timestamp>/`. The knobs are environment variables: `TRIALS`, `MIN_GAIN`, `TEST_FRAC`, `SEED`, `ROUNDS`, `MAX_COST_USD`, `CONCURRENCY`, `SANDBOX`, `AGENT_MODEL`, `PROPOSER_MODEL`, `WORK`, `SKILLSBENCH_SHA`.

The climb stops early, and cheaply, when something is wrong:

- **Exit 2, "Refusing to climb"**: after the baseline, the noise on train or test is above `--min-gain`. The message names the `--trials`, task count or `--min-gain` that would pass. Test-split noise is usually the binding one: with 8 test tasks and 5 trials, expect a 95% noise band around 0.1 to 0.2. Raise `TRIALS`, or `TEST_FRAC` (more test tasks), or `MIN_GAIN`, and run again; only the baseline was spent.
- **Exit 1, infrastructure errors**: more than 25% of an evaluation's trials had no score (sandbox, provider, or verifier failures). Their categories are in the message and in `hillclimb.json`.
- **Grader checks**: tasks whose oracle does not pass, or where doing nothing passes, are warned about and, with `--exclude-broken-tasks` (on in `climb`), dropped before the baseline.

## Cost and time

Each evaluation runs every task `TRIALS` times: 20 x 5 = 100 rollouts of the agent under test. A climb is one baseline plus one evaluation per round (five rounds at most), so at most 600 rollouts, plus one optimizer run per round and one analysis run after a stall. At roughly $0.05 to $0.30 per Haiku rollout on these tasks, and $1 to $5 per Opus optimizer run, expect about $40 to $200; the $250 cap stops it before a round it cannot afford. With 40 or more Daytona sandboxes an evaluation takes about as long as the slowest task (15 to 30 minutes, image builds included), so a full climb takes about 2 to 4 hours. The smoke run costs a few dollars.

## Reading the result

- `report.html`: the verdict (the best version against the baseline on the test split, with a 95% interval, and whether the gain exceeds noise), the score curve by round, every decision with its train and test deltas, what each optimizer run was given ("What the optimizer saw": two mounts, zero test tasks), the stall analysis, and every diff. It is one file with no network requests.
- `hillclimb.json`: the same, as data (schema: `docs/reference/schemas/benchflow-hillclimb.v1.schema.json`).
- `surface-history/`: `git log -p` there shows the skill's kept edits, one commit each.
- `evals/<id>/<split>/trial-NN/job/`: every trial, as normal BenchFlow jobs (`bench eval view`, `bench eval metrics evals/<id>/test`).
- `proposer/<id>/`: each optimizer run's workspace (exactly what was uploaded), `mounted.json` (every uploaded file with its sha256), the generated task and the rollout.

## The tasks

[`tasks.txt`](tasks.txt): `citation-check`, `court-form-filling`, `econ-detrending-correlation`, `edit-pdf`, `exceltable-in-ppt`, `financial-modeling-qa`, `invoice-fraud-detection`, `offer-letter-generator`, `organize-messy-files`, `paper-anonymizer`, `pdf-excel-diff`, `powerlifting-coef-calc`, `pptx-reference-formatting`, `reserves-at-risk-calc`, `sales-pivot-analysis`, `sec-financial-report`, `shock-analysis-demand`, `shock-analysis-supply`, `weighted-gdp-calc`, `xlsx-recover-data`. They share file formats and tools, so one skill can plausibly help across them, and the held-out ones tell whether it does.
