# Reproduce a published score

A published score is a claim about particular trials, run with particular settings and averaged a particular way. To check one: get the trials, recompute the number from them, write down the settings they ran with, rerun a matched sample, and compare with an interval. The example checks a SkillsBench leaderboard entry: OpenHands on DeepSeek V4 Flash, without skills, **27.5 ± 7.4**.

## Get the published trials

Start from the number's own definition. The leaderboard's code, [`website/src/data/leaderboard-data.ts`](https://github.com/benchflow-ai/skillsbench/blob/f860ecf02ef53412a030f8661c83d644b739ff01/website/src/data/leaderboard-data.ts#L136-L137) at SkillsBench commit f860ecf0, says how it is computed:

```ts
// Point estimates are task-macro means of per-task mean rewards (x100).
// CI: 95% task-level (Method D) — 1.96 * stdev(per-task means) / sqrt(87).
```

SkillsBench publishes its trials on Hugging Face, in the dataset [`benchflow/skillsbench-leaderboard`](https://huggingface.co/datasets/benchflow/skillsbench-leaderboard). [`skillsbench_no_skills.py`](../examples/reproduce/skillsbench_no_skills.py) downloads each trial's `result.json` at a fixed revision of the dataset, picks the trials, and prints the score both ways; [`skillsbench-v1.1-tasks.txt`](../examples/reproduce/skillsbench-v1.1-tasks.txt) lists the 87 tasks:

```bash
base=https://raw.githubusercontent.com/benchflow-ai/benchflow/main/docs/examples/reproduce
curl -fsSLO $base/skillsbench_no_skills.py && curl -fsSLO $base/skillsbench-v1.1-tasks.txt
uv run --with huggingface_hub python3 skillsbench_no_skills.py skillsbench-v1.1-tasks.txt
```

```text
430 trials in 8 folders at f1045803
published rule, copies counted: 27.490 +- 7.417 over 87 tasks
same rule, copies removed: 27.107 +- 7.364 over 87 tasks
```

That's the published 27.5 ± 7.4, and all 87 per-task scores match the leaderboard's own table, but only with a rule the leaderboard doesn't fully state: every `openhands-no-skills__*v4-flash*` folder except `openhands-no-skills__deepseek-ds-v4-flash-src-411dc68`; per task, the latest 3 trials that have a reward and no error; empty slots counted as 0. Along the way:

- 9 trials appear twice and are counted twice; without the copies the score is 27.1.
- The one folder left out is the only one whose metadata names the model `deepseek/ds-v4-flash`; keeping it gives the same 27.49, with ± 7.5.
- The leaderboard's text says "earliest 3" trials; the data match the latest 3.
- The score is a mean of rewards, some fractional (0.55), not a pass rate: 24.5% of the trials scored 1.

A number that recomputes is the first check. How it was assembled is the second, and it belongs in any comparison.

## Write down the settings

Next, read each trial's `config.json` and `result.json` for what it ran with. For the 259 trials behind this number:

| Setting | What the trials say |
|---|---|
| BenchFlow version | not recorded |
| OpenHands version | not recorded; it was installed unpinned |
| Model route | 172 trials straight to DeepSeek's API, 87 through a proxy |
| Reasoning effort | `max` on 169, unset on 90; whether it reached the model isn't recorded |
| Agent time limit | 300 seconds to 7 days |
| Task version | five commits from May 5 to June 8, all before the release tag the leaderboard names |

Each gap is something a rerun can't match. BenchFlow 0.9.1 records its own version, the model, the effort asked for and the limits in every trial's `config.json`, the agent's installed version in `agent/install-stdout.txt`, and each request as sent in `trajectory/llm_trajectory.jsonl`; 0.10.0rc1's [run card](../evaluate/before-you-run.md#the-run-card-010) puts them in one file; the [report checklist](./report.md#what-a-result-needs) lists what to publish so others can do this to your number.

## Match them in a run

Rerun a sample of tasks with the settings matched as closely as the release allows. Five tasks, three tries each, without skills, as published:

```bash
bench run skillsbench/tasks \
  -i 3d-scan-calc -i citation-check -i crystallographic-wyckoff-position-analysis \
  -i econ-detrending-correlation -i offer-letter-generator \
  -a openhands -m deepseek/deepseek-v4-flash --agent-env LLM_REASONING_EFFORT=high \
  --trials 3 -j 2 --max-cost-usd 3
```

`skillsbench/` is a clone of SkillsBench at commit f860ecf0 ([how](../evaluate/run-a-benchmark.md#run-three-tasks-three-times-each)).

The published effort couldn't be matched. `-e max` is refused for OpenHands:

```text
RuntimeError: reasoning_effort='max' was requested for agent 'openhands', but that agent does not declare or advertise an ACP effort config option
```

`--agent-env LLM_REASONING_EFFORT=max` is rejected by BenchFlow's model proxy (`request outside this run's protocol allowlist: unsupported fields reasoning`), and `high` is accepted but never sent: none of the run's 221 model requests carries an effort, so DeepSeek used its default. The run's 15 trials all scored, in 19.6 minutes, for $0.20.

## Compare with an interval

Compare each task with the same task's published trials, not with the 87-task headline:

| Task | Published | Rerun | Difference |
|---|---|---|---|
| 3d-scan-calc | 33.3 | 100.0 | +66.7 |
| citation-check | 100.0 | 100.0 | 0 |
| crystallographic-wyckoff-position-analysis | 58.0 | 55.0 | −3.0 |
| econ-detrending-correlation | 100.0 | 100.0 | 0 |
| offer-letter-generator | 0 | 100.0 | +100.0 |
| **mean of the five** | **58.3** | **91.0** | **+32.7** |

[`compare_five.py`](../examples/reproduce/compare_five.py) holds the 30 trial rewards behind this table and computes the comparison three ways (Python 3, nothing to install). Given a JSON file of your own, `{"task": [[published rewards], [rerun rewards]], ...}`, it compares those instead:

```bash
curl -fsSLO https://raw.githubusercontent.com/benchflow-ai/benchflow/main/docs/examples/reproduce/compare_five.py
python3 compare_five.py
```

```text
3d-scan-calc                                 published  33.3  rerun 100.0  difference  +66.7
citation-check                               published 100.0  rerun 100.0  difference   +0.0
crystallographic-wyckoff-position-analysis   published  58.0  rerun  55.0  difference   -3.0
econ-detrending-correlation                  published 100.0  rerun 100.0  difference   +0.0
offer-letter-generator                       published   0.0  rerun 100.0  difference +100.0

mean difference +32.7 points; 95% t interval -26.5 to +91.9
sign-flip p = 0.500 (the smallest possible with 5 tasks: 0.0625)
within-task permutation p = 0.020 (3,200,000 splits)
```

The t interval treats the five tasks as a sample, and is wide. The exact sign-flip test across tasks can never go below 0.0625 with five tasks. The permutation test within each task, p = 0.02, asks only whether these five tasks' trials differ, and two tasks carry it. So: three of the five tasks came within 3 points of their published scores and two didn't, and five tasks picked for convenience say nothing about the 87-task 27.5. "Within 3 points" is not equivalence: 3 of 3 has an exact 95% interval from 29% to 100%, so three tries per task can't resolve differences smaller than tens of points.

## When the original harness differs

Explain each difference before calling it a failure to reproduce.

`3d-scan-calc`'s verifier changed after the published trial: since May 16 it also accepts the mass computed in millimetres. All three rerun answers (34.65 g) are the millimetre answer, which the old verifier rejected, so this task isn't a like-for-like comparison.

`offer-letter-generator`'s files are unchanged since the published run. All its published trials came from one batch, through the proxy, with no effort set. The trajectories would say why they failed; without them, the cause is unknown.

The other differences can't be told apart: a different BenchFlow, a pinned OpenHands (1.16.0) against an unpinned one, the model's effort, the route, and a model served five to six months later. Each could move a score, and five tasks can't separate them.

To shrink that list next time, rerun at the task commit the published trials name, pin the agent's version, and keep the request bodies (`trajectory/llm_trajectory.jsonl`) so the effort a model received can be checked.
