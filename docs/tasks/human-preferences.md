# Human preferences

> Needs BenchFlow `0.10.0rc1`, a prerelease: [install it](../start/install.md#install-the-010-prerelease). 0.9.1 has no preference grading.

Some work can't be scored by a test: whether a landing page is good is a matter of judgment. Here a judge compares two outputs at a time, with the procedure we use to judge designs:

1. The judge never sees an agent, model or harness name, and each pair's sides are shuffled.
2. Each output is live, at true device widths, and nothing covers it unless the judge asks. The task's brief opens on request.
3. Each vote (A, B or a tie) carries reason keys and an optional note, so it can be checked later.
4. Ratings are read once every pair has its votes, with intervals.
5. A person or a model can judge the same packet, and the agreement between judges is reported.

BenchFlow 0.10.0rc1 covers part of it: `bench preferences pairs` makes blind pairs from finished trials, `import` checks and records the votes, and `rank` rates the agents and checks the judges. For the judging itself, this page uses two examples from BenchFlow's docs, which the package doesn't include: a judging page for people and a script for model judges, in [`docs/examples/design-task/judge/`](../examples/design-task/judge). Votes never change a trial's reward.

## The design task

The example task, [`repair-cafe`](../examples/design-task/repair-cafe/task.md), asks for a one-page website for a volunteer repair café, written to `/app/out/index.html`. Its test checks only that the page exists and states the facts the brief gives. Two settings in `task.md` matter here:

```yaml
artifacts:
  - source: /app/out
    destination: out
preference:
  protocol: pairwise
  randomize_sides: true
  raters_per_pair: 2
```

`artifacts` saves the page in each trial's folder. `preference` asks for pairs with shuffled sides and two judges on each. Copy the task, its rubric and the judge folder into a working folder:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/design-task/repair-cafe tasks/
cp -r benchflow/docs/examples/design-task/judge benchflow/docs/examples/design-task/rubric.json .
```

## Collect the outputs

Run each agent you want to compare. Here, Claude Code on MiMo and OpenCode on DeepSeek, two trials each:

```bash
bench run tasks/repair-cafe -a claude -m xiaomi/mimo-v2.6-flash --trials 2 -j 1 --jobs-dir jobs/claude
bench run tasks/repair-cafe -a opencode -m deepseek/deepseek-v4-flash --trials 2 -j 1 --jobs-dir jobs/opencode
```

All four trials passed the test. One Claude Code trial took screenshots of its own page and saved them beside it in `/app/out`, then ran into the 10-minute limit after 1,071,085 tokens; the page was already written, so it passed.

## Make blind pairs

```bash
bench preferences pairs jobs/*/*/trial-*/*/ --store store --rubric rubric.json --out assignment.json --seed 8
```

```text
4 pairs of 4 outputs.
For the rating tool: assignment.json, and the outputs under their opaque ids in store-outputs.
Store (import and rank read it): store
```

Each pair holds two outputs of the same task version from different agents. `assignment.json` lists the pairs and each output under an opaque id; which agent made which stays in `store/`. The copies keep the artifact folder, so the page to open is `store-outputs/<id>/out/index.html`. The rubric is frozen with the pairs: `rubric.json` has an `id`, a `version` and `criteria`, each with an `id`, a `scale` such as `[1, 5]` and a `description`.

Check the sides before anyone judges. rc1 flips a coin for each pair, and with seed 7 Claude Code's output was on the left in all four, so a lean to the left and a lean to one agent would be the same votes. This reads the private half of the store, so keep its result from the judges:

```bash
python3 -c 'import json; a = {i["item_id"]: i["agent"] for i in map(json.loads, open("store/items.v1.jsonl"))}; print([a[p["left"]] for p in map(json.loads, open("store/pairs.v1.jsonl"))])'
```

```text
['opencode', 'claude-agent-acp', 'opencode', 'claude-agent-acp']
```

A judge could still tell the agents apart. The screenshots one Claude Code trial saved are copied with its page. And pairs are only made between agents, so two outputs that never meet come from the same one. The judging page opens only the page itself, which hides the screenshots; nothing hides the pairing.

## Judge each pair

`serve.py` serves the pairs to one judge, who opens the address it prints:

```bash
python3 judge/serve.py store-outputs assignment.json --brief tasks/repair-cafe/task.md \
  --reasons judge/reasons.json --rater judge-1 --votes votes-judge-1.jsonl
```

```text
4 pairs for judge-1: open http://127.0.0.1:8700/ (votes go to votes-judge-1.jsonl; Ctrl-C stops)
```

The judge sees one pair at a time as A and B, each a live page to scroll and click, in a desktop window 1440 pixels wide or on a 390 × 844 phone. The desktop window shrinks to the space there is; the phone doesn't. On a screen about 2,000 pixels wide or more, A and B sit side by side; on a laptop, one fills the window.

| Key | What it does |
|---|---|
| A, B, T | Vote for A, B or a tie, and open the reasons |
| 1 to 8 | Mark a reason from [`reasons.json`](../examples/design-task/judge/reasons.json), such as "What, when and where are clear at a glance" |
| Tab | Type a note |
| Enter | Save the vote |
| U | Undo the last vote |
| I | Show or hide the brief |
| X | Switch between A and B, one at a time |
| D, P | Desktop or phone |
| Esc | Close the reasons or the brief |

Each vote goes into the votes file at once, in the format `bench preferences import` reads, with A and B written as the sides `pairs` issued. A test that pressed B, 1 and 3 and typed a note saved this line:

```json
{"pair_id": "p-c154f0f9354d6dd6", "choice": "right", "rater": "judge-1", "rated_at": "2026-11-05T18:42:35.701Z", "source": "judge.html", "reason": "keys: 1,3; note: the times and the address are on the first screen"}
```

The outputs come from two other ports, one per side, so a page an agent wrote can't read the judging page, see the other side, cast a vote or navigate the window.

## Let a model judge the same packet

`judge_model.py` gives a model what a person gets: the brief, the reason keys and both outputs, as pictures from headless Chromium (it needs Playwright). Each output gets three: the first screen of a 1440 × 900 window, the whole desktop page (cut at 2,700 pixels) and the whole page on a 390 × 844 phone. It calls an OpenAI-compatible endpoint that accepts images, here MiMo's, with the key and base URL from [Quick start](../start/quick-start.md#on-another-provider):

```bash
python3 judge/judge_model.py store-outputs assignment.json --brief tasks/repair-cafe/task.md \
  --reasons judge/reasons.json --shots shots --votes votes-mimo.jsonl \
  --model mimo-v2.6-flash --base-url-env XIAOMI_BASE_URL --key-env XIAOMI_API_KEY
```

```text
pair 1 of 4 (p-c154f0f9354d6dd6)
  tokens: 8917 in, 811 out
  b: keys: 1,3; note: B puts the when, time and where cards directly under the headline on both desktop and phone, so the logistics land first, while A buries them below the examples.
...
4 of 4 pairs have a vote from model:mimo-v2.6-flash in votes-mimo.jsonl
```

With `--command`, an agent judges instead: it runs in each pair's folder of pictures, reads the prompt on its input and prints its answer. Here Claude Code on Claude Opus 5.5, logged in as in [Quick start](../start/quick-start.md#run-claude-code-on-it), judged the same pictures:

```bash
python3 judge/judge_model.py store-outputs assignment.json --brief tasks/repair-cafe/task.md \
  --reasons judge/reasons.json --shots shots --votes votes-claude.jsonl \
  --model claude-opus-5-5 --command 'claude -p --model claude-opus-5-5'
```

```text
pair 1 of 4 (p-c154f0f9354d6dd6)
  b: keys: 1,2,4; note: B has the clearer headline, puts when, time and where in cards on the first screen, and looks more polished, while A's 'What you can bring' heading sits right against the hero with no gap ...
...
4 of 4 pairs have a vote from model:claude-opus-5-5 in votes-claude.jsonl
```

The votes below come from these two model judges. A person's votes file from the judging page goes through the same import.

## Import the votes

```bash
bench preferences import store --votes votes-mimo.jsonl
bench preferences import store --votes votes-claude.jsonl
```

```text
4 votes added, 0 already recorded.
Log head: sha256:8653f3b4c8f7c4130a41957ff88653232db3323ef644ff7a648609110d2bafc3 (keep a copy elsewhere: the log alone cannot show that its last lines were cut).
4 votes added, 0 already recorded.
Log head: sha256:148d3fcf592ea0bb9843990dc1cce20321aa379ecaf9545546e83a48268c4138 (keep a copy elsewhere: ...)
```

Every vote is checked before any is written: a pair that was issued, a choice of `left`, `right`, `tie` or `abstain`, a time after the pairs were made, and outputs unchanged since. Importing a file again adds nothing, and the log is hash-chained, so an edit to it is detected. Rubric scores import the same way, with `item_id`, `criterion` and `score` in place of `pair_id` and `choice`.

## Check that every pair has its votes

After the first import, `rank` already printed ratings:

```bash
bench preferences rank store --bootstrap 1000 --raters-per-pair 2
```

```text
4 votes counted (4 pairwise, 0 rubric scores) of 4 recorded; ... From 1 raters: model:mimo-v2.6-flash 4.
...
0 of 4 pairs have 2 or more raters.
...
```

The ratings table followed. Read the coverage line first and wait until it counts every pair: rc1 prints ratings either way, and prints that line only when you give `--raters-per-pair` or the task sets `raters_per_pair`.

## Rank with intervals

With both judges in, the same command gives:

```text
8 votes counted (8 pairwise, 0 rubric scores) of 8 recorded; ... From 2 raters: model:claude-opus-5-5 4, model:mimo-v2.6-flash 4.
Left side chosen in 4 of 8 decisive overall votes (z = 0.0, counting each vote apart although raters of a pair saw the same sides: no side bias large enough to show in 8 votes).
Two raters on the same pair gave the same answer 4 times in 4 (100%), over the 4 pairs judged by two or more raters; their own choice rates would agree 50% of the time by chance (kappa 1.00).
4 of 4 pairs have 2 or more raters.
Ratings: Bradley-Terry on the Elo scale (1000 is a fixed reference; 400 points means preferred 10 to 1). ...
Pairwise (overall)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━┓
┃ arm                                       ┃ rating ┃ 95% interval ┃ wins ┃ ties ┃ losses ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━┩
│ claude-agent-acp / xiaomi/mimo-v2.6-flash │   1274 │ 1274 to 1274 │    8 │    0 │      0 │
│ opencode / deepseek/deepseek-v4-flash     │    726 │   726 to 726 │    0 │    0 │      8 │
└───────────────────────────────────────────┴────────┴──────────────┴──────┴──────┴────────┘
```

Both judges preferred Claude Code's page in all four pairs (`--by agent` or `--by model` groups the ratings another way). The interval of 1274 to 1274 says only that every vote went one way: each resample of the 8 votes is the same 8 to 0, so the bootstrap has nothing to vary. If the two agents were equally good, all four pairs would go the same way 1 time in 8, and each output is in two of the pairs, so the evidence is weaker than four separate comparisons. A claim about agents across a design benchmark needs many tasks: compare the agents task by task, as in [Compare runs](../audit/compare-runs.md).

## Read the agreement between the judges

The judges gave the same answer on 4 of 4 pairs, where their own choice rates would agree half the time by chance (kappa 1.00). Over four pairs that is a weak check, and two models agreeing says nothing about whether people would. MiMo also wrote Claude Code's pages, and a model can favour its own work; Claude Opus 5.5 wrote neither page and chose the same. The left side won 4 of 8 votes with the sides balanced, so there is no sign of a side bias. With a person and a model on the same packet, rc1 pools every pair of judges into one agreement line, so compare their votes files pair by pair. `--min-agreement`, or a task's `preference.min_agreement`, sets the raw agreement `rank` checks.

## What 0.10.0rc1 doesn't do yet

- Show a pair to a judge and record the vote: no judging tool ships; the one above is an example.
- Judge with a model: no model judge ships.
- Record reasons: a vote holds one free-text `reason`; reason keys aren't defined, checked or counted.
- Show the brief: `assignment.json` carries no task instruction.
- Name the page to open: the copies keep the artifact folder (`out/index.html`), and `assignment.json` doesn't say so.
- Keep the judges blind: the bundle copies every file the trial saved, an agent's screenshots of its own page included, and pairs are only made between agents, which shows which outputs share one.
- Balance the sides: each pair gets its own coin flip, so one agent can be on the left in every pair.
- Wait for complete votes: `rank` prints ratings while pairs lack votes, and the coverage line only with `--raters-per-pair` or a task's `raters_per_pair`.
- Size the uncertainty of a few votes: a sweep gets a zero-width interval, and four left votes of four read as "a possible side bias" (z = 2.0).
- Describe the rubric format: `--rubric` shows it only in an error message.
- Read jobs copied from another machine: `pairs` stops with "Permission denied" when it can't read the tasks folder the job recorded.
- Compare particular judges, such as a model against each person, or keep a judge's second look at a pair: a later vote replaces the earlier one.

## Assessments for an outside collector

`bench assess` is a lower-level path for one blind pair at a time, recorded beside the trial's results for a separate tool that collects the judgements. Use `bench preferences` unless you run that tool; `bench assess --help` lists its commands.
