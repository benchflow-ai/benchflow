# Human preferences

> Needs BenchFlow `0.10.0rc3`, a release candidate on PyPI that you install by its version ([Install](../start/install.md#install-the-010-prerelease)). 0.9.2 has no preference grading; on 0.9.2, a model can grade each page against criteria you write, one page at a time rather than in pairs ([LLM judge](./llm-judge.md#grade-a-web-page)).

Some work can't be scored by a test: whether a landing page is good is a matter of judgment. Here a judge compares two outputs at a time, with the procedure we use to judge designs:

1. The judge never sees an agent, model or harness name, and each pair's sides are shuffled.
2. Each output is live, at true device widths, and nothing covers it unless the judge asks. The task's brief opens on request.
3. Each vote (A, B or a tie) carries reason keys and an optional note, so it can be checked later.
4. Ratings are read once every pair has its votes, with intervals.
5. A person or a model can judge the same packet, and the agreement between judges is reported.

BenchFlow 0.10.0rc3 covers part of it: `bench preferences pairs` makes blind pairs from finished trials, `import` checks and records the votes, and `rank` rates the agents and checks the judges. The runs below used 0.10.0rc1, whose preference commands are 0.10.0rc2's but for one thing: 0.10.0rc2 names each arm with a hash of its harness versions too (`/ harness-…` after the model) when its trials recorded them, as 0.10.0rc2's trials do. For the judging itself, this page uses two examples from BenchFlow's docs, which the package doesn't include: a judging page for people and a script for model judges, in [`docs/examples/design-task/judge/`](../examples/design-task/judge). Votes never change a trial's reward. On 0.10.0rc3 `rank` prints more than the outputs below: whose votes the ratings are, no interval where every resample gave one value, and a head-to-head table.

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

Run each agent you want to compare. Here, Claude Code on MiMo and OpenCode on DeepSeek, two trials each, with `XIAOMI_API_KEY`, `XIAOMI_BASE_URL` and `DEEPSEEK_API_KEY` exported ([Agents and models](../evaluate/agents.md#a-model-from-another-provider)):

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

Check the sides before anyone judges. BenchFlow flips a coin for each pair, and with seed 7 Claude Code's output was on the left in all four, so a lean to the left and a lean to one agent would be the same votes. This reads the private half of the store, so keep its result from the judges:

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

`judge_model.py` gives a model the brief, the reason keys and both outputs that a person gets, but as pictures from headless Chromium rather than live pages. It needs Playwright and its browser: `pip install playwright`, then `python -m playwright install chromium`. Each output gets three: the first screen of a 1440 × 900 window, the whole desktop page (cut at 2,700 pixels) and the whole page on a 390 × 844 phone. It calls an OpenAI-compatible endpoint that accepts images, here MiMo's, with the key and base URL from [Quick start](../start/quick-start.md#on-another-provider):

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

The ratings table followed. Read the coverage line first and wait until it counts every pair: BenchFlow prints ratings either way, and prints that line only when you give `--raters-per-pair` or the task sets `raters_per_pair`.

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

Both judges preferred Claude Code's page in all four pairs (`--by agent` or `--by model` groups the ratings another way). The interval of 1274 to 1274 says only that every vote went one way: each resample of the 8 votes is the same 8 to 0, so the bootstrap has nothing to vary. If the two agents were equally good, all four pairs would go the same way 1 time in 8, and each output is in two of the pairs, so the evidence is weaker than four separate comparisons. For an interval that means something here, count only pairs that share no output: two of the four do, and Claude Code's page won both, so a 95% exact (Clopper-Pearson) interval for its chance of winning such a pair runs from 15.8% (0.025 to the power 1/2) to 100%. A claim about agents across a design benchmark needs many tasks: compare the agents task by task, as in [Compare runs](../audit/compare-runs.md).

## Read the agreement between the judges

The judges gave the same answer on 4 of 4 pairs, where their own choice rates would agree half the time by chance (kappa 1.00). Over four pairs that is a weak check, and two models agreeing says nothing about whether people would. MiMo also wrote Claude Code's pages, and a model can favour its own work; Claude Opus 5.5 wrote neither page and chose the same. The left side won 4 of 8 votes with the sides balanced, so there is no sign of a side bias. With a person and a model on the same packet, BenchFlow pools every pair of judges into one agreement line, so compare their votes files pair by pair. `--min-agreement`, or a task's `preference.min_agreement`, sets the raw agreement `rank` checks.

## Outputs made outside BenchFlow

> New in 0.10.0rc3: 0.10.0rc2 has no `--items`, `--condition`, `--entry` or `--golden`.

A golden, a designer's version or a remix made in another tool can be a side beside the trials. List each one in a JSON-lines file and pass it with `--items`:

```json
{"task": "repair-cafe", "output": "goldens/repair-cafe/site", "kind": "golden", "condition": "golden", "agent": "golden", "model": "designer", "external_id": "golden-repair-cafe-R1"}
{"task": "repair-cafe", "output": "remixes/repair-cafe/1/site", "condition": "enriched", "agent": "remix", "model": "claude-opus-5-5", "external_id": "remix-repair-cafe-1"}
```

`task` and `output`, a folder, are required, and `agent` names the arm the output is rated under. An output joins the trials of its task's version; when those trials ran more than one version, name the one it answers with `task_digest`. A task no trial ran is paired among its listed outputs alone. `external_id` is the output's id in its own tool, so a vote from that tool can name it. A row takes only `task`, `task_digest`, `output`, `kind`, `condition`, `agent`, `model`, `harness_versions`, `external_id` and `source`, and the whole file is checked before the store is made.

```bash
bench preferences pairs jobs/*/*/trial-*/*/ --items items.jsonl --condition one-shot --entry index.html \
  --store store --out assignment.json --seed 8
```

- `--condition` labels every trial's output, and a listed output brings its own. Two arms with the same agent and model under different conditions are rated apart, and `rank --by condition` rates the conditions.
- `--entry` names the page to open: each output in `assignment.json` gets its `entry` and the `page` to open in its copy, and an output without that page is left out.
- `--golden bar` keeps a golden out of every pair, for a tool that shows it as the bar instead. The default, `side`, pairs it like any output.

## Votes as the rater's tool showed them

> New in 0.10.0rc3.

A tool that makes its own pairs can name the two outputs instead of a `pair_id`. `left_item` and `right_item` are the outputs shown on the left and on the right, by their id in `assignment.json` or their `external_id`, and `choice` is the side chosen as shown. Both outputs must answer one task version. A rater's later vote on the same two outputs replaces the earlier one, whichever side each was on. A vote can also carry `rater_kind` (`person`, `guest` or `model`), `invited` (`true` or `false`), `against` (`golden` when the golden was shown as the bar, `brief` when only the brief was) and `tags`. They are recorded with the vote and don't change its id, so a votes file from 0.10.0rc2 imports again without adding anything.

`import` also reads the Frontier Design Benchmark vote site's export (`fdb-votes/v1`) as it comes. Each of its votes names the two outputs a voter saw; a voter who came through a personal link is a `person` and anyone else a `guest`, and a skip is an abstention. The site keeps one vote per voter and pair, so a changed vote replaces the earlier one, and a vote a later export no longer has (its voter pressed Undo) is recorded as withdrawn and stops counting. An export withdraws only votes cast before it was made, so importing an older export again changes nothing, and an export with no time (`exportedAt`) or with no vote on the store's outputs withdraws nothing; an `exportedAt` that isn't an ISO 8601 time with a zone refuses the file. `--skip-unknown` leaves out the votes on outputs the store doesn't have instead of refusing the file, and `--dry-run` checks every vote and writes nothing:

```bash
bench preferences import store --votes export.json --dry-run
```

## People, guests and model judges

> New in 0.10.0rc3.

`rank` rates people's votes. Guests' votes and model judges' votes are fitted in tables of their own and never pooled with people's. A vote says who cast it with `rater_kind`; a vote without one is a person's, unless its rater is named `model:<name>`, as `judge_model.py` names its judges. With no person's vote in the store, the ratings are the first kind that voted, and `rank` says so before anything else. `--raters person,model` fits kinds together when that is what you mean.

Each model judge is measured against people. On every pair people voted on, the answer is the output most of them preferred; a judge agrees when it chose that output, and its tie counts as a miss. `rank` prints each judge's agreement with a 95% interval, split by the arms in the pair, beside one person's agreement with the rest of the people. A judge asked about a pair both ways round keeps both answers, so `rank` can say how often they match; the pair still counts once in its agreement, scored by the share of its answers that agree. A judge that gave rubric scores instead of votes is measured the same way, the output it scored higher being its choice, and its scores are shown apart from people's, unless `--raters` rates judges with people. Guests' rubric scores get a table of their own.

## Intervals and the smallest difference

> New in 0.10.0rc3.

One rater's votes share that rater's taste, and one task's votes share that task, so they aren't independent. With several raters and several tasks, `rank` draws whole raters and whole tasks for each resample (`--cluster` picks `rater+task`, `rater`, `task` or `vote` instead of the default, `auto`). The design effect says how much wider that made the intervals than redrawing single votes. The head-to-head table divides each pair of arms' votes by it, then gives the share the first arm won, its Wilson interval, and the smallest gap from an even split those votes could show. When the design effect can't be measured (with `--bootstrap 0`, or when no arm gets both a clustered and a single-vote interval), the table gives each pair's votes and share with no interval or gap, and says why. An interval that every resample gives the same value, as when all the votes went one way, is left out instead of printed with no width. A resample that drew none of an arm's votes, as happens when an arm was compared in only a few of the tasks, says nothing about it, so it is left out of that arm's interval rather than counted at the reference's 1000; the row's `resamples` and a line under the table say how many counted.

## Ranks wait for every pair

> New in 0.10.0rc3.

`rank` says how many pairs still have no vote, and until every pair has one it lists the arms by name, without ratings. `--provisional` shows the ratings anyway, and `--json` always has them, with `ranked` false.

## What 0.10.0rc2 doesn't do yet

- Show a pair to a judge and record the vote: no judging tool ships; the one above is an example.
- Judge with a model: no model judge ships. The example script shows a model three still pictures of each output, not the live page a person can scroll and click, so a person's and a model's votes rest on different views.
- Show true device widths on a laptop: the example page shrinks the 1440-pixel desktop window to fit a narrower screen; give judges a window about 2,000 pixels wide, or judge one side at a time with X.
- Record reasons: a vote holds one free-text `reason`; reason keys aren't defined, checked or counted.
- Show the brief: `assignment.json` carries no task instruction.
- Name the page to open: the copies keep the artifact folder (`out/index.html`), and `assignment.json` doesn't say so.
- Keep the judges blind: the bundle copies every file the trial saved, an agent's screenshots of its own page included, and pairs are only made between agents, which shows which outputs share one.
- Balance the sides: each pair gets its own coin flip, so one agent can be on the left in every pair.
- Wait for complete votes: `rank` prints ratings while pairs lack votes, and the coverage line only with `--raters-per-pair` or a task's `raters_per_pair`.
- Size the uncertainty of a few votes: a sweep gets a zero-width interval, and four left votes of four read as "a possible side bias" (z = 2.0).
- Describe the rubric format: `--rubric` shows it only in an error message.
- Read jobs copied from another machine: `pairs` stops with "Permission denied" when it can't read the tasks folder the job recorded. Until it does, copy the task with the jobs and set `tasks_dir` in each copied job's `evaluation.json` to the folder that holds it.
- Compare particular judges, such as a model against each person, or keep a judge's second look at a pair: a later vote replaces the earlier one.

## Assessments for an outside collector

`bench assess` is a lower-level path for one blind pair at a time, recorded beside the trial's results for a separate tool that collects the judgements. Use `bench preferences` unless you run that tool; `bench assess --help` lists its commands.
