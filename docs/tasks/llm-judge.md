# LLM judge

A [rubric review](./rubric-review.md) adds a reviewer agent to a test. When no script can check the answer, as with a piece of writing, a model can grade it alone: a verifier document with an `llm-judge` strategy takes the place of `test.sh`, and the model's verdicts on your criteria are the reward. The judge reads the files the agent left, not its trajectory, so use it when those files are all there is to grade.

The example task asks for "an original haiku about the sea" in `/app/poem.txt`. It began as `bench new sea-haiku`, and its verifier folder now holds two files. `verifier.md` picks the judge, its rubric and its model:

```markdown
---
verifier:
  strategies:
    judge:
      type: llm-judge
      rubric: judge.toml
      model: openai/mimo-v2.6-flash
---

MiMo V2.6 Flash reads `/app/poem.txt` and answers the two pass/fail questions
in `judge.toml`. The reward is 1 only when both pass.
```

`judge.toml` holds the criteria:

```toml
[judge]
files = ["poem.txt"]

[[criterion]]
name = "about_the_sea"
type = "binary"
description = """PASS when poem.txt holds a poem whose subject is the sea or the ocean \
(its water, waves, tides, shore, depths or sea life), not a poem about something else \
that mentions the sea in passing. FAIL when poem.txt is missing or empty, or when the poem is about another subject."""

[[criterion]]
name = "haiku_form"
type = "binary"
description = """PASS when poem.txt holds exactly three non-empty lines of verse with about \
5, 7 and 5 syllables (count the syllables of every word; allow a difference of at most one \
syllable per line). FAIL when poem.txt is missing or empty, has more or fewer than three lines \
of verse, or contains anything besides the poem, such as a title, a note or an explanation."""

[scoring]
aggregation = "all_pass"
```

The judge copies the agent's `/app` folder out of the sandbox (set `input_dir` in the strategy to use another) and asks the model about one criterion at a time, showing it the files the rubric lists that it can read (see the file types below); a criterion can list its own files and set a `weight`. This rubric gives 1 only when every criterion passes, and [Run the controls](#run-the-controls) has the other ways to combine them.

The task and the scripts the controls below use are in BenchFlow's repository; copy them into a working folder:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks counterexamples
cp -r benchflow/docs/examples/llm-judge/sea-haiku tasks/
cp benchflow/docs/examples/llm-judge/counterexamples/*.sh counterexamples/
```

## A model from another provider

The judge has clients for Anthropic, OpenAI and Google. It picks one from the start of the model's name, such as `anthropic/`, `openai/` or `google/`, and removes that part before it sends the name. MiMo has an OpenAI-compatible endpoint, so its model is `openai/mimo-v2.6-flash`, and the OpenAI client reads its key and endpoint from the verifier's environment, which the task sets from MiMo's own variables:

```yaml
verifier:
  env:
    OPENAI_API_KEY: "${XIAOMI_API_KEY}"
    OPENAI_BASE_URL: "${XIAOMI_BASE_URL}"
```

A task reads your variables only when you name them ([Environment variables](../reference/environment-variables.md#let-a-task-read-your-variables)), so every check and run of this one needs `--allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL`, or `BENCHFLOW_ALLOW_TASK_ENV=XIAOMI_API_KEY,XIAOMI_BASE_URL` in the environment. Without them, both stop: `refused: the task asks for XIAOMI_API_KEY, XIAOMI_BASE_URL from your environment`. In our runs that succeeded, no file of the job held the key, and a script agent that listed the variables named like a key saw none. A judge call that fails can put the key in the error it records on 0.9.1 and 0.10.0rc1 (fixed after 0.10.0rc1), so read a failed trial's files before you share a job.

## Run the controls

```bash
export XIAOMI_API_KEY=... XIAOMI_BASE_URL=...   # your MiMo key and base URL, as in the Quick start
bench check tasks/sea-haiku --allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL
bench run tasks/sea-haiku -a oracle --allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL
```

```text
✓ sea-haiku — valid (structural)
...
[sea-haiku] Running verifier...
[sea-haiku] llm-judge verifier: 2 criteria → reward 1.0000
[sea-haiku] Rewards: {'reward': 1.0}
[PASS] sea-haiku (reward=1.00, tools=0)
...
Cost:      none: no model tokens were used
```

Then the agent that does nothing, and a script that writes a haiku about a city:

```bash
bench run tasks/sea-haiku -a nop --allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL
bench run tasks/sea-haiku -a script:counterexamples/city-haiku.sh --allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL
```

| Agent | Its poem | `about_the_sea` | `haiku_form` | Reward |
|---|---|---|---|---|
| `oracle` | "Grey waves fold and break / salt wind carries the gull cries / the tide pulls back home" | pass | pass | 1.0 |
| `nop` | none | fail | fail | 0.0 |
| the city script | "Neon signs flicker / taxis honk through the wet streets / the city won't sleep" | fail | pass | 0.0 |

We ran the oracle and the city script three more times each with `--trials 3`, and every verdict held. The city haiku passes one criterion of two, so its reward depends on how the rubric combines them:

| `aggregation` | The reward | The city haiku |
|---|---|---|
| `all_pass` | 1 if every criterion passes, else 0 | 0.0 |
| `any_pass` | 1 if any criterion passes, else 0 | 1.0 |
| `weighted_mean`, the default | the criteria's scores, averaged by `weight` | 0.5 |
| `threshold` | 1 if that average reaches `threshold` (0.7 unless set), else 0 | 0.0 |

## What the judge records

The trial's `verifier/reward.json` holds the reward, and `verifier/reward-details.json` a score for each criterion. The city haiku's, shortened:

```json
{"criteria": [{"type": "dense", "reward": 0.0, "source": "criterion:about_the_sea", "step": 0, ...},
              {"type": "dense", "reward": 1.0, "source": "criterion:haiku_form", "step": 1, ...}],
 "aggregate": {"reward": 0.0, "method": "mean"}, "source": "llm-judge"}
```

The file says `"method": "mean"` whatever the rubric says; this reward of 0 came from `all_pass`. Nothing else about the judge is kept:

- **Its reasons.** No file in the job explains a verdict, so a failed criterion shows only its 0. A rubric review explains each verdict in its report.
- **Its model.** No file of the run names the judge's model. The model is in the task's verifier document, and the trial's `task_digest` in `result.json` covers it: the digest changed when we changed the model line.
- **Its tokens.** The `Cost:` line above reports no tokens, although the judge called MiMo twice, and the job's summary counts none. The run's records don't count the judge's tokens anywhere.

A judge that can't answer leaves the trial unscored, where a test that crashes would score 0 ([When the verifier can't run](./verifiers.md#when-the-verifier-cant-run)). With the model misspelled as `openai/mimo-v2.6-flsh`, the run reports `verifier crashed: JudgeScoringError: Judge error on criterion about_the_sea: BadRequestError`.

## Grade a web page

Comparing two pages by people's votes needs 0.10.0rc1 ([Human preferences](./human-preferences.md)). On 0.9.1, a model can grade each page by itself, as the task's verifier: it checks one page against criteria you write and gives that trial its reward. It can't compare two pages, and it reads the HTML source, so it never sees the page rendered. The `llm-judge` strategy above can't read the page at all: it shows the model only `.md`, `.txt`, `.json`, `.csv`, `.docx`, `.xlsx`, `.pptx` and `.pdf` files, and a copy of this task that used it scored the reference page 0. An `agent-judge` strategy reads the files it lists. [`repair-cafe-judge`](../examples/design-task/repair-cafe-judge/task.md) is the design task with this `verifier/verifier.md` in place of `test.sh`:

```markdown
---
verifier:
  strategies:
    judge:
      type: agent-judge
      role: page-judge
      isolation: verifier-only
      model: openai/mimo-v2.6-flash
      inputs: [/app/out/index.html]
---

MiMo V2.6 Flash reads the HTML of `/app/out/index.html` and checks it against
three pass/fail criteria. The reward is 1 only when all three pass.

## role:page-judge

You grade one web page, `/app/out/index.html`, from its HTML source. You can't
see it rendered. Give each criterion "pass" or "fail".

- facts: pass when the page states every fact in the brief, each correctly: what
  happens (Fix-It Friday, a free repair café run by volunteers), the three things
  to bring (a lamp, a bike, a jacket with a broken zip), when (the first and
  third Friday of each month, 6 to 9 pm) and where (the community hall, 40 Mill
  Street). Fail when any of them is missing or different.
- first_screen: pass when the head has a viewport meta tag with
  width=device-width and, in the order of the body, when (the days and the
  hours) and where (the address) come after no more than the headline, a short
  paragraph and a short list. Fail when longer content comes first, such as a
  description of each thing to bring.
- self_contained: pass when the page is one complete HTML document that works on
  its own: its CSS is inside it, it loads no other file by a relative path, and
  it has no placeholder or unfinished text, such as lorem ipsum, TODO or an empty
  section. Links within the page and inline SVG are fine. Fail otherwise.

In the JSON you return, "score" is 1 when all three pass and 0 otherwise, and
"reasoning" gives each criterion in the form "name: pass, reason" or
"name: fail, reason".
```

Its `task.md` sets the judge's `OPENAI_API_KEY` and `OPENAI_BASE_URL` from MiMo's variables, as on the [LLM judge](#a-model-from-another-provider) page, so every run needs `--allow-task-env`:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/design-task/repair-cafe-judge tasks/
export XIAOMI_API_KEY=... XIAOMI_BASE_URL=...   # your MiMo key and base URL, as in the Quick start
bench run tasks/repair-cafe-judge -a oracle --allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL
```

To grade agents' pages on 0.9.1, run the agents on this task itself; its `task.md` has no `preference:` setting, which 0.9.1 doesn't know, so it runs there:

```bash
bench run tasks/repair-cafe-judge -a claude -m xiaomi/mimo-v2.6-flash --allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL
```

To check the judge on pages we already had, two script agents wrote out the pages that the first Claude Code and OpenCode trials of [Collect the outputs](./human-preferences.md#collect-the-outputs) made (those ran on 0.10.0rc1). With the oracle and `nop`:

| Agent | `facts` | `first_screen` | `self_contained` | Reward |
|---|---|---|---|---|
| `oracle` | pass | pass | pass | 1.0 |
| `nop` | not asked | not asked | not asked | 0.0 |
| Claude Code's page | pass | pass | pass | 1.0 |
| OpenCode's page | pass | fail | pass | 0.0 |

We ran the oracle and each saved page three more times with `--trials 3`, and every verdict held. The plain reference page and Claude Code's designed one both score 1; telling them apart takes a comparison, which is what the pairs on [Human preferences](./human-preferences.md) are for. `verifier/reward-details.json` keeps the model's name and its reasoning, which names each verdict, and the run's cost line counts none of the judge's tokens. With no page, the trial scored 0 without asking the model. BenchFlow takes the score the model gives, so the all-pass rule is only in the prompt; in every trial here, the score matched the verdicts.

The model judges the layout from the order of the HTML. In Chromium on a 390 × 844 phone, Claude Code's page puts the hours 919 pixels down and the address 1,126 pixels down, below the first screen, yet `first_screen` passed. The judge also gives the model 2,048 tokens to reply, and MiMo reasons before it answers: when `first_screen` asked it to work out the layout from the CSS, it spent them all on Claude Code's page and the trial was unscored (`JudgeScoringError: agent-judge strategy 'judge' returned an invalid verdict (ValueError)`).

## Limits in 0.9.1

- **The judge reads some file types only.** It shows the model `.md`, `.txt`, `.json`, `.csv`, `.docx`, `.xlsx`, `.pptx` and `.pdf` files that sit directly in its input folder, and skips anything else, such as an `.html` page or a file in a subfolder: a copy of the repair café task graded this way scored its own reference page 0 on every criterion. For a web page, use an `agent-judge` strategy, which reads the files you name ([Grade a web page](#grade-a-web-page) shows one).
- **The judge's settings go in the verifier document.** A `task.md` that sets `verifier.type: llm-judge` and `verifier.judge` instead is refused by both check and run: `verifier/ has no runnable verifier entrypoint (expected test.sh or a valid verifier.md selected strategy)`.
- **Name the model in the strategy.** A model named in `judge.toml` is ignored. A strategy without one uses `claude-sonnet-4-6`, whose client needs the `benchflow[judge]` extra; without the extra the trial is unscored: `verifier crashed: JudgeEnvironmentError: The anthropic judge SDK is required to run model 'claude-sonnet-4-6' but is not installed.`
- **The check doesn't read the rubric.** A misspelled `aggregation = "all-pass"` passed `bench check`, and the city haiku scored 0.5, as under `weighted_mean`.
