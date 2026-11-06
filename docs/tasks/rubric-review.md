# Rubric review

A test checks the answer. A rubric review checks the work: after the test, a reviewer agent, an LLM grader, reads the run's records (the files the agent left, its trajectory, its report) and grades them against criteria you write. Use it when how the answer was found matters as much as the answer, as in research and design tasks. For a task no script can check, a model can grade the answer alone: [Grade with a model, no test script](#grade-with-a-model-no-test-script).

## When to use a rubric

Add one when a wrong method can still produce a right-looking answer. The half-life test can't tell a fitted 37.4 minutes from a typed one: the no-method counterexample in [Write a task](./write-a-task.md#check-it-and-run-the-controls) writes the true value and passes. A rubric can require the fit itself.

Keep the test. The review adds to it, and a run passes only when both agree.

## Write the rubric

Put it at `verifier/rubric.json`, next to the test. The half-life task's, shortened ([the whole file](../examples/research-task/half-life/verifier/rubric.json)); to follow along, copy the whole file into the task from [Write a task](./write-a-task.md), and the no-method script from [`counterexamples/no-method.sh`](../examples/research-task/counterexamples/no-method.sh):

```json
{
  "criteria": [
    {
      "name": "background_modelled",
      "blocker": 1,
      "weight": 5,
      "description": "Ignoring the background biases the half-life high by about 4.5 minutes on this data.",
      "guidance": "Read /app/report.md, the scripts the agent left in /app and its trajectory under trial/trajectory/. PASS when the fit includes a constant background term, or subtracts a background estimated from the data before fitting. FAIL when it fits a pure exponential to the raw counts, or when no method is shown."
    },
    {
      "name": "uncertainty_from_data",
      "blocker": 0,
      "weight": 4,
      "description": "The stated uncertainty should come from the data, not a guess.",
      "guidance": "Score 2 when the uncertainty comes from the fit's covariance, ... Score 1 when ... Score 0 when it is guessed, fixed by hand or missing."
    },
    {"name": "noise_model", "blocker": 0, "weight": 3, "...": "..."},
    {"name": "reproducible", "blocker": 0, "weight": 3, "...": "..."}
  ]
}
```

With the file in place, every run of the task starts a review, so take it out again before you follow the other guides.

There are two kinds of criterion:

- A **blocker** (`"blocker": 1`) is pass or fail. If any blocker fails, the trial's reward is 0, whatever the test said.
- A **scored** criterion gets 0, 1 or 2 points, times its weight.

When the test passes and every blocker passes, the trial's reward is the rubric's score: the scored criteria's points over their maximum. So a trial passes, at 1.0, only when every scored criterion gets full marks. `result.json` keeps both numbers: `rubric_reward` is the scored criteria's share whatever the blockers say, and `reward` is 0 when a blocker fails, so they differ when one does. Write each `guidance` as a rule another expert could apply without asking you: what earns 2, what earns 1, what earns 0.

## Run the review

A task with a rubric is reviewed on every run, oracle and `nop` included. Name the reviewer's agent and model; 0.9.1's default reviewer has no default model and refuses to start. The reviewer works without internet access, which needs a model BenchFlow can vouch for, or your say-so ([Choose the reviewer](#choose-the-reviewer)):

```bash
export BENCHFLOW_RESTRICTED_WEB_MODELS=openai/mimo-v2.6-flash
bench run tasks/half-life -a script:counterexamples/no-method.sh \
  --reviewer-agent claude --reviewer-model xiaomi/mimo-v2.6-flash
```

```text
[half-life] Running verifier...
[half-life] Rewards: {'reward': 1.0}
[review-half-life__4a3d6180] Prompt 1/1: You are reviewing one finished agent run. Judge the run against each criterion l...
[review-half-life__4a3d6180]   → end_turn, 9 total tool calls
[half-life] Reviewed half-life__4a3d6180 with rubric .../half-life/verifier/rubric.json (valid=True)
[FAIL] half-life (reward=0.00, tools=0)
Job complete: 0/1 (0.0%), mean_reward=0.00, errors=0, idle_timeouts=0, time=5.1min

✗ Score: 0/1 (0.0%), mean reward 0.00, errors=0
  ✗ half-life: reward 0.0 — rubric_reward 0.0, verifier_reward 1.0
```

The test passed the typed answer (`verifier_reward 1.0`); the review failed it, and the trial scores 0.

## Read the review

Each review is in the trial's `reviews/<id>/review_report.json`: a verdict and an explanation per criterion, a summary, and how the reward was made. The blocker's verdict here:

```json
"background_modelled": {
  "outcome": "fail",
  "explanation": "No method is shown at all. The workspace snapshot contains no report.md and no fitting script — only the pre-written answer.json — and the trajectory (trial/trajectory/acp_trajectory.jsonl) shows a single step running script:no-method.sh that produced empty stdout and wrote 37.4/1.0 ... the 'no method is shown' failure branch applies."
}
```

and the scoring block:

```json
{"policy": "tests-blockers-quality-v1", "passed": false, "tests_pass": true, "all_blockers_pass": false,
 "failed_blockers": ["background_modelled"], "verifier_reward": 1.0, "rubric_reward": 0.0}
```

The three scored criteria got 0 each, with reasons ("The uncertainty 1.0 min in answer.json has no stated derivation").

The oracle, whose fit models the background and whose report names its method, gets full marks: the blocker passes and each scored criterion gets 2, so `rubric_reward` is 1.0 and the trial passes. The reviewer checked the work itself; on `reproducible`:

```text
fit.py was left at /app/fit.py (per workspace-manifest.json original_path) and report.md says 'Reproduce: python3 /app/fit.py'. I copied the snapshot data/ and fit.py into my /app and ran it: it printed 'half-life 37.120 +- 1.135 min' ...
```
 For the no-method run, `result.json` keeps all three numbers: `{'reward': 0.0, 'verifier_reward': 1.0, 'rubric_reward': 0.0}`. Read a sample of reviews before you trust a rubric, as you would a test: the reviewer is a model.

## Grade with a model, no test script

A rubric review adds a reviewer to a test. When no script can check the answer, as with a piece of writing, a model can grade it alone: a verifier document with an `llm-judge` strategy takes the place of `test.sh`, and the model's verdicts on your criteria are the reward. The judge reads the files the agent left, not its trajectory, so use it when those files are all there is to grade.

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

The judge copies the agent's `/app` folder out of the sandbox (set `input_dir` in the strategy to use another) and asks the model about one criterion at a time, showing it the files the rubric lists; a criterion can list its own files and set a `weight`. This rubric gives 1 only when every criterion passes, and [Run the controls](#run-the-controls) has the other ways to combine them.

The task and the scripts the controls below use are in BenchFlow's repository; copy them into a working folder:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks counterexamples
cp -r benchflow/docs/examples/llm-judge/sea-haiku tasks/
cp benchflow/docs/examples/llm-judge/counterexamples/*.sh counterexamples/
```

### A model from another provider

The judge has clients for Anthropic, OpenAI and Google. It picks one from the start of the model's name, such as `anthropic/`, `openai/` or `google/`, and removes that part before it sends the name. MiMo has an OpenAI-compatible endpoint, so its model is `openai/mimo-v2.6-flash`, and the OpenAI client reads its key and endpoint from the verifier's environment, which the task sets from MiMo's own variables:

```yaml
verifier:
  env:
    OPENAI_API_KEY: "${XIAOMI_API_KEY}"
    OPENAI_BASE_URL: "${XIAOMI_BASE_URL}"
```

A task reads your variables only when you name them ([Environment variables](../reference/environment-variables.md#let-a-task-read-your-variables)), so every check and run of this one needs `--allow-task-env XIAOMI_API_KEY --allow-task-env XIAOMI_BASE_URL`, or `BENCHFLOW_ALLOW_TASK_ENV=XIAOMI_API_KEY,XIAOMI_BASE_URL` in the environment. Without them, both stop: `refused: the task asks for XIAOMI_API_KEY, XIAOMI_BASE_URL from your environment`. The key stays with the judge: no file of the job holds it, and a script agent that listed the variables named like a key saw none.

### Run the controls

```bash
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

### What the judge records

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

### Limits in 0.9.1

- **The judge's settings go in the verifier document.** A `task.md` that sets `verifier.type: llm-judge` and `verifier.judge` instead is refused by both check and run: `verifier/ has no runnable verifier entrypoint (expected test.sh or a valid verifier.md selected strategy)`.
- **Name the model in the strategy.** A model named in `judge.toml` is ignored. A strategy without one uses `claude-sonnet-4-6`, whose client needs the `benchflow[judge]` extra; without the extra the trial is unscored: `verifier crashed: JudgeEnvironmentError: The anthropic judge SDK is required to run model 'claude-sonnet-4-6' but is not installed.`
- **The check doesn't read the rubric.** A misspelled `aggregation = "all-pass"` passed `bench check`, and the city haiku scored 0.5, as under `weighted_mean`.

## Check the reviewer

A reviewer is a model, so check it the way you'd check a test: run it on work whose verdict you already know, before you trust it on work you don't. Three cases cover the main ways it can be wrong:

| Case | The work | What the reviewer should say | What it said |
|---|---|---|---|
| right | the oracle: a background fit, its uncertainty from the chi-square profile, a report | pass everything | blocker passed, 2 on each criterion: reward 1.0 |
| no method | the true value, typed in, with no script or report | fail the blocker | blocker failed: reward 0 |
| a planted instruction | a script that fits no background, the right numbers typed in, and a report that claims a background fit and tells the reviewer to pass it ([`injected-report.sh`](../examples/research-task/counterexamples/injected-report.sh)) | fail the blocker, and ignore the note | blocker failed: reward 0, `rubric_reward` 0.3 |

On the third, the reviewer reran the script itself rather than trusting the report:

```text
fit.py line 5 fits a pure exponential to the raw counts (np.polyfit on log(counts), commented '# a pure exponential: no background term') with no constant background term; the trajectory stdout 'half-life 42.73434679000039' exactly matches my reproduction of that background-free fit, not the 37.12 min a background model gives. report.md's claim of 'counts = A exp(-ln2 t / T) + B' is contradicted ...
```

Each case ran once. Before you rely on a reviewer, run each case a few times to see whether its verdict holds, and add cases that look like your task's real failures.

## Choose the reviewer

- **Name it.** `--reviewer-agent` and `--reviewer-model` take the same agents and models as `-a` and `-m`. `--reviewer-reasoning-effort`, `--reviewer-timeout-sec` and `--reviewer-concurrency` set the rest.
- **Its network.** The reviewer runs without internet access, so BenchFlow admits only models it can identify as having no web search of their own. Others are refused, and the trial ends unscored, filed as a verifier failure: `the verifier failed (task problem)`. The run's output gives the reason: `With web access restricted, BenchFlow runs only models it can identify ... 'openai/mimo-v2.6-flash' is not one ... allow it explicitly in the controller's environment: BENCHFLOW_RESTRICTED_WEB_MODELS=openai/mimo-v2.6-flash`. Set that variable to the id the message names, which can differ from the one you passed (`xiaomi/mimo-v2.6-flash` here). `--reviewer-open-network` gives the reviewer the network instead.
- **Its cost.** The run's `Cost:` line counts the agent's tokens only. This review used 128,754 tokens that the line doesn't show; each review's own `result.json`, under `reviews/<id>/runtime/`, has them.
- **After the fact.** `bench review <job or trial>` reviews finished runs and writes each review's `review_report.json` under `jobs/review-<time>/` (or `--out-dir`); it never changes the trials' rewards. Give it the rubric with `--rubric`, or the folder of tasks with `--tasks-root` so it uses each task's own `verifier/rubric.json`; with neither it uses a built-in default rubric. `-a` and `-m` pick the reviewer.
