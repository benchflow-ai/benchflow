# Rubric review

A test checks the answer. A rubric review checks the work: after the test, a reviewer agent, an LLM grader, reads the run's records (the files the agent left, its trajectory, its report) and grades them against criteria you write. Use it when how the answer was found matters as much as the answer, as in research and design tasks. For a task no script can check, a model can grade the answer alone, with no test: [LLM judge](./llm-judge.md).

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

When the test passes and every blocker passes, the trial passes, and its reward is the rubric's score: the scored criteria's points over their maximum. So a passing trial can score 0.8, and `bench results` and the run's summary count it as passed; a failed test or blocker makes the reward 0 and the trial a failure. `result.json` keeps both numbers: `rubric_reward` is the scored criteria's share whatever the blockers say, and `reward` is 0 when a blocker fails, so they differ when one does. Write each `guidance` as a rule another expert could apply without asking you: what earns 2, what earns 1, what earns 0.

## Run the review

A task with a rubric is reviewed on every run, oracle and `nop` included. Name the reviewer's agent and model; 0.9.2's default reviewer has no default model and refuses to start. The reviewer works without internet access, which needs a model BenchFlow can vouch for, or your say-so ([Choose the reviewer](#choose-the-reviewer)):

```bash
export XIAOMI_API_KEY=... XIAOMI_BASE_URL=...   # the reviewer's model: MiMo, as in the Quick start
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
- **A written deliverable.** When the work to judge is a document, such as a paper the agent compiled to PDF, give the reviewer a way to read it. `--reviewer-pdf-tools` installs PyMuPDF in its sandbox, which extracts a PDF's text and renders its pages to images; or build `docker/reviewer-pdf/Dockerfile`, which adds poppler's `pdftotext`, and pass the tag it names with `--reviewer-image` on the Docker sandbox. Without either, the reviewer has pypdf and no way to render a page. `--reviewer-submitted-file-rule` adds a paragraph to its instruction: judge the file the agent submitted, read it before judging its content, cite pages, and take a compiled PDF over its sources. Each review's `reviewer-environment.json`, under `reviews/<id>/runtime/`, names the image and its digest, the packages its setup installed, the PDF tools the image held, and the sha256 of the instruction the reviewer got.
- **Its cost.** The run's `Cost:` line counts the agent's tokens only. This review used 128,754 tokens that the line doesn't show; each review's own `result.json`, under `reviews/<id>/runtime/`, has them.
- **After the fact.** `bench review <job or trial>` reviews finished runs and writes each review's `review_report.json` under `jobs/review-<time>/` (or `--out-dir`); it never changes the trials' rewards. Give it the rubric with `--rubric`, or the folder of tasks with `--tasks-root` so it uses each task's own `verifier/rubric.json`; with neither it uses a built-in default rubric. `-a` and `-m` pick the reviewer.
