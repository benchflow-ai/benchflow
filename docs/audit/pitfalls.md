# Known pitfalls

These are problems that change a score without an error, or break a run in a way that's easy to misread. Each has the symptom you'd see, the cause, how to check for it, and the fix. Every one was hit in real runs on BenchFlow 0.9.1 or 0.10.0rc1, by us or by people running evaluations with it this autumn; each holds for 0.9.2, which is 0.9.1 with five grading-integrity fixes, and where 0.10.0rc2 fixed one, the entry says so.

Most checks read a trial's own files: `result.json` (outcome, reward, error category, usage), `config.json` (what the trial was asked to run), `trajectory/llm_trajectory.jsonl` (each model request and response that passed through BenchFlow's proxy) and, in 0.10.0rc2, the job's `run-card.json` and `report.md`.

## Find your symptom

- [Settings that don't reach the model](#settings-that-dont-reach-the-model)
  - [Scores barely move between effort levels](#scores-barely-move-between-effort-levels)
  - [An effort level is refused for an agent that has no effort option](#an-effort-level-is-refused-for-an-agent-that-has-no-effort-option)
  - [`agent integration failure [agent_model]` on the first prompt](#agent-integration-failure-agent_model-on-the-first-prompt)
  - [A run used an older model than you meant](#a-run-used-an-older-model-than-you-meant)
  - [A long answer ends early and the trial scores 0](#a-long-answer-ends-early-and-the-trial-scores-0)
  - [`Agent idle for 600s with no new tool call, ...` on a model that was still thinking](#agent-idle-for-600s-with-no-new-tool-call--on-a-model-that-was-still-thinking)
  - [Scores far from the published ones for the same model](#scores-far-from-the-published-ones-for-the-same-model)
  - [The agent version differs from a published run's](#the-agent-version-differs-from-a-published-runs)
  - [`ACP error -32603: Internal error` on the first model call](#acp-error--32603-internal-error-on-the-first-model-call)
- [Scores that count the wrong thing](#scores-that-count-the-wrong-thing)
  - [Two pass rates for one job](#two-pass-rates-for-one-job)
  - [An agent looks worse on one benchmark only](#an-agent-looks-worse-on-one-benchmark-only)
  - [A score drops on the day a provider has trouble](#a-score-drops-on-the-day-a-provider-has-trouble)
  - [A model "fails" tasks it refused](#a-model-fails-tasks-it-refused)
  - [A score rose after a rerun](#a-score-rose-after-a-rerun)
  - [A trial scores 0 though the test wrote a reward](#a-trial-scores-0-though-the-test-wrote-a-reward)
  - [Terminal-Bench 2.0 tasks fail that pass in published runs](#terminal-bench-20-tasks-fail-that-pass-in-published-runs)
  - [Correct answers score 0 on tasks whose tests use the repository's own `conftest.py`](#correct-answers-score-0-on-tasks-whose-tests-use-the-repositorys-own-conftestpy)
  - [Your number doesn't match a published one](#your-number-doesnt-match-a-published-one)
  - [Controls pass, yet agents' failures cluster on one or two tests](#controls-pass-yet-agents-failures-cluster-on-one-or-two-tests)
- [Graders an agent can fool](#graders-an-agent-can-fool)
  - [A task passes for a script that solves nothing](#a-task-passes-for-a-script-that-solves-nothing)
  - [A planted `json.py` answers for the test](#a-planted-jsonpy-answers-for-the-test)
  - [A task passes with no method behind the answer](#a-task-passes-with-no-method-behind-the-answer)
  - [An answer the task hid reaches the agent anyway](#an-answer-the-task-hid-reaches-the-agent-anyway)
  - [Controls pass, the benchmark still leaks](#controls-pass-the-benchmark-still-leaks)
  - [`--integrity audit` shows `access_unattested` for a script agent](#--integrity-audit-shows-access_unattested-for-a-script-agent)
- [Infrastructure that fails quietly](#infrastructure-that-fails-quietly)
  - [Trials end with `disk_full` midway through a run](#trials-end-with-disk_full-midway-through-a-run)
  - [Training rows are missing for some rollouts](#training-rows-are-missing-for-some-rollouts)
  - [Training rows have no tool calls](#training-rows-have-no-tool-calls)
  - [Cost is `null`](#cost-is-null)
  - [No traces in your OpenTelemetry collector](#no-traces-in-your-opentelemetry-collector)
  - [`--trials` refuses to start](#--trials-refuses-to-start)
  - [A registry dataset is refused](#a-registry-dataset-is-refused)
  - [An older dataset snapshot is refused](#an-older-dataset-snapshot-is-refused)
  - [A rubric task won't start, oracle and nop included](#a-rubric-task-wont-start-oracle-and-nop-included)
  - [`BENCHFLOW_RESTRICTED_WEB_MODELS` is set, and the model is still refused under a restricted network](#benchflow_restricted_web_models-is-set-and-the-model-is-still-refused-under-a-restricted-network)
  - [A rubric run's `Cost:` line says no model tokens were used](#a-rubric-runs-cost-line-says-no-model-tokens-were-used)
  - [The agent spends its turn on `Permission denied` and tries `su root`](#the-agent-spends-its-turn-on-permission-denied-and-tries-su-root)
  - [The agent's requests to the task's own services get HTTP 403 with `X-BenchFlow-Blocked: 1`](#the-agents-requests-to-the-tasks-own-services-get-http-403-with-x-benchflow-blocked-1)
  - [A task's digest differs between two copies](#a-tasks-digest-differs-between-two-copies)
  - [A Modal sandbox is refused at start](#a-modal-sandbox-is-refused-at-start)
- [Many runs on one machine](#many-runs-on-one-machine)

## Settings that don't reach the model

The checks in this section read each request's record in `trajectory/llm_trajectory.jsonl`; [`requests.py`](../examples/audit/requests.py) prints the model, effort, sampling, output cap, HTTP status and finish reason of every request in a job ([Usage and telemetry](../scale/telemetry.md#what-a-trial-records) shows its output).

### Scores barely move between effort levels

- Cause: the provider or a router in front of it ignored or remapped the setting; some providers map every level above `high` to `high`
- Check: each request's body in `trajectory/llm_trajectory.jsonl`; in 0.10.0rc2, the sampling line of `report.md`
- Fix: label results with the effort the requests carried, which can differ from the one you typed

### An effort level is refused for an agent that has no effort option

- Message: `reasoning_effort=... was requested for agent ..., but that agent does not declare or advertise an ACP effort config option`
- Cause: BenchFlow sets effort through an option the agent must offer, and refuses rather than drop it
- Check: the message
- Fix: use an agent that offers effort, or run without `--reasoning-effort` and report "the agent's default"

### `agent integration failure [agent_model]` on the first prompt

- Cause: the agent doesn't offer the model you named. In 0.9.2, Claude Code on a subscription login offers only `default`, `sonnet`, `opus`, `haiku` and `claude-fable-5-1`, so `-m claude-sonnet-5-5` stops the trial before any model call
- Check: the message lists the names the agent offers
- Fix: pass one of them

### A run used an older model than you meant

- Cause: `-m` was left out, so the agent used its default (Claude Code: `claude-haiku-4-5-20251001` in 0.9.2, `claude-haiku-5-5` in 0.10.0rc2)
- Check: `config.json`, `model`
- Fix: always pass `-m`

### A long answer ends early and the trial scores 0

- Cause: the agent's output cap: BenchFlow starts Claude Code with `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`
- Check: responses whose stop reason is `max_tokens` or `length` in `llm_trajectory.jsonl`
- Fix: set the cap the model supports with `--agent-env CLAUDE_CODE_MAX_OUTPUT_TOKENS=<n>`, and report the cap and how many responses were cut

### `Agent idle for 600s with no new tool call, ...` on a model that was still thinking

- Cause: the idle limit is 10 minutes by default, and some models reason that long without sending anything
- Check: the message
- Fix: raise `--idle-timeout`, with the same value for every model you compare

### Scores far from the published ones for the same model

- Cause: the agent picks its own temperature and thinking settings, and the vendor recommends others for evaluation
- Check: the run card's `Vendor` line (0.10.0rc2); the request bodies in `llm_trajectory.jsonl`
- Fix: 0.9.2 can't set sampling for Claude Code, Codex or OpenCode ([Before you run an eval](../evaluate/before-you-run.md#what-to-write-down)); match the vendor's settings where an agent's own configuration allows it, and otherwise report how yours differ

### The agent version differs from a published run's

- Cause: each BenchFlow release pins its own agent versions
- Check: `bench doctor`, "Agent versions"
- Fix: match the BenchFlow version, or report the difference

### `ACP error -32603: Internal error` on the first model call

- Cause: some agents report a rejected key as an internal error
- Check: call the provider directly with the same key
- Fix: replace the key

## Scores that count the wrong thing

### Two pass rates for one job

- Cause: trials that ended unscored or errored count as 0 over all trials, and drop out of the scored-only rate
- Check: the `Outcomes` line; `job.denominators()`
- Fix: report both, and the counts behind them

### An agent looks worse on one benchmark only

- Cause: the verifier's sandbox failed (`verifier_infra`), as with Terminal-Bench 4.1's tasks on 0.9.2 ([Bring what you have](../evaluate/bring-what-you-have.md#terminal-bench-41))
- Check: `bench results`: the Assess column says `unscored`, or `error` when the verifier itself failed
- Fix: leave infrastructure failures out of agent comparisons, and say so

### A score drops on the day a provider has trouble

- Cause: a trial where only some requests failed goes on, and is scored as the agent's work
- Check: non-200 statuses in `llm_trajectory.jsonl`, per trial
- Fix: rerun those trials before you compare, and report how many there were

### A model "fails" tasks it refused

- Cause: a refusal scores 0, like a wrong answer
- Check: the stop reasons of the last responses in `llm_trajectory.jsonl`
- Fix: report refusals beside the score

### A score rose after a rerun

- Cause: a failed attempt was retried (2 retries by default) and the retry passed
- Check: `bench results <job> --attempts all` lists every attempt; each trial's `attempts` keeps them, oldest first ([Resume, retries and regrade](../scale/resume.md#retries))
- Fix: set `--retry-attempts` on purpose, and report it

### A trial scores 0 though the test wrote a reward

- Cause: the reward was outside 0 to 1 (or the task's `verifier.reward_range`), or wasn't a number
- Check: the `Verifier scored 0:` warning; `reward_out_of_range_info` in `result.json`
- Fix: write rewards in range, or declare the range

### Terminal-Bench 2.0 tasks fail that pass in published runs

- Cause: the tasks were written for an agent that runs as root, and BenchFlow runs the agent as its own user, so it can't install system packages or write under `/usr`, `/etc` or `/opt`. In Terminal-Bench 2.0, 33 of the 89 reference solutions write as root and five tasks' tests read those writes (`build-pmars`, `build-pov-ray`, `mailman`, `nginx-request-logging`, `sqlite-with-gcov`); `headless-terminal` failed when the agent installed `pexpect` for its own user only. Twelve tasks fail a control on 0.9.1 for other causes, such as package versions their servers no longer have; [Check a benchmark](./check-a-benchmark.md#terminal-bench-20-on-091) lists them
- Check: the tests' output in `verifier/test-stdout.txt`: a missing system package, or a file missing under `/usr`, `/etc` or `/opt`
- Fix: report the sandbox user beside the score. `--sandbox-user none` (or `root`) runs the agent as root and allows those writes, but root can undo BenchFlow's protections and change what the verifier runs, so use it only for agents you trust and keep its scores apart from non-root runs ([Changelog](../versions/changelog.md), 0.9.0's known limitations)

### Correct answers score 0 on tasks whose tests use the repository's own `conftest.py`

- Seen on: SWE-bench Verified's Sphinx, Flask, pytest and seaborn tasks (`fixture 'app' not found`, `pytester`, `long_df`)
- Cause: 0.9.0, 0.9.1 and 0.9.2 remove every `conftest.py` in the workspace before grading, the repository's own included ([Changelog](../versions/changelog.md))
- Check: the tests' output in `verifier/test-stdout.txt`: a missing fixture
- Fix: on 0.9.2, set `cleanup_conftests: false` under `verifier: hardening:` in the task's `task.md` (`cleanup_conftests = false` in `[verifier.hardening]` of a `task.toml`); that keeps the repository's files, and also any `conftest.py` the agent planted, so probe such tasks with a script agent that plants a `conftest.py` beside the tests, as [Probe your verifier](../tasks/reward-hacking.md#probe-your-verifier) shows for other cheats. 0.10.0rc2 keeps a repository's unchanged `conftest.py` by itself, except in a separate verifier sandbox. Otherwise leave such tasks out of 0.9.2 results, and say so

### Your number doesn't match a published one

- Cause: a different task version
- Check: each trial's `task_digest`
- Fix: pin the benchmark's commit or registry version, and report it

### Controls pass, yet agents' failures cluster on one or two tests

- Cause: the oracle and nop check the grader at two points only; a test can demand details the instruction never states
- Check: read a sample of failed trials
- Fix: fix the task where failures cluster

## Graders an agent can fool

### A task passes for a script that solves nothing

- Cause: the test reads its expected value from a file the agent can replace (SkillsBench `3d-scan-calc`)
- Check: a cheating script agent ([Check a benchmark](./check-a-benchmark.md#probe-the-verifiers-with-a-cheating-agent))
- Fix: keep the test's inputs in `verifier/`

### A planted `json.py` answers for the test

- Cause: Python run from the workspace imports the agent's modules first; BenchFlow removes planted `conftest.py` and test-runner modules, not files named like the standard library
- Check: the `shadow-json.sh` probe ([Probe your verifier](../tasks/reward-hacking.md#probe-your-verifier))
- Fix: run the test's Python scripts with `python3 -I`, and pytest with `$BENCHFLOW_PYTEST` ([Verifiers](../tasks/verifiers.md#run-pytest-in-a-test))

### A task passes with no method behind the answer

- Cause: the test checks the number only
- Check: the no-method counterexample ([Rubric review](../tasks/rubric-review.md))
- Fix: add a rubric review with a blocker ([Rubric review](../tasks/rubric-review.md))

### An answer the task hid reaches the agent anyway

- Cause: `denylist` blocks only what it lists; a page-fetching service it doesn't list can fetch a listed page
- Check: the trajectory's fetches
- Fix: use `allowlist` or `no-network` for anything that must stay out of reach

### Controls pass, the benchmark still leaks

- Cause: oracle and nop can't find shortcuts
- Check: a trajectory search for probing ([Check a benchmark](./check-a-benchmark.md#search-trajectories-for-shortcuts))
- Fix: probe the verifiers, and read what agents tried

### `--integrity audit` shows `access_unattested` for a script agent

- Cause: no root check runs before a scripted agent's turn
- Check: the verdict's `reason`
- Fix: judge script-agent probes by their rewards; see [Stop reward hacking](../tasks/reward-hacking.md#audit-how-each-reward-was-earned)

## Infrastructure that fails quietly

### Trials end with `disk_full` midway through a run

- Cause: task images and agent install layers fill the disk: 2 to 3 GB of layer per task image for Claude Code, about 14 GB for six trials at once on our machine. Over a whole benchmark the pulled task images add up: Docker listed 87 of Terminal-Bench 2.0's 89 task images at 103 GB
- Check: `df -h`; `bench cache list`. A full disk also shows as `Unexpected: [Errno 28] No space left on device` with no category, or `Sandbox startup failed: Cannot pull image ... failed to extract layer`, which reads like a task failing
- Fix: lower `-j`; lower `BENCHFLOW_AGENT_LAYER_MAX_GB` (20 by default) on small disks; `bench cache clean --remove` between runs. It keeps every task image a run pulled on Docker (`Docker cannot atomically remove a checked reference; kept`), so remove those with `docker image rm` when a task's runs finish. Rerun the trials that met the full disk

### Training rows are missing for some rollouts

- Cause: `bench train convert` leaves out rollouts that ended with an error, such as a time limit, even when they were scored 0. In 0.10.0rc2 and earlier, `--group-advantage` still wrote the rest of their group with advantages that counted them
- Check: `--manifest stats.json`: `skipped_terminal_error`, and each withheld group's `withheld` field
- Fix: with `--group-advantage`, a scored rollout that ended with an error now writes its rows, marked with the error in their `group` block, and a group missing a member's score (no result, unscored) or a member that never reached the model is withheld whole and named in the manifest

### Training rows have no tool calls

- Cause: `bench train convert` loses a Claude Code trial's tool calls in 0.9.2; 0.10.0rc2 keeps them
- Check: `bench train validate`: `rows_with_tool_calls: 0`
- Fix: convert with 0.10.0rc2 or later, or train on other agents' trials; `bench train validate --require-tool-calls` fails such rows

### Cost is `null`

- Cause: the model is missing from BenchFlow's price table, or the run used a subscription login
- Check: `price_source` in `result.json`
- Fix: cap and compare by tokens; price them from the provider's rate card

### No traces in your OpenTelemetry collector

- Cause: agents decide whether to export OpenTelemetry, Claude Code often doesn't, and BenchFlow 0.9.2 emits none
- Check: `usage_source` in `result.json`
- Fix: read usage and calls from BenchFlow's records ([Usage and telemetry](../scale/telemetry.md)); 0.10.0rc2 writes each trial's trace to `trajectory/otel-spans.json` and sends it to the collector `OTEL_EXPORTER_OTLP_ENDPOINT` names

### `--trials` refuses to start

- Cause: on 0.9.2 a GitHub address can't be repeated: `--trials > 1 needs a task folder or --dataset`
- Check: the message
- Fix: clone the benchmark first; 0.10.0rc2 repeats a task named by its address

### A registry dataset is refused

- Cause: its entry was validated with other BenchFlow versions: `bench 0.9.2 is outside the range this dataset was validated against`
- Check: `bench hub show name@version`
- Fix: use a validated version, or the benchmark's Git repository

### An older dataset snapshot is refused

- Cause: its `task.md` files use `environment:`, which is `sandbox:` now
- Check: the message
- Fix: use a newer snapshot, or rename the key

### A rubric task won't start, oracle and nop included

- Cause: every run of a rubric task is reviewed, and 0.9.2's default reviewer has no default model: `the reviewer agent 'opencode' has no default model`
- Check: the message
- Fix: pass `--reviewer-agent` and `--reviewer-model`

### `BENCHFLOW_RESTRICTED_WEB_MODELS` is set, and the model is still refused under a restricted network

- Cause: the variable must name the model as BenchFlow resolved it (`openai/mimo-v2.6-flash`), not as you passed it (`xiaomi/mimo-v2.6-flash`)
- Check: the refusal message gives the exact line to set
- Fix: set the id the message names

### A rubric run's `Cost:` line says no model tokens were used

- Cause: the line counts the agent's tokens; the reviewer's are recorded with each review
- Check: each review's `result.json` under `reviews/<id>/runtime/`
- Fix: add the reviewer's tokens when you report cost

### The agent spends its turn on `Permission denied` and tries `su root`

- Cause: the agent's user owns only the working directory (`WORKDIR`, or `/root` when the image sets none) and its own home folder, and the task needs it to write elsewhere or install system packages
- Check: the denied path in the trajectory against `Sandbox user agent ready (workspace=...)` in the run's output
- Fix: keep the agent's work inside the working directory and install what it needs in the Dockerfile; `--sandbox-user none` runs the agent as root

### The agent's requests to the task's own services get HTTP 403 with `X-BenchFlow-Blocked: 1`

- Cause: under `denylist`, the services' names aren't on the agent's list
- Check: the response headers in the trajectory
- Fix: use `allowlist`, with the services in `allowed_hosts`: by name in 0.10.0rc2, by their private address range in 0.9.2

### A task's digest differs between two copies

- Cause: macOS added `._*` files when the folder was copied with `tar`
- Check: `find <task> -name '._*'`
- Fix: copy with `COPYFILE_DISABLE=1 tar ...`, or delete the `._*` files

### A Modal sandbox is refused at start

- Cause: a system library in the image is writable by more than root: `LoaderCustodyError: ... /usr/local/lib/libpython3.12.so ... not root's alone`
- Check: the message
- Fix: fix the file's owner and mode in the task's Dockerfile, as the message says

## Many runs on one machine

- **Many `bench` processes at once.** Start one `bench run` with `-j` instead: a hundred processes starting together can starve the checks before a run, and trials never start.
- **A second run into a busy jobs folder.** A second unnamed run into a folder where a job is running starts a job of its own: `Starting a new job ...: job ... is held by another run`. Results don't mix, but you get two jobs.
- **A temporary folder per run.** Runs on one machine share `/tmp`; give each its own `TMPDIR`.
- **Clean up by name.** Remove only the containers and images your own runs created. A sweep by creation time can remove someone else's.
- **Pin the install.** `uv tool install` ignores BenchFlow's lock file; install with the release's `constraints.txt`, as in [Install](../start/install.md).
