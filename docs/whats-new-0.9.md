# What's new in 0.9

BenchFlow 0.9.0 is a major rework of the existing Python SDK and CLI, following 0.7.11 as the first Python 0.9 release. Python remains the main implementation, with a static Linux `bfx` helper for sandbox operations. This page is for users upgrading from 0.7 or switching from the separate Rust engine 0.8.2. It lists what changes how you work. The [CHANGELOG](https://github.com/benchflow-ai/benchflow/blob/main/CHANGELOG.md) has every entry, with reasons and evidence, and its known limitations.

To move to 0.9.1, including from an exactly pinned tool installation, install the new version with its matching constraints:

```bash
uv tool install --python 3.12 --upgrade 'benchflow==0.9.1' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.1/constraints.txt
bench --version     # benchflow 0.9.1
bench doctor
```

Keep any extras your installation needs, for example `'benchflow[sandbox-daytona]==0.9.1'`. `uv tool upgrade benchflow` respects an existing version pin and therefore does not move a pinned 0.7.11 installation to 0.9.1.

Python 3.12 or newer is needed; 3.12 and 3.13 are tested. The CLI and SDK require a [POSIX host](./getting-started/install.md#host-platforms); native Windows Python is unsupported. See [Install](./getting-started/install.md), including how to keep the Rust engine's `bench` beside this one.

## Coming from the Rust engine 0.8.2

Install the Python package beside the Rust engine using [Which bench runs](./getting-started/install.md#which-bench-runs): keep Rust in its own folder and use `benchflow` for Python. To replace the command instead, add `--force` to your Python install line, preserving its version, extras and constraints; this replaces the existing `bench` executable. Check `bench --version`: Python prints `benchflow 0.9.1`, Rust prints `bench 0.8.2` with build information.

Start Python runs in a new `--jobs-dir`. `bench results` can read Rust trial folders that contain the compatible `result.json` layout. `bench eval resume` cannot reconstruct a Rust job without Python's `evaluation.json`; keep the Rust executable to resume it. Do not point a Python run at a Rust job to migrate it: sharing verbs and score rules does not make the saved run settings interchangeable.

The common verbs do not imply identical flags or task support. Python adds the `benchflow` Python API, plugin agents, task-format plugins and the rollout training APIs described below. Check this install's `bench run --help-all` before reusing a Rust command. Review the scoring changes and sandbox limitations on this page before comparing results.

## New command names

`bench` has a shorter set of verbs. The older spellings still work, and each prints one note with the new line.

| Write | Older spelling | Notes |
|---|---|---|
| `bench run <task folder \| folder of tasks \| org/repo[/path][@ref]>` | `bench eval run`, `bench eval create` | Takes every `bench eval run` flag. `bench run --help` is short; `--help-all` lists them all. |
| `bench run tasks/` (no `-a`) | `bench eval run --tasks-dir tasks/ --agent oracle` | Without `-a`, `bench run` runs the oracle. |
| `bench check` | `bench tasks check` | |
| `bench new` | `bench tasks init` | |
| `bench view` | `bench eval view` | |
| `bench results` | `bench eval list`, `bench eval inspect`, `bench eval compare` | One job, or a folder of jobs, is a summary; two jobs are compared. Flags only `inspect` or `compare` has (`--csv`, `--vary`, ...) stay on the older spelling. |
| `bench import` | `bench tasks migrate` | |
| `bench export` | `bench tasks export` | |
| `bench regrade` | `bench eval regrade` | |
| `bench branch` | `bench eval branch` | `bench eval branches` keeps its name. |
| `bench agents` | `bench agent list`, `bench agent show` | `bench agents <name>` shows one agent. |

Some flags of `bench run` have shorter names too. The older ones still work:

| Write | Older spelling |
|---|---|
| `--job` | `--job-name` |
| `--save-outputs` | `--freeze-workspace` |
| `--effort` | `--reasoning-effort` |
| `--env` | `--agent-env` |
| `--limit` | `--n-tasks` |
| `--idle-timeout` | `--agent-idle-timeout` |

These keep their names: `bench doctor`, `bench eval smoke`, `bench eval metrics`, `bench eval resume`, `bench eval branches` and `bench eval compare-lift`.

## What to do differently

> **Breaking:** the first five items below can stop a 0.7 run.

- **Use current Claude model ids.** The sandbox now installs `claude-agent-acp` 0.81.2 (Claude Code 2.1.280). It accepts `claude-haiku-4-5-20251001` (or `claude-haiku-4-5`), `claude-sonnet-5`, `claude-opus-5-5`, and the aliases `haiku`, `sonnet` and `opus`. It refuses older ids such as `claude-sonnet-4-6`. Update the `--model` of Claude Code runs and the `model:` of `claude-agent-acp` roles.
- **Check agent names.** An unknown one-word agent name now fails before any sandbox starts. In 0.7 it ran as a shell command.
- **Fix typos in `task.toml`.** A `task.toml` with an unknown key runs with a warning. A likely typo of a known key is refused, and so is any unknown key in `[sandbox]`, `[agent]`, `[verifier]` or `[steps]`. For corpora you don't control, `BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1` restores the old behaviour.
- **Codex Apps are off for scored tasks.** Scored `codex-acp` tasks no longer get your account's Apps. Pass `--codex-apps-policy inherit` to give them back. Scored Codex runs as root, or with `--sandbox-user null`, are refused.
- **Pin or bake verifier pytest plugins.** Verifiers now refuse pytest plugins whose code the agent could have changed. Bake plugins into the image, or pin their releases in `[verifier] pytest_plugin_releases` (`["pytest-asyncio==0.24.0"]`): BenchFlow installs them into its protected verifier runtime before the agent's turn, and the plugin guard loads them from there. pytest-json-ctrf for `--ctrf` comes from there without a pin. A plugin that `test.sh` installs with `uvx`, `uv add` or `pip install` is never loaded. `bench check` warns when a `test.sh` installs plugins into the workspace or `/tmp`.
- **The agents catalog is pinned per release.** OpenClaw now loads from the agents repository. To follow its `main`, set `BENCHFLOW_AGENTS_SOURCE=benchflow-ai/agents@main`. To use a local checkout, set `BENCHFLOW_AGENTS_DIR`.
- **`bf.run` uses the task's agent timeout in every form.** In 0.7, `bf.run(bf.Agent(...), bf.Environment(...))` used 900 s instead of the task's `timeout_sec`. It now uses the task's value and warns when they differ. To keep 900 s, pass `RuntimeConfig(timeout=900)`.
- **`bf.run` returns `RolloutResult` in every form**, including `Agent + Environment`. Change `isinstance(result, RuntimeResult)` to `RolloutResult`.
- **Unknown cost stays unknown.** `total_cost_usd` is `null` when cost is not known; it does not mean the run was free. Preserve null in dashboards and exports.
- **`null` means unscored.** An unscored trial's row in `results.jsonl` now has reward `null`, not `0.0`.
- **Symlinks are not uploaded.** Sandbox uploads skip symlinks. `bench check` reports a task that has them. A run skips a task whose `task.md` or `task.toml` is a link, or that has a link in a folder the run uploads (`environment/`, `verifier/`, `oracle/`, or the older layout's `tests/` and `solution/`), with a warning that names it; a link elsewhere in the task gets a warning and the task runs. Replace the links with copies.
- **Removed re-exports.** 129 unused names are gone from `benchflow.task.verifier`, `benchflow._utils.task_authoring`, `benchflow.task.acceptance_live` and `benchflow.rollout`. Import each from the module that defines it.
- **Smaller repository downloads.** A target with a folder (`bench run org/repo/path`, or `--source-repo` with `--source-path`) downloads only that folder. One SkillsBench task is now a few megabytes, not the whole repository. 0.9 clones into `.cache/datasets-v2/` and does not use 0.7's `.cache/datasets/`; delete that folder once no 0.7 run uses it.
- **`mean_reward` is a mean over every trial.** 0.7's `mean_reward` left out the trials with no score (an error, a verifier crash). 0.9's counts each of them as 0, so a job with unscored trials shows a lower figure for the same trials: a 0.7 job with 2 passes, 1 error and 2 verifier crashes printed `mean_reward=1.00`, and resumed under 0.9 it prints `0.40`. 0.7's figure is `mean_reward_scored_only` in summary.json and `bf.load_job`. summary.json gives each with its n (`mean_reward_n`, `mean_reward_scored_only_n`) and says which is which in `mean_reward_rule`. A task that fails before its trial folder exists (a `task.toml` typo) now gets a folder with its result, so `bench eval metrics` and `bf.load_job` count the same trials as the summary. The end-of-run line names both when they differ (`mean reward 0.40 over all 5 trials (unscored as 0) … scored only: 2/2 (100.0%), mean reward 1.00 (0.7's mean_reward)`), and a resumed 0.7 job says what its old summary said. Gate CI and dashboards on one of the two on purpose.
- **Reported verifier guard problems score 0.** A refusal, crash, incomplete check or load failure produces a named warning; a `registered` marker cannot override it. The absence of a reported problem does not prove that the guard loaded. Automatic rubric review still requires a completed verdict, so capture/export failures can remain unscored. See the [scoring contract](./sandboxes/threat-model.md#pytest-plugin-guard).
- **A rerun resumes a job whose comparable recorded settings match.** These include model, effort, harness, prompts, timeouts, budget, sandbox, skills and task identity when recorded. A mismatch starts a new automatic job; with `--job`, the old trials move to `<job>/.set-aside/` and the affected tasks run again. The run names the changed setting. For legacy jobs, missing fields remain unknown while explicit null is a known value; not every derived setting or package/catalog version can be proven equal. Use a fresh job for a fresh comparison. See [Resume and legacy limits](./jobs/resume.md).
- **Tasks read your variables only when you name them.** A task's `${NAME}` (in `[environment] env`, setup commands, Compose files) reads a variable of your environment only with `--allow-task-env NAME`; otherwise the task is refused before any sandbox starts. A Compose file that mounts, reads or builds from your machine is refused unless you set `BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1`. See [Environment variables](./jobs/environment-variables.md).
- **The verifier's pip reads no configuration file.** Pass index or wheelhouse options in test.sh, or set `PIP_*` variables in `[verifier] env`. See [Sandbox security](./sandboxes/security.md#changed-in-08-the-verifier-ignores-pip-configuration).
- **Cloud metadata is blocked on the default open network.** The agent's user cannot reach 169.254.169.254 and the other metadata endpoints. The agent, a root agent included, starts only once the block is proven; otherwise the trial stops before the agent. A root agent on Daytona, AgentCore or Apple container gets the provider's own metadata policy instead, with a warning. A Modal sandbox has no route to the endpoints; each trial proves it (`not_reachable`). A task about the metadata service sets `allow_cloud_metadata = true` in its environment table, or the run passes `--allow-cloud-metadata`.
- **Pulled task images stay on the machine.** Teardown removes only the images BenchFlow built. `bench cache clean --only pulled-images --remove` removes the ones BenchFlow pulled.
- **The task's test script is trusted.** BenchFlow guards the verifier's environment, but code that test.sh installs, builds or imports from the agent's files (`pip install -e .` of the workspace, or a module of the agent's that the tests import) runs in the grader and can change the result. Since 0.9.1, a `pytest.py` or another test-runner module the agent added to the workspace is removed before grading. Install test tools into the image or pin them, run tests with `$BENCHFLOW_PYTEST`, and do not run Python from the workspace folder. Name `$BENCHFLOW_PYTEST` in test.sh itself: BenchFlow prepares the runner only for a task whose tests folder or `[verifier] env` names it. See [Sandbox security](./sandboxes/security.md#what-you-must-trust).
- **Images that set bash start-up names are refused.** An image or sandbox environment that sets an exported bash function (`BASH_FUNC_name%%`), `SHELLOPTS` or `BASHOPTS` is refused at start on every sandbox, because bash would run it as root before BenchFlow's commands. Remove it from the image.
- **Names that pointed at another harness are renamed, with no alias.** `bench export --target task-toml`, `bench hub list --provider benchflow`, the `task-toml-compatible` task.md preset, `benchflow.hub.registry` and the `TaskTomlRoundTrip*` names. The hybrid verifier strategy and authoring profile are `criteria-runner`; the adapter is `benchflow.adapters.task_toml`, with `TaskTomlAdapter`, `from_task_toml_dir` and source `task-toml`. The runner receives `BENCHFLOW_CRITERIA_RUNNER_ROOT`, `BENCHFLOW_CRITERIA_RUNNER_CRITERIA` and `BENCHFLOW_CRITERIA_RUNNER_MANIFEST`, and writes `criteria-runner-manifest.json`. There are no aliases: old imports fail, old strategy values are refused, and all three old environment variables are silently ignored. An old value gets the ordinary unknown-value error, which lists the values that work. The full table is in the [CHANGELOG](https://github.com/benchflow-ai/benchflow/blob/main/CHANGELOG.md).
- **Datasets come from BenchFlow's registry only.** `-d name@version` reads BenchFlow's registry and the entries packaged with BenchFlow (`terminal-bench@2.0`, digest-checked). A dataset it does not have stops with the next step to take.
- **Only job folders BenchFlow made are resumed.** A named run refuses a folder with no BenchFlow marker (`<folder> is a folder BenchFlow did not make; nothing in it moves`); an unnamed run starts a new job beside it. A 0.7 job stopped before any trial finished now resumes.
- **`bench check` warns about a verifier's model use only when the task declares it** (`[verifier] models = true`, a provider key in the verifier env, or a recorded judge import).
- **Resuming a 0.7 job rewrites it.** Re-running the command that made a 0.7 job resumes it: its finished trials are reused, and its summary.json and results files are rewritten. The run log says the job was made by 0.7, and its own files are copied to `<job>/.pre-0.8/` first. Pass `--fresh` to leave it as it was. `bench eval resume <job>` cannot rebuild a 0.7 job's run (0.7 wrote no evaluation.json); it prints the command to re-run instead. Each 0.9 job records the BenchFlow version that made it (`created_by_benchflow_version` in evaluation.json), and each trial its own (`benchflow_version` in config.json).

## New in 0.9

- **First run:** `bench doctor` checks the machine and each agent's login, with a fix per problem. `bench eval smoke` runs a bundled hello-world task once per logged-in agent. `bench --help` and `bench new` end with the command that comes next. See [Quick start](./getting-started/quick-start.md).
- **Errors:** the end-of-run summary says why trials have no score, whose problem it is and what to do next. Most errors end with a `next:` line, `bench explain [CODE]` explains a cause, and commands that accept `--json` return errors as JSON objects on stdout. See [When a run fails](./jobs/when-a-run-fails.md).
- **Checks:** `bench check` refuses what `bench run` would refuse, before any sandbox starts: a task feature the chosen sandbox cannot run (code R401), or a missing judge or proxy extra. See [Check a task](./tasks/check.md).
- **Kept services:** a task whose tests connect to a server the agent starts lists its ports in `x-keep-services` (up to 8); the server survives the kill before grading. Published Terminal-Bench 2 tasks that need this are recognised by their content. See [Sandbox security: Kept services](./sandboxes/security.md#kept-services).
- **Agent judges:** a task.md draft 2 rubric criterion can be judged by an agent that reads the saved submission and can run it as an unprivileged user with no network. See [task.md draft 2](./tasks/taskmd-draft-2.md).
- **Job configs with a matrix:** a config that lists `agents`, `datasets` and `n_attempts` runs every agent on every dataset, `n_attempts` times each. See [Configs](./jobs/configs.md).
- **Controls:** `-a nop` does nothing, so a sound verifier scores it 0. The oracle (`bench run` with no `-a`) should score 1. Run both before any model.
- **CI:** `bench run` takes `--fresh`, `--job`, `--fail-under`, `--fail-on` and `--summary-out`. `bench eval resume JOB_DIR` finishes an interrupted job. SIGTERM cancels a run, deletes its sandboxes and exits 143. See [Auth: CI](./agents/auth.md#ci).
- **Budgets:** `--max-cost-usd`, `--max-sandbox-seconds`, `--max-tokens` and `--max-rollouts` (`bf.Budget` in Python) stop a job at a cap. See [Budgets](./reference/budget.md).
- **Reading results:** `bench results JOB` summarizes a job, and `bench results A B` compares two. In Python, use `bf.load_job` and `bf.load_trial`. Also pass@k, pass^k and versioned JSON exports. See [Analysing runs](./results/analysing-runs.md).
- **Python:** `bf.run_sync`, `bf.arun`, `bf.run_batch`, `Evaluation.stream()` and `Evaluation.resume(job_dir)`. See the [Python examples](./examples/python-sdk/).
- **Branching:** `bench branch` and `bf.branch` fork a run at a checkpoint into verifier-scored children, in parallel or nested, with the cost of each child. See [Branching](./jobs/branching.md).
- **Regrade:** `bench regrade` scores a run made with `--save-outputs` again after a verifier fix, without rerunning the agent. The new score sits beside the original. See [Regrade](./jobs/regrade.md).
- **Sandboxes:** `--sandbox remote-docker` runs tasks on a Docker host you control ([Remote Docker](./sandboxes/remote-docker.md)). A task can also score in a separate verifier sandbox that shares nothing with the agent's, and `--verifier-sandbox separate` does this for every task of a run ([Separate verifier sandboxes](./tasks/separate-verifier.md)).
- **Reward integrity (BenchShield):** `--integrity audit` writes a verdict next to each reward: `Checked`, `VectorExposed`, `AgentViolation` or `Rejected`. It never changes the reward. `--integrity strict` also runs the verifier in a separate verifier sandbox. See [Reward integrity](./integrity.md).
- **Native harness:** `--harness native` runs Claude Code and Codex through their own CLIs (`claude -p`, `codex exec`) instead of their ACP adapters. ACP stays the default. See [Native harness](./agents/harness.md).
- **Training:** `bench train stream` sends each finished rollout to a trainer while the job runs. `bench train token-coverage` checks gateway token capture, which records token ids and logprobs on `vllm/...` and `sglang/...` models. See [Streaming rollouts](./reference/rollout-stream.md) and [Token capture](./reference/token-capture.md).
- **Agents:** the Google Antigravity CLI (`antigravity`, alias `agy`) and Meta Muse Code (`muse-code`). Claude Code runs as 2.1.280 under `claude-agent-acp` 0.81.2, and Codex under `codex-acp` 2.0.1. The agents catalog is pinned to one commit per release.
- **Authoring:** `bench new` writes a canary GUID into each new task, and `bench check` warns when a task has none.
- **Task formats:** task.md draft 2 packages run natively, and a package can register a format so its own task folders run with no export step. See [task.md draft 2 packages](./tasks/taskmd-draft-2.md) and [Task formats](./tasks/task-formats.md).
- **Robots and simulators:** `benchflow.embodied` gives robot and simulator tasks an embodiment spec, the `robo` protocol, seeded rollouts and training export. See [Embodied rollouts](./tasks/embodied.md).

## Speed

A pre-release review measured build `2b5c678e6` against `dc509087a` and the Rust engine 0.8.2 on three Terminal-Bench 2 tasks (`fix-git`, `polyglot-c-py`, `pypi-server`), using mini-swe-agent 2.4.6 with recorded model replies on Docker, one trial at a time. Each arm ran three measured rounds after a warm-up. Median harness time, including test.sh, was **21.69 s** for this build, **25.24 s** for the earlier build and **22.65 s** for Rust. The paired change was **−3.30 s** against the earlier build (95% interval −3.55 to −3.10 s), and **−1.56 s** against Rust (−1.89 to +1.55 s); the Rust interval includes zero. Excluding test.sh, the medians were 13.23, 15.64 and 16.88 s. About 0.4 s of the measured reduction came from the setup-only CPU lift becoming available on this path. These are review measurements of that build, not final-release measurements.

- **Installs are built once.** On Docker, the in-sandbox model proxy and pinned agent installs become cached image layers, reused by later trials. On Daytona, the agent and model-proxy layers are used when this process already built them, or when the plan reuses the task image (`--trials` above 1 or `BENCHFLOW_DAYTONA_LAYER_REUSE=1`). One trial builds them; concurrent trials install in their sandbox instead of waiting. A single trial with no known layer installs in its sandbox. A new process does not assume Daytona retained a previous build; opt-in content snapshots have their own reuse rule. `--fresh-install` installs in every trial instead. `bench cache list` and `bench cache clean` show and remove what is cached ([Cache](./jobs/cache.md)).
- **Setup CPU only:** the direct Docker path can lift the CPU limit during agent installation. The task's limit is restored and checked before the agent starts; the agent and test.sh keep their declared CPU and memory limits.
- **Docker trials do less:** single-service tasks start without Compose, root's steps share one command channel, the next trials' sandboxes are prepared while the current ones run, and a finished sandbox is removed in the background.
- **A warm local runtime:** `bench daemon start` with `BENCHFLOW_DAEMON=auto` keeps BenchFlow and its model proxy loaded between `bench run` calls ([The local runtime](./local-runtime.md)).
- **The protected test runner only where a task names it:** a task whose tests folder and `[verifier] env` never name `$BENCHFLOW_PYTEST` skips its setup, which copied the image's `site-packages` (several GB on a torch image).

## Fixes

- A root-owned link to a missing native library no longer causes a sealing refusal when only root could create its target. This fixes the loader refusal on Terminal-Bench 2's `sam-cell-seg`; it does not promise its verifier passes. An unsafe library search path is refused before agent installation, with the path, owner, mode and remedy.
- Daytona guard setup records its steps and package-manager output in `daemon-guard.log`; a timeout names the last step. `timing.json` records `daemon_guard` separately.
- A pending model request counts as activity for up to three idle budgets; the wall deadline still applies. If a request never answers and outlives the idle budget, the result names the pending provider stall. Invalid provider responses and sandbox preparation failures have specific categories; these labels preserve the prior retry and resume decisions.

- Run logs prefix each trial line with its task. `--quiet` keeps warnings, errors and the final score. Network request chatter and routine protected-runtime messages move to DEBUG; a missing protected Python is recorded in the trial's `warnings` too.
- Argument parsing errors exit 2. `bench run` also uses 2 for an unknown agent, unsupported sandbox or empty task selection; missing credentials, missing sandbox extras and run failures exit 1. Some subcommands retain different exit rules; see the [CLI reference](./reference/cli.md).
- `bench hub show terminal-bench@2.0` lists the 89 tasks; `bench hub list` defaults to BenchFlow's dataset registry. Non-interactive trajectory commands fail with the argument to supply instead of waiting for input.
- A fresh unnamed rerun with a narrower task selection starts a new job and names the tasks left out. A named job retains its existing tasks and warns that the score includes them.
- A stopped job records a partial summary, including `--summary-out`. Result readers distinguish running, interrupted and never-started trials. A rollout stream reports a stopped job after yielding its finished rollouts.
- An unfilled task scaffold and malformed task configuration are reported before a sandbox build. A cold offline verifier cache gets a network/cache remedy; a checked warm cache still works offline.
- `total_cost_usd` is null when no trial reported cost. Treat it as unknown, not a free run. Training conversion skips control runs, and reports zero rows for a control-only job.

- Concurrent jobs on one Docker daemon no longer delete each other's containers and networks.
- SWE-bench Verified tasks keep the repository's own `conftest.py` files and run test.sh under the shell its first line names, so their tests are graded as written.
- The Python SDK's `bf.run(bf.Agent(...), bf.Environment.from_task(..., sandbox="docker"))` keeps the verifier's reward; it used to come back with a missing-reward error.
- An agent's installation no longer changes the task's environment (mini-swe no longer installs a system `python3`).
- Daytona trials are scored again: commands over 64 KiB travel as data, and folders move as one archive. An earlier 0.8.3 candidate scored none of 238 Daytona trials.
- On Docker, root's steps after the agent (the kill before grading, the workspace capture, the start-up hook clean-up) run on `bfx`, BenchFlow's own static helper, so they no longer need the image's Python.
- No image Python is started as root after the agent's turn, the egress proxy's included.
- A run in which no pytest reached the plugin guard is scored as test.sh wrote it, instead of ending unscored.
- Daytona starts survive bursts of 64 trials, a sandbox is retried only where a fresh one helps, and only sandboxes Daytona has already stopped are deleted automatically (auto-stop after 15 minutes without activity).
- Every volume a Docker trial causes is labelled when it is created and removed with the trial (one exception: see [Known limitations](#known-limitations)).
- Terminal-Bench 2's `largest-eigenval` and `sqlite-with-gcov` oracles score again, and `mteb-retrieve`, `pytorch-model-recovery` and `mteb-leaderboard` no longer run out of disk on Daytona.

## Known limitations

- **Kept services can retain an open writer to an existing workspace file.** Changing a single-name file's owner and mode does not revoke a descriptor a service already opened for writing. The close step replaces non-root files, including shared hard-linked files, but does not copy every root-owned file. A kept service can still change such a file while tests read it. Use a separate verifier when its inputs must not change during grading.
- **Closing shared files for kept services can consume extra disk.** The root-owned copy writes sparse files in full: a sparse file with a logical size of 4 GiB can require 4 GiB of disk. Without a protected Python, the fallback file walk can also fail on paths longer than 4,096 bytes; this stops hardening rather than accepting the workspace.

- **Tasks written for a root agent can fail with the default user.** The agent runs non-root while the verifier normally runs as root. It cannot install system packages or write under `/usr`, `/etc` or `/opt`; packages installed in its user site may be invisible to root's tests. The historical Terminal-Bench 2.0 scan found root-only writes in 33 of 89 reference solutions; five tests read those writes directly (`build-pmars`, `build-pov-ray`, `mailman`, `nginx-request-logging`, `sqlite-with-gcov`). The other 28 depend on the chosen solution; this scan does not establish that every alternative solution needs root. `headless-terminal` also failed when the agent installed `pexpect` only for its own user. `--sandbox-user none` runs the agent as root and permits those writes, but root can undo the PATH lock, protected runtime and workspace freeze and change what the verifier runs. Use it only for trusted agents; keep its scores separate from non-root runs.
- **Some task images cannot pass the protected runtime's loader checks.** If the native loader searches or loads libraries from a folder another user can write, BenchFlow refuses its root steps and the trial is unscored, including the oracle. Terminal-Bench 2's `make-mips-interpreter` is one such task. Use an image whose library files and search folders are root-owned and closed to other writers.
- **Ten of Terminal-Bench 2's 89 tasks keep services that native Daytona cannot freeze for a non-root model agent.** These trials are refused: `configure-git-webserver`, `git-multibranch`, `hf-model-inference`, `install-windows-3.11`, `kv-store-grpc`, `mailman`, `nginx-request-logging`, `pypi-server`, `qemu-alpine-ssh` and `qemu-startup`. Use Docker or remote Docker. The oracle and nop do not have this model-agent refusal; their separate image and verifier limitations still apply.

The [CHANGELOG](https://github.com/benchflow-ai/benchflow/blob/main/CHANGELOG.md) lists them in full. The ones task authors most often meet:

- The task's test script is trusted; see [What to do differently](#what-to-do-differently).
- Three checks of the protected runtime's native-library search paths remain incomplete in 0.9.0. Each needs an image that points a search path into a folder the agent can write. See the [threat model](./sandboxes/threat-model.md#known-gaps).
- A task that keeps services cannot be graded on an image without Python when the agent runs as a non-root user: the trial is refused with a named message. Terminal-Bench 2's `configure-git-webserver` is such a task; its oracle run scores, and runs with an agent under the default user are unscored.
- A new connection is refused while kept services retain processes from an earlier role or scene, before any later login is staged. Use a fresh rollout. See the [threat model](./sandboxes/threat-model.md#known-gaps).
- Under `denylist` and `allowlist`, an image with no root-only `python3` in `/usr/local/bin`, `/usr/bin` or `/bin` (a conda or uv-only interpreter) stops before the agent.
- A task that calls `$BENCHFLOW_PYTEST` only from inside its image receives a stub that prints an error and exits 4. The trial follows test.sh's reward, usually 0; name the runner in the tests folder or `[verifier] env` to prepare it.
- A task image tag that someone retags on the same Docker daemon during a trial's start can leave one volume behind; remove it with `docker volume rm`.
- On Docker, an agent that leaves about 90,000 files with start-up hook names, or a folder set to mode 000 under a Compose file that drops `DAC_OVERRIDE` and `DAC_READ_SEARCH`, can make the hook clean-up fail, so the trial is unscored and retried instead of scored.
- Every trial keeps a root-only copy of the pre-agent workspace; a task whose image ships a workspace of several GB can run out of disk on a small sandbox.

## Deprecated

Each of these still works and warns when used:

- `benchflow.SDK`: use `bf.run(bf.RolloutConfig(...))`, which takes the same keyword arguments.
- `bf.snapshot`, `bf.restore` and `bf.list_snapshots`: use `bf.workspace_snapshot`, `bf.workspace_restore` and `bf.list_workspace_snapshots`.
- `RuntimeResult`: use `RolloutResult`. Every use of the name warns, so an `isinstance(result, RuntimeResult)` check, which is now False, is pointed out.
- The unused `RuntimeConfig.max_rounds` and `snapshot_policy` fields, whenever they are passed (a copy made with `dataclasses.replace` stays silent), and `reward_stream` when it is not `True`.
- The older command spellings: see [New command names](#new-command-names).

## Next steps

- Check your machine: `bench doctor`.
- Run a task with the new verbs: [Quick start](./getting-started/quick-start.md).
- Every flag: [CLI reference](./reference/cli.md).
