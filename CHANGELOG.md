# Changelog

## [0.9.1]

0.9.0 with fixes for problems found after its acceptance. Each was fixed first on the 0.10 line or on public main, with a test that fails on 0.9.0 and a real run.

### Security

- **LiteLLM 1.103.3, and patched TLS and HTTP clients** (as public #1196). LiteLLM 1.91.5 carried two advisories (GHSA-7hp6-4w63-5g45 and GHSA-g5ff-637f-6q2m), a streaming stall when a cached upstream handler was collected mid-stream, and a `TypeError` in the cleanup of a disconnected stream whose request had `"metadata": null`. The floors are now `cryptography>=50`, `certifi>=2026.10.19` and `aiohttp>=3.14.4`, and oauthlib is 4.0.1 (CVE-2026-49272): `pip-audit` of the lock and of both proxy pin files the wheel ships finds nothing. The Bedrock adaptive-thinking patch and the Gemini pass-through patch follow LiteLLM 1.103's calls. Bedrock Invoke's system-message patch (#1135) now runs on top of LiteLLM's own handling: it still keeps a mid-conversation system message in place only on the models Bedrock was seen to accept one from, with only its content, and moves Claude Code's cache breakpoint off trailing system messages. The restricted-model allowlist is rebuilt from the new model map, and the no-web guard refuses every deep research model, whatever its provider calls it: 1.103's map adds Perplexity's deep research models under another provider's name.

### Fixed

- **A long streamed thought is merged in linear time** (as public #1165). Text runs were joined with `+=` per chunk, so a thought of 20,000 chunks took 33 ms per live snapshot; they are merged with one join.
- **A cancelled run's trajectory keeps its last updates** (as public #1165). The live trajectory is written at most every 250 ms, and a cancelled run never reached the final write, so the file missed up to 250 ms of updates; the held-back tail is now published when the run is torn down.
- **gemini on Daytona starts again** (#1208). Daytona refuses SSH access longer than 1440 minutes, and 0.9.0 asked for 2880, so every launch on the SSH transport failed with `expiresInMinutes must be between 1 and 1440`.
- **A root-owned reward on rootful Docker is read** (#1202). A verifier that wrote its reward as root with mode 0600 on mounted outputs scored 0 without an error.
- **A workspace `pytest.py` no longer replaces pytest.** When test.sh ran `python3 -m pytest` from the workspace, an agent's `/app/pytest.py` (or a `_pytest`, `pluggy`, `iniconfig` or `py` module or package) ran as root in place of the grader, and an unsolved task scored 1.0. Before grading, BenchFlow now removes each such module the pre-agent copy of the workspace has no entry for; a repository's own stays.
- **pi-acp connects again** (#1191). It installed the deprecated `@mariozechner/pi-coding-agent` (0.73.1), whose `session/new` never answered. Pinned to `@earendil-works/pi-coding-agent@0.81.1` and `pi-acp@0.0.34`.
- **openhands runs a model on Daytona.** uv's git cache lock was left writable by everyone, so the in-sandbox model proxy refused BenchFlow's uv cache and the trial failed before the first prompt.
- **A Docker without buildx runs local trials.** Its legacy builder cannot build a cached layer's `RUN --mount` and deletes the image-id file when it fails, so a trial that needed the model proxy's, the agent's or the verifier's layer stopped with `FileNotFoundError` on `benchflow-layer-*.iid`. A failed layer build is now reported with the build's last lines, and the trial installs in the sandbox instead.
- **A score in `reward.txt` beside a `reward.json` of details scores** (#1192). Details that are not numbers are kept beside the score; a number among them is still a metric held to the reward range.
- **`bench results <job> --json` reads a job 0.9 resumed from 0.7.** The job's `.pre-0.8/` backup made it look like a folder of jobs, and `--json` exited 2.

- **The verifier no longer uses the agent's workspace as its home**. With `WORKDIR /root` the verifier ran as root with `HOME=/root`, so an env script the uv installer's preamble sources (`~/.local/bin/env`) or a `~/.curlrc` the agent left there ran while it graded. In SkillsBench's default set (87 tasks at f860ecf), 45 source that env script as root, and in 32 of them the workspace is /root; a planted one that writes the reward scored 1 on 32 of 32 tried with BenchFlow 0.6.3 and on 7 of 7 tried with 0.9.0. When the agent could write `/root`, the verifier now gets a new root-only `HOME` (XDG folders inside, empty `.curlrc` and `.wgetrc`). A root-only `/root` stays. Tests that read the agent's output through `~` with `WORKDIR /root` use its path, or set `HOME = "/root"` in `[verifier] env`.
- **`verifier.restore_workspace` is refused with what to do instead:** check input digests in test.sh.
- **Default models their vendors no longer serve are replaced.** `bench eval smoke` tried Codex with `gpt-5.5`, which left ChatGPT sign-in on 2026-10-14, and Gemini with `gemini-2.5-flash`, which Google has closed to new API keys. It now uses `gpt-6-luna` (on every ChatGPT plan) and `gemini-3.5-flash-lite`, which is also the default for `--agent gemini` without `--model` and for `bench review`. The README's run-config example and the three `*-gemini-flash-lite.yaml` configs used `gemini-3.1-flash-lite-preview`, which Google has shut down; they use `gemini-3.5-flash-lite`. A test names any default or shipped example that uses a model on the retired list.

### Known issues

- **The model proxy's lazy start is off with LiteLLM 1.103.3.** It defers route analysis and the tokenizer load only for the LiteLLM, FastAPI and pydantic versions it was checked against, and this lock moved LiteLLM and pydantic; the proxy starts as LiteLLM does.
- **A repository's own `conftest.py` in the workspace is still removed before grading** (#1171), so a correct answer whose tests need its fixtures scores 0, as on SWE-bench Verified's Sphinx, Flask, pytest and seaborn tasks. Fixed in 0.10.0rc1. On 0.9.1, `cleanup_conftests = false` in the task's `[verifier.hardening]` keeps them, and also keeps any `conftest.py` the agent planted, which can then change the result.

### Packaging

- **The build backend is pinned (`hatchling==1.32.4`).** Built from 0.9.0's source with hatchling 1.33.0, the wheel differed from its accepted SHA-256 in one line of `WHEEL`.

## [0.9.0]

BenchFlow 0.9.0 rebuilds and expands the Python SDK and CLI after 0.7.11. Python remains the main implementation, with the static Linux bfx helper for sandbox operations. The Rust engine's 0.8.2 release is a separate implementation. Upgrading from 0.7? Read [What's new in 0.9](docs/whats-new-0.9.md) first.

**Before you compare scores with a 0.7.11 run.** These changes can move a score:

- Solve rate, pass@k, pass^k and mean reward now count every trial. A trial with no score (an error, a verifier crash) counts as 0. The 0.7 figures stay beside them as `*_scored_only`.
- A reward outside the task's range is no longer clipped. It scores 0 with a `reward_out_of_range` warning.
- Reported verifier guard problems score 0 with a named warning. Automatic rubric review still requires a completed verdict; capture/export failures may remain unscored even after script verification. See the [scoring contract](docs/sandboxes/threat-model.md#pytest-plugin-guard).
- A pytest plugin that test.sh installs after the agent stops is never loaded. Pin it in `[verifier] pytest_plugin_releases`, or install it into the image.
- The verifier's pip reads no configuration file. Pass index or wheelhouse options in test.sh.
- Running the same command again resumes a matching job based on the recorded settings BenchFlow can compare. Missing legacy fields remain unknown; use a fresh job for a fresh comparison. See [Resume](docs/jobs/resume.md).
- Several kinds of task that were graded wrongly are graded correctly now; see [Fixed](#fixed).

### What's new

- **One command set.** `bench new`, `check`, `import`, `run`, `branch`, `results`, `view`, `regrade`, `export`, `doctor` and `agents`, the same verbs as the Rust engine. `bench run` takes every `bench eval run` setting, and without `-a` it runs the oracle. The older spellings (`bench eval run`, `bench tasks check`, ...) still work and print one note with the new spelling.
- **Errors say what happened, whose problem it is and what to do next.** Most errors end with a `next:` line. `bench explain [CODE]` explains why a trial has no score. On commands that accept `--json` (such as `explain`, `results` and `hub`), errors are JSON objects on stdout (`error`, `code`, `owner`, `next`, `detail`, `exit_code`).
- **`bench check` refuses what `bench run` refuses,** before any sandbox starts: a task feature the chosen sandbox cannot run (code R401), or a judge or proxy extra that is missing.
- **Services the agent leaves running.** A task can name up to 8 TCP ports in `x-keep-services`. The agent's processes listening on them survive the kill before grading, paused while the grading inputs are closed. Published Terminal-Bench 2 tasks that need this (hf-model-inference, pypi-server and 8 more) are recognised by their content.
- **Grade any run in a separate sandbox.** `--verifier-sandbox separate` (or `--separate-verifier`) runs every task's verifier in a fresh sandbox that receives only the workspace and the files the task declares. `--separate-verifier-include PATH` carries outputs from outside the workspace. A task it cannot run is refused before a job folder exists.
- **Agent-judged rubric criteria** in task.md draft 2 tasks. The judge reads the saved submission, can run it as an unprivileged user with no network, and records each session in `verifier/judgments.jsonl`.
- **Job configs with a matrix of agents and datasets.** A job config that lists `agents`, `datasets` and `n_attempts` runs every agent on every dataset, `n_attempts` times each, and writes `matrix-summary.json` with pass@k per cell. A setting BenchFlow cannot honour exactly is refused by file, line and key.
- **`bench cache list` and `bench cache clean`** show and remove BenchFlow's cached agent, proxy and verifier layers, its Daytona content snapshots and the task images it pulled. `clean` removes nothing without `--remove`.
- **Agents.** Meta Muse Code (`muse-code`). Antigravity's agy 1.3.5, with any Gemini API model. OpenCode on Gemini with its own Google client. mini-swe works again (mini-swe-agent 2.4.6, every dependency pinned). `claude-agent-acp` 0.81.2 with Claude Code 2.1.280, which runs `claude-opus-5-5`. `codex-acp` 2.0.1, which knows `gpt-6.1-sol`. Each release pins the agents catalog to one commit.
- **Also new since 0.7.11:** Docker hosts you control (`--sandbox remote-docker`), regrade without rerunning the agent, reward integrity audits, a native harness for Claude Code and Codex, branching, budgets, rollout streams for trainers, task formats and embodied rollouts. Each is described [in detail](#in-detail-every-change-since-0711) below.
- **Records you can read back.** `timing.json` names more steps (agent install, proxy start, each hardening step, cleanup, waits). Every job and trial records the BenchFlow version that made it. `cloud-metadata.json`, `kill-before-grading.json`, `locked-system-path.json` and `verifier/trusted-path.json` say what the sandbox did.
- **Install.** A new [install guide](docs/getting-started/install.md); every release tag carries a `constraints.txt` with the exact dependency versions it was tested with; `bench doctor` says which `bench` your shell runs, this one or the Rust engine's.
- **Dependency security updates.** The lock, matching constraints and packaged model-proxy requirements include the refreshed dependency versions. Use the release's matching constraints for those pins. Package metadata keeps its existing minimum versions, so unconstrained installation is a separate resolution, not the locked environment.
- **Patched installer trust store.** Sandbox installers and CI use checksum-pinned uv 0.13.4 instead of 0.13.1, removing the affected emSign root certificate (GHSA-hq85-5cpj-mr62). The managed Python remains pinned to 3.12.15.
- **Agent dependency security updates.** deepagents keeps its agent version and updates certifi from 2026.7.22 to 2026.10.23. The pinned harvey-lab harness updates anyio 4.13.0 → 4.14.2, certifi 2026.2.25 → 2026.10.23, cryptography 46.0.6 → 50.0.2, idna 3.11 → 3.16, lxml 6.0.2 → 6.1.0, pillow 12.2.0 → 13.0.0, pyasn1 0.6.3 → 0.6.4, soupsieve 2.8.3 → 2.9 and urllib3 2.6.3 → 2.8.0. Both complete requirement lists retain exact versions and artifact hashes; the harness source commit is unchanged.

### Speed

A pre-release review measured build `2b5c678e6` against `dc509087a` and the Rust engine 0.8.2 on three Terminal-Bench 2 tasks (`fix-git`, `polyglot-c-py`, `pypi-server`), using mini-swe-agent 2.4.6 with recorded model replies on Docker, one trial at a time. Each arm ran three measured rounds after a warm-up. Median harness time, including test.sh, was **21.69 s** for this build, **25.24 s** for the earlier build and **22.65 s** for Rust. The paired change was **−3.30 s** against the earlier build (95% interval −3.55 to −3.10 s), and **−1.56 s** against Rust (−1.89 to +1.55 s); the Rust interval includes zero. Excluding test.sh, the medians were 13.23, 15.64 and 16.88 s. About 0.4 s of the measured reduction came from the setup-only CPU lift becoming available on this path. These are review measurements of that build, not final-release measurements.

- **Installs are built once and reused.** On Docker, the in-sandbox model proxy and the pinned agent installs are built into cached image layers per task image, shared by every BenchFlow process on the machine. On Daytona, the agent and model-proxy layers are used when this process already built them, or when the plan reuses the task image (`--trials` above 1 or `BENCHFLOW_DAYTONA_LAYER_REUSE=1`). One trial builds them; concurrent trials install in their sandbox instead of waiting. A single trial with no known layer installs in its sandbox. A new process does not assume Daytona retained a previous build; opt-in content snapshots have their own reuse rule. `--fresh-install` (or `BENCHFLOW_FRESH_INSTALL=1`) installs in every trial instead. uv and Python in the sandbox are pinned and checksum-verified, so images without Python work too.
- **Setup CPU only.** On Docker's direct path, BenchFlow can temporarily raise the CPU limit while installing the agent. It restores and reads back the task's declared limit before launching the agent; the agent and test.sh keep the task's CPU and memory limits.
- **Docker trials do less.** A single-service task starts and stops through the Docker Engine API instead of a dozen `docker compose` commands (`BENCHFLOW_DIRECT_DOCKER=0` keeps Compose). Root's setup and verifier steps go through one command channel. The verifier's preparation is one script, and independent steps before the verifier run side by side. Containers stop together, and a finished trial's sandbox is removed in the background while the next trial starts.
- **Preparing the next trial.** With more tasks than `--concurrency`, the next trials' sandboxes are prepared while the current ones run (`BENCHFLOW_WARM_POOL=off` turns this off). Missing task images are pulled when the job starts. On Docker, each trial's host model proxy starts while its sandbox starts.
- **More trials at once.** Each Docker trial network takes a small subnet, so `--concurrency` above about 30 works on a stock daemon. Above `--concurrency 16`, trial starts are paced without holding a slot.
- **A warm local runtime: `bench daemon start`.** Opt in with `BENCHFLOW_DAEMON=auto` to keep BenchFlow and its model proxy loaded between non-interactive calls. `--prepare N` prepares sandboxes for the next identical call.

- **CLI imports.** It imports only the command that runs.
- **Daytona** polls for command results and sends commands of 30 s or less in one request.
- **The verifier's tests run at their own speed.** coverage.py no longer traces them unasked.

BenchFlow prepares its protected test runner (`$BENCHFLOW_PYTEST`) only for a task whose tests folder or `[verifier] env` names it (see [Fixed](#fixed)), so other tasks skip copying the image's `site-packages` for it. A task that names the runner only inside its image gets none (see [Known limitations](#known-limitations)).

For six Terminal-Bench 2 tasks (`count-dataset-tokens`, `password-recovery`, `polyglot-c-py`, `sanitize-git-repo`, `sparql-university`, `vulnerable-secret`), a setup-only Daytona check used mini-swe with a loopback endpoint and no prompt. At build `56ca86da8`, the cold single-trial median was 123.9 s using fresh installs; the first round building layers for four planned trials was 136.3 s, and a warm round was 49.6 s. The earlier build `dc509087a` had median 124.9 s across four rounds. These include setup, start, agent installation, connection and cleanup, not an agent turn or verification; they are provisional measurements of those builds.

### Fixed

**Grading.**

- A repository's own `conftest.py` files are kept for grading; only the ones the agent added are removed. SWE-bench Verified's Sphinx, pytest, seaborn and Flask tasks are graded correctly again. matplotlib's SWE-bench tasks are graded on Daytona again (#1168). **Correction:** in 0.9.0 and 0.9.1 a repository's own `conftest.py` in the workspace was still removed before grading, so a correct answer whose tests need its fixtures still scored 0; fixed in 0.10.0rc1 (#1171).
- test.sh runs under the interpreter its first line names, else bash. SWE-bench Verified and SWT-bench scripts whose `#!/bin/bash` is not on line 1 used to run under dash.
- The pytest plugin guard trusts the image's own plugins under Debian's `root:staff` folders (#1176), a plugin a fresh `uv init` project lacks (#1175), and every wheel of a pinned release. A test.sh that adds pytest-json-ctrf with `uv add` and uses `--ctrf` is graded again (#1170): BenchFlow supplies `ctrf` from its protected verifier runtime. Other unpinned plugins installed by test.sh after the agent stops are refused; a reported refusal scores 0 with a named warning.
- On images without Python, the verifier keeps the image's root-owned tool folders (cargo, mvnd) on its `PATH` (#1174). Each folder it leaves out is listed in `verifier/trusted-path.json` with the reason.
- A `reward.json` with extra fields that hold no number scores (#1178).
- Daytona Compose tasks that mount the task folder find their files there (#1181).
- An ACP agent's final reply is written to `/logs/agent/output.txt`, where registry verifiers read it (#1182). OpenCode can write its temp folder (#1183).
- A task with a small disk no longer runs out of space installing the proxy and the agent (#1169).
- The workspace freeze keeps the image's own symlinks, including a venv's link to its interpreter, and removes the ones the agent added.
- The oracle can use git repositories the image built inside the workspace (Terminal-Bench 2's `git-leak-recovery` scored 0).
- The Python SDK's `Agent + Environment` form keeps its reward on Docker. `bf.run(bf.Agent(...), bf.Environment.from_task(..., sandbox="docker"))` with the task's own test script wrote the reward to one folder and read it from another, so the run came back with a missing-reward error although the verifier had scored it. This was present since 0.5. The rollout now uses the environment's own folders, and `Environment.from_task(..., jobs_dir=...)` chooses where they go.
- An agent's installation no longer changes the task's environment: mini-swe no longer installs a system `python3` into the task image, and codex-acp and OpenHands no longer leave a `umask 077` for the agent's commands.
- With no reported plugin-guard problem, a run is scored from test.sh's reward, even when no pytest reached the guard. Terminal-Bench 2's `qemu-startup` and `qemu-alpine-ssh` can therefore score 0 when test.sh fails before pytest starts. A reported guard refusal, crash, incomplete check or load failure scores 0 with a warning; a `registered` marker cannot override it. The absence of a reported problem does not prove that the guard loaded.
- When a left-out `PATH` folder is listed in `verifier/trusted-path.json`, its reason names the rule that left it out (`/tmp/tools: under /tmp`), not the folder again.
- BenchFlow prepares its protected test runner (`$BENCHFLOW_PYTEST`) only for a task whose tests folder or `[verifier] env` names it. This holds on every backend, whatever user the agent runs as, and in a separate verifier sandbox too. Preparing it copied the image's whole `site-packages` into the sandbox (5 to 8 GB on a torch image), and Terminal-Bench 2's `mteb-retrieve`, `pytorch-model-recovery` and `mteb-leaderboard` ran out of disk on Daytona. No Terminal-Bench 2 test.sh calls the runner. A task that names it only in a file inside its image gets none (see [Known limitations](#known-limitations)).
- A Python start-up hook that only root could have written during the agent's turn is kept for grading: owned by root, closed to group and others, in folders only root can write up to `/`, outside the workspace and the other agent-writable trees, and with one name (a link count of 1). A solution that pip-installs a package as root keeps its `.pth` file; Terminal-Bench 2's `largest-eigenval` (eigenpy's `cmeel.pth`) scored 0. On Docker, where `bfx` does the clean-up, the trial log names the first five hooks it kept and their total count; on other backends, and on Docker without `bfx`, the line goes to the step's own error output, which the trial records do not carry. The rule applies only when the agent runs as a non-root user in its own sandbox. When the agent runs as root, or in a separate verifier sandbox, the earlier rule stays: every new or changed hook is removed.
- A program added to a `PATH` folder during the agent's turn whose route leads into the workspace stays on the verifier's `PATH` when its name was not in that folder before the turn. Root records each `PATH` folder's names before the agent starts. Terminal-Bench 2's `sqlite-with-gcov` solution links `/usr/local/bin/sqlite3` to `/app/sqlite/sqlite3`. The agent's user cannot add a name to such a folder, so the name was added by root (the task's solution or a service) or, in a folder another user owns, by that user. The verifier then runs, under that name, whatever is at that workspace path after the turn, which can be the agent's file. A name the image already had is still refused, whatever it points to now, so an image cannot hand the verifier a `python3` or `pytest` that runs the agent's files. The rule applies only when the agent runs as a non-root user in its own sandbox; with a root agent, in a separate verifier sandbox, or when the record could not be taken, such a program is left out as before. A kept program is not listed in `verifier/trusted-path.json`.
- When BenchFlow's copy of the image's Python fails once, it is tried again, and a failed copy is not cached (`usaco/987` lost its protected Python on Daytona).

**Keeping the agent away from its grade and from your secrets.**

- The kill before grading freezes the agent's processes first, so fork loops and processes that keep changing their pid cannot outlast it.
- An agent's launch cannot gain privileges through setuid programs such as `sudo`, or take control of the firewall. An image that would hand the agent the firewall is refused before the agent starts.
- Under the default open network, the agent's user cannot reach the cloud metadata endpoints. The agent, a root agent included, starts only once that block is proven; otherwise the trial stops before the agent. The one exception is a root agent on Daytona, AgentCore or Apple container, where the provider's own metadata policy applies and the trial logs a warning (`require_cloud_metadata_block = true` stops it instead). A Modal sandbox has no route to the endpoints: each trial proves it with a connect to every endpoint and records `not_reachable`. A task about the metadata service itself opts out with `allow_cloud_metadata`.
- A task's files read a variable of your environment only when you name it (`--allow-task-env NAME`). A Compose file that mounts, reads or builds from your machine is refused unless you set `BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1`.
- An agent starts with only the credentials of the route BenchFlow chose. Logins for other agents and providers stay out of its sandbox. One case is not covered yet: a task that keeps services and has more than one role (see [Known limitations](#known-limitations)).
- When the agent runs as its own user (the default), BenchFlow's own root steps after the agent's turn never start the image's Python. They use a copy of the image's Python and standard library, made and sealed before the turn and always started isolated (`-I -S -B`) from `/`. This includes the egress proxy under `denylist` and `allowlist` (its start, its health check and its certificate lookup). The installer of the pytest plugin guard starts no interpreter at all: it reads where each interpreter looks from a record taken before the turn. An interpreter missing from that record, or changed since, gets no copy of the guard; it reaches the guard only through the verifier's `PYTHONPATH`, and the trial log has a `benchflow:` line naming it. A separate verifier sandbox takes that record before the agent's workspace is copied in. Code that test.sh itself runs is the task's (see [Known limitations](#known-limitations)).
- A task that lets the agent switch to other accounts (`x-agent-identity-switch`) is refused on sandboxes that cannot keep no-new-privileges on the agent's container (native Daytona, Modal, AgentCore, Apple container).
- Files BenchFlow copies out of the sandbox (verifier output, artifacts, judge evidence) are published without following links or crossing mounts the agent could have planted.
- The model proxy reads no `.env` file it happens to find (see [Changed](#changed)).
- One list of shell start-up variables for every sandbox. The loader and shell start-up variables (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `BASH_ENV`, `ENV` and 9 more) are emptied for BenchFlow's own root shells and given back only to the task's own commands, on every backend; Daytona used to refuse such images. An image or sandbox environment that sets an exported bash function (`BASH_FUNC_name%%`), `SHELLOPTS` or `BASHOPTS` to a value is refused at start on every backend, because bash would run it as root before any command BenchFlow sends: on Daytona an image function did run as root in root's shells. The message names the variables, never their values. An empty value is accepted.
- On Daytona, a root command too large for one request is staged in a folder that is checked to be root's alone at the moment of use, and it runs whole or not at all.
- **On Docker, root's steps after the agent no longer need the image's Python.** BenchFlow now ships `bfx`, a small static helper, in the wheel (Linux x86_64 and aarch64, pinned by hash), and mounts it read-only at `/.bfx` in every Docker sandbox, local, remote or Docker Desktop. The kill before grading, the workspace capture, the workspace's change of owner, clearing `/logs/verifier`, the start-up hook clean-up and root's command channel run on it. So the hook clean-up also runs on images without Python and after the agent removed `python3`. `BENCHFLOW_BFX_DELIVERY=off` turns it off; a machine of another architecture runs without it. Daytona, Modal, AgentCore and Apple container keep their earlier steps. A task that mounts anything at or below `/.bfx` is refused.
- Where `bfx` is not delivered, the workspace freeze leaves a file that has a second name with its owner unchanged, as `bfx` does. A start-up hook the agent's user owns and hard-linked into the workspace stays the agent's, so the hook clean-up removes it even when the workspace's `conftest.py` sweep has removed its second name first. For kept services, the close step renames root-owned copies over names that still refer to non-root files, including files with several hard links, and checks the final owners and modes.

**Runs and results.**

- A root-owned link to a missing native library no longer causes a sealing refusal when only root could create its target. This fixes the loader refusal on Terminal-Bench 2's `sam-cell-seg`; it does not promise its verifier passes. An unsafe library search path is refused before agent installation, with the path, owner, mode and remedy.
- Daytona guard setup records its steps and package-manager output in `daemon-guard.log`; a timeout names the last step. `timing.json` records `daemon_guard` separately.
- A pending model request counts as activity for up to three idle budgets; the wall deadline still applies. If a request never answers and outlives the idle budget, the result names the pending provider stall. Invalid provider responses and sandbox preparation failures have specific categories; these labels preserve the prior retry and resume decisions.

- Run logs prefix each trial line with its task. `--quiet` keeps warnings, errors and the final score. Network request chatter and routine protected-runtime messages move to DEBUG; a missing protected Python is recorded in the trial's `warnings` too.
- Argument parsing errors exit 2. `bench run` also uses 2 for an unknown agent, unsupported sandbox or empty task selection; missing credentials, missing sandbox extras and run failures exit 1. Some subcommands retain different exit rules; check the command's help and the [CLI reference](docs/reference/cli.md).
- `bench hub show terminal-bench@2.0` lists the 89 tasks; `bench hub list` defaults to BenchFlow's dataset registry. Non-interactive trajectory commands fail with the argument to supply instead of waiting for input.
- A fresh unnamed rerun with a narrower task selection starts a new job and names the tasks left out. A named job retains its existing tasks and warns that the score includes them.
- A stopped job records a partial summary, including `--summary-out`. Result readers distinguish running, interrupted and never-started trials. A rollout stream reports a stopped job after yielding its finished rollouts.
- An unfilled task scaffold and malformed task configuration are reported before a sandbox build. A cold offline verifier cache gets a network/cache remedy; a checked warm cache still works offline.
- `total_cost_usd` is null when no trial reported cost. Treat it as unknown, not a free run. Training conversion skips control runs, and reports zero rows for a control-only job.

- A closed terminal or a dropped SSH session stops a run cleanly and removes its sandboxes (exit 129).
- After a run is killed (`kill -9`, out of memory, a preempted VM), the next Docker job, `bench eval resume` or `bench sandbox cleanup` removes its containers and networks.
- A dataset run (`-d <dataset>`) whose `--include` or `--exclude` selects no task is refused instead of running the whole dataset.
- A `--matrix` run no longer stops after the first job with an error, and always writes `matrix-summary.json`.
- A run no longer hangs when Python's garbage collection cleans up a finished sandbox while another trial stages its task.
- Copies out of the sandbox get a deadline that grows with their size, and a failed copy is recorded instead of losing the manifest.
- **Daytona trials are scored again.** On an earlier 0.8.3 candidate, a Daytona run of Terminal-Bench 2 scored none of its 238 trials: the verifier's preparation went out as one shell argument, over Linux's 128 KiB limit, and the read of the image's users lost its NUL bytes. Every backend now hands its provider at most 64 KiB per command. A larger command travels as data (on the command's input, or as a root-only staged file on Daytona) and runs whole or not at all. Folders move as one archive, so a tests folder of 2,000 files or more uploads on Daytona, and a Compose environment of 16 KB or more works on Docker-in-Docker. BenchFlow's 14-task transport check (oracle and nop) scores as expected on Docker and on Daytona. Trials lost at start under load are covered by the next entry.
- **Daytona starts hold up under load, and Daytona decides when a sandbox is abandoned.** With 64 trials starting at once, the readiness check lost 13 of them before and loses none now. The run's preflight check no longer counts the Daytona SDK's import against its time limit (with many runs starting at once, that import alone could pass the limit and stop the run), and only a rejected API key stops a run. A sandbox that fails to start is retried with a fresh one only where that can help (Daytona reports `ERROR`, or a build with no fallback failed); a cached install layer that fails to build is built once, then the plain task image is used. Sandboxes are created with Daytona's auto-stop at 15 minutes without activity (it was 24 hours), and BenchFlow keeps a sandbox active while a run holds it. Automatic clean-up deletes only sandboxes Daytona has already stopped, archived or failed and that no live run holds; a running one is reported, never deleted. `bench sandbox cleanup --all` follows the same rule, `--id ID` deletes the sandbox you name in your owner scope whatever its state (it also applies the age rule to your other idle sandboxes), and `--max-age` keeps its warning. Sandboxes a run could not delete are listed in `~/.local/state/benchflow/daytona-leftovers.jsonl` (`BENCHFLOW_DAYTONA_LEFTOVERS`), which `bench sandbox cleanup` reads. A trial whose sandbox Daytona stopped says so: "Daytona stopped the sandbox (SANDBOX_NOT_RUNNING). It stops a sandbox after 15 min without activity from its controller: was this process paused, asleep or offline that long?"
- **Every volume a Docker trial causes is named and labelled when it is created.** The image's `VOLUME` paths and Compose's anonymous volumes become named volumes labelled `io.benchflow.trial-volume=<project>`, and teardown, the force-kill fallback, verifier recovery and `bench sandbox cleanup` remove the trial's volumes by that label, after a failed start too, locally and on remote Docker. A volume the trial did not create (yours, an external or adopted one, a `--mount` source, BenchFlow's `benchflow-bfx-*`) is never removed, and nothing is removed while a container still uses it. If an image or a `volumes_from` source cannot be read, the start fails instead of guessing.

Every fix, with its reasons and examples, is listed [in detail](#in-detail-every-change-since-0711) below.

### Renamed

Public names that named another harness, or its reward package, now have neutral names. No old name is kept as an alias: an old CLI value is refused with the ordinary unknown-value error, which lists the values that work (`--target 'x' is an unknown value; choose one of: task-toml`), an old preset is refused as an unknown profile, and an old module or class no longer imports. The old runner environment variables are silently ignored; update all three below.

| Where | Before | Now |
|---|---|---|
| `bench export --target` (and `bench tasks export`) | the value named after the harness whose `task.toml` split layout it writes | `task-toml` (still the default and the only value) |
| `export-report.json` | `target`: that same value | `target`: `task-toml` |
| `bench hub list --provider` | the provider named after that harness's registry | `benchflow` |
| `bench hub check --registry` default | that harness's public `registry.json` | BenchFlow's registry: `benchflow-ai/skillsbench`'s `registry.json`, then the registry packaged with BenchFlow (Terminal-Bench 2.0, 89 tasks at a pinned commit, with digests) |
| `bench hub check --cache-dir` default | `.cache/hub/` and that harness's name | `.cache/hub/registry` |
| `bench hub check` output | headings naming that harness | "Registry check (<registry>)" and "check registry tasks" |
| `bench hub check` records | a `framework` field naming that harness, `hub_url`, and an `env_uid` prefixed with its name | `registry` (the source, or `benchflow`), and `env_uid` = `<dataset>/<task>@<ref>` |
| task.md preset | the compatibility preset named after that harness, with its `compatibility.<name>` block | `task-toml-compatible`, with `compatibility.task-toml` |
| Python | the round-trip conformance report class named after that harness | `TaskTomlRoundTripConformanceReport` |
| Python | the round-trip mismatch class named after that harness | `TaskTomlRoundTripMismatch` |
| Python | the round-trip report builder named after that harness | `build_task_toml_roundtrip_conformance_report` |
| Python module | `benchflow.hub.` + that harness's name + `_registry`, with its task ref, load, select, check, uid and materialize functions and its registry and hub URL constants | `benchflow.hub.registry`: `RegistryTaskRef`, `load_hub_registry`, `select_registry_tasks`, `check_registry_tasks`, `registry_env_uid`, `materialize_registry_task`, `BENCHFLOW_REGISTRY`, `DEFAULT_CACHE_DIR` |
| Verifier strategy | the previous strategy name, in each of its spellings (one word, with a hyphen, with an underscore) | `criteria-runner`; every previous spelling gets the unknown-value error |
| task-package fields | the reward-package field and `source` value | `criteria_runner` (hybrid field and rubric source); `criteria-runner` (strategy type) |
| task-package fields | a `source` naming that harness; a step runner named after it | `source: legacy` (what `bench import` writes); `runner: custom-step-runner` |
| Python module | `benchflow.adapters.` + the previous harness name | `benchflow.adapters.task_toml`; no alias |
| Adapter class | the previous harness name in title case + `Adapter` | `TaskTomlAdapter`; no alias |
| Adapter function | `from_` + the previous harness name + `_dir` | `from_task_toml_dir`; no alias |
| Adapter source | the previous harness name in `InboundTask.source`, `compatibility.source` and materialised `benchflow.compat.source` | `task-toml`; no alias |
| Runner environment | `BENCHFLOW_` + the previous rewards-package name in uppercase with an underscore + `_ROOT` | `BENCHFLOW_CRITERIA_RUNNER_ROOT`; no alias, the old variable is silently ignored |
| Runner environment | the same old prefix + `_CRITERIA` | `BENCHFLOW_CRITERIA_RUNNER_CRITERIA`; no alias, the old variable is silently ignored |
| Runner environment | the same old prefix + `_MANIFEST` | `BENCHFLOW_CRITERIA_RUNNER_MANIFEST`; no alias, the old variable is silently ignored |
| Runner manifest | the previous hyphenated strategy name + `-manifest.json`; its version and root key | `criteria-runner-manifest.json`, version `benchflow.criteria-runner.v1`, `paths.criteria_runner_root`; no alias |
| Authoring profile | the profile named after the previous rewards package | `criteria-runner`, entrypoint `verifier/criteria_runner/reward.py`; no alias |
| Example task | `docs/examples/task-md/` + that harness's name + `-parity` | `docs/examples/task-md/task-toml-parity` |
| Dataset id in the docs | Terminal-Bench 2 under that harness's GitHub organisation | `resolve_source("terminal-bench-2")`, from BenchFlow's registry |

The repository's comparison tests and integration suites against that harness are removed; they were not part of the package.

These changes came with the renames:

- **Datasets come from BenchFlow's registry only.** `-d name@version` reads BenchFlow's published registry and the entries packaged with BenchFlow, and nothing else. `terminal-bench@2.0` is packaged: 89 tasks at a pinned commit, each with a content digest that is checked. A dataset BenchFlow's registry does not have stops with "Dataset '...' not found in BenchFlow's registry" and names the next step (`bench hub list --provider benchflow`, or `--registry` for a registry that has it).
- **Only job folders BenchFlow made are resumed.** A job folder is resumed only when it carries a BenchFlow marker: `evaluation.json`, the `summary.json` a 0.7 job writes, a trial `config.json` with a 0.8 or 0.7.x trial's keys, or a trial result that is a BenchFlow rollout result (`rollout_name`). The job lock is not a marker. A 0.7 job stopped before any trial finished now resumes. A named run refuses any other folder with files (`<folder> is a folder BenchFlow did not make; nothing in it moves`); an unnamed run starts a new job beside it.
- **`bench check` reads a verifier's model use only from what the task declares** (`[verifier] models = true`, a provider key the verifier env names, or the judge files an import recorded). A test.sh that calls a grader with none of these declared no longer gets the warning.
- **Integrity action records count unknown tool calls with no input** in `unmapped_kinds`, instead of skipping one foreign event name.
- **The Python export API takes only `task-toml`.** `export_task_to_split_layout`, `build_compatibility_export_report` and `build_task_toml_roundtrip_conformance_report` refuse any other target with the CLI's unknown-value error.

### Known limitations

- **Kept services can retain an open writer to an existing workspace file.** Changing a single-name file's owner and mode does not revoke a descriptor a service already opened for writing. The close step replaces non-root files, including shared hard-linked files, but does not copy every root-owned file. A kept service can still change such a file while tests read it. Use a separate verifier when its inputs must not change during grading.
- **Closing shared files for kept services can consume extra disk.** The root-owned copy writes sparse files in full: a sparse file with a logical size of 4 GiB can require 4 GiB of disk. Without a protected Python, the fallback file walk can also fail on paths longer than 4,096 bytes; this stops hardening rather than accepting the workspace.

- **Tasks written for a root agent can fail with the default user.** The agent runs non-root while the verifier normally runs as root. It cannot install system packages or write under `/usr`, `/etc` or `/opt`; packages installed in its user site may be invisible to root's tests. The historical Terminal-Bench 2.0 scan found root-only writes in 33 of 89 reference solutions; five tests read those writes directly (`build-pmars`, `build-pov-ray`, `mailman`, `nginx-request-logging`, `sqlite-with-gcov`). The other 28 depend on the chosen solution; this scan does not establish that every alternative solution needs root. `headless-terminal` also failed when the agent installed `pexpect` only for its own user. `--sandbox-user none` runs the agent as root and permits those writes, but root can undo the PATH lock, protected runtime and workspace freeze and change what the verifier runs. Use it only for trusted agents; keep its scores separate from non-root runs.
- **Some task images cannot pass the protected runtime's loader checks.** If the native loader searches or loads libraries from a folder another user can write, BenchFlow refuses its root steps and the trial is unscored, including the oracle. Terminal-Bench 2's `make-mips-interpreter` is one such task. Use an image whose library files and search folders are root-owned and closed to other writers.
- **Ten of Terminal-Bench 2's 89 tasks keep services that native Daytona cannot freeze for a non-root model agent.** These trials are refused: `configure-git-webserver`, `git-multibranch`, `hf-model-inference`, `install-windows-3.11`, `kv-store-grpc`, `mailman`, `nginx-request-logging`, `pypi-server`, `qemu-alpine-ssh` and `qemu-startup`. Use Docker or remote Docker. The oracle and nop do not have this model-agent refusal; their separate image and verifier limitations still apply.

- **What BenchFlow does not isolate.** BenchFlow treats test.sh, and everything it runs or installs, as the task author's code, as BenchFlow's Rust engine and other harnesses do. BenchFlow protects the verifier's environment, not its logic, and it does not isolate code that test.sh installs, builds or imports from files the agent wrote: a `pip install -e .` of the workspace, a requirements file or wheel there, a `pyproject.toml` that `uv sync` reads, or a module that test.sh's Python imports from the workspace, such as a `pytest.py` the agent left in `/app` when test.sh runs `cd /app && python -m pytest`. Such code runs as the verifier's user, usually root, and can change the result, for example by patching pytest, standing in for it, or writing the reward file. The pytest plugin guard still refuses a plugin from such an install, but that is not isolation, and a run in which no pytest reached the guard (test.sh ended before pytest, checked without pytest, imported a `pytest` that is not pytest, or ran pytest where the verifier's environment does not reach, as with `env -i`, `sudo`, ssh or another container) is scored as test.sh wrote it.
- **What task authors should do.** Install test tools into the image or pin them, run the tests with `$BENCHFLOW_PYTEST`, do not run Python from the workspace folder, and install nothing from the agent's files.
- **Where BenchFlow takes the task's word.** (1) It prepares the protected runner only when the task's tests folder or `[verifier] env` names `$BENCHFLOW_PYTEST`, on every backend and for every agent user, a separate verifier sandbox included. The next two apply only when the agent runs as a non-root user in its own sandbox; with a root agent or in a separate verifier sandbox, the earlier rules stay. (2) It keeps a Python start-up hook that only root could have written during the agent's turn (as a root-run solution's `pip install` writes). (3) It keeps on the verifier's `PATH` a program in a `PATH` folder whose route leads into the workspace, when that name was not in the folder before the turn: root (the task's solution or a service) added it, or, in a folder another user owns, that user did. The verifier then runs, under that name, whatever is at that workspace path after the turn, which can be the agent's file. What the image already had before the turn, and anything the agent's user could have written outside the workspace, is judged as before. Refusing or isolating these installs remains follow-up work after 0.9.0. See [Sandbox security](docs/sandboxes/security.md#what-you-must-trust).
- **Three checks of the protected runtime's native libraries remain incomplete.** After the agent, BenchFlow runs its root steps with a protected copy of the image's Python. Three of its checks on native-library search paths are incomplete, and each needs a task image that already points a search path into a folder the agent can write:
  - Python extension modules copied into the protected runtime keep their own library search paths (RPATH and RUNPATH).
  - The check of a tool's library search path splits paths on spaces (`/opt/shared libs` is checked as `/opt`), and on musl images it does not read RUNPATH folders.
  - The protected Python's own native dependency search, and plugin preparation and image start-up hooks, are not yet fully covered.

  0.7.11 ran the image's own Python for these steps, so none of these is new. Their closure remains follow-up work after 0.9.0.
- **On an image with no `python3` at all,** the step that removes workspace links pointing outside the workspace is skipped, and the trial log says so. It is not refused, so Go, Rust and C tasks stay scored.
- **On a sandbox other than Docker, an image whose only `python3` is one the agent can remove** (a uv-only image's alias) has its start-up hook clean-up skipped when the agent deletes it, and the trial log says so. On Docker, `bfx` does this step.
- **Reconnecting while kept services retain previous-connection processes is refused.** With `x-keep-services`, BenchFlow cannot prove that every process from the previous role or scene stopped without also killing the services. It refuses a later connection before its probes, proxy setup or credential staging. Use a fresh rollout; the first connection and its existing session remain available. Ordinary tasks still stop the agent user's processes between connections.
- **A task that keeps services cannot be graded on an image without Python when the agent runs as a non-root user.** BenchFlow cannot check the kept services' workspace there, so the trial is refused with a named message and has no score. Terminal-Bench 2's `configure-git-webserver` is such a task: its oracle run scores, and a run with an agent under the default non-root user (`nop` included) is unscored.
- **A task image tag that someone retags during a Docker start can leave one volume behind.** BenchFlow reads the image's `VOLUME` paths when the start begins. If someone retags or pulls that tag on the same daemon in between, to an image that declares another `VOLUME` path, Docker gives that path an anonymous volume, and teardown does not remove it. BenchFlow's own builds use private tags, so this needs an outside retag. Remove such a volume with `docker volume rm` once its container is gone.
- **A task that calls `$BENCHFLOW_PYTEST` only from a file inside its image,** not from its tests folder or `[verifier] env`, receives a stub that prints an error and exits 4. The trial follows test.sh's reward, usually 0. Name the runner in test.sh to prepare the protected runtime. No Terminal-Bench 2 task calls it.
- **On Docker, an agent can make the hook clean-up fail, so the trial is unscored and retried instead of scored.** Where `bfx` does the clean-up, its answer about the files it walked is one line of at most 256 MiB: a non-root agent that leaves about 90,000 files with start-up hook names makes it longer, and the step fails as a verifier infrastructure error. Under a Compose file that drops the `DAC_OVERRIDE` and `DAC_READ_SEARCH` capabilities, an agent folder set to mode 000 fails the step the same way, where the shell clean-up skipped it. The shell form, used on the other backends, passes both.
- **Every trial keeps a root-only copy of the workspace as it was before the agent's turn** (`/testbed_verify`), which the workspace checks after the turn compare against. It costs disk in proportion to the task's workspace: at most 112 MB among the 24 Terminal-Bench 2 images inspected in the historical review, but a task whose image ships a workspace of several GB can run out of space on a small sandbox, such as Daytona's 10 GB default.
- **On Modal without BenchFlow's protected runtime** (`--sandbox-user none`, on an image where the agent can write `/opt/benchflow`), a folder upload can be raced so that root changes the mode of a link's target to 700. Nothing runs, but it is a mode change made as root.
- **A folder made to look like a 0.7 job is resumed as one.** The 0.7 trial marker checks that a trial `config.json` has all four 0.7.x keys, not what they hold. No other tool writes them, so only a folder built to pass does.
- **A named run that refuses a folder BenchFlow did not make takes its job lock there first,** so the folder's modification time changes and a lock guard file stays beside it in the jobs folder. Nothing in the folder moves. A run killed at that moment can leave `.evaluation.lock` inside it.
- **The packaged Terminal-Bench 2.0 entry's repository URL redirects on GitHub to the dataset's current home.** Its tasks are fetched at the pinned commit and each is checked against its digest, so a different redirect target cannot change what runs.
- **An image or sandbox environment that sets an exported bash function, `SHELLOPTS` or `BASHOPTS` is refused** (see [Fixed](#fixed)). The refusal is reported as an infrastructure error, and on Daytona it is retried twice, so three sandboxes start before the trial fails. A task's own `[environment] env` that sets one is refused on Docker and Docker-in-Docker, and accepted on Daytona and Modal, where it never reaches root's shells. None of the 22 base images successfully inspected in the historical review set one; this is not a guarantee about other images.
- **When the agent runs as root (`sandbox_user: null`), there is no protected copy of the image's Python:** root's steps and the plugin guard's installer start the image's `python3`, isolated from `/`, as 0.7.11 did. Never run an untrusted agent as root.
- **Under `denylist` and `allowlist`, an image BenchFlow cannot copy a Python from stops before the agent** with an infrastructure error: one with no root-only `python3` in `/usr/local/bin`, `/usr/bin` or `/bin` (a conda or uv-only interpreter), or whose standard library others can write. Native Claude login on a no-web task stops the same way.
- **A Python that appears or changes during the agent's turn** (a solution that apt-installs `python3`) gets no copy of the plugin guard. If pytest reports that it cannot load the guard, the trial scores 0 with a warning naming that Python. On an image without `stat`, every Python counts as changed. Absence of a reported failure does not establish that this interpreter loaded the guard.
- **A separate verifier sandbox builds BenchFlow's protected runtime in every trial.** The historical eight-task Docker measurement added roughly 1.5 to 2 s on a loaded host; that is neither total runtime setup time nor a universal cost. A verifier image that cannot hold it (no root-owned `/bin/sh`, no `sha256sum`, or an `/opt` others can write) fails at setup.
- **Some installs still add system packages to the task image under `allowlist` and `denylist` policies:** the in-sandbox model proxy, OpenHands' tool bootstrap and the Toolathlon adapter apt-install curl, openssl and ca-certificates.
- **The verifier gets the image's `LD_LIBRARY_PATH` (and similar loader variables) empty.** An image tool that needs its own library folder can exit 127 in the verifier. Where the backend records the service image environment, the trial warns about the recorded values it dropped. A backend or service without that record has no such warning.
- **A wrapped Codex agent (`acpx/codex`, `acpx/codex-acp`) does not receive an inline subscription login** passed as `CODEX_AUTH_JSON`. Use the plain `codex-acp` agent.
- **Ignoring Python's bytecode caches in the verifier costs time.** No `.pyc` the agent planted can run, but every Python process in the verifier compiles what it imports. In the historical `fix-code-vulnerability` experiment, this added about 0.6 s per pytest process; other import workloads and machines have different costs.
- **The `gpt-5.4` family loses some tools under codex-acp 2.0.1** (see the known regression in [Fixed, in detail](#fixed-1)).

### In detail: every change since 0.7.11

#### Fixed

- **No image Python is started as root after the agent's turn.** With a sandbox user, each connection's egress proxy under `denylist` and `allowlist` started three of the image's Pythons as root, after the agent's turn too: its health check, the proxy itself and the certificate fallback. An agent's `/app/urllib.py` ran as root in the health check. All three now run on root's protected copy of the image's Python, started `-I -S -B` from `/`; a protected runtime without a Python stops the start (exit 87). The pytest plugin guard's installer no longer starts any interpreter: it installs the guard from a record, taken before the turn, of where each verifier interpreter looks (an entry must end with `end`, and an interpreter recorded as another file, or changed, counts as unrecorded). An unrecorded interpreter gets the guard through the verifier's `PYTHONPATH` and a `benchflow:` line naming it. A separate verifier sandbox takes the record before the workspace is copied in, and without a kit it never starts an interpreter under the folders the agent can write. The source scan that finds root starts of an image Python now reads interpreters held in shell variables, `$(command -v python...)`, f-strings and module constants.
- **Daytona exec transport.** What a backend hands its provider per command is at most 64 KiB, counted after every wrapper and checked at one place (AgentCore: 60,000 JSON bytes). A larger command, its environment included, goes as data and a fixed reader line runs it in the same shell, user and route: on the command's input on Docker, remote Docker, Apple container and Modal; on Daytona as a file in root's mode-700 `/opt/benchflow/run`, used only after the folder is proven root's alone and not a link, and deleted after; on Docker-in-Docker the inner command on input and an oversized VM command staged too; on AgentCore as a sealed file. A payload ends with a per-command end line, and one cut short exits 126 with nothing run. Command output that may hold NUL bytes comes back as a checked hex receipt. Directories move as one tar archive with one open file at a time (Daytona, its Docker-in-Docker VM and Modal); an image without `tar` falls back to batches of 100 files. A single environment value over 128 KiB is still refused by the kernel.
- **Without a reported guard problem, the task's reward remains authoritative.** A missing `registered` marker alone no longer refuses the trial. Terminal-Bench 2's `qemu-startup` and `qemu-alpine-ssh` therefore score the 0 that test.sh writes when setup fails before pytest starts. A signed adverse guard verdict, an incomplete check or pytest's guard-load failure message gives score 0 with a named warning, even beside a `registered` marker. Absence of that evidence does not prove loading: code test.sh imports can suppress the guard or change its reward files. The task's test script and the code it chooses to execute remain trusted. See [the threat model](docs/sandboxes/threat-model.md#pytest-plugin-guard).
- **coverage.py no longer measures the verifier's tests unasked.** The verifier ran with `COVERAGE_PROCESS_START` set to empty, meant to keep coverage.py from reading a config file, but `coverage.process_startup()` starts measuring whenever the variable is defined, even empty. So a test package, `sitecustomize` or `.pth` file that calls the hook ran its tests under the tracer, and wrote a `.coverage` file into its working folder. The verifier's command now unsets `COVERAGE_PROCESS_START` and `COVERAGE_PROCESS_CONFIG` (which coverage.py 7.10+ reads first), so the hook returns before it reads or writes anything.

- **An agent's installation no longer changes the task's environment (mini-swe, codex-acp, OpenHands).** mini-swe's install apt-installed Ubuntu `python3` into the task image when it had none, so the agent's commands found a Python the task never had; this changed command lookup and could make a correct solution fail on a Python-free task. mini-swe now uses a private pinned CPython under `/opt/benchflow` (agents #81), fetched without installing system packages, and never on the tool PATH. codex-acp and OpenHands wrote their credentials under `umask 077` and kept it, so the agent's files were created 0600 where the image's umask is 0022; the umask now applies only to those writes. A Python-free-image regression covers mini-swe, claude-agent-acp, codex-acp and OpenHands, fresh and cached.
- **A closed terminal or a dropped SSH session stops a run cleanly, exit 129.** SIGHUP killed `bench` at once and left the run's sandboxes running. It now stops the run as SIGTERM does: the running trials stop, their sandboxes (the warm pool's prepared ones, and a sharded run's workers' too) are removed, and the exit is 129. A run started under `nohup` is unaffected. While the sandboxes are being removed, a second Ctrl-C stops at once, a second SIGTERM (or a Ctrl-C after SIGTERM) is ignored and the next one stops at once, and a SIGHUP never does (a closed terminal sends it twice). A sharded run's parent now stops its worker processes and waits for their cleanup; before, a stopped parent left them running.
- **A model that never answered before the agent's wall-clock budget ran out is an integration failure in the run's own records too.** The run-time integration check judged only timeouts that carry no message, and BenchFlow's own wall-clock timeout always carries one. So such a trial was written as `error_category: timeout` with reward 0, and `summary.json` and the end-of-run score counted it as a scored 0, while `bf.load_job`, `bench eval metrics` and the viewer read the same trial as an unscored integration failure. A wall-clock timeout after no tool call, message, thought or output token is now `agent_integration` (`no_activity`) when it is written: the reward is withheld, `summary.json` counts it under `integration_failures`, and `agent_timeout_info` stays. Like the timeout, it is not retried. An idle-watchdog timeout keeps its category while the job runs, because it is retried, and is judged when read. When a result has both, the integration failure now decides `error_category`.
- **An adapter's own warnings are recorded as notices, not as the agent's words.** BenchFlow now declares ACP session notices (`clientCapabilities.session.notices`), so codex-acp 2.x and claude-agent-acp 0.81+ send what they compose themselves (a Codex warning or model reroute, a Claude model fallback or SDK warning) as `notice` updates; without it they wrote it into the agent's reply. On a model missing from the pinned Codex catalog (gpt-5.4, or gpt-5.4-nano), Codex warns as the turn starts, so a provider that never answered still left one agent message, and the trial ended `timed_out` and scored 0 instead of an unscored integration failure. A notice is now its own trajectory event (`type: "notice"` with `severity`, `title` and `description`): never the agent's answer, never activity for the idle watchdog or the integration check (whose evidence quotes it), a notice card in the viewer, and a `system` step in ATIF; training exports leave it out. A typed-failure advisory is recorded the same way instead of as `**Notice:**` agent text, and the native Codex harness records the warnings `codex exec` reports as error items as notices too (it dropped them). claude-agent-acp also stops sending its info-level SDK lines, which Claude Code itself shows only in transcript mode.
- **A spent subscription fails fast with a typed error, instead of a generic, retried `acp_error`.** On 2026-09-30 every trial on a Claude login whose weekly window was spent ended `ACP error -32603: Internal error: You've hit your weekly limit · resets Oct 2, 4pm (UTC)`, was retried twice (three sandboxes per task), and a caller could find out only by matching that text. BenchFlow now asks `claude-agent-acp` (0.81.2, checked against its source) for its typed session failures, so a spent login arrives as its own failure kind, and also reads the same text from JSON-RPC errors and codex-acp's `codexErrorInfo: "usageLimitExceeded"`. The trial is `error_category: usage_limit` with a `usage_limit_info` block (the login as a label such as `CLAUDE_CODE_OAUTH_TOKEN (environment)`, never the token; the 5-hour or 7-day window; the reset), unscored as before and never retried. An `Evaluation` then starts no more trials, lets running ones finish, lists the trials not started in `summary.json`'s `usage_limit` block, and raises `bf.UsageLimitError` (`.login`, `.window`, `.resets_at`, `.result`) so a caller can switch logins and resume; `UsageLimitError.from_result` reads the same from a `bf.run` result. Every other typed failure raises the same `ACPError` the agent raised before, so no other category or retry verdict changes; a killed Claude Code process now says it was the agent's own process. The hill-climb demo's account pool catches the error and reruns the unstarted trials on another account.
- **`bench tasks check` finds pytest plugin installs in agent-writable locations.** It resolves relative paths, `/root/.venv`, `~/.venv`, explicit `--python` paths and activated environments against the task workspace, agent home and temporary/log directories. The warning names the plugin and installation location. Pin the release in `[verifier] pytest_plugin_releases` or install it into the image; an unpinned `uvx` install does not fix the refusal. A reported runtime refusal scores 0. pytest-json-ctrf for `--ctrf` is supplied by BenchFlow's protected runtime.
- **Expected errors print one message; unexpected ones a traceback and a log.** `bench` ran the Typer app directly, so any uncaught error ended in a Rich traceback box (a task.md naming a missing environment manifest ended `bench eval branch` in 110 lines of one). The console scripts now run `benchflow.cli.main:main`: a `benchflow.errors.UserError` (task-format, flag, job-state and registry errors, a missing login, a usage limit) or a missing file prints one message and its next step; anything else prints the traceback and a crash log under `~/.cache/benchflow/logs` (`BENCHFLOW_LOG_DIR`) with the command, versions and the run's last log lines. Credential values from the environment, `.env`, `KEY=VALUE` arguments, credential-named options and the Claude and Codex login files are replaced with `***`; redaction matches names and values, so the log is still worth reading before it is attached to a public issue. A missing login is `MissingCredentialError` with a next step (`claude setup-token`, `codex login`), checked by `bench eval run`, `bf.run` and `Evaluation.run` before a job exists instead of logging a traceback in every trial.
- **A Claude model the adapter refuses fails once and names the models it offers.** `--model claude-sonnet-9` made three attempts over 203 s, each an opaque `ACP session/set_config_option failed ... -32603: Internal error`; it is now an unscored, never retried `agent integration failure [agent_model]` listing the adapter's model values, as for codex-acp.
- **The Daytona PTY's messages read as a lost connection.** Text-only classification read `PTY closed by the peer: ...`, `PTY readline timeout` and the start-marker timeout as `other`; they are `pipe_closed`, as their transport diagnostic already said.
- **Two SDK errors now say what to do.** A misspelt key in a budget mapping (`EvaluationConfig(budget={"max_usd": 5})`, or `budget:` in a YAML config) is refused with the valid keys; it used to leave the job without a budget, silently. An `Evaluation` over a tasks folder that does not exist says so and what to pass, instead of a bare `[Errno 2]`.
- **`bf.regrade` works in a notebook, and the blocking entry points match their async forms.** `bf.regrade` called `asyncio.run` and failed inside a running event loop (a Jupyter cell), where `bf.run_sync`, `bf.run_batch` and `Evaluation.run_sync` work; it now blocks the same way and takes `aregrade`'s `runner`. `bf.run_sync` declares `arun`'s parameters instead of `*args, **kwargs`, so `help()` and editors show them.
- **A trial whose agent errored keeps its artifacts.** Artifacts (`/logs/artifacts` and declared `artifacts = [...]`) were collected when the verifier phase started, which an agent error skips, so a crashed trial kept nothing the agent left, such as its session log. Cleanup now collects them before the sandbox stops when the verifier never ran (bounded to 3 minutes, and skipped when the sandbox is known to be unreachable).
- **Group-relative advantages stay inside one job and skip retried attempts.** `bench train convert --group-advantage` and `bench train stream --group-size` grouped by task, agent and model, pooling rollouts across jobs (which may have run different policy checkpoints under one model name) and across an Evaluation's retried attempts, and the stream's group size counted attempts a retry then replaced. The default key is now `task,agent,model,job` (leave `job` out to pool jobs of one policy, such as `--matrix --trials`); a replaced attempt is `excluded: "retried"` and in no baseline; the stream counts only final rollouts toward a group, emits a replaced one with `retried: true`, and names the attempt a rollout replaces in `replaces`.
- **Solve rates say when control runs were left out.** `bench eval metrics` over an oracle run printed `Score 100.0%` beside `Solve rate n/a (0 scored trials)` with no reason; the solve-rate caveats (CLI, `summary.json`, JSON export) now name the control runs (oracle, empty/nop) they leave out.
- **A plain `bench eval run` (or `Evaluation` without a `job_name`) resumes only its own job.** It resumed into the alphabetically last folder of the jobs dir, whatever it was: after `bench eval smoke`, every run landed in `jobs/smoke` (a folder of jobs) and blended its hello-world pass into another task's score, and the documented oracle-then-nop pair refused its second command. Now only a job folder is resumed (a timestamp name, a job record or trial folders; never a folder of jobs), only when it carries a BenchFlow marker (see "Only job folders carrying a BenchFlow marker are resumed" under [Changed](#changed)), and only when its `evaluation.json` records the same agent, model and tasks folder; otherwise a new job starts, with a warning naming the one it did not resume. The refusal for an explicit mismatching job names `--fresh`.
- **Budgets enforce USD and tokens between rollout starts, and count every attempt.** USD and tokens were counted only as trials finished, and only from a trial's last attempt, while the CLI called the caps hard limits: every free slot kept starting trials until the cap was passed, then the running ones were cancelled with their spend lost, and a retried trial's earlier attempts were free. Every rollout (retries included) now counts, and once rollouts have finished a trial waits to start while the running ones, at the job's mean spend so far, would reach the cap, so a job passes a USD or token cap by about one rollout's spend (the first wave, before anything has finished, starts without an estimate). A resume counts every attempt the job folder records. The CLI help and `docs/reference/budget.md` now say how each cap is enforced.
- **`results.jsonl` rows say which reward convention they use.** Since this release an unscored rollout's row has reward null, not 0.0, but nothing in the file said so. Each BenchFlow row now carries `info.schema_version` 2 (rows without it are version 1, where 0.0 could mean unscored), and `bf.load_job` reads an unscored version-1 row (reward 0.0 with no `reward` in `metrics`) as unscored instead of a failure.
- **The committed JSON Schemas accept fields added later, as the versioning rule promises.** `benchflow.trial`, `benchflow.job`, `benchflow.comparison`, `benchflow.run-summary` and `benchflow.rollout-stream.v1` set `additionalProperties: false` at every level, so a reader validating with its copy of a v1 schema refused every document that gained an optional field, although a new optional field keeps the version. The published schemas are now open; the writers stay strict.
- **Token coverage requires one logprob per sampled token id.** A call whose sampled ids and logprobs had different lengths counted as complete, so `bench train token-coverage` called the rollout training-grade while `bench train stream` could not merge it and dropped that conversation's sequence without a word. Such a call is now incomplete (`logprobs:length_mismatch` in `unavailable`), and a stream record never loses a conversation silently: the rollout is marked not training-grade with a reason.
- **`--source-env` refuses `--fail-under`, `--fail-on` and `--summary-out` instead of ignoring them.** The hosted-environment path never read them, so a CI job gated on a pass rate exited 0 and wrote no summary. vf-eval scores such a run as one mean reward, with no trials to count, so the flags now stop the run before it starts and say where the reward is.
- **Repeated rollouts in a `bf.run_batch` folder are separate trials, and `bench eval metrics` agrees with `bf.load_job`.** `load_job` kept one trial per task and folder, which is right for the retries of an Evaluation job but collapsed a batch's independent rollouts: four rollouts with rewards 1, 0, 1, 0 (newest last) gave a solve rate of 0.0, `attempts="all"` gave 0.5, and `bench eval metrics` said 100% (it kept the best of the four, across folders too). Attempts now collapse only inside an Evaluation job folder (`evaluation.json` or `summary.json`), to the scored attempt, then the newest, the rule the job's own `summary.json` uses; `bench eval metrics` counts trials by the same rule, so its `Total`/`Passed`/`Score` match its solve rate, and it lists each task name once.
- **`Evaluation(config=cfg, budget=...)` no longer changes `cfg`.** The keyword, and the task source's provenance, were written into the caller's `EvaluationConfig`, so a config reused for a second Evaluation carried the first job's budget. The Evaluation keeps its own copy.
- **The egress proxy no longer cuts model responses longer than 15 minutes.** Under a denylist or allowlist network policy, both relays gave each socket a 900 s timeout and treated the client's silence as the end of the connection, but a streaming model response is client silence by nature: every model turn longer than 15 minutes was cut with no status, and Claude Code retried it from scratch. The 900 s is now an idle timeout for the whole connection, reset by every byte in either direction. There is no absolute cap: a connection that keeps moving bytes is in use, and the proxy lives only as long as its rollout.
- **A lost tool-call update no longer kills a working agent (#1141).** The idle watchdog gave each pending tool call its own grace clock (3x the idle budget) that only that call's own updates restarted, so one call whose terminal update the ACP adapter lost ended a session that was still completing tool calls: 6.9 h of a Claude Code run on a GPU box, and two Codex runs at the 5400 s grace. A newer tool call now restarts the grace of every call already pending, since an agent that starts a new call is not waiting on the older one; a silent session still hits the grace. `BENCHFLOW_AGENT_PENDING_TOOL_GRACE_SEC` sets the grace in seconds, and the idle timeout message and `idle_timeout_info.lost_update_tool_call_ids` name the calls that got no update after later calls started ("lost update").
- **Codex no longer sends the task prompt to api.openai.com.** codex-acp (1.13.1 through 2.0.1) names each session with an ephemeral thread on the hard-wired `gpt-5.6-luna` whose input is the first user message, the task prompt. It starts that thread without the session's provider, so the request went to Codex's built-in `openai` provider at api.openai.com, which answered 401 on every rollout after the prompt had left. BenchFlow's Codex launcher now writes the run's provider into Codex's user `config.toml` as the default for every thread (the built-in `openai` provider points at the same endpoint), and the model gateway refuses a request for any model other than the run's with HTTP 400 `model_not_served`, without forwarding it, and counts refusals in a warning when the rollout ends. With the real codex-acp 1.13.1 in a container with no network, the unfixed launch tried api.openai.com 17 times after the first turn; the fixed one sent the title request only to the gateway.
- **`claude-opus-5-5` runs under `claude-agent-acp`: the adapter moves to 0.81.2 and the Claude Code CLI gets its own pin (#1137).** The 0.73.0 pin bundled Claude Code 2.1.257, and `claude-opus-5-5` refuses every turn below 2.1.280. The sandbox now installs `@anthropic-ai/claude-code@2.1.280` beside the adapter and the launcher hands it to the adapter through `CLAUDE_CODE_EXECUTABLE`, so the CLI the adapter's SDK bundles never runs; a model that needs a newer CLI is now a CLI bump rather than an adapter bump (the adapter stays on the 0.81 line: 0.82.0 changed the tool-call contract). The install fails when the CLI does not report its pin, the native Claude no-web gate verifies the CLI package, its binary and the launcher, and `bench doctor` shows and guards both pins. A `tool_progress` heartbeat that names an Agent call as its own parent no longer detaches the call and its subagent from the trajectory. Builds on #1139 (alistairjcbrown), which first moved the adapter to 0.81.0.
- **A timed-out exec is reported as a timeout, never as an empty `verifier crashed:` (#1065).** Killing a child that asyncio had already reaped raised `ProcessLookupError`, which replaced the timeout error; it carries no text, so the verifier error read `verifier crashed:` with nothing after the colon. Teardown now tolerates a child that is already gone, and verifier errors name the exception class (`verifier crashed: ConnectionResetError: ...`) (#1081, by tulerfeng).
- **A slow exec after the agent finished no longer discards the rollout (#948).** Publishing the trajectory for the verifier (`mkdir -p /logs/agent`) and scraping Gemini's trajectory ran a 10 s exec whose timeout threw away a completed rollout on a loaded host; they are now non-fatal and logged (#1043, by tulerfeng). The verifier's own setup commands (clearing its output directory, `chmod` of `test.sh`, a target service's verifier directory, the criteria runner's manifest) get 60 s instead of 10 s and two more tries, 5 s and 15 s apart, before `Verifier setup failed` (verifier infrastructure).
- **A multi-megabyte ACP message no longer kills the agent's transport or loses its tool call (#1138).** A Claude Code `Read` of a PDF returns the pages as inline base64 images, one `tool_call_update` line of several megabytes: Daytona closed the PTY websocket on it (1008) on every attempt, and Docker's reader dropped any line over 10 MB, and with it the tool call's final update. When the sandbox has `python3`, the agent's output now passes through a filter that rewrites any line over 1 MiB: image and audio blocks and other long base64 strings become a note of what was removed, then the longest remaining strings are cut until the line fits. The message's ids, `toolCallId` and `status` are kept, so the update still closes its tool call. The host reads a Docker line over 10 MB whole instead of dropping it (up to 256 MB) and applies the same rewrite to any line still over the limit, so both backends record the same message. `BENCHFLOW_ACP_LINE_LIMIT` sets the limit in bytes (`0` turns the filter off).
- **A verifier command the exec layer lost is found in 30 s, retried once, then reported as infrastructure (#1136).** After a long agent phase a Daytona exec session can lose the verifier's command. Only tasks with a verifier-only recovery contract asked the command for a start receipt; every other task waited out its whole verifier budget, then, finding no output, a second whole budget, and ended as `verifier timed out`, whose retry replays the solver. Every script verifier's command now writes a start receipt under `/run/benchflow`: when it is missing 30 s after the command was issued, the command runs once more in place, and a second miss is `verifier_wedge:` (verifier infrastructure; a task with a recovery contract goes to fresh-sandbox recovery at the first miss, as before). A command that did start is never replayed, however quiet. Each status and log request of the Daytona command poll now has its own timeout (30 s and 120 s) and is retried, so one request on a dead connection no longer holds a finished command until the verifier budget ends.
- **Reviewers get their own idle budget, and the Daytona read guard covers the pending-tool grace (#1143).** `--reviewer-idle-timeout` (`ReviewerConfig.idle_timeout_sec`, default 600, `0` or `none` disables) sets the reviewer's idle watchdog, which reviewers inherited from the rollout default with no way to change it. The PTY read guard is now at least 3 × idle + 60 s for solvers and reviewers: the idle watchdog lets a pending tool call stay silent for three idle budgets, and the transport used to cut it at idle + 60 s (a reviewer at 900 s).
- **A closed Daytona websocket fails the agent's read at once, and a reviewer that loses it is retried (#1144).** BenchFlow's Daytona PTY reader waited only for the next line, so a websocket the Daytona side closed (1006 after a reset, 1008 on an oversized message) or a network path that died without a close frame surfaced as `PTY readline timeout` once the whole silence budget had passed, and the reviewer's transport retry skips a readline timeout by design. The read now fails as soon as the SDK's PTY handle reports its websocket gone, with the close code (`PTY closed by the peer: websocket closed (close_code=1006, ...)`, transport diagnosis `pty_closed`); lines sent before the close are still read. The SDK's websockets are opened with a heartbeat (a ping every 120 s; `BENCHFLOW_DAYTONA_WS_HEARTBEAT_SEC`, `0` disables), so a silently dead path closes too. A reviewer that loses its transport is retried once with a fresh sandbox, and when it loses it again the scoring block carries `error_category: reviewer_transport`.
- **Concurrent jobs on one Docker daemon no longer delete each other's containers.** Every `bench eval run` / `Evaluation` on Docker ran `docker container prune` and `docker network prune` filtered only by `benchflow.owned=true` when it started, before each retry and when it ended. That removed every stopped BenchFlow container and unused BenchFlow network on the daemon, including another rollout's container between Compose's create and start, its network before the container joined, and the `main` container a branch restore was replacing: "container is marked for removal and cannot be started", "failed to set up container networking: network <project>_default not found" (retried since as a daemon race), "removal of container ... is already in progress". The prune is now a sweep that removes a stopped container or unused network only once the BenchFlow process that made it (new label `benchflow.process`) is gone, or once its sandbox in this process has been torn down. New containers and networks are labelled `benchflow.owned=process` instead of `true`, so an older BenchFlow's prune on the same daemon cannot match them; the sweep leaves older versions' containers alone. Each Docker sandbox also gets its own Compose project (the rollout name plus a random suffix): branch children (`n1`, `n2`, ...) and regrade's verifier had fixed names, so two of them on one daemon shared a project and tore down each other's containers. Branch restore removes the old container with `docker rm -f` straight away, which also drops a 10 s `docker stop` wait from every in-place restore.
- **A symlink that leaves the workspace no longer costs a trial its review or its verifier.** codex-acp can leave a helper link such as `/app/apply_patch` pointing outside the workspace, and every Python virtualenv links `bin/python` to the system interpreter. Workspace capture aborted on such a link, so a rubric trial ended in a scoring error and a separate-verifier trial was unscored, both with `rewards: null`. Such a link, a link whose target text climbs out of the workspace (which used to pass capture and then fail extraction), and a link loop are now `symlink_escape` exclusions that record their `link_target`, in the Python capture script and in the tar fallback for images without Python alike; no outside file is copied. (#1130)
- **A regrade restores the file permissions the solver left.** The frozen workspace's manifest now records each file's and directory's mode, and `bench eval regrade` and verifier recovery put them back (the reviewer's copy stays 0644/0755). Before, a solver that made a key private with `chmod 600` passed, and a regrade with the unchanged verifier failed it.
- **A regrade says when a verdict changed although the task did not.** Each row carries `task_changed`; a changed verdict on an unchanged task has a reason (the verifier read state a regrade does not restore, such as a package installed system-wide or a running service) and is marked in the table.
- **`bench eval regrade <job>` finds a batch job's tasks without `--tasks-dir`** through the tasks folder the job recorded, and no longer lists attempts a retry replaced as extra trials.
- **`bench eval list jobs/` lists every job.** The copy of the last summary each job writes to the jobs folder made a folder of several jobs show as one row.
- **`bench eval metrics` keeps one result per task, agent and model.** Over a folder holding an oracle job and a nop control job, every control failure was replaced by the oracle's pass.
- **A rubric task with no reviewer model names `--reviewer-model`.** The refusal said "pass --model", which sets the solver; the rubric review docs no longer claim `opencode` has a registered model.
- **Resuming a job whose task changed logs one warning** instead of a traceback before keeping the saved result.
- **Codex knows `gpt-6.1-sol`: the `codex-acp` pin moves from 1.13.1 to 2.0.1.** Codex takes a model's tool surface from the model catalog bundled with its `@openai/codex` dependency. codex 0.156.1 (behind 1.13.1) has no `gpt-6.1-sol`, so a `gpt-6.1-sol` rollout logged "Model metadata for `gpt-6.1-sol` not found. Defaulting to fallback metadata" and was offered only the fallback function tools: no code mode, no `apply_patch`, no multi-agent tools (the #1145 symptom again), and `session/set_model` to a `gpt-6.1-sol[...]` id failed. codex-acp 2.0.1 depends on `@openai/codex` ^0.159.1 (npm resolved 0.159.2 during the original pin check; both catalogs list `gpt-6.1-sol` with efforts low to max and ultra, code mode only, multi-agent v2, a 272k context). On a live docker rollout through the LiteLLM proxy (Azure), `gpt-6.1-sol` at effort `max` now gets code mode (`exec`, `wait`), `clock.sleep` and the collaboration tools (`spawn_agent`, `wait_agent`, `send_message`, `followup_task`, `interrupt_agent`, `list_agents`), and `session/set_model` accepts `gpt-6.1-sol[max]` for direct providers.
- **Codex command output stays in the trajectory under codex-acp 2.x.** codex-acp 2.0 (its tool call contract) stopped putting a finished command's output in `rawOutput.formatted_output` for a client that declares no terminal capability, which is BenchFlow: a shell command now ends with no `rawOutput` and a read / search / list command with only `{"exit_code": n}`, while the output arrives as `_meta.terminal_output_delta` chunks and the exit code in `_meta.terminal_exit`. `ToolCallRecord.raw_output` rebuilds the 1.x `{"formatted_output", "exit_code"}` shape from those chunks (an agent-sent `formatted_output` still wins), so captured trajectories keep every command's output. The trajectory viewer now shows it for shell commands too: it rendered Codex's `terminal` content block (a terminal id) instead of the command's raw output, on codex-acp 1.x as well.
- **The native harness runs the Codex core codex-acp 2.0.1 runs.** `_CODEX_CLI_PACKAGE` moves from `@openai/codex@0.156.1` to 0.159.3, the release 2.0.1's `^0.159.1` dependency resolves to, so `--harness native` and the ACP harness still drive one Codex build. The native-harness samples were re-recorded on 0.159.3 against the deterministic fake model; only the SIGINT sample changed: Codex 0.159.3 does not stop on SIGINT but finishes the turn, which the native client never applies (it drops everything after a cancel and kills the CLI after its 2 s grace). The deterministic tier's Codex scenarios move from `gpt-5.4`, which codex 0.158 dropped from its catalog, to `gpt-5.5`, which both releases list without code mode, so they keep exercising Codex's full tool surface instead of its fallback tools.
- **Known regression: the `gpt-5.4` family falls back under codex-acp 2.0.1.** The catalog comparison for this pin on 2026-10-02 covered Codex 0.156.1 and 0.159.3: the former lists `gpt-5.4` without `gpt-6.1-sol`, while the latter has the opposite coverage. This is a statement about those inspected versions. On 2.0.1, `gpt-5.4`, `gpt-5.4-mini` and `gpt-5.4-nano` log "Model metadata ... not found" and lose `apply_patch` and `tool_search`. This affects the codex-acp integration model (`.github/integration/scope_defaults.yml`, `gpt-5.4-nano`) and the `openai/gpt-5.4-mini` examples; a model the catalog knows (for example `gpt-5.5` or `gpt-5.6-luna`) keeps the full surface.
- **Codex gets its real model id, and a codex that knows it (#1145).** Under
  a BenchFlow provider (LiteLLM proxy) the `codex-acp` thread was started with
  the proxy alias as the model (`benchflow-azure-foundry-openai-gpt-5.6-luna`).
  Codex resolves model metadata by slug, warned "Model metadata for `...` not
  found. Defaulting to fallback metadata", and offered a reduced tool surface
  on every rollout: eight function tools, no native `apply_patch`, no code
  mode, no multi-agent tools, where native `codex exec` on the same provider
  offers all of them. `CODEX_CONFIG.model` now names the bare slug (which the
  proxy already serves next to the alias), the launch-config writer accepts
  it when applying the reasoning effort, and the `codex-acp` pin moves to
  1.13.1 (codex 0.156.1) because 0.148 has no metadata for `gpt-6-astra`
  even under the bare slug. Verified on hello-world rollouts: bare slug +
  0.156.1 gives `gpt-6-astra` code mode + `apply_patch` + `spawn_agent` /
  `wait_agent` / `send_message` / `followup_task`, and `gpt-5.6-luna` code
  mode + `apply_patch`, matching the native CLI's tool lists for both.
  This pin is no longer current: 0.9.0 ships `codex-acp` 2.0.1 with Codex
  0.159.3 (the two entries above).

#### Added

- **ACP and native agents run on Modal.** `ModalSandbox.live_process()` returns a `ModalProcess`: the agent's stdin and stdout are Modal `Sandbox.exec` streams. The agent starts as on Docker: root's bash by absolute path with no startup file, the loader and start-up variables empty until the drop to the sandbox user, and the agent's environment as the first line of its stdin, so no value is on a command line, in a file or in a Modal Secret; lines over the ACP limit are shortened in the sandbox. A closed stdout raises `TransportClosedError` with the exit code and the redacted stderr tail. Plan validation no longer refuses ACP and native agents on Modal; `denylist` and `allowlist` tasks are still refused there.
- **Separate verifier sandboxes run on Modal.** `verifier.sandbox_mode: separate`, a declared `verifier.sandbox` (its own GPUs included) and `--verifier-sandbox separate` / `--integrity strict` were refused on Modal before the run. The verifier's sandbox starts once the agent's has stopped and been confirmed gone (as on Daytona), so a GPU trial never holds both sandboxes' GPUs at once. A file over Modal's 5 GiB per-file transfer limit cannot cross: the capture fails with that reason and the trial is unscored, never with part of the file.
- **Cloud metadata on Modal: proven unreachable instead of blocked.** A Modal sandbox has no route to any cloud metadata endpoint (measured from a public-network sandbox: every IPv4 endpoint timed out on ports 80 and 443, every IPv6 one had no route, as root and as a non-root user). The block cannot go in there (root has no `NET_ADMIN`), so every non-root agent on an open network was stopped before it started unless the run passed `--allow-cloud-metadata`. Each agent trial on Modal now connects to every endpoint as the agent's user and records `not_reachable` in `cloud-metadata.json` when none answers; any answer, or a probe that cannot run, records `unproven` and stops the trial. Docker and Daytona keep the block.
- **Long Modal sandbox names.** A name over Modal's 64 characters, or with characters Modal refuses, is shortened and ends in a hash of the whole name; a long task name plus `-verifier` was refused at create.
- **Modal trials start on Modal's current service.** Modal's servers now refuse the legacy Sandbox filesystem API (`UnimplementedError: The legacy Sandbox filesystem API is no longer supported`), so every Modal trial failed right after its sandbox started, on the log directories. The backend makes them, uploads and downloads with `Sandbox.filesystem` (the legacy API's `FileNotFoundError`, `IsADirectoryError`, `NotADirectoryError` and `PermissionError` are kept), tests paths with `test -d`/`test -f` through its exec, and gives `Sandbox.exec` a whole-second timeout (a task's `600.0` was refused). The `sandbox-modal` extra now needs Modal SDK 1.6 (`modal>=1.6.2,<1.7`; the lock and `constraints.txt` move from 1.4.2 to 1.6.2, with synchronicity 0.12.6): the `<1.6` cap only kept the legacy file API, which the service no longer serves.
- **`bfx`, BenchFlow's own helper for root's steps on Docker, and a host-side record of what entered each sandbox.** `bfx` is a static Rust binary built reproducibly from `bfx/` (two builds in separate folders are byte-identical); the wheel carries it for Linux x86_64 (sha256 `4e2b672d116399460bef89537ebbf3746a52d35640b036ed20ce6723b0a2d534`) and aarch64 (`dbab1ae574124d6ab8eec539a2753572bc4300ccb180e6077b4e7d846b1b6bc9`), and a release fails if either is missing or differs. On Docker it is mounted read-only at `/.bfx/bfx`: by a bind of a copy under `~/.cache/benchflow/bfx/` when BenchFlow runs on Linux next to its daemon, otherwise through a volume `benchflow-bfx-<hash>` that stays on the daemon for later trials (`BENCHFLOW_BFX_DELIVERY=auto|bind|volume|off`). It runs the kill before grading (with a receipt), the workspace capture (`bfx tar-out`; a workspace path with a link or mount falls back to the earlier capture), the workspace's change of owner (uses the checked fallback walk where the helper cannot serve it), clearing `/logs/verifier`, the start-up hook clean-up and root's command channel (`bfx serve`), so none of these needs a Python in the image. Without a pinned binary for the machine, or with `off`, the earlier steps run unchanged. A remote daemon whose volume cannot be filled stops the start with a message naming `BENCHFLOW_BFX_DELIVERY=off`. A task that mounts anything at or below `/.bfx` is refused. Delivery code for Modal, Daytona and AgentCore exists but is off (`BENCHFLOW_BFX_PROVIDERS=on` delivers the binary, and no step uses it there yet). The oracle and nop controls retain their scores. BenchFlow also records, on the host, every transfer into a sandbox and who chose its bytes (see the `admission=` keyword under [Changed](#changed)): the ledger is private to your user (`~/.local/state/benchflow/trust`, or `BENCHFLOW_PRIVATE_STATE`), and each trial gets a read-only report copy, `trust-state.json`. A snapshot restored without that record counts as already touched by the agent, and `--integrity strict` refuses to restore it.
- **`--integrity audit|strict`: a reward-integrity verdict next to every reward (BenchShield, arXiv 2609.11028).** `bench eval run --integrity audit` (`integrity=` on `EvaluationConfig`, `RolloutConfig` and `TaskRuntimeConfig`; `integrity:` in a run config) records, for every trial, what the agent did to the things its reward depends on, checks that record against the task's contract, and writes `integrity/` beside the reward: the verdict (`Checked`, `VectorExposed`, `AgentViolation` or `Rejected`), the contract it was checked against, the agent's actions projected from the trajectory the host recorded, hash-chained lifecycle events, the trace check and the reward's provenance. The verdict never changes the reward. `strict` also runs every trial's verifier in the separate verifier sandbox (Docker, remote Docker, Daytona) and refuses, per trial, a task that sandbox cannot run. `bf.load_job(...).integrity()` and `bf.load_trial(...).integrity` read verdicts back, and `benchflow.integrity.audit_trial` audits a stored trial that ran without the option. The lifecycle model, event schema, trace checker and claim rules are ported from BenchShield's implementation, BenchGuard. The audit is off by default, never fails a trial (an audit that cannot record its own failure is logged and dropped) and runs off the event loop; `--integrity` on the command line overrides a `--config` file, and a bad value is refused before a job folder exists. CI's Docker tier runs `tests/test_integrity_deterministic.py`. See [Reward integrity](docs/integrity.md).
- **`bench eval regrade` and `bf.regrade`: re-score stored runs with a changed verifier, without running the agent again.** A run made with `--freeze-workspace` (rubric review and verifier recovery freeze it anyway) keeps each trial's final workspace and declared artifacts in `evidence/`, with every file's size and sha256 in a manifest. `bench eval regrade JOB_OR_TRIAL --tasks-dir FIXED_TASKS --reason "..."` (`bf.regrade`, `bf.aregrade`) builds a fresh sandbox from the current task, restores the frozen workspace after checking it byte for byte against the manifest, runs the verifier with the usual hardening, and writes the new score beside the original; `result.json` is never changed. A trial without evidence, whose evidence no longer matches its manifest, whose task folder is gone, or whose task is embodied is reported as not regradable, with the reason, and is never scored against a reconstructed or empty workspace. See [Regrade](docs/regrade.md).
- **`--harness native`: Claude Code and Codex through their own CLIs.** `bench eval run` and `bench eval branch` take `--harness native` (`harness="native"` on `EvaluationConfig`, `RolloutConfig` and `bf.branch`; `harness: native` in a run config) to run `claude-agent-acp` as `claude -p --output-format stream-json` and `codex-acp` as `codex exec --json` instead of through their ACP adapters, with the same agent entry, install, credentials, skills, MCP servers, model proxy, network policies, sandbox user, timeouts and trajectory; ACP stays the default. Each turn runs the CLI once, in its own process group, and resumes the CLI's session after the first; a cancel or timeout interrupts the CLI and kills every process of the turn. The CLIs' JSON events become the ACP session updates the adapters send (Claude Code's mapping is a port of `claude-agent-acp` 0.81.2's), so results, trajectories, ATIF exports and usage come out the same: on the deterministic tier native Claude Code reproduces the ACP golden files, and both harnesses send the model proxy the same requests apart from a written list of expected differences. Native Codex is pinned to `@openai/codex@0.159.3`, the Codex core `codex-acp` 2.0.1 runs (it was 0.156.1 with `codex-acp` 1.13.1 when this harness landed), reaches models only through BenchFlow's proxy, and under an allowlist network policy opens no other host. Ask-user handlers and `confirmation_policy: human` need the ACP harness and are refused on the native one. A Claude Code subscription login with no usage left is reported as what it is rather than as a hard task: the failed turn raises the same `bf.UsageLimitError` as the ACP harness, with the window and the reset read from the CLI's machine-readable `rate_limit_event` record, so the job stops starting trials. `config.json` records `harness_mode`, which `bench eval compare --vary harness_mode` compares. See `docs/native-harness.md`.
- **The end-of-run summary says why trials have no score, whose problem it is and what to do next.** After the score, `bench eval run` and `bench eval resume` print `Outcomes: N scored (passed, failed), N unscored, N errored`, then each unscored or errored group by cause with its fault (task, agent, infrastructure, setup), its tasks and a next step (`bench tasks check <task>` for a verifier plugin-trust refusal; switch logins or wait, then `bench eval resume <job>`, for a usage limit), the trials a usage limit left unstarted, and the run's cost and time. `benchflow.failures.cause_of` holds the table; [When a run fails](docs/when-a-run-fails.md) maps every message to its cause and fix.
- **`bench doctor` checks what a first run needs.** Docker's daemon version and free disk in its data root (warns below 10 GiB); the Claude subscription login's 5-hour and 7-day windows, used and reset, from the `anthropic-ratelimit-unified-*` headers of one 8-token Haiku request with its OAuth token (a spent or refused login warns; `--offline` skips it); the LiteLLM model proxy API-key runs go through and a custom `BENCHFLOW_PROVIDER_BASE_URL`; Linux fixes on Linux and a dead `DOCKER_HOST` named. A machine with no credential now warns instead of failing, since the oracle and nop controls need none, and the line sits with the other credential lines. `bench eval smoke --agent oracle` points at `bench eval run`.
- **A Python quickstart that runs in under a minute with no key.** The Python API's Quick Start and `docs/examples/python-sdk/quickstart.py` write a tiny task on a prebuilt image (`python:3.12-slim`) and run the oracle on it: no model credential and no image build. The oracle rollout takes about 20 seconds on Docker and the whole tour (a batch and the reading and comparing cells) about 45.
- **`print(job)` summarises a job, and solve rates carry a 95% interval.** `Job.to_markdown()` (what `str(job)` and a notebook show) gives the trials, tasks and agents, the solve rate with its interval, unscored trials grouped by reason, control runs left out, and the cost of every rollout (retries included, session-log estimates marked); `repr(job)` stays one line. `SolveRates.interval` is the Wilson interval over every trial, an unscored one counting as not solved (`interval_scored_only` is the same over scored trials only), adjusted for repeated trials of a task by the design effect, so it needs no resampling; the job and comparison exports carry it (minor 1.3). An `EvaluationResult` prints on one line instead of every task's result.
- **USD for subscription runs, from Claude Code's own session log.** A Claude subscription (`CLAUDE_CODE_OAUTH_TOKEN`) bypasses BenchFlow's model gateway, so its trials had tokens but `cost_usd` null, and a budget's `max_cost_usd` could not count them. When the gateway has no price, cleanup now copies the sandbox user's `~/.claude/projects` into the trial's `agent/claude-sessions/` (credentials redacted) and prices it: Claude Code's own `totalCostUSD` when its last `cost-state` line covers every logged response, else the logged usage at LiteLLM's list prices. The figure is labelled an estimate: `price_source` is `agent_session_log` and `usage_details.cost_estimate` gives the method (`claude-code-cost` or `usage-at-list-price`) and the log's path; `bench eval inspect` shows it as `~$…`, and the trial export carries both (minor 1.3). A rollout whose sandbox holds another rollout's turns is not priced this way -- a fork's children share the parent's log, and an isolated child or a continued checkpoint starts from a snapshot that already carries the source's session file -- and a model missing from LiteLLM's table is priced only from Claude Code's own totals.
- **Hooks and extra artifacts for an `Evaluation`.** `EvaluationConfig(pre_agent_hooks=[...])` runs Python hooks in every trial's sandbox before its agent starts, as `RolloutConfig.pre_agent_hooks` does for one rollout (config files refuse them, `evaluation.json` records their names, `Evaluation.resume(..., pre_agent_hooks=...)` takes them again, and worker processes refuse them). `config_override` (`--config-override`) may add `artifacts`, collected besides each task's own and never replacing them, so a job can keep, say, the agent's session log from every trial without editing the tasks.
- **`Trial.attempts`: the rollouts a retried trial took.** `bf.load_job` keeps each retried task's best attempt; its `attempts` now lists every attempt of that task in the Evaluation job, oldest first (itself included), so what a job ran or spent can be counted without walking folders. Records and the `benchflow.trial`/`benchflow.job` export (minor 1.3) carry the count as `attempts`.
- **`bf.Budget(max_rollouts=N)` and `bench eval run --max-rollouts N`: a cap on rollouts started, retries included.** At the cap no rollout starts (a trial whose retry is refused keeps its last attempt's result) and running ones finish; `summary.json`'s `budget.spent.rollouts` counts them. A subscription run has no USD to cap, so this and `--max-sandbox-seconds` are the caps that always bind.
- **`--sandbox remote-docker`: tasks on a Docker host you control.** Set `BENCHFLOW_REMOTE_DOCKER_HOST` (or `DOCKER_HOST`) to `ssh://user@host` or to `tcp://host:2376` with `DOCKER_TLS_VERIFY=1` and client certificates; plain TCP and local sockets are refused. It is the local Docker provider with the daemon elsewhere: same compose files, sandbox user and verifier hardening, `no-network`, `allowlist` and `denylist` modes, compose side services, separate verifier sandboxes and snapshots. Nothing on your machine is bind-mounted on the remote host (logs, verifier output and artifacts are copied back), the model proxy runs in the sandbox, every docker call targets the configured host regardless of your docker context, and the ssh user is redacted from messages. `bench eval run`, `bf.Evaluation` and reviewer sandboxes check the host with `docker info` before a job exists: an unreachable host or a task asking for more CPUs or memory than the host reports is refused with the reason, and never retried. Teardown removes the project's volumes and orphans and then removes anything still labelled with the rollout's compose project (every volume the trial causes carries that label; see [Known limitations](#known-limitations) for the one exception); when the host is gone, the warning names the command that cleans up later. See `docs/remote-docker.md`.
- **`bench train stream <job> --format jsonl` and `bf.stream_rollouts` / `bf.astream_rollouts`: rollouts to a trainer while the job runs.** Each finished rollout is emitted once, as its `result.json` appears, with reward (null when unscored), outcome, group id, the gateway's per-call prompt token ids, sampled token ids and logprobs, and, when the rollout is token-in/token-out, one merged sequence per conversation (prompt ids, completion ids, completion mask, completion logprobs). `--group-size N` emits groups with a GRPO advantage. The stream waits for a job folder that does not exist yet and ends with the job (exit 0), its dead process (exit 1) or `--timeout` (exit 3). Record schema `benchflow.rollout-stream.v1` (`docs/reference/rollout-stream.md`, JSON Schema committed); `Evaluation.job_dir` is known before the run.
- **`bench train token-coverage` names the capture path and checks token-in/token-out per conversation.** Each rollout shows the route of its captured calls (`vllm`, `sglang`, …) and its conversations (`threads`). Fix: Claude Code's tool-less helper calls (short separate prompts) counted as prefix breaks of the agent loop, so a real agent run on a self-hosted policy server was never training-grade; calls are now grouped by tool set and each tool-less call is its own conversation.
- **`sglang/...` model route with token capture.** A SGLang OpenAI-compatible server is a registered provider like `vllm` (`--model sglang/<name> --agent-env BENCHFLOW_PROVIDER_BASE_URL=...`). With `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1` the gateway asks it for prompt and sampled token ids through SGLang's `sglext` extension, which works on streamed calls too, and records them in `benchflow.token_capture.v1`. Fix: a streamed Anthropic Messages call (Claude Code) to a server that sends SGLang's final `sglext` chunk no longer breaks LiteLLM's stream adapter and loses the call record.
- **`bench train token-coverage <job or rollout> [--json]`** reports whether each rollout's gateway token capture is training-grade: prompt token ids, sampled token ids and logprobs on every call, and each prompt extending the previous call's prompt and sampled tokens (token-in/token-out). Rollouts that bypassed the gateway, such as subscription-auth runs, are named as `no_gateway_capture` instead of looking empty. Python: `benchflow.trajectories.token_capture.summarize_rollout_token_capture` / `summarize_token_capture`.
- **Separate verifier sandboxes.** task.toml tasks with `[verifier] environment_mode = "separate"` or `[verifier.environment]` (BenchFlow: `sandbox_mode`, `[verifier.sandbox]`) now run on Docker and Daytona instead of being refused. The verifier runs in a fresh sandbox built from the verifier's `docker_image`, the task's `docker_image`, `tests/Dockerfile` or (BenchFlow only) `environment/Dockerfile`, and receives only the frozen workspace, the declared artifacts and `/logs/artifacts`, each byte checked against its capture manifest. Anything else the agent planted never reaches it. The workspace replaces the image's copy of a dedicated directory such as `/app` or `/testbed` (so files the agent deleted stay deleted), with the permission bits the solver left. A failed transfer or verifier-image build leaves the trial unscored (`verifier_error` "separate verifier ...", `verifier_infra`), never 0, unless the solution's own files caused it: a workspace or `/logs/artifacts` over the capture limits, a declared artifact left as a symlink, or a `/logs/artifacts` file taking a declared artifact's place, where the same paths, measured before the agent ran, were fine. That scores 0 with no verifier error, and `verifier-sandbox.json` says why (`status` `refused`). `timing.json` gains `verifier_sandbox_setup`, `verifier_transfer`, `verifier_sandbox_teardown` and `verifier_sandbox_total`, `verifier-sandbox/verifier-sandbox.json` records the image source, sandbox id, transfer inventory and sandbox-seconds, and `--max-sandbox-seconds` counts both sandboxes. Workspace and artifact capture also work in images without Python (tar in the sandbox, rules applied on the host). See `docs/separate-verifier.md`.
- **`--source-env` runs verifiers v1 environments.** verifiers' v1 `vf-eval` (0.3.2 dev line) rejected every flag BenchFlow passed. BenchFlow now reads which CLI the installed verifiers ships and speaks it: v1 runs get the v1 flags and `--no-push` (so nothing is uploaded to Prime), and their reward, errored episodes and tokens come from `traces.jsonl`. New `--source-env-verifiers-version` pins verifiers; `--source-env-base-url` / `--source-env-api-key-var` choose the model endpoint.
- **Canary strings.** `bench tasks init` writes a canary line with a fresh GUID per task into the `task.md` frontmatter, `verifier/test.sh`, `oracle/solve.sh` and `environment/Dockerfile` (files the agent does not see), and `bench tasks check` warns, without failing, when a task has no `canary GUID <uuid>` line (`benchflow.task.canary`).
- **Hard per-job budget.** `bench eval run --max-cost-usd/--max-sandbox-seconds/--max-tokens` and `Evaluation(budget=bf.Budget(...))` (also `EvaluationConfig.budget`, YAML `budget:`) stop a job at a cap: no new trials start and running ones are cancelled (their sandboxes cleaned up, no `result.json`). `summary.json` and `EvaluationResult.budget` report the caps, what was spent, the reason, and the cancelled and not-started trials, which are never counted as failures; a resume runs them and counts what was already spent. USD counts only trials that report it; sandbox-seconds include running trials. Refused with `--worker-concurrency` and `--source-env`. See `docs/reference/budget.md`.
- **Lenient-but-safe task.toml on the run path.** A legacy `task.toml` with a genuinely unknown key outside the decision-bearing tables now runs: the key is ignored with a warning (`task.config.ignored_keys`) and `bench tasks check` prints it as a warning without failing. But leniency no longer fails open: a probable typo of a known key (a close edit distance to a sibling, e.g. `allow_internett`, `timout_sec`, a mistyped `[verifer]` table) and any unknown key inside a table that decides isolation, network, timeouts, resources or grading (`[sandbox]`, `[agent]`, `[verifier]`, `[steps]`) are refused with a `TaskConfigKeyError` and a did-you-mean, so a typo can never silently fall back to a default (a public network, the default timeout) — `main` refused the same file. A real newer task.toml key BenchFlow chooses to ignore can be added to `benchflow.task.imports.KNOWN_IGNORABLE_KEYS`, and `BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1` restores the old drop-everything leniency for corpora you do not control. `bench tasks check` reports the same refusals as errors, so the check and the run agree. Keys whose semantics BenchFlow cannot honour stay refused before launch by the runtime capability check; `[[verifier.collect]]` joins that list. `TaskConfig.model_validate_toml` and `task.md` stay strict. The task.toml docs now say what is read, ignored and refused, and that the run path reads task files directly.
- **pass@k, pass^k and partial-credit solve rates.** `summary.json`, `matrix-summary.json` (per model entry, pooling its `trial-NN` folders), `bench eval metrics`, `bench eval compare`, `Job.solve_rates()` and `bf.compare` report solve rate and the unbiased pass@k/pass^k estimators. Default statistics count every recorded trial, treating an unscored trial as not solved. Only the explicitly named `*_scored_only` variants exclude unscored trials. Control runs are excluded; each estimator omits a task when its selected population has fewer than k trials and reports the n ≥ k caveat. `--solve-threshold`/`solve_threshold=` counts reward ≥ threshold as solved for non-binary rewards. `bench eval metrics --json` no longer breaks long strings. See `docs/reference/pass-at-k.md`.
- **CI, results and JSON contracts.** CI: `bench eval run --fresh`, `--job-name`, `--fail-under`, `--fail-on timeout,error,verifier-error` and `--summary-out` (a `benchflow.run-summary` document); `bench eval resume JOB_DIR`; a resumed run says so and names `--fresh`; SIGTERM cancels the run, deletes its sandboxes and exits 143; an unknown one-word agent and a rejected `DAYTONA_API_KEY` stop the run before a job exists; `bench sandbox list` shows only your sandboxes (owner label, full IDs, `--json`) and `bench sandbox cleanup --all` deletes them regardless of age, with a reason for every skip and exit 1 on backend failure; `BENCHFLOW_LOG_LEVEL`/`BENCHFLOW_LOG_FORMAT=time`; the heartbeat stays on for a one-task job at any concurrency. A built-in `nop` agent is the empty control run. Reading results: `Job.denominators_by(by=...)`, `compare(..., by=("agent", "model"))` with a mixed-model caveat, paired-task headline rates (`a_paired`/`b_paired`), `bench eval inspect --by/--include-controls/--csv` and `compare --by`; the best attempt is kept per agent and model, not per task; records carry cache tokens, execution, assessment and run settings; `Trial.agent`/`model`/`duration_sec`; compact `Trial`/`Job` reprs; `results.jsonl` without task names and `summary.json` passed as a job are refused with a clear message; `agent_env_keys` survive a second config save. JSON contracts 1.1 (`schema_minor`): `benchflow.trial` carries every rubric review in full (`rubric_reviews`: scoring revisions from `bench eval run`/`bench eval score` and `bench review` audits, each with the rubric's criteria, weights and scales, reviewer agent/model/effort, per-criterion verdicts with cited evidence files, and the weighted reward arithmetic), `usage.cost_status`, `sandbox` and `verifier.reward_details`; `benchflow.job` adds `groups`, `interrupted`, `error_categories` and `timing_totals`; `benchflow.comparison` adds `by`, `rows[].group` and the paired denominators. An unscored trial's `results.jsonl` row has reward null, not 0.0. New page: [Analysing runs](docs/analysing-runs.md).
- **Branching: training rows v2, snapshot reuse, costs.** Branch-tree training rows are version 2: OpenAI chat-template shape (the agent's tool names, `function.arguments` JSON strings, merged assistant turns, a `tools` column), `session` copied from the fork, the source conversation prepended for `--from-checkpoint` trials (`prefix_source`, `prefix_complete`), oracle children skipped, DPO-ready same-request pairs (`prompt`/`chosen`/`rejected`), and `--min-reward`/`--expected-rows`/`--manifest` applied; `bench train validate --format branch-tree` checks them and JSON Schemas are in `docs/reference/schemas/`; SFT conversion of a branch run points at `--format branch-tree`. A fork reuses the checkpoint already taken at the fork point instead of snapshotting again (`Rollout.branch(reuse_snapshot=)`), and a `--from-checkpoint` trial starts straight from the kept snapshot on Daytona. The branch view (1.1, with `kind`/`schema_version`) and `bench eval branches` show input/output/cache tokens, a list-price USD estimate and cost per scored child, discarded parents (`unscored_by_design`) and reused snapshots; `bench eval branches` exits 2 on a missing path. Retries record `tool_calls`/`no_work`, `--retry-prompt` expands `@instruction` and `@verifier_feedback`, and the run summary prints a retries line. `bf.branch` gains `on_event`, `child.advantage`, `checkpoint=` and defaults from the checkpoint's trial, with errors naming Python keywords. Each fork folder has `labels.json`; `timing.json` itemizes `checkpoint_snapshot`.
- **Branch views, sturdier and resumable branch children.** `benchflow.branch-view/1` is one versioned JSON document per branched trial (per fork and child: request, status, reward, advantage, tokens, USD, sandbox-seconds, timings, snapshot reuse, lineage), from `bf.load_trial(...).branch_view`, `bf.load_job(...).branch_views()` or `bench eval branches PATH [--json]`, documented in `docs/reference/branch-view.md` with a pure builder a viewer can copy. A branch child that failed before its agent did anything (a provider hiccup) is retried once and siblings of a failed child keep running in `bench eval branch`/`bf.branch` (`--child-retries`, `--stop-on-child-failure`; opt-in on `Rollout.branch`). Automatic checkpoints record the agent session id, so `--from-checkpoint … --resume-session` and `--retry-from-checkpoint … --retry-resume-session` continue the conversation. Forks record `value_stderr`. A sandbox created from a branch snapshot no longer re-uploads the task files.
- **Versioned JSON of results, and `bench eval inspect` / `bench eval compare`.** `Trial`, `Job` and `Comparison` gain `to_json_dict()`/`to_json()` producing `benchflow.trial`/`benchflow.job`/`benchflow.comparison` documents (schema_version 1) built from pydantic models that also generate the committed JSON Schemas in `docs/reference/schemas/` (see `docs/reference/json-export.md`). `bf.load_trial` reads the `bench review` rubric verdict (`trial.review`), `bf.load_job` reads folders that only have `results.jsonl` rows, and `bf.compare` checks that both sides ran with comparable settings (task digest, model, harness, dataset, reasoning effort, sandbox, sandbox user, timeout, agent variables, prompts): an undeclared difference warns, or refuses with `on_mismatch="raise"`; `vary=` declares the intended ones. `bench eval inspect PATH [--json]` and `bench eval compare A B [--json]` run the same code.
- **Faster parallel branch children, branch cost accounting, retry from a checkpoint, and a Branching guide.** A sandbox created from a branch snapshot (isolated children, `--from-checkpoint`) now checks what it already holds and reuses it: the agent binary (verified present), and the verifier's pre-agent baseline, which was previously re-captured from the checkpoint state (fix: a build file tampered with before the fork could have been restored as legitimate); task `setup_commands` are no longer re-run on a checkpoint. `_snapshot_build_config` runs in one sandbox command instead of 13 at the start of every trial. With more children than `--concurrency`, the next ones are prepared while earlier ones run. Each child, fork and trial records `cost` (tokens, USD when the provider reports it, sandbox-seconds); `bench eval branch` prints a per-fork cost table and `bf.branch` results carry it. `bench eval run --checkpoints … --retry-from-checkpoint on-failure|on-timeout [--retry-prompt]` forks one retry from a failed trial's last checkpoint, verified the same way; the trial's reward is kept and the retry's is reported next to it. New `docs/branching.md`.
- **Read, compare and save from Python.** `bf.load_job` / `bf.load_trial` read finished job and trial folders (current and older layouts) into typed `Job`/`Trial` objects with trajectories, verifier output, costs, checkpoints and branch lineage; `Job.denominators()` and `bf.compare(job_a, job_b)` count like the viewer (attempted, scored, verifier errors and unscored apart; control runs left out). `Evaluation.to_yaml/to_dict/from_dict` and `RolloutConfig.to_yaml/to_dict/from_yaml/from_dict` round-trip configs with `bench eval run --config` (agent_env values only on request). `docs/reference/cli-python-parity.md` maps every `bench eval run`/`branch` flag to Python, and a test fails on drift. `docs/examples/python-sdk/quickstart.py` is a tour in `# %%` cells, and the folder's README is the examples gallery.
- **`bf.branch` / `await bf.abranch`: branch a run from Python in one call.** It builds and validates the same plan as `bench eval branch`, runs the same driver, writes the same job folder and returns a typed `BranchResult` (V(checkpoint), each child's label, status, reward and reward source, the parent's `RolloutResult`, `to_records`/`to_csv`/`to_jsonl`). Children are a `label -> prompt` mapping or `ChildSpec` items for nested forks; every CLI option is a keyword.
- **Parallel, nested and resumable branch children; automatic checkpoints; branch-tree training export.** `bench eval branch --concurrency K` (`branch(isolate_children=True, concurrency=K)`) runs a fork's children as sub-rollouts in their own sandboxes created from the snapshot (Daytona starts them straight from it), at most K at once, each a full trial folder under `branches/<fork>/children/<node>`. `--child label=…,parent=LABEL` forks again from a child's state (nested forks in one `tree.json`). `--resume-session` (`resume_session=True`) makes children resume the parent's agent conversation with ACP `session/load` (Claude Code); `tree.json` records `snapshot.agent_session`. `--checkpoints every-prompt|prompt:N,M` with `--checkpoint-keep` on `bench eval run` and `bench eval branch` keeps sandbox snapshots after chosen prompts (`checkpoints.json`), so `--from-checkpoint <trial> --checkpoint prompt:N` works on normal runs; `bench sandbox cleanup` also removes stale Docker `bf-snap-*` images. `bench train convert --format branch-tree [--pairs PATH]` exports one row per child (shared prefix, continuation, reward, value, advantage) and sibling preference pairs. Forks record `timing_sec.children`.
- **Sync, async and batch entry points in Python.** `bf.arun` (async; `bf.run` stays the same function) and `bf.run_sync` (blocking, also inside a running loop such as Jupyter); `bf.as_completed`, `bf.arun_batch` and `bf.run_batch` run a list of `RolloutConfig`s with bounded concurrency, sharing one job directory, and return a `Results` list with `n_passed`, `mean_reward` and pandas-free `to_records()`/`to_csv()`/`to_jsonl()` (also on `EvaluationResult`, via `RolloutResult.to_record()`). `Evaluation.stream()` yields results as tasks finish, `Evaluation.run_sync()` blocks, and `Evaluation.resume(job_dir)` finishes an interrupted job from the `evaluation.json` every job now writes (agent_env key names only, never values). A job holds `<job_dir>/.evaluation.lock` while it runs, so a second run of a live job is refused. Every public function and method is annotated (a test enforces it) and the SDK modules' doctests run in the suite.
- **`bench eval branch`: fork an agent run from the command line.** Runs each task up to `--checkpoint-after-prompt N`, snapshots the sandbox (or declared environment state), runs every `--child label=NAME[,prompt=…]` from that checkpoint with a fresh agent session scored by the task verifier, then continues or discards the parent (`--parent`). `--agent oracle` children run `solve.sh`; `--retain-snapshots` keeps the checkpoint and `--from-checkpoint` branches again from a kept sandbox snapshot. The job folder keeps the normal layout plus `tree.json` per trial and a children-counting `summary.json`. `Rollout.branch()` gains `restore_parent=False` (skip the parent restore after the last child; the rollout is then marked discarded and refuses to continue) and `child_requests` (what each child's runner does, recorded as `intervention.requested`); each fork records `timing_sec`. Branch lineage now reaches `result.json` (per-child `nodes`, `parent_node`, `parent_restore`), `results.jsonl` (`info.branches`) and each child's `observation.json` (`lineage`).
- **Typed results for Python callers.** `RolloutResult` gains `reward`, `passed` and `rollout_dir`, and `RolloutResult.load(path)` reads a finished rollout back from its `result.json` and trajectory. `Evaluation.run()` returns `results` (task name to `RolloutResult`, resumed tasks included) and `job_dir`. `EnvironmentManifest`, `load_manifest` and `resolve_source` are importable from `benchflow`. The Python API reference documents results, batches, manifests and deprecations, and `docs/examples/python-sdk/` has runnable scripts for each.
- **Branch runners learn which child they run, and branched results count their children.** A child runner that declares a keyword-only `child` parameter receives a `BranchChild` (index, label, node, fork id); `result.json` gains a `branches` block with the children's summed native token usage and phase time; `docs/examples/branch-agent-run.py` and `docs/composed-checkpoints.md` show how to branch a real agent run.
- **`bench doctor` and `bench eval smoke`.** `bench doctor` checks Python/uv, the Docker daemon and Colima VM, the Daytona SDK and key, each agent's credential (names, sources and expiry, never values; an expired `~/.claude/.credentials.json` is flagged), sandbox agent pins and model-endpoint egress, with a fix per problem, `--json` output and exit 1 on a required failure. `bench eval smoke` then runs the bundled hello-world task once per credentialed agent, one at a time on Docker or Daytona, and prints reward, time and trajectory path. See `docs/getting-started.md#run-everything-doctor-smoke-eval`.
- **Embodied rollouts (`benchflow.embodied`).** BenchFlow now owns the low-level layer for robot and simulator tasks: an embodiment spec (sensors with privileged vs visible state, named action groups with units, limits and control modes, typed skills, budgets) that covers arms, bimanual rigs, dexterous hands, mobile manipulators, quadrupeds, humanoids, drones and skill-only robots; the agent-facing `robo` command and wire protocol 2 (`robo act GROUP=V1,V2 ...`, `robo skill NAME ARGS`, with every protocol-1 command kept); the trusted episode server (budgets, video, judging, oracle token, vision mode, roles, a per-step state/action/reward trace); built-in arm and gripper skills; the simulator-sidecar task format base class `EmbodiedTaskFormat`; and the physical verifier. Trials with an episode record also get `trainer/embodied_steps.jsonl` and `trainer/embodied_episode.json` (per-step transitions, video index). `bench eval run --seeds 0-4` runs each task once per seed (formats that implement `materialize_variant`) and adds pass@k, mean, std and reset reproducibility per task to `summary.json`; `bench embodied report|export|check-spec` work on any job. First user: robouse. See `docs/embodied.md`.
- **Embodied tasks refuse branching, checkpoint restores and replay unless they declare a restorable world.** One declaration, `metadata.embodied` (every `EmbodiedTaskFormat` package has it), says what software may do to the task's world: `mode` (`sim`, `real` or `hil-mock`), `world_restore` and `action_replay`, both false by default; a real or hardware-in-the-loop embodiment cannot declare either. A container snapshot holds neither a simulator's process state nor a robot's arm and scene, so `bench eval branch` and `bf.branch` (with or without `--from-checkpoint`) refuse such a task before any sandbox starts, `Rollout.branch()` refuses before anything is quiesced or checkpointed, `--retry-from-checkpoint` records `status: refused` (counted as `refused` in `checkpoint_retries`), `bench eval regrade` reports its trials as not regradable, and `benchflow continue` refuses to replay it. A finished trial's own episode record also marks it embodied, and wins over the task's declaration when it names a real or hardware-in-the-loop embodiment. `bench eval branch` and `bf.branch` also take task-format folders, as `bench eval run` does. See `docs/embodied.md`.
- **Task formats: foreign task folders run natively.** A package can register a task format through the `benchflow.task_formats` entry point (or `benchflow.task.formats.register_task_format`). When BenchFlow loads a folder the format claims, the format materializes it as an ordinary native package under `~/.cache/benchflow/task-formats/<format>` (override with `BENCHFLOW_TASK_FORMAT_CACHE`), and BenchFlow runs that package. This applies to `bench eval run --tasks-dir`, `bench tasks check` and `RolloutConfig` / `Evaluation` in the Python API. The first user is robouse: `bench eval run --tasks-dir <robouse>/tasks/metaworld --agent codex --sandbox docker` runs robouse's source tasks, with the simulator as a trusted compose sidecar that runs the verifier. See `docs/task-formats.md`.
- **task.md draft 2 packages run natively (`benchflow.taskmd`).** `bench eval run --tasks-dir`, `bench tasks check`, `bf.Evaluation` and `RolloutConfig` take a task.md draft 2 folder (the instruction, then a `toml task` config block; `sandbox/`, `verifier/`, `oracle/`) through a built-in task format, `taskmd`, that never claims a native `task.md`, which opens with YAML frontmatter. BenchFlow parses with task-md's own reference parser, vendored verbatim at a pinned commit, and refuses by name, before anything runs, every run-affecting field it cannot honor yet; a batch skips a refused package with a warning. The verifier decides test criteria from the script's CTRF report and scores the rubric (gates, points, levels, continuous scores, headline), and `llm` and `agent` criteria run `judge-loop@1` with the byte-exact `judge-prompt@1` prompt over the Anthropic Messages API (an API key or a Claude Code OAuth token; `BENCHFLOW_TASKMD_JUDGE_MODEL` substitutes a model), with an agent judge's commands in the separate verifier sandbox. Families run per seed (`--seeds`), their generator in a fresh container of the task's image; chained stages become turns of the run's agent; `reward.json` records `timed_out`. `bench tasks check` prints the reference checker's report beside BenchFlow's field-by-field decisions. Automatic rubric review leaves a task.md-schema `rubric.json` to its package's verifier. See `docs/task-authoring-taskmd-v2.md`, which lists what is honored, refused, or recorded, field by field, and where the judges depart from the spec.
- **Native Google Antigravity CLI agent (`antigravity`, alias `agy`).** The
  Antigravity CLI replaced the hosted Gemini CLI in mid-2026 but ships no ACP
  mode, so BenchFlow bundles an ACP shim over its headless `stream-json`
  protocol. The registry entry pins the native `agy` release (SHA-512
  verified, no Node.js), runs in Gemini API-key mode, streams tool calls with
  raw input/output and per-turn token usage, honors `--model` and
  `--reasoning-effort` over ACP (agy requires an effort; the default is
  `high`), discovers skills from `~/.gemini/config/skills` and
  `<workspace>/.agents/skills` (a `SKILL.md` read is recorded as the skill
  invocation), delivers task MCP servers through
  `~/.gemini/config/mcp_config.json`, routes API-key runs through the LiteLLM
  Gemini pass-through like the `gemini` agent, and enforces the no-web and
  denylist policies with `PreToolUse` deny hooks.
- **Verifier recovery from preserved solver evidence.** Tasks that declare `workspace_recovery` (Docker, or Daytona with a digest-pinned image) or `submission_files` can re-run a failed verification against the saved workspace or declared outputs without replaying the solver; `bench eval score` resumes pending recoveries. Their verifier writes its start receipt under `/run/benchflow` rather than `/logs/verifier`; every script verifier's command now writes one (#1136, under Fixed above). Tasks that declare neither are not recovered this way. See `docs/verifier-recovery.md`.
- **Composed checkpoints for rollout branching.** `Rollout.branch()` can checkpoint environment and sandbox layers together, restores the parent after every child (including failed and cancelled ones), and records fork lineage in `tree.json`. Adapted from #1046 by JeremyJC67. See `docs/composed-checkpoints.md`.
- **Structured diagnostics in `results.jsonl`.** Each row carries `info.diagnostics` (schema version 1) with usage, error and provider categories, preserved through export and scoring resume. Adapted from #1038 by Ziao Yang.
- **Claude subagent attribution in trajectories.** ACP events carry `parent_tool_call_id`; ATIF exports put each subagent in `subagent_trajectories` linked from the spawning tool call, and `bench train convert` keeps subagent calls out of parent rows (`--subagent-rows` emits them separately).
- **Event timing in trajectories.** Captured events record host receipt times and the ISO-8601 `ts`, `started_at` and `finished_at` fields the trajectory viewer reads.
- **Native Claude OAuth on no-web tasks.** An in-sandbox proxy admits only model requests to the Anthropic API and refuses requests that declare server-side tools, `mcp_servers` or URL sources, so a Claude Code `advisorModel` setting is refused on no-web runs. See `docs/native-oauth-no-web.md`.
- **Opt-in token capture at the LiteLLM gateway.** `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1` records prompt and completion token ids and logprobs per call where the provider returns them, with an explicit reason where it does not (schema `benchflow.token_capture.v1`, see `docs/reference/token-capture.md`).
- **`--codex-apps-policy {disabled,inherit}`** controls whether account Apps stay enabled for managed Codex runs.
- **Trajectory viewer: subagents, execution vs assessment, lineage.** `bench eval view` nests Claude subagent steps under the tool call that spawned them, reports execution and assessment separately (a verifier failure after a clean run reads "completed but unscored") with verifier-recovery evidence in the Verifier tab, and adds a Lineage tab for branched rollouts (`tree.json`) that opens each child's trajectory.

#### Changed

- **One start-up name list for every backend, and bash's own start-up names refused.** The 13 loader and shell start-up names (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT`, `LD_DEBUG_OUTPUT`, `LD_PROFILE`, `LD_PROFILE_OUTPUT`, `GCONV_PATH`, `LOCPATH`, `NLSPATH`, `HOSTALIASES`, `MALLOC_TRACE`, `BASH_ENV`, `ENV`) are emptied at create on every backend and carried aside as `_BF_IMAGE_<name>` for the task's own command; Daytona used to refuse an image that set one. An exported bash function (`BASH_FUNC_<name>%%`), `SHELLOPTS` or `BASHOPTS` with a value, in the image's `ENV` or the sandbox's environment, is refused at start on Docker, remote Docker, Docker-in-Docker, Daytona (create and checkpoint), Modal, Apple container and AgentCore (at image build): `benchflow: <where> sets bash start-up names (<names>); bash would run them as root before any command BenchFlow sends, and they cannot be given back to the task's own command; remove them from the image or the sandbox's environment`. On Daytona an image's exported function ran as uid 0 in root's shells, and `SHELLOPTS=xtrace` ran root's commands traced.
- **Neutral names for the task.toml layout, the dataset hub, the round-trip check and a verifier strategy, with no alias.** Each name below says what it is. An old value gets the ordinary unknown-value error, which lists the values there are (`--target 'x' is an unknown value; choose one of: task-toml`).
  - `bench tasks export --target` (and `bench export`): the previous export target is now `task-toml`, the default and only value. `compatibility/export-report.json` records `"target": "task-toml"`. The Python export API (`export_task_to_split_layout`, `build_compatibility_export_report`, `build_task_toml_roundtrip_conformance_report`) refuses any other target with the same error.
  - `bench hub list --provider`: the previous benchmark-registry provider is now `benchflow`, which lists BenchFlow's dataset registry (title "BenchFlow Registry").
  - `bench hub check`: `--registry` defaults to BenchFlow's registry instead of the previous outside registry URL, and `--cache-dir` to `.cache/hub/registry` instead of the previous cache folder. Each record names the `registry` it read and an `env_uid` of `<dataset>/<task>@<ref>`, in place of the previous `framework` and `hub_url` fields and the prefixed UID. The summary line reads `Registry check (<registry>): ...`.
  - The task.md preset that marks a task exportable to task.toml is now `task-toml-compatible` (its category too; its `benchflow.compatibility` key is `task-toml`).
  - `benchflow.task`: the previous round-trip conformance names are now `TaskTomlRoundTripConformanceReport`, `TaskTomlRoundTripMismatch` and `build_task_toml_roundtrip_conformance_report`.
  - `benchflow.hub`: the module is `benchflow.hub.registry`, with `RegistryTaskRef`, `load_hub_registry`, `select_registry_tasks`, `check_registry_tasks`, `registry_env_uid`, `materialize_registry_task` and `BENCHFLOW_REGISTRY`. The previous module, its names and its default-URL constants are gone.
  - `benchflow.adapters`: the inbound adapter for task.toml folders is `benchflow.adapters.task_toml`, with `TaskTomlAdapter` and `from_task_toml_dir`. The `InboundTask.source` it returns, and the `benchflow.compat.source` a materialised task.md records, read `task-toml`. The previous module and its names are gone. Its missing-file and unknown-format errors, and the task.toml spelling errors, describe the format as task.toml.
  - The verifier strategy that runs a Python criteria runner from a root folder in `verifier/` is now `criteria-runner` (`type: criteria-runner` in `verifier/verifier.md`, and the `criteria-runner` strategy key in a hybrid task.md's `benchflow.verifier.implementation.strategies`; `criteria_runner` is accepted there as before for the `-`/`_` spelling). The previous value gets the ordinary unknown-value error, which lists the strategies there are (`verifier.strategies.<name>.type '<value>' is an unknown value; choose one of: agent-judge, criteria-runner, llm-judge, ors-episode, script, taskmd`). With it: the runner's environment variables are `BENCHFLOW_CRITERIA_RUNNER_ROOT`, `BENCHFLOW_CRITERIA_RUNNER_CRITERIA` and `BENCHFLOW_CRITERIA_RUNNER_MANIFEST`; the manifest is `/logs/verifier/criteria-runner-manifest.json`, with `"version": "benchflow.criteria-runner.v1"` and `criteria_runner_root`; the task.md authoring profile is `criteria-runner` (entrypoint `verifier/criteria_runner/reward.py`); the module is `benchflow.task.verifier_criteria_runner`. A hybrid verifier strategy named with the previous one-word spelling of the rewards package stays unrecognised.
  - **Upgrade note: a task folder that declares the previous strategy value now fails to load with that unknown-value error, before any sandbox starts.** Change its `type:` to `criteria-runner` in `verifier/verifier.md` (and the strategy key in a hybrid task.md). A runner that reads the previous environment variable names or manifest file name must read the new ones. The folder that holds the runner can keep its name; `root:` names it.
  - Docs: the schema example folder is `docs/examples/task-md/task-toml-parity`; the task-package examples show `source: legacy` (what `bench import` writes), `runner: custom-step-runner` and a `criteria_runner` strategy.
- **`-d name@version` reads BenchFlow's registry and nothing else.** BenchFlow's registry is its published list (`registry.json` in the skillsbench repo) plus entries the package now ships for datasets that list lacks: `terminal-bench@2.0`, all 89 tasks pinned to commit `69671fba` with a content digest each, so they are digest-checked. A dataset BenchFlow's registry does not have stops with `Dataset '<name>@<version>' not found in BenchFlow's registry`, and the next step says to pick one `bench hub list --provider benchflow` lists or to name a registry with `--registry <URL or file>`; no outside registry is read on its own. `bf.resolve_source` also takes BenchFlow's dataset IDs: `bf.resolve_source("terminal-bench-2", path="log-summary-date-ranges")`.
- **Only job folders carrying a BenchFlow marker are resumed.** A job folder is BenchFlow's when it has a marker BenchFlow writes as the job starts or later: `evaluation.json` (written as a 0.8 job starts), a BenchFlow 0.7 `summary.json`, or a trial BenchFlow started. A trial is BenchFlow's when its `config.json` (written as the trial starts) records `benchflow_version` or `harness_mode` (a 0.8 trial, `bf.run(job_name=...)` too) or has every key a 0.7.x trial's has (`acp_transport`, `sandbox_locked_paths`, `sandbox_setup_timeout`, `agent_idle_timeout_sec`), or when its `result.json` (or solver checkpoint) is a BenchFlow rollout result (`rollout_name`). So a 0.7 job stopped before any trial finished is resumed too. Any other folder with files is a folder BenchFlow did not make: an unnamed run starts a new job beside it, a named run refuses it (`<folder> is a folder BenchFlow did not make; nothing in it moves`), and nothing in it moves. A folder whose only contents are empty folders counts as empty. Before, only folders with two particular foreign file markers were treated that way.
- **`bench check` reads a verifier's model use from what the task declares.** `[verifier] models = true`, a provider key its verifier env names, or the judge files an import recorded. It no longer guesses from the command names test.sh runs, so a script calling a model now needs one of those declarations to get the warning.
- **Integrity action records no longer skip one foreign event name.** A tool call of an unknown kind is handled by what it carries: with a path it is witnessed as a read; with no input, or any other input, it gives no record and is counted in `unmapped_kinds`, since a tool with no arguments can still act. This replaces a list entry keyed on one foreign event name; calls of that kind are now counted like any other unknown kind.
- **A sandbox's transfer primitives take a required `admission=` keyword.** `upload_file`, `upload_dir`, `live_process` and `script_runner_upload` of every built-in backend now refuse a call without the `Admission` that `benchflow.trust.state.admit()` returns for that sandbox, service and start, raising `AdmissionRequired` before any byte moves; BenchFlow records each agent-derived ingress this way. Code that calls a built-in backend's primitive directly must admit first (see the Sandbox protocol in `docs/reference/python-api.md`), and a backend of your own must accept the keyword. The SDK's `Environment` methods admit for you and are unchanged.
- **BenchFlow's LiteLLM model proxy reads no `.env` file.** LiteLLM calls `load_dotenv()` when it is imported, so every host proxy also loaded a `.env` that python-dotenv found by itself. A real proxy imports LiteLLM from BenchFlow's start-up hook, before its script has a file, so python-dotenv searched from the proxy's working folder: its runtime folder, then `$TMPDIR` (usually `/tmp`), then `/`. A `/tmp/.env` or `/.env` left by anyone could add provider settings (`*_PROXY`, `AZURE_*`, `LITELLM_*`) that the user never passed to BenchFlow, and a proxy forked by the local runtime searched elsewhere. Every BenchFlow proxy (the host proxy, cold or from `bench daemon`, and the in-sandbox proxy) now runs with `PYTHON_DOTENV_DISABLED=1`, and BenchFlow passes everything the proxy needs in its environment. **If you relied on a stray `/tmp/.env` or `/.env` reaching the proxy, export those variables, or put them in the `.env` in the folder you run `bench` from, which BenchFlow itself reads.** The proxy also no longer puts folders it finds from its working folder on its import path (LiteLLM's proxy modules add the folder and its `../..` when they are imported); it imports from its runtime folder and installed packages only.
- **OpenClaw loads from the agents repo, and a release pins that repo to a commit (#1093).** OpenClaw's ACP shim now lives in `benchflow-ai/agents` (`acp/openclaw`) and loads on first use, like the other catalog agents. A one-word agent name that neither BenchFlow nor the catalog knows now fails before any sandbox starts instead of running as a command; commands with arguments or a path still run. The default catalog is a commit, `benchflow-ai/agents@9aaec336b68426058db1bd83fd2c463be2935ce7`, the agents `main` after #71 to #81 merged (the OpenClaw shim moved by agents #72, with #73's prompt streaming, #80's pinned mini-swe install and #81's private mini-swe Python), not the `main` branch itself, so a release keeps loading the manifests it was tested with. `BENCHFLOW_AGENTS_SOURCE=benchflow-ai/agents@main` follows `main`; `BENCHFLOW_AGENTS_DIR` points at a local checkout.
- **Every `bf.run` form uses the task's own agent timeout unless you pass one.** `RuntimeConfig.timeout` defaulted to 900 s, which the `bf.run(bf.Agent(...), bf.Environment...)` / `Runtime.execute()` form always applied, overriding the task's `[agent] timeout_sec`; it now defaults to `None`. When that form runs a task whose timeout is not 900 s without an explicit `timeout`, it emits a `FutureWarning` naming both values. Pass `RuntimeConfig(timeout=900)` to keep the old limit.
- **The `Agent + Environment` form returns `RolloutResult`**, like every other `bf.run` form, so it now carries tokens, cost, error categories and `score_outcome`. `RuntimeResult` is deprecated: constructing it warns, and `result.verified`, `result.messages` and `result.snapshots` keep working on `RolloutResult` with a `DeprecationWarning` (`messages` and `snapshots` were always empty). Code that checks `isinstance(result, RuntimeResult)` must check `RolloutResult` instead.
- **Codex Apps are disabled by default for scored `codex-acp` tasks.** The native Codex version must fall inside the range the pinned `codex-acp` adapter declares, and scored runs as root or with `--sandbox-user null` are refused; pass `--codex-apps-policy inherit` to keep the previous behaviour. See `docs/codex-apps-policy.md`.
- **Hardened verifiers check pytest plugin provenance.** The guard loads plugins from BenchFlow's protected verifier runtime, the running pytest or unchanged image files; discovery ignores workspace metadata. A reported refusal, guard crash, incomplete check or load failure during verification gives score 0 with a named warning. Initial hardening failures before a test command starts remain setup errors; unrelated controller, transport and provider failures remain unscored. The absence of a reported problem does not prove loading, and a `registered` marker cannot cancel an adverse signal. The guard directory joins the verifier `PYTHONPATH` where needed; task preflights still reject a task-supplied `PYTHONPATH`. Plugin discovery comments adapted from #1117 by tulerfeng. See `docs/sandbox-hardening.md`.
- **Verification waits for sandbox-user processes to stop** before scoring, using a POSIX-shell check that needs neither procps nor Python.
- **A hard deadline that cuts the agent phase is reported as `timeout` and the rollout is still verified.** Only an overrun of the verification grace period remains `INFRA_ERROR`. Bare-timeout reclassification adapted from #1131 by AdnanElAssadi.
- **The automatic reviewer's deadline scales** with setup-command timeouts and the size of the evidence it uploads.
- **Results awaiting assessment stay unscored.** A result that declares an `assessment` of `pending` or `unassessable` (or a legacy `status: awaiting_assessment`) has no reward in score summaries, rescoring or the viewer, whatever reward sits beside it, and resume never re-runs it; score summaries gain an `unscored` count.
- **Removed 129 unused re-exports** from `benchflow.task.verifier`, `benchflow._utils.task_authoring`, `benchflow.task.acceptance_live` and `benchflow.rollout`; import those names from their defining modules.
- **`--source-repo` with `--source-path` downloads only that path.** `bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check` cloned the whole repository to run one task: a 1.1 GB checkout (447 MB pack) plus a 644 MB snapshot worktree, about 21 s before any sandbox started. A source with a path now makes a blob-less sparse clone (`--depth 1 --filter=blob:none --sparse`) and checks out that path alone, so the task costs about 3 MB and under 2 s; later paths join the same cache, and the per-commit snapshot holds only the requested paths (it is created with `--no-checkout`, because git before 2.36 would fetch everything into a new worktree). A source with no path, a path through a symlink and a folder that holds no BenchFlow task (a foreign benchmark whose source adapter reads other files of the repository) still get the whole repository, and caches made by earlier releases are reused as they are. A missing `--source-path` lists the folders of its deepest existing parent and suggests a close match, instead of listing the repository root. Git older than 2.26, which has no `sparse-checkout add`, still clones the whole repository.
- **CLI help points a first-time user at the next step.** `bench --help` ended with no route into the docs and `bench eval --help` described the group as "Evaluation commands.". `bench --help` now ends with `New here? Run bench doctor, then follow <getting-started>`, and `bench eval --help` says what the group does. On `bench eval run`, the flags a first run needs read "Agent name", "Model", "Subpath within the repo" and "Output directory"; they now name the values they take, including the built-in `oracle` and `nop` agents, that `--source-path` is the only folder downloaded, and that a plain run resumes its latest job. (The `--agent` and `--jobs-dir` of sibling commands keep their own shorter help.) `bench tasks init` writes a scaffold that fails `bench tasks check` until its `[REPLACE: ...]` placeholders are filled in, and said nothing about that: it now ends by printing the two commands that come next, `bench tasks check <dir>` and an oracle `bench eval run`, and a link to the task-authoring guide. That run command carries its own `--jobs-dir`, so it cannot resume another experiment's job, and `--fresh`, so that re-running it after an edit runs the task again instead of reporting the cached result of the code the author just changed.

#### Deprecated

- `benchflow.SDK` (use `bf.run(bf.RolloutConfig(...))`, which takes the same keyword arguments), the `bf.snapshot` / `bf.restore` / `bf.list_snapshots` aliases (use the `workspace_*` names) and the unused `RuntimeConfig.max_rounds`, `snapshot_policy` and `reward_stream` fields each emit a `DeprecationWarning` and keep working.

#### Fixed (continued)

- **task.toml multi-step tasks are refused on `steps`.** A multi-step task has no root `instruction.md`, so `bench eval run` and `bench tasks check` failed on the missing file instead of the documented refusal, and in a batch the trial left no folder (summary.json counted it, `inspect` and `bf.load_job` did not).
- **The oracle runs as the task's `[agent].user` and in its `[environment].workdir`,** a correct solution to a task that declares either previously scored 0. With neither declared the oracle still runs as root in the image's default directory. Oracle trials also record `agent_execution` time.
- **Sandbox user setup works on busybox images (Alpine)** through `adduser`, and fails with a clear setup error when no user can be created, instead of failing later as a verifier hardening identity error.
- **A Daytona sandbox left by a failed create that names no sandbox is deleted.** A gateway 502 while the SDK waited for a new sandbox to start left it running; each create attempt now carries a unique label used for the cleanup.
- **summary.json counts a fresh run's integration failures by their recorded cause** (it said `unknown`), and **reports a null cost when no trial reported one** (it said `0.0` for subscription-login runs; `bench eval metrics` too).
- **`bench eval view` shows automatic rubric reviews** stored in the trial's `scoring/` report, and **`bench train token-coverage` no longer counts a trial's reviewer run as a rollout.**
- **A task that relies on symlinks is refused before it runs.** Sandbox uploads skip symlinks on purpose (#411), so a task whose `verifier/` linked to another task's reached the sandbox with an empty `/tests`, failed as `Verifier setup failed: chmod exited with rc=1` (a retried verifier-infrastructure error) and ran the agent again on every retry for nothing. `bench tasks check` now reports such a task (`Task uses symlinks, which are not uploaded into the sandbox: …`), a batch skips it with a warning naming it (legacy `task.toml` layouts too), and the bundled `citation-check-network` example holds copies instead of symlinks.
- **A task with symlinks no longer aborts a batch.** Source provenance for a local task in a git checkout is inferred on a best-effort basis by hashing its files, and a symlink (such as `docs/examples/task-md/real-skillsbench/citation-check-network`, which links its `environment/`, `oracle/` and `verifier/` to `citation-check`) raised there: the task failed as an unexpected exception, and building its result raised again after every task had finished, so the job exited 1 without `summary.json` or `results.jsonl`. Such a task now gets no inferred provenance, with a warning.
- **A Codex model the login does not offer fails once, with the list.** `--agent codex-acp --model <name>` for a model missing from the session's advertised models sent the name anyway, got an opaque `ACP error -32603: Internal error`, and was retried twice with new sandboxes before failing the same way; the run now stops at once with `agent integration failure [agent_model]: codex-acp does not offer model '<name>' for this login; it offers: …`. The trial is an unscored agent-integration failure (`error_category` `agent_integration`, `integration_failure_info.cause` `agent_model`), is not retried, and a repeat trips the batch breaker.
- **`bench eval compare` labels each side by its own folder.** Comparing a timestamped job folder with a named one (`jobs/batch-oracle`) labelled the named side with its parent folder (`jobs`); only a timestamped side now falls back to its parent.
- **A transposed short agent name is refused by the Python SDK too.** `bf.run_sync(RolloutConfig(agent="oracel", ...))` passed the pre-run check (the name scored just under its similarity cutoff), started a sandbox and failed minutes later as an ACP initialize timeout; two swapped neighbouring letters in a registered name (`oracel`, `gemnii`) now get the "did you mean" error before anything starts, as `bench eval run` already did.
- **Token capture asked for on a run that skips the gateway now warns.** `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1` on a subscription-auth run (or an agent the gateway cannot route) was ignored without a word, because capture happens in the gateway; the run now logs that no token ids or logprobs will be recorded.
- **`bench train convert` no longer converts a `results.jsonl` row whose capture ended inside a Claude subagent as the parent's conversation.** The results writer keeps only calls a later request consumed, so when a run stopped inside a subagent the parent's spawning call was dropped, the row's steps were the subagent's alone, and the spawn-prompt check read back from the row found nothing: one row of subagent calls was written as a parent row. When the capture shows subagent activity, the writer now tags each step's `extras` with the owner it attributes from the whole capture (`agent_role` = `parent`, `subagent` with `parent_tool_call_id`, `helper` or `unattributed`), and conversion refuses a row with a non-parent step or one that does not end on a parent call. Rows written before this change keep the old check.
- **A reviewer that loses its transport before a verdict is retried once (#1144).** A Daytona PTY reset during the review stage (websocket close 1006/1008, a closed peer) ended the trial with `reviewer did not produce a readable review-result.json`, although the verifier output was already on disk. The review now runs once more in a fresh reviewer sandbox; the failed attempt's folder is kept under `transport-retry-<n>` beside it and the trial's notes name the lost attempt. A readline timeout is not retried, since the reviewer was silent past its budget.
- **A Daytona PTY or AgentCore read no longer kills a stage its idle budget still allows (#1143).** The transport's readline guard was a fixed 900 s unless `BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT` was exported, so `--agent-idle-timeout 7200`, `--agent-idle-timeout none` or a reviewer with a raised idle budget died as `PTY readline timeout (900s)` after 15 silent minutes. Each prompt now raises the guard to its idle budget (or, with no idle watchdog, its wall budget) plus 60 s; an explicitly set `BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT` / `BENCHFLOW_AGENTCORE_READLINE_TIMEOUT` still wins.
- **The pre-verifier `conftest.py` sweep ran nowhere with GNU find.** It paired `-prune` with `-delete`, which GNU find (Debian, Ubuntu, `python:*-slim`) refuses with exit 1, and its error went to `/dev/null`, so agent-planted `conftest.py` files outside `/tests` survived hardening. The sweep now uses `-exec rm -f -- {} +`; GNU find on `python:3.12-slim` reproduces the old failure.
- **`bf.Evaluation(...).run()` from Python makes the same pre-run checks as `bf.run` and `bench eval run`** (misspelt agent or sandbox, Docker, an expired Claude login file); `Evaluation(..., preflight=False)` skips them, and the CLI passes it because it checks first.
- **Blocking SDK calls no longer log `Unclosed client session`**: `bf.run_sync`, `bf.run_batch`, `Evaluation.run_sync` and `bf.branch` close the Daytona client their private event loop created.
- **`bench eval run --retry-attempts` help** no longer calls the option reserved; it sets the retries per task.
- **The Python SDK makes `bench eval run`'s pre-run checks.** `bf.arun`/`run_sync` and the batch functions refuse a missing task directory, an unknown sandbox and a misspelt agent name before a sandbox starts (a misspelt agent used to fail 60 s later as an ACP initialize timeout), raise with `bench doctor`'s fix when Docker is not ready, and warn when Claude would fall back to an expired login file (it used to fail after the sandbox started). `BENCHFLOW_SKIP_PREFLIGHT=1` skips the host checks.
- **Environment-plane manifest services start on Daytona.** The sandbox env-file wrapper joined the caller's command with `&&`, so a detached service start (`nohup CMD ... &`) backgrounded the whole list and held the Daytona session open; every manifest rollout failed with `Command timed out after 15 seconds`.
- **`bf.run("agent", task_path=..., config=RuntimeConfig(...))` honours `rollout_name` and a changed `timeout`**, as the `Agent + Environment` form already did, and rejects an `Environment` object passed as `env` instead of building a second sandbox.
- **Branch snapshots are deleted, owner-scoped and credential-free.** `Rollout.branch()` deletes each fork's container snapshot when the fork finishes, including after a failed or cancelled child (`retain_snapshots=True` keeps it); Daytona snapshot names carry `BENCHFLOW_DAYTONA_OWNER` and `bench sandbox cleanup` plus the eval-start auto-reap delete that owner's stale ones; agent credential files are removed before capture and written back after capture and after every restore.
- **Branch scoring is honest for custom runners.** A custom runner's verifier reward is recorded as `reward_source: "verifier"`, a missing verifier reward or a `None` return leaves the child unscored instead of 0, and `execute()` inside a child fills the pending node without `node=`.
- **A plain task's branch error names `snapshot_layers={"sandbox"}`**, and `DaytonaClientManager.get_client()` loads the SDK itself and creates a new client under a new event loop.
- **The host deadline honours config overlays** for agent and build timeouts. (#1134)
- **Bedrock keeps prompt caching for Claude Code conversations.** LiteLLM moved mid-conversation system messages and their cache breakpoint into the top-level system prompt, so conversation turns were never read from cache. (#1135)
- **Claude 5-family reasoning effort reaches the provider** when LiteLLM falls back to its bundled model list.
- **No-web agent runs get the capabilities their firewall needs** on Docker and Daytona DinD, DinD uploads the capability overlay it references, and images without python3 or openssl fail early with a clear error.
- **Rollout snapshots keep distinct databases that share a filename**, and Docker restore replays the container's host configuration (runtime, init, groups, tmpfs, labels) or refuses to restore.
- **A refused review keeps a timed-out solver's verifier result**, and one failing resumed review no longer cancels the others.
- **Oversized integer rewards no longer crash aggregates or exports.**
- **ACP session updates are kept when the agent omits `sessionId`**, and unchanged tool polls no longer rewrite the trajectory file.
- **A failed verifier start probe is treated as unknown**, so a transient exec error no longer cancels a running verifier as wedged.
- **ACP prompts return as soon as they finish.** The idle watchdog slept a full poll between checks, so agent execution time was rounded up to the next 30 s and the sandbox stayed up for the difference.
- **Daytona deletes a sandbox that fails to build or start** instead of leaving it for the reaper, and the error reports the real number of attempts and the sandbox id.
- **A `task.toml` that fails to parse is named.** A batch skips the task with a warning and a single-task run raises `MalformedTaskError` naming the file, instead of scoring a bare `TOMLDecodeError` as 0.
- **`bench doctor` no longer passes a Gemini Google login.** Gemini runs go through the LiteLLM proxy, which needs `GEMINI_API_KEY`, so the login is a warning and `bench eval smoke` skips Gemini instead of failing on it.
- **An unrunnable task directory says why.** Without `--include`/`--exclude`, an empty selection names the `bench tasks check` issues of the task (or counts the unrunnable task directories) instead of blaming the filters.
- **`--sandbox daytona` without the extra fails before the job starts**, and every missing-extra message gives the same `uv tool install` / `uv sync --extra <extra> --extra dev --locked` hint.
- **A trial whose pytest the verifier guard stopped by refusing a plugin is labelled** in the run summary (`pytest did not run: the verifier guard refused pytest plugin …`).
- **`bench eval run` and `bench eval smoke` point at `bench eval view`**; the viewer flushes its URL when stdout is a pipe, and names the localhost URL when it refuses another host.
- **A scoped Daytona run no longer warns about other operators' unscoped sandboxes** as possible orphans.
- **`bench eval run --sandbox docker` checks Docker before it creates a job.** It reuses `bench doctor`'s Docker check, so a missing `docker` CLI or a stopped daemon stops the command with doctor's fix line instead of a traceback and a published 0/1 job (`BENCHFLOW_SKIP_PREFLIGHT=1` skips the check). A rollout whose compose build cannot reach the daemon is recorded as a sandbox startup failure and is not retried.
- **An expired `~/.claude/.credentials.json` is flagged before the image build.** When a Claude run would fall back to that file and its access token has expired, `bench eval run` prints doctor's warning and the `claude setup-token` fix up front; the README gives the macOS token route.
- **A sandbox build whose context is missing a path is not retried** (Daytona's `Path does not exist`, BuildKit's `failed to calculate checksum … not found`); other startup failures still retry.
- **Oracle runs no longer fetch remote agent manifests**, and `--source-repo` no longer prints git's worktree progress.
- **`bench doctor` reports a missing buildx plugin**, drops the HTTP status from reachable network lines, and keeps its columns aligned for long host names; `bench --help` and `bench eval --help` list doctor, smoke and run first.
- **Trajectory viewer:** a subagent-spawning call is badged `agent`, provisional edit titles show the file, the Lineage tab explains value and runner return, branched runs are marked in the run list, a branch child's back button opens its parent, and counts of one read "1 run".
- **Daytona retries a sandbox delete that the API gateway failed.** A delete that gets HTTP 502, 503 or 504 is tried up to four times (about 7 s of backoff); a sandbox already gone on a retry counts as deleted. If every attempt fails, the error names the sandbox id, shows an HTML error page as its title, and points at `bench sandbox cleanup`. Timeouts, connection and rate-limit errors keep the two-attempt budget.
- **Benign teardown and mount lines are quiet.** On a Mac, a job directory outside `$HOME` (which Colima does not share) no longer warns that the verifier-log bind mount is not visible; elsewhere the warning names the path and points at `DOCKER_HOST`. A Docker rollout that could not start because Docker was unavailable no longer warns that `compose down` failed. Daytona no longer warns "Sandbox not found. Please build the environment first." when there is nothing to delete, and warns only when an unfinished create may have left a sandbox behind.
- **ACP trajectories record a tool call's final title** (`Write /app/hello.txt`, not `Preparing file…`) when a later `tool_call_update` retitles it; the live progress line shows it too.
- **The verifier's uv and pip state leaves an agent-writable `$HOME`.** Under a `WORKDIR /root` image, verifier caches, tool environments, managed Pythons and configuration move into a new root-owned directory after the agent stops. An unpinned pytest plugin that test.sh installs there, including through `uvx`, is still refused; a reported refusal scores 0 with a warning. Pin it in `[verifier] pytest_plugin_releases` or install it into the image. citation-check and other tasks using `--ctrf` receive pytest-json-ctrf from BenchFlow's protected verifier runtime. See `docs/sandboxes/threat-model.md#pytest-plugin-guard`.
- **The verifier's commands run under umask 022 on every backend.** BenchFlow set no mask for test.sh, and `docker exec` on Docker-in-Docker (Docker 29.8.1, runc 1.5.1) runs with 0000, so the uv cache test.sh filled came out group- and world-writable, the pytest plugin guard refused the verifier's own `pytest-json-ctrf`, and every task that passes `--ctrf` ended unscored there, the correct oracle included (89 of 89 in Terminal-Bench 2). The mask is now set in the command string for test.sh, script strategies, hybrid strategy runners and the separate verifier's unpack, so Docker, remote Docker, Daytona and Modal all get 0644 files and 0755 directories.
- **The verifier's uv and pip state always gets a fresh directory.** The move to `/_benchflow_verifier_<hex>` used to happen only when the agent could write `$HOME`, so with any other `WORKDIR` (Terminal-Bench 2's `/app`) uv's cache stayed in `/root/.cache/uv`. Every uv and pip cache, uv tool environment and uv-managed Python now moves there after the agent stops. uv always reads a pinned configuration: a safe `UV_CONFIG_FILE` already set, else root's own `~/.config/uv/uv.toml` when the agent could not write it, else the image's first safe system `uv.toml`, else an empty one; the workspace's `uv.toml` and `[tool.uv]` are not read from there (test.sh's own `uv pip install` and `uv run` in the workspace still read them; set `UV_NO_CONFIG=1`). A warm uv cache baked into the image is no longer used. The plugin guard does not trust what test.sh installs in this directory.
- **A separate verifier sandbox distrusts only what came from the agent.** It passed `sandbox_user=None` but kept the plugin guard's rule for the agent's own sandbox, blocking `/tmp`, `/var/tmp` and `/testbed` and requiring every plugin file to be root-owned and not group- or world-writable, although no agent process ever ran there. The transfer now reports the roots it writes (the workspace, each declared artifact's bundle root, `/logs/artifacts`), and in that sandbox discovery and the guard block those and `/logs` alone and accept the verifier image's own files without the ownership rule. Plugins still come only from the protected verifier runtime, the running pytest or the image: one left in the workspace, or one test.sh installs after the transfer, is refused.
- **A separate verifier sandbox always empties `BASH_ENV`, with a warning.** The activation files it would name are transferred from the agent's sandbox, so they have no pre-agent provenance.
- **A plugin guard refusal says what it could not trust and why.** The diagnostic identifies the refused plugin, path and reason, including ownership, write permissions, duplicate registrations or a missing installation. A reported refusal gives score 0 with a warning. The plugin-file inventory remains diagnostic evidence; it is not authority to exclude a trial or establish who caused the failure.
- **Plugin-guard markers are private and signed.** Markers live in the verifier user's mode-0700 guard directory and carry an HMAC-SHA256 under a per-verification key that is not put in the environment. Invalid signatures are ignored. A signed adverse marker or pytest's load-failure message gives score 0; a `registered` marker cannot override that evidence. The verifier user can read the key, suppress failure evidence and rewrite reward files, so the signature does not authenticate that user's code. The task script and what it imports remain trusted. Markers no longer depend on downloading `/logs/verifier`.
- **`bench tasks check` warns about pytest plugins installed into the workspace.** It recognizes common venv, uv and pip installation forms in the working directory and temporary folders. The warning names the script, plugins and commands. Pin plugin releases in `[verifier] pytest_plugin_releases` or install them into the image; unpinned `uvx` installs are refused too. A reported runtime refusal scores 0. Scripts that clear `PYTEST_ADDOPTS` are told that the guard does not check that code. The historical 413-script scan flagged mailman, archive/mailman, powerlifting-coef-calc, fix-build-google-auto and fix-build-agentops; that scan is not a final-release runtime result.
- **A task generated from the `multi-agent` profile gets a reviewer that can run.** The profile paired its reviewer role's `agent: claude-agent-acp` with `model: claude-sonnet-4-6`, an id the pinned adapter (`@agentclientprotocol/claude-agent-acp@0.81.2` with Claude Code 2.1.280) refuses at `session/set_config_option` with `-32603 Internal error`, so the review scene of every such task failed on each run, after the whole retry budget, as an opaque `acp_error`. The reviewer now uses `claude-sonnet-5`. The model API still serves the refused id; only the ACP adapter rejects it, which is why this surfaced as a protocol error rather than an unknown model.

## 0.7.11 — 2026-10-16

### Added

- **`--separate-verifier-include PATH` carries outputs outside the workspace
  into a separate verifier sandbox.** Only the workspace (`sandbox.workdir`,
  else the image's `WORKDIR`) crossed, so a task whose agent writes graded
  files elsewhere scored 0 with no error, though it passed in shared mode.
  Each include path
  (repeatable; SDK and config `separate_verifier_include`) is captured beside
  the workspace into `evidence/artifacts/`, hash-checked, and laid over the
  verifier image's copy at the same path; pseudo filesystems and roots such as
  `/`, `/usr` and `/home` are refused before any sandbox starts.
  `config.json` records `verifier_include`, and the transfer inventory lists
  each path as captured, missing or already in the workspace.
- **A separate verifier lists files the agent left outside everything that
  crosses.** When the workspace is frozen, the regular files written during
  the agent phase outside the workspace and include paths (skipping system,
  scratch and provider trees, package caches and the agent's home) are logged
  as a warning with a count per directory and examples, and recorded under
  `outside_workspace_writes` in `verifier-sandbox/verifier-sandbox.json`.
  Advisory: it never fails a rollout.

### Fixed

- **The verifier ignores pip's and git's per-user config under `/root`.**
  The verifier runs as root with `HOME=/root`, the agent's workspace in many
  tasks, so `~/.config/pip/pip.conf` or `~/.gitconfig` there were whatever the
  agent left, in both modes (a separate verifier receives the `/root`
  workspace too). `VERIFIER_ENV` now sets `PIP_CONFIG_FILE=/dev/null` and
  `GIT_CONFIG_GLOBAL=/dev/null` (git 2.32+; `/etc/gitconfig` still applies),
  and drops `PIP_NO_USER_CONFIG`, which is not a pip option and did nothing.
  A task that needs either config back sets it in its verifier `env`.
- **Package-manager caches stay out of workspace evidence.** `.cache/pip`,
  `.cache/uv` and npm's `_cacache`, `_npx` and `_logs` below a captured root
  are recorded as `package_cache` exclusions instead of captured. With a
  `/root` workspace the harness's own installs fill them as root before the
  agent starts, and they can be most of what the separate verifier receives
  and of the rubric review's upload.
- **`mcp` 1.30.0 in `uv.lock`** for CVE-2026-59951 (was 1.28.1), which failed
  the `pip-audit` job and, through it, `main`'s release gate.

## 0.7.10 — 2026-10-16

### Added

- **Separate verifier sandboxes run on Docker and Daytona.** A task that sets
  `[verifier] sandbox_mode = "separate"` (task.toml `environment_mode`) or
  declares `[verifier.sandbox]` was refused with "separate verifier
  sandboxes are parsed but not executed". It now runs: after the agent stops,
  the workspace is frozen into `evidence/`, and a fresh sandbox built from
  the verifier image (`[verifier.sandbox].docker_image`, `[sandbox]
  .docker_image`, `tests/Dockerfile`, then the task's own
  `environment/Dockerfile`) receives only those hash-checked bytes before the
  normal hardening and `test.sh` run there. Nothing else from the agent's
  sandbox reaches the verifier. `bench eval run --separate-verifier` (SDK and
  config `separate_verifier`) forces the mode for every task of a run without
  editing them, so task digests do not change. A transfer or sandbox failure
  is a verifier infrastructure error, never a score;
  `verifier-sandbox/verifier-sandbox.json` records the image source, sandbox
  id, transfer inventory and timing, and `config.json` records
  `verifier_sandbox_mode`. Ported from release/0.8's separate verifier,
  without its declared-artifact and `/logs/artifacts` transfer, which main
  does not have.

### Fixed

- **Root helpers no longer import modules from the agent's workspace.** The
  helpers BenchFlow runs as root after the agent stops (disk reclaim, the
  verifier PATH check, pytest-plugin discovery, the external-symlink purge,
  the workspace-restore fallback, workspace evidence capture and its
  reviewer-side admission) ran plain `python3 -c` from the sandbox's default
  working directory, which is the agent's workspace, so a `json.py` or
  `glob.py` left there was imported as root before the verifier. They now run
  `python3 -I`; the `.pth`/`sitecustomize.py` purge keeps its environment
  but runs from `/`. Pytest-plugin discovery therefore scans only the
  interpreter's site-packages: declare a plugin installed elsewhere in
  `verifier.pytest_plugins`.

## 0.7.9 — 2026-10-14

### Fixed

- **Codex knows `gpt-6.1-sol`: the `codex-acp` pin moves from 1.13.1 to
  2.0.1.** Codex takes a model's tool surface from the model catalog bundled
  with its `@openai/codex` dependency. codex 0.156.1 (behind 1.13.1) has no
  `gpt-6.1-sol`, so a `gpt-6.1-sol` rollout logged "Model metadata for
  `gpt-6.1-sol` not found. Defaulting to fallback metadata" and was offered
  only the fallback function tools: no code mode, no `apply_patch`, no
  multi-agent tools (the #1145 symptom again), and `session/set_model` to a
  `gpt-6.1-sol[...]` id failed. codex-acp 2.0.1 depends on `@openai/codex`
  ^0.159.1 (npm resolves 0.159.2 today; both catalogs list `gpt-6.1-sol` with
  efforts low to max and ultra, code mode only, multi-agent v2, a 272k
  context). On a live docker rollout through the LiteLLM proxy (Azure),
  `gpt-6.1-sol` at effort `max` now gets code mode (`exec`, `wait`),
  `clock.sleep` and the collaboration tools (`spawn_agent`, `wait_agent`,
  `send_message`, `followup_task`, `interrupt_agent`, `list_agents`), and
  `session/set_model` accepts `gpt-6.1-sol[max]` for direct providers.
- **Codex command output stays in the trajectory under codex-acp 2.x.**
  codex-acp 2.0 (its tool call contract) stopped putting a finished
  command's output in `rawOutput.formatted_output` for a client that declares
  no terminal capability, which is BenchFlow: a shell command now ends with no
  `rawOutput` and a read / search / list command with only `{"exit_code": n}`,
  while the output arrives as `_meta.terminal_output_delta` chunks and the
  exit code in `_meta.terminal_exit`. `ToolCallRecord.raw_output` rebuilds the
  1.x `{"formatted_output", "exit_code"}` shape from those chunks (an
  agent-sent `formatted_output` still wins), so captured trajectories keep
  every command's output. The trajectory viewer now shows it for shell
  commands too: it rendered Codex's `terminal` content block (a terminal id)
  instead of the command's raw output, on codex-acp 1.x as well.
- **Known regression: the `gpt-5.4` family falls back under codex-acp 2.0.1.**
  codex 0.158 dropped `gpt-5.4` from its catalog, and no codex release lists
  both `gpt-5.4` and `gpt-6.1-sol`. On 2.0.1, `gpt-5.4`, `gpt-5.4-mini` and
  `gpt-5.4-nano` log "Model metadata ... not found" and lose `apply_patch` and
  `tool_search`. This affects the codex-acp integration model
  (`.github/integration/scope_defaults.yml`, `gpt-5.4-nano`) and the
  `openai/gpt-5.4-mini` examples; a model the catalog knows (for example
  `gpt-5.5` or `gpt-5.6-luna`) keeps the full surface.
- **Codex gets its real model id, and a codex that knows it (#1145).** Under
  a BenchFlow provider (LiteLLM proxy) the `codex-acp` thread was started with
  the proxy alias as the model (`benchflow-azure-foundry-openai-gpt-5.6-luna`).
  Codex resolves model metadata by slug, warned "Model metadata for `...` not
  found. Defaulting to fallback metadata", and offered a reduced tool surface
  on every rollout: eight function tools, no native `apply_patch`, no code
  mode, no multi-agent tools, where native `codex exec` on the same provider
  offers all of them. `CODEX_CONFIG.model` now names the bare slug (which the
  proxy already serves next to the alias), the launch-config writer accepts
  it when applying the reasoning effort, and the `codex-acp` pin moves to
  1.13.1 (codex 0.156.1) because 0.148 has no metadata for `gpt-6-astra`
  even under the bare slug. Verified on hello-world rollouts: bare slug +
  0.156.1 gives `gpt-6-astra` code mode + `apply_patch` + `spawn_agent` /
  `wait_agent` / `send_message` / `followup_task`, and `gpt-5.6-luna` code
  mode + `apply_patch`, matching the native CLI's tool lists for both.

### Added

- **`bench skills validate`: static skill checks with NVIDIA SkillEvaluator 0.5.0 Tier 1 and NVIDIA SkillSpector 2.12.0.** Takes skill directories or any directory to search for `SKILL.md`, runs both external CLIs with `--no-llm`, and prints a per-skill table (Tier 1 status, quality grade and score, SkillSpector risk and verdict) or `--json`. It exits 1 on a schema, security or dependency finding (including known vulnerabilities pip-audit finds in pinned packages) at or above `--fail-on` (default `high`), when a checker fails to run, or when a given path holds no `SKILL.md`; `--no-security` skips SkillSpector. The tools are not on PyPI, so they stay external installs (`uv tool install` from their release tags, see `docs/skill-eval.md`); a missing tool gets an install hint and an older one a version warning.
- **Native Google Antigravity CLI agent (`antigravity`, alias `agy`).** The
  Antigravity CLI replaced the hosted Gemini CLI in mid-2026 but ships no ACP
  mode, so BenchFlow bundles an ACP shim over its headless `stream-json`
  protocol. The registry entry pins the native `agy` release (SHA-512
  verified, no Node.js), runs in Gemini API-key mode, streams tool calls with
  raw input/output and per-turn token usage, honors `--model` and
  `--reasoning-effort` over ACP (agy requires an effort; the default is
  `high`), discovers skills from `~/.gemini/config/skills` and
  `<workspace>/.agents/skills` (a `SKILL.md` read is recorded as the skill
  invocation), delivers task MCP servers through
  `~/.gemini/config/mcp_config.json`, routes API-key runs through the LiteLLM
  Gemini pass-through like the `gemini` agent, and enforces the no-web and
  denylist policies with `PreToolUse` deny hooks.

## 0.7.8 — 2026-09-14

### Added

- **Automatic rubric review and scoring.** Tasks with a weighted `rubric.json`
  run a separate reviewer after verification. Passing requires all required
  tests and rubric blockers to pass; the final reward is the weighted quality
  score. Solver evidence and reviewer runs are retained, and `bench eval score`
  can retry an interrupted review without rerunning the solver. (#1126)
- **Rubric results in the trajectory viewer.** The Rubric tab displays criterion
  judgments, explanations, blocker outcomes, and weighted scores. (#1102)
- **Viewer themes and syntax highlighting.** Trajectory pages support a dark
  theme and highlighted tool content. (#1098)

### Fixed

- **Automatic rubric review works on Daytona when the task workspace is
  `/root`.** Daytona's daemon keeps its session tree in `/root/.daytona` (the
  entrypoint's FIFOs plus every session command's script, log and exit code),
  and evidence capture aborted on the first FIFO, so the rollout ended in a
  scoring error with `rewards: null` although its verifier had run. Capture now
  leaves that tree out as `sandbox_runtime` and records any other socket, FIFO
  or device node as a `special_file` exclusion instead of aborting. Capture
  limits, escaping symlinks and concurrent changes still fail closed. (#1128)
- **Automatic review supports shell-only task images.** Required Python
  capture tools are provisioned before the solver starts, so a successful
  shell task does not lose its review to a missing interpreter. (#1127)
- **Denylist enforcement closes CONNECT and HTTP Host bypasses.** TLS identity
  and HTTP authority are checked against the allowed destination on Docker and
  Daytona. (#1122)
- **IPv6 egress rules are installed when procfs reports zero-size files.**
  Sandbox firewall setup reads the kernel data instead of relying on the
  reported file size. (#1124)
- **Buffered ACP prompts respect their deadlines.** Timeout handling no longer
  repeats captured provider history. (#1125)
- **Tool captures retain their content and raw input/output.** ACP trajectories
  preserve observations needed for later review. (#1100)

## 0.7.7 — 2026-09-09

### Added
- **`network_mode: denylist` blocks a list of URLs and hosts for the agent on
  Docker and Daytona.** The task keeps internet access; `blocked_urls` and
  `blocked_hosts` are enforced by a root-owned loopback proxy behind the
  sandbox-user firewall, hosted search tools are switched off per harness,
  and every refused request lands in `trajectory/egress_denylist.jsonl`.
  Other backends refuse the mode at preflight. (#1113)

### Fixed

- **Denylist egress no longer cuts the agent off from its own model.** The
  controller registers the running LiteLLM gateway's exact
  `127.0.0.1:<port>` endpoint with the proxy, so clients that ignore
  `NO_PROXY` (Gemini's Undici `ProxyAgent` tunnels even plain HTTP) still
  reach it; the exception comes from the live gateway, never task metadata or
  agent-supplied environment, and every other private destination stays
  blocked. Reconnects re-register the current port. (#1118)
- **The denylist is a shared sandbox policy, not a per-harness one.** Every
  supported ACP harness — custom registrations included — gets the same proxy,
  certificates, and UID firewall on primary connections and later roles alike,
  independent of harness name, model id, or provider. (#1118)
- **`codex-acp` sessions are configured through native ACP settings.** Hosted
  search is switched off via `CODEX_CONFIG.web_search` (the CLI ignores `-c`
  overrides), and Codex runs in `agent-full-access` session mode when
  BenchFlow has already selected a non-root sandbox user, so its bubblewrap
  sandbox is not nested inside Docker or Daytona — BenchFlow's own user,
  filesystem restrictions, proxy, and firewall still apply. (#1118)
- **`cryptography>=44` is a core dependency.** Denylist egress mints TLS
  certificates on both Docker and Daytona, so the pin moved out of the
  `sandbox-agentcore` extra. (#1118)

## 0.7.6 — 2026-09-04

### Added
- **ACP rollout directories render as an interactive reviewer page.**
  `bench eval view <rollout>` now assembles a JSON payload (normalized
  events + result/timing/verifier metadata) and renders it client-side in a
  self-contained template (no build step, zero network requests): full
  event stream with collapse instead of truncation, harness/model/skills
  identity row, reward badge and `result.json` failure-diagnostic banners,
  Verifier and Metrics tabs, Focus/Full modes, per-kind filters and hues,
  text search, and per-event `#e42` anchors. Trajectory content is treated
  as untrusted end to end (data-only embedding, `textContent` rendering).
  Raw session JSONL files and legacy `turn*.txt` runs remain supported, and
  the `--confirm` approve/reject contract remains compatible.
- **Directories of rollouts serve a run catalog.** Pointing
  `bench eval view` at a job directory (or a whole `jobs/` tree) serves a
  browsable index: corpus counts, grouping by task or model + harness with
  per-group pass/fail aggregates and pass rates, sorting by
  name/reward/duration/cost, text filtering, collapsible groups with
  incremental pagination, and URL-preserved state — selecting a run opens
  the detail page and the back control restores the exact catalog view.
  Traces load dynamically from `/api/rollout?id=…` (ids resolve only by
  exact membership in a fresh directory scan, so crafted ids cannot reach
  the filesystem) and `?run=<id>` deep-links a run. `--confirm` on a
  multi-run directory errors out: a confirmation needs exactly one
  trajectory.
- **`hf://` sources browse HuggingFace trajectory datasets directly.**
  `bench eval view hf://<org>/<dataset>[@revision][/subpath]` fetches the
  viewer-relevant slice of a trajectory dataset into the shared
  `huggingface_hub` cache and serves it through browse mode, making the
  community ground-truth uploads reviewable with one command. The download
  allowlist is exact — trajectories, result/timing/prompts, and the four
  verifier sidecars the viewer renders — with no wildcards, so large run
  artifacts and `llm_trajectory`/`trainer` exports are never fetched. The
  CLI passes the spec as a string into a typed
  `LocalPathSource | HfDatasetSource` parser (never through `pathlib.Path`,
  whose normalization would mangle `hf://` into `hf:/`), and dataset
  subpaths are validated.
- **The viewer renders per-event timelines the moment captures provide
  timestamps.** Steps gain a `+m:ss` offset chip and tool calls a duration
  chip whenever events carry `ts` / `started_at` / `finished_at` fields;
  with today's captures (which carry none) nothing changes visually.
- **Z.ai Coding Plan routing.** Coding Plan subscriptions route through the
  provider layer, with validated generation parameters forwarded and explicit
  provider endpoints preserved by the LiteLLM proxy. (#1074)
- **`bench eval run` publishes job artifacts and eval results.**
  `--publish-bucket` syncs a job directory to a HuggingFace storage bucket (an
  alternative to `--publish-hf`'s dataset-repo upload), and
  `--eval-results-model/-dataset/-task` opens a community eval-results PR
  (`.eval_results/*.yaml`) on a model's HF repo, scored from the run's mean
  reward. (#1035)

### Fixed

- **`claude-fable-5-1` can run: the `claude-agent-acp` pin moves 0.40.0 to
  0.73.0.** The old pin bundled `@anthropic-ai/claude-agent-sdk` 0.3.160, and
  the model rejects Claude Code older than 2.1.251 with
  `claude_code_version_too_old` (HTTP 400) — so every rollout failed on its
  first API call. The new pin bundles sdk 0.3.257, and the `set_config_option`
  `"model"` / `"effort"` wiring is re-verified against it by
  `tests/test_acp_pinned_protocol_guard.py`. (#1086)
- **`codex-acp` dispatch modernized.** The shim pin moves to 1.6.0 and
  reasoning effort travels inside the model id (`modelId[effort]`) through
  `session/set_model`, rather than a config option the shim rejects. (#1044)
- **ACP usage and terminal evidence survive an agent timeout**, instead of
  being dropped with the timed-out turn. (#1080)
- **A pending tool call can no longer defer the idle watchdog without
  bound** — the deferral is capped. (#1066)
- **`bench eval` resume re-runs infra-retryable verifier-errored tasks**
  rather than treating the infrastructure failure as a settled result. (#1063)
- **Sandbox hardening execs draw on the verifier-setup budget**, so hardening
  no longer competes with the agent's own time. (#1062)
- **`--trials > 1` without `--matrix` is rejected up front.** (#1064)
- **Healthy native ACP subscription results are preserved.** (#1049)
- **The agent judge parses native edit targets.** (#1069)
- **Non-dict rewards no longer break completed-outcome classification.** (#1054)
- **Codex honors proxy-owned model selection.** (#1076)
- **Gemini routing corrections.** ACP model ids fixed and wrapped ids
  normalized, Gemma routing supported with pass-through usage captured, every
  Google key alias routed through the proxy, Google Gemini gateway routes
  normalized, and headless runs trust the sandbox workspace.
- **LiteLLM provisions the Vertex sandbox runtime.** (#985)
- **OpenClaw caps supported model output tokens.**
- **RestrictedPython updated to 8.3.**

## 0.7.5 — 2026-08-19

### Added

- **Weighted rubric-review contract (v0.2).** `bench review` now accepts the
  versionless `rubric.json` shape, where
  every criterion adds strict `blocker` (`0` or `1`) and `weight` (`1` through
  `10`) fields. Binary blockers gate publication; non-blockers receive weighted
  `0` / `1` / `2` scores with raw and gated quality plus publication bands in
  the report. Blocker weights are excluded from quality, and the wrapper reward
  remains a structural-validity signal. Existing three-field v0.1 rubrics keep
  their `pass` / `fail` / `not_applicable` behavior unchanged. Docker runs now
  probe whether verifier-log bind mounts are genuinely visible to the container
  and fall back to explicit copy-out when path translation is unavailable.

### Fixed

- **Linked-worktree `.git` pointer files stay out of workspace attachments.**
  Workspace capture now excludes `.git` files as well as directories, preventing
  local absolute worktree metadata from entering uploaded archives. (#1032)

## 0.7.4 — 2026-08-16

### Added
- **Uploads are confirmed all the way into cloud storage.** After the
  progress bar finishes, `bench traj upload` now polls the contribution
  service's new `GET /v1/uploads/{digest}` capture-status endpoint (the
  validation ledger) until the validator's verdict: `✓ Verified in cloud
  storage` once the capture is promoted to `sources/community/<digest>/`, a
  concise exit-1 error with the fixable detail if the validator rejects it,
  and a `bench traj status sha256:<digest>` handoff line if validation is
  still running when the budget (default 240 s, `BENCHFLOW_TRAJ_WAIT_SECONDS`
  override, `--no-wait` opt-out) runs out. A handshake 409 ("already
  submitted") prints the verified line immediately, and a deployed broker
  that predates the endpoint (404) keeps today's behavior unchanged. The new
  `bench traj status DIGEST` command runs one check on demand. Status polls
  consume a separate, higher rate-limit budget (`TRAJ_STATUS_RATE_LIMIT`,
  default 720/hour/IP) and reveal only the ledger state, the bounded
  rejection detail, and the public promotion prefix — never contributor
  identity or quarantine internals. The broker must be redeployed
  (`deploy-trajectory-upload` workflow or `scripts/deploy.sh`) before the
  endpoint answers in production; the CLI degrades gracefully until then.
- **The `bench traj` family shares one polished terminal design language.**
  A new presentation-only kit (`cli/_traj_tui.py`) gives `traj setup`,
  `traj upload`, and `traj status` a coherent look: a `◆ benchflow · <command>`
  banner, styled `◇` input prompts, an arrow-key recent-session picker (↑/↓,
  1-9 jump, esc to fall back to typing a path) on real terminals, colored
  step kinds in the report preview matching the browser viewer's palette,
  and rounded panels. Every interactive affordance degrades to the exact
  previous prompt-driven flow off-TTY (agents, pipes, CI, Windows), and all
  machine-read lines (`Masked for you:`, `Digest:`, `Repo:`) stay plain.
- **Uploads can carry the session's workspace folder as a zip attachment.**
  `bench traj upload` reads the session's recorded working directory (the
  same Claude `cwd` / Codex `session_meta` provenance as repo tagging) and
  archives it into the capture as `workspace/<name>.zip`, printing
  `Workspace: <path> (from session cwd; use --no-workspace to omit)` and a
  `Workspace attached:` line with size, file count, and exclusion count.
  VCS internals, dependency trees, caches, symlinks, and secret-shaped
  filenames (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `.netrc`, …) never enter
  the archive; everything else is archived as-is without content redaction,
  and the attach line says so. Workspaces over 1 GiB (measured before
  compression, so the zip is never created), over 50,000 files, missing, or
  empty are skipped with a printed reason instead of failing the upload.
  When detection fails on a real terminal, one optional prompt accepts a
  folder or skips on Enter; `--workspace-dir` overrides detection and
  `--no-workspace` opts out. The archive is staged in the upload's
  temporary directory and always deleted afterwards. Server side, the
  contribution service accepts the new `workspace/*.zip` namespace with a
  1 GiB per-archive cap (trajectory JSONL keeps 128 MiB), allows at most
  one archive per capture and never an archive alone, verifies the zip
  container format instead of JSONL strictness, promotes it with an
  `application/zip` content type, and scopes trajectory-report
  cross-checks to trajectory artifacts so an attachment cannot fail
  report equality.

## 0.7.3 — 2026-08-16

### Added
- **`bench traj upload` waits out short rate-limit responses instead of
  failing.** When the contribution service answers 429 with a short
  `Retry-After`, the handshake now sleeps it out with jitter and retries up
  to three times (two-minute cap per wait), so a crowd of simultaneous
  contributors self-heals instead of surfacing errors. Longer waits still
  fail fast with the actionable retry-after message. (#1027)

### Changed
- **The contribution service rate-limits per contributor, not per venue.**
  Upload budgets are token buckets keyed on contributor identity with a
  wide per-IP abuse backstop, refill continuously, and answer 429 with
  seconds-until-next-token instead of the remainder of the clock hour, so
  many contributors behind one NAT no longer starve each other. Contended
  bucket updates back off with jitter rather than shedding simultaneous
  crowds. (#1027, #1028)

## 0.7.2 — 2026-08-16

### Added
- **The upload preview itemizes what redaction masked, by kind.** The
  redactor now categorizes every replacement by the rule that fired — API
  keys, bearer tokens, private key blocks, passwords, URL credentials, and
  credential-bearing field values — and the terminal trajectory report shows
  a `Masked for you: 2 API keys, 1 bearer token — originals never leave this
  machine` breakdown under the masked-count row, plus a reassurance that
  redaction ran locally and the server independently rescans staged
  artifacts (or `No secrets or personal identifiers detected — nothing
  needed masking.` when nothing matched). `bench traj upload --dry-run`
  prints the same breakdown as a plain `Masked for you:` line, and
  `bench eval view --confirm` gains a display-only `--redaction-summary`
  flag that renders it in the confirm bar next to the Approve button; the
  `benchflow-traj-upload` skill stages a dry run first and passes the line
  through. The total `redaction_replacements` count and the manifest
  `trajectory_report` contract are unchanged (the server validates the
  report with a closed schema and exact recompute equality, so per-category
  counts stay display-only).
- **The trajectory viewer can collect the eval-prize confirmation in the
  browser.** `bench eval view PATH --confirm` renders the normal page plus a
  sticky site-styled bottom bar ("Submit this trajectory to the BenchFlow
  eval prize?") with **Approve & submit** / **Not this one** buttons. A click
  POSTs to `/decision`; the server prints a machine-readable
  `DECISION: approved` or `DECISION: rejected` line to stdout, shuts down,
  and the CLI exits `0` on approve and `3` on reject (non-1/2 so rejection
  never collides with error or usage exits). Without the flag, behavior is
  unchanged (no bar, no endpoint, Ctrl+C to stop). The
  `benchflow-traj-upload` skill now prefers the button flow and falls back
  to chat confirmation on CLIs older than 0.7.2.
- **Trajectory uploads are tagged with the session's repository by
  default.** Unless `--source-id` is given, `bench traj upload` reads the
  session's recorded working directory (Claude `cwd` events, Codex
  `session_meta`), resolves its git `origin` remote, and stores
  `repo/<owner>/<name>` as the manifest source id, printing
  `Repo: owner/name (from session cwd /path; use --no-repo to omit)` (the
  local path is terminal output only, never uploaded). `--no-repo` opts out
  — the `benchflow-traj-upload` skill now surfaces the detected tag during
  the confirm step so contributors can decline it for private repos — and
  undetectable repos fall back silently to the path-derived source id.

### Fixed
- **The repo tag derives only from the session's own recorded cwd.** The
  initial repo-tagging implementation (#1015) fell back to the upload
  invocation directory's git remote when the session cwd yielded nothing; a
  collector-side audit showed this mis-attributes provenance — a session
  recorded in a non-repo directory, uploaded from the benchflow checkout,
  was tagged `repo/benchflow-ai/benchflow` (two community-dataset entries
  carry the mis-tag). The fallback is removed: no session cwd, a missing
  directory, or no GitHub remote now mean no repo tag, exactly like
  `--no-repo`.
- **The trajectory viewer header no longer shows `?` badges on real Claude
  Code sessions.** `bench eval view` on a `~/.claude` session JSONL rendered
  `model: ?`, `session: ?...`, `claude code: ?`, and `total cost: $0.0000`
  because the header only read a `type: system` event that real session
  files don't contain. The header now derives its metadata from what the
  file actually carries (first assistant event's `message.model`, per-event
  `version` / `sessionId`, the filename stem as a session fallback) and
  hides any badge whose value is unknown — including the cost badge when no
  event carries cost data. Presentation only.
- **Upload progress no longer claims the broker is waking up.** A warm
  retry printed `Uploading… the first request can take a minute while the
  service wakes up` even when the service was already up. The line is now
  `Uploading… this can take up to a minute; retries are safe.`
- **`bench traj setup --list` no longer wraps session paths mid-token.**
  Each hit prints index/source/time, then the path on its own line, then
  the snippet, via plain `print` so Rich does not split a long JSONL path
  and break copy-paste.

### Changed
- **Trajectory viewer tool calls are color-coded and backgrounds are light.**
  Each tool kind now gets a muted GitHub-label-style accent on its name pill
  and a left border strip on the card: shell/exec → amber, write/edit → blue,
  read → teal, agent/task/skill → purple, web/search/fetch → cyan, everything
  else → neutral gray. Tool arguments and tool outputs render on light
  surfaces (`#f5f5f5` / white with dark ink) instead of near-black blocks;
  the dark `#141414` terminal treatment is reserved for shell-command output
  only, and the ink-black result card stays as the deliberate bento-ink
  accent. Presentation only (CSS classes + a tool-name→accent mapping);
  content strings and behavior are unchanged. Applies to all three viewer
  templates, which share one stylesheet since #1019.
- **The contributor prompt now tells the agent to upgrade BenchFlow first.**
  `CONTRIBUTOR_PROMPT` (kept in sync in `README.md`, `docs/traj-upload.md`,
  and the `benchflow-traj-upload` skill evals) is a three-line block that
  says to run `uv tool install --python 3.12 --upgrade --force benchflow`
  before reading the skill, so agents that skim the skill or hit a stale copy
  still install the latest CLI. The prompt also names OpenCode and Cursor
  sessions and the re:Agent hackathon 72-hour window; README/docs render it
  as a blockquote (soft-wraps on GitHub) behind an explicit "send this to
  your coding agent" framing, and `bench traj setup` prints the same framing.
  The skill's Discover step gains best-effort Cursor
  (`~/.cursor/projects/*/agent-transcripts/`) and OpenCode
  (`~/.local/share/opencode/opencode.db`, `opencode db path`) locations plus
  a prefer-recent-matching-sessions note. Follow-up to the version
  precondition from #1013/#1014.
- **Trajectory viewer restyled to match www.benchflow.ai.** All three
  `bench eval view` pages (stream-json/JSONL, ACP events, multi-turn trial)
  now share one inline stylesheet with the site's design language: light
  monochrome palette (`#fafafa` page, white cards, `#0a0a0a` ink, dark
  `#141414` code blocks), Satoshi/Google Sans Code font stacks with
  system-safe fallbacks, mono pill badges, and a small BenchFlow wordmark
  header with the inline SVG logo. Pages remain fully offline (no external
  font or CDN requests) and content/structure semantics are unchanged;
  follows the contributor paste-line flow from #1013.

## 0.7.1 — 2026-08-16

### Added
- **`bench traj setup` / `bench traj upload` print an upgrade hint when
  outdated.** Both commands start with a lightweight PyPI latest-version
  check (2 s timeout, completely silent on any network or parse failure) and
  print a one-line `uv tool install --python 3.12 --upgrade --force
  benchflow` hint when the installed version is older than the latest
  release; dev/prereleases of a newer-or-equal base are not outdated.
  `BENCHFLOW_SKIP_UPDATE_CHECK=1` disables the check. The
  `benchflow-traj-upload` skill and docs now tell contributors to upgrade to
  the latest BenchFlow (0.7.1+) before using the trajectory-upload flow.

### Changed
- **Trajectory contribution is a copy-paste line, plus optional setup.**
  Contributors paste one line into their agent; the agent reads
  `benchflow-traj-upload`, opens the viewer, then uploads. Optional setup is
  `npx skills add benchflow-ai/benchflow --skill benchflow-traj-upload` or
  `bench traj setup`. `bench eval view` accepts a raw session JSONL file and
  does not write `trajectory.html` next to it. The CLI infers GitHub
  username and email from `gh` / `git` before prompting, prints `Submitted` /
  `Already submitted` plus a digest for public uploads (not a private Azure
  inbox URL), waits up to 90s for broker cold start, and treats Azure
  `403 UnauthorizedBlobOverwrite` as an idempotent skip. (#1008's interactive
  report, local secret masking to `<XXX-benchflow-key-values-XXX>`, schema-1.2
  manifest report binding, and byte progress are included; the PR #1008
  operator manual now lives at `benchflow-traj-upload-ops` so the public skill
  name stays contributor-facing.)

## 0.6.9 — 2026-08-15

### Changed
- **Trajectory uploads require contributor provenance.** The single
  `bench traj upload` command now requires `--github-id` and `--email`; both
  values are validated locally and server-side and stored in `manifest.json`
  under the structured `contributor` field. Existing 0.6.8 manifests remain
  readable by the validator during the client transition.

## 0.6.8 — 2026-08-15

### Added
- **Public trajectory contribution.** `bench traj upload PATH` validates and
  structurally redacts trajectory JSONL, creates a content-addressed manifest,
  and uploads through the built-in public broker. Azure Blob quarantine,
  versioning, short-lived create-only user-delegation SAS grants, and an
  event-driven fail-closed validator keep untrusted captures out of the
  community namespace until hashes, sizes, strict JSONL (including duplicate-key
  and non-finite-number rejection), and artifact/manifest secret scans pass.
  Replaying an ingested digest is a no-op; trusted operators can opt into direct
  Azure upload with the `azure` extra and `--direct`.

### Changed
- **BREAKING (task.md): the `environment:` frontmatter key is renamed to
  `sandbox:`.** The native task-config surface now accepts only `sandbox:`
  (plus `verifier.sandbox:` for the verifier's separate sandbox spec and
  `verifier.sandbox_mode:` for shared/separate selection); `environment:`,
  `verifier.environment:`, and `verifier.environment_mode:` no longer
  validate and fail with an actionable message naming the rename. **The
  one-line fix for existing task.md files is renaming the key.**
  Legacy `task.toml` imports are unaffected: the toml loader
  converts `[environment]`, `[verifier.environment]`, and
  `environment_mode` to the `sandbox` spellings (declaring both spellings
  in one file is an error), and `bench tasks export` emits the inverse —
  an `[environment]`-spelled `task.toml`. All native emitters
  — `model_dump_toml`, `bench tasks migrate`, task scaffolding,
  skill-eval/trace/adapter task generation, rubric-review wrappers — now
  write `sandbox`. Python API: the compat property
  `TaskConfig.environment` is removed (use `TaskConfig.sandbox`),
  `VerifierConfig.environment`/`environment_mode` became
  `sandbox`/`sandbox_mode`, and the `VerifierEnvironmentMode` enum is now
  `VerifierSandboxMode`. The Environment plane
  (`--environment-manifest`, `benchflow.environment.manifest`, the
  eval-config `environment:` docker/daytona selector) is a different
  subsystem and is unchanged.

### Fixed
- **Coherent integration coverage for code-and-fixture changes.** The
  credential-free fixture job now tests pull-request source together with its
  task fixtures, while the secret-bearing smoke job retains trusted fixtures;
  the release gate requires both results. (#969)
- **Retryable Daytona transport failures stay retryable.** Transient SDK
  transport errors are stamped while their vendor type is still available,
  empty exception messages retain useful type information, and permanent
  errors remain outside the retry policy. (#970)
- **Live token counters no longer freeze behind callback-log reads.** Larger,
  bounded ranged reads distinguish EOF from read failure, expose lag, and keep
  terminal phase labels from moving backwards. (#971)
- **Fresh OpenClaw installs and GPT-5.4 calls use compatible limits.** OpenClaw
  is pinned to a Node-compatible runtime, and raw plus ACP-alias GPT-5.4 model
  IDs clamp output tokens to the provider's 128,000-token limit. (#977, #986)
- **Anthropic Vertex routes include their required runtime.** The gateway now
  installs the Google Cloud Vertex AI SDK used by LiteLLM's Anthropic Vertex
  path. (#978)
- **ACP subprocess diagnostics survive normal teardown.** A bounded, redacted
  stderr tail is retained even when stdout remains open; stderr drain failures
  cannot mask the structured transport diagnosis, and repeated process close
  calls are idempotent. (#980)

## 0.6.7 — 2026-08-09

### Added
- **Built-in environment registry.** The committed env-axis pins
  (`env0@prod`, `env0@outage`) moved from `benchmarks/_environments/` into
  the package (`benchflow/environment/_registry/`) and ship inside the
  wheel, so `--environment-manifest env0@prod` resolves on a bare
  `pip install benchflow` with no checkout and no env vars.
  `$BENCHFLOW_ENV_REGISTRY`, when set, still wins entirely; resolution
  stays content-addressed (sha256 logged), and unknown names now error
  listing the available specs. (#961)
- **Console progress heartbeat.** Single-concurrency eval runs print a
  throttled progress line (`… 6.2min, 12 tool calls (last: …)`) about every
  45 seconds while the agent works, so a long prompt is distinguishable from
  a hang. The heartbeat is auto-gated off for multi-concurrency jobs;
  `bench eval run --quiet` suppresses it, and `BENCHFLOW_PROGRESS=on`/`off`
  overrides the auto-gate. (#951)
- **Live per-task activity in the eval dashboard.** Under a TTY the
  running-now table gains an activity column ("38 calls · last:
  file_editor", plus tokens once a usage snapshot exists), polled from the
  ACP session's existing heartbeat counters. The agents-manifest autoload
  also clones quietly — one "Cloning …" summary line instead of raw git
  progress. (#956)
- **Phase labels and per-task failure reasons.** The activity cell is never
  blank while a row exists: sandbox create, agent install, and verify show
  dim phase labels ("creating sandbox…", "installing agent…",
  "verifying…"). The final block prints one dim "✗ task: reason" line per
  failed task — verifier error first, else a compact reward/metric
  breakdown, else the reward — capped at 5 lines. `--quiet` now silences
  the dashboard as well as the heartbeat. (#957)
- **Failure reasons mined from verifier artifacts.** When a displayed
  failure would read as a bare "reward X", the CLI reads a small bounded
  artifact from the rollout's own verifier dir — the CTRF report (first
  failed test plus its assertion line) or a tail of `test-stdout.txt` — and
  a dim "(details: …/verifier)" pointer names the artifact directory.
  Artifacts resolve by the recorded rollout name, never by glob. (#959)
- **Fractional rewards in console summaries.** Per-task lines carry the
  scored reward (`✗ task (reward=0.30, tools=47)`), the Score line renders
  ", mean reward 0.30" alongside the binarized counts, and
  `EvaluationResult` / `summary.json` gain a `mean_reward` field (mean over
  scored rollouts; errors excluded, not zeroed). (#960)
- **Full failure counts on per-task console lines.** A CTRF report with
  more than one failed test rolls up as " (+N more failure(s); P/T checks
  passed)" after the first failure — the count suffix is never truncated
  away — and parametrized test names keep their `[param]` ids whenever the
  report carries them. (#962)
- **Live token usage in the dashboard footer.** The footer sums completed
  tasks' trusted telemetry plus every running rollout's live ACP session
  usage, so spend is visible while the run executes (usage lands per
  completed prompt; cost stays scoring-time from the gateway log).
  Single-tool agents' activity cell drops the redundant "last:" suffix in
  favor of "38 calls · 412.0k tok" once tokens are available. (#963)
- **env0-shaped failure breakdowns.** The failure-reason tiers understand
  metrics nested one level under `metrics` / `details`, pair
  `<name>_found` with `<name>_total` into fractions ("deadlines 1/5",
  lowest-signal first), probe `verifier/reward.json` on disk between the
  CTRF and stdout tiers, and every failure block with on-disk verifier
  artifacts gets one "(details: …)" pointer. (#964)
- **Mid-prompt live token usage via the gateway's live capture.** The #963
  live tokens stepped forward only when an ACP prompt completed, so a
  single-prompt rollout showed "— tokens" for its whole agent phase. The
  proxy runtime's existing live-capture loop (which already tails the
  sandbox gateway's callback log every second) now also accumulates
  provider tokens into an O(1) counter, and the dashboard reconciles the
  two non-decreasing live signals as max(ACP, gateway) — display-only,
  still replaced by the trusted scoring import at completion, and any
  gateway-side failure degrades to the ACP-only behavior. (#965)

- **Compact flow-style arrays in emitted task.md frontmatter.** Short
  scalar-only lists render on one line (`tags: [parsing, nlp]`) instead of
  multiline bullets, so hand-written flow arrays survive `bench tasks
  migrate` / normalize round-trips. The style predicate measures each list
  with the real YAML emitter (so it can never disagree with PyYAML's own
  quoting) and falls back to block style for long, nested, or multiline
  items. Parsing is unchanged — both styles were always accepted. (#967)

### Changed
- Migrated the clawsbench `archive-amazon-shipping` task to the native
  `task.md` package layout (`bench tasks migrate --remove-legacy`), and a
  clawsbench run launched without `--environment-manifest` now fails with an
  actionable verifier message naming the exact flag to pass instead of an
  opaque connection error. (#952)

### Fixed
- Isolated pull-request and post-test integration concurrency groups so a
  credential-free workflow completion cannot cancel an active PR rollout.
- **Host-side hard rollout deadline.** An await wedged below the phase-level
  watchdogs (e.g. a Daytona PTY teardown on a dead websocket) could freeze an
  entire eval job. Every rollout attempt now runs under a host-side hard
  deadline derived from the task's own phase budgets plus a fixed margin; a
  trip returns a retryable infra-error result and abandons the sandbox after
  a bounded cleanup grace. Override with `BENCHFLOW_ROLLOUT_HARD_DEADLINE`
  (seconds; `off`/`none`/`0` disables). Verifier timeouts that produced zero
  output are retried once — that signature is an exec-layer wedge, not a slow
  verifier. Agent-sent reasoning parameters are also forwarded through the
  LLM gateway, and DeepSeek routes natively with `reasoning_effort` passed
  verbatim. (#949)
- Quieter Daytona teardown: successful runs end at the score line — the
  atexit client cleanup no longer prints a cancelled-reader traceback after
  `✓ Score`, and benign engineio PTY-disconnect errors are silenced. (#951)
- Accept the task.toml schema 1.3 `[task] version` field in task configs. It is
  informational and stored verbatim; the strict schema previously rejected
  it, which made every schema-1.3 curated task unloadable. (#953)
- Synced the committed `env0@prod` / `env0@outage` environment pins with the
  upstream `benchflow-ai/env0` manifest (the pins had drifted to stale
  service CLI names, a phantom service, and shifted ports), and a manifest
  environment now fails loud when it declares services but none are startable
  in the image, instead of passing a vacuous readiness gate and dying at the
  verifier. (#954)
- Pointed the built-in `env0@prod` / `env0@outage` pins at the org-owned
  `ghcr.io/benchflow-ai/env0:0.2.0` base image. The wheel-shipped pins named a
  personal Docker Hub image while their own comments and the environment docs
  declared the ghcr image authoritative; `base_image` is recorded as rollout
  provenance, so every env0 result carried the wrong base. Verified on Daytona
  (env0 `auth-least-privilege-summary`, 8/8 services ready, reward 1.0).
- Restored the `env0@outage` perturbation (gmail and slack removed relative
  to `env0@prod`) that the #954 upstream sync had erased by mirroring the
  full service list into both pins, and corrected the pin header's stale
  slack port. (#955)
- **ACP protocol JSON glued to PTY shell noise now decodes.** On PTY
  transports, an agent's initialize response could arrive on the same line
  as the shell prompt (ANSI/OSC-prefixed), fail the strict whole-line JSON
  parse, and get filed as noise — the handshake then "timed out" at any
  window with the answer sitting in the agent log. The PTY-facing transport
  now retries from the first `{` and once more after an ANSI scrub, but
  only until the first successfully decoded protocol message per
  connection; afterwards the strict contract rules, so log-echoed envelopes
  cannot impersonate protocol traffic mid-session. The pre-prompt handshake
  window is also env-configurable via `BENCHFLOW_ACP_HANDSHAKE_TIMEOUT`
  (seconds; default 60). (#958)

## 0.6.6 — 2026-08-04

### Added
- **Apple Container and Amazon Bedrock AgentCore sandboxes.** Apple Silicon
  users can run supported single-container arm64 tasks through Apple's
  Virtualization.framework, while `--sandbox agentcore` builds task-specific
  AWS runtimes with lease-aware cleanup for supported public-network,
  single-container arm64 tasks. (#936, #937)
- **Rubric review (`bench review`).** Detached agentic grading of finished
  rollouts against a `rubric.json` — rubric contract **v0.1**: an object with
  a `criteria` list, each entry carrying a `name` (structured-output field), a
  `description` (author documentation, never shown to the reviewer), and
  `guidance` (the grading contract). The document carries no in-file
  version key. One
  reviewer agent per rollout runs as an ordinary rollout of a throwaway
  wrapper task on a digest-pinned multi-architecture base image and no
  task-authored Dockerfile
  (AgentCore still builds a derived runtime image), reads a read-only evidence
  copy of the rollout and, when admitted from a trusted `--tasks-root` with a
  verified digest, its task, and answers every criterion with `pass` / `fail` /
  `not_applicable` plus an explanation. The wrapper's own reward means only
  "the reviewer produced a structurally valid result"; graded outcomes land
  in `review_report.json`. Reviews never modify a reviewed rollout's rewards
  or `result.json`. Rubric resolution: `-r` > the task's own
  `verifier/rubric.json` > a built-in default (`reward_hacking`,
  `task_specification`). `--passing` / `--failing` filter the rollouts under
  review; a job-level prose summary aggregates multi-rollout runs. The
  default reviewer harness is `opencode` (pinned to `1.18.11`).
  Evidence mounts at `/evidence` outside the agent workdir (root-owned,
  unwritable by the reviewer), symlinks are dropped rather than
  dereferenced, task skills, shipped rubrics, and cumulative provider-history
  trajectories are excluded while the canonical ACP trajectory is retained;
  dropped ACP tool observations and generic tool titles are repaired from
  exact-ID events in the trusted provider capture,
  reviewer egress is gateway-scoped via the sandbox lockdown flag, artifact
  consumption is pinned to each invocation's unique runtime leaf, and the
  job summary is a deterministic aggregation.
- **Sealed AgentCore uploads.** Every AgentCore **upload and staged
  environment** is now encrypted end-to-end: the sandbox generates a
  keypair, only the public key appears in command output, payloads travel
  as AES-256-CTR ciphertext with an HMAC-SHA256 tag over IV and
  ciphertext (verified before decryption), and the decrypted key never
  appears in command text. Fixes provider credentials from
  `launch_config.json` and command environments being recoverable from
  the runtime's CloudWatch command log; the generated wrapper image
  installs `openssl` when missing. Downloads are not sealed: they return
  file contents as base64 through command output, so they must only carry
  non-secret run artifacts.
- **`RolloutConfig.uploads`.** Generic post-start host→sandbox uploads
  (directory or file → absolute sandbox path), used by rubric review to
  deliver evidence into prebuilt-image sandboxes and available to any caller
  whose task data is not baked into the image.
- **Native TRL tool-calling SFT export.** `bench train convert --format
  trl-sft` emits conversational prompt/completion rows with a `tools` column,
  excludes OpenCode title/summary helper calls, and accepts rollout trees,
  canonical `results.jsonl`, or existing TRL JSONL. `bench train validate
  --format trl-sft` can render rows with a pinned tokenizer and fail closed on
  missing assistant masks or overlength samples. Tokenizer-aware
  `--context-policy message-window` keeps the harness/task prefix and the
  longest complete recent assistant/tool suffix when exact rows exceed the
  student context window. (#925)
- **LLM call-purpose provenance.** Captured LLM exchanges retain provider/model
  metadata and classify agent, title, summary, compaction, and helper calls;
  `results.jsonl` carries the metadata on each trajectory step. (#925)
- **Live trajectory streaming and token logprobs.** Redacted LLM trajectory
  snapshots are written during active rollouts, and training runs can opt into
  sampled-token logprobs with `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`. (#922, #926)

### Changed
- OpenHands supports requested reasoning effort through its LiteLLM request
  body and now fails closed if completed provider exchanges are missing from
  trainer trajectories. The built-in OpenCode harness is version-pinned for
  reproducibility. (#921, #931)

### Fixed
- Restored the documented `test` → live `integration-light` → internal-preview
  publication chain for successful `main` pushes, pinned to the exact tested
  commit and fail-closed against untrusted workflow sources.
- Fixed OpenHands startup outside root-owned workdirs, OpenCode gateway setup
  in non-Python task images, Qwen3.5 TRL SFT tokenization, and incomplete agent
  skill-path validation. (#919, #924, #929, #932)
- Moved recursive review evidence work and blocking platform probes off the
  async event loop so concurrent reviews and rollouts retain responsive
  scheduling and timeouts.
- Remove staged Docker and Apple Container credential files when live process
  launch fails before the agent can source and unlink them.
- Corrected release-facing authentication, sandbox, trajectory-artifact,
  branching, and provider documentation to match the shipped interfaces.

## 0.6.5 — 2026-07-10

### Added
- **Reproducible evaluation artifact workflows.** `bench eval run` can emit
  task/run manifests, trajectory-health summaries, canonical one-rollout-per-task
  selections, materialized trainer inputs, repeated model matrices, and
  Hugging Face dataset uploads with source provenance. (#844, #907)
- **Prime-RL supervised fine-tuning integration.** `bench train validate` and
  `bench train run sft --backend prime-rl` add fail-closed trainer-row checks,
  local/Hugging Face dataset staging, publishing hooks, and compatibility
  controls for reproducing the Mobile300 PR828 training recipe. (#842–#865)
- **Native external benchmark adapters.** BenchFlow can materialize MCP Atlas
  and Toolathlon sources, credential-backed task packages, and the service
  sidecars required by their hosted runtimes. (#878, #885, #889)
- **Registry-driven agent extension.** Agent packages can autoload through
  `benchflow.agents` entry points or declarative manifests, including namespace
  shorthand for externally registered agents. (#873, #877)
- **BenchFlow-native GRPO/TRL pipeline.** Adds reusable `TaskRuntime`, the
  optional `BenchFlowSpec` TRL adapter, selective Hugging Face task snapshots,
  paired `bench eval compare-lift` reporting, and an end-to-end GRPO runbook.
  (#901–#907)

### Changed
- BenchFlow CLI and SDK installations now explicitly require Python 3.12 or
  newer, with `uv tool install --python 3.12` as the recommended CLI path.
  (#899)

### Fixed
- Hardened LiteLLM and provider routing across Responses/chat bridges, Gemini
  custom endpoints, Claude OAuth, Harvey LAB, OpenHands Azure reasoning effort,
  diagnostics, and parallel ACP shim startup. (#868, #871, #879–#881, #886,
  #888, #911)
- Stabilized Daytona execution for Toolathlon and ACP agents, including DinD
  retries, PTY handling, orphan-free long-command heartbeats, and Gemini's SSH
  transport fallback. (#890, #892–#896, #910, #912)
- Tightened trajectory and Prime-SFT artifact integrity with secret redaction,
  tool-call validation, repaired results conversion, and reproducible
  compatibility controls. (#849–#865)

### Removed
- **`BENCHFLOW_SKILL_NUDGE` skill prompt nudge.** The optional prompt injection
  that prepended mounted-skill names/descriptions/bodies to the task
  instruction (#207) is gone. Setting the environment variable now has no
  effect: prompt resolution never reads skill directories, and mounted skills
  reach agents only through their native skill paths. This keeps prompts
  identical across skill modes and closes off accidental prompt-level skill
  leakage. (#908)

## 0.6.4 — 2026-06-27

### Added
- **Environment and config as run-time axes on `bench eval run`.** `--state`
  binds the environment (S-axis) per run — inline JSON, a registry
  `name@version` resolved through the environment registry, or a manifest path
  (takes precedence over `--environment-manifest`). `--config-override` overlays
  the task config (C-axis) — inline JSON/YAML/TOML or `@file`, deep-merged into
  each task's resolved config. `--config` also gains a `--run-config` alias.
  (#790)
- **Content-addressed environment binding.** Registry environment resolution is
  content-addressed — `env_hash = sha256(manifest)` — so a `name@version`
  resolves to an exact, pinned environment that is recorded for replay; the
  C-axis `--config-override` is likewise persisted with its content hash and the
  applied patch. Every rollout is attributable to the precise world and config
  it ran against. (#790)
- **MLE-bench adapter.** Adds an MLE-bench benchmark adapter, parity fixture, and
  task plumbing for running and auditing MLE-bench through BenchFlow. (#792)
- **Agent adapter skill.** Adds the canonical adapter skill under `.agents/skills`
  for harness-side adapter work. (#793)
- **Prime-RL SFT export.** Adds `bench train convert prime-sft` support for
  exporting BenchFlow trajectories into Prime SFT-ready JSONL artifacts. (#828)

### Changed
- **`bench continue` is now `bench eval continue`.** The command (and its
  `continue-batch` companion) moved under the `eval` group, where it is now
  discoverable in `bench eval --help` alongside `run`/`adopt`. The original
  top-level `bench continue` / `bench continue-batch` remain as hidden,
  deprecated aliases (they print a deprecation notice) so existing scripts keep
  working. (#800)
- **Routable agents always go through the LiteLLM usage proxy.** OpenCode-family
  and pi-acp model calls now stay on the proxy path so token usage, cost, and
  trajectory capture are preserved consistently. (#797, #803, #820)
- **Agent manifest loading is now the additive decoupling path.** The core agent
  manifest loader and Omnigent/session-factory seam are gated in while preserving
  existing ACP manifests and byte-identical parity coverage. (#825, #836, #837)

### Fixed
- Resolved the sharded and run-config paths so the S-axis environment and C-axis
  config overlay are applied consistently in `bench eval run`. (#804)
- Added `bench eval run --context-root` plumbing and early validation for missing
  paths. (#816)
- Fixed verifier-error resume logging and streaming `claude-agent-acp`
  trajectory emission so failed or streamed runs retain the expected evidence.
  (#819, #839)
- Resolved bare model IDs to their provider, avoided pi-acp context-window retry
  storms, and kept provider failure causes visible while preserving redaction.
  (#805, #831, #834, #835)
- Preserved Codex subscription-auth behavior and auth-file permissions in the
  launcher path. (#825)
- Rejected `.git` and `file://` source paths with clear errors. (#822)
- Hardened experiment-review and integration gates around missing trajectories,
  summaryless roots, file-editor false positives, and L3 review calibration.
  (#802, #806, #807, #808, #809, #810, #811, #812, #814, #817, #821, #823, #824)

## 0.6.3 — 2026-06-16

### Changed
- **`bench eval create` renamed to `bench eval run`.** The verb now matches what
  the command does (it runs an evaluation, single task or batch). `bench eval
  create` stays as a deprecated alias that prints a deprecation notice on use, so
  existing scripts, YAML configs, and downstream repos (e.g.
  `benchflow-ai/skillsbench`) keep working unchanged. Switch to `bench eval run`.
- **`task.md` is now the sole task authoring format.** `bench tasks init`
  scaffolds a native `task.md` package (`task.md` + `environment/` + `oracle/` +
  `verifier/`). `bench tasks init --format legacy` is retired and now exits with
  an error pointing at `bench tasks migrate <dir> --remove-legacy`. Existing
  split-layout packages remain readable, and `bench tasks migrate` /
  `bench tasks export` continue to cover the migration and compatibility paths.
  Authoring docs now lead with `task.md`; the split layout is documented only as
  a migration/export target.

### Fixed
- `bench skills eval` now exits non-zero when any eval case errors (e.g. missing
  credentials), matching `bench eval run`. A 100%-error run printed `0/1`
  but exited `0`, so CI/scripts read a total failure as success.
- The "task.md already exists" migrate error now names both surfaces
  (`--overwrite` for the CLI, `overwrite=True` for the Python API) instead of
  only the API kwarg.
- `bench eval view <job-dir>` no longer shows a blank "No trajectory files
  found" when given a job directory (the natural value from `eval run`'s
  "Artifacts:" line) — it now indexes the rollout subdirectories to drill into.
- `bench hub env list` prints a footer (`Showing N…`) with how to refine
  (`--search`/`--owner`/`--limit`/`--json`), so a small page of a large catalog
  no longer reads as "the provider only has N environments".
- `bench tasks check --level publication-grade` errors for a missing verifier
  package now include a remediation hint (author `verifier/verifier.md`; note
  that `bench tasks migrate` does not generate it), instead of a dead-end.

## 0.6.0 — 2026-06-13

### Added

- **The `task.md` task standard** — a single-file unified task format (parser,
  verifier planes, prompt sidecars, round-trip export with a machine-readable
  loss report) plus the authoring CLI: `bench tasks init / check / migrate /
  export`, with a layered `check --level` ladder up to a leaderboard-grade
  acceptance gate. See [`docs/task-standard.md`](docs/task-standard.md) and the
  [native authoring guide](docs/task-authoring-task-md.md).
- **`bench eval adopt` benchmark-adoption router** — `init` scaffolds a benchmark
  conversion per [`benchmarks/CONVERT.md`](benchmarks/CONVERT.md), `convert` drives
  the host `codex` CLI through the conversion workflow, and `verify` runs the
  parity gate (deterministic per-criterion conversion parity plus the
  agent-scale reward-distribution layer) and emits a confidence verdict, with a
  drafted support issue on divergence. `bench eval adopt verify --rerun`
  independently re-executes the benchmark's `parity_test.py` and scores its fresh
  output (instead of trusting the recorded `parity_experiment.json`), failing
  closed if the output is not scoreable; `bench eval adopt convert -c key=value`
  passes codex config overrides through to the host codex driver (e.g. to work
  around `~/.codex` drift). `bench tasks digest` recognizes native `task.md` tasks
  as well as legacy `task.toml`.
- **ATIF and ADP trajectory artifacts** — every scored rollout now emits
  `trainer/atif.json` and `trainer/adp.jsonl` (alongside the existing
  `verifiers.jsonl`), with job-level ADP aggregation. One canonical raw
  trajectory, multiple ecosystem formats out of the box.
- **OpenReward (ORS) reward-format interop** — export BenchFlow rewards in the
  Open Reward Standard shape (`benchflow.adapters.ors`) and the `ors-episode`
  verifier strategy is recognized. (The hosted-environment episode runner that
  executes ORS environments end-to-end is in progress, not in this release.)
- **Daytona sandbox auto-reap** — orphaned sandboxes are cleaned at eval start
  (TTL-tiered; failure states reaped sooner; an idle-activity guard protects
  live runs), gated by `BENCHFLOW_DAYTONA_AUTO_REAP` (any of `0`/`false`/`no`/
  `off`, case-insensitive, disables it).
- **Registry-pinned dataset runs** — `bench eval create -d name@version`
  (e.g. `-d skillsbench@1.1`) resolves a dataset from a git-backed
  `registry.json` (see skillsbench `docs/dataset-versioning.md`): tasks are
  cloned at their pinned `git_commit_id` into `.cache/datasets` and every
  task directory is verified against its sha256 content digest before
  anything runs; the entry's `bench_version` range is checked against the
  installed benchflow. `--registry` overrides the default (skillsbench)
  registry. `result.json`/`config.json` are stamped with `dataset_name`,
  `dataset_version`, and a per-task `task_digest` (`summary.json` carries
  the name/version); `--tasks-dir` dev runs carry no dataset fields but
  still stamp a live-computed `task_digest`, so every trajectory stays
  attributable to exact task content. `bench tasks digest <dir>` prints
  the digest for task authoring, and `check_results.py` audits the stamps.
  See [`docs/running-benchmarks.md`](docs/running-benchmarks.md). (#689,
  #690, #691; `packaging` promoted to a core dependency for the
  `bench_version` check.)
- **`benchflow continue <run-folder>`** — resume a previous, unfinished
  (timed-out) `openhands` run to completion. A standalone tool (it does not
  touch the normal run path) that reconstructs the run's exact workspace and
  agent memory from the recorded `llm_trajectory.jsonl` via record-replay,
  then continues with the live model — no injected prompt — and writes a new
  HF-compatible folder with `continued_from` provenance. See
  [`docs/continue-runs.md`](docs/continue-runs.md).

### Changed

- `bench metrics` → `bench eval metrics` and `bench view` → `bench eval view`
  (the deprecated hidden top-level forms are gone; use the `eval` subgroup).
- Quickstart and CLI reference now match observed run behavior — the real jobs
  directory layout and artifact map, the `<PROVIDER>_API_KEY` /
  `<PROVIDER>_BASE_URL` convention, and exit-code semantics.
- Document the public vs internal preview install/upgrade command matrix,
  including `uv tool` exact pins, internal preview upgrades, and the
  `--force` path for replacing stale entrypoint scripts.

### Renamed (aliased; old names removed in 0.7)
- Benchmark adoption is now `bench eval adopt {init,convert,verify}`. It lives
  under `eval` because `eval` is the universal benchmark entry point (`eval
  create` runs a benchmark; `eval adopt` makes a foreign one runnable). Two prior
  spellings remain as hidden deprecated aliases, each printing a one-line stderr
  notice pointing at `bench eval adopt`: the original `bench agent
  create|run|verify`, and the 0.6-dev intermediate top-level `bench adopt`.
  `bench agent` now means agent management only (`list` / `show`).
- The overloaded `bench environment` group was split and is now a hidden
  **deprecated alias group** (removed in 0.7): the local sandbox lifecycle moved
  to `bench sandbox {create,list,cleanup}`, and hosted-provider browsing to
  `bench hub env {list,show,inspect}`. The old `bench environment
  create|list|cleanup|show|inspect` (plus `list --provider`/`--hub`) still work,
  each printing a one-line stderr deprecation notice. The hosted *run* path stays
  on `bench eval create --source-env`.

### Removed
- **Removed the unwired `OTelCollector`** (`benchflow.OTelCollector` /
  `benchflow.trajectories.OTelCollector`) and its `trajectories/otel.py` module.
  It was a designed-but-never-wired OTLP receiver from the v2 rewrite — never
  instantiated, never tested, and not part of any run path (BenchFlow captures
  trajectories via ACP session events and the LiteLLM callback path instead).
  This drops it from the public `__all__`; re-add it (with a test + real wiring)
  if OpenTelemetry-based capture is revived.
- Removed two unimplemented stub methods (`read_file`, `write_file`) from the
  `@runtime_checkable` `Sandbox` Protocol. No backend implemented them (backends
  expose the `upload_file`/`download_file` family) and there were no call sites,
  so they were a latent `isinstance` trap on the contract surface.
- Dead-code purge, round 3 (no public-API impact; each symbol re-verified
  zero-reference with class context): removed `TaskMetrics.audit_outcome`,
  `OTelCollector.endpoint`, `ReplayRouter.cursor`, `RuntimeResult.to_run_result`
  (legacy SDK-compat converter, unused), the never-read dataclass fields
  `ToolCall.output` and `JudgeConfig.{reference, prompt_template}`, the write-only
  `ReplayProxy._host`, the inert `AgentProtocolError.code` annotation, and an
  unused `retry_if_exception_type` import + fallback in `sandbox/daytona.py`.
- Dead-code purge, round 2 (no public-API impact; each symbol adversarially
  verified zero-reference with class context): removed seven unused `*_path`
  `@property`s from `TaskPaths`/`RolloutPaths` (`readme_path`, `gitignore_path`,
  `verifier_document_path`, `artifacts_manifest_path`, `result_path`,
  `exception_message_path`, `log_path`), the vestigial `ModalSandbox.supports_gpus`
  / `can_disable_internet` capability properties (not on the Sandbox Protocol),
  an unused module-level `logger` in `cli/continue_cmd.py`, and the orphaned
  `mcp_service_hooks_from_config` helper.
- Dead-code purge (no public-API impact unless noted): removed the unused
  `job_config_from_yaml` helper, the nominal `TASK_REPOS` back-compat dict
  (use `TASK_ALIASES`), the `_looks_like_verifier_dep_install_error` shim
  (use `contains_verifier_dep_install_marker`), the unused `parse_binary_verdict`
  reward helper (use `parse_verdict`), the dead `SandboxBackend` type alias,
  an unused `StdioTransport._read_buffer` field, and 12 redundant `rollout`
  package re-export aliases (submodule definitions unchanged).
- Removed the deprecated, hidden `benchflow skills install` CLI command. The
  SDK function `benchflow.skills.install_skill` is unchanged.
- Retired the deprecated top-level legacy CLI (`cli/legacy.py`). The dead
  0.3-era `job`/`agents`/`eval` commands are removed; `metrics` and `view` are
  promoted to first-class `bench eval metrics` / `bench eval view`; and the
  redundant `cleanup` command is dropped in favor of the existing
  `bench environment cleanup`.
- Removed the `experiments/` research/dev tooling tree (never shipped in the
  wheel) and its 6 dependent test modules, completing the dev-tree cleanup
  alongside the earlier `dashboard/` removal and `labs/` → `docs/labs`
  migration. Benchmark result files were preserved out-of-tree, not deleted.

### Fixed
- **CLI errors now go to stderr.** `print_error` (the single CLI error sink) wrote
  to stdout, so a `bench … --json | jq` pipeline could get a non-JSON error line on
  the JSON channel. All CLI errors (and the dataset bench-version remediation hint)
  now route to stderr; exit codes are unchanged, so failures stay detectable.
- **`bench hub env list --json` now emits valid JSON at any width.** The raw
  payload was printed through Rich's console, which soft-wrapped long strings and
  injected literal newlines mid-value (unparseable JSON when piped). It is now
  written verbatim.
- **No more raw tracebacks on bad input.** Hardened the unguarded front doors a
  stress sweep surfaced: `eval create --source-repo` clone failures and
  `--tasks-dir <file>`; `eval view` on corrupt/partial trajectory artifacts
  (`prompts.json`, a bad `acp_trajectory.jsonl` line, `result.json`, a null
  `session_id`); `sandbox create` with an unknown `--sandbox` backend or a missing
  optional sandbox dependency; `tasks digest` on an unreadable file (single = clean
  error, batch = warn-and-skip); and `hub check` with a malformed/missing
  `--registry` (now a user-meaningful message, not a raw `JSONDecodeError`/`OSError`).
- **Markup-safe output.** User/author-controlled strings that look like Rich markup
  no longer crash or silently garble output: `eval list` job names, `eval metrics`
  title, `skills list` cells, and `tasks init`'s reported path are now escaped.
- **`skills eval` schema errors** no longer leak pydantic internals (private model
  name, `[type=…]` tags, the pydantic.dev URL) — just the actionable per-field text.
- **`bench environment` deprecation notice** now fires exactly once (one line,
  once per process) instead of doubling up with Typer's generic
  `DeprecationWarning`, and its aliased verbs are hidden from `--help`, matching the
  `agent` / `eval adopt` alias families.
- `benchmarks/CONVERT.md` now references the canonical `bench eval adopt verify`
  (was the deprecated `bench agent verify`) in the conversion prompt.
- `bench tasks migrate` emits minimal, canonical (`schema_version`) front
  matter instead of a full defaults dump.
- Verifier `timeout_sec` is validated as a positive, finite budget
  (fail-closed at parse time; omission inherits the documented default).
- Docker `compose up` retries on the daemon network create/attach race.
- Console error messages truncate at word boundaries instead of mid-token.
- Recorded sandbox-setup timeouts and trajectory artifacts are consistent
  across the Docker and Daytona backends.
- The `task.md` init scaffold is agent-neutral, so `--agent oracle` works on a
  freshly scaffolded task.
- `gemini/`-prefixed judge/simulated-user models now resolve to the Google
  backend instead of passing the slashed name through and 404-ing.
- Model-backed judges raise a clear error naming the provider and pointing at
  `pip install benchflow[judge]` when the judge SDK is missing, instead of the
  misleading "Missing OPENAI_API_KEY" fall-through.
- `bench tasks check` recognizes a rubric-backed `llm-judge` verifier as a valid
  entrypoint and no longer demands a `test.sh`.
- Pre-verifier disk reclaim is workspace-aware and symlink-safe: it rejects
  symlinked cache candidates and realpath-guards every deletion against the
  workspace and `/logs`, so an agent-planted `~/.cache` symlink cannot steer the
  reclaim into workspace or output state (#601).
- Bedrock Claude 4.8+ routes fail closed when LiteLLM's adaptive-thinking patch
  is inactive, instead of silently sending a request the proxy cannot satisfy
  (#602).

## 0.5.2 — 2026-06-05

### Changed

- **PyPI project README badge** — replace the dynamic PyPI version badge with
  a stable package badge so the rendered project description cannot show a
  stale external version image after a public release.
- **Release documentation refresh** — update public install snippets,
  release-channel docs, examples, and citation metadata to `0.5.2`.

## 0.5.1 — 2026-06-05

### Added

- **Daytona usage telemetry by default** — Daytona runs now start a sandbox-local provider usage proxy so token/cost telemetry works without an external tunnel; use `--usage-tracking off` to bypass proxying when needed.
- **Azure AI Foundry providers** — new `azure-foundry-openai/` and `azure-foundry-anthropic/` prefixes routing through Foundry's unified resource. Export `AZURE_API_KEY` plus `AZURE_API_ENDPOINT` (e.g. `https://<resource>.openai.azure.com/`); benchflow derives the resource name from the endpoint host, builds the per-surface base URL, and maps the key onto the agent-native auth env automatically. Missing/unrecognized endpoints and unsupported agent/provider protocol pairings fail fast with clear errors instead of falling through to the wrong endpoint.
- **Azure Foundry auth guidance** — agent discovery output and docs now call out that provider-prefixed models can use provider-specific credentials instead of the agent's native/default API key.

### Changed

- **PyPI project documentation refresh** — the public package README, install snippets, release-channel docs, examples, and citation metadata now point at `0.5.1`.

### Fixed

- Inherit `BENCHFLOW_PROVIDER_BASE_URL` / `BENCHFLOW_PROVIDER_API_KEY` from the host environment so self-hosted / OpenAI-compatible endpoints route correctly instead of falling back to `api.openai.com`; empty or whitespace-only host values are skipped so they cannot shadow the resolved provider URL (benchflow-ai/skillsbench#817).

## 0.5.0 — 2026-06-04

### Added

- **Public/internal preview release channels** — tag-driven public releases publish stable PyPI packages and GitHub Releases; merges to `main` publish internal preview `.devN` packages after CI passes.
- **v0.5 integration evidence** — release validation docs now cover urgent blocker closure, SkillsBench infra-fix validation, adapter evidence, trace-to-task evidence, hosted env compatibility, and diagnostic fields.
- **Release automation guardrails** — public release tags must point at commits contained in `main`, version tags must match `pyproject.toml`, and PyPI publishing uses Trusted Publishing/OIDC instead of stored tokens.

### Changed

- `main` now tracks the next public version as `0.5.1.dev0`; the published public SDK is `0.5.0`, and internal previews are emitted as `0.5.1.dev<N>`.
- Documentation now directs downstream users to depend on public PyPI releases by default and use prerelease-enabled internal previews only for validation before the next public cut.

### Fixed

- Closed the v0.5 release blocker set covering structured sandbox/verifier diagnostics, Daytona startup/export retries, verifier dependency classification, CTRF path consistency, and SkillsBench task compatibility evidence.

## 0.3.3 — 2026-05-15

### Added

- **Harvey LAB benchmark** — converter, agent shim, and parity validation for 1,251 legal AI tasks (#239).
- **Harvey LAB Claude Sonnet judge** — switched verifier from Gemini to `claude-sonnet-4-6`, matching the original benchmark default (#264).
- **ProgramBench integration** — new benchmark adapter; TB2 removed; `.ref/` migrated to `benchmarks/` (#237).
- **CLI progress output** — `bench eval create` / `bench run` now show progress messages by default (#264).
- **Skill nudge** — optional prompt injection for skill-enhanced agent runs (#207).
- **Self-generated skill mode** for Codex agent (#233).
- **Integration test suite**, and the `OPENAI_BASE_URL` inheritance fix (#255).
- **Modal backend support** — Dockerfile compatibility for Modal environments.
- **CITATION.cff** (#246).
- **`AGENTS.md`** — canonical contributor guide; `CLAUDE.md` deprecated (#258).

### Changed

- **Two-field source pattern** for dataset sourcing (#252).
- **Docs overhaul** — synced from www.benchflow.ai; Mintlify config added then orphaned config removed (#259, #257, #226).
- **`uv sync`** for package management (#232).

### Fixed

- Prevent `TypeError` in `metrics.collect_metrics` when reward is `None` (#243).
- Copy eval `requirements.txt` into Docker build context (#245).
- Resolve agent aliases in `bench agent show` and display aliases in `bench agent list` (#251).
- Guard ACP transports against JSON scalar logs (#236).
- Agent timeout reward fallback for Codex (#234).
- Isolate JS agent runtime installs (#231).
- Route Codex ACP through responses API (#224).
- Deploy skills and forward `solution.env` for oracle runs (#223).
- Honor no-internet tasks for agent runs; disable web tools without prompt mutation (#215).
- Propagate `OPENAI_API_KEY` for vllm provider (#3).
- Preserve arrival order of thought/message within flush windows (#214).
- Record user messages and per-turn agent text in ACP trajectory (#745).
- Chown skill-link parent dirs so sandbox user can write into them.
- Dynamic `--rootdir` in `PYTEST_ADDOPTS` based on task workspace.
- Unique env-file path in `DaytonaPtyProcess` to avoid race conditions (#200).

## 0.2.3 — 2026-04-15

### Added

- `benchmarks/tb2_multiturn-claude-haiku45.yaml` — shipped config for the README's TB2 multi-turn Claude result.
- Daytona resource clamping via `BENCHFLOW_DAYTONA_MAX_CPUS` / `MAX_MEMORY_MB`.

### Changed

- Renamed `skillsbench-claude-glm5.yaml` → `skillsbench-claude-glm51.yaml` to match the model ID.
- `codex --login` correction in `docs/getting-started.md`.
- Restricted sdist build to `src/`, `tests/`, and metadata.

### Fixed

- Verifier sandbox hardening follow-ups across several base-image and tooling edge cases.
- Preserve trusted verifier path entries and workspace answer files.
- Redirect oracle output to container log.
- Align YAML path resolution to config file location.

## 0.2.2 — 2026-04-13

### Added

- **Sandbox hardening tiers 1–3** — layered defense (env scrubbing, path lockdown, workspace
  freeze, wider snapshot, oracle privilege drop) blocking F1–F6 red-team findings.
- **`labs/reward-hack-matrix`** — per-trial timeout support and 0.2.2 sweep handoff scripts.

### Fixed

- Multiple sandbox bypass vectors identified in red-team testing.

## 0.2.1 — 2026-04-12

### Added

- **Sandbox hardening on by default** — `sandbox_user` now defaults to `"agent"` (was `None`/root). Blocks conftest-hook and answer-lookup exploit patterns.
- **Path lockdown** — new `sandbox_locked_paths` parameter makes `/solution` and `/tests` read-only before the verifier runs, blocking `.pth`-injection and similar pre-verify tampering.
- **Verifier failure isolation** — agent errors and verifier errors are now stored separately; a crashing verifier no longer masks the agent result.
- **`labs/benchjack-sandbox-hardening`** — cookbook demonstrating three exploit patterns (P1 conftest-hook, P2 answer-lookup, P7 `.pth`-injection) and their defenses.

### Fixed

- **Oracle runs as `sandbox_user`** — oracle agent now respects path lockdown instead of running as root and bypassing it.
- **Multi-endpoint provider routing** — providers with multiple endpoints now route by the agent's native API protocol.
- **Stale API key shadowing subscription auth** — emits a warning when `ANTHROPIC_API_KEY` env var is present alongside `claude login` credentials.
- **pytest `ini`-injection bypass** — closed a verifier hardening edge case.

### Changed

- Version is now single-sourced via `importlib.metadata`; no more duplicate version string in `__init__.py`.
- **User-facing docs** — new `docs/` directory with getting-started guide, CLI reference, architecture overview, task-authoring guide, and labs index. README trimmed; detailed content moved to `docs/`.

## 0.2.0 — 2026-04-09

**First public release.** A near-complete rearchitecture from the 0.1.x era. API surface has changed — assume breaking changes. Future releases will maintain compatibility within the 0.2.x line. 0.1.x users should treat this as a fresh install; see `.dev-docs/sdk-reference.md` for the new SDK.

### Added

- **Multi-agent, multi-provider, multi-auth matrix** — one YAML config, any supported agent × model × provider × auth combination.
- **Subscription auth support** — use `claude login`, `codex --login`, `gemini` OAuth credentials directly. No API keys required for host-based agent workflows.
- **Vertex AI support** — ADC auth for `google-vertex/`, `anthropic-vertex/`, `vertex-zai/` prefixed models.
- **Provider registry** — add a new LLM endpoint via a dict entry in `providers.py`, no code changes.
- **`benchmarks/` directory** with reusable YAML configs and runner scripts for TB2 and SkillsBench.
- **Auto task download** — YAML configs reference datasets as `org/repo/path` (e.g. `benchflow-ai/skillsbench/tasks`). Repos are cloned on first use and cached under `.cache/datasets/`.
- **`benchflow tasks init`** — scaffold new tasks.
- **`benchflow tasks check`** — validate task structure.
- **`benchflow cleanup`** — delete old sandboxes with `--max-age` filtering (default 24h).
- **Oracle agent support** — run `solution/solve.sh` directly for task validation.
- **Hello-world-task example** for sanity-testing the agent pipeline.
- **Model generation params** via env vars (`BENCHFLOW_TEMPERATURE`, `BENCHFLOW_TOP_P`, `BENCHFLOW_MAX_TOKENS`).
- **OpenClaw ACP shim** with trajectory parsing and skills support.
- **ACP trajectory capture** — full multi-turn agent trajectories via ACP protocol.

### Changed

- **Skill loading** — agent-targeted with proper precedence; auto-distributed from `task.toml` `skills_dir`.
- **`openclaw-gemini` merged** into `openclaw` — provider mode selected at runtime via `BENCHFLOW_PROVIDER_NAME`.

### Fixed

- **API keys leaking in `ps aux`** — env vars now written inside the container instead of passed via Docker exec `-e`.
- **Subscription auth skipped without `-m`** — `benchflow run` without `--model` now checks correctly.
- **ADC credentials break with `sandbox_user`** (#111) — credentials written to sandbox user's home instead of `/root/`.
- **Daytona sandboxes not cleaned up** (#102) — auto-delete after max age.
- **`benchflow cleanup` ignoring `--max-age`** — was deleting everything regardless of age.
- **readline buffer overflow crashes trial** (#98).
- **OpenClaw ACP shim loses tool command text** (#96).
- **OpenClaw ACP shim hardcodes `anthropic/` prefix** (#95) — now routes correctly for Gemini/GLM models.
- **Oracle agent `PermissionError`** writing `agent/oracle.txt` (#91).
- **Oracle path skips `pre_agent_hooks`** (#92) — services now start before oracle runs.
- **Richer trial data** (#90) — richer `result.json`, agent logs, per-phase timing.
- **`SDK.run()` `PermissionError`** — `jobs_dir` subdirectories created as root (#88).
- **Partial trajectory lost on timeout** — saved before timeout raises.
- **Redundant `--version` binary check** removed — was wasting 30s per trial.
- **Trajectory fallback** — scrapes agent-native files when ACP `session/update` is empty (#94).
- **`litellm` upgraded to 1.83.0** for CVE-2026-35030; transitive dep security alerts resolved (13 Dependabot alerts closed).

### Deprecated

- `BaseAgent` re-export — planned removal in 0.3.0
- `Trial` re-export — planned removal in 0.3.0
