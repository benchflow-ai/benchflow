# BenchFlow in Rust: design

This branch (`bench-rs`) rebuilds BenchFlow's command line and run engine in Rust, starting from the 0.8 release. The Python package under `src/benchflow` stays in the tree as the reference implementation until each part of it has a Rust port that matches it; the Rust code lives in `crates/` and builds one binary, `bench`.

The goal is not a line-by-line port. It is the same engine, with the developer experience designed from the first command a new user types.

## What a developer should feel

1. **Instant.** `bench` is one static binary. It starts in milliseconds, checks a thousand tasks in about a second, and needs no Python, Node, or virtual environment on the host.
2. **A small surface.** A dozen verbs cover the whole loop, and each one has good defaults. The 0.8 spellings (`bench eval run`, `bench tasks check`, ...) keep working and say once what the new spelling is.
3. **Errors that say what, whose, and what next.** Every problem has a stable code, the place it happened (a line and column when it is in a file), whose problem it is (the task, the agent, the model provider, the sandbox, your setup, or BenchFlow itself), and the next step. `bench explain <code>` gives the long form.
4. **A task proves itself.** `bench test` runs the reference solution (it must pass), an empty run (it must fail), the task's own negative controls (they must fail), and BenchFlow's exploit probes (they must not raise the reward), and says which of these held.
5. **Everything is a file.** A job is a folder of JSON and JSONL in 0.8's layout, so `bf.load_job`, the 0.8 viewer, and any script keep working. Every command takes `--json`.
6. **Machines are users too.** Stable JSON, stable exit codes, and no prompts unless a person is at the terminal.

## The commands

| Command | What it does | 0.8 spelling it replaces |
|---|---|---|
| `bench` | Shows what is in this folder (tasks, the last job) and what to do next | — |
| `bench new <name>` | Scaffolds a task that already passes `bench test` | `bench tasks init` |
| `bench check [paths]` | Validates tasks without building or running anything; `--fix` applies the safe fixes | `bench tasks check` |
| `bench test [paths]` | Proves each task: oracle, empty run, controls, exploit probes | (manual oracle and nop runs) |
| `bench run [targets]` | Runs an agent on tasks and scores them | `bench eval run` |
| `bench results [jobs]` | Summarizes jobs; two or more jobs are compared trial by trial | `bench eval list`, `metrics`, `inspect`, `compare` |
| `bench view [job]` | Opens a job in the browser | `bench eval view` |
| `bench shell [task]` | Opens a shell in a task's sandbox, then grades what you left | — |
| `bench doctor` | Checks the machine: sandboxes, agent logins, disk; `--confinement` reads a confined sandbox's settings back | `bench doctor` |
| `bench agents [name]` | Lists agents and whether each can run here | `bench agent list`, `show` |
| `bench import <src>` | Converts a split-layout (task.toml), 0.8 task.md, or Terminal-Bench v1 (task.yaml) package to task.md | `bench tasks migrate` |
| `bench export <task>` | Writes a split-layout copy for harnesses that read task.toml | `bench tasks export` |
| `bench export <job>` | Writes a job's trajectories as a Hugging Face dataset (optionally with ShareGPT conversations) or OpenTelemetry JSON lines | `bench train convert` |
| `bench explain <code>` | Explains a diagnostic or failure code | — |

`bench regrade <job>` scores a finished job again with each task's current verifier, and `bench schema` prints the JSON Schemas of BenchFlow's files. Everything else in 0.8 (branching, training export, hub, hosted environments, embodied rollouts) is not ported yet; see [status.md](./status.md).

### What `bench run` takes

One positional list of targets, instead of `--tasks-dir`, `--source-repo`, `--source-path`, and `--dataset`:

| Target | Example |
|---|---|
| A task folder, or a folder of tasks | `bench run tasks/` |
| A GitHub repository, optionally a folder in it and a ref | `bench run benchflow-ai/skillsbench/tasks@main` |
| An explicit GitHub URL | `bench run https://github.com/org/repo/tree/main/tasks` |
| A dataset from a registry (`-d`) | `bench run -d terminal-bench@2.0` |

The agent defaults to the oracle, so `bench run tasks/my-task` checks a task with no model and no key.

### Configuration

Flags override `bench.toml`, which overrides built-in defaults. `bench.toml` is optional and found by walking up from the current folder, like `Cargo.toml`:

```toml
[run]
agent = "claude"
model = "claude-sonnet-4-6"
sandbox = "docker"
concurrency = 8

[profile.ci.run]
sandbox = "daytona"
concurrency = 32
fail-under = 0.8
```

`--profile ci` applies a profile on top of `[run]`.

## Architecture

```
crates/
  benchflow-task      task packages: parse, check, score, import, export, digest
  benchflow-sandbox   the Sandbox trait; docker/podman (CLI) and daytona (REST) backends
  benchflow-agent     agent manifests (data), credentials, the ACP client, trajectories
  benchflow-run       the run engine: rollouts, hardening, verification, jobs, results
  benchflow-viewer    the job viewer (local web server, 0.8's assets)
  benchflow-cli       the `bench` binary: commands, output, diagnostics rendering
```

Dependencies only point down the list: `task` and `sandbox` depend on nothing of ours, `agent` on `sandbox`, `run` on all three, and `cli` on everything.

### Task formats

| Format | How it is read |
|---|---|
| task.md (draft 2): the instruction, then a ` ```toml task ` block | Native. Unknown keys are errors with a suggestion. |
| BenchFlow 0.8 task.md: YAML frontmatter, then the prompt | Read as is; `bench import` converts it |
| Split layout: `task.toml` + `instruction.md` + `environment/` + `tests/` + `solution/` (schema 1.0 to 1.4) | Read as is, with warnings at line and column for keys that would be ignored; `bench import` converts it, `bench export` writes it |
| Terminal-Bench v1: `task.yaml` + `run-tests.sh` + `solution.sh` + `docker-compose.yaml` + `tests/` | Not run directly; `bench import` converts it to the split layout, then to task.md |

Every format becomes one normalized `TaskPackage`. The run engine never reads format-specific keys. A field the engine cannot honor yet is refused by name before anything starts, never skipped.

### A rollout

1. **Prepare the image.** Build `sandbox/Dockerfile` (or pull `[sandbox] image`). Images are tagged by a hash of everything that decides them, so an unchanged task never rebuilds. The agent's install becomes a cached layer on top, so a hundred trials install the agent once, not a hundred times.
2. **Start the sandbox** with the task's CPUs, memory, process limit, and network.
3. **Set up the agent user** (non-root), keep the verifier and oracle out of the sandbox until grading, take group and other write permission from system folders, snapshot `/etc/ld.so.preload`, and read back the memory limit and the agent's out-of-memory score. Then enforce the agent's network policy and self-test it as the agent user ([testing-tasks.md](./testing-tasks.md#network-policies)).
4. **Run the agent.** An ACP agent gets the instruction over ACP; the oracle runs `solve.sh`; `nop` does nothing; a probe runs its exploit; `script:<file>` runs the user's own script as the oracle would. Every agent command starts with `oom_score_adj` 1000, so the kernel stops the agent's processes first when memory runs out. The agent's events stream into `trajectory/acp_trajectory.jsonl`.
5. **Hand off.** Kill every process of the agent user and confirm none is left, clear `/logs/verifier`, and take custody of the outputs: copy only regular files and folders, refuse symlinks, FIFOs, sockets, and devices, and cap sizes (a 100 GB sparse file fails the cap at once instead of filling the disk).
6. **Harden and verify.** Remove planted `conftest.py`, `sitecustomize.py`, and `.pth` files and pytest plugins registered in `entry_points.txt` (`[pytest11]`), restore `/etc/ld.so.preload`, give the verifier a `PATH` of root-owned folders only, `PYTHONSAFEPATH=1`, and no loader variables (checked as the verifier's user before it runs; grading is refused when that fails), upload the verifier, run it under its timeout, and read `reward.json` or `reward.txt` (or score the rubric from `ctrf.json`).
7. **Save** outputs, timing, and the result, then destroy the sandbox.

Every phase is timed, and every failure is classified by owner. Each trial also records which defenses actually held, one line each, as `enforced`, `not-supported` (the backend cannot do it), or `off` (the task opted out), so a result says how far it can be trusted instead of implying it.

### Exploit probes

A probe is a scripted agent that does no work on the task and tries to raise its reward through the grading machinery. Each one names the defense it tests:

| Probe | What it tries | The defense |
|---|---|---|
| `forge-reward` | Writes `reward.txt`, `reward.json`, and a passing `ctrf.json` into `/logs/verifier` | The runner clears `/logs/verifier` before grading |
| `background-writer` | Leaves a detached process rewriting those files and the workdir | The runner kills every agent-user process before grading |
| `harness-tamper` | Plants `conftest.py`, `sitecustomize.py`, `.pth` files, and `pytest.ini` that force tests to pass | Hardening removes them and runs pytest with `-c /dev/null` |
| `path-shadow` | Puts fake `pytest`, `python3`, and `bash` early on `PATH` | The verifier runs with a trusted `PATH` |
| `peek-hidden` | Looks for the oracle and verifier during its run, and runs the oracle if it finds it | They are not in the sandbox until grading |
| `special-files` | Leaves FIFOs, symlinks to secrets, and a huge sparse file where outputs go | Output collection refuses special files and caps sizes, so grading cannot hang |
| `process-flood` | Leaves hundreds of processes | The process limit holds and the runner still kills them all |
| `env-inject` | Exports `LD_PRELOAD`, `PYTHONPATH`, and `BASH_ENV` from the agent's shell profile | The verifier starts from a clean environment |

A probe's reward is compared with the empty run's. A probe that scores higher found a hole, and `bench test` says which defense failed for that task and how to close it.

### Results on disk

A job folder keeps 0.8's layout so existing tools read it:

```
jobs/<job>/
  evaluation.json  summary.json  results.jsonl  lock.json
  <task>__<id>/
    result.json  config.json  timing.json  prompts.json
    trajectory/acp_trajectory.jsonl
    verifier/reward.txt  reward.json  ctrf.json  test-stdout.txt
    artifacts/
```

### Output conventions

- Lines that report a finished thing start with a mark: `✓` (green), `✗` (red), `!` (yellow). Secondary text is dim.
- `NO_COLOR` turns color off; `--json` turns all human output off and prints one JSON document (or JSON lines for streams) on stdout.
- Progress goes to stderr; results go to stdout.
- A command that fails prints one block: what happened, whose problem it is, and the next step. A bug in BenchFlow prints the path of a crash log instead of a stack trace.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Done; for `run`, every trial was scored and every gate held |
| 1 | Done, but something failed: a check found errors, a trial went unscored, or a `--fail-under` gate failed |
| 2 | Usage error: a bad flag or argument; nothing ran |
| 3 | The environment is not ready: no sandbox backend, no login for the agent |

## Interop

BenchFlow reads split-layout tasks, computes the layout's standard content digest, and accepts its reward files (`reward.json` as a flat map of finite numbers, else `reward.txt`). `bench export` writes the split layout for harnesses that read it.

## Decisions

| Decision | Why |
|---|---|
| Keep the Python tree on this branch | It is the behavioral reference: each Rust port is checked against it, and `bf.load_job` must read what the Rust engine writes |
| Shell out to the `docker` CLI rather than the Docker API | It honors the user's contexts, credentials, and `DOCKER_HOST` with no extra setup, and podman speaks the same CLI |
| Talk to Daytona over REST, not an SDK | Its API is small and stable, and an SDK would pull a second async runtime |
| Write our own small ACP client | BenchFlow's client side is small (initialize, a session, prompts, permission requests), and the trajectory needs the raw events |
| Agents are data | An agent is a TOML manifest: how to install it, how to launch it, which credentials it reads. Adding one needs no Rust |
| The oracle is the default agent | `bench run tasks/x` with no flags checks the task and needs no key |
