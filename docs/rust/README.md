# BenchFlow in Rust

This branch rebuilds BenchFlow's command line and run engine in Rust. It is one binary, `bench`, that runs AI agents on tasks in sandboxes and scores them, reads every task format BenchFlow 0.8 reads, and writes job folders 0.8's tools read. The Python package in `src/benchflow` stays on the branch as the reference implementation until each part has a Rust port that matches it.

- [install.md](./install.md): the install script, building from source, and the supported targets
- [design.md](./design.md): what a developer should feel, the architecture, and the decisions behind it
- [cli.md](./cli.md): every command, flag, exit code, environment variable, and `bench.toml` key
- [viewer.md](./viewer.md): `bench view`'s pages: the run list, compare, outcomes, and the Pareto view
- [testing-tasks.md](./testing-tasks.md): how `bench test` proves a task, and every exploit probe
- [testing-e2e.md](./testing-e2e.md): the black-box end-to-end suite in `e2e/` that drives the `bench` binary on Docker and Daytona
- [rubrics.md](./rubrics.md): rubric criteria a model judges on the host, one clean call per criterion, against the task's reference
- [harnesses.md](./harnesses.md): agents and how they are driven: manifests, options, skills, sessions, providers
- [branching.md](./branching.md): checkpoints and `bench branch`: forking a rollout into graded children
- [status.md](./status.md): what is ported, what is not yet, and how each part was verified
- [examples/](../../examples/README.md): small tasks, scripted agents, job files, and a dataset, one per feature, each with the outcome it should have

## Build it

```bash
cargo build --release          # target/release/bench
cargo install --path crates/benchflow-cli   # or put bench on your PATH
```

The workspace pins Rust 1.99 in `rust-toolchain.toml`; `rustup` fetches it on first build. `bench` needs no Python, no Node, and no virtual environment. It needs a sandbox backend to run tasks: Docker (or Podman) on the machine or through `DOCKER_HOST`, or Daytona's cloud sandboxes with `DAYTONA_API_KEY` set.

## Ten minutes, start to finish

**1. Check the machine.**

```console
$ bench doctor
  ✓ bench.toml  none (defaults apply)
  ✓ docker      Docker 29.1.3 (server), context default
  · podman      not installed
  · daytona     DAYTONA_API_KEY is not set
                fix: for cloud sandboxes, create a key at https://app.daytona.io/dashboard/keys and
                     export DAYTONA_API_KEY
  ! agents      no agent has a login here (40 agents checked)
                fix: `codex login` or `claude setup-token` sets up the most common ones; `bench
                     agents <name>` says what any agent needs
  · providers   no key here for vercel, openai, anthropic, openrouter, zai, deepseek
  ✓ git         git version 2.34.1
  ✓ disk        32 GB free in ~/.cache/benchflow
```

Each line is one check; anything not ready says how to fix it. A `·` is worth knowing but needs nothing: here, the sandbox backends not set up beside a working Docker. `bench doctor` exits 3 when no sandbox backend is ready. The outputs in this walkthrough are real: steps 1 to 4 ran as a new user (an empty home folder) on a Linux machine with Docker, and steps 5 and 6 on a Mac against Daytona; the scripts that captured them are kept, so they can be refreshed.

**2. Create a task.** `bench new` writes a complete task that already passes its own test: an instruction, a sandbox, a verifier, and a reference solution.

```console
$ bench new fizzbuzz
  ✓ created tasks/fizzbuzz (basic template)
      task.md
      sandbox/Dockerfile
      verifier/test.sh
      verifier/rubric.json
      oracle/solve.sh

  Next:
    bench check tasks/fizzbuzz  validate it (instant)
    bench test tasks/fizzbuzz   prove it: the oracle passes, doing nothing fails, exploits fail
    bench agents                pick an agent to run on it; none has a login on this machine yet
  Then make it yours: the instruction is the top of tasks/fizzbuzz/task.md.
```

The instruction is the top of `task.md`, exactly what the agent receives; the config follows in a ` ```toml task ` block.

**3. Check it.** `bench check` validates without building or running anything, and points at the exact line and column of every problem:

```console
$ bench check
error[T201]: unknown key [agent] timout
   ╭▸ tasks/fizzbuzz/task.md:11:1
   │
11 │ timout = "5m"
   │ ━━━━━━ did you mean `timeout`?
   │
   ╰ note: [agent] holds budget, network, network_reason, on_timeout, system_prompt_append,
           timeout, timeout_basis, user, or x- extensions
help: rename `timout` to `timeout`
   ╭╴
11 │ timeout = "5m"
   ╰╴   +
  ✗ fizzbuzz  1 error · task.md
      grading needs a model: no (1 criterion decided by tests, rules, the world, or replay)
      worst-case grading time: 1m (scripts 1m; in sequence, before retries)
✗ 1 task: 0 ok, 1 with errors  9ms
  1 error can be fixed automatically: bench check --fix
```

`bench check --fix` applies the fixes that are safe, including rewriting another runner's spelling into task.md's: `timeout_sec = 600` becomes `timeout = "10m"`.

```console
$ bench check --fix
  ✓ fixed tasks/fizzbuzz/task.md: rename `timout` to `timeout`
  ✓ fizzbuzz  task.md
      grading needs a model: no (1 criterion decided by tests, rules, the world, or replay)
      worst-case grading time: 1m (scripts 1m; in sequence, before retries)
✓ 1 task: 1 ok  10ms
```

**4. Prove it.** `bench test` runs the reference solution (it must pass), an empty run (it must fail), the task's own negative controls, and eight exploit probes that try to raise the reward without doing the task, all side by side on one image:

```console
$ bench test
Testing 1 task with 10 trials on docker · 4 at a time
  ✓ fizzbuzz nop                         0   7.5s
  ✓ fizzbuzz probe:forge-reward          0   7.6s
  ✓ fizzbuzz oracle                      1   7.8s
  ...

  ✓ fizzbuzz
      ✓ oracle      1  the reference solution passes
      ✓ nop         0  doing nothing fails (reward 0)
      ✓ probes    8/8  8 exploits refused
      ! defenses not in force in every trial:
          memory_limit  off: the task sets no memory limit, so the sandbox may use all 64298 MB its
                        host has
✓ 1 task proved itself  bench view jobs/tests/2026-10-10__18-15-19
```

The `!` lines name runner defenses not in force in some trial, and why; here the task sets no memory limit. When a check fails, the report says what it means and how to fix it. [testing-tasks.md](./testing-tasks.md) shows a failing report and lists every probe.

**5. Run an agent.**

```console
$ export AI_GATEWAY_API_KEY=... DAYTONA_API_KEY=...
$ bench run tasks/ -a builtin -m vercel/deepseek/deepseek-v4-flash-0731 -s daytona -j 3 --save-outputs
Running builtin (vercel/deepseek/deepseek-v4-flash-0731) on 4 trials · daytona · 3 at a time
  ✓ hello           1    57s
  ! sidecar         –    0ms  sidecar needs services beside the agent's container (api), and the da…
  ✓ hello-split     1    58s
  ✓ fizzbuzz        1  1m25s

  Score 3/4 passed · 75.0% (30.1–95.4) · 3 of 4 scored
  ! 1 trial errored [R401] · your setup's problem
      sidecar needs services beside the agent's container (api), and the daytona backend cannot run
      them
      trials: sidecar__e0d6a9c0
      next: run it with `--sandbox docker` (or podman), which starts each service on a private
            network
  Usage  10.8K in · 789 out · $0.0058
  Time   1m25s
  Job    jobs/2026-10-10__11-14-15
  Next   bench results jobs/2026-10-10__11-14-15 · bench view jobs/2026-10-10__11-14-15
```

`builtin` is BenchFlow's own tool loop (bash, view, replace, write, submit), here calling DeepSeek through the Vercel AI Gateway; `-a claude` with `CLAUDE_CODE_OAUTH_TOKEN` set (from `claude setup-token`) runs Claude Code the same way. Besides `fizzbuzz`, `tasks/` held three of the repository's test fixtures (`crates/benchflow-run/tests/fixtures/tasks`): `hello`, `hello-split` (the split layout), and `sidecar`, which needs a service container that Daytona cannot run.

Every failure says whose problem it is and what to do. Rerunning the same command resumes the job; `--fresh` starts a new one. Targets can also be GitHub repositories: `bench run benchflow-ai/skillsbench/tasks@main` fetches just that folder. A run's settings can come from job files, local or at an `https://` address (`bench run -c https://github.com/org/evals/blob/main/nightly.yaml`); `lock.json` records each file's digest ([Job files](./cli.md#job-files)).

**6. Read the results.**

```bash
bench results                    # every job under ./jobs, with pass rates and 95% intervals
bench results jobs/<job>         # one job: per task, pass@k, failures by cause
bench results jobs/a jobs/b      # paired comparison over the tasks both ran
bench view                       # the latest job in your browser
```

After the regrade below, `bench results` lists both jobs:

```console
$ bench results
jobs/2026-10-10__11-14-15  builtin · vercel/deepseek/deepseek-v4-flash-0731  just now
  4 trials · 3 passed · 75.0% (30.1–95.4) · mean reward 1
jobs/2026-10-10__11-14-15-regrade-1  builtin · vercel/deepseek/deepseek-v4-flash-0731  just now
  3 trials · 2 passed · 66.7% (20.8–93.9) · mean reward 0.667

`bench results <job>` for one job; `bench results <a> <b>` to compare.
```

## Agents, options, skills, and sessions

`bench agents` lists every agent and whether it has a login here; `bench agents <name>` shows one: how it is installed and started, the logins and providers it takes, whether it can resume a session or load one from a file, where its skills go, and its options:

```console
$ bench agents claude
claude  Claude Code via ACP (Anthropic's Agent Client Protocol), or headless with claude -p
  aliases    claude-agent-acp, claude-code
  source     built-in agents/claude.toml
  registry   claude-acp (the registry's build runs as acp:claude-acp)
  harness    acp, headless (the first is the default)
  install    npm @agentclientprotocol/claude-agent-acp@0.81.2 (runs claude-agent-acp)
  model      claude-haiku-4-5-20251001 by default
  checked    this build was never checked; `bench agents check claude -s <backend>`
  resume     on acp and headless with --resume-sessions (its sessions live in ~/.claude/projects)
  loads      on acp and headless with --load-trajectory: a Claude Code session file named
             <session-id>.jsonl (from ~/.claude/projects/<folder>/)
  skills     copied into $HOME/.claude/skills

Options (bench run -a claude --ak <name>=<value>; unset ones are not passed)
  version               Claude Code CLI version to install (the ACP adapter drives it too).
                        takes a version · installs @anthropic-ai/claude-code · default 2.1.280
  max_turns             Most agent turns.
                        takes a whole number from 1 · flag --max-turns (headless only)
  max_thinking_tokens   Most thinking tokens per response.
                        takes a whole number from 1024 · env MAX_THINKING_TOKENS
  max_budget_usd        Stop when the session has spent this many US dollars.
                        takes a number from 0 · flag --max-budget-usd (headless only)
  ...
```

- `--ak <name>=<value>` passes one of those options (`--ak max_turns=30`; `--agent-kwarg` works too). A wrong name, a value of the wrong type, or an option that does not reach the agent on the run's harness is refused before any sandbox starts.
- A task's own skills are installed for the agent by default, into the folders its CLI reads; `--skill <folder>` adds more (or replaces one of the same name), and `--no-task-skills` leaves the task's out, BenchFlow 0.8's no-skill mode. The oracle gets none.
- `--resume-sessions` makes each graded stage after the first resume the agent's own session instead of starting a fresh one, and `--load-trajectory <file>` starts the agent on a saved Claude Code or Codex session.
- `bench handoff <trial>` copies a finished Claude Code or Codex trial's saved session to this machine and opens it in that CLI, in the current folder, under your own login.

[harnesses.md](./harnesses.md) has the details of each, and which agents support what.

## Datasets

A dataset is a named, versioned list of tasks, each pinned to a git commit. `bench run -d` resolves one from a registry and fetches only the task folders it needs, with a sparse, blobless checkout per repository and commit (cached under `~/.cache/benchflow/sources/git`):

```bash
bench run -d terminal-bench@2.0 -a claude -m claude-sonnet-4-6   # BenchFlow's registry, then the public dataset registry
bench run -d skillsbench -i 'fix-*' --dry-run                     # no version: head, else the highest release
bench run -d mine@1.0 --registry ./registry.json                  # a registry file or URL
bench run -d datasets/mine/dataset.toml                           # a local dataset
```

A fetch that fails for a network reason (no connection, a timeout, a dropped transfer, HTTP 5xx or 429) is tried twice more, after 2 s and 8 s; a missing branch or commit and refused credentials are not. git runs in these checkouts without starting anything that outlives it (the fsmonitor daemon, background gc, and the daemon of git's `cache` credential helper), because a process git leaves running keeps the checkout locked. Your other credential helpers are used as configured, so a private repository whose only helper is `cache` fails to authenticate, and the error says so.

Two registry files are read: BenchFlow 0.8's dataset registry (skillsbench's `registry.json`, which pins a content digest per task) and the public `registry.json` that lists Terminal-Bench 2 and other datasets; both are JSON arrays of `{name, version, tasks: [{name, git_url, git_commit_id, path}]}`. A local dataset is a `dataset.toml` with its tasks in folders next to it:

```toml
[dataset]
name = "mine"
version = "1.0"

[[tasks]]
name = "fix-the-parser"        # the folder ./fix-the-parser (or ./tasks/fix-the-parser)
digest = "sha256:…"            # optional; `path = "…"` names another folder
```

`--include`, `--exclude`, and `--limit` narrow the dataset before anything is fetched, and `--dry-run` lists its tasks and says whether the run would resume an earlier job (`would resume jobs/<job> (3 of 3 trials kept)`) or start a new one; it fetches nothing unless the jobs folder holds a job of the same dataset and version, or one that ran a task of the same name (a run resumes by its tasks' digests, so the same tasks under another version or from a local folder count), since telling whether that job matches needs the tasks' digests (and then it also checks the agent's login, as a run would). Every trial's `result.json` records `dataset_name`, `dataset_version`, and the digest the registry pins (`registry_digest`), and the job's `lock.json` records the commit each task was fetched at. Where the registry pins a digest, each fetched task folder is checked against it before anything runs: a task that does not match stops the run (exit 1, the task's problem) and is named, and a run prints how many pinned tasks matched (`digests 3 of 3 pinned tasks match the registry`).

## Regrading

When a verifier changes, `bench regrade` scores a finished job again without running the agent. Here `hello-split`'s verifier was changed to expect 41 instead of 42 after the run above:

```console
$ bench regrade jobs/2026-10-10__11-14-15 -s daytona -j 3
Regrading 3 trials from jobs/2026-10-10__11-14-15 on daytona · 3 at a time
  ✗ hello-split     0    18s
  ✓ hello           1    20s
  ✓ fizzbuzz        1    28s

  trial                       old       new
  fizzbuzz__f2325479            1         1
  hello-split__562dddfc         1         0  !
  hello__4739a1fa               1         1

  Changed 1 of 3 regraded trials
  ! 1 trial cannot be regraded
      it never started a sandbox (R401)
      trials: sidecar__e0d6a9c0
      next: fix what stopped it (bench results names it), then run it again
  Job    jobs/2026-10-10__11-14-15-regrade-1
  Next   bench results jobs/2026-10-10__11-14-15 jobs/2026-10-10__11-14-15-regrade-1
```

Each trial is graded in a fresh sandbox from its task's image, with the outputs it saved (`outputs.tar` or `outputs/`) restored at their paths and the task's current verifier run the way a separate verifier runs. Only regular files inside the task's outputs, folders strictly inside them, and hard links between such files are restored; saved outputs that hold anything else (links, paths outside the outputs) are not regraded. The results go to a new job folder, `<job>-regrade-<n>`, whose `evaluation.json`, `summary.json`, and `lock.json` name the job it regrades (`regrade_of`); the original is never changed. Run jobs with `--save-outputs` to keep them regradable. The outputs are copied out of the sandbox under a cap (1 GiB of content plus 64 MiB for the archive) and a 3-minute deadline; a copy past either is not saved, and the log says so, so that trial cannot be regraded (a separate verifier needs the copy, so there the trial ends as the sandbox's `R103`). Entries past the byte budget (1 GiB plus 64 MiB of files, each name, a hard link's too, counted at its file's full size) are dropped and listed in `custody.json` (`outputs_refused`), and a saved `outputs/` folder past the budget is not regraded.

## Reusing earlier jobs

`bench run --diff <job>` matches each trial a run is about to start against an earlier job's trials and prints, per trial, whether it would be reused (same task, agent, model, effort, and settings), regraded from saved outputs (only the tests or verifier changed), or rerun, and why. With `--dry-run` it stops there; otherwise reused trials are copied into the new job and regraded ones are graded without running the agent. The [CLI reference](./cli.md#reusing-earlier-jobs) lists the rules.

Against the job from step 5, after `hello-split`'s verifier changed (above), nothing runs:

```console
$ bench run tasks/ -a builtin -m vercel/deepseek/deepseek-v4-flash-0731 -s daytona --save-outputs --diff jobs/2026-10-10__11-14-15 --dry-run
Reuse plan against 1 earlier job (nothing runs with --dry-run)
  reuse    fizzbuzz     same task, agent, model, and settings; it was scored (from
                        2026-10-10__11-14-15/fizzbuzz__f2325479)
  reuse    hello        same task, agent, model, and settings; it was scored (from
                        2026-10-10__11-14-15/hello__4739a1fa)
  regrade  hello-split  the task's tests or verifier changed; the agent's side is the same and its
                        outputs were saved (from 2026-10-10__11-14-15/hello-split__562dddfc)
  rerun    sidecar      the earlier trial errored (R401)
  4 trials: 2 reuse · 1 regrade · 1 rerun
```

`sidecar` is rerun because its earlier trial errored; the other two are copied as they are.

## Exporting traces

`bench export <job>` writes a job's trajectories as a Hugging Face dataset folder (one row per agent step, and a dataset card), or with `--format otel` as OpenTelemetry traces, one per trial. Jobs from BenchFlow 0.8 export the same way, and nothing is uploaded:

```console
$ bench export jobs/2026-10-10__11-14-15
  ✓ jobs/2026-10-10__11-14-15 → jobs/2026-10-10__11-14-15-hf  11 rows from 3 of 4 trials
      ! sidecar__e0d6a9c0: no trajectory: the trial stopped while setting up the sandbox, before the
        agent ran (R401)
next: datasets.load_dataset("jobs/2026-10-10__11-14-15-hf") reads it; check the dataset card
      (README.md) before sharing
```

The [CLI reference](./cli.md#trace-export) describes the rows, the spans, and the options (`--sharegpt`, `--episodes last`, `--filter`).

## Images between runs

On Docker and Podman, BenchFlow keeps the images it builds. A task's image is tagged by a hash of its build folder, and each agent's install is a cached image on top of it, so the second run of a task starts without building anything. Under an open network (the default), the agent's sandbox also refuses the cloud metadata endpoints with iptables, so an image without iptables gets a small layer that installs it (measured on 2026-10-12 with `ubuntu:24.04`: 3.9 MB, about 20 s to build once per image, then about 1 s per trial to install and test the rule); a task that opts out (see [Network policies](./testing-tasks.md#the-cloud-metadata-endpoints)) gets none, and neither does a task whose agent runs as root, since root could remove the rule. BenchFlow 0.8 removed every image after each task instead.

On Daytona, the image is built in your Daytona account from a recipe: the task's Dockerfile, a hash of its build folder, and the layers bench adds to every image (iptables for the network guard, and a record of the image's own environment). Daytona keeps each image it built under the recipe's key, so later runs of the task, from any machine on the account, start from it. A newer bench that changes those layers changes every recipe, so each task's image is built once more on its first run after the update, which can take minutes for a large image. Each trial's `image.log` says which happened: `image ready, built now: ...` or `image ready: ... (built earlier; Daytona kept it)`. Checked on 2026-10-11 with a one-line image: 22 s to build, 2 s the second time.

After each fresh build, BenchFlow removes its least recently used images while the daemon's disk has less than `BENCHFLOW_IMAGE_CACHE_MIN_FREE_GB` free (default 10; measured when the daemon runs on the same machine) or while its images take more than `BENCHFLOW_IMAGE_CACHE_MAX_GB` (no limit unless set). It never removes an image a trial of the current run still needs (a run releases each image after the last trial that uses it, so a long run over many tasks keeps only what is left to run), one a container still uses, or one used in the last 30 minutes, and the image log says what it removed. On a remote daemon, where free space cannot be measured, set `BENCHFLOW_IMAGE_CACHE_MAX_GB` to bound the cache. Set `BENCHFLOW_IMAGE_CACHE_MIN_FREE_GB=0` to turn the free-space rule off.

On Daytona, each image is built by a builder sandbox in Daytona's cloud, and a build gets the task's build timeout plus 120 seconds, the time the builder takes to start; the run's image log says so. A build still running at that limit is left to finish rather than cancelled (deleting its builder would cancel the build, and Daytona would keep the cancelled build), and the trial errors with R102. A retry, in this run or a later one, waits for that builder instead of starting another. (Two runs that start the same image at the same moment can still each start one: the check and the start are not one step.)

## Cleaning the cache

`bench cache` lists what BenchFlow keeps between runs, with sizes and last use: the images it built (on Docker and Podman), the task sources and dataset checkouts it fetched, registry indexes, and logs. `bench cache clean` prints what it would remove and removes it only with `--remove`; `--only images,sources,registries,logs` and `--older-than 30d` narrow it.

```console
$ bench cache
BenchFlow keeps 4 items · 133 MB
  cache folder ~/.cache/benchflow
  images (docker)  benchflow/writable-bin:537ae938e175      43 MB  used 7 h ago
  images (docker)  benchflow/ships-solution:2dddc26b148c    30 MB  used 7 h ago
  images (docker)  benchflow/always-passes:45d5160498ea     30 MB  used just now
  images (docker)  benchflow/fizzbuzz:45d5160498ea          30 MB  used 5 h ago
  Next  bench cache clean to see what would be removed
```

That ran as a new user on the Linux machine, right after `bench new my-task`, `bench test`, and `bench run -a oracle` on a Docker daemon that kept images from earlier runs. `my-task`'s build folder hashes the same as the `always-passes` fixture's, so its trials used that image, and it shows as used just now. `bench cache clean` marks an image used in the last 30 minutes as kept, with the reason, and leaves it out of what it would remove; it removes the rest only with `--remove`.

Only what BenchFlow made is touched. Images must carry its owner label (`ai.benchflow.owner`, for this `BENCHFLOW_OWNER` scope); they are removed by tag and never forced, so an image a container still uses stays, and images used in the last 30 minutes are kept in case a run still needs them. Pulled base images carry no label and are never listed. In the cache folder, only the folders this `bench` writes are listed; anything else (BenchFlow 0.8's `task-formats/`, another tool's files) is named and left alone. Daytona builds each sandbox's image in the cloud and keeps it there (see [Images between runs](#images-between-runs)); nothing of it is on this machine. This is a separate command rather than a `bench doctor` flag because `doctor` only reads the machine's state and never changes it.

## Importing and exporting tasks

`bench import` converts a task to task.md: the split layout (`task.toml`), BenchFlow 0.8's task.md, or a Terminal-Bench v1 task (`task.yaml`, `run-tests.sh`, `docker-compose.yaml`). It takes one task or a folder of them (`bench import terminal-bench/original-tasks tb`). `bench export <task>` writes a task.md task in the split layout, with a report of anything the split layout cannot hold. `bench run`, `check`, and `test` read the split layout and 0.8's task.md as they are; a Terminal-Bench v1 task needs `bench import` first.

## Trust

A task is code you run. Its image decides what runs in the agent's sandbox: the shell, the system's programs, the libraries the agent's own CLI loads, and the variables in the task's `[environment] env`. BenchFlow keeps the agent away from the grade and from API keys, but it cannot keep a task's image away from the agent's runtime.

- An API key never enters the sandbox. The key forwarder holds it outside the agent's reach and passes only the run's model ([Keys and the forwarder](./harnesses.md#keys-and-the-forwarder)), so a hostile task can at most spend it on that model while the trial runs.
- A subscription login (Claude Code's OAuth token or `.credentials.json`, Codex's ChatGPT login in `auth.json`) is a file or variable the agent's own CLI needs, so it is copied into the sandbox, where the task's image, its setup, its `env`, and the agent can all read it. Its value is redacted from logs and trajectories and never saved with a session, but a task that wants it can send it anywhere its network allows.

So running a task with a subscription login trusts that task as much as a program you run under that login. For tasks you did not write or review (a dataset from a registry, a repository someone sent you), use an API key, which goes through the forwarder. `bench agents <name>` says, for each credential an agent reads, whether the forwarder holds it or the agent gets it.

Secrets an agent comes across are redacted from trajectories, logs, trainer files and exports (not from the agent's own session files, which `agent/sessions/` keeps as the agent wrote them, login files left out): token families (`sk-ant-…`, `ghp_…`, JWTs and the like), credentials in URLs and in `name: value` carriers, and private keys (PEM, OpenSSH and OpenPGP blocks, armour header lines included, and the half of a key that one output shows when the key is split across two) become `***REDACTED***`. A key read in parts (`head`, then `sed -n`, then `tail`) leaves a middle part with no marker of its own; the trainer files (each trial's `trainer/verifiers.jsonl` and `trainer/adp.jsonl`, and the job's `verifiers.jsonl` and `adp.jsonl`) remove its body lines when it starts within 8,192 characters of the serialized line after the latest of: one of the key's markers alone on its line (`head -1 id`, `tail -1 id`, `grep -n END id`) or next to its body, and the last body line of 48 or more base64 characters (a key's width) they removed. So the parts may come in any order once a part with one of the key's markers has been shown (`head`, `tail`, then `sed -n`; `tail`, `sed -n`, then `head`). Only the messages' text changes; the task id, environment, model and other fields of a record stay. The cost is that lines, or quoted strings, of 16 or more base64 characters and nothing else (a commit hash, a path such as `/usr/local/bin/python3`, `"application/json"`, also after an agent's output prefix such as Qwen Code's `Output: ` or a terminal escape such as a color code, which goes with the line) in those 8,192 characters of messages are removed too, and so are such lines after a key marker that stands alone on its line in other output (a source file's `END_MARKER = "-----END RSA PRIVATE KEY-----"`, `pem.endswith("-----END RSA PRIVATE KEY-----")`, or a JSON or dict field that holds only the marker, `{"end": "-----END RSA PRIVATE KEY-----", "next": …}`). Field names in a message's text count as such strings when they are that long (`"outputWasTruncated"` in a tool's JSON output), and the marker that opened the span may itself be redacted, so the file does not always show what removed them. The half of a key that one output shows steps over the same prefixes, in trajectories, logs and exports too, so a short word after a prefix right before an END marker (`Output: done`) is redacted with it. Known limits: a part that starts further away keeps its body lines in those files; so does a middle part read before any part that shows one of the key's markers (`sed -n 4,24p id`, then `head` and `tail`); so does a part read after one output showed the key's first and last lines together (`head -3 id; tail -3 id`, then `sed -n 4,24p id`), because that output's key is redacted whole, markers included; the trial's `trajectory/acp_trajectory.jsonl` and `trainer/atif.json` keep the middle part, because each event is redacted on its own; `grep -r` context lines under a folder whose name has a space keep the key's body lines; so do a part's lines next to a terminal escape that is not stepped over (a C1 control, a lone BEL or SI, `ESC c`, a device-control or tmux title string, a window title over 256 characters or with a backslash in it) or followed by an escape at the line's end (a color reset after the text), in the trainer files; and remainders of 15 characters or fewer (a key's short last line, the end of a line cut mid-way) are left.

## Coming from 0.8

The 0.8 spellings keep working and print the new one once:

| 0.8 | Now |
|---|---|
| `bench eval run --tasks-dir tasks/` | `bench run tasks/` |
| `bench eval run --source-repo org/repo --source-path tasks --source-ref main` | `bench run org/repo/tasks@main` |
| `bench eval run --dataset skillsbench@1.1` | `bench run -d skillsbench@1.1` |
| `bench eval list`, `metrics`, `inspect`, `compare` | `bench results` |
| `bench eval view` | `bench view` |
| `bench eval regrade` | `bench regrade` |
| `bench tasks init` | `bench new` |
| `bench tasks check` | `bench check` |
| `bench tasks migrate` | `bench import` |
| `bench tasks export` | `bench export` |
| `bench train convert` | `bench export <job> --format hf` or `--format otel` |
| `bench agent list`, `show` | `bench agents` |

Job folders keep 0.8's layout (`result.json`, `config.json`, `timing.json`, `trajectory/acp_trajectory.jsonl`, `verifier/`), so `bf.load_job` and the 0.8 viewer read Rust runs. A command that has no Rust equivalent yet says so and points to the Python CLI.
