# CLI reference

`bench` runs AI agents on tasks in sandboxes and scores them. This page lists every command and flag (generated from the binary's own definitions, so it cannot drift), then the conventions every command shares: global options, JSON output, exit codes, environment variables, and `bench.toml`.

## Global options

Every command takes these:

| Option | Meaning |
|---|---|
| `-C <DIR>` | Run as if `bench` started in `DIR` |
| `--json` | Print JSON instead of text. Commands print one JSON document on stdout; `bench run` and `bench test` print one JSON object per event while they run, then a final object |
| `-q`, `--quiet` | Print only results and errors |
| `-v`, `--verbose` | Print more detail, such as image build output; `-vv` adds debug logs |
| `--color <WHEN>` | `auto` (the default), `always`, or `never`. `NO_COLOR` turns color off; `CLICOLOR_FORCE=1` forces it |

Results go to stdout; progress, notes, and diagnostics go to stderr, so `bench results --json > results.json` captures only the results.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Done. For `run`: every trial was scored and every gate held. For `check`: no errors. For `test`: every task proved itself |
| 1 | Done, but something failed: a check found errors, a trial went unscored or errored or was stopped (Ctrl-C), a probe got through, or a `--fail-under` gate failed. Also when something bench had to fetch first could not be read (a dataset registry, a task's git repository); then nothing ran |
| 2 | A usage error: a bad flag, a missing path, an invalid `bench.toml`; nothing ran |
| 3 | The machine is not ready: no sandbox backend, or no login for the agent |
| 130 | Quit at once: a third Ctrl-C (or a second `kill`) |

## Commands

<!-- BEGIN GENERATED: bench commands -->

### bench new

Create a task that already passes `bench test`.

```text
bench new [OPTIONS] [NAME]
```

| Argument | Meaning |
|---|---|
| `<NAME>` | The task's name (`my-task`, `org/my-task`), or a path whose folder exists (`tasks/my-task`) |
| `-t`, `--template` `<TEMPLATE>` | The template to start from (default: `basic`) |
| `--dir` `<DIR>` | Where to create it (default: ./tasks/ when it exists, else .) |
| `--list` | List the templates |

```text
Examples:
  bench new fix-the-parser                 # tasks/fix-the-parser/ (or ./fix-the-parser/ outside a tasks folder)
  bench new acme/dedup-revenue -t python   # a pytest verifier and a rubric with points
  bench new --list                         # every template
```

### bench check

Validate tasks without building or running anything.

```text
bench check [OPTIONS] [PATHS]...
```

| Argument | Meaning |
|---|---|
| `<PATHS>` | Task folders, or folders of tasks |
| `--fix` | Apply the fixes that are safe to apply |
| `--strict` | Treat warnings as errors |
| `--schema` | Check only the task file, not the folders a run needs (for authoring fixtures) |
| `-w`, `--watch` | Check again whenever a file in the tasks changes |
| `--review` | Also have a model review each task's quality (its opinion, shown after the checker's) |
| `--model` `<MODEL>` | The model for --review, as <provider>/<model> (default: vercel/deepseek/deepseek-v4-flash-0731) |
| `--save` | Save each review to review.json in its task folder |

```text
Checks the task document, its config, the rubric, and the package files, and
points at the exact line and column of every problem. Tasks in any format
BenchFlow reads are checked: task.md, 0.8's task.md, and the split layout
(task.toml), where keys other runners would silently ignore are flagged.

Examples:
  bench check                  # every task under this folder (or bench.toml's [tasks] paths)
  bench check tasks/my-task    # one task
  bench check --fix            # apply the safe fixes, such as renaming a misspelled key
  bench check -w tasks/my-task # check again on every save
  bench check tasks/my-task --review          # and a model's opinion of the task's quality
  bench check tasks/my-task --review --save   # ... saved to tasks/my-task/review.json

--review sends the task's files (size-capped, secrets redacted, .env and key
files never) to a model, by default vercel/deepseek/deepseek-v4-flash-0731 on
the Vercel AI Gateway. Its findings are shown after the checker's, marked as
its opinion, and never change the exit code.
```

### bench test

Prove tasks: the oracle passes, doing nothing fails, exploits are refused.

```text
bench test [OPTIONS] [PATHS]...
```

| Argument | Meaning |
|---|---|
| `<PATHS>` | Task folders, or folders of tasks |
| `-s`, `--sandbox` `<BACKEND>` | Where trials run: docker, podman, or daytona (default: bench.toml, then the first one available) |
| `-j`, `--concurrency` `<N>` | Trials at a time |
| `--keep` `<KEEP>` | Keep sandboxes after their trials, for inspection (default: `never`); one of `never`, `on-failure`, `always` |
| `--jobs-dir` `<DIR>` | Where job folders go (default: bench.toml's, then ./jobs) |
| `--profile` `<NAME>` | Apply a [profile.<NAME>] from bench.toml |
| `--runtime` `<RUNTIME>` | Start sandboxes under this container runtime, such as runsc (gVisor), on Docker daemons that have it |
| `--probes` `<WHICH>` | Which exploit probes to run: all, none, or a comma-separated list (default: `all`) |
| `--list-probes` | List the exploit probes and what each tests |

```text
For each task, runs side by side on one image:
  the oracle         must pass (the verifier can say yes)
  an empty run       must fail (the verifier can say no)
  each control       must fail (tasks' own known-bad answers, from controls/)
  each exploit probe must score no higher than the empty run

Examples:
  bench test                         # every task here, on the default sandbox
  bench test tasks/x -s daytona      # one task, in Daytona's cloud sandboxes
  bench test --probes none           # just the oracle, the empty run, and controls
  bench test --probes forge-reward,path-shadow
```

### bench run

Run an agent on tasks and score them.

```text
bench run [OPTIONS] [TARGETS]...
```

| Argument | Meaning |
|---|---|
| `<TARGETS>` | Task folders, folders of tasks, or GitHub repositories (org/repo[/path][@ref]) |
| `-a`, `--agent` `<AGENT>` | The agent: claude, codex, gemini, opencode, builtin, terminus-2, a project agent, oracle, nop, or script:<file> (your own script, run like the oracle) |
| `-m`, `--model` `<MODEL>` | The model, in the agent's own naming, or `<provider>/<model>` to run on a provider's key (vercel, openai, anthropic, openrouter, zai, deepseek, mimo) |
| `-e`, `--effort` `<LEVEL>` | Reasoning effort, when the agent exposes one (low, medium, high, max, ...) |
| `--harness` `<HARNESS>` | How to drive the agent: acp or headless (default: the agent's own; `bench agents` lists each agent's); one of `acp`, `headless` |
| `--backend` `<URL>` | For `-a external`: the service that decides each step (an HTTP backend protocol) |
| `--ak` `<NAME=VALUE>` | An agent option (repeatable), checked before any sandbox starts; `bench agents <name>` lists an agent's |
| `--resume-sessions` | On each graded stage after the first, resume the agent's own session instead of starting a fresh one (agents that can) |
| `--skill` `<FOLDER>` | A skill to install for the agent (repeatable): a folder with a SKILL.md, or a folder of them; it replaces a task skill of the same name |
| `--no-task-skills` | Leave out the task's own skills (BenchFlow 0.8's no-skill mode); --skill folders are still installed |
| `--load-trajectory` `<FILE>` | Start the agent on this session file and resume it with the first prompt (claude: <id>.jsonl; codex: rollout-*.jsonl) |
| `-s`, `--sandbox` `<BACKEND>` | Where trials run: docker, podman, or daytona (default: bench.toml, then the first one available) |
| `-j`, `--concurrency` `<N>` | Trials at a time |
| `--keep` `<KEEP>` | Keep sandboxes after their trials, for inspection (default: `never`); one of `never`, `on-failure`, `always` |
| `--jobs-dir` `<DIR>` | Where job folders go (default: bench.toml's, then ./jobs) |
| `--profile` `<NAME>` | Apply a [profile.<NAME>] from bench.toml |
| `--runtime` `<RUNTIME>` | Start sandboxes under this container runtime, such as runsc (gVisor), on Docker daemons that have it |
| `-k`, `--trials` `<N>` | Trials per task |
| `--job` `<NAME>` | Job name (an existing job is resumed) |
| `--fresh` | Start a new job instead of resuming the latest matching one |
| `-i`, `--include` `<PATTERN>` | Only tasks whose names match: an exact name, or a pattern of `*`, `?`, `[...]` and `[!...]` (repeatable) |
| `-x`, `--exclude` `<PATTERN>` | Skip tasks whose names match: an exact name, or a pattern of `*`, `?`, `[...]` and `[!...]` (repeatable) |
| `-n`, `--limit` `<N>` | Run at most N tasks (the first N by name) |
| `--env` `<KEY=VALUE>` | Environment variable for the agent (repeatable) |
| `--timeout-multiplier` `<X>` | Scale every task's agent, verifier, and build timeouts |
| `--agent-timeout-multiplier` `<X>` | Scale every task's agent timeout (overrides --timeout-multiplier) |
| `--verifier-timeout-multiplier` `<X>` | Scale every task's verifier timeout (overrides --timeout-multiplier) |
| `--build-timeout-multiplier` `<X>` | Scale every task's image build timeout (overrides --timeout-multiplier) |
| `--idle-timeout` `<DURATION>` | Abort a prompt when the agent is silent this long (e.g. 10m) |
| `--max-tool-calls` `<N>` | Stop the agent after N tool calls (native harnesses) |
| `--max-tokens` `<N>` | Stop the agent after N tokens, input plus output (native harnesses) |
| `--judge-model` `<MODEL>` | The model that judges rubric criteria with judge "llm", as <provider>/<model> (default: the rubric's x-judge.model, else vercel/deepseek/deepseek-v4-flash-0731) |
| `--fail-under` `<RATE>` | Exit 1 when the pass rate is below this (0 to 1) |
| `--extra-instruction` `<TEXT>` | Text appended to every task's instruction |
| `--save-outputs` | Save each trial's workdir (or declared outputs) in its folder |
| `-d`, `--dataset` `<NAME@VERSION>` | A dataset from a registry, name@version (or a dataset.toml); instead of targets |
| `--registry` `<URL-OR-FILE>` | The registry for --dataset: a URL or a file (default: BenchFlow's, then the public dataset registry) |
| `-c`, `--config` `<FILE-OR-URL>` | A job file (TOML, or YAML or JSON by extension), or an https:// URL of one; repeatable, later files win |
| `--print-config` | Print the resolved settings as a job file, and exit |
| `--rerun` `<KINDS>` | On resume, also rerun trials that were: errored (a trial whose unexplained sandbox loss repeated is kept otherwise), unscored, failed (comma-separated) |
| `--diff` `<JOB>` | An earlier job (folder, or name in the jobs folder) to reuse trials from: each planned trial is reused, regraded from its saved outputs, or rerun, and the plan is printed first (repeatable) |
| `--dry-run` | Print the plan and run nothing; exits 0, or 3 (not ready) after naming what a run would stop on (a missing login) |

```text
Targets are task folders, folders of tasks, or GitHub repositories:
  bench run tasks/                                  # local tasks
  bench run benchflow-ai/skillsbench/tasks@main     # a folder of a repository, fetched sparsely
  bench run https://github.com/org/repo/tree/v1/tasks

Or a dataset from a registry, every task pinned to a git commit:
  bench run -d terminal-bench@2.0                   # BenchFlow's registry, then the public dataset registry
  bench run -d mine@1.0 --registry registry.json    # a registry file or URL
  bench run -d datasets/mine/dataset.toml           # a local dataset

Settings come from flags, then job files (-c, later files win), then
bench.toml's profile and [run]. `--print-config` shows the result.

The agent defaults to the oracle (the task's reference solution), so a bare
`bench run tasks/x` checks a task with no model and no key.

Rerunning the same command resumes its job: finished trials are kept
(errored ones run again; `--rerun failed,unscored` reruns those too).
`--fresh` starts a new job. Every job folder has a lock.json recording
exactly what ran.

`--diff <job>` matches each planned trial against an earlier job's: reuse
(same task, agent, model, effort, and settings; copied in), regrade (only the
tests or verifier changed, and outputs were saved; graded again without the
agent), or rerun, with the reason. With --dry-run it only prints the plan.

Ctrl-C drains the run: no new trial starts and running ones finish. A second
Ctrl-C stops them (their sandboxes are removed; they are left out of the pass
rate and listed apart), and a third quits at once.

Examples:
  bench run tasks/ -a claude -m claude-sonnet-4-6 -j 8
  bench run tasks/ -a codex -k 3 --fail-under 0.6      # 3 trials per task; exit 1 below 60%
  bench run -c base.toml -c claude.toml                # job files
  bench run -c https://github.com/org/evals/blob/main/nightly.yaml   # a job file from a URL
  bench run tasks/ --dry-run                           # print the plan and run nothing
  bench run tasks/ -a codex --diff jobs/last --dry-run # what an earlier job can supply
```

### bench results

Summarize jobs; with two or more, compare them.

```text
bench results [OPTIONS] [JOBS]...
```

| Argument | Meaning |
|---|---|
| `<JOBS>` | Job folders, or folders of jobs (default: ./jobs) |
| `-t`, `--trials` | List every trial, not just the summary |
| `--analyze` | Have a model explain why trials failed (with no job: the latest one) |
| `--model` `<MODEL>` | The model for --analyze, as <provider>/<model> (default: vercel/deepseek/deepseek-v4-flash-0731) |
| `--max-trials` `<N>` | Analyze at most this many failed trials in all (default: `10`) |
| `--no-save` | Do not write analysis.json into the job folder |

```text
Examples:
  bench results                         # every job under ./jobs
  bench results jobs/2026-10-10__09-12-03
  bench results jobs/a jobs/b           # paired comparison over the tasks both ran
  bench results --analyze               # a model explains why the latest job's trials failed
  bench results jobs/x --analyze --max-trials 30

--analyze sends each failed, errored, or unscored trial's record, instruction,
trajectory tail, verifier output, and agent logs (capped, secrets redacted) to
a model, by default vercel/deepseek/deepseek-v4-flash-0731, and saves what it
says to analysis.json in the job folder. What it says is its opinion; the
problem BenchFlow recorded is shown next to it.

Scores that stand but need a second look are named under the score, and in
--json: rewards outside their task's range, counted as written (0 to 1 unless
a task declares [verifier] reward_range; rewards_out_of_range),
and trials graded although locked_system_dirs failed, so the agent could
change system folders the verifier runs from (graded_unlocked_system_dirs).
Two more lines count the trials whose model agent could reach the cloud
metadata endpoints, and those whose model agent ran on Daytona without the
guard on the sandbox daemon's ports. --json lists each such trial, stopped
ones included, with its record's state and note (cloud_metadata_reachable,
root_service_unguarded); with one job it also holds every trial's
result.json, whose defenses hold all its records.
```

### bench view

Open a job in your browser.

```text
bench view [OPTIONS] [JOBS]...
```

| Argument | Meaning |
|---|---|
| `<JOBS>` | A job folder, a folder of jobs, or a trial folder (default: the latest job); two or more jobs open the compare view |
| `--port` `<PORT>` | Port to serve on (default: the first free one from 8765) |
| `--no-open` | Do not open a browser |

```text
Pages:
  /           the run list, and each trial's trajectory, verifier output, and files
  /compare    tasks by job, pass rates with 95% intervals, paired differences
  /outcomes   how trials ended, problems by code and owner, time per phase
  /pareto     pass rate against cost, time, and tokens per trial, with each frontier
  /branches   each fork of a `bench branch` job: value, checkpoint, parent, and
              each child's label, status, reward, and advantage
Each keeps its jobs and filters in the URL; /api/<page> answers the same as JSON.

Examples:
  bench view                          # the latest job
  bench view jobs/a jobs/b jobs/c     # opens on /compare; /pareto is one click away
```

### bench regrade

Score a finished job again with each task's current verifier.

```text
bench regrade [OPTIONS] <JOB>
```

| Argument | Meaning |
|---|---|
| `<JOB>` | The finished job folder |
| `-s`, `--sandbox` `<BACKEND>` | Where the fresh sandboxes run: docker, podman, or daytona (default: bench.toml, then the first one available) |
| `-j`, `--concurrency` `<N>` | Trials regraded at a time |
| `--verifier-timeout-multiplier` `<X>` | Scale every task's verifier timeout |
| `--profile` `<NAME>` | Apply a [profile.<NAME>] from bench.toml |

```text
Grades each trial again from the outputs it saved (outputs.tar or
outputs/), in a fresh sandbox from its task's image, with the task's
verifier as it is now. The agent does not run. Results go to a new job
folder next to the original, <job>-regrade-<n>, and the old and new reward
of every trial are printed side by side.

Trials that saved no outputs cannot be regraded; run them with
--save-outputs (tasks that declare outputs or use a separate verifier
save them anyway).

Examples:
  bench regrade jobs/2026-10-10__09-12-03
  bench regrade jobs/2026-10-10__09-12-03 -s daytona -j 16
```

### bench branch

Fork a rollout from a checkpoint into children, each graded.

```text
bench branch [OPTIONS] <TARGETS>...
```

| Argument | Meaning |
|---|---|
| `<TARGETS>` | Task folders, or folders of tasks |
| `-a`, `--agent` `<AGENT>` | The agent: oracle, nop, script:<file>, or a model agent (default: oracle) |
| `-m`, `--model` `<MODEL>` | The model, for model agents |
| `--effort` `<EFFORT>` | Reasoning effort, for agents that take one |
| `-e`, `--env` `<KEY=VALUE>` | Extra environment variables for the agent (KEY=VALUE, repeatable) |
| `--at` `<POINT>` | Where to fork: start, prompt:N, or stage:<name> (default: `start`) |
| `--children` `<K>` | K children (2 or more), each asked the task's own instruction (at stage:<name>, the next stage's prompt) |
| `--child` `<SPEC>` | One child: label=<name>[,prompt=<text>\|prompt-file=<path>] (repeatable; `@instruction` in a prompt is the task's) |
| `--parent` `<MODE>` | What becomes of the parent after its fork: discard, or continue (it goes on and is graded) (default: `discard`) |
| `--retain-checkpoints` | Keep the checkpoint after the fork (`bench cache` lists it) |
| `--resume-session` | At prompt:N or stage:<name>, children resume the parent's session (`session/load`) instead of starting fresh ones |
| `--job` `<NAME>` | The job folder's name under the jobs folder |
| `-s`, `--sandbox` `<BACKEND>` | Where trials run: docker, podman, or daytona (default: bench.toml, then the first one available) |
| `-j`, `--concurrency` `<N>` | Trials at a time |
| `--keep` `<KEEP>` | Keep sandboxes after their trials, for inspection (default: `never`); one of `never`, `on-failure`, `always` |
| `--jobs-dir` `<DIR>` | Where job folders go (default: bench.toml's, then ./jobs) |
| `--profile` `<NAME>` | Apply a [profile.<NAME>] from bench.toml |
| `--runtime` `<RUNTIME>` | Start sandboxes under this container runtime, such as runsc (gVisor), on Docker daemons that have it |

```text
Runs the task to the fork point, stores the sandbox's files as a checkpoint
(Docker: an image; Daytona: a snapshot), and runs each child in a sandbox of
its own made from it. Each child is graded by the task's verifier; the
fork's value is the mean of the children's rewards. The checkpoint is
deleted afterwards unless --retain-checkpoints. The parent is discarded
after its fork, or with --parent continue goes on in its own sandbox while
the children run, and is graded too (its reward is not in the fork's
value).

Fork points: `start` (after setup, before the agent), `prompt:N` (after the
parent's N-th prompt, for model agents), and `stage:<name>` (after a graded
stage of a task with stages; children run the stages after it). A child
sends its own prompt (at `stage:<name>`, in place of the next stage's), or
without one the task's instruction again (the next stage's); it starts a
fresh session in the parent's files, or with --resume-session continues the
parent's conversation through `session/load` (ACP agents that offer it).

Records (BenchFlow 0.8's): tree.json and checkpoints.json in the trial
folder, each child under branches/<fork>/children/n<k>/, and a `branches`
block in the trial's result.json.

Examples:
  bench branch tasks/x --children 3                    # the oracle, 3 children
  bench branch tasks/x -a builtin -m mimo/mimo-v2.6-flash -s daytona \
      --child label=base --child "label=hint,prompt=@instruction Start with the tests."
  bench branch tasks/x -a opencode -m vercel/deepseek/deepseek-v4-flash-0731 \
      --at prompt:1 --resume-session --child "label=check,prompt=Check your work."
  bench branches jobs/2026-10-10__20-00-00             # each fork's value and children
```

### bench branches

Show a branch job's forks: value, children, rewards.

```text
bench branches <JOB>
```

| Argument | Meaning |
|---|---|
| `<JOB>` | A job folder `bench branch` wrote |

### bench shell

Open a shell in a task's sandbox, then grade what you left.

```text
bench shell [OPTIONS] [TASK]
```

| Argument | Meaning |
|---|---|
| `<TASK>` | The task folder (default: the task in this folder) |
| `-s`, `--sandbox` `<BACKEND>` | Where the sandbox runs: docker or podman (default: bench.toml, then the first one available) |
| `--as` `<USER>` | Who to be: the agent user (default) or root (default: `agent`) |
| `--no-grade` | Exit without grading |

```text
Starts the task's sandbox as an agent would see it, as the agent user, in the
workdir. Exit the shell to grade the sandbox as it is, with the same handoff
and verifier a real run uses.

Examples:
  bench shell tasks/x              # try the task by hand, then see its score
  bench shell tasks/x --as root    # debug the image as root
  bench shell tasks/x --no-grade
```

### bench handoff

Resume a finished trial's agent session in its CLI, here.

```text
bench handoff [OPTIONS] <TRIAL> [-- <ARGS>...]
```

| Argument | Meaning |
|---|---|
| `<TRIAL>` | The trial folder (jobs/<job>/<task>__<id>) |
| `--session` `<ID>` | Resume this session, when the trial saved more than one (default: the one the harness recorded, else the newest) |
| `--replace` | Overwrite a local session with the same id and other content |
| `--dry-run` | Print what would be copied and the command, and copy nothing |
| `--trust` | Hand off a trial from outside this project's jobs folder (its session becomes your agent's context, run with your permissions) |
| `<ARGS>` | Extra arguments for the agent's CLI, after `--` |

```text
Copies the session the trial saved (agent/sessions/, recorded in result.json
as agent_session) to where the agent's own CLI on this machine looks for it,
under the session's own id, then starts the CLI resuming it in this folder:
  Claude Code  ~/.claude/projects/<this folder>/<id>.jsonl, then claude --resume <id>
  Codex        ~/.codex/sessions/..., then codex resume <id>
($CLAUDE_CONFIG_DIR and $CODEX_HOME are honored.) The CLI uses your own login
and model from here on. A local session with the same id and other content is
never overwritten unless you pass --replace.

A trial outside this project's jobs folder (bench.toml's jobs-dir, else
./jobs) is refused unless you pass --trust: its session becomes your local
agent's context, and the agent runs with your permissions on this machine.

Examples:
  bench handoff jobs/2026-10-10__09-12-03/hello__84c1190c
  bench handoff jobs/x/hello__84c1190c --dry-run      # what it would copy, and the command
  bench handoff jobs/x/hello__84c1190c -- --version   # extra arguments for the CLI
```

### bench doctor

Check this machine: sandboxes, agent logins, disk.

```text
bench doctor [OPTIONS]
```

| Argument | Meaning |
|---|---|
| `-s`, `--sandbox` `<BACKEND>` | Check only these sandbox backends |
| `--confinement` | Start one sandbox, confine it as a trial does, and report each setting read back from it |
| `--runtime` `<RUNTIME>` | With --confinement: start the sandbox under this container runtime, such as runsc |
| `--image` `<IMAGE>` | With --confinement: the image to start from (default: `ubuntu:24.04`) |

### bench cache

List what BenchFlow keeps between runs; `cache clean` removes it.

```text
bench cache [OPTIONS]
       bench cache <COMMAND>
```

| Argument | Meaning |
|---|---|
| `--only` `<KINDS>` | Only these kinds: images, sources, registries, logs, checkpoints (comma-separated) |
| `-s`, `--sandbox` `<BACKEND>` | Backends to look at: docker, podman, daytona (default: the ones installed, and daytona when DAYTONA_API_KEY is set) |
| `--older-than` `<DURATION>` | Only what was last used longer ago than this (e.g. 30d, 12h) |

```text
Lists the images BenchFlow built (on Docker or Podman, with their last use),
the task sources and dataset checkouts it fetched, registry indexes, and logs
in its cache folder (~/.cache/benchflow, or BENCHFLOW_CACHE_DIR), with sizes.

`bench cache clean` prints what it would remove; add --remove to remove it.
Only what BenchFlow made is touched: images carry its owner label (and are
removed by tag, never forced, so an image a container uses stays), and files
in the cache folder that this bench did not write are left alone. Images used
in the last 30 minutes are kept, since a run may still need them.

Examples:
  bench cache                                   # everything, with sizes and last use
  bench cache clean                             # what would be removed
  bench cache clean --only images --remove      # remove BenchFlow's images
  bench cache clean --older-than 30d --remove   # remove what was not used for 30 days
```

### bench agents

List agents and whether each can run here.

```text
bench agents [OPTIONS] [NAME]
       bench agents <COMMAND>
```

| Argument | Meaning |
|---|---|
| `<NAME>` | Show one agent in detail, including its install script (acp:<id>[@<version>] for an ACP registry entry) |
| `--registry` | List the entries of the public ACP registry, each runnable as acp:<id> |

```text
Examples:
  bench agents                       # every agent, its login, and whether this build was checked
  bench agents goose                 # how goose is installed, started, and logged in
  bench agents acp:goose             # the same for the ACP registry's goose entry
  bench agents --registry            # every entry of the public ACP registry
  bench agents check goose -s daytona

With --json, the list is an array of {name, aliases, kind, harnesses, description, options}: kind is script (oracle, nop), manifest, native, or backend (external), and harnesses are the values --harness takes, the default first. One agent's page is its entry; --registry gives {name, version, installs, built_in_as, description} per entry.
```

### bench import

Convert a task.toml (split layout), 0.8 task.md, or Terminal-Bench v1 task to task.md.

```text
bench import [OPTIONS] <SOURCE> [OUT]
```

| Argument | Meaning |
|---|---|
| `<SOURCE>` | A task folder in the split layout (task.toml), 0.8's task.md, or Terminal-Bench v1 (task.yaml), or a folder of them |
| `<OUT>` | Where to write (default: next to the source, as <name>-taskmd) |
| `--overwrite` | Replace an existing output folder |
| `--cpus` `<N>` | CPUs for Terminal-Bench v1 tasks, which declare none |
| `--memory-mb` `<MB>` | Memory in MB for Terminal-Bench v1 tasks |
| `--storage-mb` `<MB>` | Disk in MB for Terminal-Bench v1 tasks |
| `--gpus` `<N>` | GPUs for Terminal-Bench v1 tasks |

```text
A Terminal-Bench v1 task (task.yaml, run-tests.sh, solution.sh,
docker-compose.yaml) is converted to the split layout, then imported: a compose file with one service folds into the Dockerfile,
run-tests.sh becomes the verifier with a reward from its exit code, and other
files go into the sandbox's build context. A task with solution.yaml is
refused; rewrite it as solution.sh.

Examples:
  bench import tasks/my-task                    # one task, written to tasks/my-task-taskmd
  bench import terminal-bench/original-tasks tb # every Terminal-Bench v1 task, into tb/
  bench import original-tasks tb --cpus 2 --memory-mb 4096
```

### bench export

Write a task in the split layout, or a job's traces as a Hugging Face dataset or OpenTelemetry.

```text
bench export [OPTIONS] <TASK> [OUT]
```

| Argument | Meaning |
|---|---|
| `<TASK>` | A task.md task folder, or a job folder |
| `<OUT>` | Where to write (default: next to it, as <name>-split, <name>-hf, or <name>-otel.jsonl) |
| `--overwrite` | Replace an existing output |
| `--allow-degraded` | Export even when the split layout drops something that decides how the task runs, such as its rubric |
| `-f`, `--format` `<FORMAT>` | What to write: split for a task; hf (the default) or otel for a job; one of `split`, `hf`, `otel` |
| `--sharegpt` | Add a conversations_sharegpt column (ShareGPT's from/value form) |
| `--episodes` `<EPISODES>` | One row per agent step (all), or only each trial's last (last) (default: `all`); one of `all`, `last` |
| `--filter` `<FILTER>` | Which trials: all, success (reward above 0), or failure (default: `all`); one of `all`, `success`, `failure` |
| `--instruction` | Add an instruction column: each trial's first user message |
| `--verifier-output` | Add a verifier_output column: the verifier's stdout and stderr |

```text
A task folder is written in the split layout (task.toml), for harnesses that
read it. A job folder is written as traces:

  --format hf    a Hugging Face dataset folder: data/train.jsonl, one row per
                 agent step (the conversation up to it), and a README.md
                 dataset card; load it with datasets.load_dataset(<folder>)
  --format otel  OpenTelemetry: one OTLP JSON trace per trial, one per line

Each trial's trajectory is its ATIF document (trainer/atif.json), or one
rebuilt from trajectory/acp_trajectory.jsonl; jobs from BenchFlow 0.8 read the
same way.

Examples:
  bench export tasks/my-task                     # tasks/my-task-split
  bench export jobs/2026-10-10__09-12-03         # jobs/2026-10-10__09-12-03-hf
  bench export jobs/2026-10-10__09-12-03 --sharegpt --episodes last
  bench export jobs/2026-10-10__09-12-03 --format otel traces.jsonl
```

### bench explain

Explain an error code.

```text
bench explain [CODE]
```

| Argument | Meaning |
|---|---|
| `<CODE>` | The code, such as T201 or R301 (omit to list every code) |

### bench schema

Print the JSON Schema of a task, job file, result, or lock.

```text
bench schema <KIND>
```

| Argument | Meaning |
|---|---|
| `<KIND>` | Which schema; one of `task`, `job`, `result`, `lock` |

```text
Kinds:
  task    task.md's config block (the ```toml task block)
  job     a job file for `bench run -c`
  result  a trial's result.json
  lock    a job's lock.json

Examples:
  bench schema lock > lock.schema.json
```

<!-- END GENERATED -->

## bench.toml

An optional project file, found by walking up from the current folder (stopping at the repository root). Flags override the profile, which overrides `[run]`, which overrides built-in defaults. Unknown keys are errors.

```toml
[run]
agent = "claude"              # the default --agent (otherwise: oracle)
harness = "headless"          # acp or headless (otherwise: the agent's own)
model = "claude-sonnet-4-6"
reasoning-effort = "high"
sandbox = "docker"            # docker, podman, or daytona (otherwise: the first one ready)
concurrency = 8
trials = 1
jobs-dir = "jobs"
timeout-multiplier = 1.0       # scales the agent, verifier, and build timeouts
agent-timeout-multiplier = 1.0 # each overrides timeout-multiplier for one phase
verifier-timeout-multiplier = 1.0
build-timeout-multiplier = 1.0
idle-timeout = "10m"
fail-under = 0.8
sandbox-user = "agent"        # the user agents run as when a task names none
retries = 1                   # retries of a trial that failed for a transient reason
runtime = "runsc"             # the container runtime, on Docker daemons that have it
resume-sessions = true        # --resume-sessions
skills = ["skills"]           # --skill folders
task-skills = false           # --no-task-skills
load-trajectory = "session.jsonl"   # --load-trajectory

[run.env]                     # environment for the agent
MY_FLAG = "1"

[run.agent-options]           # --ak; `bench agents <name>` lists an agent's options
max_turns = 30

[tasks]
paths = ["tasks"]             # where check, test, and run look when given no path

[profile.ci.run]              # bench run --profile ci
sandbox = "daytona"
concurrency = 32
```

A `.env` file next to `bench.toml` (or in the current folder) is read for variables that are not already set. A cloned repository brings its `.env`, so it may set only keys and tokens (names ending in `_API_KEY`, `_KEY`, or `_TOKEN`, except `GIT_*`, `SSL_*`, and `DAYTONA_API_KEY`, which decides whose sandboxes get the run's keys and logins), `DAYTONA_TARGET`, and `BENCHFLOW_OWNER`; the variables it took are named in a note. A `.env` that git tracks came with the clone and is not read at all; anything else (base URLs such as `DAYTONA_API_URL`, proxies, certificate files, git's settings, `PATH`) is ignored and named in a note, and must be set in the shell. `DAYTONA_API_URL` must be https, or http to this machine, wherever it is set.

## Job files

`bench run -c job.toml` reads a run from a file. A job file takes the keys of `bench.toml`'s `[run]` at its top level, plus `targets`, `dataset`, `registry`, `include`, and `exclude`; a file ending in `.yaml` or `.yml` is read as YAML, one ending in `.json` as JSON, and any other as TOML. `bench schema job` prints its JSON Schema.

```toml
dataset = "terminal-bench@2.0"   # or: targets = ["tasks/"]
include = ["build-*"]
agent = "claude"
model = "claude-sonnet-4-6"
trials = 3
concurrency = 8

[env]
MY_FLAG = "1"
```

`-c` is repeatable. Later files override earlier ones key by key: tables merge, lists append, and an explicit empty list (`include = []`) clears what came before. Relative `targets`, `jobs-dir`, `skills`, `load-trajectory`, and a local `dataset` or `registry` file are relative to the file that names them. Flags override job files, which override the profile and `[run]`; a list given on the command line replaces the files' list. `bench run --print-config` prints the fully resolved settings as a job file and exits.

A job file can also be an `https://` URL: a raw file (`https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>`), or a GitHub page for one (`https://github.com/<owner>/<repo>/blob/<ref>/<path>`), which is read from its raw address. The file is fetched once, within 30 seconds and 1 MB, and only parsed, never run. Its syntax follows the URL's file name as for local files. Relative paths in it are relative to the current folder, and it may name no others: an absolute path, `~`, or `..` in `targets`, `jobs-dir`, `dataset`, or `registry` is refused. Settings that bring this machine's files or variables into the run (`env`, `skills`, `load-trajectory`, and a `script:` agent) are refused in a fetched file; give them as flags, or download the file, check it, and pass it as a local job file (`-c ./file.toml`). Every redirect must stay on HTTPS, at most 5. `lock.json` records every job file the run read, path or URL, with the URL fetched and the SHA-256 digest of its bytes (`job_files`), so a rerun can tell whether the file changed. A missing file (HTTP 404, which a private repository also answers), a web page instead of the file, a body that does not parse, a timeout, or a file over the cap stops the run before anything starts, with the reason and what to change. Plain `http://` is refused except from this machine.

## lock.json

Every job folder has a `lock.json`, written before the first trial starts: the `bench` version and the git commit it was built from, the resolved run options (the names of `--env` variables, never their values), the dataset when there is one, the job files read with their digests, and for every trial the task (id, absolute path, digest, the registry's digest, format), the attempt (agent, model), the image (the reference pulled or the Dockerfile built, and a hash of everything that decides it), and the sandbox backend. `bench schema lock` prints its JSON Schema. A regrade's lock also names the job it regrades (`regrade_of`).

Each trial's task also has an inputs digest: a hash of what the agent works from and of everything that runs before its outputs are saved (the instruction, the image's files, the task's agent and sandbox settings, skills, mounts, services, hooks, and for the oracle its solution). Package paths are hashed by content, so moving a task folder changes neither digest. When the task digest changes and the inputs digest does not, only grading changed. The attempt records `--effort`, and the options record `--max-tool-calls` and `--max-tokens`, when they were given. A trial brought in from an earlier job names it (`carried_from`).

## Reusing earlier jobs

`bench run --diff <job>` compares the job about to run with one or more earlier jobs (`--diff` is repeatable; a job is a folder, or a name in the jobs folder) and prints, for each planned trial, what happens to it and why:

| Action | When |
|---|---|
| `reuse` | An earlier trial ran the same task (same digest) with the same agent, model, effort, agent install, and the settings that change the agent's run (sandbox backend and user, agent and idle timeouts, process limit, `--runtime`, `--max-tool-calls`, `--max-tokens`, `--extra-instruction`, `--skill`, `--load-trajectory`, `--resume-sessions`, the names of `--env` variables, and agent options), and it was scored. Its folder is copied into the new job |
| `regrade` | Everything on the agent's side matches (same inputs digest), but the task's tests or verifier changed, or the verifier timeouts did; the earlier trial was scored or unscored and saved its outputs, and the task is not graded in stages. Its outputs are graded again in a fresh sandbox with the new verifier; the agent does not run |
| `rerun` | Anything else. The reason names the first difference found (`the effort differs: low then, high now`), or says the earlier trial errored, saved no outputs, or is already used |
| `run` | No earlier job had a trial of this task; it runs for the first time (counted as `new` in the last line) |

```text
$ bench run tasks -a oracle -s daytona --diff oracle-a --dry-run
Reuse plan against 1 earlier job (nothing runs with --dry-run)
  rerun    3d-scan-calc                 the task changed what the agent works from (instruction,
                                        settings, skills, mounts, services, or hooks)
  regrade  citation-check               the task's tests or verifier changed; the agent's side is
                                        the same and its outputs were saved (from
                                        oracle-a/citation-check__46da99fb)
  reuse    econ-detrending-correlation  same task, agent, model, and settings; it was scored (from
                                        oracle-a/econ-detrending-correlation__46e9b35d)
  3 trials: 1 reuse · 1 regrade · 1 rerun
```

Without `--dry-run` the plan is printed and the job runs: reused trials are copied in (their `result.json` gets the new trial name and `carried_from`), regraded ones are graded from their saved outputs, and the rest run as usual. Each earlier trial is used at most once, so `-k 3` against three earlier jobs of one trial each can reuse all three. A job without a `lock.json` (BenchFlow 0.8 wrote none) or a regrade job cannot be compared and is skipped with the reason; when no `--diff` job can be read, nothing runs (exit 2). `--rerun failed` also keeps failed earlier trials from being reused. Trials saved with `--save-outputs` (or of tasks that declare outputs or use a separate verifier) are the ones that can be regraded. `--json --dry-run` prints the plan as one JSON document.

## Trace export

`bench export <job>` writes a job's trajectories in two forms. `--format hf` (the default for a job) writes a Hugging Face dataset folder: `data/train.jsonl` with one row per agent step and the conversation up to it (`--episodes last` keeps one row per trial), and a `README.md` dataset card that `datasets.load_dataset("<folder>")` reads, naming the job, its agents and models, the columns, and the trials left out with the reason. `--sharegpt` adds a `conversations_sharegpt` column in ShareGPT's `{from, value}` form, `--instruction` the trial's first user message (its first message when it has none), `--verifier-output` the verifier's stdout and stderr, and `--filter success|failure` keeps trials whose reward is above 0, or the rest. `--format otel` writes OpenTelemetry: one OTLP JSON trace per trial, one per line, with an `AGENT` span for the trial, a span per turn, an `LLM` span per agent step, a `TOOL` span per tool call, and subagents under the tool call that started them; attribute names are OpenInference's, and trace and span ids are derived from the trial's name, so exporting twice gives the same ids.

Each trial's trajectory is its ATIF document: `trainer/atif.json` when the trial has one (this engine writes it for every trial; BenchFlow 0.8 did in train mode), else `agent/trajectory.json`, else one rebuilt from `trajectory/acp_trajectory.jsonl`. Jobs from either engine export the same way. A job folder that has a `result.json` of its own next to its trial folders, as other harnesses write, is read as a job (by `bench results` and `bench export` alike), with each trial's rewards read from `verifier_result.rewards`, its agent and model from `agent_info`, and its error from `exception_info` when `result.json` has no `rewards`, `agent`, or `error`; `bench view` opens such trials too, reading their steps from `agent/trajectory.json`. A folder whose own `result.json` names a task is a trial, whatever its subfolders hold. Everything an export writes is redacted, each value once: every trajectory before rows or traces are built from it (another harness's `agent/trajectory.json` included), the verifier's output with `--verifier-output`, the trial's metadata columns, and the dataset card; and the `instruction` column is the first user message. On 61 jobs from both engines, every row and trace was checked against a reference exporter's output from the same trajectories (the resource names BenchFlow as the producer). A trial with images or audio is left out of the dataset, which is text only; a job with no trajectory at all is refused. Nothing is uploaded: publish the folder with Hugging Face's tools after reading its card, since the rows hold the tasks' instructions and files.

## Environment variables

| Variable | Meaning |
|---|---|
| `DAYTONA_API_KEY` | Daytona's API key; with it, `daytona` is a sandbox backend |
| `DAYTONA_API_URL`, `DAYTONA_TARGET` | Daytona's API endpoint and region |
| `DOCKER_HOST`, `DOCKER_CONTEXT` | Where the `docker` CLI sends work, as usual |
| `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY` | Claude Code's logins, in the order it prefers them (an API key wins and is billed) |
| `OPENAI_API_KEY`, `CODEX_API_KEY` | Codex's logins; `codex login` also works |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY` | Gemini CLI's logins |
| `BENCHFLOW_CACHE_DIR` | Where sources, logs, and caches go (default `~/.cache/benchflow`) |
| `BENCHFLOW_REWARD_LENIENT` | `1` takes a reward from a `reward.json`'s `score`, else `rewards`, when it has no `reward` and there is no `reward.txt`, as BenchFlow 0.8's lenient mode does; unset, such a file leaves the trial unscored (`R302`), as 0.8's default does. Each job's `evaluation.json` records the mode as `config.reward_lenient` |
| `NO_COLOR`, `CLICOLOR_FORCE` | Color off or forced |

Credentials are passed to the agent inside its sandbox and never written to job folders: `config.json` records the names of the variables an agent got, never their values.
