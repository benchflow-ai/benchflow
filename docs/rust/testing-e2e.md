# End-to-end tests of the `bench` binary

`e2e/` drives a built `bench` from outside and checks what it does: exit codes, key lines of output, the files a job writes, and each trial's status, reward, failure code, and owner. It never looks at how the binary is written, so the same scenarios can gate any build of `bench`, on any backend. The unit and integration tests under `crates/` check the parts; this suite checks the whole command as a user runs it.

## Running it

```bash
cargo build --release -p benchflow-cli

# Everything that needs no sandbox (no Docker, no keys): seconds
e2e/run.sh --bench target/release/bench --backend none

# The smoke tier on Daytona: one sandbox scenario per area, about 10 minutes
export DAYTONA_API_KEY=...        # and DAYTONA_API_URL, DAYTONA_TARGET if not the defaults
e2e/run.sh --bench target/release/bench --backend daytona --tier smoke

# The full tier on Docker (DOCKER_HOST is honored, as by bench itself): about 12 minutes
e2e/run.sh --bench target/release/bench --backend docker --tier full

# Only some scenarios, keeping their folders for inspection
e2e/run.sh --bench target/release/bench --backend docker -k regrade -k skills --keep
```

`e2e/run.sh` runs `e2e/run.py`, which needs Python 3.11 or newer and nothing else (`curl` for the viewer scenario). The options:

| Option | Meaning |
|---|---|
| `--bench <binary>` | The binary to test (required) |
| `--backend none\|docker\|daytona` | The sandbox backend for scenarios that need one; `none` (the default) runs only the scenarios that start no sandbox |
| `--tier smoke\|full` | `smoke` (the default) is the quick set; `full` adds every scenario |
| `-k <text>` | Only scenarios whose name contains this (repeatable) |
| `--max-sandboxes <N>` | Sandbox scenarios run one at a time, each with at most N trials at once (`{j}`; default 3) |
| `-j <N>` | Scenarios without sandboxes run N at a time (default 4) |
| `--owner <name>` | The base of the suite's owner labels (default `e2e-<user>-<random>`, new each run). It must start with `e2e-` and be a label bench keeps as it is (letters, digits, `.`, `_`, `-`; at most 40 characters): the runner removes what carries it |
| `--require-sandbox` | Fail when no scenario that starts a sandbox ran (CI uses it, so a gate that skipped them all is red) |
| `--allow <need>` | Run scenarios with an extra need (none today) |
| `--gateway` | The gateway key reaches `bench` some other way; run the gateway scenarios |
| `--judge-model <provider>/<model>` | The model for the model-judged rubric and model-agent scenarios (default `vercel/deepseek/deepseek-v4-flash-0731`) |
| `--work <dir>` | Where each scenario's folder goes; a failed scenario's folder is kept, with every step's stdout and stderr |
| `--keep` | Keep every scenario's folder |
| `--json <file>` | Also write the results as JSON |

Each line of the report is one scenario: `✓` passed, `✗` failed (with the commands it ran and why), `~` failed as a known limit the scenario records, `!` passed although it is marked as a known limit (remove the mark). Skipped scenarios say why: another tier, another backend, or a missing key. The exit code is 1 when any scenario failed or passed unexpectedly.

Every step gets an owner of its own: the run's base plus a counter (`e2e-<user>-<random>-<n>`; an ambient `BENCHFLOW_OWNER` is not inherited). `bench` sweeps its owner's leftovers when it starts, so with one owner per scenario a later step would quietly remove what an earlier one left; with one per step, the runner's check after each step sees exactly what that step left. After each step of a sandbox scenario the runner lists the containers, networks, or Daytona sandboxes carrying that step's owner and this machine's host label, fails the step if anything is there (on Daytona after waiting up to a minute, since it deletes a few seconds after `bench` asks), and removes it. Nothing another machine made is ever listed or removed. When a run starts, it also removes what earlier runs of this suite by the same user left on this machine whose `bench` process is gone (a run stopped by force). A step that runs past its timeout, or a suite stopped with Ctrl-C or SIGTERM, sends SIGTERM to the running steps' whole process groups and waits until every process in them is gone, `bench` under `sh` included, then SIGKILL 150 s later (`bench` waits up to 120 s for sandboxes still being created); after an interrupt the runner removes what the running steps' owners left. A scenario that leaves a crash log in its cache folder fails, whatever its exit codes. Steps get the caller's environment without `BENCHFLOW_*`, `CLAUDE_CONFIG_DIR`, and `CODEX_HOME`, so a redirected endpoint or the developer's own Claude folder cannot change what a step does. Each scenario gets its own empty folder and its own cache folder (`BENCHFLOW_CACHE_DIR`), so runs do not see each other. A scenario file with a key the runner does not know is an error, so a typo cannot turn a check off.

### Keys and cost

| Needs | Scenarios | Where it comes from |
|---|---|---|
| a sandbox | every scenario that runs a trial | Docker (or `DOCKER_HOST`), or `DAYTONA_API_KEY` |
| `gateway` | `judge` (the rubric's model judge), `model-agent` (the builtin loop on a model), `claude-sessions` (Claude Code on the gateway's model, no Claude login) | `AI_GATEWAY_API_KEY`, or `~/.config/benchflow/vercel-ai-gateway.env`; a full run spends about $0.02 |

GitHub's `rust` workflow runs `--backend none --tier full` on Linux and macOS after `cargo test`, with the debug build. Two jobs run the sandbox scenarios on `ubuntu-latest`'s own Docker, with no secret: `e2e smoke (docker)` (`--tier smoke --require-sandbox`: `bench run`, `bench test`, `doctor`, the failure paths, mounts and hooks) on every pull request and push that touches the code, and `e2e full (docker)` (`--tier full --require-sandbox`: also the network policies, every probe, the image-hole defenses, services, and the shell) on pushes to `main` and `bench-rs`, nightly, and on manual runs; the full tier takes about 12 minutes on a warm machine and longer on a fresh runner, which is why it is not run on every pull request. The model and Daytona scenarios skip themselves there: they need secrets a pull request from a fork must not get. Branches reach `bench-rs` by a merge that one person makes, not by pull requests, so these jobs run after the merge, by design: they report a broken `bench` on `bench-rs` rather than block it. Run the full tier on Docker (and on Daytona when the change touches it) before merging.

## What it covers

| Area | Scenarios |
|---|---|
| Commands without a sandbox | `cli-basics` (version, help, usage errors, `-C`), `new`, `check` (codes with positions, `--fix`, `--strict`, `--schema`, `--json`), `explain-schema`, `agents`, `import-export` (split layout, 0.8 task.md, Terminal-Bench v1, `--allow-degraded`, round trip), `job-reports` (`results` with one and two jobs, `--json`, `-t`; `export` as a dataset and as OpenTelemetry), `view` (every page and its `/api/` answer), `handoff` (`--dry-run`, `--trust`), `cache`, `run-config` (job files in TOML, YAML, and JSON merged; `--print-config`; `--dry-run`; flags refused before anything starts: `--ak`, `--rerun`, `--load-trajectory`, `--skill`, `--diff`, `--judge-model`, `--fail-under`, a job file over plain HTTP, a missing dataset) |
| `bench run` | `run-basics` (oracle and nop rewards, the files a job writes, resuming, `-k`, `--fail-under`, variable values kept out of the job), `failure-paths`, `regrade-diff` (`--diff` reuse, regrade, and rerun; `bench regrade`; `--rerun failed`), `dataset` (a local `dataset.toml`, the public registry), `skills` (scripted attempts get none and record why; `--no-task-skills`), `judge`, `model-agent` (`-a builtin` with `--ak` and `--max-tool-calls`), `claude-sessions` (skills placed where Claude Code looks, `--resume-sessions` across two stages, `handoff --dry-run` on a real trial, `--load-trajectory` from its saved session) |
| `bench test` | `test-command` (oracle, nop, controls; a verifier that says yes to nothing; `--list-probes`), `test-probes` (every exploit probe) |
| Task features | `task-stages`, `task-separate-verifier`, `services` (Compose; Docker only, refused by name on Daytona), `task-mounts-hooks-outputs` (mounts, a task.md snapshot hook, declared outputs as artifacts), `task-hooks-split` (a split-layout collect hook), `task-network-none`, `task-network-none-edges` (a separate verifier with no network on an image without iptables, a task's own service on loopback port 53, a root agent), `task-network-allow`, `task-network-block`, `task-split-layout`, `task-format-v06`, and images that try the runner's defenses (`task-writable-bin`, `task-writable-system`, `task-ships-solution`, `task-leaves-credentials`) |
| Machine | `doctor` (the backend, `--json`, `--confinement`), `shell` (Docker only) |

The failure paths it checks, with the code and owner each trial must end with:

| Task (`e2e/tasks/`) | What goes wrong | Expected |
|---|---|---|
| `broken-image` | The Dockerfile fails | errored, `R101`, the task |
| `no-reward` | The verifier writes no reward | unscored, `R301`, the task |
| `verifier-timeout` | The verifier runs past its timeout | unscored, `R303`, the task |
| `agent-timeout` | The agent runs past its timeout | scored, reward 0, `timed_out` |
| `bad-oracle` | The solution exits non-zero and leaves nothing | scored, reward 0 |
| `judged` with a judge that cannot answer | The judge's provider refuses the scenario's fake key | unscored, `R312` |
| `sidecar` on Daytona | Services on a backend without them | errored, `R401`, you |
| `none-root` on Daytona | A root agent with no network, where Daytona's block leaves IPv6 open | errored, `R108`, you (Docker runs it: reward 1) |

## Writing a scenario

One TOML file per feature in `e2e/scenarios/`. Tasks come from `e2e/tasks/` or the repository's fixtures (`crates/benchflow-run/tests/fixtures/tasks/`) and are copied into the scenario's folder as `tasks/<name>`. Steps run in order in that folder; the first one that fails ends the scenario.

```toml
# One line on what this checks.
tier = "smoke"                 # or "full"
needs = ["sandbox"]            # and/or "gateway"
backends = ["docker"]          # only where it can run (default: all)
tasks = ["hello"]              # copied into tasks/
limit = "..."                  # a known limit: failing is expected and reported as such

[[step]]
name = "the oracle passes"
run = ["run", "tasks/hello", "-s", "{backend}", "-j", "{j}", "--job", "o"]   # bench's arguments
exit = 0                       # or a list of codes; a step without `exit` must exit 0
stdout_has = ['1/1 passed']    # regular expressions, one line of output each, never exact prose
output_has = []                # stdout or stderr
output_lacks = []
files = ["jobs/o/lock.json"]   # globs that must exist (absent = [...] must not)
json = [{ file = "jobs/o/summary.json", passed = 1 }]
trials = [{ job = "o", task = "hello", status = "scored", reward = 1.0, code = "R301", owner = "task",
            result = { "defenses.network_policy.state" = "enforced" }, files = ["verifier/reward.txt"] }]

[[step]]
sh = "echo '# changed' >> tasks/hello/verifier/test.sh"   # a shell step, for setup or a check
only_backends = ["docker"]     # skip this step elsewhere
```

Placeholders: `{bench}`, `{backend}`, `{j}`, `{work}` (the scenario's folder), `{tasks}`, `{jobs}`, `{e2e}`, `{repo}`, `{judge}`. In `json`, `stdout_json`, and `result`, a key is a dotted path (`trials.0.status`) and a value is either the value expected or a table of `min`, `max`, `in`, `not`, `has`, `matches`, `present`, or `len`. In `trials`, `job` may be a glob (`tests/*` for `bench test`'s job), and `agent` picks trials by agent (`oracle`, `nop`, `control:near-miss`).

A scenario that fails is either a bug in `bench`, fixed with a test in the crate it belongs to, or a limit, recorded with `limit = "..."` and the reason, so the report keeps showing it.

## Limits recorded

- `--resume-sessions`, `--load-trajectory`, and `handoff` run for real only with Claude Code (`claude-sessions`, on the gateway's key); Codex's sessions are checked only up to the run's plan.
- `services` and `shell` run only on Docker; on Daytona `services` checks the refusal instead.
- On Daytona, `none` needs BenchFlow's firewall for IPv6 unless the sandbox's kernel has no IPv6 at all (every Daytona sandbox seen so far has at least IPv6 loopback): an image that cannot get iptables and ip6tables, or a root agent, is refused there (`R108`, owner you) though Docker runs it.
- On a Docker daemon other tools share, a scenario can fail if something else removes an image a trial is about to start from (seen once, as R103 "Unable to find image", in `failure-paths`; not reproduced). Rerun the scenario before suspecting `bench`.
- The stored jobs in `e2e/jobs/` (one oracle and one nop job from a Daytona run) let `results`, `view`, and `export` run without a sandbox; regenerate them when a job's files change shape.
