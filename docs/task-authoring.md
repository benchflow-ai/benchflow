# Authoring tasks
A BenchFlow task packages an instruction, a sandboxed environment, and a verifier into a directory that BenchFlow runs and scores automatically.

---

## Directory layout

> [!NOTE]
> Planned support: BenchFlow aims to add first-party support for Kaggle,
> Verifiers, and OpenReward Standard. These are not current task-runtime
> capabilities unless documented by a specific adapter or runtime capability.

For new BenchFlow-native task authoring, prefer the experimental unified
`task.md` entrypoint. It keeps Harbor-compatible task config in YAML
frontmatter and puts the agent-facing prompt, role prompts, scene prompts, and
simulated-user persona in the markdown body.

The broader draft standard for `task.md`, including `oracle/` and `verifier/`
native naming, compatibility export rules, assets, evidence, teams, nudges, and
the parsed-vs-runtime-supported capability matrix, lives in
[BenchFlow Task Package Standard](./task-standard.md).

```
my-task/
├── task.md                # config + prompt + optional roles/scenes/user
├── environment/
│   └── Dockerfile         # sandbox image
├── verifier/
│   └── test.sh            # verifier entry point
└── oracle/                # optional — reference/oracle solution
    └── solve.sh
```

`verifier/` may also include `test_outputs.py` (pytest module called by
`test.sh`). Harbor/Pier-style split tasks may use `tests/` and `solution/`;
BenchFlow keeps those names as compatibility inputs and export targets.

---

## task.md (experimental native format)

```md
---
schema_version: "1.3"

# Metadata is YAML, not TOML, in this file.
metadata:
  author_name: alice
  difficulty: easy
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  network_mode: no-network
  cpus: 1
  memory_mb: 2048
  storage_mb: 10240
  env:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
---

## prompt

Create the requested files in `/app`.

## role:solver

You are responsible for the implementation.

## scene:solve

Solve the task end to end.
```

Frontmatter accepts the same modeled task config fields as Harbor-style
`task.toml`, including `task`, `metadata`, `agent`, `verifier`, `environment`,
`oracle` (legacy alias: `solution`), `source`, `artifacts`, `steps`, and
`multi_step_reward_strategy`.
Unknown config keys fail validation; arbitrary labels belong under `metadata`.
BenchFlow-only document keys are `agents`, `scenes`, `user`, and the reserved
extension namespace `benchflow`. Fields under `benchflow:` are parsed as raw
document metadata today; the current runtime does not execute every proposed
extension yet.

Schema parity is ahead of runtime parity for some newer Harbor features. Fields
such as `steps`, separate verifier environments, `environment.os`, TPU specs,
healthchecks, workdir, artifact declarations, and network allowlists parse
today, but not all of those semantics are fully executed by the current
rollout runtime.

---

## task.toml

BenchFlow still supports [Harbor-format tasks](https://www.harborframework.com/docs/tasks)
with a `task.toml` config file and separate `instruction.md`.

```toml
version = "1.0"

[metadata]                   # optional, freeform
author_name = "alice"
difficulty  = "easy"         # easy / medium / hard
category    = "programming"
tags        = ["bash", "files"]

[agent]
timeout_sec = 300            # REQUIRED — seconds before agent is killed
# user = "agent"             # optional — run agent as this user/UID

[verifier]
timeout_sec = 120            # optional (default 600)

[environment]
cpus            = 1          # default 1
memory_mb       = 2048       # default 2048
storage_mb      = 10240      # default 10240
allow_internet  = false      # default true
env             = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }  # host vars to inject
```

**Service-backed tasks** — BenchFlow ships a small service registry for task-local APIs such as Gmail, Slack, Calendar, Docs, and Drive. The runner does not auto-start services just because a Dockerfile references a binary. For Python-driven runs, start services explicitly with `pre_agent_hooks=build_service_hooks([...])`; for CLI-only task authoring, keep services inside the task's own Dockerfile/startup scripts until a dedicated service declaration is wired through the CLI.

**Install tooling to shared prefixes, not `/root`** — when a task image ships Node.js, Python tools, or agent binaries that the sandbox user must execute, install them to `/usr/local/bin`, `/usr/local/lib`, or `/opt`, not `/root/.nvm` or `/root/.local/bin`. `setup_sandbox_user()` creates the non-root user, prepares small config/auth dirs, and chowns the workspace — it does not clone `/root` into the sandbox home. Legacy images that already install tools under `/root` still work via a narrow symlink fallback, but shared prefixes are the supported path. Pre-creating the sandbox user in the Dockerfile is an optional speedup, not a requirement.

---

## Multi-container tasks

A task may ship an `environment/docker-compose.yaml` alongside the
`Dockerfile`. The agent always runs in the `main` service; any additional
services you declare become sibling containers on the same Docker network.
This supports vulhub-style CVE tasks where the agent attacks a separate target
container over the network.

> `environment/Dockerfile` is always required — `bench tasks check` rejects
> a task that ships only a `docker-compose.yaml`. If your `main` service
> uses a prebuilt `image:` and needs no build context, still include a
> minimal `Dockerfile` (e.g. `FROM <same-image>`) so structural validation
> and other tooling agree on the task package shape.

```yaml
# environment/docker-compose.yaml
services:
  main: {}            # agent container — BenchFlow injects build/image/limits
  target:             # vulnerable service the agent must exploit
    image: vulhub/struts2-s2-001:latest
    expose: ["8080"]
```

`main` reaches `target` by service name (`http://target:8080`). The verifier
can inspect *target-side* state — not just the agent's workspace — by passing
a `service` argument when running commands:

```python
# In a Python-driven run or pre/post hook
await env.exec_in_service("target", "test -f /tmp/exploit_proof.txt")
await env.exec("cat /flag", service="target")          # equivalent form
services = await env.inner.services()                  # ["main", "target"]
```

`exec(..., service=...)` works on the Docker sandbox and the Daytona DinD
(compose) sandbox. Single-container backends (Modal, direct Daytona) raise a
clear error for any non-`main` service. This lets a verifier check
write-based oracles (`/tmp/exploit.txt` in the target), database modifications,
or RCE markers without trusting the agent container.

### Target-side `test.sh` verification

For tasks whose success oracle lives in a target container — an RCE marker
file, a modified database row — point the `test.sh` verifier at that service
with `[verifier].service`:

```toml
[verifier]
service = "target"     # run verifier/test.sh inside the `target` container
```

With this set, BenchFlow uploads the task's `verifier/` directory into the
**target** container, runs `test.sh` there, and copies the resulting
`reward.txt` / `reward.json` back to the host. Legacy `tests/` directories use
the same path. `service` defaults to `"main"` (the agent container), so
existing single-container tasks are unaffected.

`[verifier].service` is the declarative, task-schema way to do cross-container
verification; the `env.exec_in_service(...)` Python API above is the
imperative equivalent for hook-driven runs.

> Use the same `service` name you declared in `docker-compose.yaml`. A
> `test.sh` running in the target reaches `main` (and vice versa) by service
> name over the Docker network, just like the agent does.

### Hardening policy for multi-container tasks

BenchFlow's pre-verification hardening — killing the sandbox user's
processes, scrubbing `PATH`/`PYTHONPATH`, restoring build-config files —
applies **only to the `main` (agent) container**. Target containers are
deliberately left unhardened: a vulhub-style target is *meant* to be
vulnerable, the agent never has a shell inside it, and hardening it would
risk breaking the very vulnerability the task exercises. `[verifier].service`
selects where `test.sh` *runs*; it does not move hardening off `main`.

---

## instruction.md

The first prompt sent to the agent. Write it as you would for a skilled developer:

- State the precise goal in the first sentence.
- Name exact files or paths the agent must create or modify.
- Specify constraints (no external libraries, must pass existing tests, etc.).
- Don't mention the verifier or `reward.txt` — those are internal.

**Multi-turn prompts** — use a Scene with multiple Turns. A `None` prompt means "use `instruction.md`":

```python
from benchflow.rollout import RolloutConfig, Scene, Role, Turn

config = RolloutConfig(
    task_path="tasks/my-task",
    scenes=[Scene(
        roles=[Role("agent", "gemini", "gemini-3.1-flash-lite-preview")],
        turns=[
            Turn("agent"),                                        # instruction.md
            Turn("agent", "Review your solution and fix any test failures."),
        ],
    )],
    environment="daytona",
)
result = await bf.run(config)
```

---

## Verifier contract (verifier/test.sh)

After the agent finishes, the BenchFlow runtime copies `verifier/` to
`/verifier/` and runs `/verifier/test.sh`. Legacy `tests/` tasks are still
copied to `/tests/` and run from there. The working directory is the
Dockerfile's `WORKDIR` (typically `/app/` in the example Dockerfile below).

**Your script must write a single float (0.0–1.0) to `/logs/verifier/reward.txt`.** The verifier should write a fresh `reward.txt` or `reward.json` and exit `0`. Current runtime treats a nonzero verifier exit with no fresh reward file as infrastructure failure; if a fresh reward file exists, BenchFlow accepts the reward.

| Path | Contents |
|---|---|
| `/app/` | Agent's working directory |
| `/verifier/` | Your native `verifier/` directory |
| `/oracle/` | Native `oracle/` files (oracle runs only) |
| `/tests/` | Legacy `tests/` directory |
| `/solution/` | Legacy `solution/` files |
| `/logs/verifier/` | Write `reward.txt` (and optionally `ctrf.json`) here |

### Pure bash verifier

```bash
#!/bin/bash
REWARD=0
if [ -f /app/hello.txt ] && [ "$(cat /app/hello.txt | tr -d '\n')" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
```

### pytest verifier

```bash
#!/bin/bash
curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
source $HOME/.local/bin/env

uvx \
  --with pytest==8.4.1 \
  --with pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA

if [ $? -eq 0 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
```

### Partial credit

```bash
python3 -c "print($PASSED / $TOTAL)" > /logs/verifier/reward.txt
```

**Security:** don't let the agent write to `/logs/verifier/reward.txt` or modify `/verifier/test.sh`. For tasks running arbitrary code, use `allow_internet = false` and verify output files only. For LLM agent runs, BenchFlow preserves the network path needed for model APIs and agent startup, then disables supported agent web browsing/fetch tools through agent config or launch controls. Oracle runs still use the environment's network policy directly.

---

## oracle/ (optional)

Include when you want to verify the task is solvable or provide a reference implementation. When BenchFlow runs with `-a oracle`, it copies `oracle/` to `/oracle/` and runs `oracle/solve.sh` instead of an ACP agent. Legacy `solution/solve.sh` remains supported.

`solve.sh` has the same filesystem access as the agent — write only to `/app/`, not to `/logs/verifier/`.

```bash
#!/bin/bash
echo "Hello, world!" > /app/hello.txt
```

---

## CLI

```bash
# Scaffold a new task from scratch
bench tasks init my-task
bench tasks init my-task --no-pytest --no-oracle
bench tasks init my-split-task --format legacy

# Mirror an existing split task into task.md
bench tasks migrate tasks/my-split-task/
bench tasks migrate tasks/my-split-task/ --remove-legacy

# Generate tasks from agent traces (personal benchmark curation)
bench tasks generate --from-local                          # from local Claude Code sessions
bench tasks generate --from-file session.jsonl --dry-run    # from a JSONL trace file
bench tasks generate --from-hf opentraces-test --limit 50   # from a HuggingFace dataset
bench tasks generate --from-local --task-format legacy      # Harbor/Pier split layout
bench tasks list-sources                                    # list known HF trace datasets

# Validate structure
bench tasks check tasks/my-task/

# Confirm oracle gets reward = 1.0
bench eval create --tasks-dir tasks/my-task/ --agent oracle --sandbox docker

# Run a real agent
bench eval create --tasks-dir tasks/my-task/ --agent gemini --sandbox daytona

# Run with task-local skills mounted
bench eval create \
  --tasks-dir tasks/my-task/ \
  --agent gemini \
  --sandbox daytona \
  --skill-mode with-skill \
  --agent-env BENCHFLOW_SKILL_NUDGE=name
```

Task-local skills are mounted through the selected agent's native skill paths.
See [Architecture: skill loading](./architecture.md#skill-loading) for the
canonical loading semantics and nudge modes.

`bench tasks migrate` mirrors a split `task.toml` + `instruction.md` pair into
`task.md`, verifies that the parsed config and prompt round-trip, and keeps the
split files unless `--remove-legacy` is provided.

`bench tasks generate` converts agent traces (Claude Code sessions, opentraces records, or HuggingFace datasets) into native `task.md` directories with `verifier/test.sh` and `oracle/solve.sh`. Use `--task-format legacy` for the Harbor/Pier split layout and `--dry-run` to preview traces before generating. See [CLI reference](./reference/cli.md#bench-tasks-generate) for all flags.

`bench tasks check` validates either `task.md` or the split `task.toml` +
`instruction.md` pair, plus `environment/Dockerfile` and non-empty `verifier/`
or compatibility `tests/`. It exits with code 1 on failure (CI-friendly).

---

## Worked example — write-fizzbuzz

```toml
# task.toml
version = "1.0"
[metadata]
difficulty = "easy"
tags = ["python"]
[agent]
timeout_sec = 180
[verifier]
timeout_sec = 60
```

```markdown
# instruction.md
Write a file `fizzbuzz.py` defining:

    def fizzbuzz(n: int) -> str

Return "FizzBuzz" / "Fizz" / "Buzz" / str(n) for divisibility by 15 / 3 / 5 / none.
No __main__ block, no print statements.
```

```dockerfile
# environment/Dockerfile
FROM ubuntu:24.04
RUN apt-get update -qq && apt-get install -y -qq python3 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
```

```python
# verifier/test_outputs.py
import importlib.util
from pathlib import Path

def _load():
    path = Path("/app/fizzbuzz.py")
    assert path.exists()
    spec = importlib.util.spec_from_file_location("fizzbuzz", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fizzbuzz

def test_fizz():    assert _load()(3) == "Fizz"
def test_buzz():    assert _load()(5) == "Buzz"
def test_fizzbuzz():assert _load()(15) == "FizzBuzz"
def test_number():  assert _load()(7) == "7"
```

```bash
# oracle/solve.sh
cat > /app/fizzbuzz.py << 'EOF'
def fizzbuzz(n: int) -> str:
    if n % 15 == 0: return "FizzBuzz"
    if n % 3 == 0:  return "Fizz"
    if n % 5 == 0:  return "Buzz"
    return str(n)
EOF
```
