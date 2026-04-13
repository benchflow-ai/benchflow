# Task Authoring Guide

A BenchFlow task packages an instruction, a sandboxed environment, and a verifier into a directory that BenchFlow runs and scores automatically.

---

## Directory layout

```
my-task/
├── task.toml              # timeouts, resources, metadata
├── instruction.md         # what the agent must do
├── environment/
│   └── Dockerfile         # sandbox image
├── tests/
│   └── test.sh            # verifier entry point
└── solution/              # optional — reference/oracle solution
    └── solve.sh
```

`tests/` may also include `test_outputs.py` (pytest module called by `test.sh`).

---

## task.toml

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

**Built-in mock services** — if the Dockerfile references a service binary (`claw-gmail`, `claw-slack`, `claw-gcal`, `claw-gdoc`, `claw-gdrive`), BenchFlow starts it automatically. No `[services]` section needed.

---

## instruction.md

The first prompt sent to the agent. Write it as you would for a skilled developer:

- State the precise goal in the first sentence.
- Name exact files or paths the agent must create or modify.
- Specify constraints (no external libraries, must pass existing tests, etc.).
- Don't mention the verifier or `reward.txt` — those are internal.

**Multi-turn prompts** — pass `prompts=` to `SDK.run()` to add follow-up turns. `None` means "use `instruction.md`":

```python
result = await sdk.run(
    task_path="tasks/my-task",
    agent="claude-agent-acp",
    prompts=[None, "Review your solution and fix any test failures."],
)
```

---

## Verifier contract (tests/test.sh)

After the agent finishes, Harbor copies `tests/` to `/tests/` and runs `/tests/test.sh`. The working directory is the Dockerfile's `WORKDIR` (typically `/app/` in the example Dockerfile below).

**Your script must write a single float (0.0–1.0) to `/logs/verifier/reward.txt`.**

| Path | Contents |
|---|---|
| `/app/` | Agent's working directory |
| `/tests/` | Your `tests/` directory |
| `/solution/` | `solution/` (oracle runs only) |
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
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

if [ $? -eq 0 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
```

### Partial credit

```bash
python3 -c "print($PASSED / $TOTAL)" > /logs/verifier/reward.txt
```

**Security:** don't let the agent write to `/logs/verifier/reward.txt` or modify `/tests/test.sh`. For tasks running arbitrary code, use `allow_internet = false` and verify output files only.

---

## solution/ (optional)

Include when you want to verify the task is solvable or provide a reference implementation. When BenchFlow runs with `-a oracle`, it copies `solution/` to `/solution/` and runs `solution/solve.sh` instead of an ACP agent.

`solve.sh` has the same filesystem access as the agent — write only to `/app/`, not to `/logs/verifier/`.

```bash
#!/bin/bash
echo "Hello, world!" > /app/hello.txt
```

---

## CLI

```bash
# Scaffold a new task
benchflow tasks init my-task
benchflow tasks init my-task --no-pytest --no-solution

# Validate structure
benchflow tasks check tasks/my-task/

# Confirm oracle gets reward = 1.0
benchflow run -t tasks/my-task/ -a oracle -e docker

# Run a real agent
benchflow run -t tasks/my-task/ -a claude-agent-acp -e docker
```

`benchflow tasks check` validates that `task.toml`, `instruction.md` (non-empty), `environment/Dockerfile`, and `tests/` (non-empty) all exist, and that `[agent].timeout_sec` is set. Exits with code 1 on failure (CI-friendly).

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
# tests/test_outputs.py
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
# solution/solve.sh
cat > /app/fizzbuzz.py << 'EOF'
def fizzbuzz(n: int) -> str:
    if n % 15 == 0: return "FizzBuzz"
    if n % 3 == 0:  return "Fizz"
    if n % 5 == 0:  return "Buzz"
    return str(n)
EOF
```
