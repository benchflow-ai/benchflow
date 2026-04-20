# Getting Started with BenchFlow

## 1. Installation

**Requirements:** Python 3.12+, Docker (local) or a [Daytona](https://app.daytona.io) account (cloud).

```bash
pip install benchflow==0.3.0a3
```

**Credentials**

BenchFlow detects subscription auth and copies it into the sandbox automatically — no API keys needed if you're already logged in:

| Agent | Login command | Credential file |
|-------|--------------|-----------------|
| `claude-agent-acp` | `claude login` | `~/.claude/.credentials.json` |
| `codex-acp` | `codex --login` | `~/.codex/auth.json` |
| `gemini` | `gemini auth login` | `~/.gemini/oauth_creds.json` |

API keys override subscription auth when both are present:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # claude-agent-acp
export OPENAI_API_KEY=sk-...          # codex-acp
export GOOGLE_API_KEY=...             # gemini
export DAYTONA_API_KEY=...            # required for daytona environment only
```

```bash
bench agent list   # list registered agents, required env vars, and protocol
```

---

## 2. Run a single task

```bash
bench eval create \
  -t path/to/task \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e daytona
```

Key flags:
- `--tasks-dir` (`-t`) — directory containing `task.toml` and `instruction.md`
- `--agent` (`-a`) — agent name from the registry (default: `gemini`)
- `--model` (`-m`) — model ID (default: `gemini-3.1-flash-lite-preview`)
- `--env` (`-e`) — `daytona` (cloud, default for Daytona users) or `docker` (local)
- `--jobs-dir` (`-o`) — where results are written (default: `jobs/`)

**Output**

```
Task:       fix-login-bug
Agent:      gemini (gemini-3.1-flash-lite-preview)
Rewards:    {'reward': 1.0}
Tool calls: 14
```

**Output directory structure**

```
jobs/{job_name}/{trial_name}/
├── config.json              # parameters used (secrets filtered)
├── result.json              # rewards, n_tool_calls, timing, error
├── timing.json              # per-phase timing
├── prompts.json             # prompts sent to the agent
├── agent/
│   ├── install-stdout.txt
│   └── claude_agent_acp.txt
├── trajectory/
│   └── acp_trajectory.jsonl # tool calls, messages, thoughts
└── verifier/
    ├── reward.txt
    └── test-stdout.txt
```

**Inspect a trajectory**

```bash
bench eval list jobs/my-job/   # show results for a job
```

**Try it with a real task**

Tasks auto-download on first run via `ensure_tasks()`. Or download explicitly:

```bash
python -c "from benchflow.task_download import ensure_tasks; ensure_tasks('terminal-bench-2')"
```

```bash
bench eval create -t .ref/terminal-bench-2/fix-git -a gemini -e daytona
bench eval create -t .ref/skillsbench/tasks/court-form-filling -a gemini -e daytona
```

---

## 3. Run a batch evaluation (SkillsBench)

`bench eval create` runs an entire task directory with configurable concurrency and retries.

The repo ships ready-to-run configs in `benchmarks/` targeting Daytona. To run locally with Docker, override with `--env docker --concurrency 1`.

**Local Docker config (`my-skillsbench.yaml`)**

```yaml
task_dir: .ref/skillsbench/tasks
jobs_dir: jobs/skillsbench-local
agent: gemini
model: gemini-3.1-flash-lite-preview
environment: docker
concurrency: 1
max_retries: 2
exclude:
  - scheduling-email-assistant
  - mhc-layer-impl
```

```bash
bench eval create -f my-skillsbench.yaml
```

Or override a shipped config inline:

```bash
bench eval create -f benchmarks/skillsbench-gemini.yaml --env docker --concurrency 1
```

Or use inline flags directly:

```bash
bench eval create \
  -t .ref/skillsbench/tasks \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -c 1 --retries 2 \
  -o jobs/skillsbench-local
```

**Output and metrics**

```
Score: 32/86 (37.2%), errors=2
```

```bash
bench eval list jobs/skillsbench-local/       # tabular summary
bench eval list jobs/skillsbench-local/ --json  # machine-readable
```

**Resume:** Re-running the same command resumes automatically. Tasks with an existing `result.json` are skipped.

---

## 4. Run a batch evaluation (TB2)

Terminal-Bench 2 tasks auto-download via `ensure_tasks("terminal-bench-2")`.

```bash
# Single-turn, Daytona
bench eval create -f benchmarks/tb2-gemini-baseline.yaml

# Multi-turn with scenes
bench eval create -f benchmarks/tb2-gemini-multiturn.yaml
```

For local Docker, override inline:

```bash
bench eval create -f benchmarks/tb2-gemini-baseline.yaml --env docker --concurrency 1
```

Or write your own config. Multi-turn uses a Scene with multiple Turns:

```yaml
task_dir: .ref/terminal-bench-2
environment: daytona
concurrency: 64
scenes:
  - name: solve-and-review
    roles:
      - name: solver
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: solver            # uses instruction.md
      - role: solver
        prompt: "Review your solution. Check for errors, test it, and fix any issues."
```

The agent retains full context (tool history, memory) between turns within a Scene.

---

## 5. Python SDK

**Single task**

```python
import asyncio
import benchflow as bf

async def main():
    result = await bf.run("gemini", task_path="path/to/task")
    print(result.rewards)      # {"reward": 1.0}
    print(result.n_tool_calls) # 17
    print(result.error)        # None on success

asyncio.run(main())
```

**With explicit TrialConfig**

```python
from benchflow.trial import TrialConfig

result = await bf.run(TrialConfig(
    agent="gemini",
    model="gemini-3.1-flash-lite-preview",
    task_path="path/to/task",
    backend="daytona",
))
```

**Multi-agent (coder + reviewer)**

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

config = TrialConfig(
    task_path="path/to/task",
    scenes=[
        Scene(name="review-loop",
              roles=[
                  Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
                  Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
              ],
              turns=[
                  Turn("coder"),
                  Turn("reviewer", "Review the solution. Write feedback."),
                  Turn("coder", "Address the reviewer's feedback."),
              ]),
    ],
    backend="daytona",
)
result = await bf.run(config)
```

**Explicit Trial lifecycle**

```python
from benchflow.trial import Trial, TrialConfig

config = TrialConfig(...)
trial = await Trial.create(config)
result = await trial.run()
```

**Load from YAML**

```python
result = await bf.run(TrialConfig.from_yaml("benchmarks/tb2-gemini-baseline.yaml"))
```

**Aggregate metrics**

```python
from benchflow import collect_metrics
metrics = collect_metrics("jobs/tb2", benchmark="TB2")
print(metrics.summary())
```

**`RunResult` fields**

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Task directory name |
| `agent` | `str` | Harness registry name (e.g. `"claude-agent-acp"`) |
| `agent_name` | `str` | ACP-reported agent name |
| `rewards` | `dict \| None` | Verifier output, e.g. `{"reward": 1.0}` |
| `trajectory` | `list[dict]` | Tool calls, messages, thoughts |
| `n_tool_calls` | `int` | Number of tool calls made |
| `n_prompts` | `int` | Number of prompts sent |
| `error` | `str \| None` | Agent error, or `None` on success |
| `verifier_error` | `str \| None` | Verifier error, or `None` on success |
| `success` | `bool` | `True` when both `error` and `verifier_error` are `None` |
| `partial_trajectory` | `bool` | `True` when trajectory was salvaged from a crashed session |
| `trajectory_source` | `str \| None` | `"acp"` (trusted), `"scraped"`, or `"partial_acp"` |
| `trial_name` | `str` | Unique trial identifier within a job run |
| `started_at` | `datetime \| None` | Wall-clock start time |
| `finished_at` | `datetime \| None` | Wall-clock end time |

---

## 6. Next steps

- **Create your own tasks** — [task-authoring.md](task-authoring.md)
- **Full CLI reference** — [cli-reference.md](cli-reference.md)
- **How it works** — [architecture.md](architecture.md)
