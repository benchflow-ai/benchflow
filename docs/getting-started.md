# Getting Started with BenchFlow

## 1. Installation

**Requirements:** Python 3.12+, Docker (local) or a [Daytona](https://app.daytona.io) account (cloud).

```bash
pip install benchflow
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
benchflow agents   # list registered agents, required env vars, and protocol
```

---

## 2. Run a single task

```bash
benchflow run \
  --task-dir path/to/task \
  --agent claude-agent-acp \
  --model claude-haiku-4-5-20251001 \
  --jobs-dir jobs/
```

Key flags:
- `--task-dir` (`-t`) — directory containing `task.toml` and `instruction.md`
- `--agent` (`-a`) — agent name from the registry (default: `claude-agent-acp`)
- `--model` (`-m`) — model ID; omit to use the agent's default
- `--env` (`-e`) — `docker` (local, default) or `daytona` (cloud)
- `--jobs-dir` (`-o`) — where results are written (default: `jobs/`)

**Output**

```
Task:       fix-login-bug
Agent:      claude-agent-acp
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
benchflow view jobs/my-job/my-trial/   # starts HTTP server at localhost:8888 — open in browser
```

**Try it with a real task**

Download tasks once (clones into `.ref/`):

```bash
python -c "from benchflow.task_download import ensure_tasks; ensure_tasks('skillsbench'); ensure_tasks('terminal-bench-2')"
```

```bash
benchflow run --task-dir .ref/skillsbench/tasks/court-form-filling --agent claude-agent-acp --model claude-haiku-4-5-20251001
benchflow run --task-dir .ref/terminal-bench-2/fix-git --agent claude-agent-acp --model claude-haiku-4-5-20251001
```

---

## 3. Run a job (SkillsBench)

`benchflow job` runs an entire task directory with configurable concurrency and retries.

The repo ships ready-to-run configs in `benchmarks/` targeting Daytona (`concurrency: 8`). To run locally with Docker, use `concurrency: 1`.

**Local Docker config (`my-skillsbench.yaml`)**

```yaml
tasks_dir: .ref/skillsbench/tasks
jobs_dir: jobs/skillsbench-local
agent: claude-agent-acp
model: claude-haiku-4-5-20251001
environment: docker
concurrency: 1
max_retries: 2
exclude:
  - scheduling-email-assistant
  - mhc-layer-impl
```

```bash
benchflow job --config my-skillsbench.yaml
```

Or override a shipped config inline:

```bash
benchflow job --config benchmarks/skillsbench-codex-gpt54.yaml --env docker --concurrency 1
```

Or use inline flags directly:

```bash
benchflow job \
  --tasks-dir .ref/skillsbench/tasks \
  --agent claude-agent-acp \
  --model claude-haiku-4-5-20251001 \
  --concurrency 1 \
  --retries 2 \
  --jobs-dir jobs/skillsbench-local
```

**Output and metrics**

```
Score: 32/86 (37.2%), errors=2
```

```bash
benchflow metrics jobs/skillsbench-local/       # tabular summary
benchflow metrics jobs/skillsbench-local/ --json  # machine-readable
```

**Resume:** Re-running the same command resumes automatically. Tasks with an existing `result.json` are skipped.

---

## 4. Run a job (TB2)

Terminal-Bench 2 ships single-turn and multi-turn variants, driven by the `prompts` field in the YAML.

```bash
python benchmarks/run_tb2.py benchmarks/tb2_single-codex-gpt54.yaml     # single-turn, Daytona
python benchmarks/run_tb2.py benchmarks/tb2_multiturn-codex-gpt54.yaml  # multi-turn, Daytona
```

For local Docker, override inline:

```bash
benchflow job --config benchmarks/tb2_single-codex-gpt54.yaml --env docker --concurrency 1
benchflow job --config benchmarks/tb2_multiturn-codex-gpt54.yaml --env docker --concurrency 1
```

Or write your own local config. Multi-turn adds a `prompts` list:

```yaml
tasks_dir: .ref/terminal-bench-2
jobs_dir: jobs/tb2_multiturn-local
agent: claude-agent-acp
model: claude-haiku-4-5-20251001
environment: docker
concurrency: 1
max_retries: 2
prompts:
  - null                # expands to task's instruction.md
  - "Review your solution. Check for errors, test it, and fix any issues."
```

The agent retains full context (tool history, memory) between turns.

---

## 5. Multi-turn via the Python SDK

**Single task**

```python
import asyncio
from benchflow import SDK

async def main():
    sdk = SDK()
    result = await sdk.run(
        task_path="path/to/task",
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        environment="docker",
    )
    print(result.rewards)      # {"reward": 1.0}
    print(result.n_tool_calls) # 17
    print(result.error)        # None on success

asyncio.run(main())
```

**Multi-turn**

```python
result = await sdk.run(
    task_path="path/to/task",
    agent="claude-agent-acp",
    prompts=[
        None,
        "Review your solution. Check for errors, test it, and fix any issues.",
    ],
    environment="docker",
)
```

**Full benchmark job**

```python
from benchflow import SDK, Job, JobConfig, collect_metrics

async def main():
    job = Job(
        tasks_dir="path/to/tasks",
        jobs_dir="jobs/tb2",
        config=JobConfig(
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            environment="docker",
            concurrency=1,
            prompts=[None, "Review your solution. Check for errors, test it, and fix any issues."],
        ),
    )
    result = await job.run()
    print(f"{result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

**Load from YAML**

```python
job = Job.from_yaml("benchmarks/tb2_multiturn-codex-gpt54.yaml")
result = await job.run()
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
