# benchflow CLI Reference

---

## benchflow run

Run a single task with an ACP agent.

| Flag | Default | Description |
|------|---------|-------------|
| `--task-dir`, `-t` | *(required)* | Task directory containing `task.toml` and `instruction.md` |
| `--agent`, `-a` | `claude-agent-acp` | Agent name from the registry |
| `--model`, `-m` | *(agent default)* | Model ID (e.g. `claude-haiku-4-5-20251001`) |
| `--env`, `-e` | `docker` | `docker` or `daytona` |
| `--prompt`, `-p` | *(instruction.md)* | Prompt(s) to send instead of the default instruction (repeatable) |
| `--jobs-dir`, `-o` | `jobs` | Output directory |
| `--ae` | — | Agent env var in `KEY=VALUE` form (repeatable) |
| `--skills-dir`, `-s` | — | Skills directory to deploy into the sandbox |
| `--sandbox-user` | `agent` | Run agent as this user; `none` for root |

```bash
benchflow run \
  --task-dir tasks/fix-login-bug \
  --agent claude-agent-acp \
  --model claude-sonnet-4-5 \
  --jobs-dir jobs/
```

Output: task name, agent, rewards dict, tool-call count. Exits non-zero on error.

```
Task:       fix-login-bug
Agent:      claude-agent-acp
Rewards:    {'reward': 1.0}
Tool calls: 14
```

---

## benchflow job

Run all tasks in a directory with concurrency and retries. Use `--config` (YAML) or `--tasks-dir` with inline flags — mutually exclusive.

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks-dir`, `-t` | — | Directory of task subdirectories |
| `--config`, `-f` | — | YAML config file (benchflow or Harbor format) |
| `--agent`, `-a` | `claude-agent-acp` | Agent name |
| `--model`, `-m` | `claude-haiku-4-5-20251001` | Model ID |
| `--env`, `-e` | `docker` | `docker` or `daytona` |
| `--concurrency`, `-c` | `4` | Max parallel tasks |
| `--retries` | `0` | Max retries per failed task |
| `--jobs-dir`, `-o` | `jobs` | Output directory |
| `--skills-dir`, `-s` | — | Skills directory to deploy into each sandbox |

```bash
# Inline flags
benchflow job \
  --tasks-dir .ref/skillsbench/tasks \
  --agent claude-agent-acp \
  --model claude-sonnet-4-5 \
  --concurrency 8 --retries 2 \
  --jobs-dir jobs/run1

# YAML config
benchflow job --config benchmarks/skillsbench-claude.yaml
```

Minimal YAML config:

```yaml
tasks_dir: .ref/skillsbench/tasks
jobs_dir: jobs/skillsbench-claude
agent: claude-agent-acp
model: claude-sonnet-4-5
environment: daytona
concurrency: 8
max_retries: 2
```

Output: `Score: 18/24 (75.0%), errors=1`

---

## benchflow agents

List all registered agents. No flags.

```
 Registered Agents
 Name              Description                         Protocol  Requires
 claude-agent-acp  Claude Code via ACP                 acp       ANTHROPIC_API_KEY (or login)
 pi-acp            Pi agent via ACP                    acp       ANTHROPIC_API_KEY
 openclaw          OpenClaw agent via ACP shim         acp
 codex-acp         OpenAI Codex agent via ACP          acp       OPENAI_API_KEY (or login)
 gemini            Google Gemini CLI via ACP           acp       GOOGLE_API_KEY (or login)
```

---

## benchflow metrics

Aggregate metrics from a completed jobs directory.

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `JOBS_DIR` (positional) | *(required)* | Jobs directory to analyze |
| `--benchmark`, `-b` | *(all)* | Filter by benchmark name |
| `--agent`, `-a` | *(all)* | Filter by agent name |
| `--model`, `-m` | *(all)* | Filter by model name |
| `--json` | false | Output raw JSON |

```bash
benchflow metrics jobs/skillsbench-run1 --agent claude-agent-acp
benchflow metrics jobs/ --json
```

Output includes total, passed, failed, errored counts, score, avg tool calls, avg duration, and task name lists.

```
 Metric          Value
 Total           24
 Passed          18
 Failed          5
 Errored         1
 Score           75.0%
 Avg tool calls  11.4
 Avg duration    92s

Passed: fix-login-bug, add-tests, ...
Errors: timeout-task
Error breakdown: {'TIMEOUT': 1}
```

---

## benchflow view

View a trial trajectory in the browser.

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `TRIAL_DIR` (positional) | *(required)* | Trial or job directory containing trajectory files |
| `--port` | `8888` | Port for the local HTTP server |

```bash
benchflow view jobs/skillsbench-run1/fix-login-bug/trial-0
```

Starts a local HTTP server. Navigate to `http://localhost:8888` (or the specified port) in your browser to view an interactive replay of the ACP trajectory.

---

## benchflow eval

Evaluate a skill against multiple tasks. A focused variant of `benchflow job` — no retries, simpler output.

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks-dir`, `-t` | *(required)* | Directory of task subdirectories |
| `--skill` | — | Path to a `SKILL.md` file; parent dir used as `skills_dir` |
| `--skills-dir`, `-s` | — | Skills directory (takes precedence over `--skill`) |
| `--agent`, `-a` | `claude-agent-acp` | Agent name |
| `--model`, `-m` | `claude-haiku-4-5-20251001` | Model ID |
| `--env`, `-e` | `docker` | `docker` or `daytona` |
| `--concurrency`, `-c` | `4` | Max concurrent tasks |
| `--jobs-dir`, `-o` | `jobs` | Output directory |

```bash
benchflow eval \
  --tasks-dir tasks/ \
  --skill skills/gws/SKILL.md \
  --agent claude-agent-acp \
  --env daytona
```

Output:

```
Skill Eval Results
  Skill: skills/gws/SKILL.md
  Score: 7/10 (70.0%), errors=0
  Elapsed: 183s
```

---

## benchflow skills

List discovered skills or install a skill.

| Flag | Default | Description |
|------|---------|-------------|
| `--dir`, `-d` | `~/.benchflow/skills`, `.claude/skills`, `skills/` | Skills directory to scan |
| `--install`, `-i` | — | Skill reference to install, e.g. `owner/repo@skill-name` |

```bash
benchflow skills                                           # list
benchflow skills --dir skills/                             # list from specific dir
benchflow skills --install anthropics/skills@find-files    # install
benchflow skills --install anthropics/skills@find-files --dir skills/
```

List output: table with Name, Version, Description, Path. Install output: `Installed: skills/find-files`

---

## benchflow tasks init

Scaffold a new task directory.

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `NAME` (positional) | *(required)* | Task name (used as directory name) |
| `--dir`, `-p` | `tasks/` | Parent directory |
| `--no-pytest` | false | Skip pytest test template |
| `--no-solution` | false | Skip solution template |

```bash
benchflow tasks init fix-auth-bug
benchflow tasks init fix-auth-bug --dir .ref/skillsbench/tasks --no-solution
```

Output:

```
Created: tasks/fix-auth-bug/
  task.toml, instruction.md, environment/Dockerfile, tests/test.sh
  solution/solve.sh
```

Exits non-zero if the directory already exists.

---

## benchflow tasks check

Validate a task directory structure.

```bash
benchflow tasks check tasks/fix-auth-bug
```

Output:

```
✓ fix-auth-bug — valid
```

```
✗ fix-auth-bug — 2 issue(s):
  → missing task.toml
  → tests/test.sh not found
```

Exits non-zero when issues are found.

---

## benchflow cleanup

Remove orphaned Daytona sandboxes older than `--max-age` minutes. Requires the `daytona` Python SDK.

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | false | List sandboxes that would be deleted without deleting |
| `--max-age` | `1440` | Age threshold in minutes (default: 24 hours) |

```bash
benchflow cleanup --dry-run --max-age 60
benchflow cleanup --max-age 120
```

Dry-run output:

```
  abc123  state=running  age=1523m (delete)
  def456  state=stopped  age=30m   (skip)

2 sandboxes found, 1 older than 60m (use without --dry-run to delete)
```

Delete output: `1 sandboxes deleted (1 skipped, younger than 1440m)`
