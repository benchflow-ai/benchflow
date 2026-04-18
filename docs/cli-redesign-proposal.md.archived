# CLI Redesign Proposal — `benchflow run` as the unified entry point

## Problem

Today's CLI has too many overlapping commands for running tasks:

```
benchflow run       — single task only
benchflow job       — batch tasks from directory or YAML
benchflow eval      — batch tasks + skill injection (thin wrapper around job)
benchflow skill-eval — generate tasks from evals.json, with/without comparison
```

Harbor has ONE command: `harbor run`. It handles single tasks, datasets,
concurrency, and cloud execution all through flags. Users never need to
choose between `run` vs `job` vs `eval`.

## Proposed: Unified `benchflow run`

```bash
# === Single task (like today) ===
benchflow run -t path/to/task -a claude-agent-acp

# === Batch: whole directory (replaces `benchflow job`) ===
benchflow run -t path/to/tasks/ -a claude-agent-acp -c 4

# === Batch: named dataset (new, like harbor) ===
benchflow run -d skillsbench -a claude-agent-acp -c 64 -e daytona
benchflow run -d terminal-bench-2 -a codex-acp -m gpt-5.4

# === From YAML config (replaces `benchflow job -f`) ===
benchflow run -f benchmarks/skillsbench-claude-glm5.yaml

# === With skills ===
benchflow run -t tasks/ -a claude-agent-acp -s skills/

# === Cloud execution ===
benchflow run -d skillsbench -a claude-agent-acp -e daytona -c 64

# === Retries ===
benchflow run -t tasks/ -a claude-agent-acp --retries 2
```

**How it works:** `run` auto-detects the mode:
- `-t path/to/file` where `task.toml` exists → single task mode
- `-t path/to/dir` where subdirs have `task.toml` → batch mode (Job)
- `-d dataset-name` → auto-download + batch mode
- `-f config.yaml` → YAML config mode

### Skills subcommands (replaces `benchflow skills` + `benchflow skill-eval` + `benchflow eval`)

```bash
# === Skill management ===
benchflow skills list                     # list discovered skills
benchflow skills install owner/repo@name  # install from skills.sh

# === Skill evaluation (replaces skill-eval) ===
benchflow skills eval my-skill/ -a claude-agent-acp
benchflow skills eval my-skill/ -a claude-agent-acp,codex-acp --export-gepa traces/
benchflow skills eval my-skill/ -a claude-agent-acp --no-baseline --dry-run
```

### Full CLI tree

```
benchflow
├── run                    # THE command — single task, batch, dataset, YAML
│   ├── -t / --task-dir    # task or tasks directory
│   ├── -d / --dataset     # named dataset (skillsbench, terminal-bench-2)
│   ├── -f / --config      # YAML config file
│   ├── -a / --agent       # agent name(s), comma-separated
│   ├── -m / --model       # model name(s), comma-separated  
│   ├── -e / --env         # environment: docker, daytona
│   ├── -c / --concurrency # max concurrent tasks (default: 1 for single, 4 for batch)
│   ├── -s / --skills-dir  # skills directory to deploy
│   ├── -o / --jobs-dir    # output directory (default: jobs/)
│   ├── -p / --prompt      # custom prompt(s)
│   ├── --retries          # max retries per task (default: 0 single, 2 batch)
│   ├── --exclude          # comma-separated task names to skip
│   ├── --ae               # agent env var (KEY=VALUE), repeatable
│   └── --sandbox-user     # run agent as non-root user
│
├── skills
│   ├── list               # list discovered skills
│   ├── install <spec>     # install from skills.sh
│   └── eval <skill-dir>   # evaluate skill (evals.json → with/without)
│       ├── -a / --agent
│       ├── -m / --model
│       ├── -e / --env
│       ├── -c / --concurrency
│       ├── --no-baseline
│       ├── --dry-run
│       └── --export-gepa <dir>
│
├── tasks
│   ├── init <name>        # scaffold new task
│   └── check <dir>        # validate task structure
│
├── agents                 # list registered agents
├── metrics <jobs-dir>     # aggregate and display results
├── view <trial-dir>       # trajectory viewer in browser
└── cleanup                # clean up orphaned Daytona sandboxes
```

## Key design decisions to review

1. **Single `run` vs `run` + `job`**: Should `run` auto-detect batch mode, or keep `job` as a separate command for power users?

2. **`-d dataset-name` auto-download**: Harbor has a dataset registry. We have `ensure_tasks()` for skillsbench and terminal-bench-2. Should we formalize this into a registry?

3. **Default concurrency**: Single task → 1, batch → 4, daytona batch → 64? Or always explicit?

4. **`skills eval` vs `skill-eval`**: Grouped under `skills` sub-app (consistent with `tasks init`/`tasks check`) vs flat top-level command (easier to type)?

5. **Backward compatibility**: Should we keep `benchflow job` and `benchflow eval` as hidden aliases during transition?

6. **Flag naming**: `-t` for task (Harbor uses `-p` for path), `-d` for dataset (Harbor uses `-d`), `-n` vs `-c` for concurrency (Harbor uses `-n`, we use `-c`)?

## What doesn't change

- `benchflow agents` — stays as-is
- `benchflow metrics` — stays as-is  
- `benchflow view` — stays as-is
- `benchflow cleanup` — stays as-is
- `benchflow tasks init/check` — stays as-is
