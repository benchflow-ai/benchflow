# Skill Eval — Design Document

> Evaluate any agent skill with one command. No Harbor task authoring required.

## Problem

Skill developers today have no lightweight way to test whether their skills
actually help agents. The current path requires:

1. Writing a full Harbor task (Dockerfile, task.toml, instruction.md, test.sh, test_outputs.py)
2. Running it with and without the skill manually
3. Comparing results by hand

Jon Snow (NVIDIA) built a Harbor adapter that auto-generates tasks from a
lightweight `evals/evals.json` inside the skill directory. We want to make this
native to benchflow so any skill developer can run:

```bash
benchflow skill-eval my-skill/ -a claude-agent-acp,codex-acp
```

## User-Facing Design

### Skill directory with evals

```
my-skill/
├── SKILL.md
├── scripts/
│   └── helper.py
└── evals/
    ├── evals.json          # Test cases (required)
    ├── Dockerfile           # Custom env (optional, default: python:3.12-slim)
    └── requirements.txt     # Extra deps (optional)
```

### evals.json schema

```jsonc
{
  "version": "1",
  "skill_name": "calculator",        // must match SKILL.md name
  "defaults": {
    "timeout_sec": 300,               // per-task timeout
    "judge_model": "claude-haiku-4-5-20251001"  // model for LLM judge
  },
  "cases": [
    {
      "id": "calculator-001",
      "question": "What is 2 + 3 * 4? Use the calculator skill.",
      "ground_truth": "14",           // expected final answer (optional)
      "expected_behavior": [          // behavioral rubric for LLM judge
        "Agent read the calculator SKILL.md",
        "Agent executed calc.py with '2 + 3 * 4'",
        "Agent reported correct result of 14"
      ],
      "expected_skill": "calculator", // which skill should be invoked
      "expected_script": "calc.py",   // which script should be called (optional)
      "environment": {}               // per-case env var overrides (optional)
    }
  ]
}
```

### CLI

```bash
# Evaluate one skill against one agent
benchflow skill-eval my-skill/ -a claude-agent-acp

# Multi-agent comparison
benchflow skill-eval my-skill/ -a claude-agent-acp,codex-acp -m haiku,gpt-5.4

# Skip baseline (with-skill only, no lift calculation)
benchflow skill-eval my-skill/ -a claude-agent-acp --no-baseline

# Custom output dir
benchflow skill-eval my-skill/ -a claude-agent-acp -o jobs/skill-eval-run

# Export traces for GEPA
benchflow skill-eval my-skill/ -a claude-agent-acp --export-gepa gepa-traces/
```

### Output

```
Skill Eval: calculator (3 cases × 2 agents × 2 modes)

Agent               Mode        Score   Avg Rubric   Skill Used
─────────────────────────────────────────────────────────────────
claude-agent-acp    with-skill  2/3     0.87         3/3
claude-agent-acp    baseline    1/3     0.42         0/3
claude-agent-acp    LIFT        +1      +0.45        +3
─────────────────────────────────────────────────────────────────
codex-acp           with-skill  3/3     0.93         3/3
codex-acp           baseline    1/3     0.38         0/3
codex-acp           LIFT        +2      +0.55        +3

Results: jobs/skill-eval/calculator/
```

## Architecture

### Module: `src/benchflow/skill_eval.py`

```
skill_eval.py
├── load_eval_dataset(skill_dir) → EvalDataset
├── generate_tasks(dataset, with_skill=True) → list[Path]  # ephemeral
├── run_comparison(dataset, agents, models, ...) → SkillEvalResult
├── cleanup_tasks(task_dirs)
└── export_gepa(result, output_dir)
```

### Data flow

```
skill_dir/evals/evals.json
    │
    ▼
load_eval_dataset()          # parse + validate
    │
    ▼
generate_tasks(with_skill=True)    generate_tasks(with_skill=False)
    │                                     │
    ▼                                     ▼
┌──────────────────┐              ┌──────────────────┐
│ _tmp/with-skill/ │              │ _tmp/baseline/    │
│ ├── case-001/    │              │ ├── case-001/     │
│ │   ├── task.toml│              │ │   ├── task.toml │
│ │   ├── inst.md  │              │ │   ├── inst.md   │
│ │   ├── env/     │              │ │   ├── env/      │
│ │   │   └── Dock │              │ │   │   └── Dock  │
│ │   └── tests/   │              │ │   └── tests/    │
│ └── case-002/    │              │ └── case-002/     │
└──────────────────┘              └──────────────────┘
    │                                     │
    ▼                                     ▼
Job.run() × N agents              Job.run() × N agents
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
          compare_results()
                   │
                   ▼
          SkillEvalResult
          ├── per_case: [{id, agent, with_score, without_score, lift, rubric_scores}]
          ├── per_agent: [{agent, with_total, without_total, lift, avg_rubric}]
          └── summary: {skill, n_cases, n_agents, best_agent, avg_lift}
```

### Ephemeral task generation

Each `evals.json` case becomes a Harbor-format task:

**`instruction.md`** — generated from `case.question`:
```markdown
{case.question}
```

**`task.toml`** — from defaults + case overrides:
```toml
[task]
timeout = 300

[environment]
dockerfile = "environment/Dockerfile"
```

**`environment/Dockerfile`** — from `evals/Dockerfile` or default:
```dockerfile
FROM python:3.12-slim
# Install skill if with_skill mode
COPY skills/ /home/user/.claude/skills/
# Install extra deps if requirements.txt exists
COPY requirements.txt /tmp/ 
RUN pip install -r /tmp/requirements.txt 2>/dev/null || true
WORKDIR /app
```

**`tests/test.sh`** — runs the LLM judge verifier:
```bash
#!/bin/bash
python3 /tests/judge.py
cat /tests/reward.txt > /logs/verifier/reward.txt
```

**`tests/judge.py`** — LLM-judged verifier (see below)

### LLM Judge Verifier

The judge grades agent trajectory against the behavioral rubric.

```python
"""LLM judge verifier for skill eval.

Reads the agent trajectory, compares against expected_behavior rubric,
and writes a reward (0.0-1.0) based on rubric adherence.
"""

import json, os, sys
from pathlib import Path

# Injected at generation time
CASE = json.loads(Path("/tests/case.json").read_text())
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")

def build_judge_prompt(trajectory_text: str, case: dict) -> str:
    rubric = "\n".join(f"- {b}" for b in case["expected_behavior"])
    ground_truth = case.get("ground_truth", "N/A")
    return f"""You are evaluating whether an AI agent correctly used a skill.

## Task
The agent was asked: {case["question"]}

## Expected behavior (rubric)
{rubric}

## Expected answer
{ground_truth}

## Agent trajectory
{trajectory_text}

## Instructions
Score each rubric item as PASS or FAIL. Then give an overall score 0.0-1.0.
Respond in JSON: {{"items": [{{"criterion": "...", "pass": true/false}}], "score": 0.0-1.0, "reasoning": "..."}}
"""

def judge(trajectory_path: str = "/logs/agent") -> float:
    # Read trajectory
    traj_files = sorted(Path(trajectory_path).glob("*.txt"))
    trajectory_text = "\n".join(f.read_text() for f in traj_files) if traj_files else "NO TRAJECTORY"
    
    # Call LLM judge
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": build_judge_prompt(trajectory_text, CASE)}],
    )
    
    # Parse score
    text = response.content[0].text
    try:
        result = json.loads(text)
        return float(result["score"])
    except (json.JSONDecodeError, KeyError):
        # Fallback: extract number
        import re
        match = re.search(r'"score"\s*:\s*([\d.]+)', text)
        return float(match.group(1)) if match else 0.0

if __name__ == "__main__":
    score = judge()
    Path("/tests/reward.txt").write_text(str(score))
    print(f"Judge score: {score}")
```

### GEPA Export Format

GEPA needs execution traces paired with scores to evolve the skill text.

```
gepa-traces/
├── skill.md              # current SKILL.md content (the artifact to evolve)
├── traces/
│   ├── case-001-claude.json
│   ├── case-001-codex.json
│   ├── case-002-claude.json
│   └── ...
└── summary.json          # aggregate scores
```

Each trace file:
```json
{
  "case_id": "calculator-001",
  "agent": "claude-agent-acp",
  "model": "claude-haiku-4-5-20251001",
  "with_skill": true,
  "score": 0.87,
  "rubric_results": [
    {"criterion": "Agent read the calculator SKILL.md", "pass": true},
    {"criterion": "Agent executed calc.py with '2 + 3 * 4'", "pass": true},
    {"criterion": "Agent reported correct result of 14", "pass": false}
  ],
  "trajectory": [...],
  "skill_text": "# Calculator\n..."
}
```

GEPA reads these, identifies failure patterns, and proposes edits to `skill.md`.

## Multi-Agent Sandbox Architecture (moltbook-in-a-box)

> Multiple agents with different tool access in a shared environment, communicating via MCP.

### Design (extends harbor-cookbook simulated-user pattern)

```
docker-compose.yml
├── orchestrator    # FastMCP server — routes messages between agents
├── agent-1         # Claude Code + MCP tools: [ask_agent_2, read_repo, gcal]
├── agent-2         # Codex + MCP tools: [ask_agent_1, write_code, gdocs]
├── agent-3         # Gemini + MCP tools: [ask_agent_1, ask_agent_2, review]
└── shared-volume   # /workspace mounted into all containers
```

Each agent gets:
- Its own ACP session (benchflow manages lifecycle)
- MCP tools to communicate with other agents (FastMCP streamable-http)
- Scoped tool access (per-agent permission config)
- Access to shared filesystem

**task.toml extension:**
```toml
[environment]
type = "multi-agent"

[[environment.agents]]
name = "coder"
agent = "claude-agent-acp"
model = "claude-haiku-4-5-20251001"
tools = ["write_code", "ask_reviewer"]
instruction = "prompts/coder.md"

[[environment.agents]]
name = "reviewer"  
agent = "codex-acp"
model = "gpt-5.4"
tools = ["read_code", "ask_coder", "approve"]
instruction = "prompts/reviewer.md"

[[environment.mcp_servers]]
name = "orchestrator"
transport = "streamable-http"
url = "http://orchestrator:8000/mcp"
```

**Orchestrator server** (extends simulated-user `server.py`):
- Maintains message queues per agent
- Exposes `ask_{agent_name}(message: str) -> str` tools dynamically
- Logs all inter-agent communication for trajectory analysis
- Supports turn limits and timeout

This is a larger feature — design doc only for now. Implementation depends on
Docker Compose support in benchflow (harbor internalization).

## Implementation Plan

### Phase 1: Skill Eval Core (this branch)
1. `src/benchflow/skill_eval.py` — EvalDataset, task generator, comparison runner
2. `src/benchflow/templates/` — Dockerfile, judge.py, test.sh templates
3. CLI: `benchflow skill-eval` command in `cli/main.py`
4. Tests: `tests/test_skill_eval.py`

### Phase 2: GEPA Integration
1. `src/benchflow/export/gepa.py` — trace export module
2. `--export-gepa` flag on skill-eval command

### Phase 3: Multi-Agent Sandbox (separate branch, needs harbor internalization)
1. Docker Compose generation from multi-agent task.toml
2. Orchestrator MCP server template
3. Multi-agent lifecycle management in SDK
4. Inter-agent trajectory capture

## Open Questions

1. **Judge model cost**: LLM judge runs per-case × per-agent × 2 (with/without). 
   For 10 cases × 3 agents = 60 judge calls. At Haiku pricing this is ~$0.30.
   Acceptable?

2. **Deterministic verifier fallback**: Should we also support non-LLM verifiers
   (e.g. `ground_truth` exact match) for cases where it's sufficient?
   Proposed: if `expected_behavior` is empty, fall back to `ground_truth` string match.

3. **GEPA version compatibility**: Need to confirm GEPA's expected input format.
   The trace format above is our best guess from the paper.

4. **Multi-agent: synchronous vs async**: Should agents take turns (synchronous)
   or run concurrently with message queues (async)? Harbor's simulated-user is
   synchronous (agent calls tool, blocks until response). Real collaboration
   may need async.
