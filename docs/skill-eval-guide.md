# Skill Eval Guide

Test whether your agent skill actually helps agents perform better.

## Install

```bash
uv tool install benchflow
```

## Overview

`benchflow skills eval` takes a skill directory with an `evals/evals.json`
file, generates benchmark tasks from it, runs them with and without the
skill installed, and reports the "lift" — how much the skill improves
agent performance.

## Quick start

### 1. Add evals to your skill

```
my-skill/
├── SKILL.md
├── scripts/
│   └── helper.py
└── evals/                    # ← add this
    └── evals.json
```

### 2. Write test cases

```json
{
  "version": "1",
  "skill_name": "my-skill",
  "defaults": {
    "timeout_sec": 300,
    "judge_model": "claude-haiku-4-5-20251001"
  },
  "cases": [
    {
      "id": "test-001",
      "question": "Do X using the my-skill skill.",
      "ground_truth": "expected output",
      "expected_behavior": [
        "Agent read the SKILL.md file",
        "Agent ran helper.py with correct arguments",
        "Agent produced the expected output"
      ]
    }
  ]
}
```

### 3. Run the eval

```bash
benchflow skills eval my-skill/ -a claude-agent-acp
```

Expected output:
```
$ benchflow skills eval ./my-skill/ -a claude-agent-acp

Skill eval: my-skill (1 cases)
  Agents: claude-agent-acp
  Environment: docker

              Skill Eval: my-skill
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Agent             ┃ Mode       ┃ Score ┃ Avg Reward ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ claude-agent-acp  │ with-skill │ 1/1   │ 0.90       │
│ claude-agent-acp  │ baseline   │ 0/1   │ 0.20       │
│ claude-agent-acp  │ LIFT       │ +1    │ +0.70      │
└───────────────────┴────────────┴───────┴────────────┘
```

## evals.json reference

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | No | Schema version (default: "1") |
| `skill_name` | string | No | Skill name (auto-detected from SKILL.md) |
| `defaults.timeout_sec` | int | No | Per-task timeout in seconds (default: 300) |
| `defaults.judge_model` | string | No | Model for LLM judge (default: claude-haiku-4-5-20251001) |

### Case fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Unique case ID (auto-generated if missing) |
| `question` | string | **Yes** | The task instruction sent to the agent |
| `ground_truth` | string | No | Expected final answer (used for exact match fallback) |
| `expected_behavior` | string[] | No | Behavioral rubric for LLM judge |
| `expected_skill` | string | No | Which skill should be invoked |
| `expected_script` | string | No | Which script should be called |
| `environment` | object | No | Per-case env var overrides |

### Grading logic

- If `expected_behavior` is provided → **LLM judge** scores the agent's
  trajectory against the rubric (0.0-1.0)
- If only `ground_truth` is provided → **exact match** checks if the
  answer appears in agent output (0.0 or 1.0)
- If neither → reward is 0.0

## Multi-agent comparison

Test your skill across multiple agents:

```bash
benchflow skills eval my-skill/ \
  -a claude-agent-acp -a codex-acp -a gemini
```

Expected output:
```
$ benchflow skills eval ./calculator/ -a claude-agent-acp -a codex-acp

Skill eval: calculator (3 cases)
  Agents: claude-agent-acp, codex-acp
  Environment: docker

              Skill Eval: calculator
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Agent             ┃ Mode       ┃ Score ┃ Avg Reward ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ claude-agent-acp  │ with-skill │ 3/3   │ 0.95       │
│ claude-agent-acp  │ baseline   │ 1/3   │ 0.38       │
│ claude-agent-acp  │ LIFT       │ +2    │ +0.57      │
│ codex-acp         │ with-skill │ 2/3   │ 0.72       │
│ codex-acp         │ baseline   │ 1/3   │ 0.35       │
│ codex-acp         │ LIFT       │ +1    │ +0.37      │
└───────────────────┴────────────┴───────┴────────────┘
```

## Custom environments

For skills that need specific dependencies, add a Dockerfile:

```
my-skill/evals/
├── evals.json
├── Dockerfile           # custom container setup
└── requirements.txt     # extra Python deps
```

The Dockerfile is used instead of the default `python:3.12-slim` base.

## GEPA integration

Export traces for GEPA skill evolution:

```bash
benchflow skills eval my-skill/ -a claude-agent-acp --export-gepa traces/
```

This creates:
```
traces/
├── skill.md              # current SKILL.md content
├── traces/               # per-case execution traces with scores
│   ├── test-001-claude-agent-acp-with.json
│   └── test-001-claude-agent-acp-without.json
└── summary.json          # aggregate lift metrics
```

Feed these to GEPA to evolve your skill:
```python
import gepa
optimizer = gepa.GEPA(traces_dir="traces/")
improved_skill = optimizer.evolve("traces/skill.md")
```

## Tips for writing good eval cases

1. **Be specific in questions** — "Use the calculator skill to compute X"
   is better than "Compute X"
2. **Write 3-5 rubric items per case** — Each should be independently
   verifiable from the trajectory
3. **Include edge cases** — Test error handling, unusual inputs, multi-step
   workflows
4. **Keep ground_truth simple** — Exact match works best for numeric or
   short-string answers
5. **Use 2-4 cases minimum** — Enough to show a pattern, not so many that
   runs get expensive
