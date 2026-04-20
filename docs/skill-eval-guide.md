# Skill Eval Guide

Test whether your agent skill actually helps agents perform better.

## Install

```bash
uv tool install benchflow
```

## Overview

`bench skills eval` takes a skill directory with an `evals/evals.json`
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
bench skills eval my-skill/ -a claude-agent-acp
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
bench skills eval my-skill/ \
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
bench skills eval my-skill/ -a claude-agent-acp --export-gepa traces/
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

## End-to-End Walkthrough

Here's a complete example evaluating a real skill from scratch.

### Step 1: Create the skill

```bash
mkdir -p gws-skill/scripts gws-skill/evals
```

Write `gws-skill/SKILL.md`:
```markdown
---
name: gws-email-drafting
description: Draft professional emails using Gmail API patterns
---

# GWS Email Drafting

Use the templates in scripts/ to draft professional emails.
```

Write `gws-skill/scripts/draft_email.py`:
```python
import sys
template = sys.argv[1] if len(sys.argv) > 1 else "general"
print(f"Email drafted using {template} template")
```

### Step 2: Write eval cases

Write `gws-skill/evals/evals.json`:
```json
{
  "skill_name": "gws-email-drafting",
  "version": "1",
  "defaults": {
    "timeout_sec": 300,
    "judge_model": "claude-haiku-4-5-20251001"
  },
  "cases": [
    {
      "id": "draft-intro-email",
      "question": "Draft a professional introduction email to a potential workshop speaker. Use the gws-email-drafting skill.",
      "ground_truth": "The agent produced a professional email with subject line, greeting, body explaining the workshop, and call to action.",
      "expected_behavior": [
        "The agent read the SKILL.md to understand the skill",
        "The agent used draft_email.py or followed the skill's patterns",
        "The email has a clear subject line",
        "The email body is professional and includes a call to action"
      ]
    },
    {
      "id": "draft-followup",
      "question": "Draft a follow-up email to someone who hasn't responded in 2 weeks. Use the gws-email-drafting skill.",
      "ground_truth": "The agent produced a polite follow-up email that references the original outreach.",
      "expected_behavior": [
        "The agent read the SKILL.md",
        "The email references a previous conversation",
        "The tone is polite but action-oriented",
        "The email is concise (under 200 words)"
      ]
    }
  ]
}
```

### Step 3: Run the eval

```bash
$ benchflow skills eval ./gws-skill/ -a claude-agent-acp -a codex-acp

Skill eval: gws-email-drafting (2 cases)
  Agents: claude-agent-acp, codex-acp
  Environment: docker

         Skill Eval: gws-email-drafting
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Agent             ┃ Mode       ┃ Score ┃ Avg Reward ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ claude-agent-acp  │ with-skill │ 2/2   │ 0.92       │
│ claude-agent-acp  │ baseline   │ 1/2   │ 0.55       │
│ claude-agent-acp  │ LIFT       │ +1    │ +0.37      │
│ codex-acp         │ with-skill │ 2/2   │ 0.88       │
│ codex-acp         │ baseline   │ 1/2   │ 0.48       │
│ codex-acp         │ LIFT       │ +1    │ +0.40      │
└───────────────────┴────────────┴───────┴────────────┘
```

### Step 4: Inspect results

Results are saved to `jobs/skill-eval/<skill-name>/`:
```
jobs/skill-eval/gws-email-drafting/
├── claude-agent-acp/
│   ├── with-skill/
│   │   ├── draft-intro-email__abc123/
│   │   │   ├── result.json
│   │   │   ├── trajectory/acp_trajectory.jsonl
│   │   │   └── timing.json
│   │   └── draft-followup__def456/
│   │       └── ...
│   └── baseline/
│       └── ...
└── codex-acp/
    └── ...
```

### Step 5: Improve with GEPA (optional)

```bash
$ benchflow skills eval ./gws-skill/ -a claude-agent-acp --export-gepa

GEPA traces exported to jobs/skill-eval/gws-email-drafting/gepa
```

Feed traces to the SkillSpin improvement pipeline to automatically
evolve the skill text based on failure patterns.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    bench skills eval                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │ evals.json  │───▶│ Task Generator   │───▶│ Ephemeral      │  │
│  │ (2-8 cases) │    │ (with/without    │    │ BenchFlow Tasks   │  │
│  └─────────────┘    │  skill mode)     │    │ (auto-deleted) │  │
│                     └──────────────────┘    └───────┬────────┘  │
│                                                      │          │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────▼────────┐  │
│  │ Lift Report │◀───│ Result Collector │◀───│ Job Engine     │  │
│  │ (per agent) │    │ (per case×mode)  │    │ (concurrency,  │  │
│  └─────────────┘    └──────────────────┘    │  retries, ACP) │  │
│                                              └────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LLM Judge (claude-haiku-4-5)                            │    │
│  │ Reads: trajectory + case.json (ground_truth, rubric)    │    │
│  │ Writes: /logs/verifier/reward.txt (0.0-1.0)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Real-World Example: Benchmark Hallucination Audit

The `benchmark-hallucination-audit` skill teaches agents to verify claims
in benchmark comparison tables by checking papers, GitHub, and HuggingFace.
Its eval cases use findings from a real audit of AlphaEval (arXiv:2604.12162).

```
benchmark-hallucination-audit/
├── skill.md                     # 5-round layered subagent methodology
└── evals/
    └── evals.json               # 8 cases from real AlphaEval audit
```

Sample case — detecting a Cross-Domain overclaim:
```json
{
  "id": "overclaim-xdom-agentbench",
  "question": "AlphaEval Table 1 marks AgentBench with Cross-Domain=✓. The definition is: 'spans 3+ distinct PROFESSIONAL domains'. AgentBench has 8 environments: OS, Database, Knowledge Graph, Card Game, Puzzles, ALFWorld, WebShop, Web Browsing. Is this correct or an overclaim?",
  "ground_truth": "OVERCLAIM. The 8 environments are TASK TYPES, not 3+ professional domains like healthcare, finance, or law.",
  "expected_behavior": [
    "The agent fetched the AgentBench paper (arXiv:2308.03688)",
    "The agent compared environments against the strict definition",
    "The agent concluded task types ≠ professional domains"
  ]
}
```

Other cases test: missing Multi-Modal marks (MLE-bench), missing Dynamic
marks (Gaia2 — title literally says "Dynamic"), correct Production marks
(SWE-Lancer — $1M real Upwork payouts), and self-audit overclaims
(AlphaEval's own Dynamic=✓ is aspirational, not mechanism-backed).

Run it:
```bash
bench skills eval ./benchmark-hallucination-audit/ -a claude-agent-acp -a codex-acp
```

This is a good template for **research skills** — where the eval cases
have verified ground truth from manual expert analysis, and the skill
teaches a systematic methodology.

## For Skill Developers (Jon Snow Adapter Pattern)

If you maintain skills and want CI-integrated eval:

```
my-skill/
├── SKILL.md
├── scripts/
│   └── do_something.py
└── evals/
    └── evals.json          ← 2-4 test cases
```

That's it. No benchmark task authoring, no Dockerfiles, no test scripts.
BenchFlow generates everything ephemeral — only results persist.

**CI integration:**
```bash
# In your skill's CI pipeline
uv tool install benchflow
bench skills eval . -a claude-agent-acp --no-baseline
# Exit code 1 if any case scores < 0.5
```

**What the adapter does (zero LLM):**
```
evals.json → Generate benchmark tasks → Run agents → Grade → Cleanup
  (static)     (deterministic)        (ACP)      (LLM)   (auto)
```

The adapter is purely deterministic — no LLM in task generation.
LLM is only used at grading time (the judge).

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
6. **Test the lift, not just correctness** — The goal is to show the skill
   improves performance vs baseline. If baseline already scores high, the
   skill isn't adding value
