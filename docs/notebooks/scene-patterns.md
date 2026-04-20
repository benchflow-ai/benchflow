# BenchFlow Scene Patterns — Multi-Turn, Multi-Round, Multi-Scene

This document + the companion notebook demonstrates all multi-agent evaluation patterns that BenchFlow's Scene lifecycle supports. Each pattern replaces hundreds of lines of custom runtime code (e.g. [harbor PR #1462](https://github.com/harbor-framework/harbor/pull/1462)) with a YAML config.

## Terminology

| Term | Definition | Example |
|------|-----------|---------|
| **Turn** | One prompt in one ACP session (one role acts) | Agent writes a regex |
| **Round** | One cycle of multi-agent exchange (A→B) | Coder submits, reviewer critiques |
| **Scene** | One interaction region with roles + turns | The entire code review loop |
| **Trial** | Sequence of scenes in a shared sandbox | Skill-gen scene → solve scene |

## Pattern 1: Single-Agent Baseline

One agent, one task, one turn. The simplest case.

```yaml
scenes:
  - name: solve
    roles: [{name: agent, agent: gemini, model: gemini-3.1-flash-lite-preview}]
    turns: [{role: agent}]
```

**Result:** reward=0.0, 3 tool calls. The agent wrote a regex but missed edge cases.

## Pattern 2: Multi-Turn Self-Review

Same agent, two sequential prompts. The agent maintains ACP session context between turns.

```yaml
scenes:
  - name: self-review
    roles: [{name: agent, agent: gemini, model: gemini-3.1-flash-lite-preview}]
    turns:
      - {role: agent}  # solve
      - {role: agent, prompt: "Review your solution. Check edge cases and fix issues."}
```

**Result:** reward=1.0, 12 tool calls. The self-review prompt caught the regex bug — agent fixed it on the second turn.

## Pattern 3: Multi-Round Code Review

Two roles taking turns — coder→reviewer→coder = 1.5 rounds. Different agents communicate via filesystem outbox.

```yaml
scenes:
  - name: code-review
    roles:
      - {name: coder, agent: gemini, model: gemini-3.1-flash-lite-preview}
      - {name: reviewer, agent: gemini, model: gemini-3.1-flash-lite-preview}
    turns:
      - {role: coder}
      - {role: reviewer, prompt: "Review the code in /app/. Write feedback to /app/.outbox/coder.json"}
      - {role: coder, prompt: "Read reviewer feedback and fix issues."}
```

**Result:** reward=0.0, 16 tool calls. Reviewer provided feedback but coder's revision didn't fix the core issue.

**At scale (267 trials):** Reviewer pattern doubles win rate — baseline 9.0% → reviewer 19.4% on TB2.

## Pattern 4: Interactive User Simulation

A "user" role reveals task information gradually. This is exactly what [harbor #1316](https://github.com/harbor-framework/harbor/issues/1316) proposed and [PR #1462](https://github.com/harbor-framework/harbor/pull/1462) built 600+ lines to implement.

```yaml
scenes:
  - name: interactive
    roles:
      - {name: user, agent: gemini, model: gemini-3.1-flash-lite-preview}
      - {name: agent, agent: gemini, model: gemini-3.1-flash-lite-preview}
    turns:
      - {role: user, prompt: "Give the agent a vague version of the task..."}
      - {role: agent}
```

**Result:** reward=0.0, 2 tool calls. The vague instruction wasn't enough — demonstrates why iterative clarification matters.

## Pattern 5: Multi-Scene BYOS (Skill Generation → Solve)

Two scenes in sequence, shared sandbox. First scene generates a skill document, second scene solves using it.

```yaml
scenes:
  - name: skill-gen
    roles: [{name: gen, agent: gemini, model: gemini-3.1-flash-lite-preview}]
    turns:
      - {role: gen, prompt: "Analyze the task and write a skill document to /app/generated-skill.md"}
  - name: solve
    roles: [{name: solver, agent: gemini, model: gemini-3.1-flash-lite-preview}]
    turns:
      - {role: solver}
```

**Result:** reward=0.0, 5 tool calls. Self-generated skills provide 0pp lift — consistent with the SkillsBench paper finding.

## Pattern 6: Realistic Contract Consultation

An anonymized real-world scenario: a startup CTO consulting with an advisor about a vendor contract. Multiple rounds of clarification and recommendation.

```yaml
scenes:
  - name: contract-review
    roles:
      - {name: client, agent: gemini, model: gemini-3.1-flash-lite-preview}
      - {name: advisor, agent: gemini, model: gemini-3.1-flash-lite-preview}
    turns:
      - {role: client}   # shares initial situation and concerns
      - {role: advisor}  # analyzes contract clause-by-clause
      - {role: client}   # clarifies top priorities
      - {role: advisor}  # gives final redline recommendations
```

This pattern shows why multi-round matters for real users — the advisor's analysis improves when the client clarifies priorities.

## Comparison: BenchFlow vs Harbor

| Pattern | BenchFlow | Harbor |
|---------|-----------|--------|
| Single-agent | `scenes: [{turns: [{role: agent}]}]` | `harbor run` |
| Multi-turn | Same scene, multiple turns | `prompts:` list in YAML |
| Multi-round | Two roles in one scene | PR #1462: BaseUser + UserFactory + per-round archiving (600+ lines) |
| Interactive user | Role with oracle access | PR #1462 required |
| Multi-scene | Two scenes in sequence | Not supported |
| Skill warmup | Scene with `scoring: null` | Not supported |

## Running the Demos

```bash
pip install benchflow==0.3.0a9
export DAYTONA_API_KEY="dtn_..."
export GEMINI_API_KEY="AIza..."

# All patterns on regex-log (TB2 task)
python docs/notebooks/multi-turn-scene-demo.py

# Realistic contract consultation (no Docker needed)
python docs/notebooks/consultation-demo.py
```
