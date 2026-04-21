# BenchFlow Scene Patterns — Contract Review Tutorial

One scenario, four patterns. Each pattern fixes a specific failure of the previous.

## Terminology

| Term | Definition |
|------|-----------|
| **Turn** | One prompt in one ACP session (one role acts) |
| **Round** | One A→B exchange between different roles |
| **Scene** | Interaction region with roles + turns |
| **Trial** | Sequence of scenes in a shared sandbox |

## The Scenario

A startup CTO is reviewing a vendor SaaS contract. The contract has buried risks: auto-renewal with uncapped price escalation, a weak SLA with no credits, and a 30-day data export window after termination. Can AI agents find them?

## Pattern 1: Single-Agent Baseline

One agent reviews the contract in a single pass.

```yaml
scenes:
  - name: review
    roles: [{name: reviewer, agent: gemini, model: gemini-3.1-flash-lite-preview}]
    turns: [{role: reviewer}]
```

**Problem:** The agent finds obvious issues (liability cap, SLA) but misses the *compound risk*: uncapped renewal pricing + auto-renewal + 90-day notice window = vendor lock-in trap.

**When to use:** Simple, well-defined tasks with clear success criteria.

## Pattern 2: Single-Agent Multi-Turn (Self-Review)

Same agent gets a second turn to re-examine for compound risks.

```yaml
scenes:
  - name: self-review
    roles: [{name: reviewer, agent: gemini, model: gemini-3.1-flash-lite-preview}]
    turns:
      - {role: reviewer}
      - {role: reviewer, prompt: "Re-examine for compound risks — combinations of clauses that together create a worse situation than any single clause alone."}
```

**Fix:** The self-review catches the compound lock-in risk (auto-renewal + uncapped pricing + short notice window).

**When to use:** Tasks where a second pass catches what the first missed. Cheap (2x cost).
**When NOT to use:** When the agent's blind spots are systematic — the same model won't catch what it missed.

## Pattern 3: Multi-Round (Two Specialists)

Two roles with different expertise — a legal reviewer focuses on liability, a commercial reviewer focuses on cost. Neither alone catches everything.

```yaml
scenes:
  - name: specialist-review
    roles:
      - {name: legal, agent: gemini, model: gemini-3.1-flash-lite-preview}
      - {name: commercial, agent: gemini, model: gemini-3.1-flash-lite-preview}
    turns:
      - {role: legal}
      - {role: commercial}
      - {role: legal, prompt: "Synthesize the commercial review with your legal findings."}
```

**Fix:** Legal flags the liability cap and termination asymmetry. Commercial flags the uncapped renewal pricing and total cost of ownership. The synthesis combines both into actionable redlines.

**When to use:** Complex tasks requiring multiple perspectives (3x cost).
**When NOT to use:** Simple tasks where a second opinion adds cost but not insight.

## Pattern 4: Interactive Multi-Round (Client + Advisor)

The client shares context and priorities, the advisor tailors recommendations. Generic advice becomes actionable.

```yaml
scenes:
  - name: consultation
    roles:
      - {name: client, agent: gemini, model: gemini-3.1-flash-lite-preview}
      - {name: advisor, agent: gemini, model: gemini-3.1-flash-lite-preview}
    turns:
      - {role: client}
      - {role: advisor}
      - {role: client, prompt: "Clarify your top priorities and acceptable compromises."}
      - {role: advisor, prompt: "Give final redline recommendations with fallback positions."}
```

**Fix:** The advisor focuses on the client's actual priority (cash flow vs lock-in vs data portability) instead of generic contract boilerplate.

**When to use:** Tasks where user context changes the answer (4x cost).
**When NOT to use:** Well-specified tasks with clear success criteria.

## Summary

| Pattern | Problem solved | Cost | Best for |
|---------|---------------|------|----------|
| 1. Baseline | — | 1x | simple tasks |
| 2. Multi-turn | missed compound risks | 2x | self-checkable tasks |
| 3. Multi-round | systematic blind spots | 3x | complex multi-perspective tasks |
| 4. Interactive | generic != actionable | 4x | user-specific decisions |

Each pattern is a TrialConfig change — same API, same verifier, same trajectory capture. No new runtime code needed.

## Running the Demo

```bash
pip install google-generativeai
export GEMINI_API_KEY="AIza..."
python docs/notebooks/scene-patterns.py
```

The script constructs the contract inline and runs all 4 patterns with actual LLM calls. No Docker or Daytona needed — it demonstrates the interaction patterns directly.
