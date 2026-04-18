# Follow-up Bench

A benchmark for evaluating how well AI agents handle multi-turn conversations where users correct, clarify, or change requirements mid-task.

## Quick start

```bash
bench skills eval ./benchmarks/followup-bench/ -a claude-agent-acp --no-baseline
```

## Cases

| ID | Scenario | What's tested |
|----|----------|---------------|
| `correction-mid-task` | "Sort by name" → "actually, sort by age descending" | Does the agent apply the correction? |
| `partial-undo` | Create config → "change only debug and log_level" | Does the agent modify only what was asked? |
| `ambiguous-followup` | Write average function → "handle the edge cases" | Does the agent handle ambiguity well? |

## Multi-turn execution

These tasks use BenchFlow's multi-turn prompt system. The `[FOLLOW-UP]` markers in each question translate to separate prompts in the ACP session:

```yaml
prompts:
  - null    # → instruction.md (initial request)
  - "Actually, sort by 'age' key in descending order instead."
```

The agent keeps its full context between turns — it sees the follow-up as a continuation, not a new task.

## What we measure

1. **Correction accuracy** — Did the agent apply the correction?
2. **Preservation** — Did the agent keep unchanged parts intact?
3. **Ambiguity handling** — Did the agent ask for clarification or handle proactively?
4. **Efficiency** — Did the agent avoid redundant work (not starting from scratch)?

## Extending

Add new multi-turn cases to `evals/evals.json`. Use `[FOLLOW-UP]` markers to indicate turn boundaries. The judge evaluates the final state against expected behavior across all turns.
