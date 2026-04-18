# Follow-up Bench

Does an independent code review improve agent performance on coding tasks?

## Concept

```
Agent A (coder)          Agent B (reviewer)
    │                         │
    ├─ solves task ──────────▶│
    │                         ├─ reads output, writes review
    │◀── /app/.outbox/coder ──┤
    ├─ revises solution       │
    │                         │
    ▼                         ▼
 Verifier scores           (review only)
```

**The question**: When a different agent reviews the coder's work and gives targeted feedback, does the coder produce a better final solution?

## Architecture: Scene-based two-agent flow

Uses BenchFlow's Scene runtime — two real ACP agents in a shared sandbox:

1. **Coder** (Role 1) attempts the TB2 task
2. **Reviewer** (Role 2) reads the coder's work, writes specific feedback to `/app/.outbox/coder.json`
3. **Coder** reads the feedback, revises
4. Scene ends when reviewer approves (LGTM) or max rounds hit
5. Verifier scores the final `/app/` state

Both roles are real ACP agents (can be same or different models). Communication is filesystem-based via the outbox convention.

## Quick start

```python
from benchmarks.followup_bench.runner import run_followup_task
from pathlib import Path

result = await run_followup_task(
    task_path=Path(".ref/terminal-bench-2/tasks/some-task"),
    coder_agent="gemini",
    coder_model="gemini-3.1-flash-lite-preview",
    reviewer_agent="gemini",
    reviewer_model="gemini-3-pro-preview",
    environment="daytona",
)
print(f"Rounds: {result.n_rounds}, Messages: {len(result.messages)}")
```

## What we measure

Compare two runs on the same TB2 tasks:

| Run | Setup | What it tests |
|-----|-------|---------------|
| Baseline | Coder only (single turn) | Raw agent capability |
| Followup | Coder + Reviewer + Coder revision | Does review help? |

**Lift** = followup score - baseline score. Positive lift = review helps. Negative = review causes regressions.

## Experiment variants

| Variant | Coder | Reviewer | Question |
|---------|-------|----------|----------|
| Self-review | Gemini | Gemini | Does same-model review help? |
| Cross-model | Gemini | Claude | Does a different model catch different bugs? |
| Strong reviewer | Haiku | Opus | Does a stronger reviewer help a weaker coder? |
| Weak reviewer | Opus | Haiku | Does a weaker reviewer hurt a stronger coder? |
