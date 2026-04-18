# Follow-up Bench

Does an independent code review improve agent performance on coding tasks?

## Concept

```
Agent A (coder)          Agent B (reviewer)
    │                         │
    ├─ solves task ──────────▶│
    │                         ├─ reviews output
    │◀── feedback ────────────┤
    ├─ revises solution       │
    │                         │
    ▼                         ▼
 Final score              (review only)
```

**The question**: When a different agent reviews the coder's work and gives feedback, does the coder produce a better final solution?

## Quick start

```bash
# With independent review skill (coder gets reviewer feedback)
benchflow skills eval ./benchmarks/followup-bench/ -a gemini --no-baseline

# Compare: with review vs without
benchflow skills eval ./benchmarks/followup-bench/ -a gemini
```

## Cases

| ID | Scenario | What's tested |
|----|----------|---------------|
| `review-improves-solution` | Reviewer finds 3 bugs in bash script | Does coder fix all 3? |
| `review-catches-bug` | Reviewer finds merge-sort remainder bug | Does coder fix correctly? |
| `review-no-change-needed` | Reviewer approves — no changes needed | Does coder avoid unnecessary edits? |

## How it works now (single-agent simulation)

Each eval case embeds the reviewer's feedback inline as a `[REVIEW]` section. The coder agent sees the task + review in one prompt. This simulates the multi-agent flow without requiring Scene runtime.

## Future: multi-agent Scene (0.4+)

In the full version, the reviewer is a real second agent:

```toml
[[agents]]
name = "coder"
agent = "claude-agent-acp"
model = "claude-haiku-4-5-20251001"

[[agents]]
name = "reviewer"
agent = "gemini"
model = "gemini-3-pro-preview"
```

The coder solves, reviewer reviews, coder revises. BenchFlow orchestrates the handoff via the Scene runtime.

## Using TB2 tasks

For a larger-scale experiment, run Terminal-Bench 2.0 tasks with multi-turn:

```yaml
# benchmarks/tb2-followup.yaml
tasks_dir: .ref/terminal-bench-2/tasks
agent: claude-agent-acp
prompts:
  - null   # task instruction
  - "An independent reviewer checked your work and found issues. Review your solution, check for errors, test it, and fix any issues."
```

Compare single-turn vs multi-turn scores across agents to measure review lift.
