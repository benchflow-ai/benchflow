# Models-as-Skills

Use one model as a "specialist" skill that another model can invoke. Test whether a generalist orchestrator + code specialist combo outperforms the orchestrator alone.

## Concept

```
┌─────────────────────────────────────────┐
│ Orchestrator (e.g. Gemini Flash)        │
│                                         │
│  "This needs a topo sort algorithm..."  │
│         │                               │
│         ▼                               │
│  ┌─────────────────────┐               │
│  │ Code Specialist Skill│               │
│  │ (e.g. GPT-5.4-nano) │               │
│  └──────────┬──────────┘               │
│             │                           │
│  "Here's the implementation..."         │
│         │                               │
│         ▼                               │
│  Agent integrates + tests               │
└─────────────────────────────────────────┘
```

## Quick start

```bash
# With specialist skill (model-as-skill installed)
bench skills eval ./benchmarks/models-as-skills/ -a gemini --no-baseline

# Compare: with vs without specialist
bench skills eval ./benchmarks/models-as-skills/ -a gemini
```

## Cases

| ID | Task | Why specialist helps |
|----|------|---------------------|
| `topo-sort-with-cycle-detection` | Graph algorithm + cycle detection | Requires specific algorithmic knowledge |
| `regex-email-parser` | Regex pattern matching | Regex is notoriously tricky; specialists are more precise |
| `optimize-quadratic-to-nlogn` | Algorithm optimization | Requires seeing the O(n) hash-set pattern |

## How it works

The SKILL.md tells the orchestrator agent *when* and *how* to delegate to a specialist. In a full models-as-skills setup:

1. The specialist model runs as an MCP sidecar tool inside the sandbox
2. The orchestrator calls it via `ask_specialist("implement topo sort with cycle detection")`
3. The specialist returns code that the orchestrator integrates

For this prototype, the skill provides guidance on delegation patterns. The with/without comparison shows whether the skill's delegation framework improves code quality.

## Future: MCP sidecar

In 0.4+, models-as-skills will use the Scene runtime with an actual MCP sidecar:

```toml
# task.toml
[[environment.services]]
name = "code-specialist"
image = "benchflow/specialist:gpt-5.4-nano"
protocol = "mcp"
```

The orchestrator gets an `ask_specialist` tool backed by a real model endpoint.
