---
profile: [code-change, harbor-compatible]
source: benchflow/wanted-features/compat-export-loss-reports
name: benchflow-wanted/compat-export-loss-reports
image: ghcr.io/astral-sh/uv:python3.12-bookworm
verifier: verifier/
oracle: oracle/
task:
  description: Add Harbor/Pier split-layout export with explicit degraded-export reports.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, harbor, pier, export]
metadata:
  feature_area: adapters
agents:
  roles:
    adapter_engineer:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: xhigh
      capabilities: [code-edit, tests]
    compatibility_reviewer:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      capabilities: [review]
scenes:
  - name: implement-export
    turns:
      - role: adapter_engineer
  - name: compatibility-review
    turns:
      - role: compatibility_reviewer
benchflow:
  compatibility:
    harbor:
      export: degraded
      emits:
        config: task.toml
        prompt: instruction.md
        oracle: solution/
        verifier: tests/
      losses: [benchflow.teams, benchflow.nudges, verifier.verifier_md]
    pier:
      export: degraded
      losses: [benchflow.teams, benchflow.nudges, verifier.verifier_md]
---

## prompt

Implement native-to-Harbor/Pier export and loss reporting.

The exporter must emit the split layout (`task.toml`, `instruction.md`,
`solution/`, `tests/`) for supported fields and a machine-readable report for
anything lost or degraded. Import mode must preserve unknown foreign extension
keys under compatibility metadata and warn, while native authoring remains
strict.

## role:adapter_engineer

Focus on canonical equivalence. Config equality should use `TaskConfig`, prompt
equality should normalize text, and verifier/oracle equality should use stable
file hash maps.

## role:compatibility_reviewer

Review for accidental root-key expansion, missing loss reports, and any wording
or behavior that treats Harbor/Pier split layout as deprecated.
