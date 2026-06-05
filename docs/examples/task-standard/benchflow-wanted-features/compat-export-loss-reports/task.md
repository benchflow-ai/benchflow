---
schema_version: "1.3"
source: benchflow/wanted-features/compat-export-loss-reports
task:
  name: benchflow-wanted/compat-export-loss-reports
  description: Add Harbor/Pier split-layout export with explicit degraded-export reports.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, harbor, pier, export]
metadata:
  category: benchflow-feature
  feature_area: adapters
agent:
  timeout_sec: 7200
  network_mode: no-network
verifier:
  timeout_sec: 1200
  user: root
environment:
  docker_image: ghcr.io/astral-sh/uv:python3.12-bookworm
  network_mode: no-network
  cpus: 4
  memory_mb: 8192
  storage_mb: 10240
  workdir: /repo
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
  document_version: "0.3"
  traceability:
    need_ids: [F1, F2, F6, F7]
    user_story: "Adapter maintainers need native task.md packages to export back to Harbor/Pier and report every native-only concept that target formats cannot express."
    acceptance:
      - "native packages export task.toml, instruction.md, solution/, and tests/"
      - "export report includes selected definitions, file hashes, alias collisions, and semantic losses"
      - "foreign import mode preserves unknown Harbor/Pier keys under compatibility metadata instead of native root-key creep"
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
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    entrypoint: verifier/test.sh
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

