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
    user_story: "Adapter maintainers need native task.md packages to export back to Harbor/Pier, prove split round-trip conformance for supported fields, and report every native-only concept that target formats cannot express."
    acceptance:
      - "native packages export task.toml, instruction.md, solution/, and tests/"
      - "export report includes selected definitions, file hashes, alias collisions, and semantic losses"
      - "foreign import mode preserves unknown Harbor/Pier keys under compatibility metadata instead of native root-key creep"
      - "split -> task.md -> split conformance proves canonical config, prompt, environment, solution, and tests file-map equality"
      - "structural checks reject mixed native/legacy drift by default"
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
  evidence:
    oracle_runs:
      required_reward: 1.0
      artifact: evidence/acceptance/oracle-run.json
    verifier:
      reruns: 3
      flake_rate: 0.0
      report: evidence/acceptance/verifier-stability-report.json
    review:
      anti_cheat: passed
      instruction_alignment: passed
      reviewer: benchflow-task-standard-dogfood
      artifact: evidence/acceptance/review.json
    calibration:
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      partial_solution_range: [0.4, 0.8]
      report: evidence/acceptance/calibration-report.json
      human_or_reference_examples:
        - name: gold-export-loss-report
          expected_reward: 1.0
          artifact: evidence/acceptance/gold-result.json
    trajectories:
      - path: evidence/acceptance/gold-trajectory.jsonl
        sha256: 1db637923331edab23a8a01fa75470669dbaccfec4fee2d04e05ac6dc6d1b98b
    artifacts:
      - path: evidence/acceptance/oracle-run.json
        sha256: c5c234231cef24ffca2bb07da57d53d2095492488c182d8780293d67019a2d76
      - path: evidence/acceptance/gold-result.json
        sha256: 22dd90c1b2c4961ed0e36a201bbb813244992d9a89164b9c3ae5d414cca3ec3e
      - path: evidence/acceptance/calibration-report.json
        sha256: c57192637876677d496d43b2eec5af37ccd626df83a4ea5826c02b2f624dde64
      - path: evidence/acceptance/verifier-stability-report.json
        sha256: 7e10f542ad8f1b517c45d351a60d9c151ac74cf4c439ec59e83fcf55590a1335
      - path: evidence/acceptance/review.json
        sha256: 035efc910b7e1fbe9c45f55e6f4809494cf4a00c7320c2d1c22a0d3157b6d131
---

## prompt

Implement native-to-Harbor/Pier export and loss reporting.

The exporter must emit the split layout (`task.toml`, `instruction.md`,
`solution/`, `tests/`) for supported fields and a machine-readable report for
anything lost or degraded. Import mode must preserve unknown foreign extension
keys under compatibility metadata and warn, while native authoring remains
strict. Harbor-compatible split tasks should also have a pure conformance check
that migrates to `task.md`, exports back to split layout, and compares canonical
config, prompt, environment, solution, and tests hashes.

## role:adapter_engineer

Focus on canonical equivalence. Config equality should use `TaskConfig`, prompt
equality should normalize text, and verifier/oracle equality should use stable
file hash maps.

## role:compatibility_reviewer

Review for accidental root-key expansion, missing loss reports, and any wording
or behavior that treats Harbor/Pier split layout as deprecated.
