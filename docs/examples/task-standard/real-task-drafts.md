# Real Task Standard Drafts

These are draft `task.md` examples for the v0.3 standard. They are intentionally
kept outside `docs/examples/task-md/**/task.md` because they use proposed
`benchflow:` fields that are parsed as raw metadata today but not executed by
the runtime yet.

Each draft should name the concrete fork, benchmark family, or BenchFlow
architecture need that justifies its proposed fields. Real draft tasks should
also show verifier package intent, not only `test.sh`. Use this pattern when a
task has hidden tests, LLM judging, agent judging, or nontrivial reward shaping:

```yaml
benchflow:
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    structured_rubric: verifier/rubrics/verifier.toml
    entrypoint: verifier/test.sh
    reward_kit: verifier/reward_kit/
    judge_agent:
      role: verifier_judge
      type: deterministic | llm-judge | agent-judge | hybrid
      input_scope: [declared_deliverables, agent_trajectory, verifier_fixtures]
      output: /logs/verifier/judge_result.json
  evidence:
    calibration:
      oracle_reward: 1.0
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      flake_reruns: 5
      flake_rate: 0.0
```

For Harbor/Pier compatibility, this verifier package exports as `tests/`.
Split-layout imports that only have `tests/test.sh` remain valid but are not
publication-grade native verifier packages until verifier intent, rubric, and
calibration evidence are supplied.

## SWE-bench Style Repo Issue

```md
---
schema_version: "1.3"
source: "benchflow-ai/swebenchpro/instance_qutebrowser__qutebrowser-01d1d1494411380d97cac14614a829d3a69cecaf-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
task:
  name: benchflow-drafts/swebench-qutebrowser-regression
  description: Fix a qutebrowser regression against hidden pytest tests.
metadata:
  category: software-engineering
  source_standard: swe-bench
  repo: qutebrowser/qutebrowser
  base_commit: "01d1d1494411380d97cac14614a829d3a69cecaf"
agent:
  timeout_sec: 3600
  network_mode: no-network
verifier:
  timeout_sec: 1200
  user: root
  hardening:
    cleanup_conftests: false
environment:
  docker_image: ghcr.io/benchflow/swebenchpro/qutebrowser:01d1d149
  network_mode: no-network
  cpus: 4
  memory_mb: 8192
  storage_mb: 10240
  workdir: /app
oracle:
  env:
    GOLD_PATCH: /oracle/gold.patch
artifacts:
  - source: /logs/artifacts/patch.diff
    destination: patch.diff
agents:
  roles:
    solver:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: high
      capabilities: [code-edit, tests]
scenes:
  - name: solve
    turns: [solver]
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F6]
    user_story: "Adapter maintainers need SWE-bench provenance, hidden test assets, and honest export loss reports."
    acceptance:
      - "base commit, gold patch hash, and hidden test patch hash survive import"
      - "Harbor/Pier export reports losses for anti-cheat and provenance fields"
  provenance:
    repo: qutebrowser/qutebrowser
    base_commit: "01d1d1494411380d97cac14614a829d3a69cecaf"
    dataset_row_id: qutebrowser__qutebrowser-01d1d1494411380d97cac14614a829d3a69cecaf
    gold_patch_sha256: "<hidden>"
    test_patch_sha256: "<hidden>"
  assets:
    - path: verifier/test_outputs.py
      visibility: hidden_verifier
      kind: hidden_pytest
      sha256: "<hash>"
    - path: verifier/test.patch
      visibility: hidden_verifier
      kind: test_patch
      sha256: "<hash>"
    - path: oracle/gold.patch
      visibility: hidden_oracle
      kind: gold_patch
      sha256: "<hash>"
  evidence:
    oracle_runs:
      required_reward: 1.0
    anti_cheat:
      forbid_paths: [verifier/test.patch, oracle/gold.patch]
    calibration:
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      flake_reruns: 5
      flake_rate: 0.0
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    entrypoint: verifier/test.sh
    implementation:
      type: test-script
      hidden_inputs: [verifier/test.patch, verifier/test_outputs.py]
      outputs:
        reward_json: /logs/verifier/reward.json
        reward_details: /logs/verifier/reward-details.json
  compatibility:
    harbor:
      export: degraded
      losses:
        - benchflow.provenance
        - benchflow.assets
        - benchflow.evidence.anti_cheat
---

## prompt

Fix the qutebrowser checkout in `/app` for the upstream issue summarized in
`/app/issue.md`. Preserve public behavior and make the hidden regression tests
pass.

## role:solver

Work like a maintainer: inspect the failing behavior, make a minimal patch, and
run the local qutebrowser tests that cover the touched area.
```

Selected runtime gaps exposed:

- hidden test patch application is not first-class
- repo/base-commit provenance is metadata only
- anti-cheat path checks are not generated from `benchflow.evidence`

## FrontierSWE Heavy Task

```md
---
schema_version: "1.3"
source: "frontierswe/inference-system-optimization"
task:
  name: benchflow-drafts/frontierswe-sglang-b200
  description: Optimize an SGLang inference server under correctness and latency constraints.
metadata:
  category: frontier-swe
  benchmark_family: FrontierSWE
  private_assets: model weights mounted read-only by provider
agent:
  timeout_sec: 21600
  network_mode: no-network
verifier:
  timeout_sec: 7200
  environment_mode: separate
  environment:
    docker_image: ghcr.io/frontierswe/private/sglang-verifier:2026-06
    network_mode: no-network
    gpus: 1
    gpu_types: [B200]
environment:
  docker_image: ghcr.io/frontierswe/private/sglang-b200:2026-06
  network_mode: no-network
  cpus: 8
  memory_mb: 65536
  storage_mb: 10240
  gpus: 1
  gpu_types: [B200]
  workdir: /workspace/sglang
multi_step_reward_strategy: final
steps:
  - name: correctness
    min_reward: 0.5
  - name: throughput-latency
    min_reward:
      tokens_per_second: 0.9
      p95_latency: 0.8
agents:
  roles:
    optimizer:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: xhigh
      capabilities: [code-edit, shell, gpu-profile]
scenes:
  - name: optimize
    turns: [optimizer]
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F3, F4]
    user_story: "FrontierSWE task owners need private GPU assets, no-search policy, separate verifier execution, and rich reward JSON without forking root task syntax."
    acceptance:
      - "unsupported sandboxes fail closed for B200, private mounts, and persistent volume"
      - "supported sandboxes preserve private asset metadata and structured reward JSON"
  runtime_policy:
    no_search: true
    no_public_internet: true
    allowed_tools: [shell, editor]
    forbidden_tools: [web_search, browser_fetch]
    persistent_volume: required
  private_mounts:
    - source: modal-volume://frontierswe-models/qwen
      target: /mnt/models/qwen
      mode: ro
  assets:
    - path: /mnt/models/qwen
      visibility: agent
      kind: external_provider_mount
      sha256: "<hash>"
  evidence:
    reward_json:
      preserve: true
      expected_keys: [reward, metrics, regressions, reasons, raw]
    calibration:
      oracle_reward: 1.0
      no_op_reward_max: 0.1
      known_bad_reward_max: 0.4
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/performance.md
    structured_rubric: verifier/reward_kit/criteria.toml
    entrypoint: verifier/test.sh
    reward_kit: verifier/reward_kit/
    implementation:
      type: hybrid
      strategies:
        deterministic: verifier/test.sh
        rewardkit: verifier/reward_kit/
      outputs:
        reward_json: /logs/verifier/reward.json
        reward_details: /logs/verifier/reward-details.json
      aggregate_policy:
        field: reward
        fallback: weighted_mean
  compatibility:
    harbor:
      export: degraded
      losses: [benchflow.runtime_policy, benchflow.private_mounts]
    pier:
      export: degraded
      losses: [benchflow.runtime_policy, benchflow.private_mounts]
---

## prompt

Improve `/workspace/sglang/benchflow_launch.py` and any server code needed to
increase decode throughput for the verifier workload while preserving exactness
tolerances.

## role:optimizer

Profile first, keep a reproducible launch script, and leave a short benchmark
note in `/workspace/sglang/RESULTS.md`.
```

Selected runtime gaps exposed:

- B200 GPU scheduling, private provider mounts, and separate verifier runtime
  need backend capability negotiation
- no-search agent semantics are launch-policy specific
- persistent volumes and process cleanup are backend capabilities, not task
  parser features

## Browser / Desktop Workflow

```md
---
schema_version: "1.3"
source: "xlang-ai/OSWorld/evaluation_examples/examples/multi_apps/5990457f-2adb-467b-a4af-5c857c92d762.json"
task:
  name: benchflow-drafts/osworld-yann-lecun-researchers-xlsx
  description: Append Yann LeCun's Google Scholar entry to researchers.xlsx using Chrome and LibreOffice Calc.
metadata:
  category: browser-desktop
agent:
  timeout_sec: 1800
verifier:
  timeout_sec: 600
  env:
    DISPLAY: ":0"
  pytest_plugins: [pytest_playwright]
environment:
  docker_image: ghcr.io/benchflow/osworld-ubuntu22:chrome-calc
  network_mode: allowlist
  allowed_hosts: [huggingface.co, scholar.google.com]
  cpus: 4
  memory_mb: 8192
  workdir: /home/user
agents:
  roles:
    desktop_agent:
      agent: openhands
      model: claude-sonnet-4-6
      capabilities: [browser, desktop, file-edit]
scenes:
  - name: desktop-workflow
    turns: [desktop_agent]
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F5, F6]
    user_story: "CUA/browser benchmark authors need desktop setup, browser action spaces, deterministic getters, and multimodal evidence in a document package."
    acceptance:
      - "setup/reset/readiness hooks are explicit environment-plane data"
      - "screenshot, accessibility, and workbook evidence are declared artifacts"
  desktop:
    kind: osworld
    id: "5990457f-2adb-467b-a4af-5c857c92d762"
    setup:
      - type: launch
        command: [google-chrome, "--remote-debugging-port=1337"]
      - type: launch
        command: [socat, "tcp-listen:9222,fork", "tcp:localhost:1337"]
      - type: download
        url: "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/5990457f-2adb-467b-a4af-5c857c92d762/researchers.xlsx"
        path: /home/user/Desktop/researchers.xlsx
      - type: launch
        command: [nautilus, /home/user/Desktop]
    browser:
      engine: chromium
      observation_space: [screenshot, accessibility_tree, terminal]
      action_space: [pyautogui, computer_use]
    verifier_getters:
      result:
        type: content_from_vm_file
        path: /home/user/Desktop/researchers.xlsx
        file_type: xlsx
        file_content: last_row
      expected:
        type: info_from_website
        url: "https://scholar.google.com/citations?user=WLN3QrAAAAAJ&hl=en"
      metric:
        func: literal_match
        options:
          type: list
          ignore_case: true
---

## prompt

Append one entry for AI researcher Yann LeCun from Google Scholar into the
existing table `/home/user/Desktop/researchers.xlsx`.

## role:desktop_agent

Use the browser to gather the current scholar fields, then update the
spreadsheet without changing existing rows.
```

Selected runtime gaps exposed:

- setup/reset/readiness hooks should be environment-plane concepts
- browser action/observation spaces are not modeled today
- deterministic verifier getters need a typed scorer interface

## Multi-Agent NudgeBench Team

```md
---
schema_version: "1.3"
source: "benchmarks/clawsbench/tasks/archive-amazon-shipping"
task:
  name: benchflow-drafts/team-archive-amazon-shipping
  description: Team workflow over mock Gmail with simulated user nudges and file handoffs.
metadata:
  category: simulated-user
agent:
  timeout_sec: 1800
verifier:
  timeout_sec: 300
environment:
  network_mode: no-network
  cpus: 2
  memory_mb: 4096
agents:
  roles:
    concierge:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      capabilities: [user-dialogue]
    operator:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: high
      capabilities: [tool-use, code-edit]
    auditor:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      capabilities: [review]
scenes:
  - name: intake
    turns:
      - role: concierge
  - name: fulfill
    turns:
      - role: operator
  - name: audit
    turns:
      - role: auditor
user:
  model: claude-haiku
  stop_rule: satisfied-or-4-rounds
  private_facts:
    sender: shipment-tracking@amazon.com
    action: archive_not_delete
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F5]
    user_story: "Agent/viewer fork maintainers need teams, handoffs, simulated user nudges, and review artifacts without changing Harbor-compatible root config."
    acceptance:
      - "role and scene prompts compose deterministically"
      - "user nudges are executable by a supported runtime or reported as metadata-only"
      - "handoff artifacts are recorded in trajectory evidence"
  teams:
    support:
      roles: [concierge, operator, auditor]
      handoff:
        trajectory_visibility: summaries
        workspace_visibility: shared
        artifacts:
          - /app/handoff/customer.json
          - /app/handoff/action_plan.md
          - /app/handoff/review.md
  nudges:
    mode: simulated-user
    branchable: true
    nudge_budget: 4
    scripted:
      - trigger: asked_for_specific_sender
        reveal: sender
      - trigger: agent_about_to_delete
        reveal: "Do not delete or trash; remove INBOX only."
    confirmation_policy:
      destructive_actions: human
  prompt:
    composition: append
    order: [base, role, scene, turn]
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/gmail-state.md
    structured_rubric: verifier/rubrics/gmail-state.toml
    entrypoint: verifier/test.sh
    judge_agent:
      role: verifier_judge
      type: agent-judge
      model: claude-sonnet-4-6
      input_scope: [declared_deliverables, agent_trajectory, verifier_fixtures]
      output: /logs/verifier/judge_result.json
      isolation: read_only
    implementation:
      type: hybrid
      strategies:
        deterministic: verifier/test.sh
        judge: verifier/judges/reviewer.md
      outputs:
        reward_json: /logs/verifier/reward.json
        reward_details: /logs/verifier/reward-details.json
  evidence:
    calibration:
      oracle_reward: 1.0
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      judge_agreement:
        required: true
        sample_count: 5
        min_pairwise_agreement: 0.8
---

## prompt

Help a user clean up Gmail. The final state must archive exactly the Amazon
shipping confirmation and leave all other messages untouched.

## role:concierge

Interview the user and write `/app/handoff/customer.json`. Ask a targeted
follow-up if the sender or action is ambiguous.

## role:operator

Read the handoff, call the mock Gmail API at `http://localhost:9001`, and write
`/app/handoff/action_plan.md`.

## role:auditor

Verify the action plan and final Gmail state. Write `/app/handoff/review.md`.

## user-persona

You are impatient and initially say only "please get the Amazon shipping email
out of my inbox." Reveal exact sender details only if asked why they are needed.
```

Selected runtime gaps exposed:

- `user` and `benchflow.nudges` are parsed-only today
- the mock Gmail API requires an environment-plane service declaration, setup
  hook, and readiness check
- team handoff visibility is not enforced
- prompt composition is fallback-based, not append/replace-based
