# BenchFlow v0.5 Integration Handoff

Generated: 2026-05-23

This handoff is for the `codex/v05-integration-merge-main` branch targeting
`v0.5-integration`.

## Current State

- Branch head before this handoff doc: `e70d724`
- Base branch for PR: `origin/v0.5-integration`
- Latest production dashboard after Linear grooming:
  `https://dashboard-benchflow.vercel.app`
- Latest deploy verified:
  `https://dashboard-1an74s6pw-benchflow.vercel.app`
- Dashboard data after grooming: 50 Linear issues, 15 active issues, 158
  visible rollout rows, 77 archived jobs, 5 archived runs.
- Do not use Docker unless the goal is specifically testing Docker. Use
  Daytona for all cloud validation and large experiments.

## What This Branch Adds

The branch contains five integration commits over `v0.5-integration`:

1. Fix Daytona live env transport secret handling.
2. Fix Daytona PTY env transport secret handling.
3. Fix Daytona verifier and upload setup.
4. Fix dashboard evidence visibility and ACP self-gen handoff.
5. Document SkillsBench rollout audits.

Important code areas:

- `src/benchflow/sandbox/process.py`
- `src/benchflow/sandbox/daytona.py`
- `src/benchflow/sandbox/_compose.py`
- `src/benchflow/sandbox/lockdown.py`
- `src/benchflow/task/verifier.py`
- `src/benchflow/rollout.py`
- `src/benchflow/self_gen.py`
- `dashboard/generate.py`
- `dashboard/serve.py`
- `dashboard/index.html`

Important durable reports:

- `docs/reports/2026-05-23-skillsbench-full94-baseline-audit.md`
- `docs/reports/2026-05-23-skillsbench-3mode-subset-audit.md`

## Handoff Verdict

BenchFlow v0.5 is substantially closer, but not release-ready as a benchmark
product. SDK/orchestration plumbing is real, dashboard data is live, and the
full SkillsBench baseline did execute current SkillsBench `main`. The remaining
release blockers are evidence quality, self-gen post-fix validation, failure
semantics, and full three-mode coverage.

Do not call the full 94-task baseline a clean release validation set. It is a
useful evidence run with several invalid/no-reward infra measurements and
verifier-contract failures that must be repaired or explicitly excluded.

## Linear Grooming State

All `ENG-148` through `ENG-161` have triage comments with source and trajectory
evidence.

Parent/state changes already applied:

| Issue | State | Parent | Status |
|---|---|---|---|
| `ENG-148` | Todo | `ENG-130` | Confirmed ACP transport rc=255 invalid measurement |
| `ENG-149` | Todo | `ENG-130` | Confirmed idle-timeout diagnostic/scoring gap |
| `ENG-150` | Todo | `ENG-130` | Confirmed verifier contract/classification bug |
| `ENG-151` | Todo | `ENG-130` | Confirmed verifier dependency/index failure |
| `ENG-152` | Todo | `ENG-130` | Confirmed quantum verifier timeout/heavy-task gap |
| `ENG-153` | Todo | `ENG-127` | Confirmed structured artifact contract gap |
| `ENG-154` | Todo | `ENG-98` | Partially done; needs clean post-fix reruns |
| `ENG-155` | Todo | `ENG-126` | Not done; self-gen run predates fix |
| `ENG-156` | Todo | `ENG-98` | Not done; full 94 x 3 modes missing |
| `ENG-157` | Todo | none | Not done; stale dashboard advisory still open |
| `ENG-158` | Todo | none | Not done; file:// fetch UX still poor |
| `ENG-159` | Todo | `ENG-131` | Not done; CLI lacks first-class `--include` |
| `ENG-160` | Todo | `ENG-127` | Not done; resume/job_name orphan risk remains |
| `ENG-161` | In Review | `ENG-155` | Done as audit work; close after human acceptance |

## Previous Run Audit Summary

### Full 94-Task SkillsBench Baseline

Run root:

```text
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64
```

Summary:

- Source: `benchflow-ai/skillsbench@main`
- Resolved SHA: `20149520474cfc8d7eb3c8000ec403d10145a9fd`
- Current `main` verified at the same SHA during audit.
- Agent/model/env: `gemini` / `gemini-3.1-flash-lite-preview` / `daytona`
- Concurrency: 64
- Summary total: 94 tasks
- Rollout rows including retries: 109
- Final outcomes: 8 pass / 76 fail / 7 error / 3 verifier_error
- Retry-row outcomes: 8 pass / 80 fail / 16 error / 5 verifier_error

Passing tasks:

- `3d-scan-calc`
- `citation-check`
- `econ-detrending-correlation`
- `mars-clouds-clustering`
- `parallel-tfidf-search`
- `pddl-tpp-planning`
- `radar-vital-signs`
- `spring-boot-jakarta-migration`

Invalid or release-blocking measurement classes:

- Agent install failed: `fix-visual-stability`, `gh-repo-analytics`,
  `pedestrian-traffic-counting`
- Artifact copy/setup failed: `drone-planning-control`,
  `pg-essay-to-audiobook`
- ACP transport/stdout closed: `mars-clouds-clustering`,
  `quantum-numerical-simulation`, `video-filler-word-remover`
- Verifier timeout: `quantum-numerical-simulation`
- Verifier contract/refusal after reward zero: `dynamic-object-aware-egomotion`,
  `threejs-structure-parser`, `threejs-to-obj`
- Idle timeout diagnostics gap: `court-form-filling`,
  `shock-analysis-supply`, `video-tutorial-indexer`

Durable audit:

```text
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/docs/reports/2026-05-23-skillsbench-full94-baseline-audit.md
```

### 9-Task Three-Mode SkillsBench Subset

Run root:

```text
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset
```

Modes:

- `baseline`: 0/9 pass, 8 fail, 1 verifier_error
- `with-task-skills`: 1/9 pass, 6 fail, 1 error, 1 verifier_error
- `self-gen`: 0/9 pass, 8 fail, 1 verifier_error

Only pass in the 27-row subset:

- `lake-warming-attribution` in `with-task-skills` mode

Critical caveat:

The self-gen run started before commit `4946481`, so it does not validate the
current ACP-native generated-skill handoff. It found that only
`creator-skills/skill-creator` was persisted under `_self_gen`; no generated
solver skill `SKILL.md` was persisted outside creator scaffolding.

Durable audit:

```text
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/docs/reports/2026-05-23-skillsbench-3mode-subset-audit.md
```

## Actual Trajectory Paths

### 9-Task Three-Mode Subset

All 27 task-mode rows have ACP trajectory files:

```text
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/data-to-d3__534b3e2e/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/grid-dispatch-operator__10b5966a/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/jax-computing-basics__f523cdb4/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/jpg-ocr-stat__97d1d558/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/lake-warming-attribution__daffb946/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/python-scala-translation__ac3cca33/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/shock-analysis-supply__de031190/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/threejs-to-obj__1d751567/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/baseline/2026-05-23__00-31-36/weighted-gdp-calc__5be9271b/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/data-to-d3__f4fd1d1c/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/grid-dispatch-operator__76c38257/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/jax-computing-basics__8c52e991/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/jpg-ocr-stat__6b929da7/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/lake-warming-attribution__128f6494/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/python-scala-translation__c3ff7696/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/shock-analysis-supply__57c6e480/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/threejs-to-obj__b68a7407/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/with-task-skills/2026-05-23__00-37-02/weighted-gdp-calc__879537c2/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/data-to-d3__b846c3e8/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/grid-dispatch-operator__8bb95f14/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/jax-computing-basics__2b9a9b0a/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/jpg-ocr-stat__310a5a20/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/lake-warming-attribution__39207e39/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/python-scala-translation__d766a9ce/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/shock-analysis-supply__d7e84e66/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/threejs-to-obj__f84eb30a/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/2026-05-23__00-44-33/weighted-gdp-calc__971d72ce/trajectory/acp_trajectory.jsonl
```

### Full-Run P0/P1 Trajectory Examples

The full 94-task audit report contains the complete rollout ledger. The most
important repair targets are:

```text
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/video-filler-word-remover__4b9e74b6/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/quantum-numerical-simulation__3031c9cf/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/quantum-numerical-simulation__a43c7238/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/dynamic-object-aware-egomotion__c905a992/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/threejs-structure-parser__94aa9b66/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/threejs-to-obj__fd7de03b/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/simpo-code-reproduction__82e67221/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/court-form-filling__4d161ed7/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/shock-analysis-supply__d95da6be/trajectory/acp_trajectory.jsonl
/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/2026-05-22__16-54-22/video-tutorial-indexer__7fe67e9d/trajectory/acp_trajectory.jsonl
```

## How To Fix All Previous-Run Issues

Work in small vertical slices. For each concrete fix, follow the operating
rule from the goal: one fix subagent using TDD, then four audit/test/verify
subagents. Keep commits scoped.

### P0: Invalid Measurements

1. `ENG-148` ACP rc=255:
   - Add structured result metadata for process source: agent process,
     Daytona shell/SSH/session, ACP protocol, or sandbox lifecycle.
   - Capture last sandbox health probe and process exit evidence in
     `result.json`.
   - Rerun `video-filler-word-remover` on Daytona.

2. `ENG-149` idle timeouts:
   - Keep watchdog counters, but surface idle timeout as a structured
     invalid-measurement class when partial trajectory or timeout error is
     present.
   - Dashboard should badge `error + reward` rows as diagnostic/invalid when
     the agent did not complete cleanly.
   - Rerun `court-form-filling`, `shock-analysis-supply`,
     `video-tutorial-indexer`, and self-gen `grid-dispatch-operator`.

3. `ENG-150` verifier rc=1 after reward zero:
   - Do not globally accept reward files from failed verifiers without a
     contract decision.
   - Preferred fix: SkillsBench verifiers that intentionally score reward 0
     should exit 0 after writing `reward.txt` and CTRF.
   - Rerun `dynamic-object-aware-egomotion`, `threejs-structure-parser`, and
     `threejs-to-obj`.

4. `ENG-151` `simpo-code-reproduction` dependency/index:
   - Fix the SkillsBench verifier dependency declaration or uv index policy for
     `torch==2.1.2+cpu`.
   - If dependency setup fails, classify it as structured setup failure rather
     than ordinary reward zero.
   - Rerun only `simpo-code-reproduction` first.

5. `ENG-152` quantum verifier timeout:
   - Profile verifier runtime.
   - Either reduce verifier runtime, raise timeout only for this heavy task, or
     mark it outside the release-gated subset.
   - Rerun `quantum-numerical-simulation`.

6. `ENG-153` structured verifier artifacts:
   - Define canonical artifact names: `reward.txt`, `ctrf.json`, verifier
     stdout/stderr, and optional task-specific attachments.
   - Treat `ctrf-report.json` as an alias only during migration.
   - Make `check_results.py` report missing structured evidence by task, not
     just by rollout schema.

### P1: Evidence And DX Gaps

1. `ENG-154` invalidated SkillsBench rows:
   - The branch has fixes for JS ACP install and Daytona upload setup.
   - Acceptance still needs clean rerun evidence for every invalidated task.

2. `ENG-155` post-fix self-gen:
   - Rerun the 9-task self-gen subset after `4946481`.
   - Audit that generated solver skills are persisted and activated.

3. `ENG-156` full 94 x 3:
   - Run all 94 SkillsBench tasks in baseline, with-task-skills, and self-gen
     modes with Gemini 3.1 Flash Lite on Daytona.
   - Use staged concurrency: 16 to shake out infra, 64 for release validation,
     then 100 only for large validation.

4. `ENG-157` stale dashboard advisory:
   - Resolve, archive, or update `OPEN-2` in `dashboard/generate.py`.
   - Regenerate and redeploy dashboard.

5. `ENG-158` file:// fetch UX:
   - Update `dashboard/index.html` error recovery text to direct users to
     `uv run python dashboard/serve.py` and `http://localhost:8777/`.

6. `ENG-159` CLI include:
   - Add `bench eval create --include` and wire it to existing
     `Evaluation.from_yaml` `include_tasks` behavior.
   - Cover with CLI tests and one real subset dry run.

7. `ENG-160` resume/job_name:
   - Scope resume scanning to the exact job root/name.
   - Do not recursively include sibling/orphan `result.json` files.
   - Add regression tests using orphan retry dirs.

## How To Complete The Vision Fast

### Fast Sequence

1. Land this PR into `v0.5-integration`.
2. Immediately fix P0 invalid-measurement semantics:
   `ENG-148`, `ENG-149`, `ENG-150`, `ENG-153`.
3. Rerun the invalidated task set only, using Daytona:
   install-failed tasks, upload-failed tasks, verifier-contract tasks,
   timeout tasks.
4. Rerun 9-task self-gen on the fixed branch and audit all generated skills:
   this closes or unblocks `ENG-155`.
5. Run the full SkillsBench 94 x 3 modes with Gemini 3.1 Flash Lite:
   this is the real `ENG-156` gate.
6. Run the adapter and architecture lanes in parallel:
   trace-to-task, Harvey LAB, ProgramBench, Terminal-Bench-style smoke,
   hosted env compatibility, sandbox/agent decoupling.
7. Regenerate dashboard, redeploy production, and close only issues with
   audited evidence.

### Release Evidence Standard

Every release-gating run should have:

- `summary.json`
- one `result.json` per rollout
- nonempty `trajectory/acp_trajectory.jsonl`
- `reward.txt`
- structured verifier report (`ctrf.json` preferred)
- verifier stdout/stderr artifacts
- source repo and resolved SHA
- model, sandbox, concurrency, timeout, and retry policy in the summary
- dashboard visibility after generation
- one durable audit report under `docs/reports/`

### Concurrency Plan

- Use concurrency 8 to 16 for diagnosis and post-fix shakedown.
- Use concurrency 64 for release validation.
- Use concurrency 100 only after no invalid-measurement classes remain.

### What Not To Do

- Do not count pre-`4946481` self-gen evidence as validation of current
  ACP-native generated-skill handoff.
- Do not mark reward-zero verifier crashes as solved until the verifier
  contract is explicit.
- Do not use Docker except for Docker-specific sandbox tests.
- Do not close Linear release blockers based only on passing unit tests.
- Do not ship v0.5 as "validated" until full 94 x 3 and adapter release lanes
  have audited evidence.

## Useful Commands

```bash
set -a; source .env; set +a
uv run python tests/integration/check_results.py <jobs-root>
uv run python tests/integration/check_adapter_evidence.py <jobs-root>
uv run python tests/integration/check_trace_to_task_evidence.py <jobs-root>
uv run ruff check .
uv run ty check src
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -p no:cacheprovider tests -q
uv run python dashboard/generate.py
```

## Dashboard And Linear

Latest dashboard deployment after grooming:

```text
https://dashboard-1an74s6pw-benchflow.vercel.app
https://dashboard-benchflow.vercel.app
```

The dashboard was verified to show:

- `ENG-148` through `ENG-161`
- `ENG-161` in `In Review`
- `ENG-148` parented to `ENG-130`
- `ENG-161` parented to `ENG-155`
- 50 total Linear issues
- 158 visible rollout rows
- 77 archived jobs
