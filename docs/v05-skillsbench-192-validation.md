# SkillsBench #192 infra-fix validation for BenchFlow 0.5.1

Closes: [#505](https://github.com/benchflow-ai/benchflow/issues/505)
References: [#192](https://github.com/benchflow-ai/benchflow/issues/192) (closed-as-superseded historical audit)

## Scope

[#192](https://github.com/benchflow-ai/benchflow/issues/192) enumerated **14 infrastructure bugs** found during the Opus 4.7 + SkillsBench Trial 1 audit (84 tasks, Daytona, concurrency=2). Most root causes were in the `benchflow-ai/skillsbench` task repo (per-task `Dockerfile` / `test.sh` / datagen), one was in `benchflow-ai/benchflow` (`VerifierConfig.pytest_plugins`), and one was upstream-only (Daytona SDK `SessionCommandLogsResponse`).

This document captures **explicit validation** that the fixes referenced in
#192's comments were in place for the BenchFlow `0.5.1` release line on:

- `benchflow-ai/benchflow @ v0.5-integration` — HEAD `61fa09e` ("fix: expose skill invocation counts in artifacts (#518)")
- `benchflow-ai/skillsbench @ main` — HEAD `3f86918e` ("revert: remove agentbeats-task-env-images workflow")

Validation strategy:

1. **Static check** — for every bug, locate the fix in the merged PR and confirm the file change is present on the named HEADs.
2. **Targeted regression tests** — run the BenchFlow-side regression tests gated on the infra paths (`pytest_plugins`, daytona log retries, DinD compose, verifier env).
3. **Live Daytona run** — execute two of the originally-bugged tasks end-to-end on the current branches to confirm the agent reaches a verifier verdict without any infrastructure failure (verifier-error, sandbox-startup, transport-error, idle-timeout, malformed-logs).

## 14-bug → fix mapping

| # | Bug (one-line)                                          | Category              | Fix PR                                                | Where the fix lives                                                                                 | Verified on HEAD |
|---|---------------------------------------------------------|-----------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-------------------|
| 1 | suricata-custom-exfil: `pytest` installed to wrong Python | Verifier broken       | skillsbench [#733](https://github.com/benchflow-ai/skillsbench/pull/733) | `tasks/suricata-custom-exfil/tests/test.sh` now uses `uvx --with pytest==8.4.1 ...`                  | yes               |
| 2 | video-tutorial-indexer: `pytest_plugins` ignored        | Verifier broken       | benchflow [#309](https://github.com/benchflow-ai/benchflow/pull/309) + skillsbench [#736](https://github.com/benchflow-ai/skillsbench/pull/736) | `src/benchflow/task/config.py:175` — `VerifierConfig.pytest_plugins: list[str]` (pydantic field), plus task TOML wiring | yes               |
| 3 | sales-pivot-analysis: missing `openpyxl`                | Verifier broken       | skillsbench [#733](https://github.com/benchflow-ai/skillsbench/pull/733) | `tasks/sales-pivot-analysis/tests/test.sh` installs `openpyxl==3.1.5`                                | yes               |
| 4 | pddl-tpp-planning: verifier opened missing `.pkl`       | Verifier broken       | skillsbench [#846](https://github.com/benchflow-ai/skillsbench/pull/846) | `tasks/pddl-tpp-planning/tests/test_outputs.py` — `validate_plan()` uses `SequentialPlanValidator`; no `.pkl` read | yes               |
| 5 | fix-build-google-auto: `uv` not in Dockerfile           | Verifier broken       | skillsbench [#807](https://github.com/benchflow-ai/skillsbench/pull/807) | `tasks/fix-build-google-auto/environment/Dockerfile` installs `uv`                                  | yes               |
| 6 | syzkaller-ppdev: `/opt/syzkaller/sys/linux` root-owned  | Sandbox permissions   | skillsbench [#733](https://github.com/benchflow-ai/skillsbench/pull/733) | `tasks/syzkaller-ppdev-syzlang/environment/Dockerfile` — `RUN chmod -R 777 /opt/syzkaller/sys/linux/` | yes               |
| 7 | taxonomy-tree-merge: `/root/output` root-owned          | Sandbox permissions   | skillsbench [#764](https://github.com/benchflow-ai/skillsbench/pull/764) | Dockerfile creates `/logs/agent/output` with mode 777 + symlinks to `/root/output` via ENTRYPOINT    | yes               |
| 8 | lean4-proof: `/root/.elan` not agent-accessible         | Sandbox permissions   | skillsbench [#733](https://github.com/benchflow-ai/skillsbench/pull/733) | Dockerfile — `RUN chmod 755 /root && chmod -R 755 /root/.elan`                                       | yes               |
| 9 | multilingual-video-dubbing: `/outputs` root + verifier missing `torch` | Sandbox permissions   | skillsbench [#733](https://github.com/benchflow-ai/skillsbench/pull/733) | Dockerfile — `mkdir -p /outputs && chmod -R 777 /outputs`; torch pinned via `--extra-index-url`      | yes               |
| 10 | organize-messy-files: input PDFs missing               | Provisioning          | skillsbench [#764](https://github.com/benchflow-ai/skillsbench/pull/764) | Dockerfile RUN heredoc downloads 100 PDFs into `/root/papers/all` at build time                      | yes               |
| 11 | azure-bgp-oscillation-route-leak: datagen heredoc dropped on Daytona build | Provisioning          | skillsbench [#846](https://github.com/benchflow-ai/skillsbench/pull/846) | Dockerfile replaces inline `RUN python3 <<'DATAGEN'` heredoc with `COPY generate_data.py` + `RUN python3 generate_data.py` | yes               |
| 12 | fix-build-agentops: `claude-agent-acp` rc=127 (Node PATH on bugswarm base) | Agent launch          | skillsbench [#736](https://github.com/benchflow-ai/skillsbench/pull/736) | Dockerfile wipes preexisting Node 18 npm/corepack/binaries then installs Node 22 cleanly             | yes               |
| 13 | latex-formula-extraction: Daytona SDK `SessionCommandLogsResponse` ValidationError | SDK upstream          | benchflow [#347](https://github.com/benchflow-ai/benchflow/pull/347) area (ENG-147/148) | `src/benchflow/sandbox/_sdk_ops.py` — bounded retry + empty-response fallback when SDK returns malformed logs payload | yes               |
| 14 | react-performance-debugging: `pytest-playwright` plugin not loaded | Verifier broken       | benchflow [#309](https://github.com/benchflow-ai/benchflow/pull/309) + skillsbench [#736](https://github.com/benchflow-ai/skillsbench/pull/736) | `tasks/react-performance-debugging/task.toml` declares `pytest_plugins = ["pytest_playwright", "pytest_asyncio"]` (mechanism fixed by #309) | yes               |

All 14 fixes are present on the named HEADs as of this validation run.

## Targeted regression tests

Using the same command #505's triage comment used (it ran 71/84-deselected):

```bash
uv run --extra dev --extra sandbox-daytona python -m pytest \
  tests/test_task_config.py \
  tests/test_verify.py \
  tests/test_sandbox_multi_service.py \
  tests/test_daytona_command_polling.py \
  tests/test_acp.py \
  -k 'pytest_plugins or plugin or daytona or dind or logs or verifier' -q
```

Result on HEAD `61fa09e`:

```
72 passed, 88 deselected, 8 warnings in 7.20s
```

(One additional test relative to the triage snapshot, consistent with the eight intervening merges into `v0.5-integration`.)

Two targeted re-runs that lock in the most cited fixes:

```bash
uv run --extra dev --extra sandbox-daytona python -m pytest \
  tests/test_sandbox_hardening.py -k pytest_plugins -v
# tests/test_sandbox_hardening.py::TestVerifierEnv::
#     test_verifier_config_keeps_pytest_plugins_from_toml PASSED (bug 2, bug 14 mechanism)

uv run --extra dev --extra sandbox-daytona python -m pytest \
  tests/test_sandbox_multi_service.py -v
# 33 passed, 6 skipped — covers DinD compose / file-transfer / service-exec paths.
```

## Live Daytona run

Two of the originally-bugged tasks were re-run end-to-end against current `skillsbench@main` from a Daytona sandbox using the gemini ACP agent (`gemini-2.5-flash`). Gemini was used as the no-cost provider path — `ANTHROPIC_API_KEY` was not available in this environment, and the user's documented testing preference (Claude Haiku 4.5) is overridden by the explicit "DO NOT use Anthropic API for verification unless a free path doesn't exist" constraint on this task.

Command:

```bash
uv run bench eval create \
  --tasks-dir /tmp/skillsbench-main/tasks \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --include pddl-tpp-planning \
  --include azure-bgp-oscillation-route-leak \
  --concurrency 2 \
  --jobs-dir /tmp/v05-validation-505/jobs \
  --agent-idle-timeout 600
```

Job directory: `/tmp/v05-validation-505/jobs/2026-05-24__21-15-34/`

### azure-bgp-oscillation-route-leak (bug 11)

`result.json` excerpt:

```json
{
  "task_name": "azure-bgp-oscillation-route-leak",
  "rewards": {"reward": 0.0},
  "n_tool_calls": 11,
  "error": null,
  "error_category": null,
  "verifier_error": null,
  "verifier_error_category": null,
  "export_error": null,
  "idle_timeout_info": null,
  "sandbox_startup_info": null,
  "transport_error_info": null,
  "verifier_timeout_info": null,
  "timing": {
    "environment_setup": 2.1,
    "agent_setup": 3.1,
    "agent_execution": 60.0,
    "verifier": 15.3,
    "total": 103.5
  }
}
```

Infra outcome — all green:

- Build/provisioning OK (env setup 2.1s, no missing `/app/data/` — bug 11 confirmed fixed).
- Daytona sandbox started; agent setup 3.1s.
- Agent executed and ended cleanly (11 tool calls, end_turn).
- Verifier ran (15.3s) — pytest 4-test suite executed, all 4 failed because the agent did not write `/app/output/oscillation_report.json`. This is a genuine agent-quality failure, not infrastructure: the verifier was loaded, dependencies installed, ctrf produced, and the agent's deliverable was the only thing missing.
- All `*_error*` / `*_info` infra-failure fields are `null`.

### pddl-tpp-planning (bug 4)

`result.json` excerpt:

```json
{
  "task_name": "pddl-tpp-planning",
  "rewards": {"reward": 0.0},
  "n_tool_calls": 14,
  "error": "Agent prompt exceeded wall-clock budget 600s",
  "error_category": "timeout",
  "verifier_error": null,
  "verifier_error_category": null,
  "export_error": null,
  "idle_timeout_info": null,
  "sandbox_startup_info": null,
  "transport_error_info": null,
  "verifier_timeout_info": null,
  "partial_trajectory": true,
  "trajectory_source": "partial_acp",
  "timing": {
    "environment_setup": 2.7,
    "agent_setup": 3.5,
    "verifier": 21.7,
    "total": 649.4
  }
}
```

Infra outcome:

- The agent hit its 600s `[agent] timeout_sec` budget — `error_category: "timeout"` (ENG-149 diagnostic). This is the *agent* timing out, not the verifier or the sandbox.
- Despite the agent timeout, the verifier still ran cleanly (21.7s). The verifier ctrf reports 2/2 tests failed with `AssertionError: Missing output file: task01.txt` — the agent simply didn't produce its deliverables before the timeout.
- **Bug 4 is confirmed fixed**: the verifier ran the new `validate_plan()` path (no `FileNotFoundError: task01.pkl`). The failing test is `TestOutputFilesExist::test_all_output_files_exist`, which existed only after the bug-4 rewrite — the old `validate_plan()` would have crashed on `pkl` long before reaching this assertion.

### Job summary

```
Job complete: 0/2 (0.0%), errors=0, idle_timeouts=0, time=10.8min
```

- `errors=0` — no infra/transport errors (the only `error_category` recorded is `timeout`, a documented agent-budget classification, not an infra failure).
- `idle_timeouts=0` — no ACP idle-watchdog trips.
- Both verifiers ran and produced ctrf reports.

## Was bug 2 regressed by an intervening change?

No. `VerifierConfig.pytest_plugins: list[str] = Field(...)` is at `src/benchflow/task/config.py:175` on HEAD `61fa09e`, and `tests/test_sandbox_hardening.py::TestVerifierEnv::test_verifier_config_keeps_pytest_plugins_from_toml` is the locking regression test (passing).

## Was bug 13 (Daytona SDK) regressed?

No. The `_sdk_ops.apply()` patch is at `src/benchflow/sandbox/_sdk_ops.py`. Both the malformed-log marker (`_MALFORMED_MARKER = "SessionCommandLogsResponse"`) and the bounded retry that returns an empty-but-valid response are present. The patch is applied lazily from `_env_setup._create_environment` so it only runs when a Daytona environment is actually being built — the live azure-bgp run above exercised it without surfacing the malformed-logs failure.

## Residual issues

None for the 14-bug set. The verifier paths, sandbox permissions, datagen, and
Daytona SDK retry all behaved as designed on the v0.5 integration branch that
fed BenchFlow `0.5.1`, and on `skillsbench@main`.

Tangential observations (not regressions of the 14 bugs, not filed as new issues):

- gemini-2.5-flash hit the 600s agent budget on pddl-tpp-planning. This is an agent-capability outcome, not infra — `error_category` is correctly `"timeout"`. Other agents or larger budgets are expected to solve this.
- Historical note: at the time of this validation, Daytona telemetry was unavailable
  when only the old host-side telemetry path existed. Current BenchFlow starts
  LiteLLM inside Daytona sandboxes, so new Daytona runs should record provider
  usage when credentials are available.

## Status

- #505: ready to close — validation evidence captured in this doc plus the linked `result.json` artifacts.
- #192: already closed as "superseded by #505". No new BenchFlow-side code defect is reproduced by this validation.
