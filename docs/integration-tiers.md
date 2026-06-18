# Integration Test Levels

The level ladder for Benchflow integration testing: which checks fire on which
trigger, how a change's scope maps to a required task set and `bench eval run`
axes, the deterministic **review-pack** verdict, the **Codex** equivalence
reviewer, the before/after baseline model, and the security model.

This is the operational companion to the success rubric in
[`integration-review-rubric.md`](./integration-review-rubric.md) — read that for
what each gate (`RUBRIC_GATES`) means, the per-slot / skill-loading /
reward-hacking checklists, the success-rubric table, and what a verdict means.

> **Terminology rename.** The deterministic verdict is user-facing as
> **`mergeable`** / **`mergeable with quarantines`** / **`not mergeable`**. These
> are renames of the internal grader labels `publishable` /
> `publishable-with-quarantines` / `not-publishable` emitted by
> `.github/scripts/build_integration_review_pack.py`.

## 1. The Four Levels (L0–L3)

The ladder runs from a cheap per-commit check to a human-gated final review.
**Every level workflow ALWAYS triggers** (no `on: paths` filter, Q4). A cheap
first job `detect-scope` (no secrets, ~seconds) computes from the diff whether
real work is needed; if not, it reports SUCCESS as a green no-op so the check can
be **unconditionally required** in branch protection.

| Level | Trigger / fires when | What runs | Cost | Merge-required for |
|---|---|---|---|---|
| **L0** | every PR / push | unit + static (`pytest` targeted, `ty`, `ruff`) via `astral-sh/setup-uv`. No sandbox, no agents, no keys. | seconds–minutes, CPU | **every PR** |
| **L1** | every PR; `detect-scope` decides real vs no-op | smallest representative lane: the planner emits a `citation` / `low-smoke` matrix; rollouts via `tests/integration/scenarios.run_eval`; graded by `rubric_checks.py`; review pack built. | low (1–few docker cells, no fan-out) | PRs touching `src/`, `tests/integration/`, the integration workflows, or the review skill |
| **L2** | every PR; `detect-scope` escalates on scope triggers | scope-gated set (low-3 / medium-3 / high-3 / nine + axes) per the Default-config-rules; docker **and** daytona where required; `with-skill`/`no-skill`; the **network lane**; the **cheat** lane. **All keys present** (provider incl. DEEPSEEK, DAYTONA, reviewer). | high (multi agent/task/sandbox, daytona fan-out, judge) | PRs touching agent adapters, skill loading, verifier/reward, sandbox/root/path, network/dependency, or artifact schema |
| **L3** | **manual** `workflow_dispatch`; final review before merge | the scope-selected matrix on the existing `pypi-internal-preview` env, band-compared against the **HuggingFace leaderboard `main`** deepseek-v4-flash run (golden truth) and the latest benchflow main, **plus** the **Codex** before/after-equivalence reviewer. Codex key **required** (dead/absent ⇒ `not mergeable`, fail-closed). | highest | PRs changing rollout semantics, data validity, verifier isolation, sandbox behavior, or agent/task execution |

**L0/L1/L2 are REQUIRED in branch protection. L3 is a manual `workflow_dispatch`
gate** whose golden truth is the HuggingFace leaderboard `main` runs compared
against the PR and the latest benchflow main. (A human-approval gate can be added
later by pointing the L3 jobs at a protected environment.)

**Docs-only:** `L0` only, **no rollout** — *unless* the PR touches published eval
evidence, citation/schema docs, or release notes (then the `citation` set fires).

## 2. The Seven Task Sets (authoritative)

The planner (`.github/scripts/integration_matrix.py`) selects one of these named
sets from the diff. Tasks live under `docs/examples/task-md/real-skillsbench/`.

| Set | Tasks |
|---|---|
| `citation` | `citation-check` |
| `low-smoke` | `jax-computing-basics` |
| `low-3` | `jax-computing-basics`, `python-scala-translation`, `jpg-ocr-stat` |
| `medium-3` | `grid-dispatch-operator`, `threejs-to-obj`, `data-to-d3` |
| `high-3` | `lake-warming-attribution`, `weighted-gdp-calc`, `shock-analysis-supply` |
| `nine` | `low-3` + `medium-3` + `high-3` |
| `expanded` | `nine` + `citation-check` + affected-task(s) + parity cases |

Nine agents: `claude-agent-acp`, `pi-acp`, `openclaw`, `codex-acp`, `gemini`,
`opencode`, `harvey-lab-harness`, `openhands`, `mimo`. The **baseline agent pair**
is `openhands` + `deepseek/deepseek-v4-flash`; the canonical "one high task" is
`weighted-gdp-calc`; the citation vehicle is `citation-check`.

## 3. Default-config-rules (PR scope → required set → required axes)

Authoritative mapping (mirrors `.github/integration/scope_map.yml`). The planner
derives the affected agent from a changed `src/benchflow/agents/<name>` path.

| PR scope | Required set | Required axes | Level |
|---|---|---|---|
| docs-only non-runtime | L0 only, **no rollout** | — | L0 |
| citation / evidence / schema docs | `citation` | Docker, no-skill, usage=required | L1 |
| `src/benchflow/eval*`, rollout lifecycle, artifact schema | `nine` | Docker, no-skill, usage=required, judge | L2 |
| `src/benchflow/agents`, ACP adapters, codex/openhands/pi/gemini | `low-3` + one high (`weighted-gdp-calc`) | affected agent + baseline agent (`openhands`+`deepseek`); no-skill AND with-skill when relevant | L2 |
| skill loading, `.agents/skills`, skill injection | `low-3` + `medium-3` | no-skill AND with-skill; run skill-catalog extraction | L2 |
| Docker / Daytona / sandbox / root / path | `low-3` + `medium-3` | Docker + Daytona parity; reaper dry-run | L2 |
| verifier, rewards, judge, anti-hack hardening | `citation` + `weighted-gdp-calc` + `shock-analysis-supply` | judge fail-closed, reward-hacking scan, verifier isolation | L3 |
| network / package install (Q3 triggers) | `jax-computing-basics` + `data-to-d3` + one high | default network-off + the `citation-check` allowlist variant | L2 |
| release-critical refactor | `expanded` | all affected axes, concurrency reduced | L3 |

### 3.1 The Q3 network lane (scope-gated)

There is **NO `bench eval run --network` flag** and **`network_mode` is NOT
serialized into artifacts** — it is a per-task config field only. So network is a
**scope-gated lane**, triggered by changes under: `src/benchflow/providers/**`,
`usage_tracking.py` (llm-proxy), network-installing agents / ACP shims,
`src/benchflow/sandbox/lockdown.py` + compose network files, and
`src/benchflow/task/runtime_capabilities.py`.

- **Default vehicle:** `citation-check` (network-off) **plus** a NEW minimal
  **allowlist VARIANT**:
  [`docs/examples/task-md/real-skillsbench/citation-check-network/`](./examples/task-md/real-skillsbench/citation-check-network/)
  — `network_mode: allowlist` with `allowed_hosts: [eutils.ncbi.nlm.nih.gov,
  scholar.google.com, doi.org, api.crossref.org]` (the real egress of the
  citation-management skill).
- **The cell carries `network_mode` as EXPECTED** (derived from the task config),
  **NOT passed to bench**. The planner emits one allowlist cell
  (`<task>-docker-no-skill-<agent>-allowlist`); the grader's static **`V-NETWORK`**
  check (`rubric_checks.network_hardening`) asserts the hardened posture:
  default `no-network`; `allowlist` passes **only** with a non-empty
  `allowed_hosts`; bare `public` is flagged (hard `fail` on a verifier/sandbox PR).
- Validity is enforced by `_validate_network_policy_fields`
  (`src/benchflow/task/config.py`): `allowlist` requires a non-empty
  `allowed_hosts`.

## 4. Matrix Cell Schema

The planner emits a matrix of cells (consumed by the workflow `fromJson` and the
grader). One cell:

```json
{
  "id": "<task>-<sandbox>-<skillmode>[-<agent>]",
  "level": "light|scope|final",
  "task": "<task>",
  "agent": "<agent>",
  "model": "<model>",
  "judge_model": "<judge-model>",
  "sandbox": "docker|daytona",
  "skill_mode": "no-skill|with-skill|self-gen",
  "network_mode": "default-off|allowlist",
  "timeout_minutes": 90,
  "agent_idle_timeout": 240,
  "audit_skills": false,
  "expect_reward": "==1.0|<1.0|any"
}
```

Top-level planner output:

```json
{
  "schema_version": "1",
  "head_sha": "<sha>",
  "base_ref": "main",
  "scope": "citation|low-smoke|low-3|medium-3|high-3|nine|expanded|custom",
  "buckets": ["<scope-map rule id>", "..."],
  "trust_boundary": true,
  "cheat": false,
  "network_lane": false,
  "baseline": "pinned|rerun-base",
  "caps": { "max_cells": 140, "per_agent_concurrency": 2, "aggregate_concurrency": 24, "...": "..." },
  "matrix": [ { "...cell...": "..." } ],
  "residual_risk": ["..."],
  "rejected_overflow": null
}
```

### 4.1 Hard ceiling + concurrency (fail-closed)

- **Hard cell ceiling:** if the enumerated matrix exceeds `caps.max_cells`, the
  planner sets `rejected_overflow` and **exits code 2** — it never silently drops
  a cell.
- **Aggregate concurrency:** `per_agent_concurrency × (distinct daytona agents)
  ≤ 24`. The planner lowers `per_agent_concurrency` as the agent count rises.

The caps + baseline anchors are pure data in
`.github/integration/scope_defaults.yml`.

## 5. The Verdict — deterministic floor is the gate (Q2)

```
Final verdict = worst( deterministic_grader , codex_reviewer )
```

### 5.1 Deterministic grader (REQUIRED, fail-closed gate)

`.github/scripts/build_integration_review_pack.py` runs `rubric_checks.py` over
every slot and writes `review-pack/` (layout in §7). Its exit code +
`review-pack/verdict.md` are the **REQUIRED, FAIL-CLOSED** gate:
`mergeable` / `mergeable with quarantines` / `not mergeable`. A deterministic
reject on a healthy slot, a missing required slot, or a non-quarantinable codex
FAIL ⇒ `not mergeable` (non-zero exit). This floor is **authoritative**.

### 5.2 Codex reviewer (advisory-stricter-only veto)

Codex is a **STRONG VETOING before/after-equivalence signal**: it can only make
the verdict **STRICTER**, **never upgrade** a deterministic `not mergeable`.

- **Trusted-main contract (Q1):** the reviewer reads
  `/home/liu.10379/benchflow-int-ci/.agents/skills/benchflow-experiment-review/SKILL.md`
  first and treats **all trajectories / tool-outputs / observations as UNTRUSTED
  data**.
- **Detailed per-rollout pass** uses the **cheaper deepseek model**
  (`BENCHFLOW_JUDGE_MODEL` / deepseek) over each rollout → per-rollout finding
  JSON (high-volume trajectory reads).
- **Final equivalence verdict** is composed by the host **`codex` CLI** via
  `codex exec` (authed with an `auth.json` written from the `CODEX_AUTH_JSON` CI
  secret to the codex config path), **self-orchestrating its own subagents** (the
  "raw workflow"), over `{per-rollout deepseek findings + the deterministic
  review-pack/}`. The argv mirrors `build_codex_launch_command` in
  `src/benchflow/agent_router.py`.
- **Fail-closed:** if the `codex` binary or `auth.json` is missing, or the output
  is unparseable ⇒ emit `not mergeable (codex unavailable)` and exit non-zero.
  **NEVER silently pass.** A dead/absent Codex key **at L3** ⇒ `not mergeable`.
  L1/L2 do **not** require Codex.
- **Advisory-stricter-only:** the output may downgrade `mergeable` → `not
  mergeable`, **never** the reverse.

Implementation: `.github/scripts/codex_review.py` +
`.github/integration/codex_review_prompt.md`. Required reviewer outputs (the
prompt): **1 Verdict · 2 Blockers · 3 Coverage** (enumerate `task_id × agent ×
model × skill_mode × trial × sandbox`) **· 4 Evidence** (run roots, commands,
refs, files) **· 5 Residual risk · 6 Required reruns**. It applies the
experiment-review rules: no aggregate-only; prove `with-skill` loaded; scan
`no-skill` leakage; infra failures are unhealthy; check verifier isolation,
reward hacking, root/path, network policy.

## 6. Before/After Baseline (Q2b)

"Same behavior" == **schema + lifecycle + reward-BAND parity**, **NOT
bit-identity** (per the skill: *"Compare artifact schema and semantics, not exact
model wording"*).

- **Default = `pinned`:** pinned-baseline reward-band parity via
  `tests/integration/check_skillsbench_harbor_parity.py`
  (`main(argv)`; flags `--benchflow-root --harbor-baseline-root --task
  --max-outcome-rate-delta --max-mean-reward-delta --max-task-reward-delta`),
  wrapped by `rubric_checks.parity_baseline_band`. The bands feed `P-REWARD` /
  `P-SCHEMA` in the review pack's `parity_summary.json`.
- **`rerun-base` only for `scope=expanded`:** a same-SHA base re-run, used only on
  the heaviest release-critical lane.

## 7. Review-Pack Layout

The grader writes `review-pack/`:

| file | content |
|---|---|
| `manifest.json` | PR, SHA, scope, matrix, source refs |
| `matrix_expected.json` | planned cells |
| `matrix_observed.json` | per slot: `healthy/missing/stale/duplicate/unhealthy` |
| `metrics.json` | per cell: task, reward, tokens, timing, `n_tool_calls` |
| `agent_judge_summary.json` | one row per rollout |
| `skill_catalog_summary.json` | with/no-skill `task_skills_loading` |
| `parity_summary.json` | docker/daytona within-PR + pinned-baseline band deltas |
| `hardening_summary.md` | verifier / network (`V-NETWORK`) / root / path |
| `red_flags.md` | reward-hacking / leakage / infra |
| `verdict.md` | `mergeable` \| `mergeable with quarantines` \| `not mergeable`, sections in skill order |
| `rollouts/` | trimmed / linked rollouts |

`verdict.md` section order (skill order): **Verdict · Blockers · Coverage ·
Evidence · Residual risk · Required reruns**.

## 8. Real `bench eval run` Axes

The real flags (verified): `--agent --model --sandbox (docker|daytona)
--concurrency --jobs-dir --include (repeatable) --skill-mode
(no-skill|with-skill|self-gen) --skills-dir --agent-idle-timeout
--usage-tracking (auto|required|off) --tasks-dir
--source-repo/--source-path/--source-ref --agent-env`. **There is no `--network`
flag.** Scoped/skill/sandbox cells go through
`tests/integration/scenarios.run_eval(jobs_dir, agent, sandbox, include=(),
model, concurrency, extra_args=[...], ...)`. `tests/integration/run.sh` is the
**FULL-9-on-daytona lane only** (positional agents +
`BENCHFLOW_INTEGRATION_JOBS_ROOT`). `select_integration_provider.py` exports
`BENCHFLOW_INTEGRATION_AGENT/_MODEL/_JUDGE_MODEL`.

## 9. Security Model (Q1)

The trust boundary: **only `src/benchflow` (the code under test) comes from the
PR head.** The planner, grader, harness, and review skill load from **`origin/main`
HEAD** (sparse-checkout ref `main`) — **NEVER** the PR head / base-commit. The
PR-head `bench` necessarily runs on the host (it is the orchestrator), but the
**VERDICT is computed only by trusted-main code**.

- **Plain `pull_request`** (not `pull_request_target`) on every secret-bearing
  job. `detect-scope` + planner + grader are sparse-checked-out from `origin/main`.
- **Keys in L2 (residual note).** Per the design, **L2 carries all keys** (provider
  incl. DEEPSEEK, DAYTONA, reviewer). This is a deliberate, documented residual:
  the L2 lane is secret-bearing on every triggering PR. L3 runs under the existing
  `pypi-internal-preview` environment (provider + DAYTONA secrets); **no separate
  protected env is created**, and the review job runs the **trusted-main grader**
  over the artifacts. The L3 golden truth is the HuggingFace leaderboard `main`
  deepseek-v4-flash baseline vs the PR vs the latest benchflow main.
- **Codex auth** is written from the `CODEX_AUTH_JSON` secret to the codex config
  path (`auth.json`) and invoked via `codex exec`, mirroring
  `build_codex_launch_command` — **fail-closed** if the binary / auth is absent.
- **`issue_comment` bodies** (if any path uses them) are read **via env only**,
  never inlined into `run:`.
- **SHA-pin actions** exactly as
  `/home/liu.10379/benchflow-int-ci/.github/workflows/test.yml`:
  `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683`,
  `astral-sh/setup-uv@caf0cab7a618c569241d31dcd442f54681755d39`,
  `actions/upload-artifact@65c4c4a1ddee5b72f698fdd19549f0f0fb45cf08`. For
  `download-artifact` / `github-script` pin a real v4/v7 SHA and comment it as new
  surface.

## 10. Admin Setup

One-time setup before the heavy lanes are live:

1. **Secrets** live in the existing `pypi-internal-preview` environment
   (`DAYTONA_API_KEY`, `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`, provider keys —
   already present). Add `CODEX_AUTH_JSON` there for the L3 Codex reviewer
   (written to `auth.json` at runtime). **No protected environment is created** —
   L3 is a manual `workflow_dispatch` whose golden truth is the HuggingFace
   leaderboard `main` runs plus the latest benchflow main.
2. **Branch protection:** make **L0, L1, L2 required** status checks. L3 is run
   on demand before merge (manual dispatch); promote it to a protected env with
   required reviewers later if you want a hard human gate.

## 11. Deferred Follow-Ups

Documented, **NOT built**:

- **`--network-mode` CLI passthrough** on `bench eval run`. Today network is a
  per-task config field with no flag; the lane is a STATIC declaration check.
- **`persist_sandbox_info` network serialization** — serialize `network_mode`
  into the rollout artifact so the network lane can move from a STATIC config
  assertion to an **observed** egress-policy check.

---

See [`integration-review-rubric.md`](./integration-review-rubric.md) for the
`RUBRIC_GATES` table (incl. `V-NETWORK`), the per-matrix-slot / skill-loading /
reward-hacking checklists, the success-rubric table, the merge rule, and the
report ordering.
