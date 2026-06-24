#!/usr/bin/env python3
"""Build the deterministic integration ``review-pack/`` for one PR.

Refactor of the earlier ``integration_review_pack.py`` into the SPEC review-pack
layout. Given a planner MATRIX (the cells the planner emitted) and an ARTIFACTS
directory of produced rollouts, this:

* classifies every planned matrix cell into a slot (X-SLOTS:
  ``missing | duplicate | stale | healthy | unhealthy``) -> ``matrix_observed``;
* grades every produced rollout through
  :func:`tests.integration.rubric_checks.grade_rollout`
  -> ``agent_judge_summary`` + ``metrics``;
* summarises skill loading per cell -> ``skill_catalog_summary``;
* compares docker/daytona within-PR parity AND, when a baseline root is given,
  the pinned-baseline reward-BAND parity -> ``parity_summary``;
* writes ``hardening_summary.md`` (network / verifier / root / path),
  ``red_flags.md`` (reward-hacking / leakage / infra), and the authoritative
  ``verdict.md`` (user-facing ``mergeable | mergeable with quarantines |
  not mergeable``).

The DETERMINISTIC verdict is authoritative and FAIL-CLOSED: any missing /
duplicate / stale / unhealthy required slot, or any parity failure, is a
blocker. The codex layer is SEPARATE (``codex_review.py``) and can only make the
verdict STRICTER, never upgrade it.

IMPORT DISCIPLINE: only stdlib at module top. The rubric grader is imported from
the sibling ``tests/integration`` tree; it lazy-imports the production enforcers
only inside the functions that grade a production rollout, so a flat-fixture run
needs no ``benchflow`` install.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
# The rubric grader is the single source of truth for gates + the verdict ladder.
sys.path.insert(0, str(REPO_ROOT / "tests" / "integration"))

import rubric_checks  # noqa: E402  (path bootstrap above is intentional)

# ------------------------------------------------------------------
# User-facing verdict names (CONTRACT Q2: deterministic gate). The internal
# rubric ladder maps onto these renamed, user-facing strings.
# ------------------------------------------------------------------
VERDICT_MERGEABLE = "mergeable"
VERDICT_QUARANTINES = "mergeable with quarantines"
VERDICT_NOT_MERGEABLE = "not mergeable"
_OK_VERDICTS = frozenset({VERDICT_MERGEABLE, VERDICT_QUARANTINES})


# ------------------------------------------------------------------
# Matrix normalization — accept BOTH the SPEC matrix-cell schema and the
# planner's existing ``cells`` schema, projecting onto one internal cell.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One normalized planned matrix cell (SPEC schema superset)."""

    id: str
    task: str | None
    agent: str | None
    model: str | None
    sandbox: str | None
    skill_mode: str | None
    network_mode: str
    audit_skills: bool
    expect_reward: str
    scenario: str | None
    include: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def is_parity(self) -> bool:
        return self.scenario == "sandbox_parity"


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def normalize_cell(raw: dict[str, Any]) -> Cell:
    """Project a planner/SPEC cell dict onto the internal :class:`Cell`.

    The SPEC cell carries a single ``task`` and ``network_mode`` /
    ``audit_skills`` / ``expect_reward``; the existing planner cell carries
    ``scenario`` + an ``include`` task list. Both are accepted so this grader
    runs over either producer.
    """
    include_raw = raw.get("include")
    include = (
        tuple(str(t) for t in include_raw) if isinstance(include_raw, list) else ()
    )
    task = raw.get("task")
    if task is None and include:
        task = include[0]
    cell_id = str(raw.get("id") or task or "?")
    return Cell(
        id=cell_id,
        task=str(task) if task is not None else None,
        agent=raw.get("agent"),
        model=raw.get("model"),
        sandbox=raw.get("sandbox"),
        skill_mode=raw.get("skill_mode"),
        network_mode=str(raw.get("network_mode") or "default-off"),
        audit_skills=_as_bool(raw.get("audit_skills")),
        expect_reward=str(raw.get("expect_reward") or "any"),
        scenario=raw.get("scenario"),
        include=include or ((task,) if task else ()),
        raw=raw,
    )


def _matrix_cells(plan: dict[str, Any]) -> list[Cell]:
    """The planned cells, from a SPEC ``matrix`` or a planner ``cells`` list."""
    raw_cells = plan.get("matrix")
    if not isinstance(raw_cells, list):
        raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list):
        return []
    return [normalize_cell(c) for c in raw_cells if isinstance(c, dict)]


# ------------------------------------------------------------------
# Rollout -> cell attribution + slot classification (X-SLOTS).
# ------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _norm(value: Any) -> Any:
    if value is None:
        return None
    return str(value).strip().lower().replace("_", "-")


def _rollout_dims(rollout_dir: Path) -> dict[str, Any]:
    cfg = _read_json(rollout_dir / "run_config.json") or {}
    result = _read_json(rollout_dir / "result.json") or {}
    task = cfg.get("task_id") or result.get("task_name")
    return {
        "scenario": cfg.get("scenario") or result.get("scenario"),
        "agent": cfg.get("harness") or cfg.get("agent") or result.get("agent"),
        "model": cfg.get("model") or result.get("model"),
        "sandbox": cfg.get("sandbox") or result.get("sandbox"),
        "skill_mode": cfg.get("skill_mode"),
        "task": task,
    }


def _dim_conflict(observed: Any, planned: Any) -> bool:
    """True when both sides are present and disagree (None on either is a wildcard)."""
    return (
        observed is not None
        and planned is not None
        and _norm(observed) != _norm(planned)
    )


def _cell_matches(cell: Cell, dims: dict[str, Any]) -> bool:
    if _dim_conflict(dims["scenario"], cell.scenario):
        return False
    if _norm(dims["agent"]) != _norm(cell.agent):
        return False
    if _dim_conflict(dims["model"], cell.model):
        return False
    if _dim_conflict(dims["sandbox"], cell.sandbox):
        return False
    if _dim_conflict(dims["skill_mode"], cell.skill_mode):
        return False
    task = dims["task"]
    if task is not None and cell.include:
        normed = {_norm(t) for t in cell.include}
        if task not in cell.include and _norm(task) not in normed:
            return False
    return True


def _resolved_source_sha(rollout_dir: Path) -> str | None:
    for name in ("run_config.json", "result.json"):
        data = _read_json(rollout_dir / name) or {}
        for key in ("head_sha", "source_sha", "source_ref", "git_sha", "commit"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        source = data.get("source")
        if isinstance(source, dict):
            for key in ("sha", "ref", "commit", "resolved_sha"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _sha_matches(resolved: str, head: str) -> bool:
    a, b = resolved.strip(), head.strip()
    if a == b:
        return True
    return len(a) >= 7 and len(b) >= 7 and (a.startswith(b) or b.startswith(a))


@dataclass
class Slot:
    """One planned cell and the rollouts that landed in it (X-SLOTS)."""

    cell: Cell
    rollouts: list[Path] = field(default_factory=list)
    status: str = "missing"  # missing|duplicate|stale|healthy|unhealthy
    grade: dict[str, Any] | None = None
    detail: str = ""

    @property
    def cell_id(self) -> str:
        return self.cell.id


def _started_at(rollout_dir: Path) -> str:
    """The rollout's start time as a sortable string ('' when absent).

    Production ``result.json`` carries a top-level ``started_at``; the flat
    fixtures nest it under ``timing``. Falls back to the finish time so
    retry-collapse still prefers the temporally-last attempt when ``started_at``
    is missing (rather than an arbitrary path/hash order).
    """
    data = _read_json(rollout_dir / "result.json") or {}
    for key in ("started_at", "finished_at"):
        value = data.get(key)
        if value:
            return str(value)
    timing = data.get("timing")
    if isinstance(timing, dict):
        return str(timing.get("started_at") or timing.get("ended_at") or "")
    return ""


def _attribute_rollout(
    rollout: Path, artifacts: Path, by_cell_id: dict[str, Slot], slots: list[Slot]
) -> Slot | None:
    """Map one produced rollout to its planned slot.

    Primary: the run-matrix writes each cell's rollouts under a directory named
    exactly by the cell id (``jobs/integration-final/<cell.id>/<ts>/<task>__<h>``),
    so attribute by that id. This keeps cells that differ only in an
    "expected-only" axis the rollout itself does not record — e.g. the
    ``-allowlist`` network-mode variant of an otherwise-identical cell — from
    colliding into a single slot (leaving the other slot spuriously "missing").
    Only path components BELOW ``artifacts`` are considered, so OS-level segments
    (``home``, ``runner``, ``work`` …) can never coincidentally match a cell id.

    Fallback: dims-based matching for legacy/flat rollouts whose path carries no
    recognizable cell-id directory.
    """
    try:
        parts = rollout.relative_to(artifacts).parts
    except ValueError:  # rollout not under artifacts (defensive)
        parts = rollout.parts
    for part in parts:
        slot = by_cell_id.get(part)
        if slot is not None:
            return slot
    dims = _rollout_dims(rollout)
    return next((s for s in slots if _cell_matches(s.cell, dims)), None)


def _dedupe_retries(rollouts: list[Path]) -> list[Path]:
    """Collapse retry attempts of one cell-job into a single rollout.

    A flaky agent makes the cell's ``scenarios.run_eval`` retry in place, leaving
    several ``<task>__<hash>`` result dirs under the SAME ``<cell-id>/<ts>`` job
    dir — attempts of one logical rollout, so keep only the latest by
    ``started_at``. Rollouts under DISTINCT job dirs are preserved, so a
    genuinely double-scheduled cell still surfaces as a ``duplicate``.
    """
    latest: dict[Path, Path] = {}
    for rollout in rollouts:
        job = rollout.parent
        current = latest.get(job)
        if current is None or _started_at(rollout) >= _started_at(current):
            latest[job] = rollout
    return sorted(latest.values())


def classify_slots(
    cells: list[Cell], artifacts: Path, expected_source_sha: str | None = None
) -> tuple[list[Slot], list[Path]]:
    """Group produced rollouts into planned slots and classify each.

    ``expected_source_sha`` is the TASK-SOURCE sha the plan pinned (e.g. the
    skillsbench commit), used only for the optional stale check; when ``None``
    no slot is marked stale. Returns ``(slots, unattributed)`` where
    ``unattributed`` are produced rollouts that matched no planned cell.
    """
    slots = [Slot(cell=cell) for cell in cells]
    by_cell_id: dict[str, Slot] = {}
    for slot in slots:
        by_cell_id.setdefault(slot.cell.id, slot)
    rollouts = sorted({p.parent for p in artifacts.rglob("result.json")})
    unattributed: list[Path] = []
    for rollout in rollouts:
        matched = _attribute_rollout(rollout, artifacts, by_cell_id, slots)
        if matched is None:
            unattributed.append(rollout)
            continue
        matched.rollouts.append(rollout)
    for slot in slots:
        slot.rollouts = _dedupe_retries(slot.rollouts)
        _classify_one(slot, expected_source_sha)
    return slots, unattributed


def _classify_one(slot: Slot, expected_source_sha: str | None) -> None:
    if not slot.rollouts:
        slot.status = "missing"
        slot.detail = "planned cell produced no rollout"
        return
    if len(slot.rollouts) > 1:
        slot.status = "duplicate"
        slot.detail = f"{len(slot.rollouts)} rollouts for one planned cell"
        return

    rollout = slot.rollouts[0]
    resolved = _resolved_source_sha(rollout)
    if (
        expected_source_sha
        and resolved
        and not _sha_matches(resolved, expected_source_sha)
    ):
        slot.status = "stale"
        slot.detail = f"task-source sha {resolved!r} != pinned {expected_source_sha!r}"
        return

    try:
        slot.grade = rubric_checks.grade_rollout(rollout)
    except FileNotFoundError as exc:
        slot.status = "unhealthy"
        slot.detail = f"ungradeable: {exc}"
        return
    if slot.grade["deterministic_reject"]:
        rejects = [
            g["id"]
            for g in slot.grade["gates"]
            if g["status"] == "fail" and g["enforcement"] == "deterministic"
        ]
        # R-OUTCOME fails when the rollout's status is not a valid SCORED outcome
        # (an errored / unscored-timeout run, often a hard task the agent didn't
        # finish in budget) — an experiment-fidelity issue on that one rollout, not
        # a regression introduced by the PR. Demote an R-OUTCOME-ONLY reject to a
        # QUARANTINE (visible, non-blocking). Any OTHER deterministic reject
        # (realness, tamper, artifact, telemetry, schema) is a real health failure
        # and still hard-blocks.
        non_outcome = [r for r in rejects if r != "R-OUTCOME"]
        if non_outcome:
            slot.status = "unhealthy"
            slot.detail = f"deterministic reject: {non_outcome}"
        else:
            slot.status = "healthy"
            # Demoted to healthy: clear the reject flag too, so the serialized
            # agent_judge_summary does not tell codex this healthy slot still has a
            # deterministic reject — a contradiction that can spuriously push the
            # codex reviewer to downgrade the verdict.
            slot.grade["deterministic_reject"] = False
            slot.grade["quarantines"] = [
                *slot.grade.get("quarantines", []),
                "R-OUTCOME: rollout produced no valid scored outcome "
                "(error / unscored timeout) — quarantined, not a PR regression",
            ]
            slot.detail = (
                f"healthy with {len(slot.grade['quarantines'])} quarantine(s) "
                "(incl. R-OUTCOME)"
            )
    else:
        slot.status = "healthy"
        if slot.grade["quarantines"]:
            slot.detail = f"healthy with {len(slot.grade['quarantines'])} quarantine(s)"
        else:
            slot.detail = "all gates green"


# ------------------------------------------------------------------
# Parity: within-PR docker/daytona pairs + pinned-baseline reward band.
# ------------------------------------------------------------------


@dataclass
class ParityResult:
    pair_id: str
    kind: str  # "within-pr" | "pinned-baseline"
    status: str
    detail: str


def within_pr_parity(slots: list[Slot]) -> list[ParityResult]:
    """P-SCHEMA over planned sandbox-parity docker+daytona pairs."""
    results: list[ParityResult] = []
    parity_slots = [s for s in slots if s.cell.is_parity]
    grouped: dict[tuple, dict[str, Slot]] = {}
    for slot in parity_slots:
        cell = slot.cell
        group = (cell.agent, cell.model, cell.include, cell.skill_mode)
        grouped.setdefault(group, {})[_norm(cell.sandbox)] = slot

    for group, sides in grouped.items():
        docker = sides.get("docker")
        daytona = sides.get("daytona")
        pair_id = f"sandbox-parity({group[0]})"
        if (
            docker is None
            or daytona is None
            or not docker.rollouts
            or not daytona.rollouts
        ):
            results.append(
                ParityResult(
                    pair_id,
                    "within-pr",
                    "fail",
                    "incomplete parity pair (missing/empty docker or daytona side)",
                )
            )
            continue
        gate_id, status, detail = rubric_checks.parity_schema_diff(
            docker.rollouts[0], daytona.rollouts[0]
        )
        results.append(
            ParityResult(pair_id, "within-pr", status, f"{gate_id}: {detail}")
        )
    return results


def baseline_parity(
    artifacts: Path, baseline_root: Path, cells: list[Cell]
) -> ParityResult | None:
    """Pinned-baseline reward-BAND parity over the SkillsBench tasks in scope.

    Lazy/subprocess wrapper via :func:`rubric_checks.parity_baseline_band`. Only
    runs over the tasks the matrix actually covers (so a non-SkillsBench PR with
    no overlapping baseline tasks is reported NA rather than fabricated).
    """
    tasks = sorted({c.task for c in cells if c.task})
    if not tasks:
        return ParityResult(
            "pinned-baseline", "pinned-baseline", "na", "no SkillsBench tasks in matrix"
        )
    band = rubric_checks.parity_baseline_band(artifacts, baseline_root, tasks)
    return ParityResult("pinned-baseline", "pinned-baseline", band.status, band.detail)


# ------------------------------------------------------------------
# Verdict (CONTRACT Q2 deterministic gate, FAIL-CLOSED).
# ------------------------------------------------------------------


@dataclass
class Verdict:
    verdict: str
    blockers: list[str] = field(default_factory=list)
    quarantines: list[str] = field(default_factory=list)


def compute_verdict(slots: list[Slot], parity: list[ParityResult]) -> Verdict:
    """Map the deterministic ladder onto the user-facing verdict (FAIL-CLOSED).

    ``not mergeable``: any deterministic reject (unhealthy slot), OR a required
    planned slot missing/duplicate/stale, OR a parity FAIL.
    ``mergeable with quarantines``: only quarantine/residual items, all
    deterministic gates green.
    ``mergeable``: full coverage, all gates green, zero quarantines.
    """
    blockers: list[str] = []
    quarantines: list[str] = []

    for slot in slots:
        if slot.status == "missing":
            blockers.append(f"missing slot: {slot.cell_id} ({slot.detail})")
        elif slot.status == "duplicate":
            blockers.append(f"duplicate slot: {slot.cell_id} ({slot.detail})")
        elif slot.status == "stale":
            blockers.append(f"stale slot: {slot.cell_id} ({slot.detail})")
        elif slot.status == "unhealthy":
            blockers.append(f"unhealthy slot: {slot.cell_id} ({slot.detail})")
        elif slot.status == "healthy" and slot.grade:
            quarantines.extend(
                f"{slot.cell_id}: {q}" for q in slot.grade["quarantines"]
            )

    for pr in parity:
        if pr.status == "fail" and pr.kind == "pinned-baseline":
            # The pinned-baseline reward-band gate currently feeds a NATIVE HF
            # leaderboard baseline to the Harbor-schema + git-pinned checker, which
            # structurally false-fails (missing Harbor fields / pin mismatch) — NOT
            # a real reward regression. Quarantine it (visible, non-blocking) until
            # check_skillsbench_harbor_parity gains a native-vs-native baseline mode
            # (tracked follow-up). Within-PR docker/daytona parity still hard-blocks.
            quarantines.append(
                f"parity {pr.kind} (advisory — gate needs native-baseline mode): "
                f"{pr.pair_id} — {pr.detail}"
            )
        elif pr.status == "fail":
            blockers.append(f"parity {pr.kind} fail: {pr.pair_id} — {pr.detail}")
        elif pr.status == "quarantine":
            quarantines.append(f"parity {pr.kind}: {pr.pair_id} — {pr.detail}")

    if blockers:
        return Verdict(VERDICT_NOT_MERGEABLE, blockers, quarantines)
    if quarantines:
        return Verdict(VERDICT_QUARANTINES, blockers, quarantines)
    return Verdict(VERDICT_MERGEABLE, blockers, quarantines)


# ------------------------------------------------------------------
# review-pack/ section builders (each returns a JSON-able object / md str).
# ------------------------------------------------------------------


def build_manifest(
    plan: dict[str, Any], artifacts: Path, baseline_root: Path | None
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "pr": plan.get("pr") or plan.get("pr_number"),
        "head_sha": plan.get("head_sha"),
        "base_ref": plan.get("base_ref"),
        "scope": plan.get("scope") or plan.get("breadth"),
        "buckets": plan.get("buckets") or [],
        "trust_boundary": plan.get("trust_boundary"),
        "network_lane": plan.get("network_lane"),
        "baseline": plan.get("baseline") or ("pinned" if baseline_root else None),
        "artifacts_root": str(artifacts),
        "baseline_root": str(baseline_root) if baseline_root else None,
        "source_refs": {
            "grader": "origin/main (trusted)",
            "code_under_test": plan.get("head_sha"),
        },
        "matrix_size": len(_matrix_cells(plan)),
    }


def matrix_expected(cells: list[Cell]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "task": c.task,
            "agent": c.agent,
            "model": c.model,
            "sandbox": c.sandbox,
            "skill_mode": c.skill_mode,
            "network_mode": c.network_mode,
            "audit_skills": c.audit_skills,
            "expect_reward": c.expect_reward,
            "scenario": c.scenario,
        }
        for c in cells
    ]


def matrix_observed(slots: list[Slot], unattributed: list[Path]) -> dict[str, Any]:
    return {
        "slots": [
            {
                "id": s.cell_id,
                "status": s.status,
                "detail": s.detail,
                "rollouts": [str(r) for r in s.rollouts],
            }
            for s in slots
        ],
        "counts": _status_counts(slots),
        "unattributed": [str(p) for p in unattributed],
    }


def _status_counts(slots: list[Slot]) -> dict[str, int]:
    counts = {
        "healthy": 0,
        "unhealthy": 0,
        "missing": 0,
        "duplicate": 0,
        "stale": 0,
    }
    for slot in slots:
        counts[slot.status] = counts.get(slot.status, 0) + 1
    return counts


def _rollout_metrics(rollout_dir: Path) -> dict[str, Any]:
    """Per-cell metrics row (task, reward, tokens, timing, n_tool_calls)."""
    try:
        ev = rubric_checks.load_evidence(rollout_dir)
    except FileNotFoundError:
        return {"error": "no readable result.json"}
    return {
        "reward": ev.reward,
        "status": ev.status,
        "n_tool_calls": ev.n_tool_calls,
        "total_tokens": ev.total_tokens,
        "started_at": ev.started_at,
        "ended_at": ev.ended_at,
        "duration_seconds": ev.duration_seconds,
        "sandbox": ev.sandbox,
    }


def metrics_summary(slots: list[Slot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in slots:
        if not slot.rollouts:
            continue
        row = {
            "cell": slot.cell_id,
            "task": slot.cell.task,
            "agent": slot.cell.agent,
            **_rollout_metrics(slot.rollouts[0]),
        }
        rows.append(row)
    return rows


def agent_judge_summary(slots: list[Slot]) -> list[dict[str, Any]]:
    """One row per graded rollout: gate statuses + attribution label."""
    rows: list[dict[str, Any]] = []
    for slot in slots:
        if not slot.grade:
            rows.append(
                {
                    "cell": slot.cell_id,
                    "status": slot.status,
                    "detail": slot.detail,
                    "gates": [],
                }
            )
            continue
        attribution = None
        try:
            ev = rubric_checks.load_evidence(slot.rollouts[0])
            attribution = rubric_checks.classify_capability(ev).__dict__
        except FileNotFoundError:
            attribution = None
        rows.append(
            {
                "cell": slot.cell_id,
                "rollout": str(slot.rollouts[0]),
                "status": slot.status,
                "schema": slot.grade["schema"],
                "deterministic_reject": slot.grade["deterministic_reject"],
                "quarantines": slot.grade["quarantines"],
                "gates": slot.grade["gates"],
                "attribution": attribution,
            }
        )
    return rows


def skill_catalog_summary(slots: list[Slot]) -> list[dict[str, Any]]:
    """with/no-skill task_skills_loading per cell (from the S-* gates)."""
    rows: list[dict[str, Any]] = []
    for slot in slots:
        if not slot.grade:
            continue
        s_with = _gate(slot.grade, "S-WITHSKILL")
        s_no = _gate(slot.grade, "S-NOSKILL")
        rows.append(
            {
                "cell": slot.cell_id,
                "skill_mode": slot.cell.skill_mode,
                "audit_skills": slot.cell.audit_skills,
                "with_skill": s_with,
                "no_skill": s_no,
            }
        )
    return rows


def _gate(grade: dict[str, Any], gate_id: str) -> dict[str, Any] | None:
    for g in grade["gates"]:
        if g["id"] == gate_id:
            return {"status": g["status"], "detail": g["detail"]}
    return None


def parity_summary(parity: list[ParityResult]) -> dict[str, Any]:
    return {
        "within_pr": [
            {"pair": p.pair_id, "status": p.status, "detail": p.detail}
            for p in parity
            if p.kind == "within-pr"
        ],
        "pinned_baseline": [
            {"pair": p.pair_id, "status": p.status, "detail": p.detail}
            for p in parity
            if p.kind == "pinned-baseline"
        ],
    }


def hardening_summary_md(
    slots: list[Slot], cells: list[Cell], verifier_or_sandbox_pr: bool
) -> str:
    """network / verifier / root / path hardening (CONTRACT F)."""
    lines = ["# Hardening summary", ""]
    lines.append("## Network policy (V-NETWORK, STATIC per-task config)")
    lines.append("")
    any_net = False
    for cell in cells:
        cfg = {
            "network_mode": _cell_network_config(cell),
            "allowed_hosts": cell.raw.get("allowed_hosts"),
        }
        gate_id, status, detail = rubric_checks.network_hardening(
            cfg, verifier_or_sandbox_pr=verifier_or_sandbox_pr
        )
        if status == "na":
            continue
        any_net = True
        lines.append(f"- {cell.id}: {gate_id}={status} — {detail}")
    if not any_net:
        lines.append(
            "- no explicit network policy declared (runtime default no-network)"
        )
    lines.append("")
    lines.append("## Verifier / root / path (per-rollout gates)")
    lines.append("")
    for slot in slots:
        if not slot.grade:
            continue
        tamper = _gate(slot.grade, "V-TAMPER")
        if tamper:
            lines.append(
                f"- {slot.cell_id}: V-TAMPER={tamper['status']} — {tamper['detail']}"
            )
    lines.append("")
    lines.append(
        "V-LIFECYCLE / V-ENVHARDEN: residual (not observable in artifacts) — "
        "codex/env-probe review."
    )
    return "\n".join(lines) + "\n"


def _cell_network_config(cell: Cell) -> str | None:
    """Map the cell's EXPECTED network_mode (Q3) to a NetworkMode literal.

    The cell carries ``network_mode`` as ``default-off`` | ``allowlist`` (Q3:
    derived from the task config, NOT passed to bench). Translate to the
    benchflow ``NetworkMode`` literals the static checker understands, or use an
    explicit per-cell ``network_mode`` override if the planner emitted one.
    """
    explicit = cell.raw.get("network_mode")
    mode = str(explicit) if explicit is not None else "default-off"
    norm = mode.strip().lower()
    if norm in {"default-off", "no-network", "off"}:
        return "no-network"
    if norm == "allowlist":
        return "allowlist"
    if norm == "public":
        return "public"
    return None


def red_flags_md(slots: list[Slot], unattributed: list[Path]) -> str:
    """reward-hacking / leakage / infra red flags (CONTRACT F)."""
    lines = ["# Red flags", ""]
    flagged = False
    for slot in slots:
        if not slot.grade:
            continue
        for g in slot.grade["gates"]:
            if g["id"] in {"V-TAMPER", "S-NOSKILL"} and g["status"] == "fail":
                flagged = True
                lines.append(
                    f"- REWARD-HACK/LEAK [{slot.cell_id}] {g['id']}: {g['detail']}"
                )
            if g["id"] == "C-ATTRIB" and g["status"] == "quarantine":
                flagged = True
                lines.append(f"- INFRA [{slot.cell_id}] C-ATTRIB: {g['detail']}")
        if slot.status == "unhealthy":
            flagged = True
            lines.append(f"- UNHEALTHY [{slot.cell_id}]: {slot.detail}")
    for path in unattributed:
        flagged = True
        lines.append(f"- UNATTRIBUTED rollout (matched no planned cell): `{path}`")
    if not flagged:
        lines.append("- none (no reward-hacking, leakage, or infra red flags)")
    return "\n".join(lines) + "\n"


def verdict_md(
    verdict: Verdict,
    slots: list[Slot],
    parity: list[ParityResult],
    plan: dict[str, Any],
) -> str:
    """The authoritative verdict.md (sections in SKILL order)."""
    lines = ["# Verdict", ""]
    lines.append(f"**{verdict.verdict}**")
    lines.append("")

    lines.append("## Blockers")
    lines.append("")
    if verdict.blockers:
        for b in verdict.blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append("| cell | task | agent | sandbox | skill_mode | status | detail |")
    lines.append("|---|---|---|---|---|---|---|")
    for slot in slots:
        c = slot.cell
        lines.append(
            f"| {slot.cell_id} | {c.task or '—'} | {c.agent or '—'} | "
            f"{c.sandbox or '—'} | {c.skill_mode or '—'} | {slot.status} | {slot.detail} |"
        )
    counts = _status_counts(slots)
    lines.append("")
    lines.append(
        "Slots: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
        + f" (planned={len(slots)})"
    )
    lines.append("")

    lines.append("## Evidence")
    lines.append("")
    for slot in slots:
        if not slot.grade:
            lines.append(f"- **{slot.cell_id}** ({slot.status}): {slot.detail}")
            continue
        gate_summary = ", ".join(
            f"{g['id']}={g['status']}" for g in slot.grade["gates"]
        )
        lines.append(f"- **{slot.cell_id}** ({slot.status})")
        lines.append(f"  - root: `{slot.rollouts[0]}`")
        lines.append(f"  - gates: {gate_summary}")
        lines.append(
            f"  - rerun: `python tests/integration/rubric_checks.py "
            f"{slot.rollouts[0]} --json`"
        )
    if parity:
        lines.append("")
        lines.append("Parity:")
        for p in parity:
            lines.append(f"- {p.kind} {p.pair_id}: {p.status} — {p.detail}")
    lines.append("")

    lines.append("## Residual risk")
    lines.append("")
    if verdict.quarantines:
        for q in verdict.quarantines:
            lines.append(f"- QUARANTINE: {q}")
    else:
        lines.append("- no deterministic quarantines raised")
    for risk in plan.get("residual_risk") or []:
        lines.append(f"- residual (from plan): {risk}")
    lines.append(
        "- V-LIFECYCLE / V-ENVHARDEN / V-REWARDHACK: codex/residual review "
        "(never faked deterministically)"
    )
    lines.append("")

    lines.append("## Required reruns")
    lines.append("")
    reruns = [
        slot.cell_id
        for slot in slots
        if slot.status in {"missing", "stale", "duplicate", "unhealthy"}
    ]
    if reruns:
        for cell_id in reruns:
            lines.append(f"- rerun cell: {cell_id}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Orchestration: build the full review-pack/ tree.
# ------------------------------------------------------------------


def build_review(
    plan: dict[str, Any], artifacts: Path, baseline_root: Path | None
) -> dict[str, Any]:
    """Classify, grade, parity, and compute the verdict for one artifacts root."""
    cells = _matrix_cells(plan)
    # Staleness is about the TASK SOURCE the rollout ran against, NOT the
    # benchflow PR head. A rollout's recorded source sha is the skillsbench
    # task-repo sha, which is unrelated to plan.head_sha (the benchflow commit) —
    # comparing them marked every healthy rollout "stale". Compare only against
    # an explicit task-source pin if the plan carries one; otherwise do not
    # fabricate staleness (the workflow guarantees freshness by running the
    # rollout at the PR head).
    expected_source_sha = plan.get("source_sha")
    slots, unattributed = classify_slots(cells, artifacts, expected_source_sha)
    parity = within_pr_parity(slots)
    if baseline_root is not None:
        base_parity = baseline_parity(artifacts, baseline_root, cells)
        if base_parity is not None:
            parity.append(base_parity)
    verdict = compute_verdict(slots, parity)
    return {
        "cells": cells,
        "slots": slots,
        "unattributed": unattributed,
        "parity": parity,
        "verdict": verdict,
    }


def _verifier_or_sandbox_pr(plan: dict[str, Any]) -> bool:
    buckets = {str(b).lower() for b in (plan.get("buckets") or [])}
    return bool(buckets & {"rewards", "sandbox", "judge-harness", "rollout"})


def write_pack(
    out: Path,
    plan: dict[str, Any],
    artifacts: Path,
    baseline_root: Path | None,
    review: dict[str, Any],
) -> None:
    """Write the full review-pack/ layout to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    cells: list[Cell] = review["cells"]
    slots: list[Slot] = review["slots"]
    unattributed: list[Path] = review["unattributed"]
    parity: list[ParityResult] = review["parity"]
    verdict: Verdict = review["verdict"]
    vsb = _verifier_or_sandbox_pr(plan)

    def _dump(name: str, obj: Any) -> None:
        (out / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")

    _dump("manifest.json", build_manifest(plan, artifacts, baseline_root))
    _dump("matrix_expected.json", matrix_expected(cells))
    _dump("matrix_observed.json", matrix_observed(slots, unattributed))
    _dump("metrics.json", metrics_summary(slots))
    _dump("agent_judge_summary.json", agent_judge_summary(slots))
    _dump("skill_catalog_summary.json", skill_catalog_summary(slots))
    _dump("parity_summary.json", parity_summary(parity))
    (out / "hardening_summary.md").write_text(
        hardening_summary_md(slots, cells, vsb), encoding="utf-8"
    )
    (out / "red_flags.md").write_text(
        red_flags_md(slots, unattributed), encoding="utf-8"
    )
    (out / "verdict.md").write_text(
        verdict_md(verdict, slots, parity, plan), encoding="utf-8"
    )
    # rollouts/ — record (link, not copy) the per-slot rollout roots so the
    # pack is self-describing without duplicating large trees.
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    rollouts_index = {slot.cell_id: [str(r) for r in slot.rollouts] for slot in slots}
    _dump("rollouts/index.json", rollouts_index)


# ------------------------------------------------------------------
# CLI.
# ------------------------------------------------------------------


def _load_matrix_arg(matrix: str) -> dict[str, Any] | None:
    """Accept ``--matrix`` as an inline JSON blob OR a path to a JSON file."""
    candidate = Path(matrix)
    if candidate.is_file():
        return _read_json(candidate)
    try:
        data = json.loads(matrix)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return {"matrix": data}
    return data if isinstance(data, dict) else None


def _emit_github_output(verdict: str) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        try:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(f"verdict={verdict}\n")
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic integration review-pack for one PR.",
    )
    parser.add_argument(
        "--matrix",
        required=True,
        help="Planner matrix: inline JSON or a path to scope-plan/matrix JSON.",
    )
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="Produced rollouts root."
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Optional pinned Harbor baseline root (enables band parity).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("review-pack"), help="review-pack output dir."
    )
    args = parser.parse_args(argv)

    plan = _load_matrix_arg(args.matrix)
    if plan is None:
        print(f"ERROR: cannot read matrix {args.matrix!r}", file=sys.stderr)
        # FAIL CLOSED: an unreadable plan is not mergeable.
        print(f"verdict={VERDICT_NOT_MERGEABLE}")
        _emit_github_output(VERDICT_NOT_MERGEABLE)
        return 1
    if not args.artifacts.exists():
        print(f"ERROR: artifacts root {args.artifacts} does not exist", file=sys.stderr)
        print(f"verdict={VERDICT_NOT_MERGEABLE}")
        _emit_github_output(VERDICT_NOT_MERGEABLE)
        return 1

    review = build_review(plan, args.artifacts, args.baseline_root)
    write_pack(args.out, plan, args.artifacts, args.baseline_root, review)

    verdict: Verdict = review["verdict"]
    print(f"verdict={verdict.verdict}")
    _emit_github_output(verdict.verdict)
    for b in verdict.blockers:
        print(f"BLOCKER: {b}", file=sys.stderr)
    for q in verdict.quarantines:
        print(f"QUARANTINE: {q}", file=sys.stderr)

    return 0 if verdict.verdict in _OK_VERDICTS else 1


if __name__ == "__main__":
    raise SystemExit(main())
