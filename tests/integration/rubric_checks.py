#!/usr/bin/env python3
"""Deterministic rubric grader for one BenchFlow integration rollout.

This module is the SINGLE SOURCE OF TRUTH for the rubric gate table
(:data:`RUBRIC_GATES`, CONTRACT section A) and the deterministic gate
predicates that decide whether a rollout is publishable. It is consumed by
:mod:`.github.scripts.build_integration_review_pack` and mirrored verbatim by
the docs.

Two artifact schemas exist and must both normalize into one evidence shape:

* **flat skill-eval fixture** — ``result.json`` with top-level ``reward`` and
  ``token_usage`` plus a sibling ``run_config.json``. Used by the standalone
  skill evals; carries no ``benchflow`` install.
* **production rollout dir** — ``result.json`` with ``rewards.reward`` /
  ``n_tool_calls`` / ``agent_result.total_tokens``, ``rewards.jsonl``,
  ``trajectory/acp_trajectory.jsonl``, ``trainer/atif.json`` /
  ``trainer/adp.jsonl``. Graded by the reused production enforcers
  (``agent_judge`` / ``check_results`` / ``scenarios``).

IMPORT DISCIPLINE: only stdlib is imported at module top so the flat-fixture
unit tests run under plain ``python3`` with NO ``benchflow`` install. The
production enforcers (``agent_judge``, ``check_results``, ``scenarios``) are
LAZY-imported inside the functions that grade a production rollout dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Bootstrap so ``python tests/integration/rubric_checks.py`` finds the sibling
# integration tools (agent_judge / check_results / scenarios) and the package
# ``src`` tree without an install, matching the other integration scripts.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2] / "src"))

Enforcement = Literal["deterministic", "quarantine", "codex", "residual"]
GateStatus = Literal["pass", "fail", "quarantine", "na"]


# ------------------------------------------------------------------
# A) RUBRIC GATE IDS — the single source of truth (CONTRACT section A).
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RubricGate:
    """One rubric gate: id, title, enforcement tier, and skill reference."""

    id: str
    title: str
    enforcement: Enforcement
    skill_ref: str


RUBRIC_GATES: list[RubricGate] = [
    RubricGate(
        "R-REAL",
        "realness (n_tool_calls>0 & total_tokens>0 & reward!=null)",
        "deterministic",
        "agent_judge.realness_issues",
    ),
    RubricGate(
        "R-OUTCOME",
        "outcome in {pass,fail,normal_timeout} with a valid score",
        "deterministic",
        "rubric_checks.check_outcome",
    ),
    RubricGate(
        "R-ARTIFACT",
        "trajectory present+parseable; ATIF/ADP well-formed; no secret leak",
        "deterministic",
        "scenarios.atif_issues/adp_issues/secret_leak_issues",
    ),
    RubricGate(
        "R-TELEMETRY",
        "token usage + timing present for newly generated data",
        "deterministic",
        "rubric_checks.check_telemetry",
    ),
    RubricGate(
        "S-WITHSKILL",
        "with-skill => task_skills_loading==1",
        "quarantine",
        "extract_harness_skills",
    ),
    RubricGate(
        "S-NOSKILL",
        "no-skill => task_skills_loading==0 AND no leakage markers",
        "deterministic",
        "rubric_checks.check_noskill_leak",
    ),
    RubricGate(
        "V-TAMPER",
        "no agent write/delete/chmod of score-defining files",
        "deterministic",
        "agent_judge._scan_verifier_tamper",
    ),
    RubricGate(
        "V-EMPTYPATCH",
        "empty/no-op patch FAILS (<1.0) while golden patch passes (1.0)",
        "deterministic",
        "null-agent lane + oracle determinism",
    ),
    RubricGate(
        "V-LIFECYCLE",
        "verifier starts only after agent exit/timeout",
        "residual",
        "no lifecycle timestamps in artifacts",
    ),
    RubricGate(
        "V-ENVHARDEN",
        "git scrubbed past base; no answer/hidden-test/scorer files visible",
        "residual",
        "sandbox-fs property; best-effort env-probe else residual",
    ),
    RubricGate(
        "V-REWARDHACK",
        "reward consistent with visible task state (no grader-gaming)",
        "codex",
        "agent_judge LLM judge",
    ),
    RubricGate(
        "P-REWARD",
        "same oracle task scores identically docker vs daytona",
        "deterministic",
        "test_sandbox_parity, oracle, reward-only",
    ),
    RubricGate(
        "P-SCHEMA",
        "token/timing/schema parity on a REAL-MODEL paired cell",
        "deterministic",
        "rubric_checks.parity_schema_diff",
    ),
    RubricGate(
        "C-ATTRIB",
        "fail/timeout = model-capability (not experiment-fidelity)",
        "deterministic",
        "check_results.INFRA_ERROR_CATEGORIES + verifier_error + required-env",
    ),
    RubricGate(
        "P-PROV",
        "source repo/path/ref/hash provenance intact",
        "deterministic",
        "check_results._source_*_truth_issues",
    ),
    RubricGate(
        "P-PATHS",
        "config root / sandbox cwd / trajectory paths / result paths agree",
        "deterministic",
        "deterministic-where-fields-exist else residual",
    ),
    RubricGate(
        "X-SLOTS",
        "every planned cell present, unique, fresh (head sha)",
        "deterministic",
        "review_pack slot classification",
    ),
    RubricGate(
        "V-NETWORK",
        "default no-network; allowlist needs non-empty allowed_hosts; "
        "public flagged (fail on verifier/sandbox PR)",
        "deterministic",
        "rubric_checks.network_hardening (STATIC per-task config)",
    ),
]

GATES_BY_ID: dict[str, RubricGate] = {gate.id: gate for gate in RUBRIC_GATES}


# ------------------------------------------------------------------
# Normalized evidence shape (the schema adapter target).
# ------------------------------------------------------------------


# No-skill leakage markers (CONTRACT S-NOSKILL): any of these appearing in a
# no-skill trajectory means a skill catalog or invocation surface leaked in.
_NOSKILL_LEAK_MARKERS = (
    "skill.md",
    ".codex/skills",
    ".agents/skills",
    ".claude/skills",
    "invoke_skill",
    "activate_skill",
    "toolsearch",
)

# Skill modes that mean "no skills were provisioned" (mirrors check_results).
_NO_SKILL_MODES = frozenset(
    {"no-skill", "no-skills", "none", "without-skill", "without-skills"}
)
_WITH_SKILL_MODES = frozenset(
    {"with-skill", "with-skills", "with_task_skills", "task-skills", "skills"}
)

# Outcome statuses that are a legitimate, scored run (CONTRACT R-OUTCOME).
_VALID_OUTCOMES = frozenset({"pass", "fail", "normal_timeout"})
# A bare ``timeout`` is normalized to ``normal_timeout`` only when the run still
# produced a score; an unscored timeout is an infra/experiment-fidelity failure.


@dataclass
class Evidence:
    """One rollout normalized across the flat-fixture and production schemas."""

    rollout_dir: Path
    schema: Literal["flat", "production"]
    reward: float | None
    status: str | None
    n_tool_calls: int
    total_tokens: int | None
    started_at: str | None
    ended_at: str | None
    skill_mode: str | None
    task_skills: list[str]
    sandbox: str | None
    source_ref: str | None
    verifier_started_after_agent: bool | None
    # Carried through for the production enforcers / attribution; not part of the
    # CONTRACT-D field list but needed by C-ATTRIB and the production gates.
    error: str | None = None
    verifier_error: str | None = None
    required_env: list[str] = field(default_factory=list)
    duration_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def normalized_skill_mode(self) -> str | None:
        if self.skill_mode is None:
            return None
        norm = self.skill_mode.strip().lower().replace("_", "-")
        if norm in _NO_SKILL_MODES:
            return "no-skill"
        if norm in _WITH_SKILL_MODES:
            return "with-skill"
        return norm


# ------------------------------------------------------------------
# Schema sniff + load_evidence (the adapter).
# ------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_flat_fixture(result: dict[str, Any], rollout_dir: Path) -> bool:
    """Sniff flat skill-eval fixture vs production rollout schema.

    The flat skill-eval fixture puts ``reward`` and ``token_usage`` at the top
    level of ``result.json`` and ships a sibling ``run_config.json``. The
    production schema instead nests the score under a ``rewards`` MAPPING
    (``rewards.reward``) and records tokens under ``agent_result``.

    Primary signal (CONTRACT): top-level ``reward`` AND ``token_usage``. But an
    infra-failure fixture may legitimately drop ``token_usage`` (it was never
    recorded), so also treat a result as flat when it has a top-level ``reward``
    and NO production ``rewards`` mapping — disambiguated by a sibling
    ``run_config.json`` (production rollouts never pair a top-level ``reward``
    with a sibling ``run_config.json``).
    """
    has_top_reward = "reward" in result
    has_token_usage = "token_usage" in result
    if has_top_reward and has_token_usage:
        return True
    rewards = result.get("rewards")
    has_prod_rewards = isinstance(rewards, dict) and "reward" in rewards
    if has_top_reward and not has_prod_rewards:
        # Flat shape (top-level reward, no production rewards mapping). Confirm
        # via the flat fixture's sibling run_config.json or its flat ``status``.
        if (rollout_dir / "run_config.json").is_file():
            return True
        if "status" in result and "timing" in result:
            return True
    return False


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_str(*values: Any) -> str | None:
    """First non-empty string among ``values``, else None."""
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _load_flat_evidence(rollout_dir: Path, result: dict[str, Any]) -> Evidence:
    """Adapt a flat skill-eval fixture (+ sibling run_config.json)."""
    run_config = _read_json(rollout_dir / "run_config.json") or {}

    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    token_usage = (
        result.get("token_usage") if isinstance(result.get("token_usage"), dict) else {}
    )
    tool_usage = (
        result.get("tool_usage") if isinstance(result.get("tool_usage"), dict) else {}
    )
    n_tool_calls = sum(
        v for v in tool_usage.values() if isinstance(v, int) and not isinstance(v, bool)
    )

    input_tokens = _coerce_int(token_usage.get("input_tokens"))
    output_tokens = _coerce_int(token_usage.get("output_tokens"))
    total_tokens: int | None = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    task_skills_raw = run_config.get("task_skills")
    task_skills = (
        [str(s) for s in task_skills_raw] if isinstance(task_skills_raw, list) else []
    )
    required_env_raw = run_config.get("required_env")
    required_env = (
        [str(s) for s in required_env_raw] if isinstance(required_env_raw, list) else []
    )

    return Evidence(
        rollout_dir=rollout_dir,
        schema="flat",
        reward=_coerce_float(result.get("reward")),
        status=str(result.get("status")) if result.get("status") is not None else None,
        n_tool_calls=n_tool_calls,
        total_tokens=total_tokens,
        started_at=timing.get("started_at"),
        ended_at=timing.get("ended_at"),
        skill_mode=(
            str(run_config.get("skill_mode"))
            if run_config.get("skill_mode") is not None
            else None
        ),
        task_skills=task_skills,
        sandbox=(
            str(run_config.get("sandbox"))
            if run_config.get("sandbox") is not None
            else None
        ),
        source_ref=(
            str(run_config.get("source_ref"))
            if run_config.get("source_ref") is not None
            else None
        ),
        verifier_started_after_agent=(
            result.get("verifier_started_after_agent")
            if isinstance(result.get("verifier_started_after_agent"), bool)
            else None
        ),
        error=result.get("error") if isinstance(result.get("error"), str) else None,
        verifier_error=(
            result.get("verifier_error")
            if isinstance(result.get("verifier_error"), str)
            else None
        ),
        required_env=required_env,
        duration_seconds=_coerce_float(timing.get("duration_seconds")),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _load_production_evidence(
    rollout_dir: Path, result: dict[str, Any], run_config: dict[str, Any] | None
) -> Evidence:
    """Adapt a production rollout dir via ``agent_judge.load_rollout_evidence``.

    Lazy-imports the production enforcer so this path is reached only when a
    production schema is detected (and ``benchflow`` is therefore installed).
    """
    from agent_judge import load_rollout_evidence  # lazy: production-only

    prod = load_rollout_evidence(rollout_dir)
    cfg = run_config or _read_json(rollout_dir / "run_config.json") or {}

    # Status is not a first-class production field; derive a best-effort outcome
    # from the recorded reward/error so R-OUTCOME has something to grade.
    status: str | None = None
    if prod.error:
        status = "timeout" if "timeout" in prod.error.lower() else "error"
    elif prod.reward is not None:
        status = "pass" if prod.reward >= 1.0 else "fail"

    skill_mode = cfg.get("skill_mode")
    task_skills_raw = cfg.get("task_skills")
    task_skills = (
        [str(s) for s in task_skills_raw] if isinstance(task_skills_raw, list) else []
    )
    required_env_raw = cfg.get("required_env")
    required_env = (
        [str(s) for s in required_env_raw] if isinstance(required_env_raw, list) else []
    )

    # Production timing lives TOP-LEVEL in result.json (`started_at` /
    # `finished_at`) with per-phase seconds under `timing` (`total`); run_config
    # is only a fallback. Reading `timing.started_at` (the flat-fixture shape)
    # would leave timing empty and false-fail R-TELEMETRY on a real rollout.
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    started_at = _first_str(result.get("started_at"), cfg.get("started_at"))
    ended_at = _first_str(
        result.get("finished_at"), result.get("ended_at"), cfg.get("ended_at")
    )
    duration_seconds = _coerce_float(
        timing.get("total")
        if timing.get("total") is not None
        else timing.get("duration_seconds")
    )

    return Evidence(
        rollout_dir=rollout_dir,
        schema="production",
        reward=prod.reward,
        status=status,
        n_tool_calls=prod.n_tool_calls,
        total_tokens=prod.total_tokens,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        skill_mode=str(skill_mode) if skill_mode is not None else None,
        task_skills=task_skills,
        sandbox=str(cfg.get("sandbox")) if cfg.get("sandbox") is not None else None,
        source_ref=(
            str(cfg.get("source_ref")) if cfg.get("source_ref") is not None else None
        ),
        verifier_started_after_agent=(
            cfg.get("verifier_started_after_agent")
            if isinstance(cfg.get("verifier_started_after_agent"), bool)
            else None
        ),
        error=prod.error,
        verifier_error=prod.verifier_error,
        required_env=required_env,
    )


def load_evidence(
    rollout_dir: Path, run_config: dict[str, Any] | None = None
) -> Evidence:
    """Normalize a rollout dir (flat fixture OR production) into one ``Evidence``.

    Raises ``FileNotFoundError`` when there is no readable ``result.json`` — an
    empty rollout cannot be graded, and silently passing it would let a missing
    measurement through the gate.
    """
    rollout_dir = Path(rollout_dir)
    result = _read_json(rollout_dir / "result.json")
    if result is None:
        raise FileNotFoundError(f"no readable result.json in {rollout_dir}")
    if _is_flat_fixture(result, rollout_dir):
        return _load_flat_evidence(rollout_dir, result)
    return _load_production_evidence(rollout_dir, result, run_config)


# ------------------------------------------------------------------
# Trajectory loading (schema-agnostic, stdlib only).
# ------------------------------------------------------------------


def _load_trajectory_events(rollout_dir: Path) -> list[dict[str, Any]]:
    """Load whichever trajectory JSONL is present, as a list of event dicts."""
    for rel in (
        "trajectory/acp_trajectory.jsonl",
        "trajectory/llm_trajectory.jsonl",
        "acp_trajectory.jsonl",
        "llm_trajectory.jsonl",
    ):
        path = rollout_dir / rel
        if not path.is_file():
            continue
        events: list[dict[str, Any]] = []
        try:
            for line in path.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
        except (OSError, json.JSONDecodeError):
            return events
        return events
    return []


def _iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_iter_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_iter_strings(item))
        return out
    return []


# ------------------------------------------------------------------
# Gate predicates. Each returns (gate_id, status, detail).
# ------------------------------------------------------------------

GateOutcome = tuple[str, GateStatus, str]


def check_realness(evidence: Evidence) -> GateOutcome:
    """R-REAL: n_tool_calls>0, total_tokens>0, reward not null."""
    issues: list[str] = []
    if evidence.n_tool_calls <= 0:
        issues.append(f"n_tool_calls={evidence.n_tool_calls} (expected > 0)")
    if not evidence.total_tokens or evidence.total_tokens <= 0:
        issues.append(f"total_tokens={evidence.total_tokens} (expected > 0)")
    if evidence.reward is None:
        issues.append("reward is null (no verifier score recorded)")
    if issues:
        return ("R-REAL", "fail", "; ".join(issues))
    return ("R-REAL", "pass", "real measurement (tools, tokens, scored)")


def check_outcome(evidence: Evidence) -> GateOutcome:
    """R-OUTCOME: outcome in {pass,fail,normal_timeout} with a valid score."""
    status = (evidence.status or "").strip().lower()
    # A scored timeout is a legitimate ``normal_timeout``; an unscored one is not.
    if status == "timeout":
        status = "normal_timeout" if evidence.reward is not None else "timeout"
    if status not in _VALID_OUTCOMES:
        return (
            "R-OUTCOME",
            "fail",
            f"status={evidence.status!r} not in {sorted(_VALID_OUTCOMES)}",
        )
    if evidence.reward is None:
        return ("R-OUTCOME", "fail", "outcome has no scored reward")
    return ("R-OUTCOME", "pass", f"{status} with reward={evidence.reward}")


def check_artifacts(evidence: Evidence) -> GateOutcome:
    """R-ARTIFACT: trajectory parseable; ATIF/ADP well-formed; no secret leak.

    For a production rollout the ATIF/ADP validators and the secret scanner are
    lazy-imported from :mod:`scenarios`. A flat fixture carries no trainer
    artifacts or trajectory, so only the secret scan applies and the ATIF/ADP
    checks are reported as NA.
    """
    rollout_dir = evidence.rollout_dir
    issues: list[str] = []

    if evidence.schema == "production":
        from scenarios import (  # lazy: production enforcers
            adp_issues,
            atif_issues,
            secret_leak_issues,
        )

        traj = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
        if traj.is_file():
            try:
                for line in traj.read_text().splitlines():
                    line = line.strip()
                    if line:
                        json.loads(line)
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"trajectory not parseable: {exc}")
        issues.extend(f"ATIF: {m}" for m in atif_issues(rollout_dir))
        issues.extend(f"ADP: {m}" for m in adp_issues(rollout_dir))
        issues.extend(f"secret-leak: {m}" for m in secret_leak_issues(rollout_dir))
        if issues:
            return ("R-ARTIFACT", "fail", "; ".join(issues))
        return ("R-ARTIFACT", "pass", "trajectory + ATIF/ADP well-formed; no leak")

    # Flat fixture: no trainer artifacts; do a stdlib-only secret scan and
    # report ATIF/ADP as NA (not applicable to this schema).
    for path in sorted(rollout_dir.glob("*.json")):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if _looks_like_secret(text):
            issues.append(f"secret-leak: {path.name}")
    if issues:
        return ("R-ARTIFACT", "fail", "; ".join(issues))
    return (
        "R-ARTIFACT",
        "na",
        "flat fixture: no trainer artifacts (secret scan clean)",
    )


def _looks_like_secret(text: str) -> bool:
    import re

    patterns = (
        r"sk-[A-Za-z0-9]{16,}",
        r"dtn_[A-Za-z0-9]{16,}",
        r"AQ\.[A-Za-z0-9_\-]{16,}",
        r"AKIA[0-9A-Z]{12,}",
    )
    return any(re.search(p, text) for p in patterns)


def check_telemetry(evidence: Evidence) -> GateOutcome:
    """R-TELEMETRY: token usage + timing present for newly generated data."""
    issues: list[str] = []
    if not evidence.total_tokens or evidence.total_tokens <= 0:
        issues.append("token usage missing or zero")
    has_timing = bool(evidence.started_at) and bool(
        evidence.ended_at or evidence.duration_seconds
    )
    if not has_timing:
        issues.append("timing (started_at/ended_at|duration) incomplete")
    if issues:
        return ("R-TELEMETRY", "fail", "; ".join(issues))
    return ("R-TELEMETRY", "pass", "token usage + timing present")


def check_noskill_leak(evidence: Evidence) -> GateOutcome:
    """S-NOSKILL: no-skill => task_skills_loading==0 AND no leakage markers.

    NA for non-no-skill cells. For a no-skill cell, scans the trajectory text
    for any skill catalog / invocation marker and fails on a hit.
    """
    if evidence.normalized_skill_mode() != "no-skill":
        return ("S-NOSKILL", "na", "cell is not no-skill")

    events = _load_trajectory_events(evidence.rollout_dir)
    hits: list[str] = []
    for event in events:
        for text in _iter_strings(event):
            lowered = text.lower()
            for marker in _NOSKILL_LEAK_MARKERS:
                if marker in lowered:
                    hits.append(marker)
    if hits:
        unique = sorted(set(hits))
        return ("S-NOSKILL", "fail", f"no-skill leakage markers: {unique}")
    return ("S-NOSKILL", "pass", "no-skill: no skill catalog/invocation leaked")


def check_withskill(
    evidence: Evidence, run_config: dict[str, Any] | None = None
) -> GateOutcome:
    """S-WITHSKILL: with-skill => task_skills_loading==1 (quarantine tier).

    Drives :mod:`extract_harness_skills` over the production
    ``trajectory/llm_trajectory.jsonl``. ``unknown`` / ``catalog_not_serialized``
    / ``skill_count==0`` / ``manual_review_required`` route to QUARANTINE (not a
    hard reject) — they are escalated to the codex reviewer.
    """
    if evidence.normalized_skill_mode() != "with-skill":
        return ("S-WITHSKILL", "na", "cell is not with-skill")

    traj = evidence.rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    if not traj.is_file():
        # No harness LLM trajectory to introspect -> can't mechanically confirm
        # loading; quarantine for codex review rather than reject.
        return (
            "S-WITHSKILL",
            "quarantine",
            "no trajectory/llm_trajectory.jsonl; task_skills_loading unverifiable",
        )

    extracted = _run_extract_harness_skills(traj, evidence.task_skills)
    if extracted is None:
        return (
            "S-WITHSKILL",
            "quarantine",
            "extract_harness_skills failed; manual review",
        )

    catalog_status = str(extracted.get("catalog_status", "unknown"))
    harness = str(extracted.get("harness", "unknown"))
    loading = extracted.get("task_skills_loading")
    manual = bool(extracted.get("manual_review_required"))
    skill_count = extracted.get("skill_count")

    if (
        harness == "unknown"
        or catalog_status in {"catalog_not_serialized", "unknown"}
        or skill_count == 0
        or manual
    ):
        return (
            "S-WITHSKILL",
            "quarantine",
            f"harness={harness} catalog_status={catalog_status} "
            f"skill_count={skill_count} manual_review={manual} -> codex",
        )
    if loading == 1:
        return ("S-WITHSKILL", "pass", "with-skill: task_skills_loading==1")
    return (
        "S-WITHSKILL",
        "quarantine",
        f"task_skills_loading={loading} (expected 1) -> codex review",
    )


def _run_extract_harness_skills(
    trajectory: Path, task_skills: list[str]
) -> dict[str, Any] | None:
    """Invoke the skill's ``extract_harness_skills.py`` and parse its JSON.

    Run as a subprocess (it is a standalone stdlib CLI, not importable as a
    package module) so this stays import-light and tolerant of the script
    living under the skill tree.
    """
    import subprocess

    script = (
        _HERE.parents[2]
        / ".agents"
        / "skills"
        / "benchflow-experiment-review"
        / "scripts"
        / "extract_harness_skills.py"
    )
    if not script.is_file():
        return None
    cmd = [sys.executable, str(script), str(trajectory)]
    for skill in task_skills:
        cmd += ["--task-skill", skill]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def check_verifier_tamper(evidence: Evidence) -> GateOutcome:
    """V-TAMPER: no agent write/delete/chmod of score-defining files.

    Reuses :func:`agent_judge._scan_verifier_tamper` over the FULL trajectory
    (the production ACP and nested ``tool_calls`` shapes) when ``benchflow`` is
    importable, and ADDITIONALLY runs a stdlib-only scan over the flat-fixture
    ``{tool, args}`` event shape so the gate stays deterministic without an
    install. The union of both scanners' findings is reported.
    """
    events = _load_trajectory_events(evidence.rollout_dir)
    if not events:
        return ("V-TAMPER", "na", "no trajectory events to scan")

    flagged: list[str] = []
    try:
        from agent_judge import _scan_verifier_tamper  # lazy: production shapes

        flagged.extend(_scan_verifier_tamper(events))
    except Exception:  # pragma: no cover - no benchflow install (flat fixtures)
        pass
    # Always run the stdlib scan too: it covers the flat-fixture trajectory
    # shape the production scanner does not, and needs no install.
    flagged.extend(_scan_flat_tamper(events))

    if flagged:
        unique = sorted(set(flagged))
        return ("V-TAMPER", "fail", f"verifier/test/grader mutations: {unique}")
    return ("V-TAMPER", "pass", "no score-defining file mutations")


# Score-defining file names (verifier/test/grader/reward) — stdlib mirror of the
# production ``agent_judge._VERIFIER_FILE_RE`` used for the flat-fixture shape.
def _verifier_file_hit(text: str) -> bool:
    import re

    return bool(
        re.search(
            r"(verif|grader|conftest|reward\.(json|txt)|run_tests|run_all|"
            r"score\.(json|txt)|results/score|\btests?/|\btests\b|"
            r"test_[\w.-]+|[\w.-]+_test\.)",
            text,
            re.IGNORECASE,
        )
    )


def _tamper_op_hit(text: str) -> bool:
    import re

    return bool(
        re.search(
            r"(>{1,2}\s|\brm\b|\bsed\s+-i\b|\bchmod\b|\bmv\b|\btruncate\b|"
            r"\btee\b|\b(echo|printf|cat)\b[^|]*>)",
            text,
            re.IGNORECASE,
        )
    )


_FLAT_WRITE_TOOLS = {
    "write_file",
    "str_replace",
    "edit_file",
    "create_file",
    "edit",
    "write",
}
_FLAT_SHELL_TOOLS = {"bash", "shell", "run", "execute", "run_command", "sh"}


def _scan_flat_tamper(events: list[dict[str, Any]]) -> list[str]:
    """Stdlib scan of the flat-fixture ``{"tool": str, "args": {...}}`` shape.

    A write-like tool targeting a score-defining file is a tamper (the mutation
    is implied by the tool); a shell tool is a tamper only when it carries a
    destructive op against a score-defining file (running tests read-only is
    fine). Mirrors the production scanner's policy for this third event shape.
    """
    flagged: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        tool = str(event.get("tool", "")).lower()
        if not tool:
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        if tool in _FLAT_WRITE_TOOLS:
            path = str(args.get("path") or args.get("file_path") or "")
            if path and _verifier_file_hit(path):
                flagged.append(f"{tool} -> {path}")
        elif tool in _FLAT_SHELL_TOOLS:
            cmd = str(args.get("cmd") or args.get("command") or "")
            if cmd and _verifier_file_hit(cmd) and _tamper_op_hit(cmd):
                flagged.append(f"{tool}: {cmd[:160]}")
    return flagged


# Attribution labels (CONTRACT C-ATTRIB / SKILL "Capability Attribution").
AttributionLabel = Literal[
    "model-capability", "experiment-fidelity", "unknown", "not-applicable"
]


@dataclass(frozen=True)
class CapabilityAttribution:
    """The cause attribution for one fail/timeout rollout.

    ``label`` is the SKILL's three-way verdict (plus ``not-applicable`` for a
    clean pass): a model-capability failure is a legitimate negative result that
    counts as healthy ``fail`` / ``normal_timeout``; an experiment-fidelity
    failure is an infra/env flaw that must be rerun or quarantined; ``unknown``
    escalates to the codex reviewer. ``reasons`` lists the concrete evidence.
    """

    label: AttributionLabel
    reasons: list[str]


def classify_capability(evidence: Evidence) -> CapabilityAttribution:
    """Attribute a fail/timeout to model-capability vs experiment-fidelity.

    Uses the production :data:`check_results.INFRA_ERROR_CATEGORIES` plus the
    ``verifier_error`` field and the run-config's required-resource list. An
    infra-category error, a verifier error, or a missing required resource the
    error blames points at experiment fidelity (a harness/env flaw, not the
    model). A clean, scored fail/timeout with real work is model-capability. A
    passing run is ``not-applicable``.
    """
    status = (evidence.status or "").strip().lower()
    reward = evidence.reward

    if reward is not None and reward >= 1.0 and status in {"pass", ""}:
        return CapabilityAttribution("not-applicable", ["passing run"])

    fidelity: list[str] = []

    infra_categories = _infra_error_categories()
    if evidence.error:
        category = _classify_error_lazy(evidence.error)
        if category in infra_categories:
            fidelity.append(f"infra error category={category}")
    if evidence.verifier_error:
        fidelity.append(f"verifier error: {evidence.verifier_error[:120]}")

    # A required resource named by the run config that the error blames is an
    # experiment-fidelity failure (the env was misprovisioned, not the model).
    if evidence.required_env and evidence.error:
        err_lower = evidence.error.lower()
        missing = [key for key in evidence.required_env if key.lower() in err_lower]
        if missing:
            fidelity.append(f"missing required resource(s): {missing}")
    # An unscored timeout with no tokens is an infra/experiment-fidelity failure.
    if reward is None and status in {"timeout", "error"} and not evidence.total_tokens:
        fidelity.append("unscored run with no token usage (infra failure)")

    if fidelity:
        return CapabilityAttribution("experiment-fidelity", fidelity)

    if reward is not None and status in {"fail", "normal_timeout"}:
        return CapabilityAttribution(
            "model-capability", ["real, scored negative result"]
        )
    if status == "timeout" and evidence.total_tokens:
        return CapabilityAttribution(
            "model-capability", ["real work before timeout, no score"]
        )
    return CapabilityAttribution("unknown", ["attribution unknown -> codex review"])


def capability_attribution(evidence: Evidence) -> GateOutcome:
    """C-ATTRIB gate: project :func:`classify_capability` onto a gate outcome.

    ``not-applicable`` -> na; ``model-capability`` -> pass (healthy negative);
    ``experiment-fidelity`` / ``unknown`` -> quarantine (rerun or codex review).
    The classification itself lives in :func:`classify_capability` so the review
    pack can attach the model-capability|experiment-fidelity|unknown label and
    reasons without re-deriving them.
    """
    attribution = classify_capability(evidence)
    detail = "; ".join(attribution.reasons) or attribution.label
    if attribution.label == "not-applicable":
        return ("C-ATTRIB", "na", "passing run; attribution not applicable")
    if attribution.label == "model-capability":
        return ("C-ATTRIB", "pass", f"model-capability: {detail}")
    if attribution.label == "experiment-fidelity":
        return ("C-ATTRIB", "quarantine", f"experiment-fidelity: {detail}")
    return ("C-ATTRIB", "quarantine", f"{detail}")


def _infra_error_categories() -> frozenset[str]:
    try:
        from check_results import INFRA_ERROR_CATEGORIES  # lazy

        return frozenset(INFRA_ERROR_CATEGORIES)
    except Exception:  # pragma: no cover - flat fixtures have no benchflow
        return frozenset(
            {
                "install_failure",
                "timeout",
                "idle_timeout",
                "pipe_closed",
                "sandbox_setup",
                "infra_failure",
            }
        )


def _classify_error_lazy(error: str) -> str | None:
    try:
        from check_results import classify_error  # lazy

        return classify_error(error)
    except Exception:  # pragma: no cover - flat fixtures have no benchflow
        lowered = error.lower()
        if "timeout" in lowered:
            return "timeout"
        if "install" in lowered:
            return "install_failure"
        if "missing" in lowered and ("key" in lowered or "env" in lowered):
            return "infra_failure"
        return None


# ------------------------------------------------------------------
# P-SCHEMA / P-REWARD parity over a paired docker+daytona cell.
# ------------------------------------------------------------------


def parity_schema_diff(dir_docker: Path, dir_daytona: Path) -> GateOutcome:
    """P-SCHEMA: token/timing/schema equality on a model-bearing paired cell.

    Compares the two sandbox rollouts' schemas (same fields present) and, for a
    model-bearing cell, that token/timing telemetry is present on both. A
    reward-only oracle pair (no model tokens) is NA for P-SCHEMA — that pair is
    covered by P-REWARD (reward equality) instead.
    """
    try:
        ev_d = load_evidence(dir_docker)
        ev_y = load_evidence(dir_daytona)
    except FileNotFoundError as exc:
        return ("P-SCHEMA", "fail", f"parity pair unreadable: {exc}")

    model_bearing = bool(ev_d.total_tokens) or bool(ev_y.total_tokens)
    if not model_bearing:
        return ("P-SCHEMA", "na", "reward-only oracle pair; P-SCHEMA not applicable")

    issues: list[str] = []
    # Token telemetry must be present on BOTH sides of a model-bearing pair.
    if not ev_d.total_tokens:
        issues.append("docker side missing token usage")
    if not ev_y.total_tokens:
        issues.append("daytona side missing token usage")
    # Timing must be present on both.
    for label, ev in (("docker", ev_d), ("daytona", ev_y)):
        if not (ev.started_at and (ev.ended_at or ev.duration_seconds)):
            issues.append(f"{label} side missing timing")
    # Schema shape: the set of populated telemetry fields must agree.
    shape_d = _telemetry_shape(ev_d)
    shape_y = _telemetry_shape(ev_y)
    if shape_d != shape_y:
        issues.append(
            f"schema shape differs: docker={sorted(shape_d)} daytona={sorted(shape_y)}"
        )
    if issues:
        return ("P-SCHEMA", "fail", "; ".join(issues))
    return ("P-SCHEMA", "pass", "token/timing/schema parity holds across sandboxes")


def _telemetry_shape(evidence: Evidence) -> set[str]:
    shape: set[str] = set()
    if evidence.total_tokens is not None:
        shape.add("total_tokens")
    if evidence.input_tokens is not None:
        shape.add("input_tokens")
    if evidence.output_tokens is not None:
        shape.add("output_tokens")
    if evidence.started_at:
        shape.add("started_at")
    if evidence.ended_at:
        shape.add("ended_at")
    if evidence.duration_seconds is not None:
        shape.add("duration_seconds")
    if evidence.reward is not None:
        shape.add("reward")
    return shape


# ------------------------------------------------------------------
# Network hardening (Q3 SCOPE-GATED LANE — STATIC assertion).
# ------------------------------------------------------------------

# Network mode literals mirror ``benchflow.task.config.NetworkMode`` (kept as
# string constants so this stays stdlib-only / importable without benchflow).
_NET_NO_NETWORK = "no-network"
_NET_ALLOWLIST = "allowlist"
_NET_PUBLIC = "public"
_VALID_NETWORK_MODES = frozenset({_NET_NO_NETWORK, _NET_ALLOWLIST, _NET_PUBLIC})


def _norm_network_mode(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower().replace("_", "-")


def network_hardening(
    task_config: dict[str, Any], *, verifier_or_sandbox_pr: bool = False
) -> GateOutcome:
    """Static network-policy hardening assertion over a task config (Q3).

    No ``bench eval run --network`` flag exists and ``network_mode`` is never
    serialized into a rollout artifact (Q3): network is a per-task config field,
    so this is a STATIC check over the task's declared policy, not over a
    produced rollout.

    Policy (CONTRACT Q3): the default safe posture is ``no-network``. Network
    access is only acceptable as ``allowlist`` with a NON-EMPTY ``allowed_hosts``
    set. A bare ``public`` mode is always flagged; on a PR that touches the
    verifier or the sandbox/lockdown surface a ``public`` mode is a hard
    ``fail`` (blocker) because that surface controls the isolation boundary.

    Returns the ``V-NETWORK`` gate outcome. ``pass`` for a hardened config,
    ``fail`` for an unsafe one (missing allowlist hosts, or ``public`` on a
    verifier/sandbox PR), ``quarantine`` for a ``public`` config on an unrelated
    PR (documented, needs human sign-off), ``na`` when no policy is declared.
    """
    mode = _norm_network_mode(task_config.get("network_mode"))
    raw_hosts = task_config.get("allowed_hosts")
    allowed_hosts = [
        str(h).strip()
        for h in (raw_hosts if isinstance(raw_hosts, list) else [])
        if str(h).strip()
    ]

    if mode is None:
        # No declared policy => the runtime default (no-network) applies; an
        # explicit allowlist list without a mode is a misconfiguration.
        if allowed_hosts:
            return (
                "V-NETWORK",
                "fail",
                "allowed_hosts declared without network_mode='allowlist'",
            )
        return ("V-NETWORK", "na", "no network_mode declared; runtime default applies")

    if mode not in _VALID_NETWORK_MODES:
        return ("V-NETWORK", "fail", f"unknown network_mode={mode!r}")

    if mode == _NET_NO_NETWORK:
        if allowed_hosts:
            return (
                "V-NETWORK",
                "fail",
                "allowed_hosts is only valid for network_mode='allowlist'",
            )
        return ("V-NETWORK", "pass", "no-network (hardened default)")

    if mode == _NET_ALLOWLIST:
        if not allowed_hosts:
            return (
                "V-NETWORK",
                "fail",
                "network_mode='allowlist' requires a non-empty allowed_hosts",
            )
        return (
            "V-NETWORK",
            "pass",
            f"allowlist hardened: hosts={sorted(allowed_hosts)}",
        )

    # mode == public
    if verifier_or_sandbox_pr:
        return (
            "V-NETWORK",
            "fail",
            "network_mode='public' on a verifier/sandbox PR (isolation boundary)",
        )
    return (
        "V-NETWORK",
        "quarantine",
        "network_mode='public' (no allowlist) — requires human sign-off",
    )


# ------------------------------------------------------------------
# Pinned-baseline reward-BAND parity (P-SCHEMA / P-REWARD before/after).
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ParityBandResult:
    """Outcome of one pinned-baseline reward-band parity comparison."""

    status: GateStatus
    detail: str
    issues: list[str]


def parity_baseline_band(
    benchflow_root: Path,
    baseline_root: Path,
    tasks: list[str],
    deltas: dict[str, float] | None = None,
) -> ParityBandResult:
    """P-SCHEMA/P-REWARD before/after band parity vs a pinned Harbor baseline.

    Wraps :func:`check_skillsbench_harbor_parity.main` (the authoritative
    pinned-baseline reward-BAND checker) by invoking it as a SUBPROCESS so this
    stays import-light: the parity checker imports ``benchflow`` at module top,
    which must not be paid by the flat-fixture unit tests. The "same behavior"
    bar is schema + lifecycle + reward-band parity, NOT bit-identity.

    ``deltas`` overrides the band tolerances (keys ``max_outcome_rate_delta``,
    ``max_mean_reward_delta``, ``max_task_reward_delta``). Returns a
    ``ParityBandResult``: ``pass`` when the bands hold, ``fail`` (with the
    surfaced issue lines) otherwise.
    """
    deltas = deltas or {}
    script = (
        _HERE.parent / "check_skillsbench_harbor_parity.py"
    )  # sibling integration tool
    if not script.is_file():
        return ParityBandResult(
            "fail",
            "check_skillsbench_harbor_parity.py not found",
            ["parity checker missing"],
        )

    cmd = [
        sys.executable,
        str(script),
        "--benchflow-root",
        str(benchflow_root),
        "--harbor-baseline-root",
        str(baseline_root),
        "--max-outcome-rate-delta",
        str(deltas.get("max_outcome_rate_delta", 0.25)),
        "--max-mean-reward-delta",
        str(deltas.get("max_mean_reward_delta", 0.25)),
        "--max-task-reward-delta",
        str(deltas.get("max_task_reward_delta", 0.0)),
    ]
    for task in tasks:
        cmd += ["--task", task]

    import subprocess

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return ParityBandResult(
            "fail", f"parity checker did not run: {exc}", [str(exc)]
        )

    output = (proc.stdout or "") + (proc.stderr or "")
    issues = [line[2:].strip() for line in output.splitlines() if line.startswith("- ")]
    if proc.returncode == 0:
        return ParityBandResult("pass", "pinned-baseline reward-band parity holds", [])
    detail = "pinned-baseline parity FAIL: " + (
        "; ".join(issues) if issues else output.strip()[-300:]
    )
    return ParityBandResult("fail", detail, issues)


# ------------------------------------------------------------------
# grade_rollout — run all deterministic+quarantine gates over one rollout.
# ------------------------------------------------------------------


@dataclass
class GateRecord:
    """One graded gate result, carrying its enforcement tier for the verdict."""

    id: str
    status: GateStatus
    detail: str
    enforcement: Enforcement

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "detail": self.detail,
            "enforcement": self.enforcement,
        }


# Gates whose enforcement is non-deterministic; they are NEVER faked here. They
# surface as markers so the review pack routes them to codex/residual review.
_NON_DETERMINISTIC_MARKERS: list[tuple[str, str]] = [
    ("V-LIFECYCLE", "no lifecycle timestamps in artifacts; residual/codex review"),
    ("V-ENVHARDEN", "sandbox-fs property; env-probe else residual/codex review"),
    ("V-REWARDHACK", "reward-consistency judged by codex LLM judge"),
]


def grade_rollout(
    rollout_dir: Path, run_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Grade one rollout across the deterministic + quarantine gates.

    Returns ``{gates, deterministic_reject, quarantines}``. ``deterministic_reject``
    is True iff any deterministic-tier gate fired ``fail``. Quarantine-tier
    ``quarantine`` outcomes collect into ``quarantines`` and never set the reject.
    """
    rollout_dir = Path(rollout_dir)
    evidence = load_evidence(rollout_dir, run_config)

    outcomes: list[GateOutcome] = [
        check_realness(evidence),
        check_outcome(evidence),
        check_artifacts(evidence),
        check_telemetry(evidence),
        check_noskill_leak(evidence),
        check_withskill(evidence, run_config),
        check_verifier_tamper(evidence),
        capability_attribution(evidence),
    ]

    gates: list[GateRecord] = []
    deterministic_reject = False
    quarantines: list[str] = []

    for gate_id, status, detail in outcomes:
        gate = GATES_BY_ID[gate_id]
        gates.append(GateRecord(gate_id, status, detail, gate.enforcement))
        if status == "fail" and gate.enforcement == "deterministic":
            deterministic_reject = True
        if status == "quarantine":
            quarantines.append(f"{gate_id}: {detail}")

    # Surface the non-deterministic gates as markers (never fabricated).
    for gate_id, detail in _NON_DETERMINISTIC_MARKERS:
        gate = GATES_BY_ID[gate_id]
        gates.append(GateRecord(gate_id, "na", detail, gate.enforcement))

    return {
        "rollout_dir": str(rollout_dir),
        "schema": evidence.schema,
        "gates": [g.to_dict() for g in gates],
        "deterministic_reject": deterministic_reject,
        "quarantines": quarantines,
    }


# ------------------------------------------------------------------
# CLI (CONTRACT section E).
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic+quarantine rubric gates over one rollout dir.",
    )
    parser.add_argument("rollout_dir", type=Path, help="Rollout dir to grade.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json override (else read from the rollout dir).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout.")
    args = parser.parse_args(argv)

    run_config = None
    if args.run_config is not None:
        run_config = _read_json(args.run_config)

    try:
        report = grade_rollout(args.rollout_dir, run_config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        verdict = "REJECT" if report["deterministic_reject"] else "OK"
        print(f"[{verdict}] {report['rollout_dir']} (schema={report['schema']})")
        for gate in report["gates"]:
            print(f"  {gate['status']:>10}  {gate['id']:<12} {gate['detail']}")
        for q in report["quarantines"]:
            print(f"  quarantine: {q}")

    return 1 if report["deterministic_reject"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
