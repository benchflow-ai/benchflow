"""Every rubric review of one trial, read for the ``benchflow.trial`` export.

A trial can be reviewed against a rubric in two ways, and both are listed:

- **revisions** — the automatic reviewer that ``bench eval run`` starts for a
  task with a rubric, and ``bench eval score`` (a rerun of it). Each attempt
  writes ``scoring/<attempt>.json`` inside the trial folder, and the one that
  ``result.json`` names (``scoring.revision``) set the trial's reward.
- **audits** — ``bench review``, a detached review that writes
  ``review*/**/review_report.json`` next to the job and never changes a
  reward. Looked up at the nearest of four folders above
  the trial that holds a report naming it.

Each entry carries the rubric definition (criteria, weights, scales), the
reviewer (agent, model, reasoning effort, the reviewer's own run), the
verdict for every criterion with the evidence paths its explanation cites,
and the weighted reward arithmetic.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_SEARCH_DEPTH = 4
_REPORT = "review_report.json"

#: Evidence the reviewer sees is mounted at /evidence/{trial,task,workspace};
#: explanations cite files there, sometimes without the /evidence prefix.
_EVIDENCE = re.compile(
    r"(?<![\w/.-])(?:/evidence/)?(trial|task|workspace)/([^\s'\"`()\[\]{},;<>]+)"
)
_LINE = re.compile(r"^(.*?):(\d+)(?:-\d+)?$")


def evidence_refs(text: str) -> list[dict[str, Any]]:
    """The evidence files an explanation cites, in order, without repeats."""
    refs: list[dict[str, Any]] = []
    for area, raw in _EVIDENCE.findall(text or ""):
        path = raw.rstrip(".:")
        line = None
        match = _LINE.match(path)
        if match:
            path, line = match.group(1), int(match.group(2))
        ref = {"area": area, "path": path, "line": line}
        if path and ref not in refs:
            refs.append(ref)
    return refs


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _kind(blocker: Any) -> str:
    if blocker is None:
        return "legacy"
    return "blocker" if blocker in (1, True) else "scored"


_SCALES: dict[str, list[Any]] = {
    "blocker": ["pass", "fail"],
    "scored": [0, 1, 2],
    "legacy": ["pass", "fail", "not_applicable"],
}


def _rubric(
    definition: dict[str, Any] | None,
    metadata: list[Any] | None,
    **provenance: Any,
) -> dict[str, Any] | None:
    """The rubric as the reviewer judged it: the snapshot, else its metadata."""
    rows = definition.get("criteria") if definition else None
    if not isinstance(rows, list):
        rows = metadata if isinstance(metadata, list) else None
    if rows is None:
        return None
    criteria = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            continue
        kind = _kind(row.get("blocker"))
        criteria.append(
            {
                "name": row["name"],
                "kind": kind,
                "weight": _int(row.get("weight")),
                "scale": _SCALES[kind],
                "description": _text(row.get("description")),
                "guidance": _text(row.get("guidance")),
            }
        )
    contract = provenance.pop("contract", None) or (
        "v0.1" if criteria and criteria[0]["kind"] == "legacy" else "v0.2"
    )
    return {"contract": contract, "criteria": criteria, **provenance}


def _verdicts(
    rubric: dict[str, Any] | None, checks: Any
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    """Per-criterion verdicts plus the weighted points they add up to."""
    if not isinstance(checks, dict) or not checks:
        return [], None, None
    known = {c["name"]: c for c in (rubric or {}).get("criteria", [])}
    names = list(known) + [n for n in checks if n not in known]
    verdicts = []
    points_total = max_total = 0
    complete = True
    for name in names:
        check = checks.get(name)
        check = check if isinstance(check, dict) else {}
        criterion = known.get(name, {})
        kind = criterion.get("kind") or ("scored" if "score" in check else "legacy")
        weight = criterion.get("weight")
        score = _int(check.get("score"))
        points = max_points = None
        if kind == "scored" and weight is not None:
            max_points = 2 * weight
            max_total += max_points
            if score is None:
                complete = False
            else:
                points = score * weight
                points_total += points
        explanation = _text(check.get("explanation")) or ""
        verdicts.append(
            {
                "name": name,
                "kind": kind,
                "weight": weight,
                "outcome": _text(check.get("outcome")),
                "score": score,
                "points": points,
                "max_points": max_points,
                "explanation": explanation,
                "evidence": evidence_refs(explanation),
            }
        )
    if not complete or max_total == 0:
        return verdicts, None, None
    return verdicts, points_total, max_total


def _points_formula(points: int | None, maximum: int | None) -> str:
    ratio = "weighted_points / max_weighted_points"
    return f"{ratio} = {points} / {maximum}" if points is not None else ratio


def _decision(passed: Any, quality: Any) -> str | None:
    """The publication band ``bench review`` would give the same verdict."""
    from benchflow.review.scoring import (
        PUBLISHABLE_QUALITY,
        REVISIONS_QUALITY,
        PublicationDecision,
    )

    if not isinstance(quality, int | float) or isinstance(quality, bool):
        return None
    if passed is not True or quality < REVISIONS_QUALITY:
        return PublicationDecision.NOT_PUBLISHABLE.value
    if quality < PUBLISHABLE_QUALITY:
        return PublicationDecision.PRESENTABLE_WITH_REVISIONS.value
    return PublicationDecision.PUBLISHABLE.value


def _recorded_at(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return None


def _revisions(trial_dir: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    folder = trial_dir / "scoring"
    if not folder.is_dir():
        return []
    committed = result.get("scoring")
    current = committed.get("revision") if isinstance(committed, dict) else None
    paths = sorted(folder.glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    entries = []
    for path in paths:
        details = _read(path)
        if details is None:
            continue
        source = f"scoring/{path.name}"
        scoring = details.get("scoring")
        scoring = scoring if isinstance(scoring, dict) else {}
        reviewer = details.get("reviewer")
        reviewer = reviewer if isinstance(reviewer, dict) else {}
        snapshot = _text(details.get("rubric_snapshot"))
        definition = _read(trial_dir / snapshot) if snapshot else None
        rubric = _rubric(
            definition,
            None,
            contract=_text(details.get("contract")),
            path=_text(details.get("rubric_path")),
            sha256=_text(details.get("rubric_sha256")),
            snapshot=snapshot,
        )
        verdicts, points, maximum = _verdicts(rubric, details.get("checks"))
        complete = scoring.get("status") == "complete"
        passed = scoring.get("passed") if complete else None
        rubric_reward = scoring.get("rubric_reward")
        entries.append(
            {
                "kind": "revision",
                "id": _text(details.get("attempt")) or path.stem,
                "source": source,
                "current": source == current,
                "recorded_at": _recorded_at(path),
                "status": "complete" if complete else "error",
                "error": _text(scoring.get("error")),
                "review_valid": details.get("review_valid") is True,
                "summary": _text(details.get("summary")),
                "reviewer": {
                    "agent": _text(reviewer.get("agent")),
                    "model": _text(reviewer.get("model")),
                    "reasoning_effort": _text(reviewer.get("reasoning_effort")),
                    "environment": _text(reviewer.get("environment")),
                    "run": _text(scoring.get("reviewer_run")),
                },
                "rubric": rubric,
                "verdicts": verdicts,
                "reward": {
                    "policy": _text(scoring.get("policy")),
                    "tests_pass": scoring.get("tests_pass"),
                    "all_blockers_pass": scoring.get("all_blockers_pass"),
                    "failed_blockers": list(scoring.get("failed_blockers") or []),
                    "weighted_points": points,
                    "max_weighted_points": maximum,
                    "rubric_reward": rubric_reward,
                    "gated_quality": (rubric_reward if passed else 0.0)
                    if complete
                    else None,
                    "decision": _decision(passed, rubric_reward) if complete else None,
                    "verifier_reward": scoring.get("verifier_reward"),
                    "passed": passed,
                    "reward": (rubric_reward if passed else 0.0) if complete else None,
                    "formula": "reward = rubric_reward if tests pass and every "
                    "blocker passes, else 0; rubric_reward = "
                    + _points_formula(points, maximum),
                },
                "notes": [],
            }
        )
    return entries


def _report_rubric(
    report_dir: Path, report: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any] | None:
    header = report.get("rubric")
    header = header if isinstance(header, dict) else {}
    path = _text(entry.get("rubric_path")) or _text(header.get("path"))
    definition = None
    if path:
        candidate = Path(path)
        definition = _read(candidate if candidate.is_absolute() else report_dir / path)
    return _rubric(
        definition,
        entry.get("criterion_metadata"),
        contract=_text(entry.get("rubric_contract")),
        path=path,
        sha256=None,
        snapshot=None,
    )


def _audits(trial_dir: Path) -> list[dict[str, Any]]:
    name = trial_dir.name
    ancestor = trial_dir
    for _ in range(_SEARCH_DEPTH):
        ancestor = ancestor.parent
        entries = []
        for review_dir in sorted(ancestor.glob("review*")):
            if not review_dir.is_dir():
                continue
            for path in sorted(review_dir.rglob(_REPORT)):
                report = _read(path)
                if report is None:
                    continue
                reviewer = report.get("reviewer")
                reviewer = reviewer if isinstance(reviewer, dict) else {}
                trials = report.get("trials")
                for entry in trials if isinstance(trials, list) else []:
                    if not isinstance(entry, dict) or entry.get("trial_name") != name:
                        continue
                    entries.append(_audit(path, ancestor, report, reviewer, entry))
        if entries:
            return entries
        if ancestor == ancestor.parent:
            break
    return []


def _audit(
    path: Path,
    root: Path,
    report: dict[str, Any],
    reviewer: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    rubric = _report_rubric(path.parent, report, entry)
    verdicts, points, maximum = _verdicts(rubric, entry.get("checks"))
    scoring = entry.get("scoring")
    scoring = scoring if isinstance(scoring, dict) else {}
    valid = entry.get("review_valid") is True
    recorded_points = _int(scoring.get("weighted_points"))
    recorded_max = _int(scoring.get("max_weighted_points"))
    points = recorded_points if recorded_points is not None else points
    maximum = recorded_max if recorded_max is not None else maximum
    return {
        "kind": "audit",
        "id": str(path.parent.relative_to(root)),
        "source": str(path),
        "current": False,
        "recorded_at": _recorded_at(path),
        "status": "complete" if valid else "error",
        "error": _text(entry.get("error")),
        "review_valid": valid,
        "summary": _text(entry.get("summary")),
        "reviewer": {
            "agent": _text(reviewer.get("agent")),
            "model": _text(reviewer.get("model")),
            "reasoning_effort": _text(reviewer.get("reasoning_effort")),
            "environment": _text(reviewer.get("environment")),
            "run": _text(entry.get("reviewer_rollout")),
        },
        "rubric": rubric,
        "verdicts": verdicts,
        "reward": {
            "policy": None,
            "tests_pass": scoring.get("deterministic_pass"),
            "all_blockers_pass": scoring.get("all_blockers_pass"),
            "failed_blockers": list(scoring.get("failed_blockers") or []),
            "weighted_points": points,
            "max_weighted_points": maximum,
            "rubric_reward": scoring.get("raw_quality"),
            "gated_quality": scoring.get("gated_quality"),
            "decision": _text(scoring.get("decision")),
            "verifier_reward": None,
            "passed": None,
            "reward": None,
            "formula": "raw_quality = "
            + _points_formula(points, maximum)
            + "; gated_quality = raw_quality if tests pass and every blocker "
            "passes, else 0 (an audit does not change the trial's reward)",
        },
        "notes": [n for n in entry.get("notes") or [] if isinstance(n, str)],
    }


def rubric_reviews(trial_dir: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Scoring revisions (oldest first), then detached audits."""
    return _revisions(trial_dir, result) + _audits(trial_dir)
