"""Read integrity verdicts back: one trial, or every trial of a job.

``IntegrityVerdict`` has a boolean ``exploited`` and a ``reason``, the shape
``benchflow.integrations.rewards.apply_integrity`` takes, so a trainer can
turn an exploited rollout into a flagged 0 in one explicit step. BenchFlow
itself never changes a reward because of a verdict.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.integrity.emit import CLAIM_VERDICT_FILE, INTEGRITY_DIRNAME

VERDICT_ORDER = ("AgentViolation", "Rejected", "VectorExposed", "Checked")


@dataclass(frozen=True)
class IntegrityVerdict:
    """A trial's integrity verdict, as written to ``integrity/claim_verdict.json``.

    ``verdict`` is one of ``Checked``, ``VectorExposed``, ``AgentViolation`` and
    ``Rejected``. ``exploited`` is True exactly for ``AgentViolation``: direct
    agent-attributed evidence of a forbidden crossing. ``severity`` is
    ``RewardRelevant`` when that happened in a run the verifier passed.
    """

    verdict: str
    exploited: bool
    reason: str
    certification: str = "Rejected"
    task_outcome: str = "TaskError"
    mode: str = "audit"
    severity: str = "None"
    agent_evidence: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    reward: float | None = None
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_claim(
        cls, claim: dict[str, Any], path: Path | None = None
    ) -> IntegrityVerdict:
        core = claim.get("core") if isinstance(claim.get("core"), dict) else {}
        verdict = str(
            claim.get("core_verdict") or core.get("core_verdict") or "Rejected"
        )
        flags = claim.get("final_flags")
        return cls(
            verdict=verdict,
            exploited=verdict == "AgentViolation",
            reason=str(claim.get("reason") or ""),
            certification=str(
                claim.get("certification") or core.get("certification") or "Rejected"
            ),
            task_outcome=str(
                claim.get("task_outcome") or core.get("task_outcome") or "TaskError"
            ),
            mode=str(claim.get("mode") or "audit"),
            severity=str(claim.get("event_severity") or "None"),
            agent_evidence=tuple(core.get("agent_evidence") or ()),
            flags=tuple(name for name, value in (flags or {}).items() if value is True)
            if isinstance(flags, dict)
            else (),
            reward=claim.get("reward")
            if isinstance(claim.get("reward"), int | float)
            else None,
            path=path,
            raw=claim,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "exploited": self.exploited,
            "reason": self.reason,
            "certification": self.certification,
            "task_outcome": self.task_outcome,
            "mode": self.mode,
            "severity": self.severity,
            "agent_evidence": list(self.agent_evidence),
            "flags": list(self.flags),
            "reward": self.reward,
            "path": str(self.path) if self.path is not None else None,
        }


def verdict_path(trial_dir: str | Path) -> Path:
    return Path(trial_dir) / INTEGRITY_DIRNAME / CLAIM_VERDICT_FILE


def read_verdict(trial_dir: str | Path) -> IntegrityVerdict | None:
    """The trial's verdict, or None when the trial was not audited."""

    path = verdict_path(trial_dir)
    try:
        claim = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(claim, dict):
        return None
    return IntegrityVerdict.from_claim(claim, path=path)


@dataclass(frozen=True)
class IntegrityReport:
    """Verdicts for a set of trials (``Job.integrity()``)."""

    verdicts: dict[Path, IntegrityVerdict | None]

    @classmethod
    def of(cls, trial_dirs: Iterable[str | Path]) -> IntegrityReport:
        return cls({Path(d): read_verdict(d) for d in trial_dirs})

    def counts(self) -> dict[str, int]:
        """Trials per verdict; ``not_audited`` counts trials with no verdict."""
        counter = Counter(
            verdict.verdict if verdict is not None else "not_audited"
            for verdict in self.verdicts.values()
        )
        return {
            key: counter[key] for key in (*VERDICT_ORDER, "not_audited") if counter[key]
        }

    def exploited(self) -> dict[Path, IntegrityVerdict]:
        """Trials with agent-attributed evidence of a forbidden crossing."""
        return {
            path: verdict
            for path, verdict in self.verdicts.items()
            if verdict is not None and verdict.exploited
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "exploited": [str(path) for path in self.exploited()],
            "trials": {
                str(path): (verdict.as_dict() if verdict is not None else None)
                for path, verdict in self.verdicts.items()
            },
        }


__all__ = [
    "IntegrityReport",
    "IntegrityVerdict",
    "VERDICT_ORDER",
    "read_verdict",
    "verdict_path",
]
