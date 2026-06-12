"""Types, constants, and dataclasses for acceptance-live validation.

This module owns the small model block shared across the acceptance-live
parsing, orchestration, and report planes. It carries no behavior beyond the
frozen dataclasses and literal definitions that describe a live acceptance
spec; the façade ``benchflow.task.acceptance_live`` re-exports every name here.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LiveAcceptanceCaseType = Literal[
    "verifier",
    "oracle",
    "no-op",
    "known-bad",
    "partial",
    "reference",
]
LiveAcceptanceCaseSource = Literal["declared", "calibration-report"]
_CASE_TYPES: set[str] = {
    "verifier",
    "oracle",
    "no-op",
    "known-bad",
    "partial",
    "reference",
}
_WORKSPACE_SOURCE_CURRENT_WORKTREE = "current-worktree"
_DEFAULT_RERUNS = 1
_MAX_RERUNS = 20
_LEADERBOARD_CALIBRATION_TYPES = frozenset(
    {"no-op", "known-bad", "partial", "reference"}
)
_STAGE_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    "__pycache__",
    "htmlcov",
    "jobs",
    "node_modules",
    "*.pyc",
)
_DEP_INSTALL_FLAKE_HINT = (
    "first failed run indicates verifier dependency install failed "
    "(see verifier/test-stdout.txt in the run artifacts)"
)


@dataclass(frozen=True)
class LiveAcceptanceWorkspace:
    source: Literal["current-worktree"]
    target: str


@dataclass(frozen=True)
class LiveAcceptanceExpectation:
    reward_min: float | None = None
    reward_max: float | None = None
    reward_range: tuple[float, float] | None = None
    reward_equals: float | None = None
    flake_rate_max: float | None = None


@dataclass(frozen=True)
class LiveAcceptanceCase:
    name: str
    case_type: LiveAcceptanceCaseType
    command: str | None
    reruns: int
    expect: LiveAcceptanceExpectation
    source: LiveAcceptanceCaseSource = "declared"


@dataclass(frozen=True)
class LiveAcceptanceRunResult:
    reward: float | None
    error: str | None
    verifier_error_category: str | None = None
    diagnostic_code: str | None = None
    artifact_hint: str | None = None


@dataclass(frozen=True)
class LiveAcceptanceLeaderboard:
    required: bool = False
    max_flake_rate: float = 0.0


@dataclass(frozen=True)
class LiveAcceptanceSpec:
    workspace: LiveAcceptanceWorkspace
    cases: tuple[LiveAcceptanceCase, ...]
    report_path: Path | None
    leaderboard: LiveAcceptanceLeaderboard
