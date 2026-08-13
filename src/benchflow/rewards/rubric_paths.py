"""Ownership rules for verifier-time and detached-review rubric paths."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LLM_JUDGE_RUBRIC_PATH = "verifier/rubrics/verifier.toml"

_RESERVED_REVIEW_RUBRIC_SLOTS = frozenset(
    {
        "verifier/rubric.json",
        "tests/rubric.json",
    }
)


class ReservedReviewRubricError(ValueError):
    """Raised when a verifier tries to consume a detached-review contract."""


def reserved_review_rubric_slot(
    path: Path | str,
    *,
    task_dir: Path | str,
) -> str | None:
    """Return the reserved task-relative slot addressed by ``path``, if any.

    This check is deliberately lexical after normalizing ``..`` components.
    The ownership boundary applies to the task-package slot itself, including
    when the file in that slot is a symlink.
    """

    root = Path(os.path.abspath(task_dir))
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        return None
    return relative if relative in _RESERVED_REVIEW_RUBRIC_SLOTS else None


def validate_llm_judge_rubric_path(
    path: Path | str,
    *,
    task_dir: Path | str,
) -> None:
    """Reject detached-review rubric slots as verifier scoring inputs."""

    slot = reserved_review_rubric_slot(path, task_dir=task_dir)
    if slot is None:
        return
    raise ReservedReviewRubricError(
        f"{slot} is reserved for post-run `bench review`. Configure the "
        "LLM-judge verifier with a separate scoring rubric, such as "
        f"{DEFAULT_LLM_JUDGE_RUBRIC_PATH}."
    )
