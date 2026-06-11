"""Agent-judge input reading, scoring, and rollout-path resolution.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. Reads and
truncates declared judge inputs, scores judge verdicts, and resolves declared
input paths against the local rollout evidence directory.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.rewards.validation import is_valid_reward_number
from benchflow.task.verifier_errors import AgentJudgeInputError

_AGENT_JUDGE_INPUT_CHAR_LIMIT = 50_000


def _agent_judge_score(verdict: dict[str, Any]) -> float:
    raw_score = verdict.get("score", verdict.get("reward"))
    if raw_score is not None:
        score = float(raw_score)
        if not is_valid_reward_number(score):
            raise ValueError("agent-judge score must be between 0.0 and 1.0")
        return score

    raw_verdict = verdict.get("verdict")
    if isinstance(raw_verdict, str):
        normalized = raw_verdict.strip().lower()
        if normalized in {"pass", "passed", "yes", "true"}:
            return 1.0
        if normalized in {"fail", "failed", "no", "false"}:
            return 0.0

    raise ValueError("agent-judge verdict must include score/reward or pass/fail")


def _read_agent_judge_input(path: Path) -> tuple[str, bool]:
    if not path.is_file():
        raise AgentJudgeInputError(
            f"agent-judge input {path} must resolve to a regular file"
        )
    content = path.read_text(errors="replace")
    if len(content) <= _AGENT_JUDGE_INPUT_CHAR_LIMIT:
        return content, False
    truncated = (
        content[:_AGENT_JUDGE_INPUT_CHAR_LIMIT]
        + f"\n[TRUNCATED: {len(content)} chars total]"
    )
    return truncated, True


def _local_rollout_input_path(
    declared_path: str,
    *,
    rollout_dir: Path,
) -> Path | None:
    path = PurePosixPath(declared_path)
    if path.is_absolute():
        logs_prefix = PurePosixPath("/logs")
        try:
            relative = path.relative_to(logs_prefix)
        except ValueError:
            return None
    else:
        relative = path

    if not relative.parts or ".." in relative.parts:
        raise AgentJudgeInputError(
            f"agent-judge input {declared_path!r} must stay inside rollout evidence"
        )
    return rollout_dir / Path(*relative.parts)


def _safe_input_filename(path: PurePosixPath) -> str:
    name = "__".join(part for part in path.parts if part != "/")
    return name or "input"
