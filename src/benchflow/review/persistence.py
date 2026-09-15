"""Commit terminal scoring without rewriting the solver's execution record."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.review.outcome import ScoringResult


@contextmanager
def scoring_lock(rollout_dir: Path) -> Iterator[None]:
    """Reject simultaneous scoring of one trial without blocking the event loop.

    The advisory lock is released by the OS on process exit, so an interrupted
    reviewer never leaves a stale lock requiring manual cleanup.
    """
    with (rollout_dir / ".scoring.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Scoring is already running for {rollout_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_json_atomic(path: Path, data: Any) -> None:
    """Replace one JSON document after its contents are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _execution_seconds(result: dict[str, Any]) -> float | None:
    """Read recorded runtime, without charging gaps between scoring attempts."""
    value = (result.get("timing") or {}).get("total")
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return float(value)
    started, finished = result.get("started_at"), result.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    duration = (
        datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    ).total_seconds()
    return duration if duration >= 0 else None


def _scoring_timing(
    rollout_dir: Path, source: dict[str, Any], scoring: ScoringResult
) -> dict[str, Any]:
    """Report solver plus selected reviewer execution, retaining unknown time.

    A retry can happen days after the solver. Parent wall timestamps describe
    when the final score was available; ``timing`` describes actual execution
    of the solver and selected reviewer. Prior attempts retain their own timing.
    """
    reviewer_seconds = None
    if scoring.reviewer_run:
        reviewer_dir = (rollout_dir / scoring.reviewer_run).resolve()
        if not reviewer_dir.is_relative_to(rollout_dir.resolve()):
            raise ValueError("Reviewer run reference escapes the parent rollout")
        reviewer_result = reviewer_dir / "result.json"
        if reviewer_result.is_file():
            reviewer_seconds = _execution_seconds(
                json.loads(reviewer_result.read_text())
            )
    solver_seconds = _execution_seconds(source)
    timing = dict(source.get("timing") or {})
    timing["review"] = (
        round(reviewer_seconds, 1) if reviewer_seconds is not None else None
    )
    timing["total"] = (
        round(solver_seconds + reviewer_seconds, 1)
        if solver_seconds is not None and reviewer_seconds is not None
        else None
    )
    return timing


def _snapshot_scoring_artifacts(
    rollout_dir: Path, result: dict[str, Any], scoring: ScoringResult
) -> None:
    """Publish immutable exports belonging to the committed scoring revision.

    Root-level files remain compatibility views. Consumers requiring a coherent
    revision follow ``scoring.revision`` and its same-stem artifact directory;
    an earlier revision remains intact while those compatibility views refresh.
    """
    if scoring.revision is None:
        return
    revision = rollout_dir / scoring.revision
    if not revision.is_file():
        raise ValueError("Cannot commit scoring without its immutable review report")
    destination = revision.with_suffix("")
    if destination.exists():
        raise FileExistsError(
            f"Scoring revision artifacts already exist: {destination}"
        )
    with tempfile.TemporaryDirectory(
        prefix=".scoring-", dir=revision.parent
    ) as temporary:
        stage = Path(temporary)
        for name in ("rewards.jsonl", "results.jsonl", "timing.json"):
            source = rollout_dir / name
            if source.is_file():
                shutil.copy2(source, stage / name)
        if (rollout_dir / "trainer").is_dir():
            shutil.copytree(rollout_dir / "trainer", stage / "trainer")
        # Avoid result.json: uncommitted revisions are not additional trials.
        write_json_atomic(stage / "parent.json", result)
        os.replace(stage, destination)


def commit_scoring_result(rollout_dir: Path, scoring: ScoringResult) -> dict[str, Any]:
    """Regenerate score exports, then atomically publish the parent result.

    solver.json is immutable across review retries. The parent result is the
    commit marker, linked to immutable revision exports. Root compatibility
    files are not a multi-file transaction and can be rebuilt after a crash.
    This operation is shared by initial scoring and resume.
    """
    from benchflow.rollout._results import _write_rewards_jsonl, _write_trainer_artifact
    from benchflow.trajectories.results import write_rollout_results_jsonl

    source = json.loads((rollout_dir / "solver.json").read_text())
    result = dict(source)
    result["scoring"] = scoring.to_dict()
    result["rewards"] = scoring.numeric_rewards()
    result["verifier_error"] = scoring.error or source.get("verifier_error")
    if scoring.error:
        result["verifier_error_category"] = "infra_failure"
    finished = datetime.now()
    result["solver_started_at"] = source.get("started_at")
    result["solver_finished_at"] = source.get("finished_at")
    result["scoring_finished_at"] = str(finished)
    result["finished_at"] = str(finished)
    timing = _scoring_timing(rollout_dir, source, scoring)
    result["timing"] = timing
    trajectory = [
        json.loads(line)
        for line in (rollout_dir / "trajectory" / "acp_trajectory.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    prompts = json.loads((rollout_dir / "prompts.json").read_text())
    agent_result = source.get("agent_result") or {}
    _write_rewards_jsonl(rollout_dir, result["rewards"], finished)
    if result["rewards"] is None:
        (rollout_dir / "rewards.jsonl").unlink(missing_ok=True)
    _write_trainer_artifact(
        rollout_dir,
        task_name=source["task_name"],
        rollout_name=source["rollout_name"],
        agent_name=source.get("agent_name") or source["agent"],
        prompts=prompts,
        trajectory=trajectory,
        rewards=result["rewards"],
        model=source.get("model"),
        verifier_error=result["verifier_error"],
        total_prompt_tokens=agent_result.get("n_input_tokens"),
        total_completion_tokens=agent_result.get("n_output_tokens"),
        total_cached_tokens=agent_result.get("n_cache_read_tokens"),
        total_cost_usd=agent_result.get("cost_usd"),
        strict=True,
    )
    write_rollout_results_jsonl(
        rollout_dir,
        task_name=source["task_name"],
        rollout_name=source["rollout_name"],
        agent=source["agent"],
        agent_name=source.get("agent_name", ""),
        model=source.get("model"),
        n_tool_calls=source.get("n_tool_calls", 0),
        prompts=prompts,
        trajectory=trajectory,
        partial_trajectory=source.get("partial_trajectory", False),
        rewards=result["rewards"],
        error=source.get("error"),
        verifier_error=result["verifier_error"],
        export_error=source.get("export_error"),
        timing=timing,
        agent_result=agent_result,
        scoring=scoring,
        purpose=source.get("purpose", "task"),
        parent_rollout=source.get("parent_rollout"),
    )
    write_json_atomic(rollout_dir / "timing.json", timing)
    _snapshot_scoring_artifacts(rollout_dir, result, scoring)
    write_json_atomic(rollout_dir / "result.json", result)
    return result
