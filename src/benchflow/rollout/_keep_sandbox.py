"""Keep a trial's sandbox for a person to look at (``--keep-sandbox``).

At teardown the rollout asks :func:`wants_keep` once. A kept sandbox is not
stopped, and the trial folder gets ``kept-sandbox.json``: what ``bench sandbox
attach`` opens a shell in and ``bench sandbox release`` deletes. The kept
sandbox is the graded state: the agent's processes were killed and the
workspace frozen before grading, so attaching resumes nothing of the agent's.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import classify_result
from benchflow.review.persistence import write_json_atomic

logger = logging.getLogger(__name__)

POLICIES = ("never", "on-failure", "always")
BACKENDS = ("docker", "daytona")
RECORD_NAME = "kept-sandbox.json"
# A kept Daytona sandbox still stops itself when idle (BenchFlow's auto-stop).
# Daytona deletes a stopped sandbox after this many minutes, so a forgotten
# one does not cost for ever; `bench sandbox attach` starts a stopped one.
DAYTONA_KEPT_DELETE_MINUTES = 24 * 60


def normalize_policy(value: str | None) -> str:
    """``never``, ``on-failure`` or ``always`` (``never`` when unset)."""
    policy = (value or "never").strip().lower()
    if policy not in POLICIES:
        raise ValueError(
            f"keep_sandbox must be never, on-failure or always, not {value!r}"
        )
    return policy


def check_backend(policy: str, environment: str) -> None:
    """Refuse a keep policy on a backend that cannot keep and attach."""
    if policy != "never" and environment not in BACKENDS:
        raise ValueError(
            f"keep_sandbox={policy!r} needs the docker or daytona sandbox, "
            f"not {environment!r}"
        )


def wants_keep(
    policy: str,
    *,
    rewards: dict[str, Any] | None,
    error: str | None,
    verifier_error: str | None,
) -> bool:
    """Whether teardown keeps the sandbox. ``on-failure`` keeps every trial
    that did not pass, by the summary's own rule (an error, a verifier error,
    no reward, or a reward that is not a pass)."""
    if policy == "always":
        return True
    if policy != "on-failure":
        return False
    reward = (rewards or {}).get("reward")
    if not isinstance(reward, int | float) or isinstance(reward, bool):
        reward = None
    outcome = classify_result(
        reward=None if reward is None else float(reward),
        error=error,
        verifier_error=verifier_error,
    )
    return outcome != "passed"


async def keep(
    env: Any,
    rollout_dir: Path,
    *,
    environment: str,
    task_copy: Path | None,
) -> dict[str, Any] | None:
    """Record ``env`` as kept in ``rollout_dir``, or None when this sandbox
    cannot be kept (the caller then stops it as usual)."""
    record: dict[str, Any] = {"version": 1, "state": "kept"}
    if environment == "docker":
        project = getattr(env, "compose_project_name", None)
        if not project:
            return None
        from benchflow.sandbox._docker_sweep import keep_project

        # Before anything else: no later sweep may take it for a leftover.
        keep_project(str(project))
        record.update(
            backend="docker",
            project=str(project),
            service="main",
            docker_host=os.environ.get("DOCKER_HOST") or None,
            docker_context=os.environ.get("DOCKER_CONTEXT") or None,
        )
    elif environment == "daytona":
        sandbox = getattr(env, "_sandbox", None)
        sandbox_id = getattr(sandbox, "id", None)
        if sandbox is None or not sandbox_id:
            return None
        from benchflow.sandbox.daytona_reaper import KEPT_LABEL

        try:
            await sandbox.set_auto_delete_interval(DAYTONA_KEPT_DELETE_MINUTES)
            # set_labels replaces them all: the ownership labels stay.
            labels = dict(getattr(sandbox, "labels", None) or {})
            await sandbox.set_labels({**labels, KEPT_LABEL: "1"})
        except Exception as exc:
            logger.warning(
                "Could not keep Daytona sandbox %s (deleting it as usual): %s",
                sandbox_id,
                exc,
            )
            return None
        record.update(
            backend="daytona",
            sandbox_id=str(sandbox_id),
            compose=bool(getattr(env, "_uses_compose", False)),
            deleted_after_stopped_minutes=DAYTONA_KEPT_DELETE_MINUTES,
        )
    else:
        return None
    record.update(
        kept_at=datetime.now(UTC).isoformat(),
        task_copy=str(task_copy) if task_copy else None,
        attach=f"bench sandbox attach {rollout_dir}",
        release=f"bench sandbox release {rollout_dir}",
    )
    write_json_atomic(rollout_dir / RECORD_NAME, record)
    return record
