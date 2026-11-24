"""Automatic checkpoints: retained sandbox snapshots after chosen prompts.

Opt-in with ``--checkpoints every-prompt|prompt:N,M`` on ``bench eval run``
and ``bench eval branch`` (``RolloutConfig(checkpoints=...)``). After each
chosen prompt the rollout's sandbox is snapshotted (agent credential files
are kept out, as for branch snapshots) and the snapshot is *kept*, so a later
``bench eval branch --from-checkpoint <trial> [--checkpoint prompt:N]`` can
fork from it. At most ``keep`` checkpoints are retained per trial; the oldest
is deleted when a newer one is taken. Everything is recorded in the trial's
``checkpoints.json``.

A checkpoint never fails the run: a sandbox without snapshot support is
recorded once as ``unsupported``, a failed snapshot as ``failed``, and the
run continues. The snapshot is taken while the agent is connected but idle
between prompts; processes the agent left running keep running.

Storage: each checkpoint is a full filesystem image held by the provider
(a Daytona snapshot, or a local Docker image). Daytona snapshots are
owner-scoped (``bf-snap2-<owner hash>-…``) and ``bench sandbox cleanup`` (and the
eval-start auto-reap) deletes an owner's snapshots once they are older than
the reap age; ``bench sandbox cleanup`` also removes Docker ``bf-snap-*``
images older than ``--max-age`` minutes (``--max-age 0`` removes all that no
container uses).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from benchflow.review.persistence import write_json_atomic

logger = logging.getLogger(__name__)

CHECKPOINTS_FILE = "checkpoints.json"
_SPEC_HELP = "--checkpoints takes every-prompt or prompt:N[,M...] (N >= 1)"


@dataclass(frozen=True)
class CheckpointPolicy:
    """Which prompts get a retained snapshot, and how many are kept."""

    every: bool = False
    after: frozenset[int] = frozenset()
    keep: int = 3

    def wants(self, prompt_number: int) -> bool:
        return self.every or prompt_number in self.after

    def to_record(self) -> dict[str, Any]:
        return {"every": self.every, "after": sorted(self.after), "keep": self.keep}


def parse_checkpoint_policy(spec: str, *, keep: int = 3) -> CheckpointPolicy:
    """Parse ``every-prompt`` or ``prompt:N[,M...]``."""
    if keep < 1:
        raise ValueError(f"--checkpoint-keep must be at least 1, got {keep}")
    spec = spec.strip()
    if spec == "every-prompt":
        return CheckpointPolicy(every=True, keep=keep)
    match = re.fullmatch(r"prompt:(\d+(?:,\d+)*)", spec)
    if match is None:
        raise ValueError(f"{_SPEC_HELP}; got {spec!r}")
    after = frozenset(int(part) for part in match.group(1).split(","))
    if 0 in after:
        raise ValueError(f"{_SPEC_HELP}; got {spec!r}")
    return CheckpointPolicy(after=after, keep=keep)


def _rows(rollout: Any) -> list[dict[str, Any]]:
    rows = getattr(rollout, "_checkpoint_rows", None)
    if rows is None:
        rows = []
        rollout._checkpoint_rows = rows
    return rows


def _write(rollout: Any, policy: CheckpointPolicy) -> None:
    write_json_atomic(
        rollout._rollout_dir / CHECKPOINTS_FILE,
        {
            "schema_version": 1,
            "kind": "benchflow-checkpoints",
            "rollout": getattr(rollout, "_rollout_name", None),
            "policy": policy.to_record(),
            "checkpoints": _rows(rollout),
        },
    )


async def after_prompt(rollout: Any, prompt_number: int) -> None:
    """Take the checkpoint ``prompt_number`` asks for, if any; never raises
    for a provider failure (cancellation still propagates)."""
    policy = getattr(getattr(rollout, "_config", None), "checkpoints", None)
    if policy is None or not policy.wants(prompt_number):
        return
    if getattr(rollout, "_rollout_dir", None) is None:
        return
    rows = _rows(rollout)
    sandbox = rollout._env
    cursor = getattr(rollout, "_cursor", None)
    row: dict[str, Any] = {
        "id": f"prompt:{prompt_number}",
        "after_prompt": prompt_number,
        "node_id": getattr(cursor, "id", None),
        # How much of the trajectory the checkpoint holds: a fork from it
        # exports the source conversation up to here as its prefix.
        "trajectory_events": len(getattr(rollout, "_trajectory", None) or []),
        # The transcript is in the snapshot (Claude Code keeps it on disk);
        # with this id a fork from the checkpoint can resume the conversation.
        "agent_session_id": getattr(
            getattr(rollout, "_session", None), "session_id", None
        ),
        "provider": None,
        "ref": None,
        "created_at": datetime.now(UTC).isoformat(),
        "seconds": None,
        "status": "failed",
        "error": None,
    }
    if not getattr(sandbox, "supports_snapshot", False):
        if any(r["status"] == "unsupported" for r in rows):
            return
        row.update(
            status="unsupported",
            error={"type": type(sandbox).__name__, "code": "no_snapshot_support"},
        )
        logger.warning(
            "Checkpoints requested, but %s cannot snapshot; the run continues "
            "without them",
            type(sandbox).__name__,
        )
        rows.append(row)
        _write(rollout, policy)
        return
    loop = asyncio.get_running_loop()
    started = loop.time()
    # The host's hard deadline doesn't count this time (seconds_spent).
    with clock(rollout, sandbox):
        try:
            image = await sandbox.snapshot()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "code": "snapshot_failed"}
            row["seconds"] = round(loop.time() - started, 3)
            # What became of a snapshot the provider went on to make (Daytona:
            # sandbox.daytona_snapshot_create): removed, none found, or left to
            # the sandbox's stop, which settles it in record_leftovers.
            leftover = getattr(exc, "benchflow_snapshot_leftover", None)
            if isinstance(leftover, dict):
                row["leftover"] = dict(leftover)
            logger.warning(
                "Checkpoint after prompt %d failed (%s: %s); the run continues",
                prompt_number,
                type(exc).__name__,
                exc,
            )
        else:
            row.update(
                **image.to_record(),
                status="kept",
                seconds=round(loop.time() - started, 3),
            )
            # The snapshot outlives this run: forget the in-memory credential
            # copy the sandbox keeps for restores of it.
            forget = getattr(
                getattr(sandbox, "_strategy", sandbox), "_snapshot_credentials", None
            )
            if isinstance(forget, dict):
                forget.pop(image.ref, None)
    rows.append(row)
    # Itemized in timing.json next to agent_execution and verifier.
    timing = getattr(rollout, "_timing", None)
    if isinstance(timing, dict):
        timing["checkpoint_snapshot"] = round(
            timing.get("checkpoint_snapshot", 0.0) + (loop.time() - started), 3
        )
    kept = [r for r in rows if r["status"] == "kept"]
    stale = kept[: max(0, len(kept) - policy.keep)]
    if stale:
        # Recorded before the older ones go: a cancel during their delete
        # must not leave the new snapshot unrecorded.
        _write(rollout, policy)
    for old in stale:
        await _delete(sandbox, old)
    _write(rollout, policy)


#: A checkpoint's most credit against the hard deadline when its sandbox
#: doesn't give its own worst case (``snapshot_budget_sec``). The deadline is
#: the backstop for an await wedged below its own bounds, so a stuck
#: checkpoint must not hold the trial for good.
_DEFAULT_CREDIT_CAP_SEC = 600.0


@contextlib.contextmanager
def clock(rollout: Any, sandbox: Any) -> Iterator[None]:
    """Count the time inside as ``rollout``'s checkpoint time
    (:func:`seconds_spent`), up to the sandbox's worst case for one snapshot."""
    budget = getattr(
        getattr(sandbox, "_strategy", sandbox), "snapshot_budget_sec", None
    )
    rollout._checkpoint_cap = (
        float(budget()) if callable(budget) else _DEFAULT_CREDIT_CAP_SEC
    )
    rollout._checkpoint_started = time.monotonic()
    try:
        yield
    finally:
        rollout._checkpoint_seconds = seconds_spent(rollout)
        rollout._checkpoint_started = None


def seconds_spent(rollout: Any) -> float:
    """Seconds ``rollout`` has spent taking checkpoints, the one in progress
    included, each counted up to its cap. A snapshot of a sandbox with an
    agent's install can take minutes (Daytona: up to
    BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC and the look after it), so the
    host's hard deadline is extended by this much."""
    spent = float(getattr(rollout, "_checkpoint_seconds", 0.0) or 0.0)
    started = getattr(rollout, "_checkpoint_started", None)
    if started is not None:
        cap = getattr(rollout, "_checkpoint_cap", _DEFAULT_CREDIT_CAP_SEC)
        spent += min(cap, max(0.0, time.monotonic() - started))
    return spent


async def _delete(sandbox: Any, row: dict[str, Any]) -> None:
    from benchflow.sandbox.protocol import SandboxImage

    try:
        deleted = await sandbox.delete_snapshot(SandboxImage.from_record(row))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        row.update(
            status="delete_failed",
            error={"type": type(exc).__name__, "code": "delete_failed"},
        )
        return
    row["status"] = "deleted" if deleted is not False else "delete_deferred"


def record_leftovers(trial_dir: Any, leftovers: list[dict[str, Any]]) -> None:
    """Write the sandbox stop's outcome for each leftover (by ``name``) into
    the rows of ``trial_dir/checkpoints.json`` that name it. Best effort, like
    the rest of the record: a failed write only logs."""
    import json

    by_name = {item.get("name"): item for item in leftovers}
    path = trial_dir / CHECKPOINTS_FILE
    if not path.is_file():
        # A branch snapshot's leftover: no checkpoint row names it.
        return
    try:
        document = json.loads(path.read_text())
        rows = document.get("checkpoints") if isinstance(document, dict) else None
        changed = False
        for row in rows if isinstance(rows, list) else []:
            leftover = row.get("leftover") if isinstance(row, dict) else None
            if isinstance(leftover, dict) and leftover.get("name") in by_name:
                row["leftover"] = dict(by_name[leftover["name"]])
                changed = True
        if changed:
            write_json_atomic(path, document)
    except (OSError, ValueError) as exc:
        logger.warning("Could not record snapshot leftovers in %s: %s", path, exc)


def load_checkpoints(trial_dir: Any) -> list[dict[str, Any]]:
    """The rows of ``trial_dir/checkpoints.json``, or [] when absent/invalid."""
    import json

    try:
        document = json.loads((trial_dir / CHECKPOINTS_FILE).read_text())
    except (OSError, ValueError):
        return []
    if (
        not isinstance(document, dict)
        or document.get("kind") != "benchflow-checkpoints"
    ):
        return []
    rows = document.get("checkpoints")
    return (
        [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    )
