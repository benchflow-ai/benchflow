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
owner-named (``bf-snap-<owner>-…``) and ``bench sandbox cleanup`` (and the
eval-start auto-reap) deletes an owner's snapshots once they are older than
the reap age; ``bench sandbox cleanup`` also removes Docker ``bf-snap-*``
images older than ``--max-age`` minutes (``--max-age 0`` removes all that no
container uses).
"""

from __future__ import annotations

import asyncio
import logging
import re
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
    try:
        image = await sandbox.snapshot()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        row["error"] = {"type": type(exc).__name__, "code": "snapshot_failed"}
        logger.warning(
            "Checkpoint after prompt %d failed (%s: %s); the run continues",
            prompt_number,
            type(exc).__name__,
            exc,
        )
    else:
        row.update(
            provider=image.provider,
            ref=image.ref,
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
    for old in kept[: max(0, len(kept) - policy.keep)]:
        await _delete(sandbox, old)
    _write(rollout, policy)


async def _delete(sandbox: Any, row: dict[str, Any]) -> None:
    from benchflow.sandbox.protocol import SandboxImage

    try:
        deleted = await sandbox.delete_snapshot(
            SandboxImage(provider=row["provider"], ref=row["ref"])
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        row.update(
            status="delete_failed",
            error={"type": type(exc).__name__, "code": "delete_failed"},
        )
        return
    row["status"] = "deleted" if deleted is not False else "delete_deferred"


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
