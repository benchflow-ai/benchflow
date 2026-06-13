"""Sandbox metadata artifact helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def persist_sandbox_info(env: Any, rollout_dir: Path | None) -> None:
    """Write sandbox.json with provider-side sandbox ID immediately after creation.

    The write is best-effort: failures to write this audit file must not abort
    an otherwise-healthy sandbox startup.
    """
    sandbox_id = getattr(env, "sandbox_id", None)
    if not isinstance(sandbox_id, str) or not rollout_dir:
        return
    provider = type(env).__name__
    info = {
        "sandbox_id": sandbox_id,
        "provider": provider,
        "created_at": str(datetime.now()),
    }
    try:
        (rollout_dir / "sandbox.json").write_text(json.dumps(info, indent=2))
    except Exception as e:  # best-effort: never fail start() over an audit file
        logger.warning(f"Failed to persist sandbox.json for {sandbox_id}: {e}")
        return
    logger.info(f"Sandbox {sandbox_id} ({provider})")


def record_snapshot_leak(
    snapshot_names: list[str], provider: str, rollout_dir: Path | None
) -> None:
    """Append leaked cloud-snapshot names to ``snapshot_leaks.json`` for cleanup.

    A snapshot benchflow created but could not delete on teardown is a real
    cloud cost/quota leak. We can't reclaim it here, so we record the names (by
    appending to any existing report) so an operator or post-mortem reaper can.
    Best-effort: a failed write must never break teardown.
    """
    if not snapshot_names or not rollout_dir:
        return
    path = rollout_dir / "snapshot_leaks.json"
    existing: list[dict[str, Any]] = []
    try:
        if path.is_file():
            loaded = json.loads(path.read_text())
            if isinstance(loaded, list):
                existing = loaded
    except Exception:  # corrupt/old report — start fresh rather than fail
        existing = []
    existing.append(
        {
            "provider": provider,
            "snapshot_names": list(snapshot_names),
            "recorded_at": str(datetime.now()),
        }
    )
    try:
        path.write_text(json.dumps(existing, indent=2))
    except Exception as e:  # best-effort: never fail teardown over an audit file
        logger.warning(f"Failed to record snapshot leak {snapshot_names}: {e}")
        return
    logger.warning(
        "Recorded %d leaked %s snapshot(s) to %s for post-mortem cleanup: %s",
        len(snapshot_names),
        provider,
        path,
        ", ".join(snapshot_names),
    )
