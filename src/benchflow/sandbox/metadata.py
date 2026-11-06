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
    path = rollout_dir / "sandbox.json"
    info: dict[str, Any] = {
        "sandbox_id": sandbox_id,
        "provider": provider,
        "created_at": _first_created_at(path, sandbox_id) or str(datetime.now()),
    }
    # Daytona: the cached install layers the image carried, Daytona's build
    # record and runner, and the disk asked for them (sandbox.daytona_layers).
    layers = getattr(env, "_install_layers_record", None)
    if isinstance(layers, dict):
        info["install_layers"] = layers
    try:
        path.write_text(json.dumps(info, indent=2))
    except Exception as e:  # best-effort: never fail start() over an audit file
        logger.warning(f"Failed to persist sandbox.json for {sandbox_id}: {e}")
        return
    logger.info(f"Sandbox {sandbox_id} ({provider})")


def _first_created_at(path: Path, sandbox_id: str) -> str | None:
    """The ``created_at`` an earlier write for this same sandbox recorded.

    sandbox.json is written again after creation (Daytona notes which install
    layers a trial used); ``created_at`` stays the time the sandbox was first
    recorded. A new sandbox id (a retried create) gets a new time.
    """
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(previous, dict) or previous.get("sandbox_id") != sandbox_id:
        return None
    created = previous.get("created_at")
    return created if isinstance(created, str) and created else None
