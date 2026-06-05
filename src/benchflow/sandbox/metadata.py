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
