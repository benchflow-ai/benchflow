"""Small host-to-sandbox file transfer primitives."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


async def upload_private_text(
    sandbox: Any,
    text: str,
    target_path: str,
    *,
    suffix: str,
) -> None:
    """Upload text with an explicit owner-only mode and remove the host temp."""

    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as temporary:
        temporary.write(text)
        source_path = Path(temporary.name)
    try:
        await sandbox.upload_file(source_path, target_path, mode="600")
    finally:
        source_path.unlink(missing_ok=True)
