"""Sandbox artifact-custody proof for provider-wire capture."""

from __future__ import annotations

import contextlib
import logging
import secrets
import shlex
from importlib.resources import files
from typing import Any

from benchflow.sandbox.files import upload_private_text
from benchflow.sandbox.lockdown import build_priv_drop_cmd

logger = logging.getLogger(__name__)


def provider_capture_custody_probe_source() -> str:
    """Read the packaged custody prober deployed beside sandbox runtimes."""

    return (
        files("benchflow.providers")
        .joinpath("resources", "provider_capture_custody.sh")
        .read_text(encoding="utf-8")
    )


async def provider_capture_has_verified_custody(
    *,
    sandbox_user: str | None,
    sandbox: Any | None,
    runtime_dir: str | None,
) -> bool:
    """Prove the agent cannot read or mutate provider capture files."""

    if sandbox is None or sandbox_user in {None, "", "root", "0"} or not runtime_dir:
        return False
    try:
        result = await sandbox.exec(
            f"id -u -- {shlex.quote(sandbox_user)}",
            user="root",
            timeout_sec=10,
        )
    except Exception as exc:
        logger.warning("Provider capture custody UID check failed: %s", exc)
        return False
    if result.return_code != 0:
        logger.warning("Provider capture custody UID check returned non-zero")
        return False
    try:
        effective_uid = int(result.stdout.strip())
    except (AttributeError, ValueError):
        logger.warning("Provider capture custody UID check returned invalid output")
        return False
    if effective_uid == 0:
        return False

    probe_path = f"/tmp/benchflow-capture-custody-{secrets.token_hex(8)}.sh"
    quoted_probe = shlex.quote(probe_path)
    quoted_runtime = shlex.quote(runtime_dir)
    try:
        await upload_private_text(
            sandbox,
            provider_capture_custody_probe_source(),
            probe_path,
            suffix=".sh",
        )
        prepared = await sandbox.exec(
            f"chown 0:0 {quoted_probe} && chmod 755 {quoted_probe}",
            user="root",
            timeout_sec=10,
        )
        if prepared.return_code != 0:
            logger.warning("Provider capture custody probe upload hardening failed")
            return False

        secured = await sandbox.exec(
            f"{quoted_probe} harden {quoted_runtime}",
            user="root",
            timeout_sec=10,
        )
        if secured.return_code != 0:
            logger.warning("Provider capture custody artifact hardening failed")
            return False
        access = await sandbox.exec(
            build_priv_drop_cmd(
                f"{quoted_probe} probe {quoted_runtime}",
                sandbox_user,
            ),
            user="root",
            timeout_sec=10,
        )
        if access.return_code != 1:
            logger.warning(
                "Provider capture custody probe found agent-accessible root data"
            )
            return False
        verified = await sandbox.exec(
            f"{quoted_probe} verify {quoted_runtime}",
            user="root",
            timeout_sec=10,
        )
        if verified.return_code != 0:
            logger.warning("Provider capture custody probe integrity check failed")
            return False
        return True
    except Exception as exc:
        logger.warning("Provider capture custody artifact probe failed: %s", exc)
        return False
    finally:
        with contextlib.suppress(Exception):
            await sandbox.exec(
                f"rm -f {quoted_probe}",
                user="root",
                timeout_sec=10,
            )
