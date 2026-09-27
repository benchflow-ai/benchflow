"""Same-daemon image leases for verifier recovery; never container commits."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_RELEASE_TIMEOUT_SEC = 60


class DockerRecoveryBaseline(BaseModel):
    """Credential-free identity for a pre-agent resolved image and runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    lease_tag: str = Field(pattern=r"^benchflow-recovery-lease:[0-9a-f]{32}$")
    effective_allow_internet: bool
    daemon_id: str = Field(min_length=1, max_length=256)
    task_digest: str = Field(min_length=1)
    effective_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    sandbox_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def config_digest(config: Any) -> str:
    """Hash effective configuration without persisting environment values."""
    data = config.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def capture_baseline(
    env: Any, task_digest: str, effective_config_digest: str
) -> DockerRecoveryBaseline:
    if env._uses_compose or env._mounts_json or env._pre_compose_hook_path.exists():
        raise ValueError(
            "Custom compose, mounts, or pre-compose hooks have no recovery baseline contract"
        )
    container_id = await env._main_container_id()
    if not container_id:
        raise ValueError("Cannot resolve original main container for recovery")
    inspected = await env._inspect_container(container_id)
    daemon = await env._docker_cli(["info", "--format", "{{.ID}}"])
    baseline = DockerRecoveryBaseline(
        image_id=inspected.get("Image"),
        lease_tag="benchflow-recovery-lease:" + uuid.uuid4().hex,
        daemon_id=daemon.stdout.strip(),
        effective_allow_internet=env.task_env_config.allow_internet,
        task_digest=task_digest,
        effective_config_digest=effective_config_digest,
        sandbox_config_digest=config_digest(env.task_env_config),
    )
    # Tag the resolved image, not its mutable original name or container state.
    await env._docker_cli(["image", "tag", baseline.image_id, baseline.lease_tag])
    try:
        await validate_baseline(env, baseline)
    except Exception:
        await release_baseline(baseline)
        raise
    return baseline


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(
            proc.communicate(), timeout=_RELEASE_TIMEOUT_SEC
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return 1, f"docker {' '.join(args)} timed out"
    return proc.returncode or 0, output.decode(errors="replace").strip()


async def release_baseline(baseline: DockerRecoveryBaseline) -> bool:
    """Remove this lease's tag; ``True`` once the lease no longer exists.

    ``docker image rm <tag>`` without ``--force`` only untags an image that
    other tags or digests still reference, deletes it only when the lease was
    its last reference, and refuses while a container uses it. A lease that is
    already gone counts as released, so every teardown path may call this.
    """
    try:
        code, output = await _docker("image", "rm", baseline.lease_tag)
    except OSError as exc:
        code, output = 1, str(exc)
    if code == 0 or "no such image" in output.lower():
        return True
    logger.warning(
        "Could not release verifier recovery image lease %s: %s",
        baseline.lease_tag,
        output,
    )
    return False


async def validate_baseline(
    env: Any, baseline: DockerRecoveryBaseline, *, sandbox_config: Any = None
) -> None:
    effective = env.task_env_config if sandbox_config is None else sandbox_config
    if config_digest(effective) != baseline.sandbox_config_digest:
        raise ValueError("Recovery sandbox configuration differs from original runtime")
    daemon = await env._docker_cli(["info", "--format", "{{.ID}}"])
    if daemon.stdout.strip() != baseline.daemon_id:
        raise ValueError("Recovery image lease belongs to a different Docker daemon")
    result = await env._docker_cli(
        ["image", "inspect", "--format", "{{.Id}}", baseline.lease_tag]
    )
    if result.stdout.strip() != baseline.image_id:
        raise ValueError("Recovery image lease is missing or changed")


def validate_task_identity(
    baseline: DockerRecoveryBaseline, task_digest: str, task_config: Any
) -> None:
    """Compare pristine effective inputs, never the hardening-mutated live task."""
    if (
        baseline.task_digest != task_digest
        or baseline.effective_config_digest != config_digest(task_config)
    ):
        raise ValueError("Original Docker recovery task/runtime identity changed")
