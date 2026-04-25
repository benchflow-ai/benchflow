"""Sandbox snapshot/restore — filesystem-level checkpointing.

Provides snapshot(name) -> ref and restore(ref) for any environment that
supports env.exec().  Works on both Docker and Daytona backends.

Implementation: tar the workspace directory into /tmp/.benchflow_snapshots/.
In-place restore by clearing and untarring.  Backend-agnostic — no Docker
daemon or Daytona snapshot API required.

For 0.3+, Daytona's _experimental_create_snapshot can be swapped in as an
optimization for branching rollouts (new sandbox from snapshot), but the
filesystem approach covers the rewind use case and is provable now.
"""

import logging
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

_SNAP_DIR = "/tmp/.benchflow_snapshots"


async def snapshot(env, name: str, workspace: str = "/app") -> str:
    """Create a named snapshot of the workspace.

    Returns a reference string suitable for restore() and for recording
    in trial metadata / rewards.jsonl.
    """
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(
            f"Snapshot name must be alphanumeric/dash/underscore, got: {name!r}"
        )
    await env.exec(f"mkdir -p {_SNAP_DIR}")
    snap_path = f"{_SNAP_DIR}/{name}.tar.gz"
    result = await env.exec(
        f"tar czf {snap_path} -C {workspace} .",
        timeout_sec=120,
    )
    if result.return_code != 0:
        raise RuntimeError(f"snapshot failed: {(result.stderr or '')}")
    ref = f"fs:{name}:{snap_path}"
    logger.info(f"Snapshot created: {ref}")
    return ref


async def restore(env, ref: str, workspace: str = "/app") -> None:
    """Restore workspace to a named snapshot.

    ref: the string returned by snapshot() — format is "fs:{name}:{path}".
    """
    parts = ref.split(":", 2)
    if len(parts) != 3 or parts[0] != "fs":
        raise ValueError(f"invalid snapshot ref: {ref}")
    snap_path = parts[2]
    check = await env.exec(f"test -f {snap_path} && echo ok || echo missing")
    if "missing" in (check.stdout or ""):
        raise FileNotFoundError(f"snapshot not found: {snap_path}")
    result = await env.exec(
        f"rm -rf {workspace}/* {workspace}/.[!.]* 2>/dev/null; "
        f"tar xzf {snap_path} -C {workspace}",
        timeout_sec=120,
    )
    if result.return_code != 0:
        raise RuntimeError(f"restore failed: {(result.stderr or '')}")
    logger.info(f"Snapshot restored: {ref}")


async def list_snapshots(env) -> list[str]:
    """List available snapshot names."""
    result = await env.exec(f"ls {_SNAP_DIR}/*.tar.gz 2>/dev/null || true")
    if not (result.stdout or "").strip():
        return []
    return [
        PurePosixPath(line.strip()).stem.removesuffix(".tar")
        for line in (result.stdout or "").strip().splitlines()
    ]
