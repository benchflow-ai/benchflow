"""Workspace snapshot/restore — filesystem-level helper, NOT the Sandbox primitive.

Provides ``workspace_snapshot(name) -> ref`` and ``workspace_restore(ref)``
for any environment that supports ``env.exec()``. Works on Docker and
Daytona sandboxes.

**Scope** (#384): this is a *workspace-only* tar/untar helper, not the
container-level snapshot/restore the Branch lifecycle calls. It captures
files under a single directory (default ``/app``); it does **not** snapshot
the container filesystem, mounted volumes, process state, sibling compose
services, or provider images.

Container-level snapshot/restore lives on the Sandbox contract itself —
see :meth:`benchflow.sandbox.protocol.Sandbox.snapshot` and the
``supports_snapshot`` capability gate. ``Rollout.branch`` composes those
with Environment-state snapshots for full Branch semantics.

The historical ``snapshot``/``restore``/``list_snapshots`` names are kept
as backward-compatible aliases.
"""

import logging
import re as _re
import shlex
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

_SNAP_DIR = "/tmp/.benchflow_snapshots"


async def workspace_snapshot(env, name: str, workspace: str = "/app") -> str:
    """Create a named *workspace-only* snapshot — tar of ``workspace``.

    Returns a reference string suitable for ``workspace_restore()`` and for
    recording in trial metadata / rewards.jsonl. This does **not** capture
    container state — use :meth:`Sandbox.snapshot` for that (#384).
    """
    if not _re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(
            f"Snapshot name must be alphanumeric/dash/underscore, got: {name!r}"
        )
    await env.exec(f"mkdir -p {_SNAP_DIR}")
    snap_path = f"{_SNAP_DIR}/{name}.tar.gz"
    result = await env.exec(
        f"tar czf {shlex.quote(snap_path)} -C {shlex.quote(workspace)} .",
        timeout_sec=120,
    )
    if result.return_code != 0:
        raise RuntimeError(f"snapshot failed: {(result.stderr or '')}")
    ref = f"fs:{name}:{snap_path}"
    logger.info(f"Snapshot created: {ref}")
    return ref


async def workspace_restore(env, ref: str, workspace: str = "/app") -> None:
    """Restore ``workspace`` to a named workspace snapshot.

    ref: the string returned by ``workspace_snapshot()`` — format is
    ``"fs:{name}:{path}"``. Does not restore container state — use
    :meth:`Sandbox.restore` for that (#384).
    """
    parts = ref.split(":", 2)
    if len(parts) != 3 or parts[0] != "fs":
        raise ValueError(f"invalid snapshot ref: {ref}")
    snap_path = parts[2]
    # Validate snap_path: must be under _SNAP_DIR and a .tar.gz file
    if not snap_path.startswith(_SNAP_DIR + "/") or not snap_path.endswith(".tar.gz"):
        raise ValueError(f"invalid snapshot ref: path must be under {_SNAP_DIR}")
    if ".." in snap_path.split("/"):
        raise ValueError("invalid snapshot ref: path traversal not allowed")
    check = await env.exec(
        f"test -f {shlex.quote(snap_path)} && echo ok || echo missing"
    )
    if "missing" in (check.stdout or ""):
        raise FileNotFoundError(f"snapshot not found: {snap_path}")
    result = await env.exec(
        f"rm -rf {shlex.quote(workspace)}/* {shlex.quote(workspace)}/.[!.]* 2>/dev/null; "
        f"tar xzf {shlex.quote(snap_path)} -C {shlex.quote(workspace)}",
        timeout_sec=120,
    )
    if result.return_code != 0:
        raise RuntimeError(f"restore failed: {(result.stderr or '')}")
    logger.info(f"Snapshot restored: {ref}")


async def list_workspace_snapshots(env) -> list[str]:
    """List available *workspace* snapshot names."""
    result = await env.exec(f"ls {_SNAP_DIR}/*.tar.gz 2>/dev/null || true")
    if not (result.stdout or "").strip():
        return []
    return [
        PurePosixPath(line.strip()).stem.removesuffix(".tar")
        for line in (result.stdout or "").strip().splitlines()
    ]


# Backward-compatibility aliases
#
# Pre-#384 these were ``snapshot`` / ``restore`` / ``list_snapshots`` and
# exported as ``bf.snapshot`` etc. The names looked like the Sandbox lifecycle
# primitive but only ever covered a single workspace directory. The renamed
# functions above make the scope explicit; the old names stay as aliases so
# external callers (proof scripts, downstream tasks) keep working.
snapshot = workspace_snapshot
restore = workspace_restore
list_snapshots = list_workspace_snapshots
