"""Keep agent credential files out of container snapshots.

A container snapshot (``docker commit``, a Daytona provider snapshot) stores
the whole filesystem at rest, including the credential files an agent install
writes: ``~/.codex/auth.json`` for codex-acp, ``~/.claude/.credentials.json``
for a host subscription login, provider files such as Vertex ADC. Branch
children still need those files, because each child starts a fresh agent
session that reads them.

So capture runs on a scrubbed filesystem. :func:`scrub_credentials` reads the
files into host memory (never as a command-line argument: Docker streams them
over ``docker exec`` stdout/stdin, Daytona uses its file API because the
Daytona daemon keeps session command output on disk inside the sandbox),
removes them, and checks that none remain; the caller captures, then
:func:`put_back_credentials` writes them back with their owner and mode, to
the live sandbox and to every sandbox restored from that snapshot. A
credential path that is a symbolic link is refused rather than followed.

The inventory is :data:`benchflow.agents.credentials.CREDENTIAL_EVIDENCE_PATHS`
under ``/root`` and every ``/home/<user>``. Credentials a task places
elsewhere are not covered.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from benchflow.sandbox import _credential_writeback
from benchflow.sandbox.protocol import ExecResult

CREDENTIAL_HOMES = ("/root", "/home/*")

# Source of the in-sandbox, symlink-safe write-back, shipped with ``-I -c``.
_WRITEBACK_SOURCE = Path(_credential_writeback.__file__).read_text()


class CredentialScrubError(RuntimeError):
    """Credential files could not be kept out of a snapshot; nothing was captured."""


@dataclass(frozen=True)
class StashedCredential:
    """One credential file held in host memory while a snapshot is taken."""

    path: str
    uid: str
    gid: str
    mode: str
    content: bytes = field(repr=False)


class CredentialOps(Protocol):
    """Root-level access to one live sandbox."""

    async def run(self, command: str) -> ExecResult: ...
    async def read(self, path: str) -> bytes: ...
    async def write(self, path: str, content: bytes) -> None: ...


def _discovery_command() -> str:
    # Deferred: benchflow.agents pulls in the agent registry.
    from benchflow.agents.credentials import CREDENTIAL_EVIDENCE_PATHS

    rels = " ".join(shlex.quote(rel) for rel in CREDENTIAL_EVIDENCE_PATHS)
    homes = " ".join(CREDENTIAL_HOMES)
    return (
        f'for h in {homes}; do [ -d "$h" ] || continue; '
        f'for r in {rels}; do p="$h/$r"; '
        'if [ -L "$p" ] || [ -e "$p" ]; then '
        'find "$p" \\( -type f -o -type l \\) '
        "-exec stat -c '%u:%g:%a:%F:%n' {} + 2>/dev/null; "
        "fi; done; done; true"
    )


async def _list_credentials(ops: CredentialOps) -> list[tuple[str, str, str, str, str]]:
    result = await ops.run(_discovery_command())
    if result.return_code != 0:
        raise CredentialScrubError(
            "could not list credential files before the snapshot: "
            f"{(result.stderr or '').strip()[:300]}"
        )
    entries = []
    for line in (result.stdout or "").splitlines():
        parts = line.split(":", 4)
        if len(parts) != 5 or not parts[4].startswith("/"):
            continue
        uid, gid, mode, kind, path = parts
        if not (uid.isdigit() and gid.isdigit() and mode.isdigit()):
            raise CredentialScrubError(f"unexpected stat output for {path}")
        entries.append((uid, gid, mode, kind, path))
    return entries


async def scrub_credentials(ops: CredentialOps) -> list[StashedCredential]:
    """Read credential files into memory and remove them from the sandbox.

    Returns the stash (empty when there is nothing to scrub). Raises
    :class:`CredentialScrubError`, with every file put back, when a file is a
    symbolic link or cannot be removed.
    """
    entries = await _list_credentials(ops)
    links = [path for _, _, _, kind, path in entries if "symbolic link" in kind]
    if links:
        raise CredentialScrubError(
            "refusing to snapshot: credential path is a symbolic link, so its "
            f"target could be captured: {', '.join(links)}"
        )
    stash = [
        StashedCredential(path, uid, gid, mode, await ops.read(path))
        for uid, gid, mode, _, path in entries
    ]
    if not stash:
        return []
    try:
        quoted = " ".join(shlex.quote(item.path) for item in stash)
        removed = await ops.run(f"rm -f -- {quoted}")
        remaining = (
            [entry[4] for entry in await _list_credentials(ops)]
            if removed.return_code == 0
            else [item.path for item in stash]
        )
        if remaining:
            raise CredentialScrubError(
                "could not remove credential files before the snapshot: "
                f"{', '.join(remaining)} {(removed.stderr or '').strip()[:300]}"
            )
    except BaseException:
        await put_back_credentials(ops, stash)
        raise
    return stash


async def _staging_dir(ops: CredentialOps) -> str:
    """Create a fresh root-owned 0700 directory the agent uid cannot enter.

    Credential content is staged here before the symlink-safe write-back copies
    it into the agent-controlled destination. Because the directory is created
    by ``mktemp -d`` as root at mode 0700, the agent cannot pre-plant a symlink
    inside it, so staging the bytes is itself safe.
    """
    result = await ops.run('umask 077 && d="$(mktemp -d)" && printf %s "$d"')
    path = (result.stdout or "").strip()
    if result.return_code != 0 or not path.startswith("/"):
        raise CredentialScrubError(
            "could not create a staging directory for the credential put-back: "
            f"{(result.stderr or '').strip()[:300]}"
        )
    return path


async def put_back_credentials(
    ops: CredentialOps, stash: list[StashedCredential]
) -> None:
    """Write stashed credential files back, following no symlinks.

    Each file is staged in a fresh root-only directory, then created at its
    destination with ``O_NOFOLLOW`` + ``O_CREAT | O_EXCL`` and its owner and
    mode set on the open descriptor (see
    :mod:`benchflow.sandbox._credential_writeback`). An agent that swaps a
    credential path — or a directory above it — for a symlink while its
    processes are alive cannot make root write through the link: the write is
    refused and named. Requires ``python3`` in the sandbox (the agent images
    that carry credentials ship it); a missing interpreter fails closed.
    """
    if not stash:
        return
    staging = await _staging_dir(ops)
    try:
        manifest = []
        for index, item in enumerate(stash):
            staged = f"{staging}/{index}"
            await ops.write(staged, item.content)
            manifest.append(
                {
                    "staged": staged,
                    "path": item.path,
                    "uid": item.uid,
                    "gid": item.gid,
                    "mode": item.mode,
                }
            )
        command = shlex.join(
            ["python3", "-I", "-c", _WRITEBACK_SOURCE, json.dumps(manifest)]
        )
        result = await ops.run(command)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "").strip()[:300]
            raise CredentialScrubError(
                "could not restore credential files without following a link "
                f"(refused a symlink swap, or python3 is missing): {detail}"
            )
    finally:
        await ops.run(f"rm -rf -- {shlex.quote(staging)}")
