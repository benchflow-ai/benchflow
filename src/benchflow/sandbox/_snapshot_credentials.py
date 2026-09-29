"""Keep agent credential files out of container snapshots.

A container snapshot (``docker commit``, a Daytona provider snapshot) stores
the whole filesystem at rest, including the credential files an agent install
writes: ``~/.codex/auth.json`` for codex-acp, ``~/.claude/.credentials.json``
for a host subscription login, provider files such as Vertex ADC. Branch
children still need those files, because each child starts a fresh agent
session that reads them.

So capture runs on a scrubbed filesystem. :func:`scrub_credentials` reads the
files into host memory, removes them and checks that none remain; the caller
captures, then :func:`put_back_credentials` writes them back, to the live
sandbox and to every sandbox restored from that snapshot. File contents never
travel as a command-line argument or through logged command output: Docker
streams them over ``docker exec`` stdin/stdout, and Daytona moves them through a
root-only staging directory with its file API, because the Daytona daemon keeps
session command output on disk inside the sandbox.

Privilege separation. The agent's processes can still be running while this
happens (automatic checkpoints snapshot a live sandbox), and the agent controls
its own home: between any two steps it can swap a credential file, or a
directory above it, for a symbolic link. So no step runs as root on a path the
agent controls. Every read, removal and write of a file runs as the owner of
the credential home it lives in (``/home/agent`` -> the agent's uid), whose
privileges a link swap cannot raise: at worst it redirects the agent's own file
operations to files the agent could already reach. Root acts only in homes that
root alone controls (every directory on the path root-owned and not group- or
world-writable, checked in the same shell), where nobody else can swap
anything. Either way a symbolic link on the path is refused and named, and the
mode put back never carries setuid, setgid or sticky bits. No interpreter is
needed in the sandbox: the steps are POSIX ``sh`` with ``stat`` and ``base64``.

The discovery listing is parsed strictly. A file name containing a newline
could forge a listing line naming any path, owner and mode, so one refuses the
snapshot, as does a path outside the credential locations, before anything is
removed.

The inventory is :data:`benchflow.agents.credentials.CREDENTIAL_EVIDENCE_PATHS`
under ``/root`` and every ``/home/<user>``. Credentials a task places
elsewhere are not covered.
"""

from __future__ import annotations

import base64
import binascii
import logging
import shlex
from dataclasses import dataclass, field
from typing import Protocol

from benchflow.sandbox.protocol import ExecResult

logger = logging.getLogger(__name__)

CREDENTIAL_HOMES = ("/root", "/home/*")


class CredentialScrubError(RuntimeError):
    """Credential files could not be kept out of a snapshot; nothing was captured."""


@dataclass(frozen=True)
class StashedCredential:
    """One credential file held in host memory while a snapshot is taken.

    ``uid``, ``gid`` and ``mode`` are the file's own. ``home_uid`` and
    ``home_gid`` own the credential home the file lives in; every operation on
    the file runs with those privileges.
    """

    path: str
    uid: str
    gid: str
    mode: str
    content: bytes = field(repr=False)
    home_uid: str = "0"
    home_gid: str = "0"


@dataclass(frozen=True)
class AsUserResult:
    """Outcome of :meth:`CredentialOps.run_as`; ``stdout`` is raw bytes."""

    return_code: int
    stdout: bytes
    stderr: str


class CredentialOps(Protocol):
    """Access to one live sandbox for the credential scrub."""

    async def run(self, command: str) -> ExecResult:
        """Run ``sh -c command`` as root. Metadata only, never file contents."""
        ...

    async def run_as(
        self, uid: int, gid: int, script: str, *, stdin: bytes | None = None
    ) -> AsUserResult:
        """Run ``sh -c script`` as ``uid:gid``; ``stdin`` and the returned
        stdout travel out of band, never as arguments or logged output."""
        ...


def _discovery_command() -> str:
    # Deferred: benchflow.agents pulls in the agent registry.
    from benchflow.agents.credentials import CREDENTIAL_EVIDENCE_PATHS

    rels = " ".join(shlex.quote(rel) for rel in CREDENTIAL_EVIDENCE_PATHS)
    homes = " ".join(CREDENTIAL_HOMES)
    return (
        # A newline inside a file name could forge a listing line; flag it.
        "nl=$(printf '\\nx'); nl=${nl%x}; "
        f'for h in {homes}; do [ -d "$h" ] || continue; found=; '
        f'for r in {rels}; do p="$h/$r"; '
        'if [ -L "$p" ] || [ -e "$p" ]; then found=1; '
        'if [ -n "$(find "$p" -name "*$nl*" -print 2>/dev/null | head -n 1)" ]; '
        'then echo "!newline:$h"; fi; '
        'find "$p" \\( -type f -o -type l \\) '
        "-exec stat -c 'F:%u:%g:%a:%F:%n' {} + 2>/dev/null; "
        "fi; done; "
        'if [ -n "$found" ]; then stat -c \'H:%u:%g:%a:%F:%n\' "$h"; fi; '
        "done; true"
    )


def _home_of(path: str) -> str | None:
    if path.startswith("/root/"):
        return "/root"
    parts = path.split("/")
    if len(parts) > 3 and parts[1] == "home" and parts[2]:
        return f"/home/{parts[2]}"
    return None


def _is_credential_path(path: str, home: str) -> bool:
    from benchflow.agents.credentials import CREDENTIAL_EVIDENCE_PATHS

    rel = path[len(home) + 1 :]
    if not rel or any(part in ("", ".", "..") for part in rel.split("/")):
        return False
    return any(
        rel == cand or rel.startswith(cand + "/") for cand in CREDENTIAL_EVIDENCE_PATHS
    )


_Homes = dict[str, tuple[str, str, str]]
_Entries = list[tuple[str, str, str, str, str]]


async def _list_credentials(ops: CredentialOps) -> tuple[_Homes, _Entries]:
    result = await ops.run(_discovery_command())
    if result.return_code != 0:
        raise CredentialScrubError(
            "could not list credential files before the snapshot: "
            f"{(result.stderr or '').strip()[:300]}"
        )
    homes: _Homes = {}
    entries: _Entries = []
    # Split on "\n" only: str.splitlines() also splits on \r, \v, \x1c ...
    for line in (result.stdout or "").split("\n"):
        if not line:
            continue
        if line.startswith("!newline:"):
            raise CredentialScrubError(
                "refusing to snapshot: a file name under the credential paths of "
                f"{line.removeprefix('!newline:')} contains a newline, which "
                "could forge a listing entry"
            )
        tag, _, rest = line.partition(":")
        parts = rest.split(":", 4)
        if tag not in ("H", "F") or len(parts) != 5:
            raise CredentialScrubError(
                f"unexpected credential listing line {line[:120]!r}"
            )
        uid, gid, mode, kind, path = parts
        if (
            not (uid.isdigit() and gid.isdigit() and mode.isdigit())
            or not path.startswith("/")
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in path)
        ):
            raise CredentialScrubError(
                f"unexpected credential listing entry {path[:120]!r}"
            )
        if tag == "H":
            homes[path] = (uid, gid, kind)
        else:
            entries.append((uid, gid, mode, kind, path))
    for *_, path in entries:
        home = _home_of(path)
        if home is None or home not in homes or not _is_credential_path(path, home):
            raise CredentialScrubError(
                f"refusing to snapshot: unexpected credential path {path!r}"
            )
    return homes, entries


def _home_owner(homes: _Homes, path: str) -> tuple[int, int]:
    home = _home_of(path)
    assert home is not None  # checked by _list_credentials
    uid, gid, kind = homes[home]
    if "directory" not in kind:
        raise CredentialScrubError(f"refusing to snapshot: {home} is not a directory")
    if uid != "0" and gid == "0":
        raise CredentialScrubError(
            f"refusing to snapshot: {home} belongs to uid {uid} with group root, "
            "so its files cannot be handled with that user's privileges alone"
        )
    return int(uid), int(gid)


# Runs as the credential home's owner (see the module docstring). ``bf_chain``
# refuses a symbolic link or non-directory on the parent chain; as root it also
# requires every directory to be root-owned and not group/world-writable, so
# nobody else can swap one between the check and the use.
_GUARD = """\
set -u
bf_uid=__UID__
[ "$(id -u)" = "$bf_uid" ] || { echo "refused: not running as uid $bf_uid" >&2; exit 98; }
bf_chain() {
  bf_d=; bf_rest=${1#/}
  while [ -n "$bf_rest" ]; do
    bf_c=${bf_rest%%/*}
    case $bf_rest in */*) bf_rest=${bf_rest#*/} ;; *) bf_rest= ;; esac
    bf_d="$bf_d/$bf_c"
    if [ -L "$bf_d" ]; then echo "refused: $bf_d is a symbolic link" >&2; return 4; fi
    if [ "${2:-}" = create ] && [ ! -e "$bf_d" ]; then mkdir -m 755 "$bf_d" || return 4; fi
    if [ ! -d "$bf_d" ]; then echo "refused: $bf_d is not a directory" >&2; return 4; fi
    if [ "$bf_uid" = 0 ]; then
      bf_o=$(stat -c %u "$bf_d") && bf_m=$(stat -c %a "$bf_d") || return 4
      if [ "$bf_o" != 0 ] || [ $((0$bf_m & 18)) -ne 0 ]; then
        echo "refused: $bf_d is writable by someone other than root" >&2; return 4
      fi
    fi
  done
  return 0
}
"""


def _script(action: str, uid: int, path: str, *lines: str, **values: str) -> str:
    assigns = "".join(
        f"{name}={shlex.quote(value)}\n" for name, value in values.items()
    )
    return (
        f"# benchflow-credential {action}\n"
        + _GUARD.replace("__UID__", str(uid))
        + f"p={shlex.quote(path)}\n"
        + assigns
        + "\n".join(lines)
        + "\n"
    )


def _read_script(uid: int, path: str) -> str:
    return _script(
        "read",
        uid,
        path,
        'bf_chain "${p%/*}" || exit 4',
        'if [ -L "$p" ]; then echo "refused: $p is a symbolic link" >&2; exit 4; fi',
        'if [ ! -f "$p" ]; then echo "refused: $p is not a regular file" >&2; exit 4; fi',
        'base64 < "$p"',
    )


def _remove_script(uid: int, path: str) -> str:
    return _script("remove", uid, path, 'bf_chain "${p%/*}" || exit 4', 'rm -f -- "$p"')


def _write_script(uid: int, path: str, mode: str, owner: str) -> str:
    return _script(
        "write",
        uid,
        path,
        'bf_chain "${p%/*}" create || exit 4',
        'if [ -L "$p" ]; then echo "refused: $p is a symbolic link" >&2; exit 4; fi',
        'if [ -e "$p" ] && [ ! -f "$p" ]; then '
        'echo "refused: $p is not a regular file" >&2; exit 4; fi',
        "umask 077",
        'base64 -d > "$p" || exit 5',
        # Only root can (and, in a root-only chain, safely may) restore the owner.
        'if [ "$bf_uid" = 0 ]; then chown "$o" "$p" || exit 5; fi',
        'chmod "$m" "$p" || exit 5',
        m=mode,
        o=owner,
    )


async def _read_as(ops: CredentialOps, uid: int, gid: int, path: str) -> bytes:
    result = await ops.run_as(uid, gid, _read_script(uid, path))
    if result.return_code != 0:
        raise CredentialScrubError(
            f"could not read {path} before the snapshot: {result.stderr.strip()[:300]}"
        )
    try:
        return base64.b64decode(b"".join(result.stdout.split()), validate=True)
    except binascii.Error as exc:
        raise CredentialScrubError(
            f"could not read {path} before the snapshot"
        ) from exc


async def scrub_credentials(ops: CredentialOps) -> list[StashedCredential]:
    """Read credential files into memory and remove them from the sandbox.

    Returns the stash (empty when there is nothing to scrub). Raises
    :class:`CredentialScrubError`, with every file put back, when a file is a
    symbolic link or cannot be read or removed.
    """
    homes, entries = await _list_credentials(ops)
    links = [path for *_, kind, path in entries if "symbolic link" in kind]
    if links:
        raise CredentialScrubError(
            "refusing to snapshot: credential path is a symbolic link, so its "
            f"target could be captured: {', '.join(links)}"
        )
    stash: list[StashedCredential] = []
    for uid, gid, mode, _, path in entries:
        home_uid, home_gid = _home_owner(homes, path)
        content = await _read_as(ops, home_uid, home_gid, path)
        stash.append(
            StashedCredential(
                path, uid, gid, mode, content, str(home_uid), str(home_gid)
            )
        )
    if not stash:
        return []
    try:
        for item in stash:
            uid = int(item.home_uid)
            removed = await ops.run_as(
                uid, int(item.home_gid), _remove_script(uid, item.path)
            )
            if removed.return_code != 0:
                raise CredentialScrubError(
                    f"could not remove credential files before the snapshot: "
                    f"{item.path} {removed.stderr.strip()[:300]}"
                )
        _, remaining = await _list_credentials(ops)
        if remaining:
            raise CredentialScrubError(
                "could not remove credential files before the snapshot: "
                + ", ".join(entry[4] for entry in remaining)
            )
    except BaseException as exc:
        try:
            await put_back_credentials(ops, stash)
        except CredentialScrubError as put_back_error:
            exc.add_note(f"putting the files back also failed: {put_back_error}")
        raise
    return stash


async def put_back_credentials(
    ops: CredentialOps, stash: list[StashedCredential]
) -> None:
    """Write stashed credential files back, with the home owner's privileges.

    Every file is attempted. A path that became a symbolic link (or whose
    parent chain did) is refused, logged and named in the
    :class:`CredentialScrubError` raised at the end; it is never written
    through. The mode is restored without setuid, setgid or sticky bits.
    """
    refused: list[str] = []
    for item in stash:
        uid = int(item.home_uid)
        mode = format(int(item.mode, 8) & 0o777, "o")
        result = await ops.run_as(
            uid,
            int(item.home_gid),
            _write_script(uid, item.path, mode, f"{item.uid}:{item.gid}"),
            stdin=base64.b64encode(item.content),
        )
        if result.return_code != 0:
            reason = result.stderr.strip()[:200] or f"exit {result.return_code}"
            logger.warning("Credential put-back of %s refused: %s", item.path, reason)
            refused.append(f"{item.path} ({reason})")
    if refused:
        raise CredentialScrubError(
            "could not restore credential files safely (never written through a "
            f"link): {'; '.join(refused)}"
        )
