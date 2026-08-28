"""Deterministic digest of a sandbox workspace directory.

The reusable form of the state-digest pipeline the branch docker proofs run
(``tests/test_branch_composed_docker.py``): file contents plus a ``stat``
listing of names and permission bits, all piped through a final
``sha256sum``. One line out, stable for identical trees — the oracle those
tests use to prove snapshot→restore losslessness, and the workspace half of
the replay cut-point accounting the rollout-branching RFC §3.5 promises
("record a content digest of the last replayed request *and the workspace*").

The value is echoed behind a marker prefix and extracted by prefix match:
sandbox ``exec`` implementations merge stderr into stdout, so compose
warnings ("Found orphan containers…") would otherwise corrupt the digest.
"""

from __future__ import annotations

import shlex
from typing import Any

#: The agent's conventional working directory inside benchflow sandboxes.
WORKSPACE_DIGEST_ROOT = "/app"

#: What the digest is computed over — recorded next to every digest so a
#: reader knows two digests are comparable before comparing them.
WORKSPACE_DIGEST_BASIS = (
    "sha256 over sorted per-file sha256sums + a sorted stat listing of names "
    "and permission bits (find|sort|sha256sum)"
)

_MARKER = "BFWSDIGEST:"


def workspace_digest_command(root: str = WORKSPACE_DIGEST_ROOT) -> str:
    """The in-sandbox pipeline: one deterministic line summarizing ``root``."""
    quoted = shlex.quote(root)
    return (
        f"cd {quoted} && {{ "
        "find . -type f | sort | xargs -r sha256sum; "
        "find . | sort | xargs -r stat -c '%n %a'; "
        "} | sha256sum"
    )


async def compute_workspace_digest(
    sandbox: Any,
    *,
    root: str = WORKSPACE_DIGEST_ROOT,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Compute the workspace digest of ``root`` inside a live sandbox.

    Returns ``{"digest": "sha256:<hex>", "basis": WORKSPACE_DIGEST_BASIS,
    "root": root}`` or raises — the caller records the failure reason instead
    of the digest; a digest is never fabricated.
    """
    command = workspace_digest_command(root)
    wrapped = f'__bf_out="$({command})" && echo "{_MARKER}${{__bf_out}}"'
    result = await sandbox.exec(wrapped, timeout_sec=timeout_sec)
    if result.return_code != 0:
        raise RuntimeError(
            f"workspace digest command failed (rc={result.return_code}): "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    values = [
        line[len(_MARKER) :].strip()
        for line in (result.stdout or "").splitlines()
        if line.startswith(_MARKER)
    ]
    if len(values) != 1:
        raise RuntimeError(
            f"workspace digest expected exactly one {_MARKER} line, got {values!r}"
        )
    hex_digest = values[0].split()[0] if values[0] else ""
    if len(hex_digest) != 64 or any(c not in "0123456789abcdef" for c in hex_digest):
        raise RuntimeError(f"workspace digest output is not a sha256: {values[0]!r}")
    return {
        "digest": f"sha256:{hex_digest}",
        "basis": WORKSPACE_DIGEST_BASIS,
        "root": root,
    }
