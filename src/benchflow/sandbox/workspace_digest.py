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
#: reader knows two digests are comparable before comparing them. The
#: null-safe/fail-closed wording is deliberate: the original ``find|sort|
#: xargs`` form word-split legal filenames and recorded a *successful wrong*
#: digest (PR #1046 second review, P1-A), so a digest recorded under that
#: basis must never read as comparable to one recorded under this one.
WORKSPACE_DIGEST_BASIS = (
    "sha256 over sorted per-file sha256sums + a sorted stat listing of names "
    "and permission bits (null-safe find -print0|sort -z|xargs -0, "
    "fail-closed staged pipeline)"
)

_MARKER = "BFWSDIGEST:"


def workspace_digest_command(
    root: str = WORKSPACE_DIGEST_ROOT, *, exclude_basename: str | None = None
) -> str:
    """The in-sandbox pipeline: one deterministic line summarizing ``root``.

    Null-safe and fail-closed (PR #1046 second review, P1-A):

    * File names travel NUL-terminated end to end
      (``find -print0 | sort -z | xargs -0``), so a legal name containing
      spaces — or even a newline — stays one argument. The original
      newline-separated form word-split ``file with spaces.txt`` into three
      arguments; the inner ``sha256sum`` failed on all of them and two
      materially different workspaces recorded the *same* digest.
    * Every stage writes to its own file under a private temp dir with its
      exit status checked by ``&&`` — an inner failure fails the whole
      command, so a digest is either correct or absent-with-reason, never
      wrong. This is intrinsic failure propagation: the sandboxes run
      ``sh -c`` (busybox ash on minimal images), where ``pipefail`` is not
      reliably available, and a trailing ``| sha256sum`` would otherwise
      succeed over a broken producer (a pipeline's status is its last
      command's). ``LC_ALL=C`` pins the sort to byte order so the digest
      does not depend on the image's locale.

    For workspaces whose filenames the old pipeline handled correctly, the
    byte stream reaching the final ``sha256sum`` is unchanged, so recorded
    digest values are stable; trees with pathological names now digest
    correctly (a genuinely different value) instead of colliding.

    ``exclude_basename`` drops files whose basename matches the glob from
    both listings — the docker proofs exclude a ``.backup``-restored sqlite
    DB that is logically, not byte-, identical.
    """
    quoted = shlex.quote(root)
    skip = f" ! -name {shlex.quote(exclude_basename)}" if exclude_basename else ""
    d = '"$__bf_td"'
    return (
        f"cd {quoted} && __bf_td=$(mktemp -d) && "
        f"find . -type f{skip} -print0 >{d}/f0 && "
        f"LC_ALL=C sort -z <{d}/f0 >{d}/f1 && "
        f"xargs -0 -r sha256sum <{d}/f1 >{d}/sums && "
        f"find .{skip} -print0 >{d}/a0 && "
        f"LC_ALL=C sort -z <{d}/a0 >{d}/a1 && "
        f"xargs -0 -r stat -c '%n %a' <{d}/a1 >{d}/stats && "
        f"cat {d}/sums {d}/stats >{d}/all && "
        f"sha256sum <{d}/all && "
        f"rm -rf {d}"
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
