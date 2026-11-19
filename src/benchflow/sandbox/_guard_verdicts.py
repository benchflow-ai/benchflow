"""What the pytest plugin guard's ``verdicts`` record says of what it refused
((g2)).

When the guard stops pytest it leaves a signed ``verdicts`` marker
(``_pytest_plugin_guard._record_verdicts`` and ``_stopping``): a JSON list
with one entry per file it refused, plus each byte-code cache Python would read
for that file, or for the file it was judging when its own code raised:

    {"v": 1, "guard": <name>, "kind": "rejected" | "guard_error",
     "path": <file>, "sha256": <digest of the bytes judged, or "">,
     "why": <the refusal, or the exception>,
     "verdict": "identical" | "changed" | "new" | "no-manifest",
     "python": <the Python's version>, "executable": <its sys.executable>}

Like every marker, test.sh's user can write it, and in a SWE-bench run that
user is root and has already run the agent's setup.py. So the record only
nominates. A rejection of a file the image does not have as it is (``changed``,
``new``, or ``no-manifest`` for the trees the manifest leaves out) lowers the
trial to 0, with the file named: a forged one only lowers the forger's own
score. The guard's own error, a rejection of the image's own file, and a stop
with no verdict cannot take a trial out of the score on the record's word.
They score a recorded 0 until the host re-judges the nominated files on the
pristine image.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.sandbox.lockdown import GuardMarker

#: A rejected file's verdict when the image does not have it as it is, and
#: how a 0 names it.
NOT_THE_IMAGES = {
    "changed": "changed since the image",
    "new": "new since the image",
    "no-manifest": "in a tree the image manifest leaves out",
}
_VERDICTS = (*NOT_THE_IMAGES, "identical")
_KINDS = ("rejected", "guard_error")
_WHY_CHARS = 300


@dataclass(frozen=True)
class VerdictEntry:
    kind: str
    path: str
    verdict: str = ""
    why: str = ""
    sha256: str = ""
    python: str = ""
    executable: str = ""


def read_verdicts(
    markers: Iterable[GuardMarker], guard: str
) -> tuple[list[VerdictEntry], str | None]:
    """The entries of the guard *guard*'s ``verdicts`` markers, and why some
    could not be read, or None."""
    entries: list[VerdictEntry] = []
    problem = None
    for marker in markers:
        try:
            data: Any = json.loads(marker.detail)
        except ValueError:
            data = None
        if not isinstance(data, list):
            problem = problem or f"verifier/{marker.name} is not a list of verdicts"
            continue
        for item in data:
            entry = _entry(item, guard)
            if entry is None:
                problem = (
                    problem or f"an entry of verifier/{marker.name} could not be read"
                )
                continue
            entries.append(entry)
    return entries, problem


def _entry(item: Any, guard: str) -> VerdictEntry | None:
    if not isinstance(item, dict) or item.get("v") != 1 or item.get("guard") != guard:
        return None
    kind = item.get("kind")
    if kind not in _KINDS:
        return None
    fields = {
        name: item.get(name, "")
        for name in ("path", "verdict", "why", "sha256", "python", "executable")
    }
    if not all(isinstance(value, str) for value in fields.values()):
        return None
    if kind == "rejected" and (
        not fields["path"] or fields["verdict"] not in _VERDICTS
    ):
        return None
    return VerdictEntry(
        kind,
        fields["path"],
        fields["verdict"],
        fields["why"][:_WHY_CHARS],
        fields["sha256"],
        fields["python"][:32],
        fields["executable"][:4096],
    )


def agents_rejection(entries: Iterable[VerdictEntry]) -> str | None:
    """Why the trial scores 0, when an entry rejects a file the image does not
    have as it is, or None."""
    foreign = [
        e for e in entries if e.kind == "rejected" and e.verdict in NOT_THE_IMAGES
    ]
    if not foreign:
        return None
    first = foreign[0]
    more = len(foreign) - 1
    return (
        f"the pytest plugin guard rejected {first.path} ("
        + NOT_THE_IMAGES[first.verdict]
        + (f": {first.why}" if first.why else "")
        + ")"
        + (f" and {more} more file{'s' if more > 1 else ''}" if more else "")
        + "; a file the image does not have as it is, so the trial scores 0"
    )


def unconfirmed(entries: Iterable[VerdictEntry], problem: str | None) -> str:
    """What the record says of a stop that is not a rejection of a file the
    image lacks, in words that neither blame nor unscore (FIX-49): the guard's
    own error, a refusal of the image's own file, or no verdict. Until the
    host re-judges it on the pristine image, such a stop is a recorded 0."""
    entries = list(entries)
    errors = [e for e in entries if e.kind == "guard_error"]
    identical = [
        e for e in entries if e.kind == "rejected" and e.verdict == "identical"
    ]
    if errors:
        e = errors[0]
        cause = ", ".join(filter(None, (e.why, e.python and f"Python {e.python}")))
        return f"the guard failed while judging {e.path or 'a file'}" + (
            f" ({cause})" if cause else ""
        )
    if identical:
        e = identical[0]
        more = len(identical) - 1
        return (
            f"the guard refused {e.path}"
            + (f" and {more} more file{'s' if more > 1 else ''}" if more else "")
            + ", byte for byte the image's own"
            + (f" ({e.why})" if e.why else "")
        )
    if problem is not None:
        return f"its verdicts record could not all be read ({problem})"
    return "no verdict of the guard names a file"
