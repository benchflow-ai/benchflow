"""The verifier's trusted PATH as hardening judged it, for the trial's record.

Hardening (``benchflow.sandbox.lockdown``) judges the image's PATH folders in
the sandbox and reports each decision here; ``_verify_rollout`` collects the
reports with :func:`recording_trusted_path`, writes the folders left out to
``verifier/trusted-path.json`` and names them in the error of a verifier that
could not find a program (#1174). The rollout kernel imports this module, not
the concrete hardening plane.

A program found only in a folder left out is not simply missing: hardening
puts a stub of that name at the end of the verifier's PATH, and the stub
leaves a hit when it runs. A hit is the evidence that the
verifier needed a refused program, whatever test.sh printed or exited with;
``_verify_rollout`` reads the hits with :func:`refused_programs_cmd`.
"""

from __future__ import annotations

import contextlib
import contextvars
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_to_bytes

# Encode every byte before a backend can normalize CR/LF or decode UTF-8.
# Shell builtins only, including on images without Python or base64.
PATH_FIELD_ENCODE_SH = r"""
bf_encode() (
  value=$1
  LC_ALL=C
  export LC_ALL
  while [ -n "$value" ]; do
    rest=${value#?}
    char=${value%"$rest"}
    printf '%%%02X' "'$char"
    value=$rest
  done
)
""".strip()
PATH_QUERY_CMD = (
    PATH_FIELD_ENCODE_SH + '\nbf_encode "${_BENCHFLOW_IMAGE_PATH-$PATH}"; printf "\\n"'
)
PYTHONPATH_QUERY_CMD = (
    PATH_FIELD_ENCODE_SH + '\nbf_encode "${PYTHONPATH-}"; printf "\\n"'
)


class VerifierPathEvidenceError(RuntimeError):
    """A verifier PATH inventory contains unsupported agent-writable names."""


def decode_path_field(value: str) -> str:
    """Decode a shell record without applying whitespace/line normalization."""
    try:
        return unquote_to_bytes(value).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerifierPathEvidenceError(
            "Verifier hardening failed: a PATH record contains a non-UTF-8 "
            "path or executable name, which is unsupported"
        ) from exc


@dataclass(frozen=True)
class TrustedPathReport:
    """What the trusted-PATH program decided about the image's PATH.

    *kept* are the extras the verifier's PATH keeps, in the image's order;
    *kept_other* the ones among them root does not own (kept because the agent
    user could never have changed them); *dropped* each left-out folder with
    the reason. *refusals* is the folder of the refusal stubs, and *stubs*
    each stubbed program with the left-out folder it is in. *service* is
    the Compose service the verifier runs in, where PATH was judged and the
    stubs made (V3-CONFIRM-5).
    """

    path: str
    kept: tuple[str, ...] = ()
    kept_other: tuple[str, ...] = ()
    dropped: tuple[tuple[str, str], ...] = ()
    refusals: str | None = None
    stubs: tuple[tuple[str, str], ...] = ()
    service: str = "main"
    omitted_tools: tuple[tuple[str, str], ...] = ()

    @property
    def run_path(self) -> str:
        """The PATH test.sh runs with: *path*, then the refusal stubs."""
        if self.refusals and self.stubs:
            return f"{self.path}:{self.refusals}/bin"
        return self.path

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": self.path,
            "kept_not_root_owned": list(self.kept_other),
            "dropped": [{"folder": f, "why": why} for f, why in self.dropped],
        }
        if self.omitted_tools:
            record["omitted_tools"] = [
                {"tool": tool, "why": why} for tool, why in self.omitted_tools
            ]
        if self.service != "main":
            record["service"] = self.service
        return record

    def dropped_text(self) -> str:
        """``<folder>: <why>; ...`` for an error or a log line."""
        return "; ".join(f"{folder}: {why}" for folder, why in self.dropped)


def refused_programs_cmd(report: TrustedPathReport) -> str | None:
    """The sandbox command that prints each stubbed program that ran, or
    ``None`` when *report* has no stubs. Shell builtins only."""
    if not report.refusals or not report.stubs:
        return None
    hits = shlex.quote(f"{report.refusals}/hits")
    count = len(dict.fromkeys(name for name, _ in report.stubs))
    # Read every pre-created marker, not a glob: missing evidence is an error.
    # Numeric IDs are independent of basename bytes and filesystem NAME_MAX.
    return (
        f'hits={hits}\ni=1\nwhile [ "$i" -le {count} ]; do\n'
        "  f=$hits/$i/state\n"
        '  [ -f "$f" ] && [ ! -L "$f" ] && [ -r "$f" ] || exit 1\n'
        "  state= used=no\n"
        "  {\n"
        '    IFS= read -r state && [ "$state" = ready ] || exit 1\n'
        '    while IFS= read -r state || [ -n "$state" ]; do\n'
        '      [ "$state" = used ] || exit 1\n'
        "      used=yes\n"
        "    done\n"
        '  } < "$f" || exit 1\n'
        '  printf "%s %s\\000" "$i" "$used"\n'
        "  i=$((i + 1))\n"
        "done\n"
    )


def refused_programs(report: TrustedPathReport, stdout: str) -> list[tuple[str, str]]:
    """The stubbed programs in *stdout* that ran, with their folders, in
    *report*'s order. IDs map to host-held names; every expected ID must have
    exactly one valid record. Missing, duplicate or corrupt evidence fails closed.
    """
    names = list(dict.fromkeys(name for name, _ in report.stubs))
    records = stdout.split("\x00")
    if records.pop() != "" or len(records) != len(names):
        raise ValueError("incomplete refusal markers")
    ran = set()
    for index, (name, record) in enumerate(zip(names, records, strict=True), 1):
        if record == f"{index} yes":
            ran.add(name)
        elif record != f"{index} no":
            raise ValueError("invalid refusal marker")
    return [stub for stub in report.stubs if stub[0] in ran]


# The reports of the hardening calls made inside ``recording_trusted_path()``.
_SINK: contextvars.ContextVar[list[TrustedPathReport] | None] = contextvars.ContextVar(
    "benchflow_trusted_path_sink", default=None
)


@contextlib.contextmanager
def recording_trusted_path() -> Iterator[list[TrustedPathReport]]:
    """Collect the :class:`TrustedPathReport` of each verifier PATH judged inside."""
    reports: list[TrustedPathReport] = []
    token = _SINK.set(reports)
    try:
        yield reports
    finally:
        _SINK.reset(token)


def report_trusted_path(report: TrustedPathReport) -> None:
    """Hand *report* to the innermost ``recording_trusted_path()``, if any."""
    sink = _SINK.get()
    if sink is not None:
        sink.append(report)
