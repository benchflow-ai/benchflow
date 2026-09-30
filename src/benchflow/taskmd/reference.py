"""task.md draft 2's reference parser, as BenchFlow calls it.

BenchFlow does not re-implement the draft 2 grammar. It calls the vendored
reference parser (``benchflow.taskmd._vendor.taskmd``, a verbatim copy of the
spec repository's ``tools/taskmd.py``), so a package parses in BenchFlow
exactly as ``python3 tools/taskmd.py json <package>`` parses it, diagnostics
included. ``tests/test_taskmd_differential.py`` holds BenchFlow to that.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchflow.taskmd._vendor import taskmd as ref
from benchflow.taskmd._vendor import vendor_pin

TaskDocument = ref.TaskDocument
Diagnostic = ref.Diagnostic


def parse_package(target: str | Path) -> Any:
    """The reference parser's ``TaskDocument`` for a package folder or a task.md file.

    Parsing never runs package code: a family's generator runs only when a
    seed is materialized (``benchflow.taskmd.family``) or in
    :func:`reference_check`, as the reference checker runs it.
    """
    return ref.parse(target)


def document_json(document: Any) -> dict[str, Any]:
    """What ``tools/taskmd.py json`` prints for ``document``, as a JSON value.

    The reference tool prints ``asdict(doc) | {"title", "ok"}`` with
    ``default=str``; the round trip applies the same conversions (TOML dates
    and times become strings).
    """
    data = asdict(document) | {"title": document.title, "ok": document.ok}
    return json.loads(json.dumps(data, default=str))


def package_json(target: str | Path) -> dict[str, Any]:
    """:func:`document_json` of :func:`parse_package`."""
    return document_json(parse_package(target))


def errors(document: Any) -> list[str]:
    """The reference parser's error diagnostics, each with where it was found."""
    return [
        diagnostic.message + (f" ({diagnostic.where})" if diagnostic.where else "")
        for diagnostic in document.diagnostics
        if diagnostic.level == "error"
    ]


@dataclass(frozen=True)
class ReferenceCheck:
    """What ``python3 tools/taskmd.py check <target>`` prints, and whether it passed."""

    ok: bool
    lines: tuple[str, ...]


def reference_check(target: str | Path) -> ReferenceCheck:
    """Run the reference checker on one target, exactly as its command line does.

    For a family, the checker runs the generator once on a sample seed,
    offline and confined (``bwrap`` on Linux, ``sandbox-exec`` on macOS), or
    says the placeholders were not verified where it cannot.
    """
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = ref.main(["check", str(target)])
    text = out.getvalue() + err.getvalue()
    return ReferenceCheck(ok=status == 0, lines=tuple(text.rstrip("\n").splitlines()))


def reference_commit() -> str:
    """The task-md commit the vendored reference tools come from."""
    return str(vendor_pin()["commit"])
