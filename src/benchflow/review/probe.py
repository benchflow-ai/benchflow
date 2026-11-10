"""What a reviewer sandbox holds, for the review's record.

``reviewer-environment.json`` and the review record that carries it name the
reviewer image by its reference and, where one is known, its digest
(:func:`image_digest`, :func:`local_image_id`), and what
:func:`probe_reviewer_image` found in it: the PDF tools on its ``PATH`` with
their versions, and the Python packages it holds, read with
``importlib.metadata`` so the probe imports and writes nothing. The probe is
the reviewer rollout's last pre-agent hook, which runs before the wrapper
task's setup commands: it sees the image as built, and the setup then adds
the record's ``artifact_packages`` at their exact pins (a failed install
fails the review). It never fails a review: an error is recorded instead.
"""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from benchflow._utils.text import describe_exception
from benchflow.sandbox._exec_output import exec_raw

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SEC = 60
#: The command-line PDF tools a reviewer reaches for.
PROBE_COMMANDS = ("pdftotext", "pdftoppm", "pdfinfo", "mutool", "gs", "qpdf")
#: The Python distributions that read or render a PDF, numpy beside them.
PROBE_PACKAGES = ("numpy", "pypdf", "PyMuPDF", "pdfplumber", "pdfminer.six", "pillow")

_PROBE_SCRIPT = (
    "for tool in "
    + " ".join(PROBE_COMMANDS)
    + """; do
  if command -v "$tool" > /dev/null 2>&1; then
    case "$tool" in
      gs|qpdf) flag=--version ;;
      *) flag=-v ;;
    esac
    printf 'command\\t%s\\tfound\\t%s\\n' "$tool" "$("$tool" "$flag" 2>&1 | head -n 1)"
  else
    printf 'command\\t%s\\tmissing\\t\\n' "$tool"
  fi
done
if command -v python3 > /dev/null 2>&1; then
  python3 -I -c 'import importlib.metadata as m, platform, sys
print("python\\t" + platform.python_version())
for name in sys.argv[1:]:
    try:
        version = m.version(name)
    except m.PackageNotFoundError:
        version = ""
    print("package\\t" + name + "\\t" + version)' """
    + " ".join(PROBE_PACKAGES)
    + """
fi
"""
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_UNPRINTABLE = re.compile(r"[^\x20-\x7e]")
_MAX_VALUE = 120


def _clean(value: str, limit: int = _MAX_VALUE) -> str:
    return _UNPRINTABLE.sub("", value).strip()[:limit]


def image_digest(image: str) -> str | None:
    """The manifest digest *image* is pinned to (``name@sha256:...``), or None."""
    name, at, digest = image.rpartition("@")
    return digest if at and name and _DIGEST.fullmatch(digest) else None


def local_image_id(image: str) -> str | None:
    """The id of the local Docker image *image* names, or None when there is
    no docker CLI or no such image."""
    docker = shutil.which("docker")
    if docker is None:
        return None
    try:
        inspected = subprocess.run(
            [docker, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = inspected.stdout.strip()
    return value if inspected.returncode == 0 and _DIGEST.fullmatch(value) else None


def parse_probe(stdout: str) -> dict[str, Any]:
    """The probe's output as the record's ``image_tools``.

    ``commands`` maps each of :data:`PROBE_COMMANDS` to its version line
    (``"unknown version"`` when it printed none) or None when it is missing;
    ``packages`` maps each of :data:`PROBE_PACKAGES` to its version or None;
    ``python`` is the image's ``python3`` version, None without one.
    """
    commands: dict[str, str | None] = dict.fromkeys(PROBE_COMMANDS)
    packages: dict[str, str | None] = dict.fromkeys(PROBE_PACKAGES)
    python: str | None = None
    for line in stdout.splitlines():
        kind, _, rest = line.partition("\t")
        if kind == "command":
            name, _, rest = rest.partition("\t")
            state, _, version = rest.partition("\t")
            if name in commands and state == "found":
                commands[name] = _clean(version) or "unknown version"
        elif kind == "package":
            name, _, version = rest.partition("\t")
            if name in packages:
                packages[name] = _clean(version) or None
        elif kind == "python":
            python = _clean(rest) or None
    return {"python": python, "packages": packages, "commands": commands}


async def probe_reviewer_image(
    sandbox: Any, *, record: dict[str, Any], record_path: Path
) -> None:
    """Add ``image_tools`` to *record* and rewrite it at *record_path*.

    A pre-agent hook of the reviewer rollout. Any error is recorded in place
    of the tools; the review goes on.
    """
    from benchflow.review.persistence import write_json_atomic

    try:
        result = await exec_raw(
            sandbox.exec,
            shlex.join(["sh", "-c", _PROBE_SCRIPT]),
            user="root",
            timeout_sec=PROBE_TIMEOUT_SEC,
        )
        tools = parse_probe(result.stdout or "")
        if result.return_code:
            tools["error"] = f"the probe exited {result.return_code}"
    except Exception as exc:
        tools = {"error": _clean(describe_exception(exc), 300)}
    record["image_tools"] = tools
    try:
        write_json_atomic(record_path, record)
    except OSError:
        logger.warning("Could not record the reviewer image's tools", exc_info=True)
