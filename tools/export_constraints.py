"""Regenerate ``constraints.txt``, the locked versions for user installs.

``uv tool install benchflow`` and ``pip install benchflow`` never read
``uv.lock``: they resolve the newest versions ``pyproject.toml`` allows on the
day, so two users of one release can run different typer, LiteLLM or httpx.
``constraints.txt`` is the lock's closure as plain ``name==version`` pins
(every extra but ``dev``, with environment markers, no hashes), committed at
the root of each release tag, so an install can ask for exactly what CI tested::

    uv tool install --python 3.12 benchflow==<version> \\
        --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v<version>/constraints.txt

Run from the repository root after any change to ``uv.lock``::

    uv run python tools/export_constraints.py

``--check`` writes nothing and exits 1 when the file differs from what the
lock gives (``tests/test_constraints_file.py`` runs the same comparison).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "constraints.txt"
REGENERATE_COMMAND = "uv run python tools/export_constraints.py"

HEADER = (
    "# BenchFlow's locked dependency versions, exported from uv.lock (every extra\n"
    "# but dev). Use it as constraints for an install that matches the release:\n"
    "#    uv tool install --python 3.12 benchflow==<version> --constraints constraints.txt\n"
    "#    pip install benchflow==<version> --constraint constraints.txt\n"
    "# Generated, do not edit; regenerate from the repository root with:\n"
    f"#    {REGENERATE_COMMAND}\n"
)


def uv_executable() -> str:
    """The uv that runs this script (``uv run`` sets ``UV``), else uv on PATH."""
    found = os.environ.get("UV") or shutil.which("uv")
    if not found:
        raise SystemExit(
            f"uv not found: run this as `{REGENERATE_COMMAND}` from the repository root"
        )
    return found


def render(*, uv: str | None = None, root: Path = ROOT) -> str:
    """The file's text, as ``uv export`` gives it from the lock.

    ``--locked`` fails when ``uv.lock`` is stale against ``pyproject.toml``.
    ``--no-emit-project`` leaves BenchFlow itself out (the user names the
    version they install); ``--no-hashes`` because a constraint with a hash
    turns on hash checking for the whole install under pip, which then
    refuses every package the file does not list.
    """
    result = subprocess.run(
        [
            uv or uv_executable(),
            "export",
            "--locked",
            "--quiet",
            "--all-extras",
            "--no-extra",
            "dev",
            "--no-default-groups",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--format",
            "requirements.txt",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"uv export failed ({result.returncode}): {result.stderr.strip()}"
        )
    return HEADER + result.stdout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate constraints.txt from uv.lock."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 when constraints.txt differs from the lock's export",
    )
    args = parser.parse_args(argv)
    text = render()
    current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else None
    if current == text:
        return 0
    if args.check:
        print(f"stale {OUTPUT.name}; regenerate it with: {REGENERATE_COMMAND}")
        return 1
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
