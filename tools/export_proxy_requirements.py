"""Regenerate the LiteLLM proxy's hashed requirement files from ``uv.lock``.

The model proxy BenchFlow installs inside a sandbox (and on the host when a
run adds extra requirements) installs exactly these files, with hash
checking, instead of resolving ``litellm[proxy]`` afresh: what runs in the
sandbox is then the locked set that ``pip-audit`` of the lock covers. Users
have no ``uv.lock``, so the export ships in the wheel, under
``src/benchflow/providers/proxy_requirements/``.

Run from the repository root after any change to ``uv.lock`` or to the
``litellm-proxy`` dependency groups in ``pyproject.toml``::

    uv run python tools/export_proxy_requirements.py

``--check`` writes nothing and exits 1 when a file differs from what the lock
gives (``tests/test_proxy_requirements.py`` runs the same comparison).
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
OUT_DIR = ROOT / "src" / "benchflow" / "providers" / "proxy_requirements"
GROUPS = ("litellm-proxy", "litellm-proxy-vertex")
REGENERATE_COMMAND = "uv run python tools/export_proxy_requirements.py"

HEADER = (
    "# BenchFlow's LiteLLM model proxy: dependency group {group!r} of\n"
    "# pyproject.toml, exported from uv.lock with hashes. Generated, do not edit;\n"
    "# regenerate from the repository root with:\n"
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


def output_path(group: str) -> Path:
    return OUT_DIR / f"{group}.txt"


def render(group: str, *, uv: str | None = None, root: Path = ROOT) -> str:
    """The file's text for one group, as ``uv export`` gives it from the lock.

    ``--locked`` fails when ``uv.lock`` is stale against ``pyproject.toml``,
    so a group edited without re-locking is caught too. ``--only-group``
    leaves out BenchFlow's own dependencies and the dev group.
    """
    result = subprocess.run(
        [
            uv or uv_executable(),
            "export",
            "--locked",
            "--quiet",
            "--only-group",
            group,
            "--no-emit-project",
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
            f"uv export of group {group!r} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return HEADER.format(group=group) + result.stdout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the proxy's hashed requirement files from uv.lock."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 when a file differs from the lock's export",
    )
    args = parser.parse_args(argv)
    uv = uv_executable()
    stale: list[str] = []
    for group in GROUPS:
        text = render(group, uv=uv)
        path = output_path(group)
        current = path.read_text() if path.is_file() else None
        if current == text:
            continue
        stale.append(str(path.relative_to(ROOT)))
        if not args.check:
            path.write_text(text)
    for name in stale:
        print(f"{'stale' if args.check else 'wrote'} {name}")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
