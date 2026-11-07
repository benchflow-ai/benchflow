"""Check that every built distribution ships both pinned bfx binaries.

The release workflows run this after ``uv build``: each wheel and sdist in
the folder must hold ``benchflow/_bfx/bin/bfx-<arch>`` for every
architecture in ``src/benchflow/_bfx/pins.py``, as a regular executable file
whose SHA-256 is the pin. Anything else fails the release before it is
published. Run with ``--no-project`` (it imports nothing from benchflow).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / "src" / "benchflow" / "_bfx" / "pins.py"


def read_pins(path: Path = PINS) -> dict[str, str]:
    """``SHA256`` from pins.py, read without importing it."""
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SHA256"
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict) and value:
                return {str(k): str(v) for k, v in value.items()}
    raise ValueError(f"{path} has no SHA256 table")


def _check(found: dict[str, tuple[bytes, int]], pins: dict[str, str]) -> list[str]:
    problems = []
    for arch, want in sorted(pins.items()):
        name = f"bfx-{arch}"
        if name not in found:
            problems.append(f"{name} is missing")
            continue
        data, mode = found[name]
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            problems.append(f"{name} has SHA-256 {got}, not the pinned {want}")
        if not stat.S_ISREG(mode) or mode & 0o111 != 0o111:
            problems.append(f"{name} is not a regular executable file ({mode:o})")
    return problems


def wheel_problems(path: Path, pins: dict[str, str]) -> list[str]:
    found: dict[str, tuple[bytes, int]] = {}
    with zipfile.ZipFile(path) as wheel:
        for info in wheel.infolist():
            parent, _, name = info.filename.rpartition("/")
            if parent == "benchflow/_bfx/bin":
                mode = info.external_attr >> 16
                found[name] = (wheel.read(info), mode or stat.S_IFREG)
    return _check(found, pins)


def sdist_problems(path: Path, pins: dict[str, str]) -> list[str]:
    found: dict[str, tuple[bytes, int]] = {}
    with tarfile.open(path) as sdist:
        for member in sdist.getmembers():
            parts = member.name.split("/")
            if parts[1:-1] == ["src", "benchflow", "_bfx", "bin"]:
                f = sdist.extractfile(member) if member.isreg() else None
                mode = member.mode | (stat.S_IFREG if member.isreg() else 0)
                found[parts[-1]] = (f.read() if f else b"", mode)
    return _check(found, pins)


def dist_problems(folder: Path, pins: dict[str, str]) -> dict[str, list[str]]:
    """Problems per distribution file; a folder without any is a problem."""
    out: dict[str, list[str]] = {}
    for path in sorted(folder.iterdir()):
        if path.name.endswith(".whl"):
            out[path.name] = wheel_problems(path, pins)
        elif path.name.endswith(".tar.gz"):
            out[path.name] = sdist_problems(path, pins)
    if not out:
        out[str(folder)] = ["no wheel or sdist"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args(argv)
    pins = read_pins()
    failed = False
    for name, problems in dist_problems(args.dist, pins).items():
        for problem in problems:
            print(f"::error::{name}: {problem}")
            failed = True
        if not problems:
            print(f"{name}: bfx " + ", ".join(sorted(pins)) + " match the pins")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
