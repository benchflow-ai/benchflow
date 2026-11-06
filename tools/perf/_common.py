"""Pieces every perf script shares: arms, provenance and CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# An arm's name is a folder name in the scripts' work folders: no separators,
# no leading dot (so never ``.`` or ``..``).
ARM_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class Arm:
    """One build under test: a name, the commit it was built from, a venv."""

    name: str
    commit: str
    venv: Path

    def exe(self, name: str) -> Path:
        return self.venv / "bin" / name


def parse_arm(spec: str) -> Arm:
    """``NAME=COMMIT:VENV``, e.g. ``base=2ea2ad00:/home/me/w/T0/venv-base``."""
    name, eq, rest = spec.partition("=")
    commit, colon, venv = rest.partition(":")
    if not (eq and colon and name and commit and venv):
        raise argparse.ArgumentTypeError(
            f"arm {spec!r}: expected NAME=COMMIT:VENV (the venv holds that commit's bench)"
        )
    if not ARM_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"arm {spec!r}: the name may hold only letters, digits, _ . and - "
            "and may not start with a dot"
        )
    return Arm(name=name, commit=commit, venv=Path(venv).expanduser().resolve())


def check_arms(arms: Sequence[Arm], *, need: str = "bench") -> None:
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"arm names must differ: {names}")
    for arm in arms:
        if not arm.exe(need).exists():
            raise SystemExit(f"arm {arm.name}: {arm.exe(need)} does not exist")


def interleaved(arms: Sequence[Arm], round_index: int) -> list[Arm]:
    """Arms in a rotating order, so no arm always runs first in a round."""
    shift = round_index % len(arms)
    return [*arms[shift:], *arms[:shift]]


def output(args: Sequence[str | os.PathLike[str]], **kwargs: Any) -> str:
    """A command's stdout, stripped; empty when it cannot run."""
    try:
        done = subprocess.run(
            [str(a) for a in args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def image_digest(ref: str) -> str:
    """The repo digest of a local image (its image id when it has none)."""
    return output(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}",
            ref,
        ]
    )


def host_facts() -> dict[str, Any]:
    cpu = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "host": platform.node(),
        "cpus": os.cpu_count(),
        "cpu_model": cpu,
        "os": platform.platform(),
        "tool_python": sys.version.split()[0],
        "uv": output(["uv", "--version"]) if shutil.which("uv") else "",
        "docker": output(["docker", "version", "--format", "{{.Server.Version}}"]),
        "tool_commit": output(["git", "-C", REPO, "rev-parse", "HEAD"]),
    }


def arm_facts(arm: Arm) -> dict[str, Any]:
    python = arm.exe("python")
    return {
        "arm": arm.name,
        "commit": arm.commit,
        "venv": str(arm.venv),
        "benchflow": output(
            [
                python,
                "-I",
                "-c",
                "import importlib.metadata as m; print(m.version('benchflow'))",
            ]
        ),
        "python": output(
            [python, "-I", "-c", "import sys; print(sys.version.split()[0])"]
        ),
    }


def provenance(
    script: str,
    *,
    arms: Iterable[Arm] = (),
    images: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What a run used: commits, versions, image digests, the machine."""
    return {
        "script": script,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        **host_facts(),
        "arms": [arm_facts(arm) for arm in arms],
        "images": dict(images or {}),
        **dict(extra or {}),
    }


def report_provenance(record: Mapping[str, Any], out_csv: Path) -> None:
    """Print the record and keep it beside the CSV (one JSON line per run)."""
    line = json.dumps(record, sort_keys=True)
    print(f"# provenance {line}", flush=True)
    sidecar = out_csv.with_name(out_csv.name + ".provenance.jsonl")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("a") as handle:
        handle.write(line + "\n")


def append_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Append rows to a CSV, writing the header when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="raise"
        )
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value == value else ""
    return value


def dir_bytes(path: Path) -> int:
    """Bytes of every regular file under ``path`` (symlinks not followed)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file = Path(root) / name
            if not file.is_symlink():
                total += file.stat().st_size
    return total
