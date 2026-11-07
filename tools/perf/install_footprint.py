"""B3: a cold ``uv tool install`` of each arm's wheel: time, packages, sizes.

Each run installs the wheel with an empty ``UV_CACHE_DIR`` and empty tool
folders made for that run, then records the wall time, the number of
installed distributions, and the bytes of the cache and of the tool's venv.
The managed Python comes from ``UV_PYTHON_INSTALL_DIR`` as set by the caller
(not cold: install it once beforehand, as the README says), so the numbers are
BenchFlow's install, not the interpreter download. Arms alternate run by run.

    uv run python -m tools.perf.install_footprint \\
        --wheel base=2ea2ad00:$W/wheels/base/benchflow-0.8.0-py3-none-any.whl \\
        --wheel tip=abc12345:$W/wheels/tip/benchflow-0.8.0-py3-none-any.whl \\
        --runs 3 --work $W/perf/b3-work --out $W/perf/b3.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tools.perf._common import (
    append_rows,
    dir_bytes,
    output,
    provenance,
    report_provenance,
)

FIELDS = (
    "arm",
    "commit",
    "round",
    "seconds",
    "exit_code",
    "packages",
    "cache_bytes",
    "venv_bytes",
)


@dataclass(frozen=True)
class Wheel:
    name: str
    commit: str
    path: Path


def parse_wheel(spec: str) -> Wheel:
    """``NAME=COMMIT:WHEEL``."""
    name, eq, rest = spec.partition("=")
    commit, colon, path = rest.partition(":")
    if not (eq and colon and name and commit and path):
        raise argparse.ArgumentTypeError(f"wheel {spec!r}: expected NAME=COMMIT:WHEEL")
    return Wheel(name, commit, Path(path).expanduser().resolve())


def count_distributions(venv: Path) -> int:
    return sum(1 for _ in venv.glob("lib/python*/site-packages/*.dist-info"))


def install_once(wheel: Wheel, work: Path, python: str) -> dict[str, object]:
    run_dir = Path(tempfile.mkdtemp(prefix=f"b3-{wheel.name}-", dir=work))
    try:
        cache, tools, bins = run_dir / "cache", run_dir / "tools", run_dir / "bin"
        env = dict(os.environ)
        env.update(
            UV_CACHE_DIR=str(cache),
            UV_TOOL_DIR=str(tools),
            UV_TOOL_BIN_DIR=str(bins),
            UV_NO_CONFIG="1",
        )
        started = time.perf_counter()
        done = subprocess.run(
            ["uv", "tool", "install", "--python", python, str(wheel.path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            check=False,
        )
        seconds = time.perf_counter() - started
        venv = tools / "benchflow"
        return {
            "seconds": seconds,
            "exit_code": done.returncode,
            "packages": count_distributions(venv) if venv.is_dir() else None,
            "cache_bytes": dir_bytes(cache) if cache.is_dir() else None,
            "venv_bytes": dir_bytes(venv) if venv.is_dir() else None,
        }
    finally:
        # This run's own scratch folder, made above.
        shutil.rmtree(run_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wheel", type=parse_wheel, action="append", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    wheels: list[Wheel] = args.wheel
    for wheel in wheels:
        if not wheel.path.is_file():
            raise SystemExit(f"wheel {wheel.name}: {wheel.path} does not exist")
    args.work.mkdir(parents=True, exist_ok=True)
    report_provenance(
        provenance(
            "install_footprint",
            extra={
                "wheels": [
                    {
                        "arm": w.name,
                        "commit": w.commit,
                        "wheel": str(w.path),
                        "sha256": output(["sha256sum", w.path]).split(" ")[0],
                    }
                    for w in wheels
                ],
                "python": args.python,
                "uv_python_install_dir": os.environ.get("UV_PYTHON_INSTALL_DIR", ""),
            },
        ),
        args.out,
    )
    for round_index in range(args.runs):
        shift = round_index % len(wheels)
        for wheel in [*wheels[shift:], *wheels[:shift]]:
            row = install_once(wheel, args.work, args.python)
            row.update(arm=wheel.name, commit=wheel.commit, round=round_index)
            append_rows(args.out, FIELDS, [row])
            print(f"{wheel.name} round {round_index}: {row['seconds']:.1f} s")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
