"""B2: CLI start-up time of ``bench`` per arm, with the interpreter's floor.

For each arm (a commit's venv) it times, after warm-ups, interleaved run by
run: ``bench --version``, ``bench --help``, ``bench eval run --help``,
``bench agent list``, and the floor in the same venv: a bare ``python -c pass``
and ``python -c 'import typer'``. Once per arm it also saves
``python -X importtime -c 'import benchflow.cli.main'`` beside the CSV.

    uv run python -m tools.perf.cli_startup \\
        --arm base=2ea2ad00:$W/venv-base --arm tip=abc12345:$W/venv-tip \\
        --runs 30 --warmups 3 --out $W/perf/b2.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from tools.perf._common import (
    Arm,
    append_rows,
    check_arms,
    interleaved,
    parse_arm,
    provenance,
    report_provenance,
)

FIELDS = ("arm", "commit", "command", "round", "warmup", "seconds", "exit_code")

# (label, executable in the arm's venv, arguments)
COMMANDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("bench --version", "bench", ("--version",)),
    ("bench --help", "bench", ("--help",)),
    ("bench eval run --help", "bench", ("eval", "run", "--help")),
    ("bench agent list", "bench", ("agent", "list")),
    ("python -c pass", "python", ("-c", "pass")),
    ("python -c import typer", "python", ("-c", "import typer")),
)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    # No update check or telemetry on the timed path.
    env["BENCHFLOW_SKIP_UPDATE_CHECK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def time_once(argv: list[str], env: dict[str, str]) -> tuple[float, int]:
    started = time.perf_counter()
    done = subprocess.run(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        check=False,
    )
    return time.perf_counter() - started, done.returncode


def save_importtime(arm: Arm, out: Path, env: dict[str, str]) -> Path:
    path = out.with_name(f"{out.stem}.importtime-{arm.name}.txt")
    done = subprocess.run(
        [str(arm.exe("python")), "-X", "importtime", "-c", "import benchflow.cli.main"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    path.write_text(done.stderr)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", type=parse_arm, action="append", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms: list[Arm] = args.arm
    check_arms(arms)
    env = _env()
    report_provenance(
        provenance(
            "cli_startup",
            arms=arms,
            extra={"runs": args.runs, "warmups": args.warmups},
        ),
        args.out,
    )
    for arm in arms:
        print(f"importtime {arm.name}: {save_importtime(arm, args.out, env)}")
    for round_index in range(args.warmups + args.runs):
        rows = []
        for label, exe, extra in COMMANDS:
            for arm in interleaved(arms, round_index):
                seconds, code = time_once([str(arm.exe(exe)), *extra], env)
                rows.append(
                    {
                        "arm": arm.name,
                        "commit": arm.commit,
                        "command": label,
                        "round": round_index,
                        "warmup": int(round_index < args.warmups),
                        "seconds": seconds,
                        "exit_code": code,
                    }
                )
        append_rows(args.out, FIELDS, rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
