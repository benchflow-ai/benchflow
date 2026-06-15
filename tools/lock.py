"""Refresh the uv dependency-cooldown cutoff to `now - COOLDOWN_DAYS`, then lock.

The repo enforces a hard dependency cooldown: `uv lock` must never resolve to a
release younger than `COOLDOWN_DAYS` (guarded by `tests/test_dep_cooldown.py`).
The cutoff lives in `[tool.uv] exclude-newer` in pyproject.toml as a *static*
ISO timestamp — and it has to stay static, because CI runs `uv sync --locked` /
`uv export --locked` and a committed lock can only be deterministic against a
fixed cutoff. A literally-live "now - 7d" value in pyproject would change on
every CI run and break `--locked`.

This helper is how that static value gets set without hand-typing a date: it
computes midnight UTC of `(today - COOLDOWN_DAYS)` and writes it into
`exclude-newer`, then re-locks. Run it whenever you intentionally take
dependency updates — the date "rolls" forward to the newest still-vetted day,
the committed pyproject date + `uv.lock` remain a fixed reviewable artifact, and
the cooldown invariant holds by construction.

    python tools/lock.py            # set cutoff to now-7d, then `uv lock`
    python tools/lock.py --no-lock  # only rewrite the cutoff date
    python tools/lock.py --check    # print the cutoff it would write; change nothing

A genuinely urgent fix younger than the cutoff stays a per-package exception in
`[tool.uv.exclude-newer-package]`; this helper never touches that table.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# Single source of truth for the cooldown window. `tests/test_dep_cooldown.py`
# imports this so the CI gate and this writer can never drift apart.
COOLDOWN_DAYS = 7

# Matches the `[tool.uv] exclude-newer = "<ts>"` assignment only. The sibling
# `[tool.uv.exclude-newer-package]` is a table *header*, never a `key =` line,
# so it can't match here.
_EXCLUDE_NEWER_RE = re.compile(
    r'^(?P<prefix>exclude-newer\s*=\s*)"[^"]*"',
    re.MULTILINE,
)


class LockError(RuntimeError):
    """Raised when the cooldown cutoff cannot be refreshed."""


def compute_cooldown_cutoff(
    now: datetime.datetime, cooldown_days: int = COOLDOWN_DAYS
) -> str:
    """Return midnight-UTC of `(now - cooldown_days)` as a uv `exclude-newer` value.

    Truncating to midnight makes a same-day re-run idempotent (no lock churn),
    and pins the cutoff to a whole day for easy review.
    """
    if cooldown_days < 0:
        raise LockError(f"cooldown_days must be non-negative, got {cooldown_days}.")
    cutoff_date = (
        now.astimezone(datetime.UTC) - datetime.timedelta(days=cooldown_days)
    ).date()
    return f"{cutoff_date.isoformat()}T00:00:00Z"


def rewrite_exclude_newer(pyproject_text: str, cutoff: str) -> str:
    """Return `pyproject_text` with `[tool.uv] exclude-newer` set to `cutoff`.

    Raises if the key is missing or appears more than once — we only ever want
    to touch a single, known assignment.
    """
    matches = _EXCLUDE_NEWER_RE.findall(pyproject_text)
    if len(matches) != 1:
        raise LockError(
            'Expected exactly one `[tool.uv] exclude-newer = "..."` line in '
            f"pyproject.toml, found {len(matches)}."
        )
    return _EXCLUDE_NEWER_RE.sub(rf'\g<prefix>"{cutoff}"', pyproject_text)


def _run_uv_lock(pyproject_path: Path) -> int:
    try:
        # Fixed argv, no shell — uv is resolved from PATH.
        completed = subprocess.run(
            ["uv", "lock"],
            cwd=pyproject_path.resolve().parent,
            check=False,
        )
    except FileNotFoundError as exc:
        raise LockError(
            "`uv` was not found on PATH; install uv or pass --no-lock to only "
            "rewrite the cutoff date."
        ) from exc
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml (default: ./pyproject.toml).",
    )
    parser.add_argument(
        "--cooldown-days",
        type=int,
        default=COOLDOWN_DAYS,
        help=f"Days the cutoff trails today (default: {COOLDOWN_DAYS}).",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Rewrite the cutoff date only; skip `uv lock`.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the cutoff that would be written and exit; change nothing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cutoff = compute_cooldown_cutoff(
            datetime.datetime.now(datetime.UTC), args.cooldown_days
        )
        if args.check:
            print(cutoff)
            return 0

        original = args.pyproject.read_text(encoding="utf-8")
        updated = rewrite_exclude_newer(original, cutoff)
        if updated != original:
            args.pyproject.write_text(updated, encoding="utf-8")
            print(f"Set [tool.uv] exclude-newer = {cutoff!r} in {args.pyproject}.")
        else:
            print(f"[tool.uv] exclude-newer already {cutoff!r}; no change.")

        if args.no_lock:
            return 0
        return _run_uv_lock(args.pyproject)
    except LockError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
