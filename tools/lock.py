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
import tomllib
from collections.abc import Mapping, Sequence
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
_PACKAGE_NAME_RE = re.compile(r"[-_.]+")


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


def normalize_package_name(name: str) -> str:
    """Return the package-name shape uv/PyPI use for comparisons."""
    return _PACKAGE_NAME_RE.sub("-", name).lower()


def _parse_lock_timestamp(raw: str) -> datetime.datetime:
    ts = datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.UTC)
    return ts


def _parse_override_timestamp(package: str, raw: object) -> datetime.datetime:
    if not isinstance(raw, str):
        raise LockError(
            "[tool.uv.exclude-newer-package] "
            f"{package!r} must be an ISO timestamp string, got {raw!r}."
        )
    try:
        return _parse_lock_timestamp(raw)
    except ValueError as exc:
        raise LockError(
            "[tool.uv.exclude-newer-package] "
            f"{package!r} must be an ISO timestamp, got {raw!r}."
        ) from exc


def newest_upload_times(
    lock: dict,
) -> dict[str, tuple[str, datetime.datetime]]:
    """Map each locked package to `(version, newest sdist/wheel upload time)`.

    Packages with no registry `upload-time` (git/local/editable sources) are
    skipped — they have no PyPI release subject to the cooldown.
    """
    newest: dict[str, tuple[str, datetime.datetime]] = {}
    for pkg in lock.get("package", []):
        name = pkg.get("name")
        if not name:
            continue
        version = str(pkg.get("version", ""))
        raw_times: list[str] = []
        sdist = pkg.get("sdist")
        if isinstance(sdist, dict) and sdist.get("upload-time"):
            raw_times.append(sdist["upload-time"])
        for wheel in pkg.get("wheels") or []:
            if isinstance(wheel, dict) and wheel.get("upload-time"):
                raw_times.append(wheel["upload-time"])
        for raw in raw_times:
            ts = _parse_lock_timestamp(raw)
            if name not in newest or ts > newest[name][1]:
                newest[name] = (version, ts)
    return newest


def find_cooldown_violations(
    lock_text: str,
    exempt: set[str],
    now: datetime.datetime,
    cooldown_days: int = COOLDOWN_DAYS,
) -> list[tuple[str, str, datetime.datetime]]:
    """Return locked packages published less than `cooldown_days` before `now`.

    This is the dynamic half of the cooldown: it reads the `upload-time` uv bakes
    into `uv.lock` and compares each resolved package against `now - cooldown_days`
    — so the window tracks the current date with no stored date to trust and no
    network calls. `exempt` is the set of package names that carry a documented
    `[tool.uv.exclude-newer-package]` override (allowed to be younger).

    Returns `(name, version, upload_time)` tuples, newest first. Never flags an
    unchanged lock over time: upload times are immutable, so packages only age
    past the window.
    """
    floor = now.astimezone(datetime.UTC) - datetime.timedelta(days=cooldown_days)
    lock = tomllib.loads(lock_text)
    normalized_exempt = {normalize_package_name(name) for name in exempt}
    violations = [
        (name, version, ts)
        for name, (version, ts) in newest_upload_times(lock).items()
        if ts > floor and normalize_package_name(name) not in normalized_exempt
    ]
    violations.sort(key=lambda item: item[2], reverse=True)
    return violations


def find_expired_cooldown_overrides(
    overrides: Mapping[str, object],
    now: datetime.datetime,
    cooldown_days: int = COOLDOWN_DAYS,
) -> list[tuple[str, str, datetime.datetime]]:
    """Return package overrides whose temporary cutoff is no longer needed.

    `[tool.uv.exclude-newer-package]` is the explicit escape hatch for a package
    that must resolve newer than the global cooldown cap. Once the override's
    timestamp is older than the active cooldown floor, the global cap can cover
    the package and the override must be removed instead of becoming a permanent
    package allowlist.
    """
    floor = now.astimezone(datetime.UTC) - datetime.timedelta(days=cooldown_days)
    expired: list[tuple[str, str, datetime.datetime]] = []
    for package, raw_cutoff in overrides.items():
        cutoff = _parse_override_timestamp(package, raw_cutoff)
        if cutoff <= floor:
            expired.append((package, str(raw_cutoff), cutoff))
    expired.sort(key=lambda item: normalize_package_name(item[0]))
    return expired


def _read_pyproject(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LockError(f"Could not read {path}: {exc}") from exc


def _write_pyproject(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise LockError(f"Could not write {path}: {exc}") from exc


def _run_uv_lock(pyproject_path: Path) -> int:
    cwd = pyproject_path.resolve().parent
    try:
        # Fixed argv, no shell — uv is resolved from PATH.
        completed = subprocess.run(
            ["uv", "lock"],
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError as exc:
        raise LockError(
            "`uv` was not found on PATH; install uv or pass --no-lock to only "
            "rewrite the cutoff date."
        ) from exc
    except OSError as exc:
        raise LockError(f"Could not run `uv lock` in {cwd}: {exc}") from exc
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

        original = _read_pyproject(args.pyproject)
        updated = rewrite_exclude_newer(original, cutoff)
        if updated != original:
            _write_pyproject(args.pyproject, updated)
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
