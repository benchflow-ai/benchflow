"""Hard rule: the dependency cooldown — never resolve to a release younger than
a 7-day window.

Two layers, both evaluated against *today*:

1. `test_uv_exclude_newer_caps_resolution` — `[tool.uv] exclude-newer` is the
   static cutoff `uv lock` applies at resolve time; it must be >= 7 days old so
   the default resolution can't reach a brand-new release. Set it with
   `python tools/lock.py` (computes `now - 7d` and re-locks).
2. `test_locked_packages_respect_cooldown` — the dynamic half: it reads the
   `upload-time` uv bakes into `uv.lock` and fails if any *resolved* package is
   younger than `now - 7d`. This tracks the current date with no stored date to
   trust and no network calls.

A genuinely urgent (e.g. security) fix younger than the window is allowed only
via an explicit, commented `[tool.uv.exclude-newer-package]` override, which
exempts that package from layer 2.
"""

from __future__ import annotations

import datetime
import tomllib
from pathlib import Path

from tools.lock import (
    COOLDOWN_DAYS,
    find_cooldown_violations,
    find_expired_cooldown_overrides,
)

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_LOCK = Path(__file__).resolve().parents[1] / "uv.lock"


def test_uv_exclude_newer_caps_resolution() -> None:
    cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    raw = cfg.get("tool", {}).get("uv", {}).get("exclude-newer")
    assert raw, (
        "[tool.uv] exclude-newer is missing from pyproject.toml — it enforces the "
        f"{COOLDOWN_DAYS}-day dependency cooldown and must be set."
    )
    cutoff = datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=datetime.UTC)
    age = datetime.datetime.now(datetime.UTC) - cutoff
    assert age >= datetime.timedelta(days=COOLDOWN_DAYS), (
        f"[tool.uv] exclude-newer ({raw}) is only {age.days}d old; the dependency "
        f"cooldown requires it to be >= {COOLDOWN_DAYS} days in the past. Re-lock "
        "with `python tools/lock.py` when taking dependency updates."
    )


def test_locked_packages_respect_cooldown() -> None:
    cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    exempt = set(cfg.get("tool", {}).get("uv", {}).get("exclude-newer-package", {}))
    violations = find_cooldown_violations(
        _LOCK.read_text(encoding="utf-8"),
        exempt=exempt,
        now=datetime.datetime.now(datetime.UTC),
    )
    assert not violations, (
        f"uv.lock resolves packages younger than the {COOLDOWN_DAYS}-day cooldown "
        "without a documented [tool.uv.exclude-newer-package] override:\n"
        + "\n".join(f"  {n} {v} (uploaded {ts:%Y-%m-%d})" for n, v, ts in violations)
        + "\nRe-lock with `python tools/lock.py`, or add a commented override."
    )


def test_exclude_newer_package_overrides_expire_with_cooldown() -> None:
    cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    overrides = cfg.get("tool", {}).get("uv", {}).get("exclude-newer-package", {})
    now = datetime.datetime.now(datetime.UTC)
    expired = find_expired_cooldown_overrides(overrides, now=now)
    floor = now - datetime.timedelta(days=COOLDOWN_DAYS)
    assert not expired, (
        "[tool.uv.exclude-newer-package] overrides are temporary security/urgent "
        f"escape hatches and must be removed once their cutoff is older than the "
        f"{COOLDOWN_DAYS}-day cooldown floor ({floor:%Y-%m-%dT%H:%M:%SZ}):\n"
        + "\n".join(
            f"  {name} = {raw!r} (override cutoff {cutoff:%Y-%m-%dT%H:%M:%SZ})"
            for name, raw, cutoff in expired
        )
        + "\nRemove the stale override and re-lock with `python tools/lock.py`, "
        "or keep an override only for a still-fresh security/urgent pin."
    )
