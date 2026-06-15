"""Hard rule: the dependency cooldown — never resolve to a release younger than
a 7-day window.

`[tool.uv] exclude-newer` in pyproject caps every `uv lock` at a fixed timestamp.
This test fails if that timestamp is ever within the last 7 days, so the lock can
never include a brand-new (and thus less-vetted) release. When you intentionally
take dependency updates, advance `exclude-newer` to a date still >= 7 days in the
past. A genuinely urgent (e.g. security) fix younger than the cap is allowed only
via an explicit, commented `[tool.uv.exclude-newer-package]` override.
"""

from __future__ import annotations

import datetime
import tomllib
from pathlib import Path

COOLDOWN_DAYS = 7
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_uv_exclude_newer_enforces_dependency_cooldown() -> None:
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
        f"cooldown requires it to be >= {COOLDOWN_DAYS} days in the past. Bump it "
        f"to an older date (>= {COOLDOWN_DAYS}d ago) when taking dependency updates."
    )
