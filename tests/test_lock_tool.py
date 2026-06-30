"""Unit tests for the dependency-cooldown re-lock helper (`tools/lock.py`).

These cover the pure date/text transforms and the lock-freshness audit against
synthetic lock text; `uv lock` itself is not invoked.
"""

from __future__ import annotations

import datetime
import tomllib

import pytest

from tools import lock as lock_tool
from tools.lock import (
    COOLDOWN_DAYS,
    LockError,
    compute_cooldown_cutoff,
    find_cooldown_violations,
    find_expired_cooldown_overrides,
    main,
    newest_upload_times,
    rewrite_exclude_newer,
)

_NOW = datetime.datetime(2026, 6, 15, 12, 0, tzinfo=datetime.UTC)

# floor (now - 7d) = 2026-06-08T12:00Z: old-pkg is older, fresh-pkg is younger.
_SYNTH_LOCK = """
[options]
exclude-newer = "2026-06-08T00:00:00Z"

[[package]]
name = "old-pkg"
version = "1.0.0"
sdist = { url = "https://example/old.tar.gz", upload-time = "2026-05-01T00:00:00Z" }

[[package]]
name = "fresh-pkg"
version = "2.0.0"
sdist = { url = "https://example/fresh.tar.gz", upload-time = "2026-06-13T00:00:00Z" }
wheels = [
    { url = "https://example/fresh.whl", upload-time = "2026-06-14T08:00:00Z" },
]

[[package]]
name = "git-pkg"
version = "0.1.0"
source = { git = "https://example/repo.git" }
"""


def test_cutoff_trails_today_by_cooldown_window_at_midnight() -> None:
    now = datetime.datetime(2026, 6, 15, 9, 30, tzinfo=datetime.UTC)
    assert compute_cooldown_cutoff(now) == "2026-06-08T00:00:00Z"


def test_cutoff_is_idempotent_within_a_day() -> None:
    morning = datetime.datetime(2026, 6, 15, 0, 1, tzinfo=datetime.UTC)
    night = datetime.datetime(2026, 6, 15, 23, 59, tzinfo=datetime.UTC)
    assert compute_cooldown_cutoff(morning) == compute_cooldown_cutoff(night)


def test_cutoff_normalizes_non_utc_input() -> None:
    # 2026-06-15 02:00 in UTC+9 is still 2026-06-14 17:00 UTC -> minus 7d.
    tz = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime(2026, 6, 15, 2, 0, tzinfo=tz)
    assert compute_cooldown_cutoff(now) == "2026-06-07T00:00:00Z"


def test_cutoff_default_window_is_the_cooldown_constant() -> None:
    now = datetime.datetime(2026, 6, 15, 12, 0, tzinfo=datetime.UTC)
    expected = (now - datetime.timedelta(days=COOLDOWN_DAYS)).date()
    assert compute_cooldown_cutoff(now).startswith(expected.isoformat())


def test_cutoff_rejects_negative_window() -> None:
    now = datetime.datetime(2026, 6, 15, tzinfo=datetime.UTC)
    with pytest.raises(LockError):
        compute_cooldown_cutoff(now, cooldown_days=-1)


def test_rewrite_replaces_only_the_exclude_newer_value() -> None:
    text = (
        "[tool.uv]\n"
        'exclude-newer = "2026-06-01T00:00:00Z"\n'
        "\n"
        "[tool.uv.exclude-newer-package]\n"
        'litellm = "2026-06-14T00:00:00Z"\n'
    )
    out = rewrite_exclude_newer(text, "2026-06-08T00:00:00Z")
    assert 'exclude-newer = "2026-06-08T00:00:00Z"' in out
    # The per-package override table is untouched.
    assert 'litellm = "2026-06-14T00:00:00Z"' in out


def test_rewrite_requires_exactly_one_assignment() -> None:
    with pytest.raises(LockError):
        rewrite_exclude_newer("[tool.uv]\n", "2026-06-08T00:00:00Z")


def test_newest_upload_time_picks_latest_artifact_and_skips_sourceless() -> None:
    newest = newest_upload_times(tomllib.loads(_SYNTH_LOCK))
    # The wheel (08:00) beats the sdist (00:00) for the same package.
    assert newest["fresh-pkg"][1] == datetime.datetime(
        2026, 6, 14, 8, 0, tzinfo=datetime.UTC
    )
    # A git-sourced package has no upload-time and is omitted entirely.
    assert "git-pkg" not in newest


def test_find_cooldown_violations_flags_only_young_unexempted_packages() -> None:
    violations = find_cooldown_violations(_SYNTH_LOCK, exempt=set(), now=_NOW)
    assert [name for name, _, _ in violations] == ["fresh-pkg"]


def test_find_cooldown_violations_honors_exemptions() -> None:
    violations = find_cooldown_violations(_SYNTH_LOCK, exempt={"fresh-pkg"}, now=_NOW)
    assert violations == []


def test_find_cooldown_violations_normalizes_exemption_names() -> None:
    """Guards the fix from PR #788 so overrides are not spelling-sensitive."""
    violations = find_cooldown_violations(_SYNTH_LOCK, exempt={"Fresh_Pkg"}, now=_NOW)
    assert violations == []


def test_find_cooldown_violations_only_grows_more_lenient_over_time() -> None:
    # The same lock, evaluated a month later: fresh-pkg has aged past the window.
    later = _NOW + datetime.timedelta(days=30)
    assert find_cooldown_violations(_SYNTH_LOCK, exempt=set(), now=later) == []


def test_expired_cooldown_overrides_flags_aged_out_entries() -> None:
    """Guards the PR #788 override escape hatch from becoming a package allowlist."""
    overrides = {
        "old-pkg": "2026-06-08T00:00:00Z",
        "fresh-pkg": "2026-06-14T00:00:00Z",
    }
    expired = find_expired_cooldown_overrides(overrides, now=_NOW)
    assert [(name, raw) for name, raw, _ in expired] == [
        ("old-pkg", "2026-06-08T00:00:00Z")
    ]


def test_expired_cooldown_overrides_rejects_invalid_timestamp() -> None:
    """Guards the PR #788 override gate against unclear TOML diagnostics."""
    with pytest.raises(
        LockError,
        match=r"\[tool\.uv\.exclude-newer-package\] 'bad-pkg' must be an ISO timestamp",
    ):
        find_expired_cooldown_overrides({"bad-pkg": "not-a-date"}, now=_NOW)


def test_main_wraps_pyproject_read_oserror(tmp_path, capsys) -> None:
    """Guards the PR #788 lock helper from leaking raw OSError tracebacks."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--pyproject", str(tmp_path), "--no-lock"])

    assert exc_info.value.code == 1
    assert "Could not read" in capsys.readouterr().err


def test_run_uv_lock_wraps_oserror(monkeypatch, tmp_path) -> None:
    """Guards the PR #788 lock helper from leaking subprocess OSError."""

    def raise_oserror(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(lock_tool.subprocess, "run", raise_oserror)

    with pytest.raises(LockError, match="Could not run `uv lock`"):
        lock_tool._run_uv_lock(tmp_path / "pyproject.toml")
