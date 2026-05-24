"""Guards dashboard ``data.json`` from leaking absolute host paths (#408).

The published ``data.json`` is served from Vercel and checked into the repo,
so any absolute host path written there leaks local usernames, worktree
names, temp-dir layout, and private runner paths. Issue #408 documents the
specific evidence: ``jobs.source.path`` and per-artifact ``path`` fields
were rendered as raw ``str(Path)`` outputs from the producer machine.

These tests pin two complementary properties:

1. A generated payload with fixtures placed under ``tmp_path`` carries no
   ``/Users/``, ``/home/``, or ``/private/`` prefixes anywhere — neither
   on ``jobs.source.path`` nor on any nested artifact ``path``.
2. Paths under HOME are scrubbed to ``~/`` form, which still round-trips
   through ``Path(...).expanduser().resolve()`` so the dashboard's
   "remembered jobs root" feature keeps working when an operator points
   at a directory inside their home tree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dashboard import generate


def _walk_strings(value: object):
    """Yield every string anywhere inside a nested JSON-like structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_strings(v)


def _write_rollout(jobs_root: Path) -> Path:
    rollout = jobs_root / "2026-05-22__01-30-00" / "task-a__abc123"
    (rollout / "verifier").mkdir(parents=True)
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "codex",
                "rewards": {"reward": 1.0},
                "timing": {},
            }
        )
    )
    (rollout / "config.json").write_text(json.dumps({"agent": "codex"}))
    (rollout / "verifier" / "reward.txt").write_text("1.0\n")
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "step"}) + "\n"
    )
    return rollout


def test_collect_jobs_strips_host_prefixes_from_published_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``collect_jobs()`` must not embed absolute host paths anywhere."""
    jobs_root = tmp_path / "previous-worktree" / "jobs"
    _write_rollout(jobs_root)

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs_root))

    payload = generate.collect_jobs()

    forbidden_prefixes = ("/Users/", "/home/", "/private/")
    for s in _walk_strings(payload):
        for prefix in forbidden_prefixes:
            assert not s.startswith(prefix), (
                f"data.json string starts with host prefix {prefix!r}: {s!r}"
            )
        # The fixture path itself must never appear verbatim. ``tmp_path`` on
        # macOS resolves under ``/private/var/folders/...``, on Linux under
        # ``/tmp/...`` — either way, the literal string would be a leak.
        assert str(tmp_path) not in s, (
            f"data.json string echoes the host tmp dir verbatim: {s!r}"
        )


def test_collect_jobs_scrubs_home_dir_to_tilde_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A jobs root under HOME is rendered as ``~/…`` and round-trips."""
    fake_home = tmp_path / "home" / "operator"
    jobs_root = fake_home / "work" / "jobs"
    _write_rollout(jobs_root)
    monkeypatch.setenv("HOME", str(fake_home))

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs_root))

    payload = generate.collect_jobs()

    source_path = payload["source"]["path"]
    assert source_path.startswith("~/"), (
        f"HOME-relative source.path should start with ~/: {source_path!r}"
    )
    assert str(fake_home) not in source_path

    # The scrubbed ``~/…`` form must round-trip back to the original absolute
    # path via the same ``expanduser().resolve()`` call the dashboard already
    # uses in ``remembered_jobs_root``.
    recovered = Path(source_path).expanduser().resolve()
    assert recovered == jobs_root.resolve()


def test_collect_jobs_uses_repo_relative_path_for_local_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the jobs root sits under the repo, ``path`` is relative-to-ROOT."""
    repo = tmp_path / "repo"
    jobs_root = repo / "jobs"
    _write_rollout(jobs_root)
    monkeypatch.setattr(generate, "ROOT", repo)
    monkeypatch.setattr(generate, "DASH", repo / "dashboard")
    monkeypatch.setattr(generate, "OUT", repo / "dashboard" / "data.json")

    monkeypatch.setenv(generate.JOBS_ROOT_ENV, str(jobs_root))

    payload = generate.collect_jobs()

    # Relative to ROOT — no absolute prefix, just ``jobs``.
    assert payload["source"]["path"] == "jobs"
    # Artifacts are likewise relative-to-ROOT (no absolute host prefix).
    artifact_paths = [
        artifact["path"]
        for group in payload["groups"]
        for run in group["runs"]
        for task in run["tasks"]
        for artifact in task["artifacts"]
    ]
    assert artifact_paths, "fixture should have produced at least one artifact"
    for p in artifact_paths:
        assert not os.path.isabs(p), f"artifact path is absolute: {p!r}"
        assert p.startswith("jobs/"), (
            f"artifact path should be repo-relative: {p!r}"
        )


def test_scrub_host_path_falls_back_to_basename_outside_root_and_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths outside both ROOT and HOME drop the host layout entirely."""
    repo = tmp_path / "repo"
    fake_home = tmp_path / "home"
    repo.mkdir()
    fake_home.mkdir()
    monkeypatch.setattr(generate, "ROOT", repo)
    monkeypatch.setenv("HOME", str(fake_home))

    # Use a path that lives nowhere we recognise.
    outside = tmp_path / "elsewhere" / "data" / "rollout.jsonl"
    outside.parent.mkdir(parents=True)
    outside.write_text("{}\n")

    scrubbed = generate._scrub_host_path(outside)

    # No absolute host prefix may survive scrubbing.
    assert not scrubbed.startswith("/")
    assert str(tmp_path) not in scrubbed
    # Drops the host layout — leaves only the basename for context.
    assert scrubbed == "rollout.jsonl"
