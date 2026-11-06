"""A release loads its agents catalog from a pinned commit, with overrides.

Since #1093 moved OpenClaw out of core, the default catalog decides which
OpenClaw shim a run installs. A branch such as ``main`` would change what a
released BenchFlow runs; a full commit SHA is fetched as-is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.benchmark_repos import _looks_like_commit_sha
from benchflow.agents import remote_manifests


@pytest.fixture
def fetched(monkeypatch, tmp_path: Path) -> list[tuple[str, str | None]]:
    calls: list[tuple[str, str | None]] = []

    def fake_resolve_source(repo, path=None, ref=None):
        calls.append((repo, ref))
        return tmp_path

    monkeypatch.setattr(
        "benchflow._utils.benchmark_repos.resolve_source", fake_resolve_source
    )
    monkeypatch.delenv(remote_manifests.AGENTS_DIR_ENV, raising=False)
    monkeypatch.delenv(remote_manifests.AGENTS_SOURCE_ENV, raising=False)
    return calls


def _fetch_effective_source() -> None:
    spec, _local = remote_manifests._effective_source()
    remote_manifests._source_root(remote_manifests._parse_source(spec))


def test_default_catalog_is_a_pinned_agents_commit(fetched) -> None:
    _fetch_effective_source()
    [(repo, ref)] = fetched
    assert repo == "benchflow-ai/agents"
    assert ref is not None and _looks_like_commit_sha(ref), ref


def test_agents_source_env_overrides_the_pin(fetched, monkeypatch) -> None:
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "benchflow-ai/agents@main")
    _fetch_effective_source()
    assert fetched == [("benchflow-ai/agents", "main")]


def test_agents_dir_overrides_the_pin_without_fetching(
    fetched, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(remote_manifests.AGENTS_DIR_ENV, str(tmp_path))
    assert remote_manifests._effective_source() == (str(tmp_path), True)
    _fetch_effective_source()
    assert fetched == []
