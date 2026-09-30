"""A path-scoped source fetches that path only.

Guards the dx/first-run sparse source fetch. Before it, `bench eval run
--source-repo benchflow-ai/skillsbench --source-path tasks/citation-check`
cloned the whole repository to run one task: on 2026-09-30 that was a
1.1 GB checkout (a 447 MB pack) plus a 644 MB per-commit snapshot, about
21 s on a GCP VM. These tests run real git against a local repository served
over ``file://``, and check which file contents were fetched at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br

_IDENTITY = ("-c", "user.name=BenchFlow Test", "-c", "user.email=test@example.com")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _remote(tmp_path: Path) -> Path:
    """A bare repository with two tasks, docs, a non-task folder and a symlink."""
    src = tmp_path / "src"
    _write(src / "README.md", "# bench\n")
    _write(src / "tasks/citation-check/task.md", "---\n---\nCheck the citations.\n")
    _write(src / "tasks/citation-check/environment/Dockerfile", "FROM ubuntu:24.04\n")
    _write(src / "tasks/citation-check/verifier/test.sh", "exit 0\n")
    _write(src / "tasks/court-form/task.md", "---\n---\nFill the form.\n")
    _write(src / "tasks/court-form/big.bin", "x" * 100_000)
    _write(src / "docs/guide.md", "guide\n")
    _write(src / "data/foreign/items.csv", "id,prompt\n1,hi\n")
    (src / "link").symlink_to("tasks", target_is_directory=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(src))
    _git(src, "add", "-A")
    _git(src, *_IDENTITY, "commit", "-q", "-m", "one")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(bare))
    _git(bare, "config", "uploadpack.allowFilter", "true")
    _git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")
    return bare


@pytest.fixture
def bench(tmp_path, monkeypatch) -> Path:
    """Serve the repository as acme/bench; return the cache checkout's path."""
    bare = _remote(tmp_path)
    monkeypatch.setattr(br, "_repo_url", lambda org, repo: f"file://{bare}")
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path / "cache")
    return tmp_path / "cache" / "acme" / "bench"


def _missing_blobs(repo: Path) -> set[str]:
    """Objects HEAD needs that were never fetched (listing them fetches nothing)."""
    listing = _git(repo, "rev-list", "--objects", "--missing=print", "HEAD")
    return {line[1:] for line in listing.splitlines() if line.startswith("?")}


def _blob(repo: Path, path: str) -> str:
    return _git(repo, "ls-tree", "HEAD", "--", path).split()[2]


def test_source_path_fetches_and_checks_out_only_that_path(bench):
    resolved = br.resolve_source_with_metadata(
        "acme/bench", path="tasks/citation-check"
    )

    snapshot = resolved.path.parents[1]
    assert (resolved.path / "task.md").read_text().endswith("Check the citations.\n")
    for checkout in (bench, snapshot):
        assert (checkout / "tasks/citation-check/environment/Dockerfile").is_file()
        assert not (checkout / "tasks/court-form").exists()
        assert not (checkout / "docs").exists()
    missing = _missing_blobs(bench)
    assert _blob(bench, "tasks/court-form/big.bin") in missing
    assert _blob(bench, "docs/guide.md") in missing
    assert _blob(bench, "tasks/citation-check/task.md") not in missing
    assert resolved.provenance["path"] == "tasks/citation-check"
    assert resolved.provenance["dirty"] is False
    assert set(resolved.provenance["file_hashes"]) == {
        "task.md",
        "environment/Dockerfile",
        "verifier/test.sh",
    }


def test_a_second_path_joins_the_same_checkout_and_snapshot(bench):
    first = br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")
    second = br.resolve_source_with_metadata("acme/bench", path="tasks/court-form")

    assert first.path.parents[1] == second.path.parents[1]
    assert (first.path / "task.md").is_file()
    assert (second.path / "big.bin").stat().st_size == 100_000
    assert (bench / "tasks/court-form/big.bin").is_file()
    assert not (second.path.parents[1] / "docs").exists()
    assert _blob(bench, "docs/guide.md") in _missing_blobs(bench)


def test_plain_resolve_source_is_sparse_too(bench):
    path = br.resolve_source("acme/bench", path="tasks/court-form")

    assert path == bench / "tasks" / "court-form"
    assert (path / "task.md").is_file()
    assert not (bench / "tasks/citation-check").exists()


def test_a_request_for_the_whole_repository_turns_sparse_off(bench):
    br.resolve_source("acme/bench", path="tasks/citation-check")

    root = br.resolve_source("acme/bench")

    assert root == bench
    assert (root / "docs/guide.md").read_text() == "guide\n"
    assert (root / "tasks/court-form/big.bin").is_file()
    assert not br._is_sparse_checkout(root)


def test_a_folder_without_benchflow_tasks_gets_the_whole_repository(bench):
    """Foreign benchmarks keep working: their source adapters (Toolathlon,
    MCP-Atlas) read files outside the path, such as db/ or configs/."""
    path = br.resolve_source("acme/bench", path="data/foreign")

    assert (path / "items.csv").is_file()
    assert (bench / "docs/guide.md").is_file()
    assert not br._is_sparse_checkout(bench)


def test_a_path_through_a_symlink_gets_the_whole_repository(bench):
    resolved = br.resolve_source_with_metadata("acme/bench", path="link/court-form")

    assert resolved.provenance["path"] == "tasks/court-form"
    assert (resolved.path / "big.bin").is_file()
    assert not br._is_sparse_checkout(bench)


def test_a_missing_path_lists_its_parent_and_suggests_a_match(bench):
    with pytest.raises(FileNotFoundError) as caught:
        br.resolve_source_with_metadata("acme/bench", path="tasks/citaton-check")

    message = str(caught.value)
    assert "not found in acme/bench. Available: " in message
    assert "['tasks/citation-check', 'tasks/court-form']" in message
    assert message.endswith("Did you mean 'tasks/citation-check'?")
    # Reporting the typo fetched no task's files.
    assert _blob(bench, "tasks/court-form/big.bin") in _missing_blobs(bench)


def test_a_file_path_is_refused_as_not_a_directory(bench):
    with pytest.raises(ValueError, match="must resolve to a directory"):
        br.resolve_source_with_metadata("acme/bench", path="docs/guide.md")


def test_a_commit_ref_checks_out_that_commit(bench, tmp_path):
    src = tmp_path / "src"
    first = _git(src, "rev-parse", "HEAD")
    _write(src / "tasks/court-form/task.md", "---\n---\nFill the new form.\n")
    _git(src, *_IDENTITY, "commit", "-qam", "two")
    _git(src, "push", "-q", str(tmp_path / "remote.git"), "main")

    resolved = br.resolve_source_with_metadata(
        "acme/bench", path="tasks/court-form", ref=first
    )

    assert resolved.provenance["resolved_sha"] == first
    assert (resolved.path / "task.md").read_text().endswith("Fill the form.\n")
    assert not (resolved.path.parents[1] / "tasks/citation-check").exists()


def test_a_full_clone_made_before_sparse_fetches_is_reused_as_is(bench, tmp_path):
    bench.parent.mkdir(parents=True)
    _git(tmp_path, "clone", "-q", f"file://{tmp_path / 'remote.git'}", str(bench))

    resolved = br.resolve_source_with_metadata("acme/bench", path="tasks/court-form")

    assert (resolved.path / "big.bin").is_file()
    assert (resolved.path.parents[1] / "docs/guide.md").is_file()
    assert not br._is_sparse_checkout(bench)


def test_the_sparse_fetch_prints_nothing(bench, capfd):
    capfd.readouterr()

    br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")
    br.resolve_source_with_metadata("acme/bench", path="tasks/court-form")

    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""
