"""Guard P78-CONFIRM-4 at 4cb6f9d4 without migrating legacy caches."""

import subprocess

import pytest

from benchflow._utils import benchmark_repos as br


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize("namespace", ["datasets", "datasets-v2"])
@pytest.mark.parametrize("layout", ["checkout", "snapshot"])
def test_cache_provenance_never_becomes_the_enclosing_project(
    tmp_path, monkeypatch, namespace, layout
):
    """Guards 4cb6f9d4 and #492 for both retained cache representations."""
    project = tmp_path / "project"
    project.mkdir()
    _git(project, "init", "-q")
    _git(project, "remote", "add", "origin", "https://github.com/acme/project.git")
    (project / "dummy").write_text("enclosing project marker\n")
    _git(project, "add", ".")
    _git(
        project,
        "-c",
        "user.name=BenchFlow Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    root = project / ".cache" / namespace / "acme" / "tasks"
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    task = root / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.1"\n')
    (task / "instruction.md").write_text("dataset marker\n")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=BenchFlow Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    sha = _git(root, "rev-parse", "HEAD")
    if layout == "snapshot":
        snapshot = root.parent / "tasks__snapshots" / sha
        snapshot.parent.mkdir()
        _git(root, "worktree", "add", "-q", "--detach", str(snapshot), sha)
        task = snapshot / "tasks" / "task-a"
    monkeypatch.chdir(project)
    before = (task / "instruction.md").read_bytes()
    provenance = br.infer_task_source_provenance(task)
    assert provenance is not None
    assert provenance["repo"] == "acme/tasks"
    assert provenance["resolved_sha"] == sha
    assert provenance["path"] == "tasks/task-a"
    assert not provenance["dirty"]
    assert (task / "instruction.md").read_bytes() == before
    assert br._cache_dir() == project / ".cache/datasets-v2"
