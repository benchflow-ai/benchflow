"""Unit tests for the repo-root build-context stager."""

from __future__ import annotations

from pathlib import Path

from benchflow._utils.build_context_stage import stage_task, stage_tasks


def _make_env0_style_task(root: Path, name: str) -> Path:
    """A task with a repo-root COPY (the smolclaws/env0 convention)."""
    t = root / name
    (t / "environment").mkdir(parents=True)
    (t / "data").mkdir()
    (t / "data" / "needles.py").write_text("X = 1\n")
    (t / "environment" / "Dockerfile").write_text(
        f"FROM base:latest\nCOPY tasks/{name}/data /tasks/{name}/data\nRUN seed\n"
    )
    (t / "solution").mkdir()
    (t / "solution" / "solve.sh").write_text("echo ok\n")
    return t


def test_stage_rewrites_repo_root_copy_and_stages_data(tmp_path):
    src = tmp_path / "tasks"
    src.mkdir()
    _make_env0_style_task(src, "slack-channel-reorg")
    out = tmp_path / "staged"

    names = stage_tasks(src, out)
    assert names == ["slack-channel-reorg"]

    df = (out / "slack-channel-reorg" / "environment" / "Dockerfile").read_text()
    assert "COPY data /tasks/slack-channel-reorg/data" in df  # rewritten
    assert "COPY tasks/" not in df  # no repo-root COPY left
    # the referenced dir is now inside the build context (environment/)
    assert (
        out / "slack-channel-reorg" / "environment" / "data" / "needles.py"
    ).exists()
    # rest of the task is carried through unchanged
    assert (out / "slack-channel-reorg" / "solution" / "solve.sh").exists()


def test_environment_relative_copy_is_left_unchanged(tmp_path):
    # A benchflow-native task (COPY relative to environment/) must not be touched.
    src = tmp_path / "tasks"
    (src / "native" / "environment").mkdir(parents=True)
    (src / "native" / "environment" / "claw-gmail").mkdir()
    (src / "native" / "environment" / "Dockerfile").write_text(
        "FROM base:latest\nCOPY claw-gmail /tmp/claw-gmail\n"
    )
    out = tmp_path / "staged"
    stage_tasks(src, out)
    df = (out / "native" / "environment" / "Dockerfile").read_text()
    assert "COPY claw-gmail /tmp/claw-gmail" in df  # untouched


def test_stage_tasks_skips_underscore_and_non_task_dirs(tmp_path):
    src = tmp_path / "tasks"
    src.mkdir()
    _make_env0_style_task(src, "real-task")
    (src / "_manifests").mkdir()  # not a task
    (src / "_manifests" / "env0.toml").write_text("x = 1\n")
    (src / "loose").mkdir()  # no environment/Dockerfile
    names = stage_tasks(src, tmp_path / "staged")
    assert names == ["real-task"]


def test_stage_task_returns_false_when_no_repo_root_copy(tmp_path):
    src = tmp_path / "native"
    (src / "environment").mkdir(parents=True)
    (src / "environment" / "Dockerfile").write_text("FROM base:latest\nRUN echo hi\n")
    assert stage_task(src, tmp_path / "out") is False
