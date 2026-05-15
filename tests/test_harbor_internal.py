"""Tests for the internal Harbor compatibility adapter."""

from pathlib import Path

from benchflow import _harbor


def _minimal_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 60\n\n'
        "[agent]\ntimeout_sec = 60\n\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Solve it.")
    return task_dir


def test_make_task_returns_current_harbor_task(tmp_path: Path) -> None:
    task = _harbor.make_task(_minimal_task_dir(tmp_path))
    assert task.__class__.__module__.startswith("harbor")
    assert task.config.agent.timeout_sec == 60


def test_make_trial_paths_preserves_current_shape(tmp_path: Path) -> None:
    paths = _harbor.make_trial_paths(tmp_path / "trial")
    paths.mkdir()
    assert paths.trial_dir == tmp_path / "trial"
    assert paths.agent_dir == tmp_path / "trial" / "agent"
    assert paths.verifier_dir == tmp_path / "trial" / "verifier"


def test_make_verifier_constructs_harbor_verifier(tmp_path: Path) -> None:
    task = _harbor.make_task(_minimal_task_dir(tmp_path))
    paths = _harbor.make_trial_paths(tmp_path / "trial")
    env = object()
    verifier = _harbor.make_verifier(task, paths, env)
    assert verifier.__class__.__module__.startswith("harbor")
