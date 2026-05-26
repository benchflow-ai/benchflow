"""RolloutConfig coerces string path inputs to :class:`pathlib.Path`.

Regression for #368.2 (ENG-166): passing ``task_path='tasks/foo'`` used to
leave ``self.task_path`` as ``str``, then downstream callers that access
``task_path.name`` raised ``AttributeError: 'str' object has no attribute
'name'``. ``RolloutConfig.__post_init__`` now normalises ``task_path``,
``context_root``, ``skills_dir``, and ``jobs_dir`` to ``Path``.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.rollout import RolloutConfig, Scene


def test_rollout_config_coerces_task_path_string() -> None:
    cfg = RolloutConfig(
        task_path="tasks/foo",
        scenes=[Scene.single(agent="claude-agent-acp")],
    )

    assert isinstance(cfg.task_path, Path)
    assert cfg.task_path == Path("tasks/foo")
    # The downstream attribute that originally crashed must now resolve.
    assert cfg.task_path.name == "foo"


def test_rollout_config_coerces_optional_paths() -> None:
    cfg = RolloutConfig(
        task_path="tasks/foo",
        scenes=[Scene.single(agent="claude-agent-acp")],
        context_root="ctx/root",
        skills_dir="skills",
        jobs_dir="custom-jobs",
    )

    assert isinstance(cfg.context_root, Path)
    assert cfg.context_root == Path("ctx/root")
    assert isinstance(cfg.skills_dir, Path)
    assert cfg.skills_dir == Path("skills")
    assert isinstance(cfg.jobs_dir, Path)
    assert cfg.jobs_dir == Path("custom-jobs")


def test_rollout_config_keeps_path_when_already_path() -> None:
    task_path = Path("tasks/foo")
    context_root = Path("ctx/root")
    skills_dir = Path("skills")
    jobs_dir = Path("custom-jobs")

    cfg = RolloutConfig(
        task_path=task_path,
        scenes=[Scene.single(agent="claude-agent-acp")],
        context_root=context_root,
        skills_dir=skills_dir,
        jobs_dir=jobs_dir,
    )

    # No double-wrapping: the stored value is the same Path instance the
    # caller supplied, not ``Path(Path(...))``.
    assert cfg.task_path is task_path
    assert cfg.context_root is context_root
    assert cfg.skills_dir is skills_dir
    assert cfg.jobs_dir is jobs_dir


def test_rollout_config_context_root_none_stays_none() -> None:
    """Optional fields that default to ``None`` are left untouched."""
    cfg = RolloutConfig(
        task_path="tasks/foo",
        scenes=[Scene.single(agent="claude-agent-acp")],
    )

    assert cfg.context_root is None
    assert cfg.skills_dir is None
