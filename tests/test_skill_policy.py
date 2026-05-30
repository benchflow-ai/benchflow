from pathlib import Path

import pytest

from benchflow.skill_policy import (
    resolve_runtime_skills_dir,
    resolve_task_skill_policy,
    strip_task_bundled_skills,
    validate_container_mount_path,
)


def _make_task_skills(task_path: Path) -> Path:
    skills = task_path / "environment" / "skills" / "alpha"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# Alpha\n")
    return task_path / "environment" / "skills"


def test_task_skills_are_stripped_from_no_skills_task_copy(tmp_path: Path) -> None:
    """Guards PR #586 against Docker build-context leaks in no-skills runs."""
    task = tmp_path / "task"
    _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
        include_task_skills=False,
    )

    assert policy.enabled is False
    assert policy.host_dir is None
    assert policy.sandbox_dir is None
    assert policy.prompt_dir is None
    assert policy.needs_task_copy is True
    assert policy.strip_bundled_dir_from_copy is True

    strip_task_bundled_skills(task)
    assert not (task / "environment" / "skills").exists()


def test_explicit_task_skills_path_keeps_task_bundle(tmp_path: Path) -> None:
    """Guards PR #586 so --skills-dir auto/task-path still enables task skills."""
    task = tmp_path / "task"
    skills_root = _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        runtime_skills_dir=skills_root,
        declared_sandbox_skills_dir=None,
        include_task_skills=False,
    )

    assert policy.enabled is True
    assert policy.host_dir == skills_root
    assert policy.sandbox_dir == "/skills"
    assert policy.prompt_dir == skills_root
    assert policy.strip_bundled_dir_from_copy is False


def test_declared_task_skills_only_apply_when_included(tmp_path: Path) -> None:
    """Guards PR #586 against task.toml skills overriding no-skills mode."""
    task = tmp_path / "task"
    skills_root = _make_task_skills(task)

    disabled = resolve_task_skill_policy(
        task_path=task,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir="/skills",
        include_task_skills=False,
    )
    enabled = resolve_task_skill_policy(
        task_path=task,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir="/skills",
        include_task_skills=True,
    )

    assert disabled.prompt_dir is None
    assert disabled.sandbox_dir is None
    assert disabled.strip_bundled_dir_from_copy is True
    assert enabled.enabled is True
    assert enabled.host_dir == skills_root
    assert enabled.sandbox_dir == "/skills"
    assert enabled.prompt_dir == skills_root
    assert enabled.strip_bundled_dir_from_copy is False


def test_include_task_skills_honors_declared_sandbox_path(tmp_path: Path) -> None:
    """Guards PR #586 so task-local skills use the task-declared mount path."""
    task = tmp_path / "task"
    skills_root = _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir="/opt/benchflow/skill-eval",
        include_task_skills=True,
    )

    assert policy.enabled is True
    assert policy.host_dir == skills_root
    assert policy.sandbox_dir == "/opt/benchflow/skill-eval"
    assert policy.strip_bundled_dir_from_copy is False


def test_include_task_skills_rejects_unsafe_sandbox_path(tmp_path: Path) -> None:
    """Guards PR #586 against Dockerfile injection via task-declared skills_dir."""
    task = tmp_path / "task"
    _make_task_skills(task)

    with pytest.raises(ValueError, match=r"environment\.skills_dir"):
        resolve_task_skill_policy(
            task_path=task,
            runtime_skills_dir=None,
            declared_sandbox_skills_dir="/opt/skills; touch /tmp/PWNED",
            include_task_skills=True,
        )


@pytest.mark.parametrize(
    "path",
    [
        "/skills",
        "/opt/benchflow/skill-eval",
        "/a/b_c.d-1",
    ],
)
def test_validate_container_mount_path_accepts_simple_absolute_paths(path: str) -> None:
    """Guards PR #586 so valid sandbox skill mount paths still work."""
    assert validate_container_mount_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "relative/skills",
        "/",
        "/opt/skills with spaces",
        "/opt/skills;touch",
        "/opt/../skills",
        "/opt//skills",
    ],
)
def test_validate_container_mount_path_rejects_unsafe_paths(path: str) -> None:
    """Guards PR #586 against shell metacharacters and ambiguous paths."""
    with pytest.raises(ValueError, match="simple absolute container path"):
        validate_container_mount_path(path)


def test_skills_dir_auto_resolves_at_rollout_boundary(tmp_path: Path) -> None:
    """Guards PR #586 so direct RolloutConfig/SDK paths share --skills-dir auto."""
    task = tmp_path / "task"
    skills_root = _make_task_skills(task)

    assert resolve_runtime_skills_dir(task, "auto") == skills_root
    assert resolve_runtime_skills_dir(task, Path("auto")) == skills_root
