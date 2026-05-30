from pathlib import Path

from benchflow.skill_policy import resolve_task_skill_policy, strip_task_bundled_skills


def _make_task_skills(task_path: Path) -> Path:
    skills = task_path / "environment" / "skills" / "alpha"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# Alpha\n")
    return task_path / "environment" / "skills"


def test_task_skills_are_stripped_from_no_skills_task_copy(tmp_path: Path) -> None:
    """Guards PR #860 against Docker build-context leaks in no-skills runs."""
    task = tmp_path / "task"
    _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
        include_task_skills=False,
    )

    assert policy.prompt_bundled_dir is None
    assert policy.needs_task_copy is True
    assert policy.strip_bundled_dir_from_copy is True

    strip_task_bundled_skills(task)
    assert not (task / "environment" / "skills").exists()


def test_explicit_task_skills_path_keeps_task_bundle(tmp_path: Path) -> None:
    """Guards PR #860 so --skills-dir auto/task-path still enables task skills."""
    task = tmp_path / "task"
    skills_root = _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        runtime_skills_dir=skills_root,
        declared_sandbox_skills_dir=None,
        include_task_skills=False,
    )

    assert policy.prompt_bundled_dir == skills_root
    assert policy.strip_bundled_dir_from_copy is False


def test_declared_task_skills_only_apply_when_included(tmp_path: Path) -> None:
    """Guards PR #860 against task.toml skills overriding no-skills mode."""
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

    assert disabled.prompt_bundled_dir is None
    assert disabled.strip_bundled_dir_from_copy is True
    assert enabled.prompt_bundled_dir == skills_root
    assert enabled.strip_bundled_dir_from_copy is False
