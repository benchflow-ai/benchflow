from pathlib import Path

import pytest

from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
    SKILL_SOURCE_NONE,
    SKILL_SOURCE_SELF_GENERATED,
    SKILL_SOURCE_TASK_BUNDLED,
    normalize_skill_mode,
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
        skill_mode=SKILL_MODE_NO_SKILL,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
    )

    assert policy.mode == SKILL_MODE_NO_SKILL
    assert policy.source == SKILL_SOURCE_NONE
    assert policy.enabled is False
    assert policy.include_task_skills is False
    assert policy.host_dir is None
    assert policy.sandbox_dir is None
    assert policy.prompt_dir is None
    assert policy.needs_task_copy is True
    assert policy.strip_bundled_dir_from_copy is True

    strip_task_bundled_skills(task)
    assert not (task / "environment" / "skills").exists()


def test_task_skill_strip_removes_direct_dockerfile_skill_copies(
    tmp_path: Path,
) -> None:
    """Guards SkillsBench a8eefb4 radar-vital-signs no-skill sandbox startup."""
    task = tmp_path / "task"
    _make_task_skills(task)
    dockerfile = task / "environment" / "Dockerfile"
    dockerfile.write_text(
        "\n".join(
            [
                "FROM python:3.11-slim",
                "COPY recordings /root/recordings",
                "COPY skills /root/.claude/skills",
                "COPY skills /root/.codex/skills",
                "RUN echo ready",
            ]
        )
        + "\n"
    )

    strip_task_bundled_skills(task)

    assert not (task / "environment" / "skills").exists()
    assert dockerfile.read_text() == (
        "FROM python:3.11-slim\nCOPY recordings /root/recordings\nRUN echo ready\n"
    )


def test_with_skill_mode_keeps_task_bundle(tmp_path: Path) -> None:
    """Guards PR #586 so canonical with-skill mode enables task skills."""
    task = tmp_path / "task"
    skills_root = _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        skill_mode=SKILL_MODE_WITH_SKILL,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
    )

    assert policy.mode == SKILL_MODE_WITH_SKILL
    assert policy.source == SKILL_SOURCE_TASK_BUNDLED
    assert policy.enabled is True
    assert policy.include_task_skills is True
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
        skill_mode=SKILL_MODE_NO_SKILL,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir="/skills",
    )
    enabled = resolve_task_skill_policy(
        task_path=task,
        skill_mode=SKILL_MODE_WITH_SKILL,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir="/skills",
    )

    assert disabled.prompt_dir is None
    assert disabled.sandbox_dir is None
    assert disabled.strip_bundled_dir_from_copy is True
    assert enabled.enabled is True
    assert enabled.host_dir == skills_root
    assert enabled.sandbox_dir == "/skills"
    assert enabled.prompt_dir == skills_root
    assert enabled.strip_bundled_dir_from_copy is False


def test_with_skill_honors_declared_sandbox_path(tmp_path: Path) -> None:
    """Guards PR #586 so task-local skills use the task-declared mount path."""
    task = tmp_path / "task"
    skills_root = _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        skill_mode=SKILL_MODE_WITH_SKILL,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir="/opt/benchflow/skill-eval",
    )

    assert policy.enabled is True
    assert policy.host_dir == skills_root
    assert policy.sandbox_dir == "/opt/benchflow/skill-eval"
    assert policy.strip_bundled_dir_from_copy is False


def test_with_skill_rejects_unsafe_sandbox_path(tmp_path: Path) -> None:
    """Guards PR #586 against Dockerfile injection via task-declared skills_dir."""
    task = tmp_path / "task"
    _make_task_skills(task)

    with pytest.raises(ValueError, match=r"environment\.skills_dir"):
        resolve_task_skill_policy(
            task_path=task,
            skill_mode=SKILL_MODE_WITH_SKILL,
            runtime_skills_dir=None,
            declared_sandbox_skills_dir="/opt/skills; touch /tmp/PWNED",
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


def test_self_gen_policy_records_generated_skill_source(tmp_path: Path) -> None:
    """Guards PR #233 so self-gen artifacts do not look like no-skill runs."""
    task = tmp_path / "task"
    _make_task_skills(task)

    policy = resolve_task_skill_policy(
        task_path=task,
        skill_mode=SKILL_MODE_SELF_GEN,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir="/skills",
    )

    assert policy.mode == SKILL_MODE_SELF_GEN
    assert policy.source == SKILL_SOURCE_SELF_GENERATED
    assert policy.enabled is False
    assert policy.include_task_skills is False
    assert policy.strip_bundled_dir_from_copy is True


@pytest.mark.parametrize(
    "raw",
    [
        "no-skill",
        "with-skill",
        "self-gen",
    ],
)
def test_normalize_skill_mode_accepts_canonical_modes(raw: str) -> None:
    """Guards PR #586 so the global skill-mode switch stays canonical."""
    assert normalize_skill_mode(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "baseline",
        "without_skills",
        "with-task-skills",
        "self_generated",
        "",
    ],
)
def test_normalize_skill_mode_rejects_legacy_aliases(raw: str) -> None:
    """Guards PR #586 so removed compatibility aliases cannot drift back in."""
    with pytest.raises(ValueError, match="skill_mode must be one of"):
        normalize_skill_mode(raw)
