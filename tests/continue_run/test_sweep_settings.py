"""Guards FF-sw-evaluation continuation fixes for 363d815b."""

import pytest

from benchflow.continue_run.orchestrator import (
    build_rollout_config,
)
from benchflow.continue_run.run_folder import RunFolder, RunFolderError


@pytest.mark.parametrize(
    "field,value,target",
    [
        ("base_image_override", "benchflow/custom-tools:v1", "base_image_override"),
        ("sandbox_locked_paths", ["/opt/fixed"], "sandbox_locked_paths"),
        ("sandbox_setup_timeout", 75, "sandbox_setup_timeout"),
        ("skip_install", True, "skip_agent_install"),
        (
            "config_override",
            {"patch": {"agent": {"timeout_sec": 99}}},
            "config_override",
        ),
        ("agent_env", {"EXTRA_MODE": "test"}, "agent_env"),
        ("integrity", "audit", "integrity"),
        ("freeze_workspace", True, "freeze_workspace"),
    ],
)
def test_continuation_restores_recorded_settings(tmp_path, field, value, target):
    run = RunFolder(tmp_path, {field: value}, {}, ["go"], [])
    config = build_rollout_config(
        run,
        task_path=tmp_path,
        live_model="test",
        agent_env={},
        timeout=30,
        output_dir=tmp_path,
        rollout_name="continued",
    )
    assert getattr(config, target) == (
        value["patch"] if field == "config_override" else value
    )


@pytest.mark.parametrize("field", ["context_root", "requested_skills_dir"])
def test_continuation_requires_recorded_local_inputs(tmp_path, field):
    run = RunFolder(
        tmp_path,
        {field: str(tmp_path / "missing"), "skill_mode": "with-skill"},
        {},
        ["go"],
        [],
    )
    with pytest.raises(RunFolderError, match="cannot reconstruct"):
        build_rollout_config(
            run,
            task_path=tmp_path,
            live_model="test",
            agent_env={},
            timeout=30,
            output_dir=tmp_path,
            rollout_name="continued",
        )


def test_continuation_restores_skill_inputs(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    run = RunFolder(
        tmp_path,
        {"skill_mode": "with-skill", "requested_skills_dir": str(skills)},
        {},
        ["go"],
        [],
    )
    config = build_rollout_config(
        run,
        task_path=tmp_path,
        live_model="test",
        agent_env={},
        timeout=30,
        output_dir=tmp_path,
        rollout_name="continued",
    )
    assert config.skill_mode == "with-skill"
    assert str(config.skills_dir) == str(skills)


@pytest.mark.parametrize(
    "recorded",
    [
        {"environment_manifest": {"name": "summarized"}},
        {"skill_mode": "self-gen"},
        {"harness_mode": "native"},
        {"loop": {"strategy": "verify-retry"}},
        {"config_override": {"keys": ["agent"]}},
        {"review": {"contract": "weighted"}},
        {"skill_source": "custom_runtime", "skill_mode": "with-skill"},
        {"scenes": [{"roles": [{"agent": "openhands", "env_keys": ["MODE"]}]}]},
    ],
)
def test_continuation_refuses_incomplete_execution_records(tmp_path, recorded):
    run = RunFolder(tmp_path, recorded, {}, ["go"], [])
    with pytest.raises(RunFolderError, match="cannot reconstruct"):
        build_rollout_config(
            run,
            task_path=tmp_path,
            live_model="test",
            agent_env={},
            timeout=30,
            output_dir=tmp_path,
            rollout_name="continued",
        )


def test_continuation_retains_reviewer_runtime(tmp_path):
    from benchflow.review.options import ReviewerConfig

    reviewer = ReviewerConfig(agent="opencode", model="test-reviewer", timeout_sec=321)
    run = RunFolder(
        tmp_path,
        {"review": {"reviewer": reviewer.to_config_artifact()}},
        {},
        ["go"],
        [],
    )
    config = build_rollout_config(
        run,
        task_path=tmp_path,
        live_model="test",
        agent_env={},
        timeout=30,
        output_dir=tmp_path,
        rollout_name="continued",
    )
    assert config.reviewer == reviewer
