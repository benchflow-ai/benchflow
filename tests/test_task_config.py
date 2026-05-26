"""Tests for task.toml parsing into TaskConfig."""

import pytest

from benchflow.task.config import TaskConfig


def test_task_config_reads_expected_skills_from_verifier_memory():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    cfg = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier.memory]\nexpected_skills = ["git-bisect", "rg"]\n'
    )
    assert cfg.expected_skills == ["git-bisect", "rg"]


def test_task_config_public_toml_dump_omits_expected_skills_fixture():
    """Guards hidden Memory-space fixtures from public task serialization."""
    cfg = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier.memory]\nexpected_skills = ["git-bisect", "rg"]\n'
    )

    dumped = cfg.model_dump_toml()

    assert cfg.expected_skills == ["git-bisect", "rg"]
    assert "expected_skills" not in dumped
    assert "git-bisect" not in dumped
    assert "rg" not in dumped


def test_task_config_preserves_empty_expected_skills_fixture():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    cfg = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier.memory]\nexpected_skills = []\n'
    )
    assert cfg.expected_skills == []


def test_task_config_absent_expected_skills_fixture_is_none():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    cfg = TaskConfig.model_validate_toml('version = "1.0"\n')
    assert cfg.expected_skills is None


def test_task_config_rejects_malformed_expected_skills():
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    with pytest.raises(ValueError, match="expected_skills"):
        TaskConfig.model_validate_toml(
            'version = "1.0"\n[verifier.memory]\nexpected_skills = "git-bisect"\n'
        )
