"""TDD: config<->manifest consistency validator (ENG-265 slice 3)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "integration"))
from _consistency import config_drifts

_MANIFEST = {
    "axes": {
        "agents": {"credentialed": ["codex-acp", "openhands"]},
        "models": {"default": {"codex-acp": "gpt-5.4-nano"}},
        "task_sets": {
            "skillsbench_release_subset": {
                "include": ["task-a", "task-b"],
            }
        },
    }
}


def test_aligned_config_has_no_drift():
    configs = {
        "codex-acp": {
            "agent": "codex-acp",
            "model": "gpt-5.4-nano",
            "include": ["task-a", "task-b"],
        }
    }
    assert config_drifts(_MANIFEST, configs) == []


def test_agent_not_in_credentialed_axis_is_drift():
    configs = {
        "mimo": {
            "agent": "mimo",
            "model": "xiaomi/mimo-v2.5",
            "include": ["task-a", "task-b"],
        }
    }
    drifts = config_drifts(_MANIFEST, configs)
    assert any("mimo" in d and "credentialed" in d for d in drifts), drifts


def test_model_mismatch_against_default_is_drift():
    configs = {
        "codex-acp": {
            "agent": "codex-acp",
            "model": "gpt-OLD",
            "include": ["task-a", "task-b"],
        }
    }
    drifts = config_drifts(_MANIFEST, configs)
    assert any("codex-acp" in d and "model" in d for d in drifts), drifts


def test_agent_without_default_pin_has_no_model_drift():
    # openhands is credentialed but axes.models.default does not pin it -> no model check
    configs = {
        "openhands": {
            "agent": "openhands",
            "model": "anything",
            "include": ["task-a", "task-b"],
        }
    }
    assert config_drifts(_MANIFEST, configs) == []


def test_include_mismatch_against_task_set_is_drift():
    configs = {
        "codex-acp": {
            "agent": "codex-acp",
            "model": "gpt-5.4-nano",
            "include": ["task-a"],
        }
    }
    drifts = config_drifts(_MANIFEST, configs)
    assert any("include" in d for d in drifts), drifts
