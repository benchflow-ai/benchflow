"""config<->manifest consistency validator (ENG-265 slice 3).

Covers the membership invariant in BOTH directions (a config's agent must be
known; a credentialed agent must have a config), model pins, order-insensitive
includes, source validation, undefined task sets, and robustness on an empty
axis. The order-insensitive include case and the reverse-drift case are what
distinguish the fixed validator from the pre-fix one, so a revert goes red here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "integration"))
from _consistency import config_drifts

_SOURCE = {"repo": "benchflow-ai/skillsbench", "path": "tasks", "ref": "main"}

_MANIFEST = {
    "axes": {
        "agents": {
            "credentialed": ["codex-acp", "openhands"],
            "experimental": ["mimo"],
        },
        "models": {"default": {"codex-acp": "gpt-5.4-nano"}},
        "task_sets": {
            "skillsbench_release_subset": {
                "include": ["task-a", "task-b"],
                "source": dict(_SOURCE),
            }
        },
    }
}


def _cfg(agent, *, model="x", include=("task-a", "task-b"), source=None):
    return {
        "agent": agent,
        "model": model,
        "include": list(include),
        "source": dict(_SOURCE) if source is None else source,
    }


def _aligned():
    """Every credentialed agent has a config; pinned agents use the pinned model."""
    return {
        "codex-acp": _cfg("codex-acp", model="gpt-5.4-nano"),
        "openhands": _cfg("openhands"),
    }


def test_aligned_config_has_no_drift():
    assert config_drifts(_MANIFEST, _aligned()) == []


def test_experimental_agent_is_not_drift():
    # mimo is experimental -> allowed even though not credentialed (the resolution).
    configs = _aligned()
    configs["mimo"] = _cfg("mimo", model="xiaomi/mimo-v2.5")
    assert config_drifts(_MANIFEST, configs) == []


def test_agent_in_neither_axis_is_drift():
    configs = _aligned()
    configs["rogue"] = _cfg("rogue")
    drifts = config_drifts(_MANIFEST, configs)
    assert any("rogue" in d and "credentialed" in d for d in drifts), drifts


def test_credentialed_agent_without_config_is_drift():
    # Reverse direction: openhands is credentialed but ships no config.
    configs = {"codex-acp": _cfg("codex-acp", model="gpt-5.4-nano")}
    drifts = config_drifts(_MANIFEST, configs)
    assert any("openhands" in d and "no config" in d for d in drifts), drifts


def test_model_mismatch_against_default_is_drift():
    configs = _aligned()
    configs["codex-acp"] = _cfg("codex-acp", model="gpt-OLD")
    drifts = config_drifts(_MANIFEST, configs)
    assert any("codex-acp" in d and "model" in d for d in drifts), drifts


def test_agent_without_default_pin_has_no_model_drift():
    # openhands is credentialed but unpinned in axes.models.default -> no model check.
    configs = _aligned()
    configs["openhands"] = _cfg("openhands", model="anything")
    assert config_drifts(_MANIFEST, configs) == []


def test_include_reorder_is_not_drift():
    # Order-insensitive: distinguishes the fixed set comparison from list==.
    configs = _aligned()
    configs["openhands"] = _cfg("openhands", include=["task-b", "task-a"])
    assert config_drifts(_MANIFEST, configs) == []


def test_include_missing_task_is_drift():
    configs = _aligned()
    configs["openhands"] = _cfg("openhands", include=["task-a"])
    drifts = config_drifts(_MANIFEST, configs)
    assert any("include" in d for d in drifts), drifts


def test_source_mismatch_is_drift():
    # Same task names from a different repo/ref are different tasks.
    configs = _aligned()
    configs["openhands"] = _cfg(
        "openhands", source={"repo": "evil/repo", "path": "tasks", "ref": "attacker"}
    )
    drifts = config_drifts(_MANIFEST, configs)
    assert any("source" in d and "openhands" in d for d in drifts), drifts


def test_undefined_task_set_is_drift():
    drifts = config_drifts(_MANIFEST, _aligned(), task_set="does-not-exist")
    assert any("task set" in d for d in drifts), drifts


def test_empty_experimental_axis_does_not_crash():
    manifest = {
        "axes": {
            "agents": {"credentialed": ["codex-acp"], "experimental": None},
            "models": {"default": {"codex-acp": "gpt-5.4-nano"}},
            "task_sets": {
                "skillsbench_release_subset": {
                    "include": ["task-a"],
                    "source": dict(_SOURCE),
                }
            },
        }
    }
    configs = {"codex-acp": _cfg("codex-acp", model="gpt-5.4-nano", include=["task-a"])}
    assert config_drifts(manifest, configs) == []


def test_no_configs_is_a_noop():
    # With no configs there is nothing to compare; the caller decides whether an
    # absent configs/ dir is itself an error (load_suite treats it as skip).
    assert config_drifts(_MANIFEST, {}) == []
