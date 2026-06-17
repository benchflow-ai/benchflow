"""Unit tests for the C-axis config overlay (benchflow._utils.config_override)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchflow._utils.config_override import (
    apply_config_override,
    deep_merge,
    load_config_override,
    overlay_hash,
    validate_overlay,
)
from benchflow.task.config import TaskConfig


def _cfg() -> TaskConfig:
    return TaskConfig.model_validate(
        {
            "version": "1.0",
            "agent": {"timeout_sec": 300},
            "verifier": {"timeout_sec": 120},
            "environment": {"cpus": 1, "memory_mb": 2048},
        }
    )


# ---- deep_merge -----------------------------------------------------------


def test_deep_merge_nested_tables_merge_and_scalars_replace():
    out = deep_merge(
        {"agent": {"timeout_sec": 300, "model": "x"}, "verifier": {"timeout_sec": 120}},
        {"agent": {"timeout_sec": 42}},
    )
    assert out["agent"] == {"timeout_sec": 42, "model": "x"}  # sibling key kept
    assert out["verifier"] == {"timeout_sec": 120}  # untouched section kept


def test_deep_merge_lists_replace_wholesale():
    assert deep_merge({"k": [1, 2, 3]}, {"k": [9]})["k"] == [9]


def test_deep_merge_does_not_mutate_base():
    base = {"agent": {"timeout_sec": 300}}
    deep_merge(base, {"agent": {"timeout_sec": 42}})
    assert base == {"agent": {"timeout_sec": 300}}


def test_overlay_hash_stable_across_key_order():
    assert overlay_hash({"a": 1, "b": 2}) == overlay_hash({"b": 2, "a": 1})
    assert overlay_hash({"a": 1}) != overlay_hash({"a": 2})


# ---- parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"agent":{"timeout_sec":42}}',
        "agent:\n  timeout_sec: 42",
        "[agent]\ntimeout_sec=42",
    ],
)
def test_load_config_override_parses_json_yaml_toml(raw):
    assert load_config_override(raw) == {"agent": {"timeout_sec": 42}}


def test_load_config_override_at_file(tmp_path):
    f = tmp_path / "ov.yaml"
    f.write_text("agent:\n  timeout_sec: 77\n")
    assert load_config_override(f"@{f}") == {"agent": {"timeout_sec": 77}}


def test_load_config_override_empty_returns_none():
    assert load_config_override(None) is None
    assert load_config_override("") is None


def test_load_config_override_non_mapping_rejected():
    with pytest.raises(ValueError, match="mapping"):
        load_config_override("[1,2,3]")


def test_load_config_override_unparseable_names_all_formats():
    with pytest.raises(ValueError, match="JSON, YAML, or TOML"):
        load_config_override("{this is : not valid : anything ]")


# ---- validate_overlay (fail-closed allowlist) -----------------------------


@pytest.mark.parametrize("section", ["agent", "sandbox", "metadata"])
def test_validate_overlay_allows_config_sections(section):
    assert validate_overlay({section: {}}) == {section: {}}


@pytest.mark.parametrize(
    "section", ["verifier", "reward", "solution", "oracle", "steps", "source"]
)
def test_validate_overlay_rejects_non_config_sections(section):
    with pytest.raises(ValueError, match="may only patch"):
        validate_overlay({section: {}})


# ---- apply_config_override ------------------------------------------------


def test_apply_noop_when_none():
    cfg = _cfg()
    assert apply_config_override(cfg, None) is cfg
    assert apply_config_override(cfg, {}) is cfg


def test_apply_overrides_agent_and_preserves_siblings():
    out = apply_config_override(_cfg(), {"agent": {"timeout_sec": 42}})
    assert out.agent.timeout_sec == 42
    assert out.verifier.timeout_sec == 120  # sibling section untouched
    assert out.sandbox.cpus == 1


def test_apply_overrides_sandbox_by_field_name():
    # Regression: merging against by_alias=True made `sandbox` (alias
    # `environment`) un-overridable via its field name. Must work now.
    out = apply_config_override(_cfg(), {"sandbox": {"cpus": 8}})
    assert out.sandbox.cpus == 8


def test_apply_enforces_allowlist():
    with pytest.raises(ValueError, match="may only patch"):
        apply_config_override(_cfg(), {"verifier": {"timeout_sec": 1}})


def test_apply_revalidates_and_rejects_bad_value():
    with pytest.raises(ValidationError):
        apply_config_override(_cfg(), {"agent": {"timeout_sec": "nope"}})
