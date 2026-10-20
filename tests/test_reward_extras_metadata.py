"""Extra reward.json fields and out-of-range rewards no longer discard a trial.

Guards the fix for benchflow-ai/benchflow issue #1178: CooperBench-style
``{"reward": 1, "merge_ok": true, "strategy": "union"}``, KUMO-style numeric
diagnostics (``seed``, ``actions_used``) and AlgoTune speedups above 1 were
rejected as verifier errors. ``validate_reward_map``'s own strict default is
unchanged (see tests/test_reward_lenient.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.rewards.validation import (
    UNBOUNDED_REWARD_RANGE,
    has_out_of_range,
    move_non_numeric_extras_to_metadata,
    record_out_of_range,
    validate_reward_map,
)
from benchflow.rollout._setup import _ensure_canonical_rewards
from benchflow.task.verifier_core import Verifier


def _verifier(tmp_path: Path, **files: str) -> Verifier:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    paths = SimpleNamespace(
        reward_text_path=tmp_path / "reward.txt",
        reward_json_path=tmp_path / "reward.json",
    )
    return Verifier(SimpleNamespace(config=None), paths, None)  # type: ignore[arg-type]


def test_cooperbench_shape_is_kept_as_metadata() -> None:
    raw = {"reward": 1, "feature1_passed": True, "merge_ok": True, "strategy": "union"}
    moved, keys = move_non_numeric_extras_to_metadata(raw)
    assert keys == ["feature1_passed", "merge_ok", "strategy"]
    parsed = validate_reward_map(moved, source="reward JSON")
    assert parsed["reward"] == 1
    assert parsed["metadata"] == {
        "feature1_passed": True,
        "merge_ok": True,
        "strategy": "union",
    }


def test_merges_into_existing_metadata_mapping() -> None:
    moved, _ = move_non_numeric_extras_to_metadata(
        {"reward": 0, "metadata": {"a": 1}, "error": "boom"}
    )
    assert moved["metadata"] == {"a": 1, "error": "boom"}


def test_clean_map_untouched() -> None:
    raw = {"reward": 1.0, "metrics": {"acc": 0.5}, "details": "ok"}
    moved, keys = move_non_numeric_extras_to_metadata(raw)
    assert keys == [] and moved == raw


def test_record_out_of_range_keeps_raw_values() -> None:
    out = record_out_of_range({"reward": 5.0, "seed": 42, "acc": 0.5})
    assert out["reward"] == 5.0
    assert out["metadata"]["out_of_range"] == {"reward": 5.0, "seed": 42}


def test_record_out_of_range_noop_in_range() -> None:
    rewards = {"reward": 1.0, "acc": 0.5}
    assert record_out_of_range(rewards) == rewards


def test_declared_range_respected() -> None:
    assert not has_out_of_range(record_out_of_range({"reward": 5.0}, (0.0, 10.0)))


def test_algotune_reward_txt_above_one_is_kept_and_flagged(tmp_path: Path) -> None:
    rewards = _verifier(tmp_path, **{"reward.txt": "1.0002\n"})._parse_reward_text()
    assert rewards["reward"] == 1.0002
    assert rewards["metadata"]["out_of_range"] == {"reward": 1.0002}
    # The final canonicalization gate must not re-reject the flagged value.
    assert _ensure_canonical_rewards(rewards)["reward"] == 1.0002


def test_kumo_shape_reward_txt_plus_diagnostic_json(tmp_path: Path) -> None:
    diagnostics = {
        "task_id": "t",
        "answer": "A",
        "is_exact_match": True,
        "seed": 42,
        "actions_used": 12,
        "relative_action_count": 1.5,
    }
    v = _verifier(
        tmp_path,
        **{"reward.txt": "1\n", "reward.json": json.dumps(diagnostics)},
    )
    rewards = v._parse_reward_json_with_text_compat()
    assert rewards["reward"] == 1
    assert rewards["metadata"]["task_id"] == "t"
    assert rewards["metadata"]["out_of_range"] == {
        "seed": 42,
        "actions_used": 12,
        "relative_action_count": 1.5,
    }


@pytest.mark.parametrize("text", ["nan", "inf", "abc"])
def test_non_finite_reward_text_still_fails(tmp_path: Path, text: str) -> None:
    from benchflow.task.verifier_core import VerifierOutputParseError

    with pytest.raises(VerifierOutputParseError):
        _verifier(tmp_path, **{"reward.txt": text})._parse_reward_text()


def test_unbounded_range_still_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        validate_reward_map(
            {"reward": float("inf")}, reward_range=UNBOUNDED_REWARD_RANGE
        )


def test_declared_range_still_enforced_for_reward_json(tmp_path: Path) -> None:
    from benchflow.task.verifier_core import VerifierOutputParseError

    v = _verifier(tmp_path, **{"reward.json": json.dumps({"reward": 1.5})})
    v._reward_range = (0.0, 1.0)
    with pytest.raises(VerifierOutputParseError):
        v._parse_reward_json()


def test_undeclared_reward_json_above_one_is_kept_and_flagged(tmp_path: Path) -> None:
    v = _verifier(tmp_path, **{"reward.json": json.dumps({"reward": 1.2})})
    rewards = v._parse_reward_json()
    assert rewards["reward"] == 1.2
    assert rewards["metadata"]["out_of_range"] == {"reward": 1.2}
