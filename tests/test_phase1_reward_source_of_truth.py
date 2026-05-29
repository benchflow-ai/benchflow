"""Guards v0.5 Phase 1 — the canonical VerifyResult is the live source of truth.

Phase 1 routes the reward path through ``Reward.score(node) -> VerifyResult``
(``docs/architecture.md`` § "The four contracts"): the verifier's reward map is
lifted ONCE during scoring into a canonical ``VerifyResult``
(``verify_result_from_reward_map``), persisted to ``verifier/verify_result.json``,
and the trainer export READS it instead of re-deriving the reward from the legacy
dict at export time. ``result.json['rewards']`` stays the legacy dict for the
existing consumers.

These tests pin that direction so a later refactor can't quietly reintroduce the
downgrade-on-export, or default the ``(space, granularity)`` tag at the writer.
"""

from __future__ import annotations

import json
from datetime import datetime

from benchflow.rewards.node import verify_result_from_reward_map
from benchflow.rewards.protocol import VerifyResult
from benchflow.rollout import _build_rollout_result
from benchflow.trajectories.export import (
    ROLLOUT_ARTIFACT_RELPATH,
    write_rollout_verifiers_jsonl,
)


def test_verify_result_from_reward_map_builds_canonical():
    """The single conversion point lifts dict -> tagged VerifyResult + events."""
    vr = verify_result_from_reward_map(
        {
            "reward": 0.75,
            "exact_match": 1.0,
            "rubric": [{"name": "clarity", "score": 0.5}],
        }
    )
    assert isinstance(vr, VerifyResult)
    assert vr.reward == 0.75
    assert vr.items["exact_match"] == 1.0
    assert vr.items["clarity"] == 0.5
    assert vr.space == "output"
    assert vr.granularity == "terminal"
    # One terminal Output event for the headline + one process event per rubric.
    terminal = [e for e in vr.events if e.type == "terminal"]
    process = [e for e in vr.events if e.type == "process"]
    assert len(terminal) == 1
    assert terminal[0].space == "output" and terminal[0].granularity == "terminal"
    assert len(process) == 1
    assert process[0].source == "clarity"
    assert process[0].granularity == "step"


def test_verify_result_from_reward_map_handles_none():
    """A crashed/timed-out verifier yields reward 0.0 with error populated."""
    vr = verify_result_from_reward_map(None, error="verifier timed out")
    assert vr.reward == 0.0
    assert vr.items == {}
    assert vr.events == []
    assert vr.error == "verifier timed out"


def test_export_reads_verify_result_not_the_dict():
    """write_rollout_verifiers_jsonl READS the VerifyResult when supplied.

    Pass a VerifyResult and a deliberately DIFFERENT rewards dict; the record
    must reflect the VerifyResult (0.9), never the dict (0.1) — proving export
    reads the canonical result rather than re-deriving from the legacy map.
    """
    import tempfile
    from pathlib import Path

    vr = verify_result_from_reward_map({"reward": 0.9})
    with tempfile.TemporaryDirectory() as d:
        rec = write_rollout_verifiers_jsonl(
            Path(d),
            task_id="t1",
            prompts=["do it"],
            trajectory=[],
            rewards={"reward": 0.1},  # intentionally different — must be ignored
            verify_result=vr,
            model="m",
            environment="e",
        )
    assert rec["reward"] == 0.9
    assert rec["info"]["reward_metadata"]["space"] == "output"
    assert rec["info"]["reward_metadata"]["granularity"] == "terminal"


def test_build_result_writes_canonical_verify_result_json(tmp_path):
    """_build_rollout_result persists verify_result.json and keeps the legacy dict.

    The canonical Reward-plane artifact carries (space, granularity) and the
    event list; result.json['rewards'] stays the legacy dict for back-compat;
    and the headline reward agrees across both, sourced from the VerifyResult.
    """
    rewards = {"reward": 1.0, "rubric": [{"name": "clarity", "score": 0.5}]}
    vr = verify_result_from_reward_map(rewards)
    _build_rollout_result(
        tmp_path,
        task_name="hello-world-task",
        rollout_name="r1",
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards=rewards,
        verify_result=vr,
        started_at=datetime.now(),
        timing={},
    )

    vr_json = json.loads((tmp_path / "verifier" / "verify_result.json").read_text())
    assert vr_json["reward"] == 1.0
    assert vr_json["space"] == "output"
    assert vr_json["granularity"] == "terminal"
    assert any(e["type"] == "process" for e in vr_json["events"])

    result_json = json.loads((tmp_path / "result.json").read_text())
    # Legacy dict preserved for the ~10 existing consumers (no breaking change).
    assert result_json["rewards"] == rewards

    # The trainer artifact's headline agrees with the canonical result.
    rec = json.loads((tmp_path / ROLLOUT_ARTIFACT_RELPATH).read_text().strip())
    assert rec["reward"] == 1.0


def test_build_result_derives_verify_result_when_absent(tmp_path):
    """Callers that pass only the legacy dict still get a sourced verify_result.json.

    SDK/Evaluation builders don't thread a VerifyResult; _build_rollout_result
    must derive one from the dict so every rollout emits the canonical artifact.
    """
    _build_rollout_result(
        tmp_path,
        task_name="t",
        rollout_name="r",
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 0.5},
        started_at=datetime.now(),
        timing={},
    )
    vr_json = json.loads((tmp_path / "verifier" / "verify_result.json").read_text())
    assert vr_json["reward"] == 0.5
    assert vr_json["space"] == "output"
