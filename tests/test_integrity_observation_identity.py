"""Every observation has an opaque, unique identity fixed before any redaction.

The route relations a live run stores are keyed on it, and it is stored in
the trajectory the same way, so a re-audit finds them whatever redaction
does to a tool call's own id: an id it rewrites, or two ids it makes equal.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from benchflow.integrity.contract import TaskBinding
from benchflow.integrity.emit import INTEGRITY_DIRNAME, emit_integrity
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    stamp_observation_ids,
)
from tests.test_integrity import _evidence, _tool

HOST = "http://grading-service:9005"
ADMIN = [("admin", f"{HOST}/_admin", "Hidden")]


def _binding() -> TaskBinding:
    return TaskBinding.model_validate(
        {
            "resources": [
                {"id": rid, "selector": sel, "class": cls, "reason": "x"}
                for rid, sel, cls in ADMIN
            ]
        }
    )


def _stored(trajectory):
    return [
        json.loads(line)
        for line in redact_acp_trajectory_jsonl(trajectory).splitlines()
        if line.strip()
    ]


def _agent_events(evidence):
    path = evidence.trial_dir / INTEGRITY_DIRNAME / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [
        (e["event_type"], e["resource"], e["resource_class"])
        for e in events
        if e["actor_class"] == "Agent"
    ]


def _live_and_reaudit(tmp_path, trajectory):
    stamp_observation_ids(trajectory)  # as the rollout does, before any write
    evidence = _evidence(tmp_path, trajectory, binding=_binding())
    live = emit_integrity(evidence)
    live_events = _agent_events(evidence)
    audit = dataclasses.replace(
        evidence, trajectory=_stored(trajectory), source="audit_trial"
    )
    again = emit_integrity(audit)
    return (live, live_events), (again, _agent_events(audit))


def _same(live, again):
    (claim, events), (claim2, events2) = live, again
    assert claim2["core_verdict"] == claim["core_verdict"]
    assert claim2["exploited"] == claim["exploited"]
    assert claim2["final_flags"] == claim["final_flags"]
    assert events2 == events


@pytest.mark.parametrize(("path", "blamed"), [("/public", False), ("/_admin/x", True)])
def test_a_call_id_redaction_rewrites_keeps_its_relations(tmp_path, path, blamed):
    trajectory = [_tool("navigate", call_id="api_key=dummy-id", url=HOST + path)]
    assert "dummy-id" not in redact_acp_trajectory_jsonl(trajectory)
    live, again = _live_and_reaudit(tmp_path, trajectory)
    _same(live, again)
    assert live[0]["exploited"] is blamed
    assert not live[0]["final_flags"]["unknown_event"]


def test_two_calls_whose_ids_collide_after_redaction_stay_two(tmp_path):
    trajectory = [
        _tool("navigate", call_id="api_key=first-call-id", url=f"{HOST}/_admin/x"),
        _tool("navigate", call_id="api_key=second-call-id", url=f"{HOST}/public"),
    ]
    stored_ids = {event["tool_call_id"] for event in _stored(trajectory)}
    assert len(stored_ids) == 1  # redaction makes them equal
    live, again = _live_and_reaudit(tmp_path, trajectory)
    _same(live, again)
    fetches = [e for e in live[1] if e[0] in {"AgentRead", "AgentNetworkRequest"}]
    assert fetches == [
        ("AgentRead", "admin", "Hidden"),
        ("AgentNetworkRequest", None, None),
    ]
    assert live[0]["exploited"]
    assert not live[0]["final_flags"]["unknown_event"]


def test_identities_are_opaque_unique_shared_by_sightings_and_kept():
    trajectory = [
        _tool("fetch", call_id="a", url=f"{HOST}/x"),
        {**_tool("fetch", call_id="a", url=f"{HOST}/x"), "status": "pending"},
        _tool("fetch", call_id="b", url=f"{HOST}/y"),
        *(_tool("fetch", call_id="", url=f"{HOST}/{n}") for n in range(200)),
    ]
    stamp_observation_ids(trajectory)
    ids = [event.get("observation_id") for event in trajectory]
    assert all(isinstance(i, str) and i.startswith("obs-") for i in ids)
    assert ids[0] == ids[1] and len(set(ids)) == len(ids) - 1
    assert redact_trajectory_obj(trajectory) == trajectory  # redaction keeps them
    before = list(ids)
    stamp_observation_ids(trajectory)
    assert [event["observation_id"] for event in trajectory] == before


def test_the_result_and_the_stored_trajectory_carry_the_same_identities(tmp_path):
    """The rollout's result builder fixes the identities before the final
    write, on the list the result keeps and the live audit reads."""
    from datetime import datetime

    from benchflow.rollout._results import _build_rollout_result

    trajectory = [
        _tool("navigate", call_id="api_key=first-call-id", url=f"{HOST}/_admin/x"),
        _tool("navigate", call_id="api_key=second-call-id", url=f"{HOST}/public"),
    ]
    rollout_dir = tmp_path / "hello__abcd1234"
    rollout_dir.mkdir()
    result = _build_rollout_result(
        rollout_dir,
        task_name="hello",
        rollout_name="hello__abcd1234",
        agent="claude-agent-acp",
        agent_name="claude",
        model="claude-haiku-4-5",
        n_tool_calls=2,
        prompts=["hi"],
        error=None,
        verifier_error=None,
        trajectory=trajectory,
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime(2026, 1, 1, 12, 0, 0),
        timing={"agent": 3.0},
    )
    path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    stored = [json.loads(line) for line in path.read_text().splitlines() if line]
    kept = [event.get("observation_id") for event in result.trajectory]
    assert [event.get("observation_id") for event in stored] == kept
    assert all(kept) and len(set(kept)) == 2
    assert "first-call-id" not in path.read_text()


def test_a_stored_call_without_an_identity_is_unknown(tmp_path):
    """An older trial, or a trajectory written without identities: the
    relations cannot be found, so the fetch is unknown, never exact."""
    trajectory = [_tool("navigate", call_id="c1", url=f"{HOST}/public")]
    stamp_observation_ids(trajectory)
    evidence = _evidence(tmp_path, trajectory, binding=_binding())
    emit_integrity(evidence)
    stored = [
        {k: v for k, v in event.items() if k != "observation_id"}
        for event in _stored(trajectory)
    ]
    claim = emit_integrity(
        dataclasses.replace(evidence, trajectory=stored, source="audit_trial")
    )
    assert claim["final_flags"]["unknown_event"] and not claim["exploited"]


def test_a_result_read_back_from_its_stored_trajectory_is_reaudited(tmp_path):
    """Resumed scoring reads the stored (redacted) trajectory back: the audit
    reuses the live run's relations instead of deriving new ones from it."""
    from benchflow.integrity.trial import rollout_evidence_source
    from benchflow.models import RolloutResult

    trajectory = [_tool("navigate", call_id="api_key=dummy-id", url=f"{HOST}/x")]
    stamp_observation_ids(trajectory)
    run = tmp_path / "run"
    (run / "trajectory").mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"task_name": "t"}))
    (run / "trajectory" / "acp_trajectory.jsonl").write_text(
        redact_acp_trajectory_jsonl(trajectory) + "\n"
    )
    assert rollout_evidence_source(RolloutResult.load(run)) == "rollout_stored"
    assert rollout_evidence_source(RolloutResult(task_name="t")) == "rollout"

    evidence = _evidence(tmp_path / "trial", trajectory, binding=_binding())
    live = emit_integrity(evidence)
    resumed = emit_integrity(
        dataclasses.replace(
            evidence, trajectory=_stored(trajectory), source="rollout_stored"
        )
    )
    assert resumed["final_flags"] == live["final_flags"]
    keys = json.loads(
        (evidence.trial_dir / INTEGRITY_DIRNAME / "route_keys.json").read_text()
    )
    assert list(keys["observations"]) == [f"{trajectory[0]['observation_id']}#0"]
