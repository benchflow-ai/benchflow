"""Guard PR #14's saved-runtime and frozen-rubric provenance boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow import assessment as a
from benchflow.assessment.store import EVENTS_FILE
from tests.assessment._fixtures import RUBRIC_ID, RUBRIC_VERSION, world


@pytest.mark.parametrize("change", ["substitute", "omit", "relabel", "extra"])
def test_supplied_reference_cannot_drop_the_result_sandbox(tmp_path: Path, change: str):
    """PR #14: a different observed runtime cannot prove the result's release."""
    w = world(tmp_path)
    refs = [
        w.first.ref(runtimes=[a.parse_runtime_ref("docker:extra-verifier:verifier")]),
        w.second.ref(),
    ]
    before = (w.first.root / "result.json").read_bytes()
    identity = refs[0]["attempt_id"]
    runtimes = refs[0]["runtime_refs"]
    if change == "substitute":
        runtimes[0]["runtime_id"] = "unrelated-already-gone"
    elif change == "omit":
        del runtimes[0]
    elif change == "relabel":
        runtimes[0]["source"] = "declared"
    else:
        runtimes.append(dict(runtimes[0], runtime_id="another-result-claim"))
    with pytest.raises(a.EvidenceRejected, match=r"result\.sandbox_id"):
        a.verify_attempt(refs[0])
    with pytest.raises(a.EvidenceRejected, match=r"result\.sandbox_id"):
        a.create(
            w.sidecar,
            refs,
            rubric=w.rubric,
            rubric_id=RUBRIC_ID,
            rubric_version=RUBRIC_VERSION,
            protocol=w.protocol,
        )
    assert not (w.sidecar / EVENTS_FILE).exists()
    assert refs[0]["attempt_id"] == identity
    assert (w.first.root / "result.json").read_bytes() == before


def test_declared_runtime_does_not_fabricate_a_result_source(tmp_path: Path):
    """PR #14: declared runtimes remain supported when the result has no sandbox."""
    w = world(tmp_path)
    result = w.first.root / "result.json"
    document = json.loads(result.read_text())
    document["sandbox_id"] = None
    result.write_text(json.dumps(document))
    ref = w.first.ref(runtimes=[a.parse_runtime_ref("docker:declared-allocation")])
    a.verify_attempt(ref)
    ref["runtime_refs"][0]["source"] = "result.sandbox_id"
    with pytest.raises(a.EvidenceRejected, match=r"result\.sandbox_id"):
        a.verify_attempt(ref)


@pytest.mark.parametrize("phase", ["distillation", "evaluation"])
def test_taskless_collector_assignments_cannot_claim_a_frozen_rubric(
    tmp_path: Path, phase: str
):
    """PR #14 / frozen contract §5: missing rubric identity stays research history."""
    w = world(tmp_path)
    w.create()
    w.release()
    left = w.collector.add_artifact(w.first)
    right = w.collector.add_artifact(w.second)
    assignment = w.collector.assign(left, right, phase=phase)
    before = (w.sidecar / EVENTS_FILE).read_bytes()
    with pytest.raises(a.EvidenceRejected, match=r"task version|frozen rubric"):
        w.assign(assignment)
    assert (w.sidecar / EVENTS_FILE).read_bytes() == before
    assert a.load(w.sidecar).state == "awaiting_assignment"


@pytest.mark.parametrize("phase", ["distillation", "evaluation"])
def test_rubric_bound_collector_assignments_remain_eligible(tmp_path: Path, phase: str):
    """PR #14: either collector phase works when all frozen identities are present."""
    w = world(tmp_path)
    w.create()
    w.release()
    task_id = "explicit-frozen-task"
    rubric = a.load(w.sidecar).created.payload["rubric"]["sha256"]
    w.collector.add_task(task_id, rubric_hash=rubric)
    left = w.collector.add_artifact(w.first, task_id=task_id)
    right = w.collector.add_artifact(w.second, task_id=task_id)
    w.assignment_id = w.collector.assign(left, right, phase=phase, task_id=task_id)
    w.assign()
    w.vote("left")
    w.import_vote()
    assert a.reduce(w.sidecar).label["value"] == 1
