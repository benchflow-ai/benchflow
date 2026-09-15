"""Guards automatic review planning and commit after baseline PR #1126."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import benchflow
from benchflow._utils.result_paths import iter_task_result_paths
from benchflow._utils.scoring import classify_score_outcome
from benchflow.review import automatic, persistence
from benchflow.review.evidence import EvidenceManifest
from benchflow.review.options import ReviewerConfig
from benchflow.rollout import _review
from benchflow.rollout._results import _build_rollout_result
from tests.test_review_evidence import LocalTransport
from tests.test_review_runtime import (
    WEIGHTED_RUBRIC,
    FakeRun,
    good_weighted_review,
    make_task,
)
from tests.test_review_runtime_automatic import make_bundle


@pytest.fixture
def auth(monkeypatch):
    resolver = Mock(return_value={"AZURE_API_KEY": "private-test-secret"})
    monkeypatch.setattr("benchflow.agents.env.resolve_agent_env", resolver)
    monkeypatch.setattr("benchflow.review.preflight.validate_reviewer_backend", Mock())
    return resolver


@pytest.fixture
def prepared(tmp_path, auth):
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    config = ReviewerConfig(
        agent="codex",
        model="azure/gpt5.6terra",
        environment="daytona",
        agent_env={"AZURE_API_KEY": "explicit-test-secret"},
    )
    plan = automatic.prepare_review(task, config)
    assert plan is not None
    return plan


@pytest.fixture
def saved(tmp_path, prepared):
    rollout = tmp_path / "jobs/physics__trial"
    _build_rollout_result(
        rollout,
        task_name="physics",
        rollout_name="physics__trial",
        agent="codex",
        agent_name="codex",
        model="azure/gpt5.6terra",
        n_tool_calls=1,
        prompts=["Calculate the energy."],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={"verify": 0.1},
        task_digest=prepared.task_digest,
        result_filename="solver.json",
    )
    (rollout / "config.json").write_text(
        json.dumps({"task_digest": prepared.task_digest})
    )
    make_bundle(tmp_path).rename(rollout / "evidence")
    return rollout


def _edit_solver(rollout: Path, **updates):
    path = rollout / "solver.json"
    data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data))


def test_no_rubric_never_requires_reviewer_credentials(tmp_path, auth):
    """Guards unchanged no-rubric execution from PR #1126."""
    task = make_task(tmp_path)
    auth.side_effect = AssertionError("A no-rubric task must not resolve reviewer auth")
    assert automatic.prepare_review(task, ReviewerConfig()) is None
    auth.assert_not_called()


@pytest.mark.parametrize(
    "payload", ["{", "[]", '{"criteria": []}', '{"criteria": [{"name":"x"}]}']
)
def test_malformed_rubric_fails_before_reviewer_auth(tmp_path, auth, payload):
    """Guards fail-closed task discovery after PR #1126."""
    task = make_task(tmp_path)
    (task / "rubric.json").write_text(payload)
    with pytest.raises(ValueError):
        automatic.prepare_review(task, ReviewerConfig())
    auth.assert_not_called()


def test_ambiguous_rubrics_fail_before_execution(prepared, auth):
    """Guards unambiguous automatic rubric selection after PR #1126."""
    auth.reset_mock()
    (prepared.task_path / "rubric.json").write_bytes(prepared.rubric_path.read_bytes())
    with pytest.raises(ValueError, match="ambiguous"):
        automatic.prepare_review(prepared.task_path, prepared.config)
    auth.assert_not_called()


def test_symlink_rubric_cannot_import_an_untrusted_file(tmp_path, auth):
    """Guards task-local rubric authority after PR #1126."""
    task = make_task(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text(json.dumps(WEIGHTED_RUBRIC))
    (task / "rubric.json").symlink_to(target)
    with pytest.raises(ValueError, match="regular task file"):
        automatic.prepare_review(task, ReviewerConfig())
    auth.assert_not_called()


def test_verifier_judge_dialect_is_not_automatically_reviewed(tmp_path, auth):
    """Guards the distinct legacy verifier dialect from PR #1126."""
    task = make_task(tmp_path)
    (task / "verifier/rubric.json").write_text(
        json.dumps(
            {"criteria": [{"id": "correct", "match_criteria": "Correct answer"}]}
        )
    )
    assert automatic.prepare_review(task, ReviewerConfig()) is None
    auth.assert_not_called()


def test_explicit_reviewer_routing_and_private_auth(prepared, auth):
    """Guards shared provider routing and credential redaction after PR #1126."""
    auth.assert_called_once_with(
        "codex-acp", "azure/gpt5.6terra", {"AZURE_API_KEY": "explicit-test-secret"}
    )
    assert prepared.config.environment == "daytona"
    assert prepared.config.agent_env == {"AZURE_API_KEY": "private-test-secret"}
    metadata = prepared.metadata()
    assert metadata["reviewer"]["agent_env_keys"] == ["AZURE_API_KEY"]
    assert "test-secret" not in json.dumps(metadata)


def test_missing_reviewer_auth_has_actionable_preflight_error(prepared, auth):
    """Guards fail-before-solver authentication after PR #1126."""
    auth.side_effect = ValueError("AZURE_API_KEY or supported OAuth login is required")
    with pytest.raises(ValueError, match=r"Reviewer preflight.*AZURE_API_KEY"):
        automatic.prepare_review(prepared.task_path, prepared.config)


def test_solver_checkpoint_is_not_published_as_a_completed_trial(saved):
    """Guards checkpoint/final artifact separation after PR #1126."""
    assert (saved / "solver.json").is_file()
    assert (saved / "trajectory/acp_trajectory.jsonl").is_file()
    assert (saved / "prompts.json").is_file()
    assert not (saved / "result.json").exists()
    assert not (saved / "results.jsonl").exists()
    assert not (saved / "rewards.jsonl").exists()
    assert not (saved / "trajectory/trajectory.json").exists()
    assert iter_task_result_paths(saved.parent) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tests_pass", "blocker_pass", "reward", "passed"),
    [
        (True, True, 0.8, True),
        (False, True, 0.0, False),
        (True, False, 0.0, False),
    ],
)
async def test_finish_review_and_commit_preserve_all_components(
    prepared,
    saved,
    monkeypatch,
    tests_pass,
    blocker_pass,
    reward,
    passed,
):
    """Guards the full mocked reviewer/commit flow after PR #1126."""
    _edit_solver(saved, rewards={"reward": float(tests_pass)})
    original = (saved / "solver.json").read_bytes()
    payload = good_weighted_review(saved.name)
    payload["checks"]["safety_gate"]["outcome"] = "pass" if blocker_pass else "fail"
    fake = FakeRun(review_payload=payload)
    monkeypatch.setattr(benchflow, "run", fake)
    scoring = await automatic.finish_review(prepared, saved)
    assert scoring.status == "complete"
    assert not (saved / "result.json").exists()
    reviewer = saved / scoring.reviewer_run
    assert (reviewer / "result.json").exists()
    assert fake.configs[0].purpose == "reviewer"
    assert fake.configs[0].parent_rollout == saved.name
    assert len(list((saved / "scoring").glob("*.json"))) == 1
    committed = persistence.commit_scoring_result(saved, scoring)
    assert committed["rewards"] == {
        "reward": reward,
        "rubric_reward": 0.8,
        "verifier_reward": float(tests_pass),
    }
    assert committed["scoring"]["passed"] is passed
    assert classify_score_outcome(committed) == ("passed" if passed else "failed")
    assert json.loads((saved / "result.json").read_text()) == committed
    assert (saved / "solver.json").read_bytes() == original
    trainer = json.loads((saved / "results.jsonl").read_text())
    assert trainer["passed"] is passed
    assert trainer["reward"] == reward
    assert iter_task_result_paths(saved.parent) == [saved / "result.json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_reward", [None, True, float("nan"), float("inf")])
async def test_invalid_verifier_reward_becomes_unscored_error(
    prepared, saved, monkeypatch, raw_reward
):
    """Guards finite scoring error persistence after PR #1126."""
    reviewer = AsyncMock(side_effect=AssertionError("reviewer must not run"))
    monkeypatch.setattr("benchflow.review.runner.run_review", reviewer)
    _edit_solver(saved, rewards={"reward": raw_reward})
    scoring = await automatic.finish_review(prepared, saved)
    assert scoring.status == "error" and scoring.passed is None
    assert scoring.tests_pass is None
    assert scoring.numeric_rewards() is None
    assert scoring.error
    reviewer.assert_not_awaited()
    assert len(list((saved / "scoring").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_detail_free_reviewer_exception_still_saves_failure(
    prepared, saved, monkeypatch
):
    """Guards transport exception evidence after PR #1126."""
    monkeypatch.setattr(
        "benchflow.review.runner.run_review", AsyncMock(side_effect=TimeoutError())
    )
    scoring = await automatic.finish_review(prepared, saved)
    assert scoring.status == "error"
    assert "TimeoutError" in scoring.error
    assert scoring.tests_pass is True
    committed = persistence.commit_scoring_result(saved, scoring)
    assert committed["rewards"] is None
    assert committed["verifier_error"]
    assert classify_score_outcome(committed) == "verifier_errored"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["rubric", "task", "workspace"])
async def test_changed_evidence_rejected_before_reviewer(
    prepared, saved, monkeypatch, change
):
    """Guards immutable review inputs after PR #1126."""
    reviewer = AsyncMock(side_effect=AssertionError("reviewer must not run"))
    monkeypatch.setattr("benchflow.review.runner.run_review", reviewer)
    target = {
        "rubric": prepared.rubric_path,
        "task": prepared.task_path / "task.md",
        "workspace": saved / "evidence/workspace/paper.txt",
    }[change]
    target.write_text("Changed after solver execution")
    scoring = await automatic.finish_review(prepared, saved)
    assert scoring.status == "error"
    assert scoring.numeric_rewards() is None
    reviewer.assert_not_awaited()


@pytest.mark.asyncio
async def test_workspace_freezes_before_capture_and_keeps_actual_cwd(
    tmp_path, monkeypatch
):
    """Guards terminal capture ordering after PR #1126."""
    order = []

    async def disconnect():
        order.append("disconnect")

    async def stop(env, user):
        order.append("stop")

    async def capture(env, cwd, destination, *, artifacts, excluded_paths):
        assert excluded_paths == ()
        assert artifacts == []
        assert cwd == "/research/work"
        assert destination == tmp_path / "evidence"
        order.append("capture")

    rollout = SimpleNamespace(
        _review_plan=object(),
        _config=SimpleNamespace(purpose="task", sandbox_user="agent"),
        _env=object(),
        _agent_env={},
        _planes=SimpleNamespace(quiesce_agent=stop),
        _task=SimpleNamespace(config=SimpleNamespace(artifacts=[])),
        _agent_cwd="/research/work",
        disconnect=disconnect,
        _require_rollout_dir=lambda: tmp_path,
    )
    monkeypatch.setattr(_review, "capture_task_evidence", capture)
    await _review.capture_terminal_workspace(rollout)
    assert order == ["disconnect", "stop", "capture"]


@pytest.mark.asyncio
async def test_daytona_session_fifos_keep_terminal_evidence_scorable(tmp_path):
    """Guards terminal capture on Daytona /root workspaces after PR #1126.

    Daytona's entrypoint FIFOs under /root/.daytona made capture fail, which
    set _export_error; finish_review then ended in a scoring error with
    rewards null even though the verifier had run.
    """
    workspace = tmp_path / "root"
    entrypoint = workspace / ".daytona/sessions/entrypoint/entrypoint_command"
    entrypoint.mkdir(parents=True)
    os.mkfifo(entrypoint / "input.pipe")
    (workspace / "paper.pdf").write_bytes(b"%PDF-1.7 solver paper")
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()

    async def idle(*_args):
        return None

    rollout = SimpleNamespace(
        _review_plan=object(),
        _config=SimpleNamespace(purpose="task", sandbox_user=None),
        _env=LocalTransport(),
        _agent_env={},
        _planes=SimpleNamespace(quiesce_agent=idle),
        _task=SimpleNamespace(config=SimpleNamespace(artifacts=[])),
        _agent_cwd=str(workspace),
        disconnect=idle,
        _require_rollout_dir=lambda: rollout_dir,
    )

    await _review.capture_terminal_workspace(rollout)

    assert getattr(rollout, "_export_error", None) is None
    manifest = EvidenceManifest.model_validate_json(
        (rollout_dir / "evidence" / "manifest.json").read_text()
    )
    assert [(entry.original_path, entry.reason) for entry in manifest.exclusions] == [
        (str(workspace / ".daytona"), "sandbox_runtime")
    ]
    assert (
        rollout_dir / "evidence" / "workspace" / "paper.pdf"
    ).read_bytes() == b"%PDF-1.7 solver paper"


def test_reviewer_child_never_recursively_preflights(monkeypatch):
    """Guards child execution recursion after PR #1126."""
    prepare = Mock(side_effect=AssertionError("reviewer cannot review itself"))
    monkeypatch.setattr(_review, "prepare_review", prepare)
    _review.prepare_terminal_review(
        SimpleNamespace(_config=SimpleNamespace(purpose="reviewer"))
    )
    prepare.assert_not_called()


def test_commit_failure_never_publishes_partial_parent(saved, monkeypatch):
    """Guards final-result commit ordering after PR #1126."""
    from tests.test_automatic_review_scoring import _score

    monkeypatch.setattr(
        "benchflow.trajectories.results.write_rollout_results_jsonl",
        Mock(side_effect=OSError("disk full while saving trainer result")),
    )
    with pytest.raises(OSError, match="disk full"):
        persistence.commit_scoring_result(saved, _score())
    assert not (saved / "result.json").exists()
    assert (saved / "solver.json").exists()
