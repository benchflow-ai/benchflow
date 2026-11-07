"""Guard the K1/K2/P78 merge of 6f9d482a and 7b4907c8."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import kill_record
from benchflow.rollout import _review
from benchflow.rollout_planes import DefaultRolloutPlanes
from benchflow.sandbox.lockdown import (
    _kill_sandbox_user_procs_keeping,
    harden_before_verify,
)
from tests.test_keep_services_kill import _Async, _env_on, _NetProc, _task
from tests.test_verifier_process_quiescence import _LIVE, _ROOT, UID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt,status",
    [
        ("BF killed 2\n", 0),
        ("BF killed 2\nBF killed 3\n", 0),
        ("BF killed 2 end\n", None),
        ("BF killed 2 end\n", False),
        ("BF killed 2 end\nBF killed 3", 0),
    ],
)
async def test_keeping_requires_the_same_complete_kill_evidence(receipt, status):
    """Guards K1 receipt validation when merged into K2 at 7b4907c8."""
    result = SimpleNamespace(
        return_code=status,
        stdout="BF waited 0\nBF kept count 0\n" + receipt,
        stderr="",
    )
    env = SimpleNamespace(exec=AsyncMock(return_value=result))
    record = {}
    with kill_record.recording_into(record), pytest.raises(RuntimeError):
        await _kill_sandbox_user_procs_keeping(env, "agent", (5000,), wait=True)
    assert record == {}


@pytest.mark.asyncio
async def test_keeping_never_kills_a_root_replacement(tmp_path):
    """Guards K1-R2-2 in K2's per-pid kill after merging 7b4907c8."""
    proc = _NetProc(tmp_path)
    service = proc.add(_LIVE)
    proc.hold(service, proc.listen(5000))
    parent = proc.add(_LIVE)
    child = proc.add(_LIVE)
    (proc.root / str(parent) / "pdeath").write_text(str(child))
    outcome = await _kill_sandbox_user_procs_keeping(
        _Async(proc), "agent", (5000,), wait=False
    )
    assert (proc.root / str(child) / "status").read_text() == _ROOT
    assert [p.pid for p in outcome.kept] == [service]
    assert proc.stopped(service)
    assert outcome.killed == 2


@pytest.mark.asyncio
async def test_kept_service_release_and_end_write_to_the_branch_child(
    tmp_path, monkeypatch
):
    """Guards K1 child custody through K2's release/end hooks at 7b4907c8."""
    from tests.test_keep_services_verify import _Planes
    from tests.test_verifier_process_quiescence import _verifying_rollout

    trial = _verifying_rollout(tmp_path, monkeypatch, hardening_count=0)
    planes = _Planes()
    planes.order = []
    trial._planes = planes
    parent = trial._rollout_dir
    original = {"user": "agent", "killed": 9}
    (parent / "kill-before-grading.json").write_text(json.dumps(original))
    child = parent / "branches" / "f1" / "children" / "c1"
    child.mkdir(parents=True)
    trial._rollout_dir = child
    trial._branch_child_active = True
    monkeypatch.setattr(
        "benchflow.rollout.capture_terminal_workspace",
        _review.capture_terminal_workspace,
    )
    assert await trial.verify() == {"reward": 1.0}
    written = json.loads((child / "kill-before-grading.json").read_text())
    assert written["killed"] == written["killed_total"] == 3
    assert written["kept_services_released"] == 1
    assert written["kept_services_ended"] == 2
    assert planes.order == ["test", "end"]
    assert json.loads((parent / "kill-before-grading.json").read_text()) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [False, True])
async def test_turn_end_and_grading_counts_survive_without_optional_capture(
    tmp_path, monkeypatch, keep
):
    """Guards K1 accounting at P78's mandatory handoff in 7b4907c8."""
    proc = _NetProc(tmp_path)
    task = _task([{"port": 5000}] if keep else [])
    service = proc.add(_LIVE) if keep else None
    if service is not None:
        proc.hold(service, proc.listen(5000))
    writer = proc.add(_LIVE)
    root = proc.add(_ROOT)
    trial = SimpleNamespace(
        _branch_child_active=False,
        _review_plan=None,
        _config=SimpleNamespace(
            purpose="task", sandbox_user="agent", freeze_workspace=False
        ),
        _env=proc,
        _task=task,
        _planes=DefaultRolloutPlanes(),
        disconnect=AsyncMock(),
        _require_rollout_dir=lambda: tmp_path,
    )
    monkeypatch.setattr(_review, "_freeze_requested", lambda _: False)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _: "no recovery contract",
    )
    await _review.capture_terminal_workspace(trial)
    record = json.loads((tmp_path / "kill-before-grading.json").read_text())
    assert record["quiesce_killed"] == record["killed_total"] == 1
    assert str(writer) not in proc.live_pids()
    assert (proc.root / str(root) / "status").read_text() == _ROOT
    with kill_record.recording_into(record):
        await harden_before_verify(_env_on(proc), task, "agent", workspace="/app")
    assert record["killed"] == 0
    assert record["killed_total"] == record["quiesce_killed"] == 1
    if service is not None:
        assert [p["pid"] for p in record["kept_services"]] == [service]
        assert record["kept_services_turn_end"]["killed"] == 1
        assert proc.stopped(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_keeping_counts_uid_role_transitions_once(tmp_path, reverse):
    """Guards K1-R3-2 while keeping K2's numeric socket identities."""
    proc = _NetProc(tmp_path)
    service = proc.add(_LIVE)
    proc.hold(service, proc.listen(5000))
    effective = f"State:\tS (sleeping)\nUid:\t0\t{UID}\t0\t{UID}\nThreads:\t1\n"
    before, after = (_LIVE, effective) if reverse else (effective, _LIVE)
    writer = proc.add(before, term_status=after)
    outcome = await _kill_sandbox_user_procs_keeping(
        _Async(proc), "agent", (5000,), wait=False
    )
    assert outcome.killed == 1
    assert str(writer) not in proc.live_pids()
    assert [p.pid for p in outcome.kept] == [service]
    assert proc.stopped(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [False, True])
async def test_what_a_collect_hook_leaves_is_stopped_before_the_capture(
    tmp_path, monkeypatch, keep
):
    """Guards the fix to PR #84's collect hooks: a main hook runs as the
    agent's account, so what it starts there (the agent's git config can
    start anything) is killed after the hooks, a kept service stays kept, and
    the counts add to the turn-end record."""
    from benchflow.task.config import VerifierCollectConfig

    users = []

    class Hooked(_NetProc):
        async def exec(self, command, **kwargs):
            if command == "snapshot":
                users.append(kwargs.get("user"))
                left.append(self.add(_LIVE))  # left running by what it ran
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            return await super().exec(command, **kwargs)

    left = []
    proc = Hooked(tmp_path)
    task = _task([{"port": 5000}] if keep else [])
    task.config.verifier.collect = [VerifierCollectConfig(command="snapshot")]
    service = proc.add(_LIVE) if keep else None
    if service is not None:
        proc.hold(service, proc.listen(5000))
    writer = proc.add(_LIVE)
    trial = SimpleNamespace(
        _branch_child_active=False,
        _review_plan=None,
        _config=SimpleNamespace(
            purpose="task",
            sandbox_user="agent",
            freeze_workspace=False,
            primary_agent="claude-agent-acp",
        ),
        _env=proc,
        _task=task,
        _planes=DefaultRolloutPlanes(),
        disconnect=AsyncMock(),
        _require_rollout_dir=lambda: tmp_path,
    )
    monkeypatch.setattr(_review, "_freeze_requested", lambda _: False)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _: "no recovery contract",
    )
    await _review.capture_terminal_workspace(trial)
    assert users == ["agent"]
    assert str(writer) not in proc.live_pids()
    assert left and not {str(pid) for pid in left} & set(proc.live_pids())
    record = json.loads((tmp_path / "kill-before-grading.json").read_text())
    # The turn-end kill took the writer, the second one what the hook left.
    assert record["quiesce_killed"] == record["killed_total"] == 2
    assert record["killed"] == 1
    if service is not None:
        assert str(service) in proc.live_pids() and proc.stopped(service)
