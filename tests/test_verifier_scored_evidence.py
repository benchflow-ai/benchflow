"""FIX-49 r4: verifier-controlled evidence cannot drop a rollout.

The changed-behavior cases fail at c278f22a6; the own-failure and valid-reward
controls also run there. The fake sandbox only supplies what a
real test.sh can write; the real Verifier and rollout scoring boundary run.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.rollout import _setup
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.lockdown import GuardMarker
from benchflow.task import Task
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier
from benchflow.task.verifier_errors import PluginGuardMarkersUnreadableError

GUARD = "_benchflow_guard_fix49r4"


async def run(
    tmp_path,
    monkeypatch,
    files,
    *,
    markers=None,
    rc=0,
    hang=False,
    exec_error=None,
    later_hardening_failure=None,
    marker_output=None,
):
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "task.toml").write_text('version = "1.0"\n')
    (task_dir / "instruction.md").write_text("Do the task.\n")
    task = Task(task_dir)
    task.config.verifier.timeout_sec = 0.05 if hang else 60
    task.config.verifier.env = {"PYTEST_ADDOPTS": f"-p {GUARD}"}
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()

    async def execute(command, **kwargs):
        if "test-stdout.txt" in command:
            if exec_error is not None:
                raise exec_error
            for name, contents in files.items():
                path = paths.verifier_dir / name
                if contents == "<directory>":
                    path.mkdir()
                elif contents == "<fifo>":
                    os.mkfifo(tmp_path / "trial/verifier" / name)
                elif contents == "<symlink>":
                    (tmp_path / "trial/verifier" / name).symlink_to("/dev/zero")
                elif contents == "<oversized>":
                    path.write_text(" " * (16 * 1024 * 1024) + "1")
                elif isinstance(contents, bytes):
                    path.write_bytes(contents)
                else:
                    path.write_text(contents)
            if hang:
                await asyncio.sleep(60)
            return ExecResult(stdout="", stderr="", return_code=rc)
        return ExecResult(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(is_mounted=True, upload_dir=AsyncMock(), exec=execute)
    verifier = Verifier(task, paths, sandbox)
    if isinstance(markers, Exception):
        reader = AsyncMock(side_effect=markers)
    else:
        reader = AsyncMock(return_value=markers or {})
    if marker_output is None:
        monkeypatch.setattr(verifier, "_plugin_guard_markers", reader)
    else:
        monkeypatch.setattr(
            "benchflow.task.verifier_core.receipt_exec",
            AsyncMock(return_value=ExecResult(stdout=marker_output, return_code=0)),
        )
    planes = SimpleNamespace(
        harden_before_verify=AsyncMock(), verifier=lambda **kwargs: verifier
    )
    with recording_trial_warnings() as warnings:
        rewards, error, diagnostic = await _setup._verify_rollout(
            sandbox, task, paths, {}, planes
        )
        if later_hardening_failure is not None:
            planes.harden_before_verify.side_effect = later_hardening_failure
            rewards, error, diagnostic = await _setup._verify_rollout(
                sandbox, task, paths, {}, planes
            )
    return rewards, error, warnings, diagnostic


@pytest.mark.parametrize(
    "files,reason",
    [
        ({}, "reward.txt"),
        ({"reward.txt": ""}, "reward.txt"),
        ({"reward.txt": "1 passed\n1\n"}, "reward.txt"),
        ({"reward.txt": "NaN"}, "reward.txt"),
        ({"reward.txt": "2"}, "reward.txt"),
        ({"reward.txt": "-1"}, "reward.txt"),
        ({"reward.txt": "<directory>"}, "reward.txt"),
        ({"reward.txt": "<fifo>"}, "reward.txt"),
        ({"reward.txt": "<symlink>"}, "reward.txt"),
        ({"reward.txt": "<oversized>"}, "reward.txt"),
        ({"reward.txt": b"\xff"}, "reward.txt"),
        ({"reward.json": "[" * 2000}, "reward.json"),
        ({"reward.txt": "0", "reward.json": '{"reward": "x"}'}, "reward.json"),
        ({"reward.txt": "1", "reward.json": "{"}, "reward.json"),
        ({"reward.txt": "0", "reward.json": '{"reward": 1}'}, "reward.json"),
        ({"reward.txt": "broken", "reward.json": '{"reward": 1}'}, "reward.txt"),
        ({"reward.json": '{"reward": 2}'}, "reward.json"),
        ({"reward.json": "[]"}, "reward.json"),
    ],
)
async def test_bad_reward_evidence_scores_zero(tmp_path, monkeypatch, files, reason):
    rewards, error, warnings, _ = await run(tmp_path, monkeypatch, files)
    assert error is None
    assert rewards == {"reward": 0.0}
    assert any(reason in warning and "scored 0" in warning for warning in warnings)


async def test_a_marker_that_fails_validation_cannot_leave_a_pass(
    tmp_path, monkeypatch
):
    output = f"{GUARD}.1-deadbeef.crashed\t" + "0" * 64 + '\t"changed"\n!end\n'
    rewards, error, warnings, _ = await run(
        tmp_path, monkeypatch, {"reward.txt": "1"}, marker_output=output
    )
    assert rewards == {"reward": 0.0} and error is None
    assert any("guard not confirmed" in warning for warning in warnings)


@pytest.mark.parametrize("registered", [False, True])
async def test_printed_guard_error_cannot_drop_a_trial_or_be_cancelled_by_a_marker(
    tmp_path, monkeypatch, registered
):
    markers = (
        {"registered": [GuardMarker(f"{GUARD}.1.registered", "registered", "")]}
        if registered
        else {}
    )
    rewards, error, warnings, _ = await run(
        tmp_path,
        monkeypatch,
        {"reward.txt": "1", "test-stdout.txt": f'Error importing plugin "{GUARD}"'},
        markers=markers,
    )
    assert error is None and rewards == {"reward": 0.0}
    assert any("guard not confirmed" in warning for warning in warnings)


@pytest.mark.parametrize(
    "kind",
    [
        "loading",
        "checking",
        "crashed",
        "installed",
        "unproven",
        "unlisted",
        "unrecorded",
        "refused",
    ],
)
async def test_every_adverse_guard_marker_scores_zero(tmp_path, monkeypatch, kind):
    rewards, error, warnings, _ = await run(
        tmp_path,
        monkeypatch,
        {"reward.txt": "1"},
        markers={kind: [GuardMarker(f"{GUARD}.1.{kind}", kind, "forged")]},
    )
    assert error is None and rewards == {"reward": 0.0}
    assert any("guard not confirmed" in warning for warning in warnings)


async def test_unreadable_guard_markers_score_zero(tmp_path, monkeypatch):
    rewards, error, warnings, _ = await run(
        tmp_path,
        monkeypatch,
        {"reward.txt": "1"},
        markers=PluginGuardMarkersUnreadableError("marker limit reached"),
    )
    assert error is None and rewards == {"reward": 0.0}
    assert warnings


async def test_a_test_timeout_scores_zero_even_with_a_passing_reward(
    tmp_path, monkeypatch
):
    rewards, error, warnings, diagnostic = await run(
        tmp_path, monkeypatch, {"reward.txt": "1"}, hang=True
    )
    assert error is None and rewards == {"reward": 0.0}
    assert any("timed out" in warning for warning in warnings)
    assert diagnostic is not None


@pytest.mark.parametrize("rc", [0, 1, 4, 127, 137])
async def test_exit_status_never_discards_a_valid_reward(tmp_path, monkeypatch, rc):
    rewards, error, _, _ = await run(tmp_path, monkeypatch, {"reward.txt": "0"}, rc=rc)
    assert error is None and rewards == {"reward": 0.0}


async def test_matching_valid_json_keeps_its_extra_metrics(tmp_path, monkeypatch):
    rewards, error, _, _ = await run(
        tmp_path,
        monkeypatch,
        {"reward.txt": "0.5", "reward.json": '{"reward": 0.5, "tests": 1}'},
    )
    assert error is None and rewards == {"reward": 0.5, "tests": 1}


async def test_deleted_start_receipts_cannot_drop_a_trial(tmp_path, monkeypatch):
    from tests.test_verifier_start_receipts import ReceiptVerifier, _env, _planes, _task

    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.01)
    monkeypatch.setattr(_setup, "_VERIFIER_RECEIPT_POLL_SEC", 0.005)
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    with recording_trial_warnings() as warnings:
        rewards, error, _ = await _setup._verify_rollout(
            _env(set()), _task(1), paths, {}, _planes(ReceiptVerifier(["hang"]))
        )
    assert error is None and rewards == {"reward": 0.0}
    assert any("start receipt" in warning for warning in warnings)


@pytest.mark.parametrize("failure", [RuntimeError("sandbox lost"), TimeoutError()])
async def test_benchflows_hardening_failure_remains_unscored(tmp_path, failure):
    """Control: failure in a host-selected step before any verifier code runs."""
    from tests.test_verifier_start_receipts import _task

    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    planes = SimpleNamespace(
        harden_before_verify=AsyncMock(side_effect=failure), verifier=Mock()
    )
    rewards, error, _ = await _setup._verify_rollout(
        SimpleNamespace(), _task(60), paths, {}, planes
    )
    assert rewards is None and error
    planes.verifier.assert_not_called()


async def test_a_backend_exec_failure_remains_unscored(tmp_path, monkeypatch):
    """Control: a provider exception comes from the API, not script output/status."""
    rewards, error, warnings, _ = await run(
        tmp_path, monkeypatch, {}, exec_error=RuntimeError("sandbox no longer exists")
    )
    assert rewards is None and "sandbox no longer exists" in error
    assert not warnings


async def test_a_backend_script_deadline_scores_zero(tmp_path, monkeypatch):
    from benchflow.sandbox.protocol import ExecTimeoutError

    rewards, error, warnings, _ = await run(
        tmp_path, monkeypatch, {}, exec_error=ExecTimeoutError(60)
    )
    assert rewards == {"reward": 0.0} and error is None
    assert any("timed out" in warning for warning in warnings)


async def test_a_fifo_in_separate_verifier_outputs_cannot_discard_its_score(tmp_path):
    from benchflow.rollout._separate_verifier import run_separate_verifier
    from tests.test_separate_verifier import (
        FakeVerifierSandbox,
        _agent_trial,
        _fake_rollout,
        _task_dir,
    )

    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box")

    async def verify(env, task, paths, *args, **kwargs):
        (paths.verifier_dir / "reward.txt").write_text("1\n")
        os.mkfifo(paths.rollout_dir / "verifier" / "planted-fifo")
        return {"reward": 1.0}, None

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *args: box, verify=verify
    )
    assert rewards == {"reward": 0.0} and error is None
    assert any("publication failed" in w for w in rollout._warnings)
    assert box.stopped == [True]


async def test_separate_verifier_preserves_its_named_zero_warning(
    tmp_path, monkeypatch
):
    from benchflow._utils.trial_warnings import report_trial_warning
    from benchflow.rollout._separate_verifier import _default_verify

    async def score(*args, **kwargs):
        report_trial_warning("Verifier scored 0: verifier/reward.json is malformed")
        return {"reward": 0.0}, None, None

    monkeypatch.setattr(_setup, "_publish_trajectory_for_verifier", AsyncMock())
    monkeypatch.setattr(_setup, "_verify_rollout", score)
    rollout = SimpleNamespace(
        _trajectory=[],
        _planes=SimpleNamespace(),
        _config=SimpleNamespace(primary_agent="oracle"),
    )
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    assert await _default_verify(rollout)(
        SimpleNamespace(), SimpleNamespace(), paths, {}, workspace="/app"
    ) == ({"reward": 0.0}, None)
    assert rollout._warnings == ["Verifier scored 0: verifier/reward.json is malformed"]


async def test_a_post_test_input_error_cannot_drop_the_trial(tmp_path, monkeypatch):
    from benchflow.taskmd.verify import TaskMdVerifierError

    rewards, error, warnings, _ = await run(
        tmp_path,
        monkeypatch,
        {"reward.txt": "1"},
        markers=TaskMdVerifierError("verifier/judge was replaced by the verifier user"),
    )
    assert rewards == {"reward": 0.0} and error is None
    assert any("verifier/judge" in warning for warning in warnings)


async def test_a_path_refusal_marker_cannot_drop_the_trial(tmp_path, monkeypatch):
    from benchflow._utils.trusted_path import TrustedPathReport, report_trusted_path
    from tests.test_verifier_start_receipts import _task

    report = TrustedPathReport(
        path="/usr/bin:/bin",
        dropped=(("/app/bin", "agent writable"),),
        refusals="/refused",
        stubs=(("grade", "/app/bin"),),
    )

    async def harden(*args, **kwargs):
        report_trusted_path(report)

    monkeypatch.setattr(
        _setup, "_refused_programs_run", AsyncMock(return_value=[("grade", "/app/bin")])
    )
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = SimpleNamespace(
        execution_receipt=None,
        verify=AsyncMock(return_value=SimpleNamespace(rewards={"reward": 1.0})),
    )
    planes = SimpleNamespace(
        harden_before_verify=harden, verifier=lambda **kwargs: verifier
    )
    with recording_trial_warnings() as warnings:
        rewards, error, _ = await _setup._verify_rollout(
            SimpleNamespace(), _task(60), paths, {}, planes
        )
    assert rewards == {"reward": 0.0} and error is None
    assert any("PATH refusal" in warning and "grade" in warning for warning in warnings)


@pytest.mark.parametrize(
    "failure", [RuntimeError("protected kit changed"), TimeoutError()]
)
async def test_prior_verifier_code_cannot_make_later_hardening_discard_the_trial(
    tmp_path, monkeypatch, failure
):
    """FIX-49 r4: soft verification runs code as the verifier user too.

    That user can change root-owned runtime files before a later hardening.
    The host remembers that code ran; writable receipts cannot erase it.
    """
    rewards, error, warnings, _ = await run(
        tmp_path, monkeypatch, {"reward.txt": "1"}, later_hardening_failure=failure
    )
    assert rewards == {"reward": 0.0} and error is None
    assert any("scored 0" in w for w in warnings)


@pytest.mark.parametrize("evidence", ["fifo-output", "linked-folder"])
async def test_judge_only_writable_evidence_cannot_drop_the_trial(tmp_path, evidence):
    from benchflow.taskmd.verify import _host_write, _real_dirs
    from tests.test_verifier_start_receipts import _task

    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    target = paths.rollout_dir / "verifier/judge-output"
    if evidence == "fifo-output":
        os.mkfifo(target)
    else:
        target.symlink_to(tmp_path, target_is_directory=True)

    async def verify():
        if evidence == "fifo-output":
            _host_write(target, "verdict")
        else:
            _real_dirs(target)

    verifier = SimpleNamespace(test_command_issued=False, verify=verify)
    planes = SimpleNamespace(
        harden_before_verify=AsyncMock(), verifier=lambda **_: verifier
    )
    with recording_trial_warnings() as warnings:
        rewards, error, _ = await _setup._verify_rollout(
            SimpleNamespace(), _task(60), paths, {}, planes
        )
    assert rewards == {"reward": 0.0} and error is None
    assert any("judge-output" in warning for warning in warnings)


async def test_judge_shell_remembers_that_submitted_code_can_run(monkeypatch):
    from benchflow.taskmd import verify as taskmd

    sandbox = SimpleNamespace()
    runner = taskmd.SandboxRunner(sandbox=sandbox, kept=[], views={})
    runner.ready = True
    runner.drop = ("setpriv", "/usr/bin/setpriv", None)
    monkeypatch.setattr(runner, "command_script", lambda *args: "run submitted code")
    monkeypatch.setattr(
        taskmd,
        "root_exec",
        AsyncMock(return_value=ExecResult(stdout="0 0\n\n", return_code=0)),
    )
    await runner("python /app/submission.py", 10)
    assert getattr(sandbox, "_benchflow_test_command_issued", False) is True


async def test_prior_verifier_cannot_drop_the_trial_by_blocking_publication(
    tmp_path, monkeypatch
):
    from benchflow.rollout._setup import VerifierPublicationError
    from tests.test_verifier_wedge_retry import _rollout

    rollout = _rollout(tmp_path, AsyncMock(), recovery=False)
    rollout._env._benchflow_test_command_issued = True
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier",
        AsyncMock(
            side_effect=VerifierPublicationError("agent/trajectory.json blocked")
        ),
    )
    assert await rollout.verify() == {"reward": 0.0}
    assert rollout._verifier_error is None
    assert any("agent/trajectory.json" in warning for warning in rollout._warnings)


@pytest.mark.parametrize(
    "step", ["capture_terminal_workspace", "_probe_integrity_access"]
)
async def test_prior_verifier_cannot_drop_the_trial_before_final_hardening(
    tmp_path, monkeypatch, step
):
    """FIX-49 r4: earlier verifier code can change the kit used by later probes."""
    from tests.test_verifier_wedge_retry import _rollout

    rollout = _rollout(tmp_path, AsyncMock(), recovery=False)
    rollout._env._benchflow_test_command_issued = True
    monkeypatch.setattr(
        f"benchflow.rollout.{step}",
        AsyncMock(side_effect=RuntimeError("protected kit changed")),
    )
    assert await rollout.verify() == {"reward": 0.0}
    assert rollout._verifier_error is None
    assert any("protected kit changed" in warning for warning in rollout._warnings)


@pytest.mark.parametrize("kind", ["deep-json", "oversized-json"])
def test_taskmd_timeout_annotation_cannot_reopen_a_bad_reward_failure(
    tmp_path, monkeypatch, caplog, kind
):
    """FIX-49 r4: the final task.md annotation reads verifier-owned bytes too."""
    from benchflow.rollout import Rollout

    contents = "[" * 20_000 if kind == "deep-json" else " " * (16 * 1024 * 1024) + "{}"
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    paths.reward_json_path.write_text(contents)
    rollout = SimpleNamespace(
        _rollout_paths=paths,
        _rewards={"reward": 0.0},
        _config=SimpleNamespace(task_path=tmp_path / "task"),
    )
    monkeypatch.setattr("benchflow.taskmd.materialize.taskmd_metadata", lambda _: {})
    Rollout._record_taskmd_timed_out(rollout)
    assert rollout._rewards == {"reward": 0.0}
    assert paths.reward_json_path.read_text() == contents
    assert "Cannot record timed_out in unreadable" in caplog.text
