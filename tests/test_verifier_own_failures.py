"""Guards FIX-49 r5 against own failures scored zero at 4f1f5e585.

Only the operation failing is injected; the real final scoring boundary runs.
No provider request, real outage or exhausted controller disk is required.
"""

import errno
import json
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.rollout import _setup
from benchflow.sandbox._base import ExecResult
from benchflow.task import Task
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier
from benchflow.taskmd import judging
from benchflow.taskmd import verify as taskmd


def _fault(kind):
    if kind == "transport":
        return httpx.ConnectError("fixture controller transport disconnected")
    if kind == "api":
        return RuntimeError("fixture sandbox no longer exists")
    return OSError(getattr(errno, kind), "fixture private controller write failed")


@pytest.mark.parametrize(
    "backend_error",
    [
        "docker-transport",
        "docker-api",
        "daytona-api",
        "daytona-missing",
        "daytona-timeout",
    ],
)
async def test_later_hardening_preserves_typed_backend_errors(
    tmp_path, monkeypatch, backend_error
):
    """Guards FIX-49 r5: backend API failures are own errors after soft verify."""
    from benchflow.sandbox._engine_api import EngineError, EngineUnavailable
    from tests.test_verifier_scored_evidence import run

    if backend_error == "docker-transport":
        fault = EngineUnavailable("fixture Docker daemon unavailable")
    elif backend_error == "docker-api":
        fault = EngineError(
            "POST", "/containers/fixture/exec", 404, "fixture API refused"
        )
    else:
        sdk = pytest.importorskip("daytona")
        kind = {
            "daytona-api": sdk.DaytonaError,
            "daytona-missing": sdk.DaytonaNotFoundError,
            "daytona-timeout": sdk.DaytonaTimeoutError,
        }[backend_error]
        fault = kind("fixture Daytona API unavailable")
    rewards, error, warnings, _ = await run(
        tmp_path,
        monkeypatch,
        {"reward.txt": "1"},
        later_hardening_failure=fault,
    )
    assert rewards is None and error is not None
    assert type(fault).__name__ in error and "fixture" in error
    assert not warnings


async def _assess(sandbox, verifier):
    planes = SimpleNamespace(
        harden_before_verify=AsyncMock(), verifier=lambda **_: verifier
    )
    with recording_trial_warnings() as warnings:
        rewards, error, _ = await _setup._verify_rollout(
            sandbox, verifier._task, verifier._rollout_paths, {}, planes
        )
    return rewards, error, warnings


def _task(tmp_path, kind="test-script"):
    root = tmp_path / "task"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "instruction.md").write_text("Grade the saved answer.\n")
    (root / "task.toml").write_text(
        f'version = "1.0"\n[verifier]\ntype = "{kind}"\ntimeout_sec = 5\n'
        '[verifier.judge]\nrubric_path = "tests/rubric.json"\ninput_dir = "/app/output"\n'
    )
    (root / "tests/rubric.json").write_text(
        json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
    )
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    sandbox = SimpleNamespace(
        is_mounted=True,
        supports_system_shell=True,
        exec=AsyncMock(return_value=ExecResult(stdout="", return_code=0)),
        upload_dir=AsyncMock(),
    )
    return sandbox, Verifier(Task(root), paths, sandbox)


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["transport", "api", "capability"])
@pytest.mark.parametrize("judge", ["llm", "agent"])
async def test_native_capture_api_failures_are_own_steps(
    tmp_path, monkeypatch, prior, kind, judge
):
    sandbox, verifier = _task(tmp_path, "llm-judge")
    sandbox._benchflow_test_command_issued = prior
    if kind == "capability":
        sandbox.supports_system_shell = False
    else:
        sandbox.exec.side_effect = _fault(kind)
    provider = AsyncMock()
    monkeypatch.setattr("benchflow.rewards.llm.call_judge", provider)
    if judge == "agent":

        async def capture():
            await verifier._resolve_agent_judge_input(
                "/app/answer.txt", inputs_dir=tmp_path / "private", index=0
            )

        monkeypatch.setattr(verifier, "verify", capture)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert rewards is None and error is not None
    assert "DownloadVerifierDirError" in error
    assert ("protected root shell" if kind == "capability" else "fixture") in error
    assert not warnings
    provider.assert_not_awaited()
    if kind != "capability":
        sandbox.exec.assert_awaited_once()


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "outcome", ["environment", "provider", "malformed", "pass", "fail"]
)
async def test_native_judge_keeps_public_errors_and_scoring_contract(
    tmp_path, monkeypatch, prior, outcome
):
    from benchflow.rewards.llm import JudgeEnvironmentError
    from tests.test_llm_judge_verifier import _make_sandbox

    _, verifier = _task(tmp_path, "llm-judge")
    verifier._task.config.verifier.judge.input_dir = "/app"
    sandbox = _make_sandbox({"answer.txt": "saved answer"})
    sandbox._benchflow_test_command_issued = prior
    verifier._sandbox = sandbox
    provider = AsyncMock()
    if outcome == "environment":
        provider.side_effect = JudgeEnvironmentError("fixture SDK unavailable")
    elif outcome == "provider":
        provider.side_effect = RuntimeError("fixture provider failed; PRIVATE_TOKEN")
    elif outcome == "malformed":
        provider.return_value = "invalid provider response; PRIVATE_TOKEN"
    else:
        provider.return_value = json.dumps({"verdict": outcome, "reasoning": "fixture"})
    monkeypatch.setattr("benchflow.rewards.llm.call_judge", provider)
    rewards, error, warnings = await _assess(sandbox, verifier)
    provider.assert_awaited_once()
    sandbox.download_file.assert_awaited_once()
    assert not warnings
    if outcome in ("pass", "fail"):
        assert error is None
        assert rewards == {"reward": 1.0 if outcome == "pass" else 0.0}
    else:
        assert rewards is None and error is not None
        assert (
            "JudgeEnvironmentError" if outcome == "environment" else "JudgeScoringError"
        ) in error
        assert "PRIVATE_TOKEN" not in error
        assert not verifier._rollout_paths.reward_json_path.exists()


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("capability", ["host-descriptors", "judge-shell"])
async def test_controller_capability_failures_remain_own_steps(
    tmp_path, monkeypatch, prior, capability
):
    sandbox, verifier = _task(tmp_path, "llm-judge")
    sandbox._benchflow_test_command_issued = prior
    if capability == "host-descriptors":

        def unavailable():
            raise RuntimeError("fixture host descriptors unavailable")

        monkeypatch.setattr(
            "benchflow._utils.reachable_write.require_descriptor_io", unavailable
        )
        monkeypatch.setattr(verifier, "verify", verifier._verify_llm_judge)
    else:
        sandbox.supports_system_shell = False
        runner = taskmd.SandboxRunner(sandbox=sandbox, kept=[], views={})
        monkeypatch.setattr(verifier, "verify", lambda: runner("true", 5))
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert rewards is None and error is not None and not warnings
    assert (
        "capability unavailable" in error
        if capability == "host-descriptors"
        else "cannot run in this sandbox" in error
    )
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["transport", "api", "deadline"])
async def test_taskmd_tool_execution_has_the_script_error_contract(
    tmp_path, monkeypatch, prior, kind
):
    sandbox, verifier = _task(tmp_path)
    sandbox._benchflow_test_command_issued = prior
    runner = taskmd.SandboxRunner(sandbox=sandbox, kept=[], views={})
    runner.ready = True
    runner.drop = ("setpriv", "/usr/bin/setpriv", None)
    monkeypatch.setattr(runner, "command_script", lambda *_: "fixture command")
    from benchflow.sandbox.protocol import ExecTimeoutError

    execute = AsyncMock(
        side_effect=ExecTimeoutError(5) if kind == "deadline" else _fault(kind)
    )
    monkeypatch.setattr(taskmd, "root_exec", execute)
    monkeypatch.setattr(verifier, "verify", lambda: runner("true", 5))
    rewards, error, warnings = await _assess(sandbox, verifier)
    execute.assert_awaited_once()
    assert sandbox._benchflow_test_command_issued is True
    if kind == "deadline":
        assert rewards == {"reward": 0.0} and error is None
        assert any("timed out" in warning for warning in warnings)
    else:
        assert rewards is None and "VerifierExecutionError" in error
        assert "fixture" in error and not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["EIO", "ENOSPC", "EACCES"])
async def test_private_judge_writes_remain_own_failures(
    tmp_path, monkeypatch, prior, kind
):
    sandbox, verifier = _task(tmp_path)
    sandbox._benchflow_test_command_issued = prior
    fault = _fault(kind)
    destination = SimpleNamespace(
        write=lambda *_args, **_kwargs: (_ for _ in ()).throw(fault)
    )
    monkeypatch.setattr(taskmd, "destination", lambda _: destination)

    async def write():
        taskmd._write_json(tmp_path / "private-session.json", {"end": "accepted"})

    monkeypatch.setattr(verifier, "verify", write)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert rewards is None and "private-session.json" in error
    assert "fixture private controller write failed" in error
    assert "link or a special file" not in error and not warnings


@pytest.mark.parametrize(
    "step",
    [
        "capture_terminal_workspace",
        "_probe_integrity_access",
        "_publish_trajectory_for_verifier",
    ],
)
@pytest.mark.parametrize("kind", ["transport", "ENOSPC"])
async def test_later_preparation_and_publication_preserve_own_errors(
    tmp_path, monkeypatch, step, kind
):
    from tests.test_verifier_wedge_retry import _rollout

    rollout = _rollout(tmp_path, AsyncMock(), recovery=False)
    rollout._env._benchflow_test_command_issued = True
    operation = AsyncMock(side_effect=_fault(kind))
    monkeypatch.setattr(f"benchflow.rollout.{step}", operation)
    assert await rollout.verify() is None
    assert "fixture" in rollout._verifier_error
    assert not any("scored 0" in w for w in getattr(rollout, "_warnings", []))
    operation.assert_awaited_once()


@pytest.mark.parametrize("step", ["exec", "upload_file"])
@pytest.mark.parametrize("kind", ["api", "deadline"])
async def test_publication_keeps_the_actual_api_cause(
    tmp_path, monkeypatch, step, kind
):
    from tests.test_verifier_wedge_retry import _rollout

    rollout = _rollout(tmp_path, AsyncMock(), recovery=False)
    rollout._env._benchflow_test_command_issued = True
    rollout._env.exec = AsyncMock(return_value=ExecResult(return_code=0))
    rollout._env.upload_file = AsyncMock()
    from benchflow.sandbox.protocol import ExecTimeoutError

    getattr(rollout._env, step).side_effect = (
        ExecTimeoutError(5) if kind == "deadline" else _fault("api")
    )
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier",
        _setup._publish_trajectory_for_verifier,
    )
    rewards = await rollout.verify()
    if kind == "deadline":
        assert rewards == {"reward": 0.0} and rollout._verifier_error is None
        assert any("publication" in warning for warning in rollout._warnings)
    else:
        assert rewards is None
        assert "fixture sandbox no longer exists" in rollout._verifier_error
        assert "publication" in rollout._verifier_error
    assert getattr(rollout._env, step).await_count >= 1


@pytest.mark.parametrize("kind", ["transport", "ENOSPC"])
async def test_separate_capture_preserves_own_failures(tmp_path, monkeypatch, kind):
    from benchflow.rollout import _separate_verifier as separate
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
    capture = AsyncMock(side_effect=_fault(kind))
    monkeypatch.setattr(separate, "_publish_off_loop", capture)
    rewards, error = await separate.run_separate_verifier(
        rollout,
        create_environment=lambda *_: box,
        verify=AsyncMock(return_value=({"reward": 1.0}, None)),
    )
    assert rewards is None and "fixture" in error
    assert "publication" in error and box.stopped == [True]
    capture.assert_awaited_once()


@pytest.mark.parametrize("script", [False, True])
@pytest.mark.parametrize("outcome", ["credentials", "provider", "pass", "fail"])
async def test_taskmd_rubric_own_errors_do_not_depend_on_prior_script(
    tmp_path, monkeypatch, script, outcome
):
    from benchflow.task.formats import materialize_task_dir
    from tests.test_taskmd_agent_judge import FIXTURE
    from tests.test_taskmd_verify import _Sandbox

    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "format-cache"))
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    if script:
        (package / "verifier/test.sh").write_text("#!/bin/sh\nexit 0\n")
    native = materialize_task_dir(package)
    root = tmp_path / "sandbox"
    (root / "app").mkdir(parents=True)
    (root / "app/main.py").write_text("print(42)\n")
    sandbox = _Sandbox(root)
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = Verifier(Task(native), paths, sandbox)
    credentials = judging.Credentials("api-key", "fixture-placeholder")
    monkeypatch.setattr(
        judging,
        "resolve_credentials",
        lambda: (
            (None, "fixture credentials unavailable")
            if outcome == "credentials"
            else (credentials, None)
        ),
    )
    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())

    async def call(*_args, **_kwargs):
        if outcome == "provider":
            raise judging.JudgeError("fixture provider unavailable")
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": "submit",
                    "name": "submit_review",
                    "input": {
                        "verdicts": [
                            {
                                "id": "prints-42",
                                "verdict": outcome,
                                "citations": [
                                    {
                                        "source": "file",
                                        "path": "/app/main.py",
                                        "quote": "print(42)",
                                        "lines": [1, 1],
                                    }
                                ],
                                "rationale": "fixture verdict",
                            }
                        ],
                        "tags": [],
                    },
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    provider = AsyncMock(side_effect=call)
    monkeypatch.setattr(judging, "MessagesClient", lambda _: provider)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert getattr(sandbox, "_benchflow_test_command_issued", False) is script
    if outcome in ("pass", "fail"):
        assert error is None
        assert rewards["reward"] == float(outcome == "pass")
    else:
        assert rewards is None and error is not None
        assert f"fixture {outcome}" in error
        assert not warnings
    if outcome == "credentials":
        provider.assert_not_awaited()
    else:
        assert provider.await_count > 0
