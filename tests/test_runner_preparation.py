"""FIX-49 r4 M1: c278f22a6 skipped the named runner without pins or --ctrf."""

import json
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.sandbox import _verifier_runtime as runtime
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.task import Task


@pytest.fixture(params=["comment", "env"])
def named_task(tmp_path, request):
    (tmp_path / "tests").mkdir()
    (tmp_path / "instruction.md").write_text("Do the task.\n")
    toml = 'version = "1.0"\n'
    test = "#!/bin/sh\n/opt/grade\n"
    if request.param == "comment":
        test += "# uses $BENCHFLOW_PYTEST\n"
    else:
        toml += '[verifier.env]\nRUNNER_NAME = "BENCHFLOW_PYTEST"\n'
    (tmp_path / "task.toml").write_text(toml)
    (tmp_path / "tests/test.sh").write_text(test)
    return Task(tmp_path)


@pytest.mark.verifier_runtime
async def test_named_runner_is_prepared_before_the_agent(named_task, monkeypatch):
    expected = {"runner": "/opt/runtime/verifier/bin/pytest"}
    prepare = AsyncMock(return_value=expected)
    monkeypatch.setattr(runtime, "prepare_verifier_runtime", prepare)
    sandbox = SimpleNamespace()
    assert (
        await lockdown.prepare_verifier_runtime(sandbox, named_task, "agent")
        == expected
    )
    prepare.assert_awaited_once()


@pytest.mark.verifier_runtime
async def test_named_runner_manifest_is_checked_after_the_agent(
    named_task, monkeypatch
):
    expected = {"runner": "/opt/runtime/verifier/bin/pytest"}
    read = AsyncMock(
        return_value=ExecResult(stdout=json.dumps(expected), return_code=0)
    )
    monkeypatch.setattr(lockdown, "_read_exec", read)
    assert (
        await lockdown._verifier_runtime_manifest(
            SimpleNamespace(), named_task, "agent", build=False
        )
        == expected
    )
    read.assert_awaited_once()


async def test_named_runner_reaches_the_runtime_builder(named_task, monkeypatch):
    expected = {
        "runner": "/opt/runtime/verifier/bin/pytest",
        "image_sites": [],
        "plugins": {},
    }
    monkeypatch.setattr(runtime, "probe_interpreter", AsyncMock(return_value=None))
    monkeypatch.setattr(
        runtime, "resolve", Mock(return_value=SimpleNamespace(runner=True, wheels=[]))
    )
    build = Mock(return_value=expected)
    monkeypatch.setattr(runtime, "build_tree", build)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(stdout="", return_code=0)),
        upload_dir=AsyncMock(),
    )
    assert (
        await runtime.prepare_verifier_runtime(sandbox, named_task, tests_dir="/tests")
        == expected
    )
    assert build.call_args.kwargs["runner_wanted"] is True


async def test_named_runner_never_silently_becomes_unset(named_task, monkeypatch):
    monkeypatch.setattr(runtime, "probe_interpreter", AsyncMock(return_value=None))
    monkeypatch.setattr(
        runtime, "resolve", Mock(return_value=SimpleNamespace(runner=False))
    )
    with pytest.raises(runtime.RuntimeUnavailable, match=r"names.*BENCHFLOW_PYTEST"):
        await runtime.prepare_verifier_runtime(
            SimpleNamespace(), named_task, tests_dir="/tests"
        )


def test_the_runner_does_not_write_its_report_into_the_scalar_reward(
    tmp_path, monkeypatch
):
    """FIX-49 r4 advice.sh: an enclosing if block redirects stdout to reward.txt."""
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "1 passed\\n"\n')
    python.chmod(0o755)
    monkeypatch.setattr(runtime, "KIT_PYTHON", str(python))
    monkeypatch.setattr(runtime, "custody_command", lambda: "true")
    reward = tmp_path / "reward.txt"
    launcher = tmp_path / "runner"
    launcher.write_text(
        runtime.runner_script().replace(
            "/logs/verifier/reward.txt", shlex.quote(str(reward))
        )
    )
    result = subprocess.run(
        [
            "sh",
            "-c",
            f"if sh {shlex.quote(str(launcher))}; then echo 1; else echo 0; fi > {shlex.quote(str(reward))}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert reward.read_text() == "1\n"
    assert result.stderr == "1 passed\n"
