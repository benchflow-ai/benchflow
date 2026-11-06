"""Guards FF-sw-docker-C13 at bfb6377c and integ 668bbccb (raw exec readers)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult, with_system_path
from benchflow.sandbox.lockdown import _keep_route_path
from tests._receipt_fakes import inner_command, receipt_answer
from tests.test_compose_raw_output import emit_process as emit_raw_process
from tests.test_compose_raw_output import transport


def emit_process(monkeypatch, output):
    """Retain C13 assertions through integration a4367969's receipt framing."""

    def answer(args):
        stdout, stderr, code = output(
            (
                *args[:-1],
                inner_command(args[-1])
                .removeprefix(_keep_route_path(""))
                .removeprefix(with_system_path("")),
            )
        )
        result = receipt_answer(
            args[-1], ExecResult(stdout=stdout, stderr=stderr, return_code=code)
        )
        return result.stdout, result.stderr, result.return_code

    emit_raw_process(monkeypatch, answer)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("route", ["container", "compose"])
@pytest.mark.parametrize("filename", lockdown._BUILD_CONFIG_FILES)
async def test_restore_preserves_every_trusted_manifest_key(
    tmp_path, monkeypatch, backend, route, filename
):
    """C13: an ordinary SSH login must not turn restoration into deletion.

    Commands are recorded, never executed: deletion is proved by the selected
    target/operation, with only dummy fixture paths in the plan.
    """
    env = transport(tmp_path, backend, filename)
    env._main_container = "dummy-container" if route == "container" else None
    manifest = {name: name == filename for name in lockdown._BUILD_CONFIG_FILES}
    commands = []

    def output(args):
        command = args[-1]
        commands.append(command)
        return (
            json.dumps(manifest) if command.startswith("cat ") else "",
            "",
            0,
        )

    emit_process(monkeypatch, output)
    await lockdown._restore_build_config(env, "/fixture/project")
    selected = [
        cmd for cmd in commands if cmd.startswith(f"rm -f /fixture/project/{filename}")
    ]
    assert len(selected) == 1
    assert "cp --preserve=timestamps" in selected[0], selected[0]
    assert f"{lockdown._SNAPSHOT_DIR}/{filename}" in selected[0]
    for name in set(lockdown._BUILD_CONFIG_FILES) - {filename}:
        assert f"rm -f /fixture/project/{name}" in commands


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("route", ["container", "compose"])
@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("return_code", [0, 1, 125])
async def test_batched_verifier_reads_keep_data_and_stderr_separate(
    tmp_path, monkeypatch, backend, route, quote, return_code
):
    """C13: merged receipts retain every field even on a partial failure."""
    env = transport(tmp_path, backend, "setup.py")
    env._main_container = "dummy-container" if route == "container" else None
    tag = "@@benchflow-merged-fixture"
    monkeypatch.setattr(lockdown, "new_tag", lambda: tag)
    # One value per batched read; PT's batch has no PYTHONPATH read (root's
    # plumbing never has it; it is read on the task route instead).
    values = {
        "path": f"/fixture/{quote}setup.py{quote}/bin",
        "pythonpath": f"/fixture/{quote}setup.py{quote}",
        "os_release": 'ID="setup.py"',
        "python3": "",
    }
    outputs = [values[key] for key, _ in lockdown._VERIFIER_READS]
    stdout = "".join(
        f"{tag}:{i}:begin\n{value}\n{tag}:{i}:end:0\n"
        for i, value in enumerate(outputs)
    )
    # A stderr record that looks like a receipt must not enter the stdout parser.
    stderr = f"{tag}:0:begin\nnoise\n{tag}:0:end:0\n"
    emit_process(monkeypatch, lambda args: (stdout, stderr, return_code))
    answers = await lockdown._probe_verifier_reads(env)
    assert {key: answer.output for key, answer in answers.items()} == dict(
        zip((key for key, _ in lockdown._VERIFIER_READS), outputs, strict=True)
    )


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("route", ["container", "compose"])
@pytest.mark.parametrize("batched", [True, False])
async def test_pythonpath_batched_and_fallback_preserve_json_paths(
    tmp_path, monkeypatch, backend, route, batched
):
    """C13: the fallback and the trusted-path JSON reader share raw exec."""
    env = transport(tmp_path, backend, "setup.py")
    env._main_container = "dummy-container" if route == "container" else None
    path = "/fixture/'setup.py'/lib"
    commands = []

    def output(args):
        commands.append(args[-1])
        return (
            path
            if args[-1] == lockdown._PRINTENV_PYTHONPATH_CMD
            else "K " + path + "\n",
            "",
            0,
        )

    emit_process(monkeypatch, output)
    reads = {"pythonpath": lockdown.StepAnswer(path, 0)} if batched else {}
    token = lockdown._verifier_reads.set((env, reads))
    try:
        assert await lockdown._trusted_verifier_pythonpath(env, "agent") == path
    finally:
        lockdown._verifier_reads.reset(token)
    assert any(path in command or "setup.py" in command for command in commands)
    assert (lockdown._PRINTENV_PYTHONPATH_CMD in commands) is not batched


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("route", ["container", "compose", "sidecar"])
@pytest.mark.parametrize("code", [0, 1, 125])
@pytest.mark.parametrize(
    "command", ["cat /fixture/manifest", "sh -c 'read-files'", "python3 -I -S probe.py"]
)
async def test_raw_api_is_independent_of_command_status_and_route(
    tmp_path, monkeypatch, backend, route, code, command
):
    """C13 contract: no command spelling or successful-exec exception list."""
    from benchflow.sandbox._exec_output import diagnostic_result, exec_raw

    env = transport(tmp_path, backend, "setup.py")
    env._main_container = "dummy-container" if route != "compose" else None
    out, err = '{"setup.py": true}\n', "warning from 'setup.py'\n"
    emit_process(monkeypatch, lambda args: (out, err, code))
    result = await exec_raw(
        env.exec, command, service="worker" if route == "sidecar" else "main"
    )
    assert (result.stdout, result.stderr, result.return_code) == (out, err, code)
    diagnostic = diagnostic_result(result)
    if backend == "remote-docker":
        assert "setup.py" not in (
            diagnostic.stdout.render() + diagnostic.stderr.render()
        )
    assert (result.stdout, result.stderr) == (out, err)
    assert diagnostic is not result


async def test_raw_scope_is_nested_concurrent_and_cancellation_safe():
    """C13 contract: a raw reader cannot disable another task's diagnostics."""
    from benchflow.sandbox._exec_output import exec_raw, raw_output_requested

    started, finish = asyncio.Event(), asyncio.Event()

    async def read():
        assert raw_output_requested()
        started.set()
        await finish.wait()
        assert raw_output_requested()
        return ExecResult(stdout="data", stderr="", return_code=0)

    task = asyncio.create_task(exec_raw(read))
    await started.wait()
    assert not raw_output_requested()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not raw_output_requested()

    async def inner():
        assert raw_output_requested()
        raise ValueError("fixture")

    async def outer():
        with pytest.raises(ValueError, match="fixture"):
            await exec_raw(inner)
        assert raw_output_requested()

    await exec_raw(outer)
    assert not raw_output_requested()


@pytest.mark.parametrize("transient", [False, True])
@pytest.mark.parametrize("dind", [False, True])
@pytest.mark.parametrize("code", [0, 1])
async def test_daytona_session_and_dind_return_raw_streams(transient, dind, code):
    """C13 contract covers SDK sessions and DinD, with no live Daytona call."""
    from benchflow.sandbox._exec_output import exec_raw
    from benchflow.sandbox.daytona import (
        DaytonaSandbox,
        _DaytonaDirect,
        _load_daytona_sdk,
    )
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    _load_daytona_sdk()
    out, err = '{"setup.py": true}\n', "warning 'setup.py'\n"

    class Process:
        async def create_session(self, session_id):
            pass

        async def execute_session_command(self, session_id, request, timeout=None):
            return SimpleNamespace(cmd_id="fixture-command")

        async def get_session_command(self, session_id, command_id):
            return SimpleNamespace(id="fixture-command", exit_code=code)

        async def get_session_command_logs(self, session_id, command_id):
            return SimpleNamespace(stdout=out, stderr=err)

        async def delete_session(self, session_id):
            pass

    env = DaytonaSandbox.__new__(DaytonaSandbox)
    env.default_user = None
    from benchflow.task.config import SandboxConfig

    env.environment_name = "fixture"
    env.task_env_config = SandboxConfig()
    env._persistent_env = {}
    env._sandbox = SimpleNamespace(process=Process())
    env._strategy = _DaytonaDinD(env) if dind else _DaytonaDirect(env)
    if dind:
        env._strategy._compose_cmd = lambda command: (
            "fixture compose " + " ".join(command)
        )
    run = env.exec_transient if transient else env.exec
    result = await exec_raw(run, "cat /fixture/manifest", timeout_sec=5)
    assert (result.stdout, result.stderr, result.return_code) == (out, err, code)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("route", ["container", "compose"])
@pytest.mark.parametrize("fallback", [False, True])
async def test_pristine_output_inventory_preserves_paths_and_fallback_counts(
    tmp_path, monkeypatch, backend, route, fallback
):
    """C13: separate grading inventories parse raw JSON and shell fallback lines."""
    from benchflow.rollout._separate_verifier import measure_outputs

    env = transport(tmp_path, backend, "setup.py")
    env._main_container = "dummy-container" if route == "container" else None
    paths = ["/fixture/'setup.py'", "/logs/artifacts", "/fixture/'setup.py'/answer"]
    commands = []

    def output(args):
        commands.append(args[-1])
        if fallback and len(commands) == 1:
            return "", "python3: not found\n", 127
        if fallback:
            return "1 0 2 7\n" * len(paths), "", 0
        info = {"exists": True, "link": False, "entries": 2, "bytes": 7}
        return json.dumps(dict.fromkeys(paths, info)), "", 0

    emit_process(monkeypatch, output)
    result = await measure_outputs(env, workspace=paths[0], artifacts=["answer"])
    assert set(result["paths"]) == set(paths)
    # This transport fake deliberately emits the older probe layout. The
    # new clean-control folder bit is unknown, while raw paths/counts survive.
    assert all(
        value == {"exists": True, "link": False, "entries": 2, "bytes": 7, "dir": None}
        for value in result["paths"].values()
    )
    assert len(commands) == (2 if fallback else 1)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_lockdown_error_uses_diagnostic_copy(tmp_path, monkeypatch, backend):
    """C13: changing the parser's stream must not expose remote login diagnostics."""
    env = transport(tmp_path, backend, "setup.py")
    emit_process(
        monkeypatch, lambda args: ('{"setup.py": true}', "warning 'setup.py'", 1)
    )
    with pytest.raises(RuntimeError) as error:
        await lockdown._checked_exec(env, "read-fixture", "fixture reader")
    assert ("setup.py" in str(error.value)) is (backend == "docker")
    assert "stdout:" in str(error.value) and "stderr:" in str(error.value)


async def test_replay_proxy_log_fallback_is_a_diagnostic(tmp_path, monkeypatch):
    """C13: a stderr-file fallback must redact its display copy, not the live data."""
    from unittest.mock import AsyncMock

    from benchflow.continue_run import sandbox_proxy

    env = transport(tmp_path, "remote-docker", "setup.py")
    env.download_file = AsyncMock(side_effect=FileNotFoundError("fixture"))
    monkeypatch.setattr(sandbox_proxy, "asyncio", SimpleNamespace(sleep=AsyncMock()))
    emit_process(
        monkeypatch,
        lambda args: (
            ('failure from "setup.py"', "", 0)
            if args[-1].startswith("cat ")
            else ("", "", 1)
        ),
    )
    proxy = sandbox_proxy.SandboxReplayProxy.__new__(sandbox_proxy.SandboxReplayProxy)
    proxy.sandbox = env
    proxy.port = 8955
    proxy.stderr_path = "/fixture/stderr"
    with pytest.raises(RuntimeError, match="did not become healthy") as error:
        await proxy._wait_until_ready()
    assert "setup.py" not in str(error.value)
    assert "failure from" in str(error.value)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "runner",
    ["_docker_cli", "_run_docker_compose_command", "_run_docker_container_command"],
)
@pytest.mark.parametrize("code", [0, 1, 125])
async def test_raw_api_also_reaches_backend_readers(
    tmp_path, monkeypatch, backend, runner, code
):
    """C13: shared raw selection works below exec, without command-name exemptions."""
    from benchflow.sandbox._exec_output import diagnostic_result, exec_raw

    env = transport(tmp_path, backend, "setup.py")
    out, err = '{"setup.py": true}', "warning 'setup.py'"
    emit_process(monkeypatch, lambda args: (out, err, code))
    result = await exec_raw(getattr(env, runner), ["new-data-command"], check=False)
    assert (result.stdout, result.stderr, result.return_code) == (out, err, code)
    if backend == "remote-docker":
        diagnostic = diagnostic_result(result)
        assert "setup.py" not in diagnostic.stdout.render() + diagnostic.stderr.render()
