"""Guards DT2-CONF-3/4 at 212f67e4: cleanup trust and retained services."""

import asyncio
import shlex
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox.lockdown import (
    _kill_sandbox_user_procs_keeping,
    end_kept_services,
    release_kept_services,
)
from benchflow.task import Task, keep_services
from tests.test_script_agent import FakeShell
from tests.test_script_agent_docker import (
    Container,
    RemoteContainer,
    run_scored,
    started_box,
)

PYTHON_IMAGE = (
    "python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b"
)
BACKENDS = [Container, RemoteContainer]
SYSTEM_DIRS = [
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
]


def test_cleanup_path_matches_protected_script_namespace():
    """Guards DT2-CONF-3: a cleanup PATH change must extend this preflight."""
    from benchflow.rollout._setup import _SCRIPT_DEFAULT_PATH
    from benchflow.sandbox.lockdown import _SAFE_VERIFIER_PATH

    assert _SCRIPT_DEFAULT_PATH == _SAFE_VERIFIER_PATH
    assert _SCRIPT_DEFAULT_PATH.split(":") == SYSTEM_DIRS


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", SYSTEM_DIRS)
@pytest.mark.parametrize(
    "route",
    [
        "absolute",
        "relative",
        "dot_parent",
        "ancestor",
        "writable",
        "group_writable",
        "dormant",
        "hidden",
        "dangling",
    ],
)
async def test_cleanup_namespace_is_immutable(tmp_path, backend, directory, route):
    """Guards DT2-CONF-3: even future helpers cannot enter via writable names."""
    with started_box() as box:
        box = backend(box.name)
        name = ".dt2-future-helper" if route == "hidden" else "dt2-future-helper"
        target = f"{directory}/{name}"
        setup = "mkdir -p /app/tools; cp /usr/bin/true /app/tools/tool; "
        if route in ("writable", "dormant", "group_writable"):
            setup += f"cp /usr/bin/true {target}; "
            setup += f"chown {'root:agent' if route == 'group_writable' else 'agent:agent'} {target}; "
            mode = {"dormant": "644", "group_writable": "775"}.get(route, "755")
            setup += f"chmod {mode} {target}"
        elif route == "ancestor":
            setup += "ln -s /app/tools /opt/dt2-tools; "
            setup += f"ln -s /opt/dt2-tools/tool {target}"
        elif route == "relative":
            setup += (
                f"ln -s $(realpath --relative-to={directory} /app/tools/tool) {target}"
            )
        elif route == "dot_parent":
            setup += f"ln -s /usr/bin/../../app/./tools/tool {target}"
        elif route == "dangling":
            setup += f"ln -s /app/not-created-yet {target}"
        else:
            setup += f"ln -s /app/tools/tool {target}"
        assert (await box.exec(setup)).return_code == 0
        result = await run_scored(box, tmp_path, "printf ran > /app/ran\n")
        assert "before the script started" in (result.error or "")
        assert result.rewards is None
        assert (await box.exec("test -f /app/ran")).return_code == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("directory", SYSTEM_DIRS)
@pytest.mark.parametrize("route", ["ordinary", "absolute_missing", "relative_missing"])
async def test_cleanup_protected_aliases_remain_usable(
    tmp_path, backend, directory, route
):
    """Guards DT2-CONF-3: immutable aliases, including missing targets, are safe."""
    with started_box() as box:
        box = backend(box.name)
        target = f"{directory}/dt2-future-helper"
        if route == "ordinary":
            setup = f"cp /usr/bin/true {target}"
        elif route == "absolute_missing":
            setup = f"ln -s /opt/dt2-missing/deeper/tool {target}"
        else:
            setup = f"ln -s $(realpath -m --relative-to={directory} /opt/dt2-missing/deeper/tool) {target}"
        assert (await box.exec(setup)).return_code == 0
        result = await run_scored(box, tmp_path, "printf ran > /app/ran\n")
        assert result.error is None and result.rewards == {"reward": 0.0}
        assert (await box.exec("test -f /app/ran")).return_code == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("helper", ["setpriv", "grep"])
async def test_timeout_cannot_execute_replaced_cleanup_helper(
    tmp_path, backend, helper
):
    """Guards DT2-CONF-3: only the agent replaces initially trusted bytes."""
    with started_box() as box:
        box = backend(box.name)
        setup = (
            f"cp /usr/bin/{helper} /app/{helper}; "
            f"ln -s /app/{helper} /usr/local/bin/{helper}; "
            "printf dummy-marker > /root/dt2-marker; chmod 600 /root/dt2-marker"
        )
        assert (await box.exec(setup)).return_code == 0
        replacement = (
            "#!/bin/sh\ncat /root/dt2-marker > /app/leaked\n"
            f'exec /usr/bin/{helper} "$@"\n'
        )
        body = (
            f"rm /app/{helper}; printf %s {shlex.quote(replacement)} > /app/{helper}; "
            f"chmod 755 /app/{helper}; sleep 60\n"
        )
        result = await run_scored(box, tmp_path, body, timeout=1)
        leaked = await box.exec("test -s /app/leaked")
        assert leaked.return_code == 1, "root cleanup disclosed the dummy marker"
        assert "before the script started" in (result.error or "")
        assert result.rewards is None


async def service_rollout(
    box, tmp_path, *, keep, failure="watchdog", verifier_user=None
):
    """Use production dispatch and cleanup; check freeze/release at verification."""
    assert (await box.exec("mkdir -p /logs/verifier")).return_code == 0
    task_path = tmp_path / "task"
    task_path.mkdir()
    metadata = "[metadata]\nx-keep-services = [{port = 8080}]\n" if keep else ""
    verifier = (
        f'[verifier]\nuser = "{verifier_user}"\n' if verifier_user is not None else ""
    )
    (task_path / "task.toml").write_text('version = "1.0"\n' + metadata + verifier)
    (task_path / "instruction.md").write_text("Serve the dummy answer.\n")
    server = (
        "import socket; s=socket.socket(); s.bind(('127.0.0.1',8080)); s.listen(); "
        "open('/app/ready','w').close(); "
        "exec(\"while True:\\n c,a=s.accept(); c.sendall(b'42'); c.close()\")"
    )
    script = tmp_path / "service.sh"
    script.write_text(
        "setsid sleep 60 </dev/null >/dev/null 2>&1 & echo $! > /app/stray.pid\n"
        f"setsid /usr/local/bin/python -I -c {shlex.quote(server)} "
        "</dev/null >/dev/null 2>&1 &\n"
        "while [ ! -e /app/ready ]; do sleep .01; done\n"
        "sleep 60\n"
    )
    rollout = Rollout(
        RolloutConfig(task_path=task_path, agent=f"script:{script}", timeout=1)
    )
    rollout._task = Task(task_path)
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "service-check"
    rollout._started_at = datetime.now()
    rollout._agent_cwd = "/app"
    rollout._timeout = 1
    rollout._env = box
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())
    if failure != "watchdog":
        original = box.script_runner_exec

        async def exec_with_timeout(command, **kwargs):
            result = await original(command, **kwargs)
            if command.startswith("timeout -s TERM"):
                if failure == "transport":
                    raise RuntimeError("Command timed out after 16 seconds")
                raise TimeoutError("host waiter expired")
            return result

        box.script_runner_exec = exec_with_timeout
    observed = {}

    async def verify():
        stray = await box.exec("cat /proc/$(cat /app/stray.pid)/status 2>/dev/null")
        observed["stray_ended"] = stray.return_code != 0 or "State:\tZ" in stray.stdout
        record = keep_services.turn_end(rollout._task)
        observed["record"] = record
        observed["waited"] = keep_services.waited(rollout._task)
        if keep and record and record["kept"]:
            pid = record["kept"][0]["pid"]
            state = await box.exec(f"cat /proc/{pid}/status")
            observed["frozen"] = "State:\tT" in state.stdout
            # The normal pre-grading kill must still find the frozen listener.
            outcome = await _kill_sandbox_user_procs_keeping(
                box, "agent", (8080,), wait=False
            )
            pids = tuple(p.pid for p in outcome.kept)
            released = await release_kept_services(box, "agent", pids)
            observed["released"] = released.released
        probe = (
            "import socket; s=socket.create_connection(('127.0.0.1',8080),1); "
            "assert s.recv(2)==b'42'"
        )
        response = await box.exec(
            f"/usr/local/bin/python -I -c {shlex.quote(probe)}",
            user=verifier_user or "root",
        )
        rollout._rewards = {"reward": float(response.return_code == 0)}
        await end_kept_services(box, "agent")
        closed = await box.exec(f"/usr/local/bin/python -I -c {shlex.quote(probe)}")
        observed["ended"] = closed.return_code != 0

    rollout.verify = verify
    result = await rollout.run()
    return result, observed


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("failure", ["watchdog", "transport", "timeout_error"])
async def test_timeout_retains_only_declared_services(tmp_path, backend, keep, failure):
    """Guards DT2-CONF-4 at 212f67e4 through real dispatch, freeze and release."""
    with started_box(image=PYTHON_IMAGE) as box:
        result, observed = await service_rollout(
            backend(box.name), tmp_path, keep=keep, failure=failure
        )
        assert result.error is None
        assert result.rewards == {"reward": float(keep)}
        assert result.trajectory[0]["timed_out"] is True
        assert observed["ended"]
        assert observed["stray_ended"]
        if keep:
            assert observed["waited"] and observed["frozen"]
            assert observed["released"] == 1
            assert observed["record"]["kept"][0]["port"] == 8080
        else:
            assert observed["record"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["absent", "kept", "root", "nonroot", "separate"])
@pytest.mark.parametrize(
    "ending",
    ["watchdog", "transport", "timeout_error", "normal", "cancel", "startup", "runner"],
)
async def test_script_cleanup_policy_and_failure_boundaries(tmp_path, policy, ending):
    """Guards DT2-CONF-4 at 212f67e4: only graded timeouts retain services."""
    from benchflow.rollout._setup import _run_script

    path = tmp_path / "task"
    path.mkdir()
    metadata = "[metadata]\nx-keep-services = [{port = 8080}]\n"
    verifier = {
        "root": '[verifier]\nuser = "root"\n',
        "nonroot": '[verifier]\nuser = "grader"\n',
        "separate": '[verifier]\nsandbox_mode = "separate"\n',
    }.get(policy, "")
    (path / "task.toml").write_text(
        'version = "1.0"\n' + (metadata if policy != "absent" else "") + verifier
    )
    (path / "instruction.md").write_text("Serve the answer.\n")
    task = Task(path)
    shell = FakeShell(
        code=0 if ending == "normal" else 124 if ending == "watchdog" else 127,
        marker="0" if ending == "normal" else None,
        started=ending != "startup",
    )
    original = shell.script_runner_exec

    async def execute(command, **kwargs):
        if command.startswith("timeout -s TERM"):
            if ending == "transport":
                raise RuntimeError("Command timed out after 16 seconds")
            if ending == "timeout_error":
                raise TimeoutError("waiter expired")
            if ending == "cancel":
                raise asyncio.CancelledError
        return await original(command, **kwargs)

    shell.script_runner_exec = execute
    outcome = SimpleNamespace(
        killed=2, waited_sec=0.25, kept=[SimpleNamespace(port=8080, pid=123)]
    )
    planes = SimpleNamespace(quiesce_agent=AsyncMock(return_value=outcome))
    attempted_timeout = ending in ("watchdog", "transport", "timeout_error")
    retains = attempted_timeout and policy in ("kept", "root")
    if ending in ("cancel", "startup", "runner"):
        with pytest.raises(
            asyncio.CancelledError if ending == "cancel" else RuntimeError
        ):
            await _run_script(
                shell,
                b"true\n",
                "script:x",
                1,
                "agent",
                "/app",
                {},
                planes=planes,
                task=task,
            )
    else:
        events, _ = await _run_script(
            shell,
            b"true\n",
            "script:x",
            1,
            "agent",
            "/app",
            {},
            planes=planes,
            task=task,
        )
        assert events[0]["timed_out"] is attempted_timeout
    if ending == "normal":
        planes.quiesce_agent.assert_not_awaited()
    else:
        planes.quiesce_agent.assert_awaited_once()
        assert planes.quiesce_agent.await_args.kwargs == (
            {"keep_ports": (8080,)} if retains else {}
        )
    assert keep_services.waited(task) is retains
    assert (keep_services.turn_end(task) is not None) is retains
