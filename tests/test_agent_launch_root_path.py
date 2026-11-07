"""Guard C8 on the agent launch (PT-CONF-6 at 666fb157).

Root starts every agent and drops to the sandbox user from there. Each test
runs the launch text a backend really builds, here, with the image's PATH led
by a folder of planted tools (as an agent can leave in ``/app/.venv/bin``)
and the planted tools recording that they ran. Nothing before the privilege
drop may be one of them. The drop itself fails here (not root), which is fine:
what ran before it is the point. Only marker files are written.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox.lockdown import build_priv_drop_cmd

PLANTED = (
    "sh",
    "bash",
    "setpriv",
    "grep",
    "su",
    "rm",
    "python3",
    "tr",
    "cat",
    "stty",
    "which",
    "printenv",
    "sed",
    "chmod",
    "mkdir",
    "find",
)


def _plant(tmp_path: Path) -> tuple[Path, Path, str]:
    planted = tmp_path / "app/.venv/bin"
    planted.mkdir(parents=True)
    marks = tmp_path / "marks"
    marks.mkdir()
    for tool in PLANTED:
        exe = planted / tool
        exe.write_text(f"#!/bin/sh\n: > {marks}/{tool}\nexit 97\n")
        exe.chmod(0o755)
    return planted, marks, f"{planted}:/usr/local/bin:/usr/bin:/bin"


def _ran(marks: Path) -> list[str]:
    return sorted(p.name for p in marks.iterdir())


def _run(argv: list[str], image_path: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        env={"PATH": image_path, "HOME": "/nonexistent", **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def _fake_proc() -> AsyncMock:
    proc = AsyncMock()
    proc.pid = 1
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"", b""))
    return proc


def test_priv_drop_as_an_exec_runs_no_planted_tool(tmp_path):
    """The wrapper's own text (an exec as root, codex's probe) pins PATH first."""
    _, marks, image_path = _plant(tmp_path)
    _run(["/bin/sh", "-c", build_priv_drop_cmd("true", "nobody")], image_path)
    assert _ran(marks) == []


@pytest.mark.asyncio
async def test_docker_launch_runs_no_planted_tool_before_the_drop(tmp_path):
    from benchflow.sandbox.process import DockerProcess

    _, marks, image_path = _plant(tmp_path)
    proc = DockerProcess(project_name="p", project_dir="/d", compose_files=["/d/c.yml"])
    captured: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        captured.append(list(args))
        return _fake_proc()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await proc.start(command=build_priv_drop_cmd("agent-acp", "nobody"))

    argv = captured[-1]
    main = argv.index("main")
    assert "-e" in argv[:main] and "LD_PRELOAD=" in argv[:main]
    assert argv[main + 1].startswith("/")
    _run(argv[main + 1 :], image_path)
    assert _ran(marks) == []


@pytest.mark.asyncio
async def test_docker_root_agent_gets_the_task_environment_back(tmp_path):
    """No sandbox user: the agent runs as root, with the task's PATH and loader values."""
    from benchflow.sandbox.process import DockerProcess

    _, marks, image_path = _plant(tmp_path)
    proc = DockerProcess(project_name="p", project_dir="/d", compose_files=["/d/c.yml"])
    captured: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        captured.append(list(args))
        return _fake_proc()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await proc.start(
            command='printf "%s|%s|%s" "$PATH" "${LD_LIBRARY_PATH-unset}" "${BASH_ENV-unset}"'
        )
    argv = captured[-1]
    ran = _run(
        argv[argv.index("main") + 1 :], image_path, LD_LIBRARY_PATH="/opt/a b/$(x)"
    )
    assert ran.stdout.endswith(f"{image_path}|/opt/a b/$(x)|unset"), ran.stderr
    # The ACP line filter runs integ's isolated /usr/bin/python3 by absolute
    # path (df6e024f3), so no planted tool runs at all.
    assert _ran(marks) == []


@pytest.mark.asyncio
async def test_apple_launch_runs_no_planted_tool_before_the_drop(tmp_path):
    from benchflow.sandbox.process.apple import AppleContainerProcess

    _, marks, image_path = _plant(tmp_path)
    captured: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        captured.append(list(args))
        return _fake_proc()

    with patch(
        "benchflow.sandbox.process.apple.asyncio.create_subprocess_exec",
        side_effect=fake_exec,
    ):
        await AppleContainerProcess("bf_run").start(
            build_priv_drop_cmd("agent-acp", "nobody")
        )
    argv = captured[-1]
    name = argv.index("bf_run")
    assert "LD_PRELOAD=" in argv[:name]
    _run(argv[name + 1 :], image_path)
    assert _ran(marks) == []


async def _modal_launch(
    monkeypatch, command: str, env: dict[str, str] | None = None
) -> tuple[list[str], dict[str, str], bytes]:
    """The argv, the Secret's values and the stdin ModalProcess hands Modal."""
    import sys
    import types

    from benchflow.sandbox.process.modal import ModalProcess

    modal = types.ModuleType("modal")
    secrets: list[dict[str, str]] = []

    class Secret:
        @classmethod
        def from_dict(cls, values):
            secrets.append(dict(values))
            return cls()

    modal.Secret = Secret
    monkeypatch.setitem(sys.modules, "modal", modal)

    class _Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    written: list[bytes] = []
    proc = MagicMock()
    proc.stdout = _Stream()
    proc.stderr = _Stream()
    proc.stdin.write = written.append
    proc.stdin.drain.aio = AsyncMock(return_value=None)
    captured: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        captured.append(list(args))
        return proc

    sandbox = MagicMock()
    sandbox.exec.aio = fake_exec
    launch = ModalProcess(sandbox)
    await launch.start(command, env=env)
    await launch.close()
    return captured[-1], secrets[-1], b"".join(written)


@pytest.mark.asyncio
async def test_modal_launch_runs_no_planted_tool_before_the_drop(tmp_path, monkeypatch):
    """Modal starts root's bash by absolute path with the loader empty."""
    from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS

    _, marks, image_path = _plant(tmp_path)
    argv, secret, _ = await _modal_launch(
        monkeypatch, build_priv_drop_cmd("agent-acp", "nobody")
    )
    assert argv[0].startswith("/") and argv[1:4] == ["--noprofile", "--norc", "-p"]
    assert secret == dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, "")
    _run(argv, image_path, **secret)
    assert _ran(marks) == []


@pytest.mark.asyncio
async def test_modal_root_agent_gets_its_env_line_and_the_task_environment(
    tmp_path, monkeypatch
):
    """The env line is applied after the pin, read once, and the ACP
    bytes after it reach the agent intact; a root agent gets the task's PATH
    and loader values back, and the launch env's own loader values too."""
    _, marks, image_path = _plant(tmp_path)
    value = "it's $(x) `y`"
    argv, _, stdin = await _modal_launch(
        monkeypatch,
        'printf "%s|%s|%s|%s\\n" "$PATH" "${LD_LIBRARY_PATH-unset}" '
        '"$API_KEY" "${NODE_OPTIONS-unset}"; /bin/cat',
        env={"API_KEY": value, "NODE_OPTIONS": "--max-old-space-size=64"},
    )
    assert value not in " ".join(argv)
    acp = b'{"jsonrpc":"2.0","id":1}\n'
    ran = subprocess.run(
        argv,
        input=stdin + acp,
        env={
            "PATH": image_path,
            "HOME": "/nonexistent",
            "LD_LIBRARY_PATH": "/opt/a b/$(x)",
        },
        capture_output=True,
        timeout=60,
    )
    assert ran.stdout.decode() == (
        f"{image_path}|/opt/a b/$(x)|{value}|--max-old-space-size=64\n" + acp.decode()
    ), ran.stderr
    assert _ran(marks) == []


def test_agentcore_launch_line_runs_no_planted_tool(tmp_path):
    from benchflow.sandbox.process.agentcore import AgentCoreProcess

    _, marks, image_path = _plant(tmp_path)
    line = AgentCoreProcess._launch_command(
        build_priv_drop_cmd("agent-acp", "nobody"), None, str(tmp_path)
    )
    _run(["/bin/sh", "-c", line], image_path)
    assert _ran(marks) == []


def _daytona_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.create_ssh_access = AsyncMock(return_value=MagicMock(token="t"))
    sandbox.process.exec = AsyncMock(
        return_value=MagicMock(exit_code=0, result="__BENCHFLOW_BOOTSTRAP_DONE__\n")
    )
    return sandbox


class _Stop(Exception):
    pass


async def _daytona_remote_cmd(is_dind: bool, command: str, tmp_path: Path) -> str:
    from benchflow.sandbox.process.daytona import DaytonaProcess

    captured: list[str] = []

    def fake_ssh_args(config_path, remote_cmd):
        captured.append(remote_cmd)
        raise _Stop

    proc = DaytonaProcess(
        sandbox=_daytona_sandbox(),
        is_dind=is_dind,
        compose_cmd_base="docker compose -p t" if is_dind else "",
    )
    with (
        patch.object(
            DaytonaProcess,
            "_write_ssh_config",
            return_value=str(tmp_path / "ssh_config"),
        ),
        patch.object(DaytonaProcess, "_ssh_args", side_effect=fake_ssh_args),
        pytest.raises(_Stop),
    ):
        await proc.start(command=command)
    return captured[0]


@pytest.mark.asyncio
async def test_daytona_ssh_launch_runs_no_planted_tool_before_the_drop(tmp_path):
    _, marks, image_path = _plant(tmp_path)
    remote_cmd = await _daytona_remote_cmd(
        False, build_priv_drop_cmd("agent-acp", "nobody"), tmp_path
    )
    _run(["/bin/sh", "-c", remote_cmd], image_path)
    assert _ran(marks) == []


@pytest.mark.asyncio
async def test_daytona_dind_launch_runs_no_planted_tool_in_the_container(tmp_path):
    _, marks, image_path = _plant(tmp_path)
    remote_cmd = await _daytona_remote_cmd(
        True, build_priv_drop_cmd("agent-acp", "nobody"), tmp_path
    )
    parts = shlex.split(remote_cmd)
    main = parts.index("main")
    assert "LD_PRELOAD=" in parts[:main]
    _run(parts[main + 1 :], image_path)
    assert _ran(marks) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("compose", [False, True])
async def test_daytona_pty_launch_runs_no_planted_tool(tmp_path, compose):
    from benchflow.sandbox.process import daytona as daytona_process

    _, marks, image_path = _plant(tmp_path)
    scripts: list[str] = []

    async def fake_script_file(
        sandbox, *, remote_script_path, script, error_label, route=None
    ):
        scripts.append(script)
        raise _Stop

    proc = daytona_process.DaytonaPtyProcess(
        _pty_sandbox(), "", "docker compose -p t" if compose else ""
    )
    with (
        patch.object(
            daytona_process, "_bootstrap_daytona_script_file", fake_script_file
        ),
        pytest.raises(_Stop),
    ):
        await proc.start(command=build_priv_drop_cmd("agent-acp", "nobody"))
    script = scripts[0]
    if compose:
        # The DinD VM runs compose; root's shell in the container is the point.
        parts = shlex.split(script[script.index("exec docker compose") :])
        parts = parts[parts.index("main") + 1 :]
        assert parts[0].startswith("/")
        _run(parts, image_path)
    else:
        path = tmp_path / "script.sh"
        path.write_text(script)
        _run(["/bin/sh", str(path)], image_path)
    assert _ran(marks) == []


def _pty_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.process.exec = AsyncMock(
        return_value=MagicMock(exit_code=0, result="__BENCHFLOW_BOOTSTRAP_DONE__\n")
    )

    class _Pty:
        def __init__(self, on_data):
            self._on_data = on_data
            self.inputs: list[str] = []

        async def wait_for_connection(self):
            return None

        async def send_input(self, payload):
            self.inputs.append(payload)
            if "echo '" in payload:
                marker = payload.split("echo '", 1)[1].split("'", 1)[0]
                await self._on_data(f"{marker}\n".encode())

        async def kill(self):
            return None

        async def disconnect(self):
            return None

    sandbox.ptys = []

    async def create_pty_session(*, id, on_data, envs=None):
        pty = _Pty(on_data)
        sandbox.ptys.append(pty)
        return pty

    sandbox.process.create_pty_session = AsyncMock(side_effect=create_pty_session)
    return sandbox


@pytest.mark.asyncio
async def test_daytona_pty_typed_lines_run_no_planted_tool(tmp_path):
    """What is typed into root's terminal on a direct sandbox: stty and the start."""
    from benchflow.sandbox.process import daytona as daytona_process

    _, marks, image_path = _plant(tmp_path)
    sandbox = _pty_sandbox()
    proc = daytona_process.DaytonaPtyProcess(sandbox, "", "")
    await proc.start(command=build_priv_drop_cmd("agent-acp", "nobody"))
    stty_line, start_line = sandbox.ptys[0].inputs
    # The script file the start line names is not here: point it at a no-op.
    noop = tmp_path / "noop.sh"
    noop.write_text("true\n")
    words = shlex.split(start_line)
    start_line = start_line.replace(words[-1], str(noop))
    _run(["/bin/sh", "-c", stty_line.split("; echo", 1)[0]], image_path)
    _run(["/bin/sh", "-c", start_line], image_path)
    assert _ran(marks) == []


def test_daytona_env_bootstrap_runs_no_planted_tool(tmp_path):
    """The env file root writes before the launch: builtins and system tools."""
    from benchflow.sandbox.process.daytona import DaytonaProcess

    _, marks, image_path = _plant(tmp_path)
    env_file = tmp_path / "agent.env"
    command = DaytonaProcess._bootstrap_env_command(
        remote_env_path=str(env_file), env_keys=["KEY_A"], shell_exports=True
    )
    ran = _run(["/bin/sh", "-c", command], image_path, KEY_A="it's $(x) `y`")
    assert ran.returncode == 0, ran.stderr
    assert _ran(marks) == []
    probe = _run(
        ["/bin/sh", "-c", f'. {env_file}; printf %s "$KEY_A"'], "/usr/bin:/bin"
    )
    assert probe.stdout == "it's $(x) `y`"


@pytest.mark.asyncio
async def test_agent_lookup_uses_the_shell_not_a_which_on_path(tmp_path):
    """connect_acp's agent-path lookup runs as the default user (root)."""
    from benchflow.acp import runtime

    _, marks, image_path = _plant(tmp_path)
    commands: list[str] = []

    async def exec_(command, **kwargs):
        commands.append(command)
        ran = _run(["/bin/sh", "-c", command], image_path)
        raise _Stop(ran)

    env = MagicMock()
    env.exec = exec_
    with pytest.raises(_Stop):
        await runtime.connect_acp(
            env=env,
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp",
            agent_env={},
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="daytona",
            agent_cwd="/app",
        )
    assert _ran(marks) == []


def test_no_bare_shell_in_the_live_process_backends():
    """Siblings: no live-process backend starts root's shell by name."""
    root = Path(__file__).resolve().parents[1] / "src/benchflow/sandbox/process"
    for path in root.glob("*.py"):
        text = path.read_text()
        for bare in ('"bash", "-c"', '"bash", "-lc"', "exec bash ", "exec sh "):
            assert bare not in text, f"{path.name}: {bare}"
    assert os.path.exists("/bin/sh")
