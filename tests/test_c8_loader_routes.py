"""Guard C8's loader routes (PT-C8-1 to PT-C8-5 at 0aee7fba).

Every test runs the text or argv a backend really builds, here, with the
environment that backend's process would start with: the image's, which
names an agent-writable library in ``LD_PRELOAD``. The library is missing
on purpose, so glibc names it on stderr for every program that starts with
it set ("cannot be preloaded"): a marker that needs no compiler. BenchFlow's
plumbing must never start a program with it; a task's own command gets the
image's value back just before it runs. Only marker files are written; the
privilege drop fails here (not root), which is fine: what ran before it is
the point.
"""

from __future__ import annotations

import contextlib
import logging
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.lockdown import build_priv_drop_cmd

#: The library an image's ENV names under the agent's workspace (missing here).
MARK = "/nonexistent/app/lib/bf-agent-profile.so"
IMAGE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _image_env(**extra: str) -> dict[str, str]:
    return {"PATH": IMAGE_PATH, "HOME": "/nonexistent", "LD_PRELOAD": MARK, **extra}


def _run(argv, env, stdin_data: bytes | None = None) -> ExecResult:
    ran = subprocess.run(
        list(argv),
        env=dict(env),
        input=None if stdin_data is None else stdin_data.decode(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return ExecResult(return_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr)


def _loaded(result: ExecResult) -> bool:
    """Whether some program started with the image's LD_PRELOAD."""
    return MARK in (result.stderr or "")


def _task_command(sandbox):
    enter = getattr(sandbox, "task_command", None)
    return enter() if callable(enter) else contextlib.nullcontext()


def _docker_exec_env(argv: list[str], image_env: dict[str, str]):
    """``docker exec [flags] <container> <cmd...>``: the env and the command."""
    env = dict(image_env)
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-e", "--env"):
            name, _, value = argv[i + 1].partition("=")
            env[name] = value
            i += 2
        elif arg in ("-u", "-w", "--user", "--workdir"):
            i += 2
        elif arg.startswith("-"):
            i += 1
        else:
            return env, argv[i + 1 :]
    raise AssertionError(f"no container in {argv}")


def _docker_sandbox(image_env, persistent=None):
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = dict(persistent or {})
    sandbox._main_container = "ctr"

    async def run(argv, check=False, timeout_sec=None, **kwargs):
        env, command = _docker_exec_env(argv, image_env)
        # Docker's exec carries the env on stdin.
        return _run(command, env, kwargs.get("stdin_data"))

    sandbox._run_docker_container_command = run
    return sandbox


# PT-C8-1: BenchFlow's own root steps (the kill before grading, the
# hardening chmods) and the image user's (user=None) plumbing.


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["root", None])
async def test_docker_root_plumbing_starts_no_program_with_the_image_preload(user):
    sandbox = _docker_sandbox(_image_env())
    result = await sandbox.exec("cat /dev/null && printf ok", user=user)
    assert result.stdout == "ok", result.stderr
    assert not _loaded(result), result.stderr


# PT-C8-2: a loader value in the task's persistent env (the environment
# manifest's) must not come back in a system-shell or plumbing exec.


@pytest.mark.asyncio
@pytest.mark.parametrize("system_shell", [True, False])
async def test_docker_plumbing_drops_the_persistent_env_loader_values(system_shell):
    sandbox = _docker_sandbox(
        {"PATH": IMAGE_PATH, "HOME": "/nonexistent"},
        persistent={"LD_PRELOAD": MARK, "TASK_FLAG": "kept"},
    )
    result = await sandbox.exec(
        'cat /dev/null && printf %s "$TASK_FLAG"',
        user="root",
        system_shell=system_shell,
    )
    assert result.stdout == "kept", result.stderr
    assert not _loaded(result), result.stderr


def _modal_sandbox(image_env, persistent=None):
    from benchflow.sandbox.modal_impl import ModalSandbox

    sandbox = object.__new__(ModalSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = dict(persistent or {})
    seen: dict = {}

    async def exec_aio(*argv, secrets=(), **kwargs):
        env = dict(image_env)
        for secret in secrets:
            env.update(secret)
        seen.setdefault("envs", []).append(env)
        result = _run(argv, env)
        process = MagicMock()
        process.stdout.read.aio = AsyncMock(return_value=result.stdout)
        process.stderr.read.aio = AsyncMock(return_value=result.stderr)
        process.wait.aio = AsyncMock(return_value=result.return_code)
        return process

    sandbox._sandbox = MagicMock()
    sandbox._sandbox.exec.aio = exec_aio
    return sandbox, seen


@pytest.mark.asyncio
async def test_modal_system_shell_keeps_the_persistent_preload_out(monkeypatch):
    pytest.importorskip("modal")
    import modal

    # Secret.from_dict needs no network: the dict is what reaches the process.
    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(lambda d: dict(d)))
    sandbox, _ = _modal_sandbox(
        {"PATH": IMAGE_PATH, "HOME": "/nonexistent"},
        persistent={"LD_PRELOAD": MARK},
    )
    # As the image's user: a root exec adds an su this test cannot replay.
    result = await sandbox.exec("cat /dev/null && printf ok", system_shell=True)
    assert result.stdout == "ok", result.stderr
    assert not _loaded(result), result.stderr


@pytest.mark.asyncio
async def test_agentcore_and_daytona_plumbing_stage_no_persistent_loader_value():
    """What the sealed channel (AgentCore) and the session (Daytona) source."""
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    sandbox = object.__new__(AgentCoreSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {"LD_PRELOAD": MARK, "TASK_FLAG": "kept"}
    sandbox.runtime_arn = "arn"
    sandbox.runtime_session_id = "s"
    staged: list[dict] = []

    async def stage_env(env, owner=None):
        staged.append(dict(env))
        return "/tmp/bf-env"

    sandbox._sealed = SimpleNamespace(stage_env=stage_env)
    sandbox._invoke_command = lambda payload, timeout: ExecResult(return_code=0)
    await sandbox.exec("true", user="root")
    assert staged and "LD_PRELOAD" not in staged[0], staged
    assert staged[0]["TASK_FLAG"] == "kept"


# The task's own command (test.sh as [verifier] user = "root"): its shell
# starts without the image's preload, which comes back just before it.


@pytest.mark.asyncio
async def test_docker_task_command_as_root_gets_the_preload_only_itself():
    # PID 1's copy of the image's value, as capture_pid1_image_env leaves it.
    sandbox = _docker_sandbox(_image_env(_BF_IMAGE_LD_PRELOAD=MARK))
    with _task_command(sandbox):
        result = await sandbox.exec(
            'printf %s "${LD_PRELOAD-unset}"', user="root", env={"A": "1"}
        )
    assert not _loaded(result), result.stderr
    assert result.stdout == MARK


# PT-C8-3: an agent env that names a loader variable (the first agent may
# have replaced that library before another role launches).


def _daytona_sandbox(env_seen: list) -> MagicMock:
    sandbox = MagicMock()
    sandbox.create_ssh_access = AsyncMock(return_value=MagicMock(token="t"))

    async def process_exec(command, env=None, timeout=None):
        env_seen.append((command, dict(env or {})))
        return MagicMock(exit_code=0, result="__BENCHFLOW_BOOTSTRAP_DONE__\n")

    sandbox.process.exec = AsyncMock(side_effect=process_exec)
    return sandbox


class _Stop(Exception):
    pass


async def _daytona_launch(is_dind: bool, env: dict, tmp_path: Path):
    from benchflow.sandbox.process.daytona import DaytonaProcess

    captured: list[str] = []
    seen: list = []

    def fake_ssh_args(config_path, remote_cmd):
        captured.append(remote_cmd)
        raise _Stop

    proc = DaytonaProcess(
        sandbox=_daytona_sandbox(seen),
        is_dind=is_dind,
        compose_cmd_base="docker compose -p t" if is_dind else "",
    )
    with (
        patch.object(
            DaytonaProcess, "_write_ssh_config", return_value=str(tmp_path / "c")
        ),
        patch.object(DaytonaProcess, "_ssh_args", side_effect=fake_ssh_args),
        pytest.raises(_Stop),
    ):
        await proc.start(command=build_priv_drop_cmd("agent-acp", "nobody"), env=env)
    return captured[0], seen


@pytest.mark.asyncio
async def test_daytona_direct_env_bootstrap_starts_without_the_agent_preload(tmp_path):
    _, seen = await _daytona_launch(False, {"LD_PRELOAD": MARK, "K": "v"}, tmp_path)
    env_file = tmp_path / "agent.env"
    bodies = []
    for command, env in seen:
        # Root's shells (the env-file write, its clean-up), with the env
        # Daytona starts them with.
        command = re.sub(r"/tmp/benchflow_env_[0-9a-f]+\.env", str(env_file), command)
        result = _run(["/bin/sh", "-c", command], {"PATH": IMAGE_PATH, **env})
        assert not _loaded(result), result.stderr
        if env_file.exists():
            bodies.append(env_file.read_text())
    # The agent's value is still there, under an inert name, for after the drop.
    body = bodies[0]
    assert "_BENCHFLOW_SET_LD_PRELOAD=" in body and "export LD_PRELOAD=" not in body


@pytest.mark.asyncio
async def test_daytona_dind_root_shell_starts_without_the_agent_preload(tmp_path):
    remote_cmd, _ = await _daytona_launch(True, {"LD_PRELOAD": MARK}, tmp_path)
    parts = shlex.split(remote_cmd[remote_cmd.index("docker compose") :])
    main = parts.index("main")
    # compose exec: the VM shell sourced the agent env, --env NAME forwards
    # it, -e NAME=value sets it; then root's shell starts in the container.
    vm_env = {"LD_PRELOAD": MARK, "_BENCHFLOW_SET_LD_PRELOAD": MARK}
    env = {"PATH": IMAGE_PATH, "HOME": "/nonexistent"}
    flags = parts[parts.index("exec") + 1 : main]
    i = 0
    while i < len(flags):
        if flags[i] in ("--env", "-e"):
            name, sep, value = flags[i + 1].partition("=")
            env[name] = value if sep else vm_env.get(name, "")
            i += 2
        else:
            i += 1
    result = _run(parts[main + 1 :], env)
    assert not _loaded(result), result.stderr


# PT-C8-4: shells the provider starts itself (Daytona's toolbox, AgentCore).


def _fake_daytona_provider(image_env):
    """A Daytona whose toolbox runs each command in a shell with the image's
    ENV under the sandbox's create-time env_vars."""
    created: list = []

    class _Sandbox:
        def __init__(self, params):
            env = {**image_env, **(getattr(params, "env_vars", None) or {})}
            self.env = env

            async def process_exec(command, env=None, timeout=None):
                ran = _run(["/bin/sh", "-c", command], {**self.env, **(env or {})})
                return SimpleNamespace(
                    result=ran.stdout + ran.stderr, exit_code=ran.return_code
                )

            self.process = SimpleNamespace(exec=process_exec)

        async def delete(self, timeout=60, wait=False):
            # Guards PR #13: admission refusal confirms cleanup before returning.
            assert wait is True and 0 < timeout <= 60
            return None

    return created, _Sandbox


@pytest.mark.asyncio
async def test_daytona_toolbox_shells_start_without_the_image_preload():
    """A Daytona image that sets a loader value is refused before the
    agent's turn (integ's admission, kept by PT round 9): no sandbox is
    left, and no later shell can start with that value."""
    pytest.importorskip("daytona")
    import daytona

    from benchflow.sandbox.daytona import DaytonaSandbox

    created, fake = _fake_daytona_provider(_image_env())
    sandbox = object.__new__(DaytonaSandbox)
    sandbox.logger = logging.getLogger("test")

    async def create_sandbox(params):
        created.append(params)
        sandbox._sandbox = fake(params)

    sandbox._create_sandbox = create_sandbox
    sandbox._require_sandbox = lambda: sandbox._sandbox
    params = daytona.CreateSandboxFromSnapshotParams(snapshot="s")
    # Refused either way: the admission names the loader settings, or the
    # probe's answer (here with glibc's preload warning in it) is not one.
    with pytest.raises(
        Exception, match=r"loader settings|Could not verify Daytona toolbox"
    ):
        await sandbox._create_sandbox_without_toolbox_bash_env(params)
    assert len(created) == 1
    assert sandbox._sandbox is None


def _dockerfile_env(text: str, image_env: dict[str, str]) -> dict[str, str]:
    """The ENV instructions' effect, as Docker expands them."""
    env = dict(image_env)
    for line in text.splitlines():
        if not line.startswith("ENV "):
            continue
        before = dict(env)
        for pair in shlex.split(line[4:]):
            name, _, value = pair.partition("=")
            env[name] = re.sub(
                r"\$\{(\w+)\}", lambda m, b=before: b.get(m.group(1), ""), value
            )
    return env


@pytest.mark.asyncio
async def test_agentcore_runtime_shells_start_without_the_image_preload(monkeypatch):
    from benchflow.sandbox.agentcore import AgentCoreSandbox
    from benchflow.sandbox.agentcore_image import AgentCoreImagePublisher

    publisher = object.__new__(AgentCoreImagePublisher)
    publisher._task_env_config = SimpleNamespace(docker_image="ubuntu:24.04")
    runtime_env = _dockerfile_env(publisher.generated_dockerfile_text(), _image_env())

    sandbox = object.__new__(AgentCoreSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox.runtime_arn = "arn"
    sandbox.runtime_session_id = "s"
    payloads: list[str] = []

    def invoke(payload, timeout):
        payloads.append(payload)
        return _run(["/bin/sh", "-c", payload], runtime_env)

    monkeypatch.setattr(sandbox, "_invoke_command", invoke)
    plumbing = await sandbox.exec("cat /dev/null && printf ok")
    assert plumbing.stdout == "ok", plumbing.stderr
    assert not _loaded(plumbing), plumbing.stderr
    # The task's own command still gets the image's value (builtins only).
    with _task_command(sandbox):
        task = await sandbox.exec('printf %s "${LD_PRELOAD-unset}"')
    assert task.stdout == MARK, task.stderr
    assert not _loaded(task), task.stderr


@pytest.mark.asyncio
async def test_terminal_backends_pin_before_stty():
    """Daytona PTY (direct) and AgentCore type into root's terminal: the
    keep-and-pin builtins come before stty, the first program it starts."""
    from benchflow.sandbox._root_launch import typed_root_prelude
    from benchflow.sandbox.process import daytona as daytona_process

    sandbox = MagicMock()
    sandbox.process.exec = AsyncMock(
        return_value=MagicMock(exit_code=0, result="__BENCHFLOW_BOOTSTRAP_DONE__\n")
    )
    inputs: list[str] = []

    class _Pty:
        def __init__(self, on_data):
            self._on_data = on_data

        async def wait_for_connection(self):
            return None

        async def send_input(self, payload):
            inputs.append(payload)
            if "echo '" in payload:
                marker = payload.split("echo '", 1)[1].split("'", 1)[0]
                await self._on_data(f"{marker}\n".encode())

        async def kill(self):
            return None

        async def disconnect(self):
            return None

    async def create_pty_session(*, id, on_data, envs=None):
        return _Pty(on_data)

    sandbox.process.create_pty_session = AsyncMock(side_effect=create_pty_session)
    proc = daytona_process.DaytonaPtyProcess(sandbox, "", "")
    await proc.start(command=build_priv_drop_cmd("agent-acp", "nobody"))
    assert inputs[0].startswith(typed_root_prelude()), inputs[0]

    from benchflow.sandbox.process import agentcore as agentcore_process

    source = Path(agentcore_process.__file__).read_text()
    stty_line = next(line for line in source.splitlines() if "{stty} raw -echo" in line)
    assert "typed_root_prelude(" in stty_line, stty_line


# PT-C8-5: an agent env that sets PATH to exactly the pinned system PATH.


def test_an_explicit_system_path_in_the_launch_env_is_kept(tmp_path):
    from benchflow.sandbox._base import SYSTEM_PATH
    from benchflow.sandbox._root_launch import root_launch_script

    env_file = tmp_path / "agent.env"
    env_file.write_text(f"export PATH={SYSTEM_PATH}\n")
    out = tmp_path / "path"
    script = root_launch_script(
        f'printf %s "$PATH" > {shlex.quote(str(out))}', env_path=str(env_file)
    )
    ran = _run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script],
        {"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/nonexistent"},
    )
    assert ran.return_code == 0, ran.stderr
    assert out.read_text() == SYSTEM_PATH


def test_the_launch_restores_provider_kept_and_launch_env_values(tmp_path):
    """After the drop (here: an agent as root) the agent gets the image's
    value a provider kept aside, and the launch env's own over it."""
    from benchflow.sandbox._root_launch import root_launch_script

    out = tmp_path / "out"
    command = f'printf "%s|%s" "$LD_PRELOAD" "$GCONV_PATH" > {shlex.quote(str(out))}'
    env_file = tmp_path / "agent.env"
    # As the launchers stage it (inert_task_env): under an inert name.
    env_file.write_text("export _BENCHFLOW_SET_GCONV_PATH=/agent/gconv\n")
    ran = _run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            root_launch_script(command, env_path=str(env_file)),
        ],
        {
            "PATH": IMAGE_PATH,
            "HOME": "/nonexistent",
            "LD_PRELOAD": "",
            "_BF_IMAGE_LD_PRELOAD": "/image/lib/x.so",
        },
    )
    assert ran.return_code == 0, ran.stderr
    assert out.read_text() == "/image/lib/x.so|/agent/gconv"
