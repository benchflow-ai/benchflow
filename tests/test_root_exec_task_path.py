"""Guard PT-CONF-1, -2 and -3 at 7a220231: a task's own PATH is data for root.

A task's persistent environment (task ``[environment] env`` or the environment
manifest) can set PATH, say ``/app/.venv/bin:/usr/bin:/bin``, with ``/app``
the agent's to write. Root plumbing (``chown -R root:root /app`` before the
verifier) sourced that PATH after pinning the system one, so a planted
``chown`` ran as root. The credential snapshot's ``docker exec ... sh`` found
``sh`` on the image's PATH the same way. And the verifier's PATH probe, which
read the image PATH saved before the persistent env was sourced, lost a
trusted folder that PATH added. The commands are built by the real backends
and launched here as the transport launches them, with no daemon or mount.
"""

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ROOT_EXEC_PATH, ExecResult
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox.docker import DockerSandbox, _DockerCredentialOps

_ROOT_SPELLINGS = ["root", 0, "0", "0:0", "root:root"]


def _planted(tmp_path):
    planted = tmp_path / "app" / ".venv" / "bin"
    planted.mkdir(parents=True)
    for tool in ("sh", "bash", "chown", "find", "base64", "rm", "cat", "chmod"):
        executable = planted / tool
        executable.write_text(
            # To stderr as well, so a redirected call still shows it ran.
            f"#!/bin/sh\nprintf 'PLANTED-{tool}-uid-%s\\n' \"$(/usr/bin/id -u)\" "
            "| /usr/bin/tee /dev/stderr\n"
        )
        executable.chmod(0o755)
    return planted


def _docker(monkeypatch, persistent_env, image_path="/usr/bin:/bin"):
    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = dict(persistent_env)
    # The image's agent user: uid 1000, as exec_route's lookup would read.
    sandbox._named_uids = (sandbox._exec_generation, {("main", "agent"): 1000})

    async def native_exec(argv, **kwargs):
        command = argv[argv.index("main") + 1 :]
        # Docker's exec carries the env on stdin.
        data = kwargs.get("stdin_data")
        ran = subprocess.run(
            command,
            env={"PATH": image_path},
            input=None if data is None else data.decode(),
            capture_output=True,
            text=True,
        )
        return ExecResult(
            return_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr
        )

    monkeypatch.setattr(sandbox, "_run_docker_compose_command", native_exec)
    return sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("user", _ROOT_SPELLINGS)
async def test_root_plumbing_ignores_a_planted_tool_on_the_persistent_path(
    tmp_path, monkeypatch, user
):
    """PT-CONF-1: no explicit PATH on the call, the task's PATH planted."""
    planted = _planted(tmp_path)
    sandbox = _docker(monkeypatch, {"PATH": f"{planted}:/usr/bin:/bin", "T": "1"})
    result = await sandbox.exec(
        "find /dev/null -maxdepth 0 >/dev/null; chown --help >/dev/null; "
        'printf "real-%s" "$T"',
        user=user,
    )
    assert "PLANTED" not in result.stdout + result.stderr, result
    assert result.stdout.endswith("real-1"), (result.stdout, result.stderr)


@pytest.mark.asyncio
async def test_task_roles_and_the_verifier_keep_their_path(tmp_path, monkeypatch):
    """The boundary: the task role keeps its PATH, the verifier its own."""
    planted = _planted(tmp_path)
    task_path = f"{planted}:/usr/bin:/bin"
    sandbox = _docker(monkeypatch, {"PATH": task_path})
    for user in (None, "agent"):
        # The task's own command (C8's task route).
        with as_task_command(sandbox):
            result = await sandbox.exec('printf %s "$PATH"', user=user)
        assert result.stdout == task_path
    # The verifier runs test.sh on the task route (verifier_core), where its
    # validated PATH is the command's; root's plumbing never takes a PATH
    # from its caller.
    with as_task_command(sandbox):
        verifier = await sandbox.exec(
            'printf %s "$PATH"', user="root", env={"PATH": "/usr/bin:/bin"}
        )
    assert verifier.stdout == "/usr/bin:/bin"


@pytest.mark.parametrize("user", _ROOT_SPELLINGS)
def test_merge_env_carries_the_task_path_as_data_for_root(user):
    """Every backend's exec merges through this (Docker, Daytona, DinD,
    Modal, Apple, AgentCore)."""
    sandbox = object.__new__(DockerSandbox)
    sandbox._persistent_env = {"PATH": "/app/.venv/bin:/usr/bin", "K": "v"}
    merged = sandbox._merge_env(None, user)
    assert merged == {"_BENCHFLOW_IMAGE_PATH": "/app/.venv/bin:/usr/bin", "K": "v"}
    assert sandbox._merge_env({"PATH": "/v"}, user) == {"PATH": "/v", "K": "v"}
    for task_user in (None, "agent", 1000):
        assert sandbox._merge_env(None, task_user)["PATH"] == "/app/.venv/bin:/usr/bin"


@pytest.mark.asyncio
async def test_the_verifier_path_probe_sees_the_persistent_path(monkeypatch):
    """PT-CONF-3: a folder the persistent env adds reaches validation."""
    task_path = "/opt/task-tools/bin:/usr/bin:/bin"
    sandbox = _docker(monkeypatch, {"PATH": task_path})
    seen = []

    def extras_cmd(raw_path, blocked, agent=None):
        seen.append(raw_path.strip())
        return "printf '[]'"

    monkeypatch.setattr(lockdown, "_trusted_path_extras_cmd", extras_cmd)
    await lockdown._trusted_verifier_path(sandbox, "agent", "/app")
    assert seen == [task_path]
    # The PATH lock reads the same variable.
    lock_view = await sandbox.exec(
        'printf %s "${_BENCHFLOW_IMAGE_PATH-${PATH:-}}"', user="root"
    )
    assert lock_view.stdout == task_path


@pytest.mark.asyncio
async def test_credential_snapshot_run_as_root_uses_the_system_shell(
    tmp_path, monkeypatch
):
    """PT-CONF-2: ``docker exec -u 0:0 ... sh`` resolved sh on the image PATH."""
    planted = _planted(tmp_path)
    image_path = f"{planted}:/usr/bin:/bin"
    real_exec = asyncio.create_subprocess_exec

    async def native(*argv, **kwargs):
        command = argv[argv.index("ctr") + 1 :]
        kwargs["env"] = {"PATH": image_path}
        return await real_exec(*command, **kwargs)

    monkeypatch.setattr(
        "benchflow.sandbox.docker.asyncio.create_subprocess_exec", native
    )
    sandbox = SimpleNamespace(
        _docker_client_env=lambda: {}, docker_host=None, root_exec_path=ROOT_EXEC_PATH
    )
    ops = _DockerCredentialOps(sandbox, "ctr")  # type: ignore[arg-type]
    script = "find /dev/null -maxdepth 0 >/dev/null; base64 </dev/null; cat"
    result = await ops.run_as(0, 0, script, stdin=b"real-stdin")
    assert "PLANTED" not in result.stdout.decode() + result.stderr, result
    assert result.stdout == b"real-stdin", (result.stdout, result.stderr)


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["root", None])
async def test_modal_root_keeps_the_task_path_as_data_only(tmp_path, monkeypatch, user):
    """Modal hands the env over as process environment, not an env file."""
    import os

    import modal

    from benchflow.sandbox.modal_impl import ModalSandbox

    planted = _planted(tmp_path)
    task_path = f"{planted}:/usr/bin:/bin"
    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(lambda env: env))
    ran = {}

    async def aio(*argv, secrets=(), **kwargs):
        payload = argv[-1]
        if os.geteuid() != 0:
            # Modal runs as root; a non-root host emulates only that switch.
            payload = payload.replace("/bin/su root -s /bin/bash -c", "/bin/sh -c")
        env = {"PATH": "/usr/bin:/bin"}
        for secret in secrets:
            env.update(secret)
        done = subprocess.run(
            [*argv[:-1], payload], env=env, capture_output=True, text=True
        )
        ran["done"] = done

        async def read(name):
            return getattr(done, name)

        async def wait():
            return done.returncode

        return SimpleNamespace(
            stdout=SimpleNamespace(read=SimpleNamespace(aio=lambda: read("stdout"))),
            stderr=SimpleNamespace(read=SimpleNamespace(aio=lambda: read("stderr"))),
            wait=SimpleNamespace(aio=wait),
        )

    sandbox = object.__new__(ModalSandbox)
    sandbox.default_user = "root"
    sandbox._persistent_env = {"PATH": task_path}
    sandbox._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=aio))
    result = await sandbox.exec(
        'find /dev/null -maxdepth 0 >/dev/null; printf "%s|%s" "$PATH" '
        '"${_BENCHFLOW_IMAGE_PATH-}"',
        user=user,
    )
    assert "PLANTED" not in result.stdout + result.stderr, result
    path, task_seen = result.stdout.split("|")
    assert str(planted) not in path
    assert task_seen == task_path
