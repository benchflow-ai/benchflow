"""Guards CONFIRM-lockdown-7: what lockdown decides on arrives as the sandbox
printed it, on every backend, whatever the backend's redaction rewrites.

Remote Docker masks the ssh user and the host in exec output. A workspace
mount alias, a plugin guard marker or any other path that contains them
must still reach the check unchanged, or the read must fail.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest

from benchflow._utils.reachable_write import ReachablePath
from benchflow.sandbox import lockdown
from benchflow.sandbox import remote_docker as remote
from benchflow.sandbox._base import ExecResult

USER = "worker"
HOST = "build.example.test"
TOKEN_USER = "tokenuser42"


@pytest.fixture(autouse=True)
def no_ssh_config(monkeypatch):
    # No connection or local ssh configuration is needed.
    if hasattr(remote.RemoteDockerHost, "ssh_view"):
        monkeypatch.setattr(
            remote.RemoteDockerHost,
            "ssh_view",
            lambda self: remote._SshView((), (), True, False),
        )


def _remote_transport(user: str):
    sandbox = object.__new__(remote.RemoteDockerSandbox)
    sandbox.docker_host = remote.RemoteDockerHost(f"ssh://{user}@{HOST}", user=user)
    return lambda result: sandbox._redact_result(result, command_output=True)


def _local_docker_transport(result):
    # Compose puts the exec's stderr into its stdout.
    return ExecResult(
        stdout=(result.stdout or "") + (result.stderr or ""),
        stderr="",
        return_code=result.return_code,
    )


def _daytona_transport(result):
    return result


BACKENDS = {
    "docker": _local_docker_transport,
    "remote-docker": _remote_transport(USER),
    "remote-docker-token-user": _remote_transport(TOKEN_USER),
    "daytona": _daytona_transport,
}


def _sandbox_answer(command: str, stdout: str, return_code: int = 0) -> ExecResult:
    """What the sandbox's shell prints for *command* when what it wraps
    printed *stdout*: the receipt line when lockdown asked for one."""
    from benchflow.sandbox._exec_receipt import (
        encode_receipt,
        unwrap_receipt_command,
    )

    unwrapped = unwrap_receipt_command(command)
    if unwrapped is not None:
        return ExecResult(
            stdout=encode_receipt(stdout.encode(), return_code, unwrapped[1]),
            stderr="",
            return_code=0,
        )
    return ExecResult(stdout=stdout, stderr="", return_code=return_code)


def _env(transport, answer):
    async def execute(command, **kwargs):
        return transport(_sandbox_answer(command, answer(command)))

    return SimpleNamespace(exec=execute)


# Every spelling remote Docker's redaction rewrites in a command's own output,
# each as part of a mount target the kernel can print (no raw spaces).
ALIASES = [
    f"/opt/{USER}@plugins",
    f"/opt/{TOKEN_USER}@plugins",
    f"/opt/'{USER}'/plugins",
    f'/opt/"{USER}"/plugins',
    f"/opt/x@{HOST}/plugins",
    f"/opt/[{HOST}/plugins",
    f"/opt/{HOST}:22/plugins",
    f"/srv/{HOST}/plugins",
]


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("alias", ALIASES)
async def test_workspace_mount_alias_reaches_the_provenance_check_unchanged(
    backend, alias
):
    mountinfo = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:2 / /app/plugins rw - tmpfs tmpfs rw\n"
        f"3 1 0:2 / {alias} rw - tmpfs tmpfs rw\n"
    )

    def answer(command):
        assert "/proc/self/mountinfo" in command
        return mountinfo + "# end\n" + "/app\0" + "\0"

    found = await lockdown._blocked_before_freeze(
        _env(BACKENDS[backend], answer), ("/app",)
    )

    assert alias in found
    assert not any("***" in path or "<remote host>" in path for path in found)


def _signed_line(name, detail, guard):
    body = json.dumps(detail)
    key = lockdown.pytest_plugin_guard_key(guard).encode()
    mac = hmac.new(key, f"{name}\n{body}".encode(), hashlib.sha256).hexdigest()
    return f"{name}\t{mac}\t{body}"


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("alias", ALIASES)
async def test_guard_refusal_marker_survives_the_backend(tmp_path, backend, alias):
    """A marker that makes the run unscored is never dropped in transit."""
    from benchflow.task.verifier_core import Verifier

    guard = "_benchflow_guard_" + "e" * 32
    name = f"{guard}.7-0123abcd.loading"
    line = _signed_line(name, f"plugin {alias}/evil.py", guard)

    def answer(command):
        return line + "\n"

    verifier = object.__new__(Verifier)
    verifier._sandbox = _env(BACKENDS[backend], answer)
    # On this branch a rollout's verifier/ is a held capability (R5-1).
    verifier._rollout_paths = SimpleNamespace(
        verifier_dir=ReachablePath.directory(tmp_path)
    )
    verifier._logger = lockdown.logger

    markers = await verifier._plugin_guard_markers(guard, "main")

    assert [m.name for m in markers.get("loading", [])] == [name]
    assert alias in markers["loading"][0].detail
