"""Behavioral guards for U5a-CONF-1, introduced by merge 8d34888f."""

import os
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult


@pytest.fixture(params=["docker", "docker-compose", "remote-docker", "daytona", "dind"])
def backend(request, tmp_path, monkeypatch):
    """Replay actual exec builders, replacing only the external transport/UID drop."""
    image_env = dict(os.environ)
    calls = []

    def run(argv, extra=None, stdin_data=None):
        calls.append(argv)
        done = subprocess.run(
            argv,
            cwd=tmp_path,
            env={**image_env, **(extra or {})},
            input=None if stdin_data is None else stdin_data.decode(),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return ExecResult(
            stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
        )

    async def compose(parts, **kwargs):
        extra = {}
        for index, part in enumerate(parts[:-1]):
            if part == "-e":
                key, value = parts[index + 1].split("=", 1)
                extra[key] = value
        # Docker's exec carries the env on stdin (main-sync b156).
        return run(parts[-3:], extra, kwargs.get("stdin_data"))

    name = request.param
    if name in ("docker", "docker-compose", "remote-docker"):
        from benchflow.sandbox.docker import DockerSandbox
        from benchflow.sandbox.remote_docker import RemoteDockerSandbox

        cls = RemoteDockerSandbox if name == "remote-docker" else DockerSandbox
        box = cls.__new__(cls)
        box._main_container = None if name == "docker-compose" else "dummy"
        box._run_docker_container_command = compose
        box._run_docker_compose_command = compose
    else:
        pytest.importorskip("daytona")
        from benchflow.sandbox import _su
        from benchflow.sandbox.daytona import (
            DaytonaSandbox,
            _DaytonaDinD,
            _DaytonaDirect,
            _load_daytona_sdk,
        )

        _load_daytona_sdk()
        box = DaytonaSandbox.__new__(DaytonaSandbox)
        if name == "dind":
            strategy = _DaytonaDinD.__new__(_DaytonaDinD)
            strategy._env = box
            strategy._compose_exec = compose
        else:
            # CI need not be root. The root-container repro separately runs
            # the genuine su; here only its identity switch is a transport fake.
            def same_uid(command, user, **kwargs):
                assert user == "root"
                return command

            monkeypatch.setattr(_su, "su_command", same_uid)
            results = []

            class Process:
                async def create_session(self, session):
                    pass

                async def execute_session_command(self, session, req, timeout=None):
                    results.append(run(["/bin/sh", "-c", req.command]))
                    return SimpleNamespace(cmd_id="dummy")

            async def poll(*args, **kwargs):
                return results.pop(0)

            box._sandbox = SimpleNamespace(process=Process())
            box._poll_response = poll
            strategy = _DaytonaDirect(box)
        box._strategy = strategy
    box.default_user = None
    box._persistent_env = {}
    return box, image_env, calls


@pytest.mark.parametrize("source", ["image", "persistent"])
@pytest.mark.parametrize(
    "spelling", ["absolute", "relative", "dot", "empty", "link", "spaces", "honest"]
)
@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "fallback"])
async def test_preparation_never_runs_planted_tools(
    backend, tmp_path, monkeypatch, source, spelling, outcome
):
    """Guards U5a-CONF-1 at 000895d1: helpers, steps and fallback use protected tools."""
    box, image_env, calls = backend
    planted = tmp_path / "agent bin"
    planted.mkdir()
    marker = tmp_path / "planted-ran"
    for tool in (
        "sh",
        "bash",
        "env",
        "base64",
        "mkdir",
        "rm",
        "cat",
        "sleep",
        "timeout",
        "mktemp",
        "find",
        "chmod",
        "chown",
        "readlink",
        "stat",
        "od",
        "tr",
        "cksum",
    ):
        real = shutil.which(tool)
        assert real is not None, tool
        script = planted / tool
        script.write_text(
            f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(tool)} >> {shlex.quote(str(marker))}\n"
            f'exec {shlex.quote(real)} "$@"\n'
        )
        script.chmod(0o755)
    (tmp_path / "alias").symlink_to(planted, target_is_directory=True)
    if spelling in ("dot", "empty"):
        for tool in planted.iterdir():
            (tmp_path / tool.name).symlink_to(tool)
    prefix = {
        "absolute": str(planted),
        "relative": "agent bin",
        "dot": ".",
        "empty": "",
        "link": str(tmp_path / "alias"),
        "spaces": str(planted) + ":" + str(planted),
    }.get(spelling)
    path = lockdown._SAFE_VERIFIER_PATH
    if prefix is not None:
        path = prefix + ":" + path
    image_env["PATH"] = path if source == "image" else lockdown._SAFE_VERIFIER_PATH
    box._persistent_env = {"PATH": path} if source == "persistent" else {}

    parent = tmp_path / "sticky"
    parent.mkdir(mode=0o1700)
    parent.chmod(0o1700)
    work = parent / "prep"
    monkeypatch.setattr(lockdown, "_PREP_DIR_PREFIX", str(work))
    monkeypatch.setattr(lockdown.secrets, "token_hex", lambda n: "a" * (2 * n))
    if outcome == "fallback":
        parent.chmod(0o700)  # Safe refusal before the batch starts any step.
    answer = tmp_path / "answer"
    data = shlex.quote(str(tmp_path / "data"))
    command = (
        f"mkdir -p {data}; printf honest > {data}/input; "
        f"cat {data}/input > {shlex.quote(str(answer))}; rm {data}/input; "
        "/bin/sleep 0.05"
    )
    if outcome == "failure":
        command += "; echo expected-error >&2; exit 7"
    elif outcome == "timeout":
        command += "; sleep 30"
    step = lockdown._PrepStep("probe", "probe", command, 1, "probe failed")
    error = None
    try:
        await lockdown._run_prep_steps(box, [step])
    except RuntimeError as exc:
        error = str(exc)
    assert not marker.exists(), marker.read_text() if marker.exists() else ""
    assert answer.read_text() == "honest"
    assert not work.with_name(work.name + "a" * 16).exists()
    assert len(calls) == (2 if outcome == "fallback" else 1)
    if outcome == "failure":
        assert error and "rc=7" in error and "expected-error" in error
    elif outcome == "timeout":
        assert error == "Command timed out after 1 seconds"
    else:
        assert error is None, error
