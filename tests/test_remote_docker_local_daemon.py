"""What assumed the Docker daemon is on this machine, checked for ``remote-docker``.

The ssh user of a remote host (often an access token) never reaches a
message, record or log: addresses docker would echo whole are refused
unquoted, every form ssh prints the user in is hidden, and the paths that
ran docker outside the redacting runners (``docker commit``, the task's
pre-compose hook, the agent's live process) hide it too. ``tcp://`` carries
the TLS settings on every call, and ``--sandbox docker`` keeps the user's own
``DOCKER_HOST`` and context (``bench doctor``: ``test_doctor_remote_docker.py``).
The fake ``docker`` CLI (``tests/_fake_docker_cli.py``) records what each
call saw.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.sandbox.remote_docker import (
    REMOTE_DOCKER_HOST_ENV,
    RemoteDockerConfigError,
    RemoteDockerHost,
    RemoteDockerSandbox,
    probe_remote_docker,
    resolve_remote_docker_host,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests import _fake_docker_cli
from tests._trust_admission import admitted

TOKEN = "tok3nlikeuser9f8e7d6c5b4a"
TOKEN_B = "s3condhalf0f1e2d3c4b5a69"
SSH_URL = f"ssh://{TOKEN}@builder.example.test:2222"


# What docker 29.1.3 prints for ``docker info`` with these DOCKER_HOST values
# (captured on a build VM): it ends the user
# at the first /, ? or # and shows the rest, and it quotes an address holding
# a control character with that character escaped.
DOCKER_29_SAYS = {
    f"ssh://{TOKEN}/{TOKEN_B}@box.example": (
        'error during connect: Get "http://docker.example.com/v1.52/info": '
        f"command [ssh -o ConnectTimeout=30 -T -- {TOKEN} docker "
        f"'--host=unix:///{TOKEN_B}@box.example' system dial-stdio] has exited "
        "with exit status 255, make sure the URL is valid, and Docker 18.09 or "
        "later is installed on the remote host: stderr=ssh: Could not resolve "
        f"hostname {TOKEN}: Temporary failure in name resolution"
    ),
    f"ssh://{TOKEN}?{TOKEN_B}@box.example": (
        "Failed to initialize: ssh host connection is not valid: invalid SSH "
        f'URL: query parameters are not allowed: "{TOKEN_B}@box.example"'
    ),
    f"ssh://{TOKEN}#{TOKEN_B}@box.example": (
        "Failed to initialize: ssh host connection is not valid: invalid SSH "
        f'URL: fragments are not allowed: "{TOKEN_B}@box.example"'
    ),
    f"ssh://{TOKEN}\t{TOKEN_B}@box.example:22": (
        f'Failed to initialize: parse "ssh://{TOKEN}\\t{TOKEN_B}@box.example:22": '
        "net/url: invalid control character in URL"
    ),
}


def _ssh_refused(user: str) -> str:
    """What docker prints when its ssh helper cannot log in as ``user``."""
    return (
        'error during connect: Get "http://docker.example.com/v1.55/info": '
        f"command [ssh -l {user} -p 2222 -o ConnectTimeout=30 -T -- "
        "builder.example.test docker system dial-stdio] has exited with exit "
        "status 255, make sure the URL is valid, and Docker 18.09 or later is "
        f"installed on the remote host: stderr={user}@builder.example.test: "
        "Permission denied (publickey)."
    )


@pytest.fixture(autouse=True)
def _clean_docker_env(monkeypatch):
    for name in (
        REMOTE_DOCKER_HOST_ENV,
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return _fake_docker_cli.install(tmp_path, monkeypatch)


def _certs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (root / name).write_text("pem\n")
    return root


def _environment(tmp_path: Path) -> Path:
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    return environment


def _sandbox(tmp_path: Path, cls=RemoteDockerSandbox, **config):
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return cls(
        environment_dir=_environment(tmp_path),
        environment_name="remote-task",
        session_id="remote-task__abc123",
        rollout_paths=paths,
        task_env_config=SandboxConfig(**config),
    )


# --- addresses docker would echo whole are refused, quoting no user ----------


@pytest.mark.parametrize(
    "value",
    [
        f"{TOKEN}@build-host.example:2222",
        f"ssh:{TOKEN}@build-host.example",
        f"unix:{TOKEN}@build-host.example",
    ],
)
def test_an_address_that_is_not_a_url_is_refused_without_its_user(monkeypatch, value):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, value)
    with pytest.raises(RemoteDockerConfigError) as info:
        resolve_remote_docker_host()
    assert TOKEN not in str(info.value)
    # Nor the address: the message names the variable.
    assert "build-host.example" not in str(info.value)
    assert REMOTE_DOCKER_HOST_ENV in str(info.value)


@pytest.mark.parametrize(
    "value",
    [
        f"ssh://{TOKEN}/{TOKEN_B}@build-host.example",
        f"ssh://{TOKEN}?{TOKEN_B}@build-host.example",
        f"ssh://{TOKEN}#{TOKEN_B}@build-host.example",
        f"tcp://{TOKEN}/{TOKEN_B}@build-host.example:2376",
    ],
)
def test_a_user_with_an_unencoded_delimiter_is_refused_unquoted(monkeypatch, value):
    # docker's parser ends the user at the first /, ? or #, takes the first
    # half for the host and quotes the rest ("extra path after the host").
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, value)
    with pytest.raises(RemoteDockerConfigError, match="percent-encode") as info:
        resolve_remote_docker_host()
    assert TOKEN not in str(info.value)
    assert TOKEN_B not in str(info.value)


@pytest.mark.parametrize(
    ("value", "says"),
    [
        (f"ssh://{TOKEN}%zz@build-host.example", "percent-encode"),
        (f"ssh://{TOKEN} x@build-host.example", "percent-encode"),
        (f"ssh://{TOKEN}:{TOKEN_B}@build-host.example", "password"),
        (f"ssh://{TOKEN}@build-host.example:22x", "unsupported"),
        # urlsplit drops tab, CR and LF; docker keeps them and quotes the
        # address with them escaped, which no form of the user matches.
        (f"ssh://{TOKEN}\t{TOKEN_B}@build-host.example:22", "control character"),
        (f"ssh://{TOKEN}\n{TOKEN_B}@build-host.example", "control character"),
        (f"ssh://{TOKEN}\r{TOKEN_B}@build-host.example", "control character"),
        (f"ssh://{TOKEN}@build\thost.example", "control character"),
        # urlsplit raises a plain ValueError for these, one of them quoting
        # the netloc (a fullwidth # that NFKC normalisation turns into #).
        (f"ssh://{TOKEN}@[::1", "not a URL docker can read"),
        (f"ssh://{TOKEN}@[gpu-box:22", "not a URL docker can read"),
        (f"ssh://{TOKEN}@[gpu-box]:22", "not a URL docker can read"),
        (f"ssh://{TOKEN}\uff03{TOKEN_B}@box.example", "not a URL docker can read"),
    ],
)
def test_an_address_docker_would_quote_in_its_error_is_refused_first(
    monkeypatch, value, says
):
    # Go's url.Parse refuses a bad escape, a space or a bad port with an
    # error holding the whole address; docker refuses a password.
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, value)
    with pytest.raises(RemoteDockerConfigError, match=says) as info:
        resolve_remote_docker_host()
    assert TOKEN not in str(info.value)
    assert TOKEN_B not in str(info.value)


def test_tcp_refusals_and_display_hide_a_user_part(monkeypatch, tmp_path):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, f"tcp://{TOKEN}@build-host:2376")
    with pytest.raises(RemoteDockerConfigError, match="needs TLS") as info:
        resolve_remote_docker_host()
    assert TOKEN not in str(info.value)
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", str(tmp_path / "none"))
    with pytest.raises(RemoteDockerConfigError, match=r"ca\.pem") as info:
        resolve_remote_docker_host()
    assert TOKEN not in str(info.value)
    monkeypatch.setenv("DOCKER_CERT_PATH", str(_certs(tmp_path / "certs")))
    host = resolve_remote_docker_host()
    assert host.display == "remote-docker[tcp]"
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS", "1")
    assert resolve_remote_docker_host().display == "tcp://***@build-host:2376"


# --- every form ssh prints the user in is hidden -------------------------------


def test_a_percent_encoded_user_is_hidden_as_ssh_prints_it(monkeypatch, fake):
    # docker decodes the user before it passes it to ssh's -l.
    encoded, decoded = f"{TOKEN}%2Fx", f"{TOKEN}/x"
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, f"ssh://{encoded}@builder.example.test")
    fake.set(fail={"info": _ssh_refused(decoded)})
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert "Permission denied (publickey)" in str(info.value)
    assert TOKEN not in str(info.value)


def test_a_long_reason_is_redacted_before_it_is_cut(monkeypatch, fake):
    # The reason line is cut at 300 characters; the user straddles the cut.
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(fail={"info": "stderr=" + "x" * 285 + f" {TOKEN}@builder: refused"})
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert TOKEN[:8] not in str(info.value)


def test_a_short_login_name_is_hidden_only_where_it_is_the_user():
    host = RemoteDockerHost(url="ssh://ubuntu@box.example", user="ubuntu")
    text = (
        "ssh -l ubuntu -p 22; ubuntu@box.example: Permission denied; "
        "as 'ubuntu'; FROM ubuntu:24.04"
    )
    assert host.redact(text) == (
        "ssh -l *** -p 22; ***@<remote host>: Permission denied; "
        "as '***'; FROM ubuntu:24.04"
    )


def test_a_token_user_is_hidden_everywhere_but_in_a_commands_own_output():
    host = RemoteDockerHost(url=SSH_URL, user=TOKEN)
    assert host.redact(f"no such user {TOKEN} here") == "no such user *** here"
    assert host.redact(f"echo {TOKEN}; -l {TOKEN}", command_output=True) == (
        f"echo {TOKEN}; -l ***"
    )


@pytest.mark.parametrize("url", list(DOCKER_29_SAYS))
def test_the_local_providers_docker_host_user_is_hidden_unvalidated(url):
    # The user's own DOCKER_HOST is not validated; docker splits a user with
    # an unencoded /, ? or # and escapes a control character, so its message
    # is replaced by one that says how to fix the address.
    from benchflow.sandbox.remote_docker import redact_docker_host

    out = redact_docker_host(DOCKER_29_SAYS[url], url)
    assert TOKEN not in out and TOKEN_B not in out
    assert "percent-encode" in out


def test_a_readable_docker_host_keeps_dockers_own_message():
    from benchflow.sandbox.remote_docker import redact_docker_host

    url = f"ssh://{TOKEN}@box.example"
    out = redact_docker_host(f"stderr={TOKEN}@box.example: Permission denied", url)
    assert out == "stderr=***@box.example: Permission denied"
    socket = "unix:///run/user@1000/docker.sock"
    assert redact_docker_host("no such file", socket) == "no such file"


async def test_a_failed_snapshot_names_no_ssh_user(tmp_path, monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    fake.set(fail={"commit": _ssh_refused(TOKEN)})
    with pytest.raises(RuntimeError, match="Permission denied") as info:
        await sandbox.snapshot("s1")
    assert TOKEN not in str(info.value)


async def test_a_failing_pre_compose_hook_names_no_ssh_user(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    sandbox = _sandbox(tmp_path)
    (sandbox.environment_dir / "benchflow-pre-compose.sh").write_text(
        f"echo '{_ssh_refused(TOKEN)}'\nexit 3\n"
    )
    with pytest.raises(RuntimeError, match="Permission denied") as info:
        await sandbox.start(force_build=False)
    assert TOKEN not in str(info.value)


@pytest.mark.parametrize("with_env", [True, False])
async def test_the_agent_process_names_no_ssh_user(
    tmp_path, monkeypatch, fake, with_env
):
    from benchflow.diagnostics import TransportClosedError

    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    fake.set(fail={"exec": _ssh_refused(TOKEN)})
    process = await sandbox.live_process(admission=admitted(sandbox))
    # Both modes now use one live exec; there is no separate file writer.
    try:
        await process.start("cat", env={"A": "1"} if with_env else None)
        with pytest.raises(TransportClosedError) as closed:
            while True:
                await process.readline()
        assert "Permission denied" in str(closed.value)
        assert TOKEN not in str(closed.value)
        assert TOKEN not in (closed.value.diagnostic.stderr_snippet or "")
        assert TOKEN not in process.stderr_tail
    finally:
        await process.close()


async def test_no_log_line_names_the_ssh_user(tmp_path, monkeypatch, fake, caplog):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    with caplog.at_level(logging.DEBUG):
        await sandbox.start(force_build=True)
        await sandbox.exec("echo hi")
        fake.set(fail={"down": _ssh_refused(TOKEN), "ps": _ssh_refused(TOKEN)})
        await sandbox.stop(delete=True)
    assert "remote-docker[ssh]" in caplog.text
    assert "builder.example.test" not in caplog.text
    assert TOKEN not in caplog.text


# --- tcp:// carries the TLS settings; --sandbox docker is unchanged ------------


async def test_every_tcp_call_carries_the_tls_settings(tmp_path, monkeypatch, fake):
    certs = _certs(tmp_path / "certs")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "tcp://builder.example.test:2376")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", str(certs))
    monkeypatch.setenv("DOCKER_CONTEXT", "some-local-context")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    await sandbox.exec("echo hi")
    process = await sandbox.live_process(admission=admitted(sandbox))
    await process.start("true")
    await process.close()
    await sandbox.stop(delete=True)
    calls = fake.calls()
    assert {c["argv"][0] for c in calls} >= {"info", "compose"}
    # On the command line (the shim), never in the environment Compose reads.
    assert {
        tuple(c["global"][2:] if c["global"][:1] == ["--config"] else c["global"])
        for c in calls
    } == {
        (
            "-H",
            "tcp://builder.example.test:2376",
            "--tlsverify",
            "--tlscacert",
            str(certs / "ca.pem"),
            "--tlscert",
            str(certs / "cert.pem"),
            "--tlskey",
            str(certs / "key.pem"),
        )
    }
    assert {
        (c["DOCKER_HOST"], c["DOCKER_TLS_VERIFY"], c["DOCKER_CERT_PATH"]) for c in calls
    } == {(None, None, None)}
    assert {c["DOCKER_CONTEXT"] for c in calls} == {None}


async def test_local_docker_keeps_the_users_own_docker_host(
    tmp_path, monkeypatch, fake
):
    """``--sandbox docker`` with the user's own DOCKER_HOST, context and TLS
    settings sends every call exactly there, never to the remote-docker host,
    and makes no host check."""
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import Task

    monkeypatch.setenv("DOCKER_HOST", "tcp://my-own-daemon.test:2376")
    monkeypatch.setenv("DOCKER_CONTEXT", "my-context")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", "/tmp/my-certs")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    (task / "instruction.md").write_text("Say hi.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = _create_sandbox_environment("docker", Task(task), task, "task__1", paths)
    assert type(sandbox) is DockerSandbox
    await sandbox.start(force_build=True)
    await sandbox.exec("echo hi")
    process = await sandbox.live_process(admission=admitted(sandbox))
    await process.start("true")
    await process.close()
    await sandbox.stop(delete=True)
    calls = fake.calls()
    assert calls
    for call in calls:
        flags = call["global"]
        if call["argv"][:1] == ["compose"] or "-H" in flags:
            # S1's isolated Compose process, and the commands on main made
            # with its connection (docker exec once main is known), carry
            # connection values only in flags; raw Docker retains the
            # operator's environment.
            assert flags[flags.index("-H") + 1] == "tcp://my-own-daemon.test:2376"
            assert "--tlsverify" in flags
            assert flags[flags.index("--tlscacert") + 1] == "/tmp/my-certs/ca.pem"
            assert flags[flags.index("--tlscert") + 1] == "/tmp/my-certs/cert.pem"
            assert flags[flags.index("--tlskey") + 1] == "/tmp/my-certs/key.pem"
            assert all(
                call[key] is None
                for key in (
                    "DOCKER_HOST",
                    "DOCKER_CONTEXT",
                    "DOCKER_TLS_VERIFY",
                    "DOCKER_CERT_PATH",
                )
            )
        else:
            assert (
                call["DOCKER_HOST"],
                call["DOCKER_CONTEXT"],
                call["DOCKER_TLS_VERIFY"],
                call["DOCKER_CERT_PATH"],
            ) == ("tcp://my-own-daemon.test:2376", "my-context", "1", "/tmp/my-certs")
    # Local Docker has no remote capacity check. U14 still probes the daemon
    # for subnet identity and pools, using the client's own settings above.
    assert ["info", "--format", "{{json .}}"] not in fake.argvs()
    # The local provider bind-mounts the rollout folders.
    files = [a for argv in fake.argvs() for a in argv if a.endswith(".yaml")]
    assert any(f.endswith("docker-compose-base.yaml") for f in files)
