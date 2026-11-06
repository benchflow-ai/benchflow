"""``remote-docker``: data read exactly, one address for Compose, every ssh hop.

- What BenchFlow parses (``compose config``, inspect and list calls) is read
  exactly as docker printed it: a login or host name that collides with a
  key (``services``, ``volumes``) or a name cannot blind the checks.
- Compose gets the address once, on its command line (``compose_args``); its
  environment (and the pre-compose hook's) holds no connection setting, SSH
  agent, registry configuration, login name or shim; the agent reaches only
  the provider's ssh, and a build that asks for it (``build.ssh``) is
  refused.
- A failed data call (``compose ps``, ``inspect``) or a restore names
  neither the ssh user nor the host.
- A one-label host (``tcp://docker:2376``) is hidden where it names the
  host, never in docker's or BenchFlow's own words.
- Every ssh hop, ProxyJump hops included, reads the provider's ssh
  configuration, which sets ``StrictHostKeyChecking yes`` first; a hop
  through a ProxyCommand is refused.
- Every hop's login and host name is hidden; when a hop cannot be read, an
  ssh failure is named by its kind alone.
- Host names are hidden in any letter case.

The real ``ssh -G`` (OpenSSH; it reads configuration and connects nowhere)
resolves the configurations written here; the fake ``docker``
(``tests/_fake_docker_cli.py``) records each call.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from benchflow._utils.scoring import is_unrecoverable_startup_error
from benchflow.sandbox._compose import docker_daemon_unreachable_reason
from benchflow.sandbox.protocol import (
    SandboxImage,
    SandboxRestoreHostConfigUnavailable,
    SandboxStartupError,
)
from benchflow.sandbox.remote_docker import (
    REMOTE_DOCKER_HOST_ENV,
    RemoteDockerSandbox,
    probe_remote_docker,
    redact_docker_host,
    resolve_remote_docker_host,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests import _fake_docker_cli
from tests._trust_admission import admitted

pytestmark = pytest.mark.skipif(
    shutil.which("ssh") is None, reason="needs OpenSSH's ssh -G"
)

SSH_CONFIG_ENV = "BENCHFLOW_REMOTE_DOCKER_SSH_CONFIG"
_CLEAR = (
    REMOTE_DOCKER_HOST_ENV,
    "BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_TLS",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "DOCKER_CONFIG",
    "BUILDX_BUILDER",
)

_FAKE_REAL_SSH = r"""#!{python}
import json, sys
from pathlib import Path
with (Path({here!r}) / "ssh-argv.jsonl").open("a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\n")
sys.exit(255)
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in _CLEAR:
        monkeypatch.delenv(name, raising=False)
    # The operator's own ssh configuration, for these tests: none.
    user_config = tmp_path / "ssh-user-config"
    user_config.write_text("")
    monkeypatch.setenv(SSH_CONFIG_ENV, str(user_config))
    return user_config


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return _fake_docker_cli.install(tmp_path, monkeypatch)


def _ssh_config(tmp_path: Path, text: str) -> None:
    (tmp_path / "ssh-user-config").write_text(text)


def _certs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (root / name).write_text("pem\n")
    return root


def _tls(monkeypatch, tmp_path: Path, url: str) -> None:
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, url)
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", str(_certs(tmp_path / "certs")))


def _sandbox(tmp_path: Path, *, hook: str | None = None) -> RemoteDockerSandbox:
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    if hook is not None:
        (environment / "benchflow-pre-compose.sh").write_text(hook)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return RemoteDockerSandbox(
        environment_dir=environment,
        environment_name="remote-task",
        session_id="remote-task__abc123",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )


def _subcommand(argv: list[str]) -> str:
    rest = argv[1:]
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("--project-name", "-p", "--project-directory", "-f"):
            rest.pop(0)
    return rest[0] if rest else ""


_BIND = {
    "services": {
        "main": {
            "volumes": [
                {"type": "bind", "source": "/tmp/rd-marker", "target": "/marker"}
            ]
        }
    }
}


# --- data is read exactly as docker printed it -------------------------------


async def test_a_login_named_like_a_key_cannot_hide_a_bind_mount(
    tmp_path, monkeypatch, fake
):
    # "services" is eight characters: a token-length login is replaced
    # everywhere in a message, which must never reach the parsed JSON.
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://services@builder.example.test")
    fake.set(project=_BIND)
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="/tmp/rd-marker"):
        await sandbox.start(force_build=True)
    assert "up" not in [_subcommand(a) for a in fake.argvs() if a[:1] == ["compose"]]


@pytest.mark.parametrize(
    "volumes",
    [
        {"data": {"name": "rd-review-shared", "external": True}},
        {"data": {"name": "rd-review-shared"}},
    ],
)
async def test_a_host_named_like_a_key_cannot_hide_a_volume(
    tmp_path, monkeypatch, fake, volumes
):
    _tls(monkeypatch, tmp_path, "tcp://volumes:2376")
    fake.set(
        project={"services": {"main": {}}, "volumes": volumes},
        volumes_on_host={"rd-review-shared": {"Name": "rd-review-shared"}},
    )
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="refuses this task"):
        await sandbox.start(force_build=True)


async def test_a_volume_named_like_the_login_is_inspected_by_its_real_name(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://benchflow1@builder.example.test")
    fake.set(
        project={"services": {"main": {}}, "volumes": {"d": {"name": "benchflow1-d"}}},
        volumes_on_host={"benchflow1-d": {"Name": "benchflow1-d", "Labels": {}}},
    )
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError) as info:
        await sandbox.start(force_build=True)
    assert ["volume", "inspect", "benchflow1-d"] in fake.argvs()
    # The message, a diagnostic, still never names the login.
    assert "benchflow1" not in str(info.value)


async def test_teardown_removes_leftovers_by_their_real_names(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://benchflow1@builder.example.test")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    fake.set(leftovers={"volumes": ["benchflow1-vol"], "containers": ["c1"]})
    await sandbox.stop(delete=False)
    assert fake.script["leftovers"]["volumes"] == []
    # The shared teardown may remove it before the remote leftover pass.
    assert any(
        a[:2] == ["volume", "rm"] and a[-1] == "benchflow1-vol" for a in fake.argvs()
    )


# --- one address for Compose; nothing for a task's ${...} or the hook -------


async def test_compose_gets_the_address_once_and_nothing_in_its_environment(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/rd-agent.sock")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "docker-config"))
    monkeypatch.setenv("USER", "rd-operator")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    composes = [c for c in fake.calls() if c["argv"][:1] == ["compose"]]
    assert composes
    for call in composes:
        assert call["global"].count("-H") + call["global"].count("--host") == 1
        assert call["host"] == "ssh://worker@builder.example.test"
        from benchflow.task.compose_env import hermetic_docker_config

        assert call["global"][:2] == ["--config", hermetic_docker_config()]
        assert call["global"][1] != str(tmp_path / "docker-config")
        # Neither the agent (a task's ${SSH_AUTH_SOCK}, a build's ssh:) nor
        # the registry configuration nor the login name reaches Compose.
        assert call["SSH_AUTH_SOCK"] is None
        assert call["DOCKER_CONFIG"] is None
        assert call["USER"] is None
        assert call["BUILDX_BUILDER"] == "default"
        # Compose's PATH has the provider's ssh, but no docker shim.
        first = Path(call["PATH"].split(os.pathsep)[0])
        assert (first / "ssh").is_file() and not (first / "docker").exists()
        # That ssh is handed the agent itself.
        assert "SSH_AUTH_SOCK=/tmp/rd-agent.sock" in (first / "ssh").read_text()
    env = sandbox._docker_compose_env()
    for name in ("SSH_AUTH_SOCK", "DOCKER_CONFIG", "USER", "LOGNAME", "DOCKER_HOST"):
        assert name not in env


async def test_the_pre_compose_hook_gets_no_shim_agent_or_login(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/rd-agent.sock")
    monkeypatch.setenv("USER", "rd-operator")
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    dump = tmp_path / "hook.txt"
    sandbox = _sandbox(
        tmp_path, hook=f'env > "{dump}"; command -v docker >> "{dump}"\n'
    )
    await sandbox.start(force_build=True)
    seen = dump.read_text()
    assert "SSH_AUTH_SOCK" not in seen and "rd-operator" not in seen
    assert "worker@" not in seen and "benchflow-remote-docker" not in seen
    # Its docker is the real one (the fake here), never the address shim.
    assert seen.strip().splitlines()[-1] == str(fake.root / "docker")


def test_the_launch_gate_gets_the_same_flags_and_transport(tmp_path, monkeypatch, fake):
    _tls(monkeypatch, tmp_path, "tcp://builder.example.test:2376")
    host = resolve_remote_docker_host()
    environ = {"DOCKER_CONFIG": "/tmp/rd-docker-config", "PATH": os.environ["PATH"]}
    certs = tmp_path / "certs"
    assert host.compose_args(environ) == [
        "--config",
        "/tmp/rd-docker-config",
        "-H",
        "tcp://builder.example.test:2376",
        "--tlsverify",
        "--tlscacert",
        str(certs / "ca.pem"),
        "--tlscert",
        str(certs / "cert.pem"),
        "--tlskey",
        str(certs / "key.pem"),
    ]
    # tcp:// needs no agent and no ssh; the builder is pinned either way.
    assert host.compose_transport_env(
        {**environ, "SSH_AUTH_SOCK": "/tmp/rd-agent.sock"}
    ) == {"BUILDX_BUILDER": "default"}


# --- every ssh hop checks its host key ----------------------------------------


def test_the_provider_ssh_holds_every_hop_to_strict_host_key_checking(
    tmp_path, monkeypatch, fake
):
    import subprocess

    _ssh_config(
        tmp_path,
        "Host gateway.example.test\n  StrictHostKeyChecking no\n"
        "Host builder.example.test\n  ProxyJump gateway.example.test\n",
    )
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    env = resolve_remote_docker_host().client_env()
    shim = Path(env["PATH"].split(os.pathsep)[0]) / "ssh"
    # The shim runs ssh with -F: ssh hands that file to the ssh it starts
    # for a ProxyJump hop, so the hop reads the policy too.
    argv = shim.read_text()
    assert " -F " in argv and "StrictHostKeyChecking=yes" in argv
    config = argv.split(" -F ", 1)[1].split()[0].strip("'")
    lines = Path(config).read_text().splitlines()
    policy = lines.index("StrictHostKeyChecking yes")
    assert policy < next(i for i, ln in enumerate(lines) if ln.startswith("Include"))
    # What ssh itself makes of the gateway through that file (ssh -G
    # connects nowhere): strict, although the operator's file says no.
    out = subprocess.run(
        [str(shim), "-G", "gateway.example.test"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "stricthostkeychecking true" in out.splitlines()


def test_the_jump_hop_gets_the_provider_configuration(tmp_path, monkeypatch, fake):
    # A stand-in for the real ssh records what the shim passes on.
    real = fake.root / "ssh"
    real.write_text(_FAKE_REAL_SSH.format(python=sys.executable, here=str(fake.root)))
    real.chmod(real.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    import subprocess

    env = resolve_remote_docker_host().client_env()
    subprocess.run(["ssh", "-T", "--", "builder.example.test"], env=env, check=False)
    (argv,) = [
        json.loads(line)
        for line in (fake.root / "ssh-argv.jsonl").read_text().splitlines()
    ]
    assert argv[0] == "-F" and Path(argv[1]).is_file()
    assert argv[2:4] == ["-o", "StrictHostKeyChecking=yes"]


def test_a_proxy_command_is_refused_before_any_call(tmp_path, monkeypatch, fake):
    _ssh_config(tmp_path, "Host builder.example.test\n  ProxyCommand nc %h %p\n")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    with pytest.raises(SandboxStartupError, match="ProxyCommand") as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert "builder.example.test" not in str(info.value)
    assert fake.argvs() == []


# --- every hop's identity is hidden --------------------------------------------


def _ssh_refused(text: str) -> str:
    return (
        "error during connect: command [ssh -o ConnectTimeout=30 -T -- target "
        f"docker system dial-stdio] has exited with exit status 255: stderr={text}"
    )


def test_a_jump_hops_login_and_host_are_hidden(tmp_path, monkeypatch, fake):
    _ssh_config(
        tmp_path,
        "Host builder.example.test\n  ProxyJump gatewaylogin@gateway.internal.test\n",
    )
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    fake.set(
        fail={
            "info": _ssh_refused(
                "gatewaylogin@gateway.internal.test: Permission denied (publickey)."
            )
        }
    )
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    message = str(info.value)
    assert "Permission denied (publickey)" in message
    assert "gatewaylogin" not in message
    assert "gateway.internal.test" not in message


def test_an_unreadable_hop_is_refused_before_any_call(tmp_path, monkeypatch, fake):
    # A ProxyJump entry ssh -G prints but BenchFlow cannot parse: the hop's
    # identity and settings are unknown, so nothing connects.
    _ssh_config(tmp_path, "Host builder.example.test\n  ProxyJump hop@[::1\n")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    with pytest.raises(SandboxStartupError, match="for every hop") as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert "hop@" not in str(info.value)
    assert fake.argvs() == []


def test_an_unknown_login_in_sshs_form_is_masked(monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    host = resolve_remote_docker_host()
    out = host.redact("other.login@elsewhere.test: Permission denied (publickey).")
    assert out == "***@<remote host>: Permission denied (publickey)."


# --- host names in any letter case ---------------------------------------------


def test_a_capitalised_tls_host_is_hidden(tmp_path, monkeypatch, fake):
    _tls(monkeypatch, tmp_path, "tcp://Docker-Host.corp:2376")
    fake.set(
        fail={
            "info": 'error during connect: Get "https://Docker-Host.corp:2376/v1.47/'
            'info": dial tcp: lookup Docker-Host.corp on 10.0.0.2:53: no such host'
        }
    )
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert "docker-host.corp" not in str(info.value).lower()


def test_a_capitalised_ssh_host_is_hidden(monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@Builder.Example.Test")
    fake.set(
        fail={
            "info": _ssh_refused(
                "ssh: connect to host Builder.Example.Test port 22: Connection refused"
            )
        }
    )
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert "builder.example.test" not in str(info.value).lower()
    assert "Connection refused" in str(info.value)


# --- review round 1: failed data calls, a one-label host, the SSH agent --------

_LONG_LOGIN = "rdlogin7c1e0f9a4b2d"
_HELPER_FAILED = (
    "error during connect: command [ssh -o ConnectTimeout=30 -T -l {user} -- "
    "builder.example.test docker system dial-stdio] has exited with exit "
    "status 255: stderr={user}@builder.example.test: Permission denied (publickey)."
)


@pytest.mark.parametrize("failing", [("ps", "inspect"), ("inspect",)])
async def test_a_restore_after_an_ssh_failure_names_neither_user_nor_host(
    tmp_path, monkeypatch, fake, failing
):
    # A branch child or --from-checkpoint restores while ssh fails: compose
    # ps and docker inspect come back with docker's helper message.
    monkeypatch.setenv(
        REMOTE_DOCKER_HOST_ENV, f"ssh://{_LONG_LOGIN}@builder.example.test"
    )
    fake.set(fail={sub: _HELPER_FAILED.format(user=_LONG_LOGIN) for sub in failing})
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxRestoreHostConfigUnavailable) as info:
        await sandbox.restore(SandboxImage(provider="docker", ref="bf-snap-rd:1"))
    message = str(info.value)
    assert _LONG_LOGIN not in message
    assert "builder.example.test" not in message


async def test_a_failed_data_calls_stderr_is_redacted_and_its_stdout_kept(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(
        REMOTE_DOCKER_HOST_ENV, f"ssh://{_LONG_LOGIN}@builder.example.test"
    )
    fake.set(fail={"inspect": _HELPER_FAILED.format(user=_LONG_LOGIN)})
    sandbox = _sandbox(tmp_path)
    result = await sandbox._docker_cli(["inspect", "c0ffee"], check=False)
    assert result.return_code == 1
    assert "Permission denied (publickey)" in result.stderr
    assert _LONG_LOGIN not in result.stderr
    assert "builder.example.test" not in result.stderr


def test_a_redacted_error_keeps_its_class(monkeypatch, tmp_path, fake):
    monkeypatch.setenv(
        REMOTE_DOCKER_HOST_ENV, f"ssh://{_LONG_LOGIN}@builder.example.test"
    )
    sandbox = _sandbox(tmp_path)
    error = sandbox._remote_error(
        SandboxRestoreHostConfigUnavailable(f"inspect failed: {_LONG_LOGIN}@x: no")
    )
    assert isinstance(error, SandboxRestoreHostConfigUnavailable)
    assert _LONG_LOGIN not in str(error)


async def test_a_refusal_on_a_host_named_docker_is_not_retried(
    tmp_path, monkeypatch, fake
):
    # tcp://docker:2376 is the docker:dind service's usual address in CI.
    _tls(monkeypatch, tmp_path, "tcp://docker:2376")
    fake.set(project=_BIND)
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError) as info:
        await sandbox.start(force_build=True)
    message = str(info.value)
    assert message.startswith("Remote Docker host refuses this task: ")
    assert is_unrecoverable_startup_error(message)


def test_a_host_named_docker_leaves_dockers_own_words(tmp_path, monkeypatch, fake):
    _tls(monkeypatch, tmp_path, "tcp://docker:2376")
    host = resolve_remote_docker_host()
    daemon = host.redact(
        "Cannot connect to the Docker daemon at tcp://docker:2376. "
        "Is the docker daemon running?"
    )
    assert daemon == (
        "Cannot connect to the Docker daemon at remote-docker[tcp]. "
        "Is the docker daemon running?"
    )
    assert docker_daemon_unreachable_reason(daemon)
    api = host.redact("failed to connect to the docker API at tcp://docker:2376")
    assert docker_daemon_unreachable_reason(api)
    lookup = host.redact(
        'error during connect: Get "https://docker:2376/v1.51/info": dial tcp: '
        "lookup docker on 10.0.0.2:53: no such host"
    )
    assert lookup == (
        "remote-docker[tcp]: cannot connect to the Docker daemon; "
        "check the remote Docker host, TLS and proxy configuration"
    )


@pytest.mark.parametrize(
    "said",
    [
        "ssh: Could not resolve hostname gpu: Name or service not known",
        "ssh: connect to host gpu port 22: Connection refused",
        "[ssh -o ConnectTimeout=30 -T -l worker -- gpu docker system dial-stdio]",
        "No ED25519 host key is known for gpu and you have requested strict checking.",
        "Host key for gpu has changed and you have requested strict checking.",
        "worker@gpu: Permission denied (publickey).",
    ],
)
def test_a_one_label_ssh_host_is_hidden_where_ssh_names_it(monkeypatch, fake, said):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@gpu")
    out = resolve_remote_docker_host().redact(said)
    assert "gpu" not in out.replace("<remote host>", "")


async def test_a_build_that_asks_for_the_ssh_agent_is_refused(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/rd-agent.sock")
    fake.set(
        project={"services": {"main": {"build": {"context": ".", "ssh": ["default"]}}}}
    )
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="builds with `ssh:`") as info:
        await sandbox.start(force_build=True)
    assert is_unrecoverable_startup_error(str(info.value))
    ran = [_subcommand(a) for a in fake.argvs() if a[:1] == ["compose"]]
    assert "build" not in ran and "up" not in ran


_FAKE_REAL_SSH_ENV = r"""#!{python}
import json, os
from pathlib import Path
with (Path({here!r}) / "ssh-env.jsonl").open("a") as fh:
    fh.write(json.dumps(os.environ.get("SSH_AUTH_SOCK")) + "\n")
"""


def test_composes_ssh_gets_the_agent_its_environment_lacks(tmp_path, monkeypatch, fake):
    import subprocess

    real = fake.root / "ssh"
    real.write_text(
        _FAKE_REAL_SSH_ENV.format(python=sys.executable, here=str(fake.root))
    )
    real.chmod(real.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/rd-agent.sock")
    host = resolve_remote_docker_host()
    env = {**host.compose_env(dict(os.environ)), **host.compose_transport_env()}
    assert "SSH_AUTH_SOCK" not in env
    subprocess.run(["ssh", "-T", "--", "builder.example.test"], env=env, check=False)
    seen = (fake.root / "ssh-env.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in seen] == ["/tmp/rd-agent.sock"]


@pytest.mark.parametrize("url", ["ssh://me@gpu-box\\", "tcp://me@host\\1:2376"])
def test_an_address_with_a_backslash_does_not_crash_redaction(url):
    out = redact_docker_host(f"docker: cannot use {url}: bad address", url)
    assert "me@" not in out and "bad address" in out


def test_the_docs_promise_no_hook_opt_in_the_code_lacks():
    import benchflow.sandbox.remote_docker as module

    source = Path(module.__file__).parents[1]
    code = "".join(p.read_text() for p in source.rglob("*.py"))
    page = (source.parents[1] / "docs" / "sandboxes" / "remote-docker.md").read_text()
    if "ALLOW_TASK_HOST_ACCESS" not in code:
        # No opt-in gates the pre-compose hook: nothing may say one does.
        assert "opt-in" not in page
        assert "opt-in" not in (module.__doc__ or "")
        assert "runs it whenever the task has one" in page


# --- Codex round 3 ----------------------------------------------------------


async def test_a_pre_compose_hook_is_refused_without_the_switch(
    tmp_path, monkeypatch, fake
):
    """The hook ran as the operator before any check, and could
    read the provider's shim folder (the address, the agent socket, the TLS
    key's path) and copy it into the image. No environment stops a process
    running as the operator, so it now runs only behind the launch gate's
    switch."""
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    marker = tmp_path / "hook-ran"
    sandbox = _sandbox(tmp_path, hook=f'touch "{marker}"\n')
    with pytest.raises(SandboxStartupError) as info:
        await sandbox.start(force_build=True)
    message = str(info.value)
    assert "BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1" in message
    assert is_unrecoverable_startup_error(message)
    assert not marker.exists()
    ran = [_subcommand(a) for a in fake.argvs() if a[:1] == ["compose"]]
    assert "build" not in ran and "up" not in ran


def _operator_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@builder.example.test")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/rd-agent.sock")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "docker-config"))
    monkeypatch.setenv("USER", "rd-operator")
    monkeypatch.setenv("LOGNAME", "rd-operator")


def _like_every_compose_call(call: dict, tmp_path) -> None:
    assert call["global"].count("-H") + call["global"].count("--host") == 1
    from benchflow.task.compose_env import hermetic_docker_config

    assert call["global"][:2] == ["--config", hermetic_docker_config()]
    assert call["global"][1] != str(tmp_path / "docker-config")
    assert call["SSH_AUTH_SOCK"] is None
    assert call["DOCKER_CONFIG"] is None
    assert call["USER"] is None
    first = Path(call["PATH"].split(os.pathsep)[0])
    assert not (first / "docker").exists()


async def test_attach_uses_composes_flags_and_environment(
    tmp_path, monkeypatch, fake, capsys
):
    """Attach ran `docker compose ... exec` and `down` with the
    raw calls' environment (the agent socket, DOCKER_CONFIG, USER, LOGNAME,
    the docker shim), so `down` re-read the task's files with other values
    than every other call (a `${SSH_AUTH_SOCK:-1}` scale), and printed
    docker's error, login included, to the terminal."""
    _operator_env(monkeypatch, tmp_path)
    execs: list = []

    def no_exec(*args):
        execs.append(args)
        raise RuntimeError("attach replaced the process")

    monkeypatch.setattr(os, "execvpe", no_exec)
    monkeypatch.setattr(os, "execvp", no_exec)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    before = len(fake.calls())
    fake.set(
        fail={
            "exec": "error during connect: stderr=worker@builder.example.test: "
            "Permission denied (publickey).",
            "down": "services.main.scale must be an integer: worker /tmp/x",
        }
    )
    await sandbox.attach()
    assert execs == []
    calls = fake.calls()[before:]
    subs = [_subcommand(c["argv"]) for c in calls if c["argv"][:1] == ["compose"]]
    assert subs[:2] == ["exec", "down"]
    for call in calls:
        _like_every_compose_call(call, tmp_path)
    shown = capsys.readouterr().err
    assert "Permission denied (publickey)" in shown and "scale" in shown
    assert "worker@" not in shown and "builder.example.test" not in shown


async def test_the_live_process_uses_composes_flags_and_environment(
    tmp_path, monkeypatch, fake
):
    from benchflow.sandbox.process import DockerProcess

    _operator_env(monkeypatch, tmp_path)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    process = DockerProcess.from_sandbox_env(sandbox)
    before = len(fake.calls())
    await process.start("true")
    await process._process.wait()
    await process.close()
    (call,) = fake.calls()[before:]
    assert _subcommand(call["argv"]) == "exec"
    _like_every_compose_call(call, tmp_path)


def _hop_chain(count: int) -> str:
    """target -> hop1 -> ... -> hop<count>, the last through a ProxyCommand."""
    lines = ["Host target.example.test", "  ProxyJump hop1.example.test"]
    for n in range(1, count):
        lines += [f"Host hop{n}.example.test", f"  ProxyJump hop{n + 1}.example.test"]
    lines += [
        f"Host hop{count}.example.test",
        "  ProxyCommand /usr/bin/ssh -o StrictHostKeyChecking=no -W %h:%p gw.test",
    ]
    return "\n".join(lines) + "\n"


def test_a_chain_longer_than_the_hop_limit_is_refused_before_any_call(
    tmp_path, monkeypatch, fake
):
    """Discovery stopped at the hop limit before reaching the
    ProxyCommand (complete=False, proxy_command=False) and docker connected
    anyway; a later error naming an uncollected hop also went through
    unmasked. An incomplete view is now refused before any connection."""
    _ssh_config(tmp_path, _hop_chain(8))
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@target.example.test")
    with pytest.raises(SandboxStartupError, match="for every hop") as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert "hop8" not in str(info.value)
    assert fake.argvs() == []


def test_a_chain_within_the_limit_is_still_checked_for_a_proxy_command(
    tmp_path, monkeypatch, fake
):
    _ssh_config(tmp_path, _hop_chain(3))
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://worker@target.example.test")
    with pytest.raises(SandboxStartupError, match="ProxyCommand"):
        probe_remote_docker(resolve_remote_docker_host())
    assert fake.argvs() == []


@pytest.mark.parametrize(
    "said",
    [
        "tls: failed to verify certificate: x509: certificate is valid for "
        "other.example.test, not Docker-Host",
        "tls: failed to verify certificate: x509: certificate is not valid for "
        "any names, but wanted to match Docker-Host",
    ],
)
def test_a_one_label_tls_host_is_hidden_in_a_name_mismatch(
    tmp_path, monkeypatch, fake, said
):
    """Go names the dialed host after ", not " or "wanted to
    match ", contexts the one-label rule did not know, so tcp://Docker-Host
    stayed in startup errors and later Compose diagnostics."""
    _tls(monkeypatch, tmp_path, "tcp://Docker-Host:2376")
    fake.set(
        fail={
            "info": f'error during connect: Get "https://Docker-Host:2376/v1.47/info": {said}'
        }
    )
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    message = str(info.value)
    assert "docker-host" not in message.lower()
    assert "other.example.test" not in message
    # The same text after start, as a Compose call's error.
    later = resolve_remote_docker_host().redact(f"compose up failed: {said}")
    assert "docker-host" not in later.lower()
    assert later == (
        "remote-docker[tcp]: cannot connect to the Docker daemon; "
        "check the remote Docker host, TLS and proxy configuration"
    )


async def test_a_failed_volume_removal_warning_names_no_login(
    tmp_path, monkeypatch, fake, caplog
):
    """The leftover warning joined the exact volume names, so a
    volume named like the login (benchflow1-vol) put the login in the log."""
    import logging

    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://benchflow1@builder.example.test")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    fake.set(
        leftovers={"volumes": ["benchflow1-vol"]},
        fail={"volume rm": "Error response from daemon: volume is in use"},
    )
    with caplog.at_level(logging.WARNING):
        await sandbox.stop(delete=False)
    # Removal used the exact name; it failed, so the volume is still listed.
    assert ["volume", "rm", "-f", "benchflow1-vol"] in fake.argvs()
    warnings = [r.getMessage() for r in caplog.records if "left on" in r.getMessage()]
    assert warnings
    assert all("benchflow1" not in w for w in warnings)


@pytest.mark.parametrize("transport", ["ssh", "tls"])
@pytest.mark.parametrize("show_address", [False, True])
async def test_live_startup_logs_omit_transport_arguments(
    tmp_path, monkeypatch, fake, caplog, transport, show_address
):
    """Guards ea3e3de4's argv logging leak."""
    import logging

    if transport == "ssh":
        monkeypatch.setenv(
            REMOTE_DOCKER_HOST_ENV, "ssh://rd-test-login@builder.example.test"
        )
    else:
        _tls(monkeypatch, tmp_path, "tcp://Docker-Host:2376")
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS", str(int(show_address)))
    sandbox = _sandbox(tmp_path)
    process = await sandbox.live_process(admission=admitted(sandbox))
    try:
        with caplog.at_level(logging.DEBUG):
            await process.start("cat")
        assert "DockerProcess:" in caplog.text
        for hidden in (
            "rd-test-login",
            "builder.example.test",
            "Docker-Host",
            str(tmp_path / "certs"),
        ):
            assert hidden not in caplog.text
        # The actual subprocess still receives exactly the selected transport.
        await process._process.wait()
        assert fake.calls()[-1]["host"] == sandbox.docker_host.url
    finally:
        await process.close()


_TLS_FAILURES = [
    "tls: failed to verify certificate: x509: certificate is valid for "
    "other.example.test, not Docker-Host",
    "tls: failed to verify certificate: x509: certificate is not valid for "
    "any names, but wanted to match Docker-Host",
    "tls: failed to verify certificate: x509: cannot validate certificate for "
    "192.0.2.44 because it doesn't contain any IP SANs",
]


@pytest.mark.parametrize("said", _TLS_FAILURES)
@pytest.mark.parametrize("consumer", ["exec", "raw-exec", "live-with-env", "live"])
async def test_tls_failures_are_hidden_at_exec_and_live_boundaries(
    tmp_path, monkeypatch, fake, said, consumer
):
    """Guards ea3e3de4's command-output TLS bypass."""
    from benchflow.diagnostics import TransportClosedError

    dialed_host = "192.0.2.44" if "IP SANs" in said else "Docker-Host"
    _tls(monkeypatch, tmp_path, f"tcp://{dialed_host}:2376")
    sandbox = _sandbox(tmp_path)
    # A connection failure after a successful start, such as cert rotation.
    await sandbox.start(force_build=True)
    fake.set(
        fail={
            "exec": f'error during connect: Get "https://{dialed_host}:2376/": {said}'
        }
    )
    process = await sandbox.live_process(admission=admitted(sandbox))
    try:
        if consumer == "exec":
            result = await sandbox.exec("true")
            assert result.return_code == 1
            shown = (result.stdout or "") + (result.stderr or "")
        elif consumer == "raw-exec":
            result = await sandbox._docker_cli(["exec", "c0ffee", "true"], check=False)
            assert result.return_code == 1
            shown = (result.stdout or "") + (result.stderr or "")
        else:
            await process.start(
                "cat", env={"RD_DUMMY": "1"} if consumer == "live-with-env" else None
            )
            with pytest.raises(TransportClosedError) as info:
                await process.readline()
            shown = "\n".join(
                [
                    str(info.value),
                    process.stderr_tail,
                    info.value.diagnostic.stderr_snippet or "",
                    info.value.diagnostic.raw_message,
                ]
            )
        assert "failed to verify certificate" in shown
        assert "docker-host" not in shown.lower()
        assert "192.0.2.44" not in shown
        assert "other.example.test" not in shown
    finally:
        await process.close()


@pytest.mark.parametrize("show_address", [False, True])
async def test_tls_redaction_preserves_task_text_and_address_opt_in(
    tmp_path, monkeypatch, fake, show_address
):
    """Guards the task-output contract while fixing the TLS bypass at ea3e3de4."""
    _tls(monkeypatch, tmp_path, "tcp://Docker-Host:2376")
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS", str(int(show_address)))
    sandbox = _sandbox(tmp_path)
    task_text = "task data: Docker-Host other.example.test; certificate is valid for sample, not sample2"
    fake.set(fail={"exec": task_text})
    result = await sandbox.exec("false")
    assert result.stdout == task_text + "\n"
    process = await sandbox.live_process(admission=admitted(sandbox))
    try:
        await process.start("cat")
        await process._process.wait()
        await process._finish_stderr_drain(cancel_on_timeout=False)
        assert process.stderr_tail == task_text + "\n"
    finally:
        await process.close()
    host = sandbox.docker_host
    for said in _TLS_FAILURES:
        redacted = host.redact(said, command_output=True)
        assert host.redact(redacted, command_output=True) == redacted
        # A task can print the complete TLS prefix too.
        # Only a failed HTTP request to our daemon identifies its TLS error.
        assert redacted == said
