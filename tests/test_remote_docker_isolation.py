"""``remote-docker`` keeps a task's files away from the transport and the host.

What a remote Docker host must keep from a task and from logs:

- the address and TLS settings never reach the environment Compose fills a
  task's ``${...}`` from, or the pre-compose hook's: the provider's docker
  shim passes them as flags;
- a task's bind mounts, ``file:`` secrets, host-path and ``external``
  volumes, and a volume someone else already has on the host, are refused
  before anything is built (a Compose warning on stderr is not part of the
  project it reads);
- teardown removes only what carries the trial's project label or was built
  for it: never ``down --volumes`` or ``--rmi all``;
- builds go to buildx's ``default`` builder whatever ``BUILDX_BUILDER`` says;
- the login ssh's configuration picks, and the host name it resolves, are
  hidden like a user in the URL;
- ssh runs with ``StrictHostKeyChecking=yes`` whatever its configuration says;
- messages, records and logs name the host by an opaque label unless
  ``BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS=1``.

The fake ``docker`` (``tests/_fake_docker_cli.py``) records each call; a fake
``ssh`` answers ``ssh -G`` and records everything else.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.sandbox.remote_docker import (
    REMOTE_DOCKER_HOST_ENV,
    RemoteDockerSandbox,
    probe_remote_docker,
    resolve_remote_docker_host,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests import _fake_docker_cli
from tests._trust_admission import admitted

TOKEN = "tok3nlikeuser9f8e7d6c5b4a"
SSH_URL = f"ssh://{TOKEN}@builder.example.test:2222"
_CONNECTION = (
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_TLS",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)

_FAKE_SSH = r"""#!{python}
import json, sys
from pathlib import Path

HERE = Path({here!r})
argv = sys.argv[1:]
if "-G" in argv:
    config = json.loads((HERE / "ssh-config.json").read_text())
    for key, value in config.items():
        print(key, value)
    sys.exit(0)
with (HERE / "ssh-calls.jsonl").open("a") as fh:
    fh.write(json.dumps(argv) + "\n")
sys.exit(255)
"""


@pytest.fixture(autouse=True)
def _clean_docker_env(monkeypatch):
    for name in (
        REMOTE_DOCKER_HOST_ENV,
        "BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS",
        "BUILDX_BUILDER",
        *_CONNECTION,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return _fake_docker_cli.install(tmp_path, monkeypatch)


def _fake_ssh(root: Path, **config: str) -> Path:
    """A fake ``ssh`` in ``root`` (put first on PATH by ``fake``)."""
    exe = root / "ssh"
    exe.write_text(_FAKE_SSH.format(python=sys.executable, here=str(root)))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    (root / "ssh-config.json").write_text(json.dumps(config))
    return exe


def _certs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (root / name).write_text("pem\n")
    return root


def _sandbox(
    tmp_path: Path, *, hook: str | None = None, **config
) -> RemoteDockerSandbox:
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
        task_env_config=SandboxConfig(**config),
    )


def _compose_calls(fake) -> list[dict]:
    return [c for c in fake.calls() if c["argv"][:1] == ["compose"]]


def _subcommand(argv: list[str]) -> str:
    rest = argv[1:]
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("--project-name", "-p", "--project-directory", "-f"):
            rest.pop(0)
    return rest[0] if rest else ""


# --- no connection setting where a task's files can read it --------


async def test_compose_never_sees_the_address_or_the_tls_settings(
    tmp_path, monkeypatch, fake
):
    certs = _certs(tmp_path / "certs")
    monkeypatch.setenv("DOCKER_HOST", "tcp://builder.example.test:2376")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", str(certs))
    monkeypatch.setenv("DOCKER_TLS", "1")
    monkeypatch.setenv("DOCKER_CONTEXT", "some-local-context")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    await sandbox.exec("echo hi")
    process = await sandbox.live_process(admission=admitted(sandbox))
    await process.start("true")
    await process.close()
    await sandbox.stop(delete=True)

    calls = _compose_calls(fake)
    assert {_subcommand(c["argv"]) for c in calls} >= {"config", "build", "up"}
    # The command ran on main: docker exec with Compose's daemon flags once
    # main is known, compose exec before. Both are among the calls below.
    assert any(_subcommand(c["argv"]) == "exec" for c in calls) or any(
        c["argv"][:1] == ["exec"] for c in fake.calls()
    )
    for name in (
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "DOCKER_CONTEXT",
    ):
        assert {c[name] for c in fake.calls()} == {None}, name
    # Compose fills ${...} from its own environment: nothing there names the
    # host or the certificate folder.
    env = sandbox._docker_compose_env()
    for name in _CONNECTION:
        assert name not in env
    assert "builder.example.test" not in json.dumps(env)
    assert str(certs) not in json.dumps(env)
    assert {c["host"] for c in fake.calls()} == {"tcp://builder.example.test:2376"}


async def test_the_pre_compose_hook_gets_no_connection_setting(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    monkeypatch.setenv("DOCKER_HOST", SSH_URL)
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    dump = tmp_path / "hook-env.txt"
    sandbox = _sandbox(tmp_path, hook=f'env > "{dump}"\n')
    await sandbox.start(force_build=False)
    seen = dump.read_text()
    assert TOKEN not in seen
    assert "builder.example.test" not in seen
    assert "DOCKER_HOST=" not in seen


# --- a task cannot mount the remote host's files -------------------


@pytest.mark.parametrize(
    ("project", "named"),
    [
        (
            {
                "services": {
                    "main": {
                        "volumes": [
                            {
                                "type": "bind",
                                "source": "/tmp/rd-remote-marker",
                                "target": "/rd-marker",
                            }
                        ]
                    }
                }
            },
            "/tmp/rd-remote-marker",
        ),
        (
            {
                "services": {"main": {}},
                "secrets": {"rd": {"name": "rd", "file": "/tmp/rd-secret-marker"}},
            },
            "secret 'rd'",
        ),
        (
            {
                "services": {"main": {}},
                "volumes": {
                    "host": {
                        "name": "p_host",
                        "driver_opts": {
                            "type": "none",
                            "o": "bind",
                            "device": "/tmp/rd-device-marker",
                        },
                    }
                },
            },
            "/tmp/rd-device-marker",
        ),
    ],
)
async def test_a_task_that_reaches_host_paths_is_refused_before_any_build(
    tmp_path, monkeypatch, fake, project, named
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(project=project)
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError) as info:
        await sandbox.start(force_build=True)
    message = str(info.value)
    assert "Remote Docker host refuses this task" in message
    assert named in message
    subs = {_subcommand(c["argv"]) for c in _compose_calls(fake)}
    assert subs == {"config"}


def test_a_task_refusal_is_not_retried():
    from benchflow.evaluation import RetryConfig

    message = (
        "Sandbox startup failed: Remote Docker host refuses this task: service "
        "'main' bind-mounts '/tmp/x', a path on the remote host"
    )
    assert RetryConfig().should_retry(message) is False


# --- volumes and images the trial does not own stay ---------------


async def test_a_named_volume_someone_else_has_is_refused(tmp_path, monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(
        project={
            "services": {"main": {}},
            "volumes": {"data": {"name": "rd-review-shared"}},
        },
        volumes_on_host={
            "rd-review-shared": {"Name": "rd-review-shared", "Labels": None}
        },
    )
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="rd-review-shared"):
        await sandbox.start(force_build=False)
    subs = [_subcommand(c["argv"]) for c in _compose_calls(fake)]
    assert "up" not in subs
    await sandbox.stop(delete=True)
    assert not any(a[:2] == ["volume", "rm"] for a in fake.argvs())


async def test_an_external_volume_is_refused(tmp_path, monkeypatch, fake):
    # RD-R1-B-2: `external: true` mounts whatever volume the task names on the
    # host (another team's database) as it is.
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(
        project={
            "services": {
                "main": {
                    "volumes": [
                        {"type": "volume", "source": "shared", "target": "/data"}
                    ]
                }
            },
            "volumes": {"shared": {"name": "other-team-db", "external": True}},
        },
        volumes_on_host={
            "other-team-db": {"Name": "other-team-db", "Labels": {}},
        },
    )
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError) as info:
        await sandbox.start(force_build=False)
    message = str(info.value)
    assert message.startswith("Remote Docker host refuses this task")
    assert "other-team-db" in message
    assert "external" in message
    subs = {_subcommand(c["argv"]) for c in _compose_calls(fake)}
    assert subs == {"config"}


async def test_a_compose_warning_on_stderr_does_not_refuse_the_task(
    tmp_path, monkeypatch, fake
):
    # RD-R1-A-1: Compose warns on stderr (an obsolete `version:` key, a
    # variable that is not set) before it prints the project; read together
    # with stdout, the JSON failed to parse and a valid task was refused.
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(
        config_stderr=(
            'time="2026-10-16T12:00:00Z" level=warning msg="The '
            '\\"HOST_AGENT_LOGS_PATH\\" variable is not set. Defaulting to a '
            'blank string."\n'
            'time="2026-10-16T12:00:00Z" level=warning msg="docker-compose.yaml: '
            "the attribute `version` is obsolete, it will be ignored, please "
            'remove it to avoid potential confusion"\n'
        )
    )
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    subs = [_subcommand(c["argv"]) for c in _compose_calls(fake)]
    assert "config" in subs
    assert "up" in subs


async def test_own_volumes_start(tmp_path, monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    project = sandbox.compose_project_name
    fake.set(
        project={
            "services": {"main": {}},
            "volumes": {
                "data": {"name": f"{project}_data"},
            },
        },
        volumes_on_host={
            f"{project}_data": {
                "Name": f"{project}_data",
                "Labels": {"com.docker.compose.project": project},
            },
        },
    )
    await sandbox.start(force_build=False)
    inspected = [a for a in fake.argvs() if a[:2] == ["volume", "inspect"]]
    assert inspected == [["volume", "inspect", f"{project}_data"]]


async def test_teardown_never_removes_every_volume_or_image_of_the_project(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    project = sandbox.compose_project_name
    await sandbox.start(force_build=True)
    from tests.test_teardown_confirmation import InventoryDaemon

    # U10 owns image inventory/reference semantics on every Docker host.
    daemon = InventoryDaemon(project)
    daemon.built("sha256:aaa", "bf__remote-task:latest", "main")
    daemon.built("sha256:bbb", f"{project}-api:latest", "api")
    daemon.add_image("sha256:ccc", [], project)
    daemon.built("sha256:ddd", "bf__remote-task:old", "main")
    daemon.images["sha256:ddd"]["tags"].append("operator/keep:1")
    sandbox._built_image_tags = {"bf__remote-task:latest", f"{project}-api:latest"}
    sandbox._private_built_image_tags = {f"{project}-api:latest"}
    transport = sandbox._docker_cli

    async def cli(args, check=True, *, raw_output=False):
        if args[:1] == ["image"]:
            return await daemon.cli(args, check=check, raw_output=raw_output)
        return await transport(args, check=check, raw_output=raw_output)

    sandbox._docker_cli = cli
    fake.set(leftovers={"volumes": [f"{project}_data"]})
    await sandbox.stop(delete=True)
    downs = [a for a in fake.argvs() if a[:1] == ["compose"] and "down" in a]
    assert downs
    for argv in downs:
        assert "--volumes" not in argv and "-v" not in argv
        assert "--rmi" not in argv
    listed = [a for a in daemon.cli_calls if a[:2] == ["image", "ls"]]
    assert listed and f"label=com.docker.compose.project={project}" in listed[0]
    assert daemon.tags() == {
        "bf__remote-task:latest",
        "bf__remote-task:old",
        "operator/keep:1",
    }
    assert "sha256:ccc" not in daemon.images
    assert daemon.parent_present
    assert fake.script["leftovers"]["volumes"] == []


async def test_teardown_without_delete_keeps_the_built_images(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    fake.set(built_images=["sha256:aaa bf__remote-task:latest"])
    await sandbox.stop(delete=False)
    assert not any(a[:2] == ["image", "rm"] for a in fake.argvs())


def test_built_image_refs_keep_a_tag_someone_else_added():
    """RD shares U10's policy, including sole foreign tags."""
    from benchflow.sandbox._compose_teardown import built_image_refs
    from benchflow.sandbox.docker import DockerSandbox

    assert (
        RemoteDockerSandbox._remove_built_images is DockerSandbox._remove_built_images
    )
    images = [
        {"Id": "sha256:1", "RepoTags": ["bf__t:latest"], "RepoDigests": []},
        {
            "Id": "sha256:2",
            "RepoTags": ["p1-api:latest", "p1-api:v2"],
            "RepoDigests": [],
        },
        {"Id": "sha256:3", "RepoTags": ["operator/keep:1"], "RepoDigests": []},
        {"Id": "sha256:4", "RepoTags": [], "RepoDigests": []},
    ]
    assert built_image_refs(json.dumps(images), build_tags={"bf__t:latest"}) == [
        "sha256:1",
        "sha256:4",
    ]


# --- builds go to the remote daemon's own builder -----------------


async def test_an_ambient_buildx_builder_is_overridden(tmp_path, monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    monkeypatch.setenv("BUILDX_BUILDER", "local-builder")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=True)
    await sandbox.stop(delete=True)
    builds = [
        c for c in _compose_calls(fake) if _subcommand(c["argv"]) in ("build", "up")
    ]
    assert builds
    assert {c["BUILDX_BUILDER"] for c in fake.calls()} == {"default"}


# --- the login ssh's configuration picks is hidden ----------------


@pytest.mark.parametrize("login", ["review-login-from-config", "rvw"])
def test_a_login_from_the_ssh_configuration_is_hidden(
    tmp_path, monkeypatch, fake, login
):
    _fake_ssh(fake.root, user=login, hostname="real-host.internal.test")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://rd-alias")
    fake.set(
        fail={
            "info": "error during connect: command [ssh -o ConnectTimeout=30 -T -- "
            "rd-alias docker system dial-stdio] has exited with exit status 255: "
            f"stderr={login}@real-host.internal.test: Permission denied (publickey)."
        }
    )
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    message = str(info.value)
    assert "Permission denied (publickey)" in message
    assert f"{login}@" not in message
    assert "real-host.internal.test" not in message
    assert "rd-alias" not in message


# --- a changed host key always fails ------------------------------


def test_ssh_runs_with_strict_host_key_checking(tmp_path, monkeypatch, fake):
    import subprocess

    _fake_ssh(fake.root, user="u", hostname="h")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    env = resolve_remote_docker_host().client_env()
    shims = Path(env["PATH"].split(os.pathsep)[0])
    proc = subprocess.run(
        ["ssh", "-l", TOKEN, "--", "builder.example.test", "docker"],
        env=env,
        capture_output=True,
    )
    assert proc.returncode == 255
    (argv,) = [
        json.loads(line)
        for line in (fake.root / "ssh-calls.jsonl").read_text().splitlines()
    ]
    # Before every other option, so no ssh configuration file can undo it.
    # The provider's ssh configuration (-F) and the policy come first.
    assert argv[0] == "-F" and Path(argv[1]).is_file()
    assert argv[2:4] == ["-o", "StrictHostKeyChecking=yes"]
    assert argv[4:] == ["-l", TOKEN, "--", "builder.example.test", "docker"]
    assert (shims / "docker").is_file()
    assert oct((shims / "ssh").stat().st_mode & 0o777) == "0o700"
    assert oct(shims.stat().st_mode & 0o777) == "0o700"


def test_a_changed_host_key_is_named_by_ssh_s_verdict(monkeypatch, fake):
    # OpenSSH 9.6's text, as docker quotes it (on a build VM, 2026-10-16).
    banner = "@" * 59
    fake.set(
        fail={
            "info": "error during connect: command [ssh -o ConnectTimeout=30 -T -- "
            "builder.example.test docker system dial-stdio] has exited with exit "
            f"status 255: stderr={banner}\n"
            "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n"
            f"{banner}\nIT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!\n"
            "Host key for [builder.example.test]:2222 has changed and you have "
            "requested strict checking.\nHost key verification failed."
        }
    )
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    assert str(info.value) == (
        "Remote Docker host unreachable: remote-docker[ssh]: "
        "remote-docker[ssh]: Host key verification failed; "
        "verify the remote SSH host key and known_hosts"
    )


def test_tcp_gets_no_ssh_shim(tmp_path, monkeypatch, fake):
    _fake_ssh(fake.root, user="u", hostname="h")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "tcp://builder.example.test:2376")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", str(_certs(tmp_path / "certs")))
    env = resolve_remote_docker_host().client_env()
    shims = Path(env["PATH"].split(os.pathsep)[0])
    assert (shims / "docker").is_file()
    assert not (shims / "ssh").exists()


# --- the address appears only when asked for ----------------------


async def test_logs_and_errors_name_the_host_by_its_label(
    tmp_path, monkeypatch, fake, caplog
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://builder.example.test:2222")
    sandbox = _sandbox(tmp_path, cpus=16)
    with (
        caplog.at_level(logging.DEBUG, logger="benchflow"),
        pytest.raises(SandboxStartupError) as info,
    ):
        await sandbox.start(force_build=False)
    assert "remote-docker[ssh]" in str(info.value)
    assert "builder.example.test" not in str(info.value)
    sandbox = _sandbox(tmp_path / "second")
    with caplog.at_level(logging.DEBUG, logger="benchflow"):
        await sandbox.start(force_build=False)
        await sandbox.stop(delete=True)
    assert "remote-docker[ssh]" in caplog.text
    assert "builder.example.test" not in caplog.text


def test_a_config_refusal_names_the_variable_not_the_address(monkeypatch):
    from benchflow.sandbox.remote_docker import RemoteDockerConfigError

    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "tcp://internal-builder.test:2376")
    with pytest.raises(RemoteDockerConfigError) as info:
        resolve_remote_docker_host()
    assert REMOTE_DOCKER_HOST_ENV in str(info.value)
    assert "internal-builder.test" not in str(info.value)


def test_the_address_is_shown_only_when_asked(monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS", "1")
    fake.set(fail={"info": "ssh: connect to host builder.example.test port 2222"})
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    message = str(info.value)
    assert "ssh://***@builder.example.test:2222" in message
    assert "connect to host builder.example.test" in message
    assert TOKEN not in message
