"""A task's Compose files reach this machine only with the operator's opt-in.

The launch gate runs ``docker compose config`` on the task's files in the
environment the run gives Compose (PATH, HOME, the docker connection
variables, the named variables, the task's values: no operator credential)
and checks what Compose resolved: bind and volume sources, build contexts and
Dockerfiles, env, label, secret and config files, and the service settings
that hand a container this machine. The files Compose reads but
does not show (includes, extends, their ``.env``) are checked on the files.

Every test goes through the real gate (``Task`` and
``_validate_task_runtime_for_launch``) or the sandbox's own Compose
environment, so each fails on the code before the change by what it does.
The ``test_r2_*`` tests are the reproductions of a review's
findings; credentials in them are dummies.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner


def _compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        found = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return found.returncode == 0


pytestmark = pytest.mark.skipif(
    not _compose_available(),
    reason="the gate runs `docker compose config`, which needs docker compose",
)

ALLOW = "BENCHFLOW_ALLOW_TASK_ENV"
HOST_ACCESS = "BENCHFLOW_ALLOW_TASK_HOST_ACCESS"
SECRET = "s1-operator-value-never-in-a-task"
AWS = "s1-dummy-operator-aws-secret"


@pytest.fixture(autouse=True)
def _operator_env(monkeypatch, tmp_path):
    """The operator has S1_SECRET and an AWS key set, names nothing, has no
    .env and has not allowed host access."""
    monkeypatch.setenv(ALLOW, "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setenv("S1_SECRET", SECRET)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", AWS)
    monkeypatch.delenv("S1_UNSET", raising=False)
    monkeypatch.delenv(HOST_ACCESS, raising=False)


def _name(monkeypatch, *names: str) -> None:
    monkeypatch.setenv(ALLOW, ",".join(names))


def _write_task(task_dir: Path, toml_body: str = "") -> Path:
    """A split-layout task that passes ``bench tasks check``."""
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task_dir / "solution").mkdir(exist_ok=True)
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/sh\necho ok\n")
    (task_dir / "instruction.md").write_text("Write ok.\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 60\n\n'
        f"[verifier]\ntimeout_sec = 60\n\n{toml_body}"
    )
    return task_dir


def _compose_task(
    root: Path, compose: str, toml_body: str = "", files: dict[str, str] | None = None
) -> Path:
    task = _write_task(root, toml_body)
    environment = task / "environment"
    (environment / "docker-compose.yaml").write_text(compose)
    for name, text in (files or {}).items():
        path = environment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return task


def _gate(path: Path, sandbox_type: str = "docker") -> None:
    from benchflow.sandbox.setup import _validate_task_runtime_for_launch
    from benchflow.task import Task

    _validate_task_runtime_for_launch(
        Task(path), sandbox_type=sandbox_type, task_path=path
    )


def _refused(path: Path, sandbox_type: str = "docker") -> str:
    """The gate's refusal, whitespace folded."""
    with pytest.raises(ValueError) as raised:
        _gate(path, sandbox_type)
    return " ".join(str(raised.value).split())


def _sandbox(task: Path, name: str = "t__1"):
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task import Task

    config = Task(task).config.sandbox
    return DockerSandbox(task / "environment", "t", name, None, config)


def _resolved(task: Path) -> dict:
    """What `docker compose config` resolves for the task with the sandbox's
    own files and environment: what its run would get."""
    sandbox = _sandbox(task)
    # Its daemon flags (the connection, kept out of the environment), where
    # the sandbox has them.
    flags = getattr(sandbox, "_docker_compose_global_args", list)()
    command = [
        "docker",
        *flags,
        "compose",
        "--project-name",
        "s1-oracle-test",
        "--project-directory",
        str(sandbox.environment_dir.resolve()),
    ]
    for path in sandbox._docker_compose_paths:
        command += ["-f", str(path.resolve())]
    out = subprocess.run(
        [*command, "config", "--format", "json"],
        env=sandbox._docker_compose_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(out.stdout)


def _service_env(task: Path, service: str = "main") -> dict:
    return _resolved(task)["services"][service].get("environment") or {}


# Each reproduction of the second Codex review


def test_r2_1_a_dotenv_escape_reads_no_operator_value(tmp_path):
    """Compose decodes \\0044 to $ in a double-quoted env-file
    value and then fills ${AWS_SECRET_ACCESS_KEY}; Compose's environment has
    no operator credential to fill it with."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    env_file: leak.env\n",
        files={"leak.env": 'LEAK="\\0044{AWS_SECRET_ACCESS_KEY}"\n'},
    )
    _gate(task)
    assert "AWS_SECRET_ACCESS_KEY" not in _sandbox(task)._docker_compose_env()
    assert _service_env(task)["LEAK"] != AWS


@pytest.mark.parametrize(
    "compose",
    [
        'services:\n  main:\n    volumes: ["${S1_MOUNT:-/:/host:ro}"]\n',
        "services:\n  main:\n    volumes:\n"
        '      - type: "${S1_TYPE:-bind}"\n        source: /\n'
        "        target: /host\n        read_only: true\n",
        "services:\n  main:\n    build:\n      context: .\n"
        '      additional_contexts: {extra: "${S1_ABSENT:+https://}/"}\n',
        "services:\n  main:\n    volumes: [hostroot:/host]\n"
        "volumes:\n  hostroot:\n    driver_opts:\n"
        "      type: none\n      o: ${S1_O}\n      device: /\n",
    ],
    ids=["short-form", "long-form-type", "additional-context", "driver-opts"],
)
def test_r2_2_paths_are_checked_after_compose_fills_them(
    monkeypatch, tmp_path, compose
):
    """Compose fills ${VAR} before it reads a volume, its type,
    a context or a driver option; so does the check, as it reads Compose's
    resolved project."""
    task = _compose_task(tmp_path / "task", compose, files={".env": "S1_O=bind\n"})
    message = _refused(task)
    assert "outside the task's folder" in message
    assert f"{HOST_ACCESS}=1" in message
    # Daytona: the VM's root, refused before the sandbox.
    assert "outside the task's folder" in _refused(task, "daytona")
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


_PWN = (
    "services:\n  side:\n    image: alpine\n    privileged: true\n"
    "    environment: [AWS_SECRET_ACCESS_KEY]\n"
)


def test_r2_3_a_path_with_whitespace_is_the_file_compose_reads(tmp_path):
    """Compose keeps a path's trailing space, so it includes
    the file ``extra.yaml `` and not ``extra.yaml``."""
    task = _compose_task(
        tmp_path / "task",
        'include: ["extra.yaml "]\nservices:\n  main: {}\n',
        files={
            "extra.yaml": "services: {side: {image: alpine}}\n",
            "extra.yaml ": _PWN,
        },
    )
    assert "runs service side privileged" in _refused(task)
    assert "AWS_SECRET_ACCESS_KEY" not in _sandbox(task)._docker_compose_env()
    assert _service_env(task, "side").get("AWS_SECRET_ACCESS_KEY") is None


@pytest.mark.parametrize(
    "dotenv",
    ["NM=bridge\nNM=host\n", "A=host\nNM=${A}\n"],
    ids=["last-assignment", "dependent-value"],
)
def test_r2_4_dotenv_values_are_compose_s(tmp_path, dotenv):
    """Compose keeps a .env's last assignment and fills a
    value from earlier ones."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    image: alpine\n    network_mode: ${NM:-bridge}\n",
        files={".env": dotenv},
    )
    assert "gives service main network_mode: host" in _refused(task)


def test_r2_5_an_include_list_is_one_project(tmp_path):
    """An include's later files are overrides in the project of
    its first file, whose folder their relative paths resolve from."""
    task = _compose_task(
        tmp_path / "task",
        "include:\n  - path: [base.yaml, sub/override.yaml]\nservices:\n  main: {}\n",
        files={
            "base.yaml": "services: {side: {image: alpine}}\n",
            "sub/override.yaml": 'services:\n  side:\n    volumes: ["../../:/host:ro"]\n',
        },
    )
    message = _refused(task)
    assert f"mounts {tmp_path} for service side, outside the task's folder" in message


def test_r2_6_an_include_has_its_own_environment(tmp_path):
    """An include's env_file fills its own files, and sibling
    includes do not share their .env files."""
    explicit = _compose_task(
        tmp_path / "explicit",
        "include:\n  - path: sub/inc.yaml\n    env_file: inc.env\n"
        "services:\n  main: {}\n",
        files={
            "inc.env": "NM=host\n",
            "sub/inc.yaml": "services:\n  side:\n    image: alpine\n"
            "    network_mode: ${NM:-bridge}\n",
        },
    )
    assert "gives service side network_mode: host" in _refused(explicit)

    siblings = _compose_task(
        tmp_path / "siblings",
        "include: [one/a.yaml, two/b.yaml]\nservices:\n  main: {}\n",
        files={
            "one/.env": "NM=bridge\n",
            "one/a.yaml": "services:\n  a:\n    image: alpine\n"
            "    network_mode: ${NM:-bridge}\n",
            "two/.env": "NM=host\n",
            "two/b.yaml": "services:\n  b:\n    image: alpine\n"
            "    network_mode: ${NM:-bridge}\n",
        },
    )
    message = _refused(siblings)
    assert "gives service b network_mode: host" in message
    assert "service a network_mode" not in message


def test_r2_7_an_included_projects_dotenv_outside_the_task_is_refused(
    monkeypatch, tmp_path
):
    """Compose reads an included project's own .env, links
    followed; one outside the task is refused before Compose reads it."""
    operator = tmp_path / "s1-operator.env"
    operator.write_text("S1_FILE_TOKEN=s1-dummy-file-token\n")
    task = _compose_task(
        tmp_path / "task",
        "include: [sub/inc.yaml]\nservices:\n  main: {}\n",
        files={
            "sub/inc.yaml": "services:\n  side:\n    image: alpine\n"
            "    environment:\n      LEAK: ${S1_FILE_TOKEN:-}\n",
        },
    )
    (task / "environment" / "sub" / ".env").symlink_to(operator)
    message = _refused(task)
    # Refused when the task is copied, before Compose could read it.
    assert "environment/sub/.env is a link to" in message
    assert "outside the task's folder" in message
    assert "s1-dummy-file-token" not in message
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


def test_r2_8_an_outside_context_with_the_tasks_dockerfile_is_refused(tmp_path):
    """A build context outside the task with the task's own
    Dockerfile, or an inline one merged in through extends, copies this
    machine's files into the image."""
    review = tmp_path / "s1-review"
    (review / "private.txt").parent.mkdir(parents=True)
    (review / "private.txt").write_text("s1-dummy-operator-file\n")
    task = _compose_task(
        review / "task",
        "services:\n  main:\n    build:\n      context: ../..\n"
        "      dockerfile: task/environment/Dockerfile\n",
    )
    (task / "environment" / "Dockerfile").write_text(
        "FROM alpine\nCOPY private.txt /copied\n"
    )
    assert "with a Dockerfile of the task's" in _refused(task)

    merged = _compose_task(
        tmp_path / "s1-review-2" / "task",
        "services:\n  main:\n    extends: {file: base.yaml, service: base}\n"
        "    build:\n      dockerfile_inline: |\n"
        "        FROM alpine\n        COPY private.txt /copied\n",
        files={"base.yaml": "services:\n  base:\n    build: {context: ../..}\n"},
    )
    assert "with a Dockerfile of the task's" in _refused(merged)


def test_r2_9_a_task_never_steers_the_pass_credential_helper(monkeypatch, tmp_path):
    """docker-credential-pass hands PASSWORD_STORE_GPG_OPTS to
    gpg; a task's value for it is never in Compose's environment, and a
    Compose file reading it is told so."""
    task = _compose_task(
        tmp_path / "task",
        "x-keep: ${PASSWORD_STORE_GPG_OPTS}\nservices:\n  main: {}\n",
        '[environment.env]\nPASSWORD_STORE_GPG_OPTS = "--output /tmp/s1-pass-output"\n',
    )
    assert "sandbox.env sets PASSWORD_STORE_GPG_OPTS" in _refused(task)
    for host in ("", "1"):
        monkeypatch.setenv(HOST_ACCESS, host)
        assert "PASSWORD_STORE_GPG_OPTS" not in _sandbox(task)._docker_compose_env()


@pytest.mark.parametrize(
    ("env_file", "text"),
    [
        ("literal.env", "TEMPLATE='${HOME}'\n"),
        ("[{path: literal.env, format: raw}]", "TEMPLATE=${HOME}\n"),
    ],
    ids=["single-quoted", "raw-format"],
)
def test_r2_10_a_literal_env_template_is_not_refused(tmp_path, env_file, text):
    """Compose leaves a single-quoted or raw-format value as
    written, so the task gets the text ${HOME}."""
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    env_file: {env_file}\n",
        files={"literal.env": text},
    )
    _gate(task)
    # Compose prints a literal $ as $$.
    assert _service_env(task)["TEMPLATE"] in ("${HOME}", "$${HOME}")


# Compose's environment


def test_compose_gets_no_operator_credential_whatever_its_files_read(
    monkeypatch, tmp_path
):
    """Every way a task's files read a variable gets nothing of the
    operator's: they are reported (as a default), not refused."""
    monkeypatch.setenv("HTTPS_PROXY", "http://op:s1-fake-proxy@proxy.invalid:3128")
    task = _compose_task(
        tmp_path / "task",
        "include: [extra.yaml]\nservices:\n  main:\n"
        "    env_file: task.env\n"
        "    environment:\n"
        "      - DIRECT=${S1_SECRET}\n      - DEFAULTED=${S1_SECRET:-d}\n"
        "      - COPIED_ENV=${PROJ:-none}\n      - AWS_SECRET_ACCESS_KEY\n"
        "    extends: {file: base.yaml, service: b}\n",
        files={
            "task.env": 'A=${S1_SECRET}\nexport B="${HTTPS_PROXY}"\nC: ${S1_SECRET}\n'
            "S1_SECRET\nHTTPS_PROXY\n",
            ".env": "PROJ=${S1_SECRET}\n",
            "extra.yaml": "services:\n  side:\n    image: alpine\n"
            "    environment: {INC: '${S1_SECRET}'}\n",
            "base.yaml": "services:\n  b:\n    environment: {EXT: '${S1_SECRET}'}\n",
        },
    )
    _gate(task)
    env = _sandbox(task)._docker_compose_env()
    for name in ("S1_SECRET", "HTTPS_PROXY", "AWS_SECRET_ACCESS_KEY"):
        assert name not in env
    resolved = json.dumps(_resolved(task))
    assert SECRET not in resolved
    assert AWS not in resolved
    assert "s1-fake-proxy" not in resolved

    _name(monkeypatch, "S1_SECRET")
    assert _service_env(task)["DIRECT"] == SECRET


def test_compose_process_env_keeps_no_credential_and_no_connection(monkeypatch):
    """Compose's environment holds no operator credential and no connection
    setting: the daemon goes on the command line."""
    from benchflow.task.compose_env import (
        compose_process_env,
        docker_connection_args,
    )

    environ = {
        "PATH": "/usr/bin",
        "HOME": "/tmp/s1-home",
        "DOCKER_HOST": "unix:///tmp/s1-docker.sock",
        "DOCKER_CONFIG": "/tmp/s1-docker-config",
        "DOCKER_AUTH_CONFIG": "{}",
        "SSH_AUTH_SOCK": "/tmp/s1-agent.sock",
        "XDG_RUNTIME_DIR": "/tmp/s1-run",
        "COMPOSE_PROJECT_NAME": "s1",
        "COMPOSE_EXPERIMENTAL_GIT_REMOTE": "true",
        "DOCKER_PASSWORD": "s1-fake-docker",
        "AWS_SECRET_ACCESS_KEY": "s1-fake-aws",
        "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/s1-gcp.json",
        "PASSWORD_STORE_DIR": "/tmp/s1-pass",
        "S1_OTHER": "dropped",
    }
    env = compose_process_env(environ=environ, allowed=(), host_access=False)
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/tmp/s1-home",
        "COMPOSE_EXPERIMENTAL_GIT_REMOTE": "false",
        "COMPOSE_EXPERIMENTAL_OCI_REMOTE": "false",
    }
    named = compose_process_env(
        environ=environ,
        allowed=("AWS_SECRET_ACCESS_KEY", "DOCKER_HOST"),
        host_access=True,
    )
    assert named["AWS_SECRET_ACCESS_KEY"] == "s1-fake-aws"
    assert "DOCKER_HOST" not in named  # a connection setting, named or not
    assert "COMPOSE_EXPERIMENTAL_GIT_REMOTE" not in named
    assert docker_connection_args(environ) == [
        "--config",
        "/tmp/s1-docker-config",
        "-H",
        "unix:///tmp/s1-docker.sock",
    ]
    assert docker_connection_args(
        {
            "DOCKER_HOST": "tcp://192.0.2.1:2376",
            "DOCKER_TLS_VERIFY": "1",
            "DOCKER_CERT_PATH": "/tmp/s1-certs",
            "DOCKER_CONTEXT": "ignored",
        }
    ) == [
        "-H",
        "tcp://192.0.2.1:2376",
        "--tlsverify",
        "--tlscacert",
        "/tmp/s1-certs/ca.pem",
        "--tlscert",
        "/tmp/s1-certs/cert.pem",
        "--tlskey",
        "/tmp/s1-certs/key.pem",
    ]
    assert docker_connection_args({"DOCKER_CONTEXT": "s1"}) == ["--context", "s1"]


@pytest.mark.parametrize(
    "name",
    [
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "CLOUDSDK_PYTHON",
        "AWS_PROFILE",
        "GIT_SSH_COMMAND",
        "PASSWORD_STORE_GPG_OPTS",
        "PASSWORD_STORE_DIR",
        "GPG_AGENT_INFO",
        "GNUPGHOME",
        "DOCKER_CONFIG",
        "DOCKER_HOST",
        "COMPOSE_EXPERIMENTAL_OCI_REMOTE",
        "SSH_AUTH_SOCK",
        "HTTPS_PROXY",
        "PATH",
        "HOME",
        "GODEBUG",
    ],
)
def test_a_task_value_that_steers_a_host_program_is_not_passed(monkeypatch, name):
    from benchflow.task.compose_env import compose_process_env

    environ = {"PATH": "/usr/bin", "HOME": "/tmp/s1-home"}
    task = {name: "s1-task-value", "TASK_KEY": "v"}
    env = compose_process_env(task, environ=environ, allowed=(), host_access=True)
    assert env.get(name) != "s1-task-value"
    assert env["TASK_KEY"] == "v"
    named = compose_process_env(
        task, environ=environ, allowed=(name,), host_access=True
    )
    if name in ("DOCKER_CONFIG", "DOCKER_HOST"):
        assert name not in named  # a connection setting is never in it
    else:
        assert named[name] == "s1-task-value"


def test_the_sandbox_gives_compose_the_tasks_values_and_benchflows_last(
    monkeypatch, tmp_path
):
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    environment:\n      T: ${TASK_KEY}\n",
        '[environment.env]\nTASK_KEY = "v"\nS1_UNREAD = "u"\n'
        'LD_PRELOAD = "/tmp/s1.so"\nDOCKER_HOST = "tcp://s1-task.invalid:2375"\n',
    )
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/s1-operator.sock")
    sandbox = _sandbox(task)
    env = sandbox._docker_compose_env()
    assert env["TASK_KEY"] == "v"
    assert env["S1_UNREAD"] == "u"
    assert "LD_PRELOAD" not in env
    # The daemon is on the command line, the operator's, never the task's,
    # after the client configuration without proxies.
    assert "DOCKER_HOST" not in env
    flags = sandbox._docker_compose_global_args()
    assert flags[0] == "--config"
    assert flags[2:] == ["-H", "unix:///tmp/s1-operator.sock"]
    assert env["MAIN_IMAGE_NAME"]
    assert env["BENCHFLOW_PROCESS"]


def test_a_compose_file_never_reads_the_docker_connection(monkeypatch, tmp_path):
    """${DOCKER_HOST} (or the context, the TLS files, the docker
    config) is not in the environment Compose fills a task's files from."""
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    monkeypatch.setenv("DOCKER_CONFIG", str(Path.home() / ".docker"))
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    environment:\n"
        "      H: ${DOCKER_HOST:-none}\n      C: ${DOCKER_CONFIG:-none}\n",
    )
    # CONF-8: even an explicit Unix endpoint does not prove that the
    # daemon shares the client filesystem, so its log binds now refuse.
    assert "local daemon" in _refused(task)
    assert _service_env(task) == {"H": "none", "C": "none"}


def test_remote_docker_keeps_its_connection_out_of_compose_s_environment(
    monkeypatch, tmp_path
):
    """The remote URL and its TLS files go on the command line;
    over ssh:// Compose needs the operator's agent, which no Compose file may
    read and no build may use, and which the pre-compose hook never gets."""
    from benchflow.sandbox.remote_docker import RemoteDockerHost

    ssh = RemoteDockerHost("ssh://s1@192.0.2.1", user="s1")
    assert ssh.compose_args({}) == ["-H", "ssh://s1@192.0.2.1"]
    transport = ssh.compose_transport_env({"SSH_AUTH_SOCK": "/tmp/s1-agent.sock"})
    assert set(transport) == {"PATH", "BUILDX_BUILDER"}
    assert transport["BUILDX_BUILDER"] == "default"
    wrapper = Path(transport["PATH"].split(":")[0]) / "ssh"
    assert "SSH_AUTH_SOCK=/tmp/s1-agent.sock; export SSH_AUTH_SOCK" in (
        wrapper.read_text()
    )
    tcp = RemoteDockerHost("tcp://192.0.2.1:2376", cert_path="/tmp/s1-certs")
    assert tcp.compose_args({"DOCKER_CONFIG": "/tmp/s1-cfg"}) == [
        "--config",
        "/tmp/s1-cfg",
        "-H",
        "tcp://192.0.2.1:2376",
        "--tlsverify",
        "--tlscacert",
        "/tmp/s1-certs/ca.pem",
        "--tlscert",
        "/tmp/s1-certs/cert.pem",
        "--tlskey",
        "/tmp/s1-certs/key.pem",
    ]
    assert tcp.compose_transport_env({"SSH_AUTH_SOCK": "/tmp/s1-agent.sock"}) == {
        "BUILDX_BUILDER": "default"
    }

    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://s1@192.0.2.1")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/s1-agent.sock")
    agent = _compose_task(
        tmp_path / "agent",
        "services:\n  main:\n    build:\n      context: .\n      ssh: [default]\n",
    )
    # Integration confines the agent to the provider's ssh wrapper. Compose
    # can therefore admit a build asking for an agent without receiving ours.
    _gate(agent, "remote-docker")
    _gate(agent)  # local docker: Compose has no agent
    reads = _compose_task(
        tmp_path / "reads",
        "services:\n  main:\n    environment:\n      A: ${SSH_AUTH_SOCK:-none}\n"
        "      H: ${DOCKER_HOST:-none}\n",
    )
    _gate(reads, "remote-docker")
    assert _service_env(reads) == {"A": "none", "H": "none"}
    _gate(reads)

    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    sandbox = RemoteDockerSandbox(
        reads / "environment",
        "t",
        "t__1",
        None,
        _sandbox(reads).task_env_config,
        docker_host=ssh,
    )
    assert "SSH_AUTH_SOCK" not in sandbox._docker_compose_env()  # the hook's
    transport = sandbox._compose_transport_env()
    assert set(transport) == {"PATH", "BUILDX_BUILDER"}
    wrapper = Path(transport["PATH"].split(":")[0]) / "ssh"
    assert "SSH_AUTH_SOCK=/tmp/s1-agent.sock; export SSH_AUTH_SOCK" in (
        wrapper.read_text()
    )
    flags = sandbox._docker_compose_global_args()
    assert flags[0] == "--config" and flags[2:] == ["-H", "ssh://s1@192.0.2.1"]
    _name(monkeypatch, "SSH_AUTH_SOCK")
    _gate(agent, "remote-docker")
    _gate(reads, "remote-docker")


@pytest.mark.parametrize(
    "compose",
    [
        "services:\n  main:\n    volumes: ['./data:/data']\n",
        "services:\n  main:\n    volumes:\n"
        "      - {type: bind, source: ./data, target: /data}\n",
        "services:\n  main:\n    volumes: [data:/d]\nvolumes:\n  data:\n"
        "    driver: local\n"
        "    driver_opts: {type: none, o: bind, device: ./data}\n",
        "services:\n  main:\n    secrets: [s]\nsecrets:\n  s: {file: ./data/s.txt}\n",
        "services:\n  main:\n    configs: [c]\nconfigs:\n  c: {file: ./data/c.txt}\n",
    ],
    ids=["short-bind", "long-bind", "bind-volume", "secret-file", "config-file"],
)
def test_remote_docker_refuses_every_host_bind(monkeypatch, tmp_path, compose):
    """On remote-docker a bind source (and a file secret or
    config, which Compose bind-mounts) is a path on the remote host, not the
    task's file, wherever it points; a named volume is fine."""
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://s1@192.0.2.1")
    task = _compose_task(
        tmp_path / "task",
        compose,
        files={"data/s.txt": "s\n", "data/c.txt": "c\n"},
    )
    _gate(task)  # local docker: the task's own folder
    message = _refused(task, "remote-docker")
    assert "a path on the remote Docker host" in message
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task, "remote-docker")

    volume = _compose_task(
        tmp_path / "volume",
        "services:\n  main:\n    volumes: [cache:/cache]\nvolumes:\n  cache: {}\n",
    )
    monkeypatch.delenv(HOST_ACCESS)
    _gate(volume, "remote-docker")


def test_a_path_the_operator_named_may_be_mounted(monkeypatch, tmp_path):
    """robohub mounts the worker-endpoints file the operator names through
    ${ROBOUSE_GPU_REMOTE}: that path (or below it) passes once named, and a
    path built to climb out of it does not."""
    endpoints = tmp_path / "s1-operator" / "remote.json"
    endpoints.parent.mkdir()
    endpoints.write_text("{}\n")
    monkeypatch.setenv("S1_GPU_REMOTE", str(endpoints))
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes:\n"
        "      - ${S1_GPU_REMOTE:?set it}:/run/remote.json:ro\n",
    )
    assert "does not resolve with docker compose config" in _refused(task)
    _name(monkeypatch, "S1_GPU_REMOTE")
    _gate(task)
    climbing = _compose_task(
        tmp_path / "climbing",
        "services:\n  main:\n    volumes:\n      - ${S1_GPU_REMOTE}/../..:/up:ro\n",
    )
    assert "outside the task's folder" in _refused(climbing)


# Remote includes: refused before Compose could fetch them


@pytest.mark.parametrize(
    "compose",
    [
        "include:\n  - https://github.com/s1-attacker/x.git#main:c.yaml\n"
        "services:\n  main: {}\n",
        "include:\n  - path: oci://registry.example.invalid/s1/inc:latest\n"
        "services:\n  main: {}\n",
        "include: [git@github.com:s1-attacker/x.git]\nservices:\n  main: {}\n",
        "services:\n  main:\n    extends:\n"
        "      file: git@github.com:s1-attacker/x.git\n      service: b\n",
        "include: [inc.yaml]\nservices:\n  main: {}\n",
    ],
    ids=["git-include", "oci-include", "ssh-include", "git-extends", "nested"],
)
def test_a_remote_include_is_refused_before_compose_runs(
    monkeypatch, tmp_path, compose
):
    from benchflow.task import compose_env

    task = _compose_task(
        tmp_path / "task",
        compose,
        files={"inc.yaml": "include: [https://github.com/s1-attacker/y.git]\n"},
    )

    def no_compose(*args, **kwargs):
        raise AssertionError("docker compose ran on a remote include")

    monkeypatch.setattr(compose_env.subprocess, "Popen", no_compose)
    monkeypatch.setattr(compose_env.subprocess, "run", no_compose)
    message = _refused(task)
    assert "a remote file BenchFlow cannot check" in message
    assert f"{HOST_ACCESS}=1" in message


@pytest.mark.parametrize(
    ("include", "disabled"),
    [
        ("https://github.com/s1-attacker/x.git", "COMPOSE_EXPERIMENTAL_GIT_REMOTE"),
        ("git@github.com:s1-attacker/x.git", "COMPOSE_EXPERIMENTAL_GIT_REMOTE"),
        (
            "oci://registry.example.invalid/s1/inc:latest",
            "COMPOSE_EXPERIMENTAL_OCI_REMOTE",
        ),
    ],
)
def test_compose_fetches_no_remote_include_in_its_environment(
    tmp_path, include, disabled
):
    """Compose 2.40.3 and v5.6.0 fetch git and OCI includes unless these
    switches are false, and a task's .env cannot turn them back on: pinned
    on the Compose that runs here."""
    from benchflow.task.compose_env import compose_process_env

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(
        f'include: ["{include}"]\nservices:\n  main:\n    image: alpine\n'
    )
    (environment / ".env").write_text(
        "COMPOSE_EXPERIMENTAL_GIT_REMOTE=true\nCOMPOSE_EXPERIMENTAL_OCI_REMOTE=true\n"
    )
    found = subprocess.run(
        ["docker", "compose", "--project-directory", str(environment), "config"],
        env=compose_process_env(allowed=(), host_access=False),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert found.returncode != 0
    assert f'disabled by "{disabled}"' in found.stderr


# Host paths in the resolved project


@pytest.mark.parametrize(
    ("compose", "toml_body"),
    [
        ("services:\n  main:\n    volumes: [/var/run/docker.sock:/s]\n", ""),
        ("services:\n  main:\n    volumes: ['{outside}:/h:ro']\n", ""),
        ("services:\n  main:\n    volumes: ['../../..:/up']\n", ""),
        (
            "services:\n  main:\n    volumes: ['${{S1_DIR}}:/d']\n",
            '[environment.env]\nS1_DIR = "{outside}"\n',
        ),
        ("services:\n  main:\n    volumes: ['${{S1_UNSET:-{outside}}}:/d']\n", ""),
        ("services:\n  main:\n    env_file: {outside}/keys.env\n", ""),
        ("services:\n  main:\n    label_file: {outside}/keys.labels\n", ""),
        ("include: ['{outside}/compose.yaml']\nservices:\n  main: {{}}\n", ""),
        (
            "services:\n  main:\n    extends: {{file: '{outside}/compose.yaml', "
            "service: side}}\n",
            "",
        ),
        (
            "services:\n  main:\n    build:\n      context: {outside}\n"
            "      dockerfile_inline: |\n        FROM alpine\n        COPY . /leak\n",
            "",
        ),
        (
            "services:\n  main:\n    build:\n      context: .\n"
            "      dockerfile: {outside}/Dockerfile\n",
            "",
        ),
        (
            "services:\n  main:\n    build:\n      context: .\n"
            "      additional_contexts:\n        up: {outside}\n",
            "",
        ),
        (
            "services:\n  main:\n    build:\n      context: .\n"
            "      ssh: ['k={outside}/id']\n",
            "",
        ),
        (
            "services:\n  main:\n    volumes: [data:/d]\n"
            "volumes:\n  data:\n    driver_opts:\n"
            "      type: none\n      o: bind\n      device: {outside}\n",
            "",
        ),
        (
            "services:\n  main:\n    secrets: [t]\n"
            "secrets:\n  t:\n    file: {outside}/keys.env\n",
            "",
        ),
        (
            "services:\n  main:\n    configs: [c]\n"
            "configs:\n  c:\n    file: {outside}/keys.env\n",
            "",
        ),
    ],
    ids=[
        "docker-socket",
        "absolute",
        "relative-escape",
        "task-env-value",
        "default-value",
        "env_file",
        "label_file",
        "include",
        "extends",
        "foreign-context-inline-dockerfile",
        "outside-dockerfile",
        "additional-context",
        "ssh-key",
        "bind-volume",
        "secret",
        "config",
    ],
)
def test_a_host_path_outside_the_task_is_refused(
    monkeypatch, tmp_path, compose, toml_body
):
    outside = tmp_path / "victim-a"
    outside.mkdir()
    for name in ("keys.env", "keys.labels", "id"):
        (outside / name).write_text("K=v\n")
    (outside / "Dockerfile").write_text("FROM alpine\n")
    (outside / "compose.yaml").write_text("services:\n  side:\n    image: alpine\n")
    task = _compose_task(
        tmp_path / "task",
        compose.format(outside=outside),
        toml_body.format(outside=outside),
    )
    message = _refused(task)
    # The task's folder, or for a file Compose reads before it resolves
    # anything, the task's environment folder (the copy holds only it).
    assert "outside the task's" in message
    assert "reaches outside the task" in message
    assert "--allow-task-env" not in message
    # Daytona: a path outside the uploaded folder is one of the sandbox's VM,
    # refused before the sandbox is created.
    assert "outside the task's" in _refused(task, "daytona")
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


def test_a_home_path_is_a_host_path(tmp_path):
    """Compose fills ~ from HOME, the operator's (left as it is: the docker
    CLI finds a per-user Compose plugin through it)."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes:\n      - ~/.s1-oracle-creds:/creds:ro\n",
    )
    home = Path.home()
    assert f"mounts {home}/.s1-oracle-creds for service main" in _refused(task)


def test_a_symlink_out_of_the_task_is_refused(tmp_path):
    outside = tmp_path / "victim-a"
    outside.mkdir()
    (outside / "keys.env").write_text("K=v\n")
    task = _compose_task(
        tmp_path / "task", "services:\n  main:\n    volumes:\n      - ./link:/l\n"
    )
    (task / "environment" / "link").symlink_to(outside)
    # Refused when the task is copied: the copy never follows it.
    assert "environment/link is a link to" in _refused(task)

    other = _write_task(tmp_path / "other")
    (other / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main: {}\n"
    )
    (other / "environment" / ".env").symlink_to(outside / "keys.env")
    assert "environment/.env is a link to" in _refused(other)


def test_paths_inside_the_task_and_benchflows_own_pass(tmp_path):
    runtime = tmp_path / "runtime-s1"
    runtime.mkdir()
    (runtime / "Dockerfile").write_text("FROM alpine\n")
    task = _compose_task(
        tmp_path / "task",
        "include: [sub/inc.yaml]\n"
        "services:\n"
        "  main:\n"
        "    build:\n      context: ${CONTEXT_DIR}\n"
        "    env_file: [main.env, {path: absent.env, required: false}]\n"
        "    label_file: main.labels\n"
        "    volumes:\n"
        "      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}\n"
        "      - ${HOST_ARTIFACTS_PATH}:/artifacts-alias\n"
        "      - ./data:/data\n"
        "      - ../tests:/task-tests:ro\n"
        "      - cache:/cache\n"
        "  sim:\n"
        f"    build:\n      context: {runtime}\n"
        "volumes:\n  cache: {}\n",
        files={
            "main.env": "MODE=fast\n",
            "main.labels": "team=s1\n",
            "sub/inc.yaml": "services:\n  side:\n    image: alpine\n",
            "data/marker": "dummy-data\n",
            "sub/.env": "MODE=slow\n",
        },
    )
    _gate(task)


@pytest.mark.parametrize(
    "volume",
    [
        "${CONTEXT_DIR}/../../../../../../../../:/host:ro",
        "${CONTEXT_DIR}/../..:/host:ro",
        "${HOST_AGENT_LOGS_PATH}/../../../..:/host:ro",
        "${HOST_ARTIFACTS_PATH}/..:/host:ro",
        "${ENV_AGENT_LOGS_PATH}:/host:ro",
    ],
    ids=["context-root", "context-parent", "logs-up", "artifacts-up", "env-path"],
)
def test_a_path_climbing_out_of_benchflows_folders_is_refused(tmp_path, volume):
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    image: alpine\n    volumes:\n      - {volume}\n",
    )
    assert "outside the task's folder" in _refused(task)


def test_include_and_extends_files_resolve_where_compose_resolves_them(tmp_path):
    """An include's project_directory is where its files' paths resolve from;
    an extended file's paths resolve from its own folder."""
    task = _compose_task(
        tmp_path / "task",
        "include:\n  - path: sub/inc.yaml\n    project_directory: ..\n"
        "services:\n  main:\n    image: alpine\n",
        files={
            "sub/inc.yaml": "services:\n  side:\n    image: alpine\n"
            "    volumes:\n      - ../:/host:ro\n",
        },
    )
    # The project folder .. is the task's folder, outside the environment
    # folder the copy holds: refused before Compose reads its .env.
    assert "includes from the project folder" in _refused(task)

    variable = _compose_task(
        tmp_path / "variable",
        "include:\n  - path: inc.yaml\n    project_directory: ${S1_UNSET:-.}\n"
        "services:\n  main:\n    image: alpine\n",
        files={"inc.yaml": "services:\n  side:\n    image: alpine\n"},
    )
    assert "a file named through a variable" in _refused(variable)

    extended = _compose_task(
        tmp_path / "extended",
        "services:\n  main:\n    extends:\n      file: sub/base.yaml\n"
        "      service: base\n",
        files={
            "sub/base.yaml": "services:\n  base:\n    image: alpine\n"
            "    volumes:\n      - ../../:/task:ro\n",
        },
    )
    _gate(extended)


# Service settings that hand a container this machine


@pytest.mark.parametrize(
    ("service", "shown"),
    [
        ("privileged: true", "runs service main privileged"),
        ('pid: "host"', "gives service main pid: host"),
        ("network_mode: host", "gives service main network_mode: host"),
        ("ipc: host", "gives service main ipc: host"),
        ("userns_mode: host", "gives service main userns_mode: host"),
        ('pid: "container:s1-other"', "gives service main pid: container:s1-other"),
        ("cap_add: [SYS_ADMIN]", "gives service main the capability SYS_ADMIN"),
        ("cap_add: [ALL]", "gives service main the capability ALL"),
        (
            "cap_add: [CAP_DAC_READ_SEARCH]",
            "gives service main the capability CAP_DAC_READ_SEARCH",
        ),
        (
            'security_opt: ["seccomp:unconfined"]',
            "runs service main with security_opt seccomp:unconfined",
        ),
        (
            'security_opt: ["apparmor=unconfined"]',
            "runs service main with security_opt apparmor=unconfined",
        ),
        (
            'devices: ["/dev/sda:/dev/sda"]',
            "gives service main the host device /dev/sda",
        ),
        (
            'volumes_from: ["container:s1-other"]',
            "gives service main the volumes of container:s1-other",
        ),
        (
            "build:\n      context: .\n      network: host",
            "builds service main on the host network",
        ),
        (
            "build:\n      context: .\n      entitlements: [security.insecure]",
            "builds service main with the entitlement security.insecure",
        ),
        (
            "use_api_socket: true",
            "gives service main the Docker API socket and your registry logins",
        ),
        (
            'provider:\n      type: ./hook.sh\n      options: {x: "1"}',
            "runs the program ./hook.sh on this machine for service main (provider)",
        ),
        ("network_mode: ${S1_NET:-host}", "gives service main network_mode: host"),
        ("cap_add: [$S1_CAP]", "gives service main the capability SYS_ADMIN"),
    ],
    ids=[
        "privileged",
        "pid-host",
        "network-host",
        "ipc-host",
        "userns-host",
        "pid-container",
        "cap-sys-admin",
        "cap-all",
        "cap-dac-read-search",
        "seccomp-unconfined",
        "apparmor-unconfined",
        "devices",
        "volumes-from-container",
        "build-network-host",
        "build-entitlement",
        "use-api-socket",
        "provider",
        "default-value",
        "task-value",
    ],
)
def test_a_service_setting_that_reaches_this_machine_is_refused(
    monkeypatch, tmp_path, service, shown
):
    """Root in a privileged container, or one with the host's PID namespace,
    reads the bench process's environment and mounts the host's disk; the
    host network reaches this machine's local services."""
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    image: alpine\n    {service}\n",
        '[environment.env]\nS1_CAP = "SYS_ADMIN"\n',
    )
    message = _refused(task)
    assert f"{shown}, which reaches this machine" in message
    assert "--allow-task-env" not in message
    # Daytona: the sandbox's VM, refused before the sandbox.
    assert shown in _refused(task, "daytona")
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


def test_a_host_network_and_an_included_privileged_service_are_refused(tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "include:\n  - extra.yaml\n"
        "services:\n  main:\n    networks: [outside]\n"
        "networks:\n  outside:\n    name: host\n    external: true\n",
        files={
            "extra.yaml": "services:\n  side:\n    image: alpine\n    privileged: true\n"
        },
    )
    message = _refused(task)
    assert "puts network outside on the host network" in message
    assert "runs service side privileged" in message


def test_namespaced_service_settings_pass(tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "services:\n"
        "  main:\n"
        "    image: alpine\n"
        "    privileged: false\n"
        "    cap_add: [NET_ADMIN, SYS_PTRACE, cap_net_raw]\n"
        '    security_opt: ["no-new-privileges:true"]\n'
        "    network_mode: ${S1_NET:-bridge}\n"
        "    volumes_from: [side]\n"
        "  side:\n"
        "    image: alpine\n",
    )
    _gate(task)


def test_a_compose_file_of_several_documents_is_one_project(monkeypatch, tmp_path):
    """Compose merges a file's --- documents; so does what it resolves."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    image: busybox\n---\n"
        "services:\n  main:\n    privileged: true\n",
    )
    assert "runs service main privileged" in _refused(task)
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


@pytest.mark.parametrize(
    ("compose", "name", "body"),
    [
        ("include: [yes]\nservices:\n  main: {}\n", "yes", _PWN),
        ("include:\n  - path: On\nservices:\n  main: {}\n", "On", _PWN),
    ],
    ids=["include-yes", "include-path-On"],
)
def test_a_path_written_as_yes_or_on_is_the_file_compose_reads(
    tmp_path, compose, name, body
):
    """Compose (YAML 1.2) reads an unquoted yes or On as text (S1-R3-B-1)."""
    task = _compose_task(tmp_path / "task", compose, files={name: body})
    message = _refused(task)
    assert "runs service side privileged" in message
    assert AWS not in message


@pytest.mark.parametrize(
    "compose",
    [
        "include: [1]\nservices:\n  main: {}\n",
        "include:\n  - path: [true]\nservices:\n  main: {}\n",
        "include: [2026-10-16]\nservices:\n  main: {}\n",
        "services:\n  main:\n    extends: {file: 1.5, service: x}\n",
        "services:\n  main:\n    extends: !!omap [{file: /etc/hostname}]\n",
    ],
)
def test_an_include_or_extends_path_that_is_not_text_is_refused(tmp_path, compose):
    message = _refused(_compose_task(tmp_path / "t", compose))
    assert "is not text to Compose" in message or "uses the YAML tag omap" in message


@pytest.mark.parametrize(
    "compose",
    [
        "services:\n  main: [\n",
        "- services\n",
        "include: [missing.yaml]\nservices:\n  main: {}\n",
        "services:\n  main:\n    environment: {K: '${S1_UNSET:?the task needs it}'}\n",
    ],
    ids=["broken", "a-list", "missing-include", "required"],
)
def test_a_project_compose_cannot_resolve_is_refused(monkeypatch, tmp_path, compose):
    task = _compose_task(tmp_path / "task", compose)
    message = _refused(task)
    assert (
        "does not resolve with docker compose config" in message
        or "is not a Compose file BenchFlow could read" in message
    )
    monkeypatch.setenv(HOST_ACCESS, "1")
    assert "does not resolve with docker compose config" in _refused(task)


# Reporting, the job gate and the CLI


def test_a_read_compose_cannot_fill_warns_naming_it(monkeypatch, tmp_path, caplog):
    """A variable the files read that Compose does not have gets its default
    or nothing, never the operator's value: the job gate and bench tasks
    check warn, naming it, never its value."""
    from benchflow.cli.main import app
    from benchflow.evaluation import _refuse_unnamed_task_env

    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    environment:\n"
        "      C: ${S1_COMPOSE:-none}\n      K: ${S1_SECRET}\n      U: ${S1_UNSET:-y}\n",
    )
    monkeypatch.setenv("S1_COMPOSE", "s1-compose-value")
    caplog.set_level(logging.WARNING, logger="benchflow.evaluation")
    _refuse_unnamed_task_env([task], manifest=None, sandbox="docker")
    logged = " ".join(caplog.text.split())
    assert "S1_COMPOSE, S1_SECRET are set in your environment or .env" in logged
    assert "--allow-task-env S1_COMPOSE --allow-task-env S1_SECRET" in logged
    assert "S1_UNSET" not in logged
    assert SECRET not in caplog.text
    assert "s1-compose-value" not in caplog.text

    checked = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert checked.exit_code == 0, checked.output
    assert "warning: S1_COMPOSE, S1_SECRET are set" in " ".join(checked.output.split())

    caplog.clear()
    _name(monkeypatch, "S1_SECRET", "S1_COMPOSE")
    _refuse_unnamed_task_env([task], manifest=None, sandbox="docker")
    assert "are set in your environment" not in caplog.text


def test_a_default_the_tasks_own_dotenv_fills_does_not_warn(monkeypatch, tmp_path):
    from benchflow.task.compose_env import check_compose

    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    environment:\n      C: ${S1_COMPOSE:-x}\n",
        files={".env": "S1_COMPOSE=from-the-task\n"},
    )
    monkeypatch.setenv("S1_COMPOSE", "s1-compose-value")
    assert check_compose(task / "environment").defaulted == []


def test_tasks_check_names_the_host_access_opt_in(monkeypatch, tmp_path):
    from benchflow.cli.main import app

    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes:\n      - /tmp/victim-a:/h\n",
    )
    refused = CliRunner().invoke(app, ["tasks", "check", str(task)])
    flat = " ".join(refused.output.split())
    assert refused.exit_code == 1
    assert "mounts /tmp/victim-a for service main, outside the task's folder" in flat
    assert f"{HOST_ACCESS}=1" in flat
    monkeypatch.setenv(HOST_ACCESS, "1")
    allowed = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert allowed.exit_code == 0, allowed.output


def test_a_task_cannot_set_benchflows_compose_variables(monkeypatch, tmp_path):
    """docker-compose-base.yaml bind-mounts ${HOST_ARTIFACTS_PATH} and the
    log paths read-write; a task's literal value must not choose them."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    environment:\n      S1_MARK: '1'\n",
        '[environment.env]\nHOST_ARTIFACTS_PATH = "/tmp/victim-a"\n',
    )
    message = _refused(task)
    assert (
        "sets HOST_ARTIFACTS_PATH, which BenchFlow sets for Compose itself" in message
    )
    monkeypatch.setenv(HOST_ACCESS, "1")
    assert "BenchFlow sets its own Compose" in _refused(task)
    sandbox = _sandbox(task)
    env = sandbox._docker_compose_env()
    assert env["HOST_ARTIFACTS_PATH"] == sandbox._env_vars.host_artifacts_path


# The YAML loader the include and extends walk reads files with


def test_plain_compose_yaml_loads_as_compose_reads_it():
    """go-yaml v3's plain scalars (what `docker compose config` prints for
    them, checked against Compose v5.6.0 and 2.40.3), not YAML 1.1's. A date
    stays text here (Compose prints a time), but a path written as one is
    refused (test_an_include_or_extends_path_that_is_not_text_is_refused)."""
    from benchflow.task.compose_env import _load_all

    cases = {
        "yes": "yes", "Yes": "Yes", "NO": "NO", "on": "on", "Off": "Off",
        "y": "y", "n": "n", "1:30": "1:30", "190:20:30.15": "190:20:30.15",
        "=": "=", "2026-10-16": "2026-10-16",
        "true": True, "False": False, "TRUE": True, "tRue": "tRue",
        "~": None, "null": None, "NULL": None,
        "12": 12, "-3": -3, "+4": 4, "1_000": 1000, "0x1F": 31, "0o17": 15,
        "0777": 511, "0b101": 5, "089": 89.0, "1e3": 1000.0, "1.5": 1.5,
        ".5": 0.5, "1.": 1.0,
    }  # fmt: skip
    text = "".join(f"k{i}: {raw}\n" for i, raw in enumerate(cases))
    assert _load_all(text) == [{f"k{i}": want for i, want in enumerate(cases.values())}]
    loaded = _load_all("a: .inf\nb: -.Inf\nc: .NaN\n")
    assert loaded is not None
    assert loaded[0]["a"] == float("inf") and loaded[0]["b"] == float("-inf")
    assert loaded[0]["c"] != loaded[0]["c"]


def test_env_files_compose_does_not_show_fail_closed(tmp_path):
    """Compose 2.40.3 reads env files into the project whatever
    --no-env-resolution says; v5.6.0 keeps their paths. Their paths are now
    checked before Compose runs, on the task's copy, so the
    resolved project needs to show none, on either version, and a variable
    in an enumerated field no longer leaves them unchecked."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    env_file: main.env\n    volumes:\n"
        '      - type: "${S1_TYPE:-volume}"\n        source: cache\n'
        "        target: /cache\nvolumes:\n  cache: {}\n",
        files={"main.env": "MODE=fast\n"},
    )
    _gate(task)
    outside = tmp_path / "s1-operator.env"
    outside.write_text("MODE=s1-dummy\n")
    (task / "environment" / "docker-compose.yaml").write_text(
        f"services:\n  main:\n    env_file: {outside}\n"
    )
    assert f"reads the env file {outside}" in _refused(task)


# Daytona DinD: Compose resolves in the sandbox's VM


def _dind(tmp_path: Path, task: Path):
    """A DinD strategy whose VM is a folder here: /benchflow is mapped onto
    it, with BenchFlow's Compose files and the uploaded environment folder
    (links skipped, as Daytona's upload does), and its commands run the real
    docker compose."""
    import os
    from types import SimpleNamespace

    from benchflow.sandbox._base import ExecResult
    from benchflow.sandbox._compose import compose_definition_path
    from benchflow.sandbox.daytona import _DaytonaDinD
    from benchflow.task import Task

    vm = (tmp_path / "vm").resolve() / "benchflow"
    (vm / "compose").mkdir(parents=True)
    for path in _DaytonaDinD._BENCHFLOW_COMPOSE_FILES:
        shutil.copy(path, vm / "compose" / path.name)
    shutil.copytree(
        task / "environment",
        vm / "task" / "environment",
        ignore=lambda folder, names: [
            n for n in names if (Path(folder) / n).is_symlink()
        ],
    )
    for name in ("verifier", "agent", "artifacts"):
        (vm / "logs" / name).mkdir(parents=True)
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    strategy._env = SimpleNamespace(
        environment_name="t",
        session_id="s1-dind",
        task_env_config=Task(task).config.sandbox,
        _persistent_env={},
        environment_dir=task / "environment",
        # The task's Compose file, as DaytonaSandbox finds it (f091d8903).
        _environment_docker_compose_path=compose_definition_path(task / "environment"),
        logger=logging.getLogger("test.s1.dind"),
    )
    strategy._use_prebuilt = False
    strategy._resolved_task_env = {}
    # What __init__ sets since the DinD build overlays (436ca3e8b).
    strategy._runtime_compose_path = None
    strategy._build_compose_path = None
    strategy._build_image_name = "bs1dind"
    commands: list[str] = []

    async def vm_exec(command, cwd=None, env=None, timeout_sec=None, **_):
        commands.append(command)
        here = str(vm)
        mapped = {k: v.replace("/benchflow", here) for k, v in (env or {}).items()}
        done = subprocess.run(
            ["sh", "-c", command.replace("/benchflow", here)],
            env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], **mapped},
            capture_output=True,
            text=True,
            timeout=60,
        )
        return ExecResult(
            stdout=done.stdout.replace(here, "/benchflow"),
            stderr=done.stderr.replace(here, "/benchflow"),
            return_code=done.returncode,
        )

    strategy._vm_exec = vm_exec  # type: ignore[method-assign]
    return strategy, vm, commands


async def _dind_refused(strategy) -> str:
    with pytest.raises(ValueError) as raised:
        await strategy._check_compose_project()
    return " ".join(str(raised.value).split())


@pytest.mark.parametrize(
    ("compose", "shown"),
    [
        (
            "services:\n  main:\n    volumes: ['/:/host:ro']\n",
            "mounts / for service main",
        ),
        (
            "services:\n  main:\n    volumes: ['../:/up']\n",
            "mounts /benchflow/task writable over the launch source",
        ),
        (
            "services:\n  main:\n    volumes: [/var/run/docker.sock:/s]\n",
            "mounts /var/run/docker.sock for service main",
        ),
        (
            "services:\n  main:\n    privileged: true\n",
            "runs service main privileged, which reaches the sandbox's VM",
        ),
        (
            "services:\n  main:\n    volumes: ['${HOST_AGENT_LOGS_PATH}/../..:/l']\n",
            "/logs/agent/../.. for service main, outside the task's folder",
        ),
    ],
    ids=["root", "parent", "docker-socket", "privileged", "climbing-logs"],
)
async def test_dind_refuses_what_reaches_the_sandbox_vm(
    monkeypatch, tmp_path, compose, shown
):
    """On Daytona DinD, Compose runs in the VM: what it resolves there may
    reach only the uploaded environment folder and BenchFlow's log folders."""
    task = _compose_task(tmp_path / "task", compose)
    strategy, _, _ = _dind(tmp_path, task)
    assert shown in await _dind_refused(strategy)
    monkeypatch.setenv(HOST_ACCESS, "1")
    await strategy._check_compose_project()


async def test_dind_passes_the_tasks_own_folders(tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    env_file: main.env\n    volumes:\n"
        "      - ./data:/data\n      - ${HOST_AGENT_LOGS_PATH}:/more\n"
        "      - cache:/cache\nvolumes:\n  cache: {}\n",
        files={"main.env": "MODE=fast\n", "data/x": "x\n"},
    )
    strategy, _, _ = _dind(tmp_path, task)
    await strategy._check_compose_project()


@pytest.mark.parametrize("read_only", [False, True])
async def test_dind_uploaded_task_root_keeps_mutable_source_checks(tmp_path, read_only):
    """Guards the 7b4907c8/S1 merge: uploaded task sources use the same invariant."""
    task = _compose_task(
        tmp_path / "task",
        'services:\n  main:\n    volumes: ["${CONTEXT_DIR}/..:/task'
        + (":ro" if read_only else "")
        + '"]\n',
    )
    strategy, _, _ = _dind(tmp_path, task)
    if read_only:
        _gate(task, sandbox_type="daytona")
        await strategy._check_compose_project()
    else:
        assert "writable" in _refused(task, "daytona")
        assert "writable" in await _dind_refused(strategy)


async def test_dind_follows_a_link_the_hook_made(tmp_path):
    """The task's pre-compose hook runs in the VM before Compose and can
    replace a folder with a link to /; the check after it follows the link."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: ['./data:/data']\n",
        files={"data/x": "x\n"},
    )
    strategy, vm, _ = _dind(tmp_path, task)
    await strategy._check_compose_project()
    shutil.rmtree(vm / "task" / "environment" / "data")
    (vm / "task" / "environment" / "data").symlink_to("/")
    assert "/benchflow/task/environment/data for service main" in await _dind_refused(
        strategy
    )


def test_dind_gate_keeps_includes_in_the_uploaded_folder(tmp_path):
    """Daytona uploads only the environment folder: the gate refuses an
    include or extends reaching outside it, before any sandbox."""
    task = _compose_task(
        tmp_path / "task",
        "include: ['../tests/extra.yaml']\nservices:\n  main: {}\n",
    )
    (task / "tests" / "extra.yaml").write_text("services: {side: {image: alpine}}\n")
    assert "outside the task's environment folder" in _refused(task, "daytona")
    remote = _compose_task(
        tmp_path / "remote",
        "include: [https://github.com/s1-attacker/x.git]\nservices:\n  main: {}\n",
    )
    assert "a remote file BenchFlow cannot check" in _refused(remote, "daytona")
    # The resolved project too, before any sandbox.
    privileged = _compose_task(
        tmp_path / "plain", "services:\n  main:\n    privileged: true\n"
    )
    assert "runs service main privileged" in _refused(privileged, "daytona")


async def test_dind_checks_before_the_hook_and_again_after_it(monkeypatch, tmp_path):
    """The check runs before any task-controlled step in the VM (the hook)
    and, when the task has a hook, again before Compose builds."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.sandbox import daytona as sdk

    task = _compose_task(tmp_path / "task", "services:\n  main: {}\n")
    (task / "environment" / "benchflow-pre-compose.sh").write_text("true\n")
    strategy, _, _ = _dind(tmp_path, task)
    env = strategy._env
    env._kwargs = {}
    env._auto_delete_interval = env._auto_stop_interval = 0
    env._create_sandbox = AsyncMock()
    env._sdk_upload_file = AsyncMock()
    env._sdk_upload_dir = AsyncMock()
    monkeypatch.setattr(
        sdk.DaytonaClientManager, "get_instance", AsyncMock(return_value=None)
    )
    # The Daytona SDK's parameter types: no sandbox is made.
    monkeypatch.setattr(sdk, "Resources", lambda **kw: kw)
    monkeypatch.setattr(sdk, "Image", SimpleNamespace(base=lambda image: image))
    monkeypatch.setattr(sdk, "CreateSandboxFromImageParams", lambda **kw: kw)
    steps: list[str] = []

    async def record(name, result=None):
        steps.append(name)
        return result

    strategy._vm_exec = lambda *a, **k: record(  # type: ignore[method-assign]
        "vm", SimpleNamespace(stdout="", stderr="", return_code=0)
    )
    strategy._wait_for_docker_daemon = lambda: record("daemon")  # type: ignore[method-assign]
    strategy._guard_daemon_from_containers = lambda: record("guard")  # type: ignore[method-assign]
    strategy._check_compose_project = lambda: record("check")  # type: ignore[method-assign]
    strategy._run_pre_compose_hook = lambda: record("hook")  # type: ignore[method-assign]

    async def compose_exec(subcommand, **_):
        from benchflow.sandbox._base import ExecResult

        steps.append(subcommand[0])
        # The build plan Compose resolves (_prepare_compose_builds, 436ca3e8b).
        resolved = '{"services": {"main": {"image": "alpine:3.20"}}}'
        return ExecResult(
            stdout=resolved if subcommand[0] == "config" else "",
            stderr="",
            return_code=0,
        )

    strategy._compose_exec = compose_exec  # type: ignore[method-assign]
    strategy._compose_up_with_retry = lambda _env: record("up")  # type: ignore[method-assign]
    strategy._wait_for_main_container = lambda: record("main")  # type: ignore[method-assign]
    # The bridge guard after Compose made its networks (f8e348db7).
    strategy._guard_compose_networks = lambda: record("networks")  # type: ignore[method-assign]
    strategy._pin_or_verify_bind_targets = lambda: record("bind-proof")  # type: ignore[method-assign]
    strategy._prepare_inert_bootstrap = lambda: record("bootstrap")  # type: ignore[method-assign]
    from benchflow.sandbox import _inert_bootstrap

    monkeypatch.setattr(_inert_bootstrap, "release", lambda *_: record("release"))
    await strategy.start(force_build=False)
    task_steps = [s for s in steps if s in ("check", "hook", "build", "up")]
    assert task_steps == ["check", "hook", "check", "build", "up"]
    assert steps[-5:] == ["up", "main", "networks", "bind-proof", "release"]
    assert steps.index("bootstrap") < steps.index("up")
