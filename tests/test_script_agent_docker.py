"""Guards fc2b9e9a with real uid, output and process checks in fresh containers."""

import asyncio
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox._base import SandboxConfig
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerSandbox

IMAGE = "ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"


class Container(DockerSandbox):
    def __init__(self, name, persistent=None):
        self.name = name
        # DockerSandbox adds its task and manifest env to every exec.
        self.persistent = dict(persistent or {})

    async def _main_container_id(self):
        return self.name

    async def _docker_cli(self, args, check=True, *, raw_output=False):
        # raw_output: production parsers ask for the undisplayed record; this
        # transport never redacts, so both forms are the same here.
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", *args],
            capture_output=True,
            text=True,
            check=check,
        )
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    async def _run_docker_compose_command(
        self,
        command,
        check=True,
        timeout_sec=None,
        *,
        merge_stderr=True,
        raw_output=False,
        stdin_data=None,
        unset_env=(),
    ):
        # Exercise production script_runner_exec/upload with raw Docker as
        # the transport, with production's signature (a fake without
        # raw_output failed every call, so these tests checked nothing:
        # SCRIPT-AGENT-DOCKER-F1). raw_output selects separate streams, as
        # in production. The warning fixture below uses actual Compose too.
        merge_stderr = merge_stderr and not raw_output
        if command[0] == "cp":
            argv = [
                "docker",
                "cp",
                command[1],
                command[2].replace("main:", f"{self.name}:", 1),
            ]
        else:
            assert command[0] == "exec"
            service = command.index("main")
            flags = [arg for arg in command[1:service] if arg != "-T"]
            argv = ["docker", "exec", *flags, self.name, *command[service + 1 :]]
        result = await asyncio.to_thread(
            subprocess.run,
            argv,
            input=stdin_data.decode() if stdin_data is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
            # As production: a removed name (bare -e NAME) is not filled in
            # from the client's environment.
            env={k: v for k, v in os.environ.items() if k not in unset_env},
        )
        if check:
            assert result.returncode == 0, result.stderr
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    async def exec(
        self, command, *, user="root", cwd=None, timeout_sec=30, env=None, **kwargs
    ):
        argv = ["docker", "exec", "-u", user]
        for key, value in {**self.persistent, **(env or {})}.items():
            argv += ["-e", f"{key}={value}"]
        if cwd:
            argv += ["-w", cwd]
        argv += [self.name, "bash", "-c", command]
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # DockerSandbox.exec raises this for an exec past timeout_sec.
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds"
            ) from None
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    async def upload_file(self, source, target, *, mode: int | None = None, **_kwargs):
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "cp", str(source), f"{self.name}:{target}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        if mode is not None:
            await self.script_runner_exec(
                f"chmod {mode:o} -- {shlex.quote(target)}", timeout_sec=30
            )


class RemoteContainer(Container, RemoteDockerSandbox):
    """Exercise remote Docker's inherited runner over the local test daemon."""


BUSYBOX_IMAGE = (
    "bash@sha256:a54fb4422b18f05dd3107c36f39d67b26334fda7ec89f4126052b45e228e2f15"
)
UBUNTU_SETUP = "useradd -m -s /bin/bash agent"
# Alpine/BusyBox with bash installed (at /usr/local/bin) and BusyBox timeout/su.
BUSYBOX_SETUP = "ln -s /usr/local/bin/bash /bin/bash && adduser -D -s /bin/bash agent"


@contextlib.contextmanager
def started_box(image=IMAGE, setup=UBUNTU_SETUP, env=(), run_flags=()):
    if not shutil.which("docker"):
        pytest.skip("Docker is unavailable")
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
        pytest.skip("Docker daemon is unavailable")
    name = f"dt2-test-{uuid.uuid4().hex[:12]}"
    flags = [arg for pair in env for arg in ("-e", pair)]
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "target=DT2",
            "--network",
            "none",
            *flags,
            *run_flags,
            image,
            "sleep",
            "300",
        ],
        check=True,
        capture_output=True,
    )
    try:
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "root",
                name,
                "/usr/bin/env",
                "-i",
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "/bin/sh",
                "-c",
                f"{setup} && mkdir -p /app /logs/agent && chown agent:agent /app /logs/agent",
            ],
            check=True,
            capture_output=True,
        )
        yield Container(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


@pytest.fixture
def box():
    with started_box() as container:
        yield container


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [7, 124, 137])
async def test_script_real_agent_user_output_exit_and_clean_environment(
    box, tmp_path, exit_code, monkeypatch
):
    monkeypatch.setenv("DT2_HOST_ONLY", "must-not-reach-script")
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[solution.env]\nDT2_SOLUTION_ONLY = "must-not-reach-script"\n'
    )
    (task / "instruction.md").write_text("Write hello.txt\n")
    script = tmp_path / "attempt.sh"
    script.write_text(
        f'whoami; pwd; printf "host=%s solution=%s\\n" "${{DT2_HOST_ONLY-unset}}" "${{DT2_SOLUTION_ONLY-unset}}"; stat -Lc "%U %a" "$0"; printf "err  \\n" >&2; printf "last  \\n"; exit {exit_code}\n'
    )
    rollout = Rollout(
        RolloutConfig(task_path=task, agent=f"script:{script}", timeout=10)
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "container-check"
    rollout._started_at = datetime.now()
    rollout._agent_cwd = "/app"
    rollout._timeout = 10
    rollout._env = box
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())

    async def verify():
        rollout._rewards = {"reward": 0.0}

    rollout.verify = verify
    result = await rollout.run()
    assert result.error is None
    event = result.trajectory[0]
    assert event["return_code"] == exit_code and event["timed_out"] is False
    assert (
        event["stdout"] == "agent\n/app\nhost=unset solution=unset\nroot 755\nlast\nerr"
    )
    captured = await box.exec("cat /logs/agent/script.txt")
    assert captured.stdout == event["stdout"]
    assert result.rewards == {"reward": 0.0}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_timeout_stops_detached_writer_before_verification(box, tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Do nothing\n")
    script = tmp_path / "slow.sh"
    script.write_text(
        "setsid bash -c 'trap \"\" TERM; while :; do echo tick >> /app/writes; sleep 0.1; done' &\necho $! > /app/writer.pid\nsleep 100\n"
    )
    rollout = Rollout(
        RolloutConfig(task_path=task, agent=f"script:{script}", timeout=1)
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "timeout-check"
    rollout._started_at = datetime.now()
    rollout._agent_cwd = "/app"
    rollout._timeout = 1
    rollout._env = box
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())

    async def verify():
        observed = await box.exec(
            'before=$(wc -l < /app/writes); sleep 0.3; after=$(wc -l < /app/writes); test "$before" = "$after" && ! ps -u agent -o stat= | grep -qv \'^Z\''
        )
        assert observed.return_code == 0, observed.stdout + observed.stderr
        rollout._rewards = {"reward": 0.0}

    rollout.verify = verify
    result = await rollout.run()
    assert result.error is None and result.verifier_error is None
    assert result.rewards == {"reward": 0.0}
    event = result.trajectory[0]
    assert event["return_code"] == 124 and event["timed_out"] is True
    assert json.loads((tmp_path / "result.json").read_text())["model"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_log_publication_cannot_follow_a_symlink_as_root(box):
    from benchflow.rollout._setup import _run_script

    prepared = await box.exec(
        "printf untouched > /tmp/dt2-victim; chmod 600 /tmp/dt2-victim"
    )
    assert prepared.return_code == 0
    events, _ = await _run_script(
        box,
        b"ln -s /tmp/dt2-victim /logs/agent/script.txt; printf output\n",
        "script:attempt.sh",
        10,
        "agent",
        "/app",
        {},
    )
    assert events[0]["return_code"] == 0 and events[0]["stdout"] == "output"
    observed = await box.exec("cat /tmp/dt2-victim")
    assert observed.stdout == "untouched"


async def run_scored(container, tmp_path, body, timeout=10, workdir="/app"):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Write hello.txt\n")
    script = tmp_path / "attempt.sh"
    script.write_text(body)
    rollout = Rollout(
        RolloutConfig(task_path=task, agent=f"script:{script}", timeout=timeout)
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "container-check"
    rollout._started_at = datetime.now()
    rollout._agent_cwd = workdir
    rollout._timeout = timeout
    rollout._env = container
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())

    async def verify():
        rollout._rewards = {"reward": 0.0}

    rollout.verify = verify
    return await rollout.run()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_agent_fifo_at_log_destination_still_scores(box, tmp_path):
    """Guards DT2-R1-A-1: a FIFO at script.txt blocked publishing past 10 s."""
    result = await run_scored(
        box, tmp_path, "mkfifo /logs/agent/script.txt; printf output\n"
    )
    assert result.error is None and result.rewards == {"reward": 0.0}
    assert result.trajectory[0]["stdout"] == "output"
    published = await box.exec(
        "test -f /logs/agent/script.txt && cat /logs/agent/script.txt"
    )
    assert published.stdout == "output"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 3])
async def test_script_runs_under_busybox_timeout(tmp_path, exit_code):
    """Guards DT2-R1-A-3: GNU-only timeout options never started the script."""
    with started_box(BUSYBOX_IMAGE, BUSYBOX_SETUP) as container:
        result = await run_scored(
            container, tmp_path, f"whoami; printf ran > /app/ran; exit {exit_code}\n"
        )
        assert result.error is None and result.rewards == {"reward": 0.0}
        event = result.trajectory[0]
        assert event["return_code"] == exit_code and event["timed_out"] is False
        assert event["stdout"] == "agent"
        assert (await container.exec("cat /app/ran")).stdout == "ran"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_busybox_watchdog_is_a_timeout(tmp_path):
    """Guards DT2-R1-A-3: BusyBox's watchdog exits 143, not GNU's 124."""
    with started_box(BUSYBOX_IMAGE, BUSYBOX_SETUP) as container:
        result = await run_scored(container, tmp_path, "sleep 100\n", timeout=1)
        assert result.error is None and result.rewards == {"reward": 0.0}
        event = result.trajectory[0]
        assert event["return_code"] == 124 and event["timed_out"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_gets_container_env_not_exec_env(tmp_path):
    """Guards DT2-R1-A-4 and DT2-R2-A-2: env -i kept only PATH and the locale,
    so image ENV such as RUSTUP_HOME or LD_LIBRARY_PATH never reached the
    script, while Rust's exec as the agent keeps it."""
    image_path = (
        "/opt/tool/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    with started_box(
        setup=f"{UBUNTU_SETUP} && mkdir -p /opt/tool/bin && printf '#!/bin/sh\\necho tool-ran\\n' > /opt/tool/bin/mytool && chmod 755 /opt/tool/bin/mytool",
        env=(
            f"PATH={image_path}",
            "LANG=C.UTF-8",
            "RUSTUP_HOME=/usr/local/rustup",
            "LD_LIBRARY_PATH=/opt/tool/lib",
            "DT2_IMAGE_VALUE=with spaces and = sign",
        ),
    ) as container:
        container.persistent = {"DT2_EXEC_ONLY": "not-for-script"}
        result = await run_scored(
            container,
            tmp_path,
            'mytool; printf "%s|%s|%s|%s|%s|%s|%s\\n" "$PATH" "${LANG-unset}" '
            '"${RUSTUP_HOME-unset}" "${LD_LIBRARY_PATH-unset}" '
            '"${DT2_IMAGE_VALUE-unset}" "${DT2_EXEC_ONLY-unset}" "$HOME"\n',
        )
        assert result.error is None
        event = result.trajectory[0]
        assert event["return_code"] == 0
        assert event["stdout"] == (
            f"tool-ran\n{image_path}|C.UTF-8|/usr/local/rustup|/opt/tool/lib|"
            "with spaces and = sign|unset|/home/agent"
        )


VOLATILE = ("PWD", "OLDPWD", "SHLVL", "_", "USER", "LOGNAME")


def env_names_and_values(text):
    return sorted(
        line
        for line in text.splitlines()
        if line and line.partition("=")[0] not in VOLATILE
    )


@pytest.fixture
def warning_compose(tmp_path, monkeypatch):
    """A task compose, with real warnings and no host mounts or network."""
    if not shutil.which("docker"):
        pytest.skip("Docker is unavailable")
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
        pytest.skip("Docker daemon is unavailable")
    monkeypatch.delenv("DT2_UNSET_COMPOSE_VALUE", raising=False)
    (tmp_path / "docker-compose.yaml").write_text(
        'version: "3.8"\nservices:\n  main:\n'
        f'    image: "{IMAGE}"\n'
        '    command: ["sleep", "300"]\n    network_mode: none\n'
        "    labels: {target: DT2}\n"
        '    environment:\n      DT2_VALUE: "${DT2_UNSET_COMPOSE_VALUE}"\n'
        '      PATH: "/app/bin:/usr/bin:/bin"\n'
    )

    class TaskCompose(DockerSandbox):
        @property
        def _docker_compose_paths(self):
            return [self._environment_docker_compose_path]

    sandbox = TaskCompose(
        environment_dir=tmp_path,
        environment_name="warning-task",
        session_id=f"dt2-test-{uuid.uuid4().hex[:12]}",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    argv = [
        "docker",
        "compose",
        "-p",
        sandbox.compose_project_name,
        "-f",
        str(tmp_path / "docker-compose.yaml"),
    ]
    try:
        subprocess.run([*argv, "up", "-d"], check=True, capture_output=True)
        probe = subprocess.run(
            [*argv, "ps", "-q", "main"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "warning" in probe.stderr.lower()
        yield sandbox, probe.stdout.strip()
    finally:
        subprocess.run([*argv, "down"], check=True, capture_output=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_main_container_id_with_warning_compose(warning_compose):
    """Guards DT2-RC1-A-1 at 5756d715 with an actual warning-producing task."""
    sandbox, container_id = warning_compose
    assert await sandbox._main_container_id() == container_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_container_config_with_warning_compose(warning_compose):
    """Guards DT2-RC1-A-1 at 5756d715 through the script's production reader."""
    sandbox, _ = warning_compose
    config = await sandbox.script_container_config()
    assert "DT2_VALUE=" in config["Env"]
    assert config["Image"] == IMAGE


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        "cat",
        "wc",
        "tail",
        "base64",
        "chmod",
        "chown",
        "id",
        "mkdir",
        "stat",
        "readlink",
        "su",
        "timeout",
        "env",
        "sh",
        "bash",
    ],
)
async def test_script_root_helpers_ignore_agent_path(warning_compose, tool):
    """Root must not run PATH shadows."""
    from benchflow.rollout._setup import _run_script

    container, container_id = warning_compose
    subprocess.run(
        [
            "docker",
            "exec",
            container_id,
            "/usr/bin/env",
            "-i",
            "PATH=/usr/sbin:/usr/bin:/bin",
            "/bin/sh",
            "-c",
            f"{UBUNTU_SETUP}; mkdir -p /app /logs/agent; chown agent:agent /app /logs/agent",
        ],
        check=True,
        capture_output=True,
    )
    body = (
        "mkdir -p /app/bin\n"
        f'printf \'#!/bin/sh\\n[ "$(/usr/bin/id -u)" != 0 ] || '
        f'printf root >> /app/root-shadow\\nexec /usr/bin/{tool} "$@"\\n\' '
        f"> /app/bin/{tool}\n"
        f"chmod 755 /app/bin/{tool}\n"
        "printf '%s' \"$PATH\" > /app/agent-path\nprintf output\n"
    )
    events, _ = await _run_script(
        container, body.encode(), "script:attempt.sh", 10, "agent", "/app", {}
    )
    shadow = subprocess.run(
        ["docker", "exec", container_id, "/usr/bin/test", "-e", "/app/root-shadow"],
        capture_output=True,
        check=False,
    )
    assert shadow.returncode == 1, "an agent helper executed as root"
    assert events[0]["stdout"] == "output"
    path = subprocess.run(
        ["docker", "exec", container_id, "/bin/cat", "/app/agent-path"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert path.stdout == "/app/bin:/usr/bin:/bin"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_script_root_launcher_ignores_image_bash_startup(timeout):
    """Guards the fix at 90d0c606 before the first root shell starts."""
    from benchflow.rollout._setup import _run_script

    with started_box(env=("BASH_ENV=/app/startup.sh",)) as container:
        subprocess.run(
            [
                "docker",
                "exec",
                container.name,
                "/bin/sh",
                "-c",
                "printf '[ \"$(/usr/bin/id -u)\" != 0 ] || printf root >> /app/root-shadow\\n' > /app/startup.sh; chown agent:agent /app/startup.sh",
            ],
            check=True,
            capture_output=True,
        )
        events, _ = await _run_script(
            container,
            b"printf output\n" + (b"sleep 100\n" if timeout else b""),
            "script:attempt.sh",
            1 if timeout else 10,
            "agent",
            "/app",
            {},
        )
        shadow = subprocess.run(
            [
                "docker",
                "exec",
                container.name,
                "/usr/bin/test",
                "-e",
                "/app/root-shadow",
            ],
            capture_output=True,
            check=False,
        )
        assert shadow.returncode == 1
        assert events[0]["stdout"] == "output"
        assert events[0]["timed_out"] is timeout


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote_docker"])
@pytest.mark.parametrize("origin", ["container", "agent"])
@pytest.mark.parametrize(
    "startup,path,loaded",
    [
        ("/app/startup.sh", "/app/startup.sh", True),
        ("startup.sh", "/app/startup.sh", True),
        ("./startup.sh", "/app/startup.sh", True),
        ("$HOME/startup.sh", "/tmp/dt2-home/startup.sh", True),
        ("${HOME}/startup.sh", "/tmp/dt2-home/startup.sh", True),
        ("${DT2_MISSING:-/app}/startup.sh", "/app/startup.sh", True),
        ('$(printf %s "$HOME")/startup.sh', "/tmp/dt2-home/startup.sh", True),
        ('`printf %s "$HOME"`/startup.sh', "/tmp/dt2-home/startup.sh", True),
        # Bash may reuse fd 3 for a read-only startup input during expansion;
        # what must be gone is the writable receipt, not the descriptor number.
        (
            '$(test "$(id -u)" != 0 && ! (printf receipt-probe >&3) 2>/dev/null && printf /app)/startup.sh',
            "/app/startup.sh",
            True,
        ),
        ("/app/startup$((6 * 7)).sh", "/app/startup42.sh", True),
        ("/app/startup with spaces.sh", "/app/startup with spaces.sh", True),
        ("/app/startup*.sh", "/app/startup*.sh", True),
        ("/app/startup;literal.sh", "/app/startup;literal.sh", True),
        ("path-only.sh", "/app/bin/path-only.sh", False),
        ("/app/missing.sh", "/app/startup.sh", False),
        ("", "/app/startup.sh", False),
        (None, "/app/startup.sh", False),
    ],
)
async def test_script_startup_matches_native_bash(
    tmp_path, backend, origin, startup, path, loaded
):
    """Guards DT2-CONF-1: 886d70bb lost Bash's startup filename expansion."""
    from benchflow.rollout._setup import _run_script

    variables = {"HOME": "/tmp/dt2-home", "PATH": "/app/bin:/usr/bin:/bin"}
    if startup is not None:
        variables["BASH_ENV"] = startup
    configured = variables if origin == "container" else {"BASH_ENV": "/missing"}
    with started_box(env=tuple(f"{k}={v}" for k, v in configured.items())) as box:
        container = RemoteContainer(box.name) if backend == "remote_docker" else box
        assert (
            await container.script_runner_exec("mkdir -p /tmp/dt2-home /app/bin")
        ).return_code == 0
        hook = tmp_path / "startup.sh"
        hook.write_text(
            '[ "$(id -u)" != 0 ] || exit 91\n'
            "[ ! -e /proc/$$/fd/3 ] || exit 92\n"
            "startup_calls=$(( ${startup_calls:-0} + 1 ))\n"
            "ANSWER=42\n"
            "startup_function() { printf function; }\n"
        )
        await container.script_runner_upload(str(hook), path)
        script = (
            'printf "%s:%s:%s" "${ANSWER:-unset}" "${startup_calls:-0}" '
            '"$(type -t startup_function)"\n'
        )
        attempt = tmp_path / "attempt.sh"
        attempt.write_text(script)
        await container.script_runner_upload(str(attempt), "/app/native.sh")
        native = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "agent",
                "-w",
                "/app",
                container.name,
                "/usr/bin/env",
                "-i",
                *(f"{k}={v}" for k, v in variables.items()),
                "/bin/bash",
                "--noprofile",
                "--norc",
                "/app/native.sh",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert native.returncode == 0, native.stderr
        expected = "42:1:function" if loaded else "unset:0:"
        assert native.stdout == expected
        # HOME is normally the sandbox user's home; use a made-up test home.
        overrides = variables if origin == "agent" else {"HOME": variables["HOME"]}
        if origin == "agent" and startup is None:
            # An explicit empty value disables the inherited startup hook.
            overrides["BASH_ENV"] = ""
        events, _ = await _run_script(
            container,
            script.encode(),
            "script:attempt.sh",
            10,
            "agent",
            "/app",
            overrides,
        )
        assert events[0]["return_code"] == native.returncode
        assert events[0]["timed_out"] is False
        assert events[0]["stdout"] == "\n".join(
            part for part in (native.stdout.rstrip(), native.stderr.rstrip()) if part
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote_docker"])
@pytest.mark.parametrize("origin", ["container", "agent"])
@pytest.mark.parametrize("startup", [None, "", "$PWD/startup.sh"])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"POSIXLY_CORRECT": ""},
        {"POSIXLY_CORRECT": "0"},
        {"POSIXLY_CORRECT": "1"},
        {"SHELLOPTS": "posix"},
        {"SHELLOPTS": "errexit:nounset:posix"},
        {"SHELLOPTS": "noexec"},
        {"SHELLOPTS": "privileged"},
        {"SHELLOPTS": "verbose:xtrace"},
        {"BASHOPTS": "extdebug"},
        {"POSIXLY_CORRECT": "1", "SHELLOPTS": "posix", "BASHOPTS": "nullglob"},
    ],
    ids=[
        "default",
        "posix-empty",
        "posix-zero",
        "posix-one",
        "shell-posix",
        "shell-combined",
        "shell-noexec",
        "shell-privileged",
        "shell-trace",
        "bash-extdebug",
        "all-settings",
    ],
)
async def test_script_startup_settings_match_native_bash(
    tmp_path, backend, origin, startup, settings
):
    """Guards DT2-CONF-2 at e1b4ab00: task settings cannot disable receipts."""
    from benchflow.rollout._setup import _run_script

    variables = {"HOME": "/tmp/dt2-home", "PATH": "/usr/bin:/bin", **settings}
    if startup is not None:
        variables["BASH_ENV"] = startup
    configured = variables if origin == "container" else {}
    with started_box(env=tuple(f"{k}={v}" for k, v in configured.items())) as box:
        container = RemoteContainer(box.name) if backend == "remote_docker" else box
        hook = tmp_path / "startup.sh"
        hook.write_text(
            'test "$(id -u)" != 0 || exit 91\n'
            "if (printf receipt-probe >&3) 2>/dev/null; then exit 92; fi\n"
            "startup_calls=$(( ${startup_calls:-0} + 1 ))\n"
        )
        await container.script_runner_upload(str(hook), "/app/startup.sh")
        script = (
            'test "$(id -u)" != 0 || exit 93\n'
            "if (printf receipt-probe >&3) 2>/dev/null; then exit 94; fi\n"
            'printf "argc=%s startup=%s posix=%s opts=%s bashopts=%s\\n" '
            '"$#" "${startup_calls:-0}" "${POSIXLY_CORRECT-unset}" '
            '"$SHELLOPTS" "$BASHOPTS"\n'
            "printf 42 > /app/answer\n"
        )
        attempt = tmp_path / "attempt.sh"
        attempt.write_text(script)
        await container.script_runner_upload(str(attempt), "/app/native.sh")
        # Use the same held-input path so native diagnostics match too.
        native = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "agent",
                "-w",
                "/app",
                container.name,
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                "/bin/sh",
                "-c",
                'exec 6< /app/native.sh; exec /usr/bin/env -i "$@" '
                "/bin/bash --noprofile --norc /dev/fd/6",
                "native",
                *(f"{k}={v}" for k, v in variables.items()),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert native.returncode == 0, native.stderr
        native_answer = await container.script_runner_exec("cat /app/answer")
        assert native_answer.stdout == (
            "" if settings.get("SHELLOPTS") == "noexec" else "42"
        )
        assert (
            await container.script_runner_exec("rm -f /app/answer")
        ).return_code == 0
        overrides = variables if origin == "agent" else {"HOME": variables["HOME"]}
        events, _ = await _run_script(
            container,
            script.encode(),
            "script:attempt.sh",
            10,
            "agent",
            "/app",
            overrides,
        )
        assert events[0]["return_code"] == native.returncode
        assert events[0]["timed_out"] is False
        assert events[0]["stdout"] == "\n".join(
            part for part in (native.stdout.rstrip(), native.stderr.rstrip()) if part
        )
        answer = await container.script_runner_exec("cat /app/answer")
        assert (answer.return_code, answer.stdout) == (
            native_answer.return_code,
            native_answer.stdout,
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", ["/opt/runtime", "/tmp/work", "/app/link"])
@pytest.mark.parametrize("runner", ["legacy", "discovered"])
async def test_script_cannot_replace_runner_through_workspace_parent(
    box, tmp_path, workdir, runner
):
    """Guards the fix at 90d0c606 with a dummy private marker."""
    prepared = await box.exec(
        "mkdir -p /opt/runtime /tmp/work; chown agent:agent /opt/runtime /tmp/work; "
        "ln -s /tmp/work /app/link; "
        "printf dt2-private-marker > /root/dt2-private; chmod 600 /root/dt2-private"
    )
    assert prepared.return_code == 0
    result = await run_scored(
        box,
        tmp_path,
        (
            "r=/opt/runtime/operator-script\n"
            if runner == "legacy"
            else 'for r in /.runtime-script-* /opt/runtime/operator-script; do [ -d "$r" ] && break; done\n'
        )
        + 'if mv "$r" "$PWD/saved-runner" 2>/dev/null; then\n'
        'mkdir "$r"; printf 1 > "$r/started"; '
        'ln -s /root/dt2-private "$r/script.stdout"; : > "$r/script.stderr"\n'
        "fi\nprintf safe-output\n",
        workdir=workdir,
    )
    assert result.error is None
    assert result.trajectory[0]["stdout"] == "safe-output"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_inaccessible_legacy_parent_does_not_score_unattempted(
    box, tmp_path
):
    """Prepare before start proof."""
    prepared = await box.exec("mkdir -p /opt/runtime; chmod 700 /opt/runtime")
    assert prepared.return_code == 0
    result = await run_scored(box, tmp_path, "printf ran > /app/ran\n")
    ran = await box.exec("test -f /app/ran")
    assert ran.return_code == 0 or (result.error and result.rewards is None)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe", ["root", "utility_parent", "utility_symlink"])
async def test_script_refuses_replaceable_root_or_utility_paths(box, tmp_path, unsafe):
    """Guards the fix at 90d0c606 for the new runner's actual parents."""
    commands = {
        "root": "chmod 777 /",
        "utility_parent": "chmod 777 /usr/bin",
        "utility_symlink": "cp /usr/bin/tail /app/tail; ln -sf /app/tail /usr/bin/tail",
    }
    assert (await box.exec(commands[unsafe])).return_code == 0
    result = await run_scored(box, tmp_path, "printf ran > /app/ran\n")
    assert "before the script started" in (result.error or "")
    assert result.rewards is None
    assert (await box.exec("test -f /app/ran")).return_code == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename",
    ["image.env", "script.stdout", "script.stderr", "attempt.sh", "started.bash"],
)
async def test_script_agent_setup_failure_has_no_start_receipt(box, tmp_path, filename):
    """Guards the fix at 90d0c606 for each post-su prerequisite."""

    class BrokenSetup(Container):
        async def break_setup(self, command):
            if "su -s /bin/bash" in command:
                directory = re.search(
                    r"/\.runtime-script-[0-9a-f]{32}|/opt/runtime/operator-script",
                    command,
                )[0]
                await super().exec(f"chmod 600 {directory}/{filename}")

        async def exec(self, command, **kwargs):
            await self.break_setup(command)
            return await super().exec(command, **kwargs)

        async def script_runner_exec(self, command, **kwargs):
            await self.break_setup(command)
            return await super().script_runner_exec(command, **kwargs)

    result = await run_scored(
        BrokenSetup(box.name), tmp_path, "printf ran > /app/ran\n"
    )
    assert "before the script started" in (result.error or "")
    assert result.rewards is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_with_warning_compose(
    warning_compose, tmp_path, monkeypatch
):
    """Guards the fix at 90d0c606 through production exec/cp."""
    # Compose 2.40.3 warns on this fixture's ps/config but can optimize exec
    # without reloading it. Inject one stderr diagnostic at the CLI boundary
    # too, so both machine reads must tolerate a warning on every invocation.
    original = asyncio.create_subprocess_exec

    async def noisy_compose(*argv, **kwargs):
        # docker <global flags> compose ...: production gives the daemon and
        # the hermetic client configuration (--config) before "compose".
        if argv[:1] == ("docker",) and "compose" in argv:
            argv = (
                "/bin/sh",
                "-c",
                'printf "dt2-compose-warning\\n" >&2; exec "$@"',
                "dt2-compose",
                *argv,
            )
        return await original(*argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", noisy_compose)
    sandbox, _ = warning_compose
    prepared = await sandbox.exec(
        f"export PATH=/usr/sbin:/usr/bin:/bin; {UBUNTU_SETUP}; mkdir -p /app /logs/agent; chown agent:agent /app /logs/agent",
        user="root",
    )
    assert prepared.return_code == 0
    result = await run_scored(sandbox, tmp_path, "printf output; printf error >&2\n")
    assert result.error is None
    assert result.trajectory[0]["stdout"] == "output\nerror"
    observed = await sandbox.script_runner_exec("cat /logs/agent/script.txt")
    assert observed.stdout == "output\nerror"
    assert "warning" in observed.stderr.lower()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entries", [(), ("JAVA_HOME=/opt/jdk",)], ids=["null-env", "no-path"]
)
async def test_script_imported_image_env_matches_plain_exec(tmp_path, entries):
    """Guards DT2-RC1-A-2/B-1 at 5756d715 with a PATH-less docker import."""
    # Export drops image metadata, so import yields Config.Env=null without
    # depending on a registry's image build tool or default PATH settings.
    image = f"dt2-test-import-{uuid.uuid4().hex[:12]}"
    archive = tmp_path / "rootfs.tar"
    with started_box() as source:
        subprocess.run(
            ["docker", "export", "-o", str(archive), source.name],
            check=True,
            capture_output=True,
        )
    try:
        subprocess.run(
            ["docker", "import", str(archive), image], check=True, capture_output=True
        )
        with started_box(image=image, setup="true", env=entries) as container:
            config = await container.script_container_config()
            assert config["Env"] == (list(entries) or None)
            result = await run_scored(container, tmp_path, "env > /app/script.env\n")
            assert result.error is None
            assert result.trajectory[0]["return_code"] == 0
            script_env = (await container.exec("cat /app/script.env")).stdout
            plain = subprocess.run(
                ["docker", "exec", "-u", "agent", "-w", "/app", container.name, "env"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            assert (
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                in plain.splitlines()
            )
            assert env_names_and_values(script_env) == env_names_and_values(plain)
    finally:
        subprocess.run(
            ["docker", "image", "rm", image], check=True, capture_output=True
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_flags", [(), ("--user", "65534")], ids=["root", "nonroot"]
)
async def test_script_env_matches_a_plain_exec_as_the_agent(tmp_path, run_flags):
    """Guards DT2-R3-A-1 at 0490a06c and DT2-R2-A-2: even with a non-root
    PID 1, the script gets Rust's `docker exec -u agent` environment."""
    with started_box(
        env=(
            "RUSTUP_HOME=/usr/local/rustup",
            "JAVA_HOME=/opt/jdk",
            "LD_LIBRARY_PATH=/opt/tool/lib",
        ),
        run_flags=run_flags,
    ) as container:
        container.persistent = {"DT2_EXEC_ONLY": "not-for-script"}
        result = await run_scored(container, tmp_path, "env > /app/script.env\n")
        assert result.error is None
        assert result.trajectory[0]["return_code"] == 0
        script_env = (await container.exec("cat /app/script.env")).stdout
        plain = subprocess.run(
            ["docker", "exec", "-u", "agent", "-w", "/app", container.name, "env"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "RUSTUP_HOME=/usr/local/rustup" in plain.splitlines()
        assert env_names_and_values(script_env) == env_names_and_values(plain)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "run_flags"),
    [
        (f"{UBUNTU_SETUP} && rm -f /usr/bin/su /bin/su", ()),
        (UBUNTU_SETUP, ("--cap-drop", "SETUID", "--cap-drop", "SETGID")),
    ],
    ids=["no-su", "no-setuid"],
)
async def test_script_su_failure_is_an_error_not_a_score(tmp_path, setup, run_flags):
    """Guards DT2-R2-A-1: su's own failure (127 without su, 1 without
    SETUID/SETGID) was recorded as the script's exit and the untouched
    workspace was scored; Rust's exec as the agent needs no su."""
    with started_box(setup=setup, run_flags=run_flags) as container:
        result = await run_scored(container, tmp_path, "printf ran > /app/ran\n")
        # Integ 7b4907c8 also requires a working user switch to prove process
        # quiescence. With SETUID/SETGID removed, that cleanup refuses first;
        # either failure must leave this unstarted attempt unscored.
        expected = (
            "sandbox-user process quiescence check exited with rc=2"
            if run_flags
            else "before the script started"
        )
        assert expected in (result.error or "")
        assert result.rewards is None
        missing = await container.exec("test -e /app/ran")
        assert missing.return_code == 1
