"""Guards the fix against K1-R15-1/2/3/4 at 766dd68d.

The backend selects the account; the fallback must never request root.
Docker establishes no_new_privs at creation for every effective service.
"""

from __future__ import annotations

import os
import pwd
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rollout.session_factory_runtime import SessionFactorySandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("service", [None, "main", "target", "attacker"])
@pytest.mark.parametrize("switch", [(), ("worker",)])
@pytest.mark.parametrize("user", ["agent", "nobody", "65534", 65534])
async def test_docker_exec_keeps_the_backend_user_and_command(service, switch, user):
    inner = SimpleNamespace(
        agent_container_no_new_privileges=True,
        agent_identity_switch=switch,
        exec=AsyncMock(),
        exec_in_service=AsyncMock(),
    )
    env = SessionFactorySandbox(inner, {}, None, "agent")
    command = "printf ok"
    if service:
        await env.exec_in_service(service, command, user=user)
        args = inner.exec_in_service.await_args
        assert args.args == (service, command)
    else:
        await env.exec(command, user=user)
        args = inner.exec.await_args
        assert args.args == (command,)
    assert args.kwargs["user"] == user


@pytest.mark.asyncio
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
async def test_native_exec_never_upgrades_to_root_and_preserves_shell(shell, tmp_path):
    # The test user is already non-root in the standard gate. A separate
    # container run selects nobody:nogroup to exercise that real account.
    user = pwd.getpwuid(os.getuid()).pw_name
    if user == "root":
        user = "nobody"
    inner = SimpleNamespace(
        agent_exec_shell=shell,
        agent_container_no_new_privileges=False,
        exec=AsyncMock(),
    )
    command = (
        "a=(ok); printf '%s\\n' \"${a[0]}\"; "
        if shell.endswith("bash")
        else "printf 'ok\\n'; "
    ) + "/bin/sh -c \"/bin/grep '^NoNewPrivs:' /proc/self/status\"; /usr/bin/id -g"
    await SessionFactorySandbox(inner, {}, None, user).exec(command)
    call = inner.exec.await_args
    assert call.kwargs["user"] == user
    argv = ["/bin/sh", "-c", call.args[0]]
    if os.getuid() == 0:
        argv = [
            "/usr/bin/setpriv",
            f"--reuid={user}",
            f"--regid={pwd.getpwnam(user).pw_gid}",
            "--init-groups",
            "--",
            *argv,
        ]
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "ok",
        "NoNewPrivs:\t1",
        str(pwd.getpwnam(user).pw_gid),
    ]


@pytest.mark.asyncio
async def test_writable_path_program_is_not_a_bootstrap(tmp_path):
    user = pwd.getpwuid(os.getuid()).pw_name
    if user == "root":
        pytest.skip("Run this property under the standard gate's non-root user")
    marker = tmp_path / "marker"
    tools = tmp_path / "bin"
    tools.mkdir()
    # Every name the old root launcher or its likely next spelling could use.
    for name in ("setpriv", "id", "env", "sh", "bash", "su", "getent", "grep"):
        path = tools / name
        path.write_text(
            f"#!/bin/sh\nprintf planted > {shlex.quote(str(marker))}\nexit 91\n"
        )
        path.chmod(0o755)
    inner = SimpleNamespace(agent_exec_shell="/bin/sh", exec=AsyncMock())
    await SessionFactorySandbox(inner, {}, None, user).exec("/bin/echo ok")
    call = inner.exec.await_args
    assert call.kwargs["user"] == user
    result = subprocess.run(
        ["/bin/sh", "-c", call.args[0]],
        text=True,
        capture_output=True,
        env={"PATH": f"{tools}:/usr/bin:/bin", "TASK_VALUE": "x' $()\nvalue"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n"
    assert not marker.exists()


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
def test_main_container_gets_no_new_privileges_at_creation(backend, tmp_path):
    """A task without a Compose file gets the option from its own overlay; a
    task's own project gets it on every service once Compose resolved it
    (tests/test_compose_no_new_privileges.py), never from the base file,
    whose list a task's own option would duplicate."""
    import yaml

    from benchflow.sandbox._compose import (
        COMPOSE_NO_NEW_PRIVILEGES_PATH,
        compose_file_paths,
    )
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox
    from benchflow.task.config import SandboxConfig

    cls = DockerSandbox if backend == "docker" else RemoteDockerSandbox
    config = yaml.safe_load(cls._DOCKER_COMPOSE_BASE_PATH.read_text())
    assert "security_opt" not in config["services"]["main"]
    (tmp_path / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = compose_file_paths(
        tmp_path, SandboxConfig(), base=cls._DOCKER_COMPOSE_BASE_PATH
    )
    assert COMPOSE_NO_NEW_PRIVILEGES_PATH in paths
    overlay = yaml.safe_load(COMPOSE_NO_NEW_PRIVILEGES_PATH.read_text())
    assert overlay["services"]["main"]["security_opt"] == ["no-new-privileges:true"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona-dind"])
@pytest.mark.parametrize(
    "filename",
    ["compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"],
)
async def test_every_effective_compose_service_gets_the_option(
    backend, filename, tmp_path, monkeypatch
):
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(
        "!override", lambda loader, node: loader.construct_sequence(node, deep=True)
    )

    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths, Task

    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "task.toml").write_text('version = "1.0"\n')
    (task_dir / "environment" / "Dockerfile").write_text("FROM debian:bookworm-slim\n")
    (task_dir / "environment" / filename).write_text(
        # A valid project: Daytona DinD resolves it with docker compose
        # where the CLI is installed (_dind_preflight) and refuses a service
        # with neither an image nor a build.
        "services:\n  main: {}\n  target:\n    image: dummy:latest\n"
    )
    if backend == "remote-docker":
        monkeypatch.setenv("DOCKER_HOST", "ssh://dummy@example.invalid")
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk"),
        patch("benchflow.sandbox._sdk_ops.apply"),
        # integ's factory checks the Compose project with the docker CLI at
        # creation; this test reads only the build overlay.
        patch("benchflow.sandbox.setup._freeze"),
    ):
        sandbox = _create_sandbox_environment(
            "daytona" if backend == "daytona-dind" else backend,
            Task(task_dir),
            task_dir,
            "trial",
            RolloutPaths(rollout_dir=tmp_path / "trial"),
        )
    import json

    inventory = SimpleNamespace(
        stdout=json.dumps(
            {
                "services": {
                    name: {
                        "image": "dummy:latest",
                        "security_opt": ["no-new-privileges=false", "label=disable"],
                    }
                    for name in ("main", "target", "profiled", "included", "extended")
                }
            }
        ),
        return_code=0,
    )
    if backend == "daytona-dind":
        strategy = sandbox._strategy
        strategy._compose_exec = AsyncMock(return_value=inventory)
        uploaded = []

        async def upload(source, destination):
            uploaded.append((Path(source).read_text(), destination))

        sandbox._sdk_upload_file = upload
        await strategy._prepare_compose_builds()
        assert uploaded, "DinD must upload a security overlay before compose up"
        document, destination = uploaded[0]
        assert strategy._compose_file_flags()[-2:] == ["-f", destination]
        assert strategy._compose_exec.await_args.args[0] == [
            "config",
            "--format",
            "json",
        ]
    else:
        sandbox._run_docker_compose_command = AsyncMock(return_value=inventory)
        await sandbox._prepare_compose_builds()
        paths = sandbox._docker_compose_paths
        assert paths[-1].name.startswith("benchflow-build-")
        document = paths[-1].read_text()
        assert sandbox._run_docker_compose_command.await_args.kwargs == {
            "raw_output": True,
            "timeout_sec": 30,
        }
        paths[-1].unlink()
    services = yaml.load(document, Loader=Loader)["services"]
    assert set(services) == {"main", "target", "profiled", "included", "extended"}
    for service in services.values():
        assert service["security_opt"] == ["label=disable", "no-new-privileges:true"]


def test_backend_shell_contracts():
    from benchflow.sandbox.agentcore import AgentCoreSandbox
    from benchflow.sandbox.apple_container import AppleContainerSandbox
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.modal_impl import ModalSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    assert DockerSandbox.agent_container_no_new_privileges is True
    assert RemoteDockerSandbox.agent_container_no_new_privileges is True
    assert ModalSandbox.agent_exec_shell == "/bin/bash"
    assert AgentCoreSandbox.agent_exec_shell == "/bin/bash"
    assert AppleContainerSandbox.agent_exec_shell == "/bin/sh"
    for compose in (False, True):
        env = SimpleNamespace(_uses_compose=compose)
        assert DaytonaSandbox.agent_container_no_new_privileges.fget(env) is compose
        assert DaytonaSandbox.agent_exec_shell.fget(env) == (
            "/bin/sh" if compose else "/bin/bash"
        )


@pytest.mark.asyncio
async def test_bash_startup_happens_only_after_the_barrier(tmp_path):
    user = pwd.getpwuid(os.getuid()).pw_name
    if user == "root":
        pytest.skip("The standard gate exercises the non-root bootstrap")
    marker = tmp_path / "startup-status"
    startup = tmp_path / "startup.sh"
    startup.write_text(
        f"/bin/grep '^NoNewPrivs:' /proc/self/status > {shlex.quote(str(marker))}\n"
    )
    inner = SimpleNamespace(agent_exec_shell="/bin/bash", exec=AsyncMock())
    await SessionFactorySandbox(inner, {}, None, user).exec("a=(ok); /bin/echo ${a[0]}")
    command = inner.exec.await_args.args[0]
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "BASH_ENV": str(startup)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n"
    assert marker.read_text() == "NoNewPrivs:\t1\n"


@pytest.mark.asyncio
async def test_identity_switch_cleanup_still_has_the_native_release_blocker(
    monkeypatch,
):
    """K1-R15-2 at 766dd68d: a deterministic grace-handler schedule.

    No processes or account switching: the stand-in models the population a
    successful per-UID cleanup observes, including the later grace callback.
    Docker's creation-time barrier forbids the cross-account exec itself.
    """
    from benchflow.sandbox import lockdown

    population = {"agent", "worker"}

    async def kill(env, user, **kwargs):
        population.discard(user)
        if user == "agent":
            population.add("worker")
        return 1

    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs", kill)
    env = SimpleNamespace(agent_identity_switch=("worker",))
    await lockdown.kill_switched_accounts(env)
    await lockdown._kill_sandbox_user_procs(env, "agent")
    assert population == {"worker"}


@pytest.mark.parametrize(
    "existing",
    [
        [],
        ["no-new-privileges"],
        ["no-new-privileges=true"],
        ["no-new-privileges:false"],
        ["no-new-privileges=false"],
        ["no-new-privileges:true", "no-new-privileges=false"],
    ],
)
def test_container_security_replaces_the_option_by_identity(existing):
    import yaml

    from benchflow.sandbox._compose_runtime import private_build_override

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(
        "!override", lambda loader, node: loader.construct_sequence(node, deep=True)
    )
    config = {
        "services": {
            "123": {
                "image": "dummy:latest",
                "security_opt": [*existing, "seccomp=unconfined"],
            }
        }
    }
    try:
        document, _ = private_build_override(
            config, "dummy", prebuilt=True, agent_security=True
        )
    except TypeError:
        # At 766dd68d the old builder emits no security override for this input.
        document, _ = private_build_override(config, "dummy", prebuilt=True)
    options = yaml.load(document, Loader=Loader)["services"]["123"]["security_opt"]
    assert options == ["seccomp=unconfined", "no-new-privileges:true"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value,body,code,output",
    [
        ("SHELLOPTS", "errexit", "false; /bin/echo should-not-run", 1, ""),
        (
            "BASHOPTS",
            "nullglob",
            "a=(/tmp/k1-no-such-file-*); printf '%s\\n' ${#a[@]}",
            0,
            "0\n",
        ),
    ],
)
async def test_backend_bash_receives_its_startup_options_after_the_barrier(
    name, value, body, code, output
):
    """R15-4 / D868: Bash itself interprets its readonly startup options."""
    user = pwd.getpwuid(os.getuid()).pw_name
    if user == "root":
        pytest.skip("The standard gate exercises the non-root bootstrap")
    inner = SimpleNamespace(agent_exec_shell="/bin/bash", exec=AsyncMock())
    await SessionFactorySandbox(inner, {}, None, user).exec(body)
    command = inner.exec.await_args.args[0]
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", name: value},
    )
    assert result.returncode == code, result.stderr
    assert result.stdout == output, result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["daytona", "modal", "agentcore", "apple"])
async def test_native_provider_dispatch_starts_the_typed_bootstrap_in_system_sh(
    kind, monkeypatch
):
    """D868: exercise each provider's actual dispatch, without remote calls."""
    from unittest.mock import MagicMock

    from benchflow.sandbox.agent_exec import agent_exec_call

    command, user = agent_exec_call(
        "printf ok", "nobody", sandbox=SimpleNamespace(agent_exec_shell="/bin/bash")
    )
    result = SimpleNamespace(stdout="ok", stderr="", return_code=0)
    if kind == "modal":
        from benchflow.sandbox.modal_impl import ModalSandbox

        env = ModalSandbox.__new__(ModalSandbox)
        process = MagicMock()
        process.stdout.read.aio = AsyncMock(return_value="ok")
        process.stderr.read.aio = AsyncMock(return_value="")
        process.wait.aio = AsyncMock(return_value=0)
        env._sandbox = MagicMock()
        env._sandbox.exec.aio = AsyncMock(return_value=process)
    elif kind == "agentcore":
        from benchflow.sandbox.agentcore import AgentCoreSandbox

        env = AgentCoreSandbox.__new__(AgentCoreSandbox)
        env.runtime_arn = "dummy"
        env.runtime_session_id = "dummy"
        env._invoke_command = MagicMock(return_value=result)
    elif kind == "apple":
        from benchflow.sandbox import apple_container

        env = apple_container.AppleContainerSandbox.__new__(
            apple_container.AppleContainerSandbox
        )
        env._container_name = "dummy"
        dispatch = AsyncMock(return_value=result)
        monkeypatch.setattr(apple_container, "_run_cli", dispatch)
    else:
        from benchflow.sandbox import daytona

        daytona._load_daytona_sdk()
        env = daytona.DaytonaSandbox.__new__(daytona.DaytonaSandbox)
        env._strategy = daytona._DaytonaDirect(env)
        env._sandbox = MagicMock()
        env._sandbox.process.create_session = AsyncMock()
        env._sandbox.process.execute_session_command = AsyncMock(
            return_value=SimpleNamespace(cmd_id="dummy")
        )
        env._poll_response = AsyncMock(return_value=result)
        env._exec_counts = lambda: SimpleNamespace(
            exec_calls=0,
            transient_exec_calls=0,
            exec_seconds=0.0,
            transient_exec_seconds=0.0,
        )
    env._persistent_env = {}
    env.default_user = None
    # PT's route reads the account's uid once (exec_route); the image would
    # answer 65534 for nobody (C8).
    env._named_uids = (env._exec_generation, {("main", "nobody"): 65534})
    if kind == "daytona":
        await env._sandbox_exec(command, user=user, route=await env.exec_route(user))
        payload = env._sandbox.process.execute_session_command.await_args.args[
            1
        ].command
        assert "/bin/su nobody -s /bin/sh -c " in payload
        assert "/usr/bin/env -u BASH_ENV" in payload
    else:
        await env.exec(command, user=user)
        if kind == "modal":
            argv = env._sandbox.exec.aio.await_args.args
        elif kind == "agentcore":
            argv = shlex.split(env._invoke_command.call_args.args[0])
        else:
            argv = dispatch.await_args.args
            assert argv[:4] == ("exec", "--user", "nobody", "dummy")
            argv = argv[4:]
        assert list(argv[:5]) == [
            "/usr/bin/setpriv",
            "--no-new-privs",
            "--",
            "/bin/sh",
            "-c",
        ]
        payload = argv[5]
        if kind != "apple":
            assert "/bin/su nobody -s /bin/sh" in payload
    assert "/usr/bin/setpriv --no-new-privs" in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("mode", ["plain", "prebuilt", "recovery"])
async def test_single_container_creation_paths_keep_the_barrier(
    backend, mode, tmp_path, monkeypatch
):
    """D868: a provider may use its own base file without any Compose overlay."""
    import yaml

    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths, Task

    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text('version = "1.0"\n')
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "environment" / "Dockerfile").write_text("FROM debian:bookworm-slim\n")
    if backend == "remote-docker":
        monkeypatch.setenv("DOCKER_HOST", "ssh://dummy@example.invalid")
    sandbox = _create_sandbox_environment(
        backend,
        Task(task_dir),
        task_dir,
        "trial",
        RolloutPaths(rollout_dir=tmp_path / "trial"),
    )
    assert sandbox._uses_compose is False
    sandbox._use_prebuilt = mode != "plain"
    if mode == "recovery":
        sandbox._recovery_baseline = object()
    from benchflow.sandbox._compose import COMPOSE_NO_NEW_PRIVILEGES_PATH

    base, launcher, barrier = sandbox._docker_compose_paths[:3]
    assert base == sandbox._DOCKER_COMPOSE_BASE_PATH
    assert launcher == (
        sandbox._DOCKER_COMPOSE_BUILD_PATH
        if mode == "plain"
        else sandbox._DOCKER_COMPOSE_PREBUILT_PATH
    )
    # Without a task Compose file, its own overlay sets the option.
    assert barrier == COMPOSE_NO_NEW_PRIVILEGES_PATH
    config = yaml.safe_load(barrier.read_text())
    assert "no-new-privileges:true" in config["services"]["main"].get(
        "security_opt", ()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exported_capture_name", [False, True])
async def test_native_environment_crosses_the_barrier_without_an_argv_copy(
    exported_capture_name,
):
    """D868: valid environment data must not become one oversized argv entry."""
    user = pwd.getpwuid(os.getuid()).pw_name
    if user == "root":
        pytest.skip("The standard gate exercises the non-root bootstrap")
    keys = ("K1_LARGE_A", "K1_LARGE_B", "K1_LARGE_C", "K1_LARGE_D")
    environment = {key: "v" * 40000 for key in keys}
    environment["PATH"] = "/usr/bin:/bin"
    if exported_capture_name:
        environment["_bf_exports"] = "caller-owned"
    body = '/usr/bin/python3 -I -c \'import os; print([len(os.environ[k]) for k in ("K1_LARGE_A", "K1_LARGE_B", "K1_LARGE_C", "K1_LARGE_D")])\''
    inner = SimpleNamespace(agent_exec_shell="/bin/bash", exec=AsyncMock())
    await SessionFactorySandbox(inner, {}, None, user).exec(body)
    result = subprocess.run(
        ["/bin/sh", "-c", inner.exec.await_args.args[0]],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "[40000, 40000, 40000, 40000]\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value,body,code,output",
    [
        (
            "BASHOPTS",
            "nullglob",
            "a=(/tmp/k1-no-such-file-*); printf '%s\\n' ${#a[@]}",
            0,
            "0\n",
        ),
        ("SHELLOPTS", "errexit", "false; /bin/echo should-not-run", 1, ""),
        ("UID", "424242", "/usr/bin/id -u", 0, None),
        (
            "SHELLOPTS",
            "hashall",
            'case ":$SHELLOPTS:" in *:posix:*) printf wrong;; *) printf backend-bash;; esac',
            0,
            "backend-bash",
        ),
        (
            "BASHOPTS",
            "nullglob",
            "if shopt -q inherit_errexit; then printf wrong; else printf backend-bash; fi",
            0,
            "backend-bash",
        ),
    ],
)
@pytest.mark.parametrize("through_account_selector", [False, True])
async def test_a_bash_implementation_of_system_sh_can_restore_readonly_metadata(
    field, value, body, code, output, through_account_selector
):
    """D868: /bin/sh may itself be Bash rather than Dash or BusyBox."""
    user = pwd.getpwuid(os.getuid()).pw_name
    if user == "root":
        pytest.skip("The standard gate exercises the non-root bootstrap")
    inner = SimpleNamespace(agent_exec_shell="/bin/bash", exec=AsyncMock())
    await SessionFactorySandbox(inner, {}, None, user).exec(body)
    # Exercise the fixed system-sh invocations as Bash in POSIX mode,
    # independent of which /bin/sh the test runner's image supplies.
    command = inner.exec.await_args.args[0]
    if through_account_selector:
        from benchflow.sandbox._su import su_command

        command = su_command(command, user, shell="/bin/sh", trusted_tools=True)
        # Model su's shell without authentication: this test already runs
        # as the selected account. Root-container checks cover the real drop.
        command = command.replace(
            f"/bin/su {user} -s /bin/sh -c", "/bin/bash --posix -c"
        )
    command = command.replace("/bin/sh -c", "/bin/bash --posix -c")
    result = subprocess.run(
        ["/bin/bash", "--posix", "-c", command],
        env={"PATH": "/usr/bin:/bin", field: value},
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stderr
    assert result.stdout == (output if output is not None else f"{os.getuid()}\n"), (
        result.stderr
    )
    assert "readonly variable" not in result.stderr
