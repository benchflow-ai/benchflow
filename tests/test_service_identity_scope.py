"""Guards DT2-CONF-8/9, introduced by b02a18f3 and present at e28ebc01."""

import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import SYSTEM_PATH, SandboxConfig
from benchflow.sandbox._identities import Identity, kill_prelude, snapshot
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task import keep_services
from tests.receipt_fakes import receipt_answer


def result(stdout="", code=0):
    return SimpleNamespace(stdout=stdout, stderr="", return_code=code)


def su_probe(passwd, command):
    """What the oracle's su prints for the identity probe: su looks the
    operand up as an account name only (BusyBox and util-linux alike)."""
    # PT adds the root route before the same task-native su invocation.
    prefix, separator, invocation = command.partition("\nsu -s /bin/bash ")
    if separator:
        assert "export PATH" in prefix
        command = "su -s /bin/bash " + invocation
    words = shlex.split(command)
    assert words[:3] == ["su", "-s", "/bin/bash"] and words[4] == "-c", command
    record = next(
        (
            r
            for r in (line.split(":") for line in passwd.splitlines())
            if r[0] == words[3]
        ),
        None,
    )
    if record is None:
        return result(f"su: unknown user {words[3]}\n", code=1)
    return result(f"benchflow-id:{record[2]}:{record[3]}\n")


def task(user, service="main"):
    return SimpleNamespace(
        config=SimpleNamespace(
            metadata={"x-keep-services": [{"port": 8080}]},
            verifier=SimpleNamespace(user=user, service=service),
        )
    )


def sandbox(tmp_path, backend, *, grader_uid=2000, bash=True):
    """Model each trusted service's own Config and immutable account files."""
    main = "a" * 64
    grader = "b" * 64
    configs = {
        main: {"User": "root", "Env": ["ENV=/app/startup", "LD_PRELOAD=/app/hook"]},
        grader: {"User": "checker", "Env": ["BASH_ENV=/app/startup"]},
    }
    accounts = {
        "main": "root:x:0:0::/:/bin/sh\nagent:x:1000:3000::/:/bin/sh\n",
        "grader": f"root:x:0:0::/:/bin/sh\nchecker:x:{grader_uid}:4000::/:/bin/sh\n",
    }
    reads = []

    def execute(argv):
        # The account read crosses as a receipt (its NUL separator).
        return receipt_answer(argv[-1], lambda read: _execute([*argv[:-1], read]))

    def _execute(argv):
        service = next(a for a in argv if a in accounts)
        if "benchflow-id" in argv[-1]:
            return su_probe(accounts[service], argv[-1])
        if not bash and "/bin/bash" in argv:
            return result(code=127)
        reads.append((service, argv))
        return result(accounts[service] + "\0root:x:0:\nteam:x:4000:\n")

    async def cli(argv, **kwargs):
        if argv[0] == "ps":
            service = argv[-1].removeprefix("label=com.docker.compose.service=")
            return result({"main": main, "grader": grader}.get(service, "") + "\n")
        assert argv[0] == "inspect"
        return result(json.dumps([{"Config": configs[argv[1]]}]))

    if backend == "dind":
        from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDinD

        env = DaytonaSandbox.__new__(DaytonaSandbox)
        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        env._strategy = strategy
        strategy._env = env

        async def compose(argv, **kwargs):
            if argv[0] == "ps":
                return result({"main": main, "grader": grader}[argv[-1]] + "\n")
            return execute(argv)

        async def vm(command, **kwargs):
            if command.startswith("docker inspect"):
                return result(json.dumps(configs[command.split()[-1]]))
            # identity_exec: a plain docker exec on the container's ID.
            argv = shlex.split(command.partition("; docker ")[2])
            return execute([{main: "main", grader: "grader"}.get(a, a) for a in argv])

        strategy._compose_exec = AsyncMock(side_effect=compose)
        strategy._vm_exec = AsyncMock(side_effect=vm)

        # The old generic path used exec for both root reads and default id.
        async def ordinary(command, service="main", **kwargs):
            return receipt_answer(command, lambda read: _ordinary(read, service))

        def _ordinary(command, service):
            if "benchflow-id" in command:
                return su_probe(accounts[service], command)
            if command.startswith("id"):
                uid = grader_uid if service == "grader" else 0
                return result(f"{uid}\n4000\n")
            return execute([service])

        env.exec = AsyncMock(side_effect=ordinary)
    else:
        cls = DockerSandbox if backend == "docker" else RemoteDockerSandbox
        env = cls.__new__(cls)
        DockerSandbox.__init__(
            env,
            environment_dir=tmp_path,
            environment_name="dt2-scope",
            session_id="dt2-scope",
            rollout_paths=None,
            task_env_config=SandboxConfig(docker_image="dt2-trusted-image"),
        )
        env._docker_cli = AsyncMock(side_effect=cli)
        env._run_docker_compose_command = AsyncMock(
            side_effect=lambda argv, **kwargs: execute(argv)
        )
    return env, accounts, configs, reads


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
@pytest.mark.parametrize(
    "user",
    [
        "checker",
        "2000",
        "002000",
        "+002000",
        2000,
        "2000:team",
        "2000:004000",
        None,
        "",
    ],
)
async def test_verifier_uses_selected_services_accounts(tmp_path, backend, user):
    """Guards CONF-8: names, numbers, groups and defaults belong to grader."""
    env, accounts, _, reads = sandbox(tmp_path, backend)
    trial = task(user, "grader")
    await keep_services.prepare_identities(env, trial, "agent")
    assert trial._benchflow_service_identities == (
        Identity(1000, 3000),
        Identity(2000, 4000),
    )
    assert keep_services.trial_keep(trial, "agent") is None
    # Later changes cannot redirect retention or the main container's cleanup.
    accounts["grader"] = "checker:x:0:0::/:/bin/sh\n"
    await keep_services.prepare_identities(env, trial, "agent")
    assert trial._benchflow_service_identities[1] == Identity(2000, 4000)
    from benchflow.sandbox.lockdown import _KILL_PRELUDE

    assert "uid=1000\ngid=3000\n" in kill_prelude(env, _KILL_PRELUDE, "agent")
    assert [service for service, _ in reads] == ["main", "grader"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
@pytest.mark.parametrize("uid", [0, 2000])
async def test_same_account_name_can_have_different_service_uids(
    tmp_path, backend, uid
):
    """Guards CONF-8: equal names in different services are independent."""
    env, accounts, _, _ = sandbox(tmp_path, backend, grader_uid=uid)
    accounts["main"] += f"checker:x:{2000 if uid == 0 else 0}:0::/:/bin/sh\n"
    trial = task("checker", "grader")
    if uid == 0:
        # Root in another service keeps nothing beside it (DT2-CONF-10).
        with pytest.raises(RuntimeError, match="service 'grader'"):
            await keep_services.prepare_identities(env, trial, "agent")
    else:
        await keep_services.prepare_identities(env, trial, "agent")
        assert keep_services.trial_keep(trial, "agent") is None
    assert trial._benchflow_service_identities[1] == Identity(uid, 4000)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
@pytest.mark.parametrize(
    "user", ["checker ", "checker\n", "2000\n", "002000\t", "2000:team\r", "missing"]
)
async def test_selected_service_rejects_unresolved_selectors(tmp_path, backend, user):
    """Guards CONF-8: unresolved selectors never silently disable policy."""
    env, _, _, _ = sandbox(tmp_path, backend)
    with pytest.raises(RuntimeError, match=r"use an existing.*without whitespace"):
        await keep_services.prepare_identities(env, task(user, "grader"), "agent")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
async def test_identity_preparation_needs_only_posix_sh(tmp_path, backend):
    """Guards CONF-9: nop/ACP retention does not require operator-script Bash."""
    env, _, _, reads = sandbox(tmp_path, backend, bash=False)
    await keep_services.prepare_identities(env, task("root"), "agent")
    table = await snapshot(env)
    assert table.resolve("agent") == Identity(1000, 3000)
    assert len(reads) == 1
    argv = reads[0][1]
    assert argv[argv.index("main") + 1 : -1] == [
        "/usr/bin/env",
        "-i",
        f"PATH={SYSTEM_PATH}",
        "HOME=/root",
        "/bin/sh",
        "-c",
    ]
    # A loader name is removed (bare -e NAME), never set empty
    # (SCRIPT-AGENT-DOCKER-F1); other start-up names are set empty.
    assert "ENV=" in argv and "LD_PRELOAD" in argv and "LD_PRELOAD=" not in argv
    assert "--noprofile" not in argv


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
@pytest.mark.parametrize("default", ["checker", "0", "000", "+00", "-00", "0:team"])
async def test_root_defaults_belong_to_selected_service(tmp_path, backend, default):
    """Guards CONF-8: inherited root names/numbers use grader, not main."""
    env, _, configs, _ = sandbox(tmp_path, backend, grader_uid=0)
    configs["b" * 64]["User"] = default
    configs["a" * 64]["User"] = "agent"
    trial = task(None, "grader")
    # Root in another service keeps nothing beside it (DT2-CONF-10).
    with pytest.raises(RuntimeError, match="service 'grader'"):
        await keep_services.prepare_identities(env, trial, "agent")
    assert trial._benchflow_service_identities[0] == Identity(1000, 3000)
    assert trial._benchflow_service_identities[1].uid == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
async def test_missing_verifier_service_refuses_without_main_fallback(
    tmp_path, backend
):
    """Guards CONF-8: an unavailable service cannot inherit main's identity."""
    env, _, _, _ = sandbox(tmp_path, backend)
    if backend == "dind":
        original = env._strategy._compose_exec

        async def compose(argv, **kwargs):
            if argv == ["ps", "-q", "grader"]:
                return result()
            return await original(argv, **kwargs)

        env._strategy._compose_exec = AsyncMock(side_effect=compose)
    else:
        original = env._docker_cli

        async def cli(argv, **kwargs):
            if argv[0] == "ps" and argv[-1].endswith("service=grader"):
                return result()
            return await original(argv, **kwargs)

        env._docker_cli = AsyncMock(side_effect=cli)
    with pytest.raises(
        RuntimeError, match=r"check verifier.service.*start its container"
    ):
        await keep_services.prepare_identities(env, task("root", "grader"), "agent")
