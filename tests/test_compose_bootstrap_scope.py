"""Guards S1-B61-1/2 introduced/left at 03ca4742: native peers and clean IDs."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _inert_bootstrap as bootstrap
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult


class Project:
    """Stopped Docker Config fixtures; the peer has only a native binary."""

    def __init__(self, *, warning=False):
        self.ids = {"main": "a" * 64, "peer": "b" * 64}
        self.project = {"services": {"main": {}, "peer": {}}}
        self.records = {
            self.ids[name]: {
                "State": {"Running": False, "ExitCode": 0},
                "Config": {
                    "Entrypoint": ["/native-service"],
                    "Cmd": [],
                    "Env": [],
                    "Labels": {"com.docker.compose.service": name},
                },
                "Mounts": [],
            }
            for name in self.ids
        }
        self.records[self.ids["main"]]["Mounts"] = [
            {"Type": "bind", "Destination": "/data", "Source": "/tmp/task-data"}
        ]
        self.env = SimpleNamespace(
            _inspect_container=AsyncMock(side_effect=self.inspect),
            _docker_cli=AsyncMock(side_effect=self.cli),
        )
        self.warning = warning
        self.compose = AsyncMock(side_effect=self.run_compose)
        self.proved = False

    async def inspect(self, cid):
        assert cid in self.records, "Compose diagnostic used as a container ID"
        return self.records[cid]

    async def cli(self, args, **kwargs):
        if args[0] == "volume":
            return ExecResult(return_code=0, stdout='[{"Options": null}]', stderr="")
        assert self.ids["peer"] not in args, "shell-less peer needs no injected tool"
        return ExecResult(return_code=0, stdout="", stderr="")

    async def run_compose(self, args, **kwargs):
        output = ""
        if args[0] == "config":
            output = json.dumps(self.project)
        elif args[0] == "ps":
            output = "\n".join(self.ids.values())
            if self.warning and kwargs.get("merge_stderr", True):
                output = "warning: obsolete version field\n" + output
        elif args[0] == "up":
            assert self.proved, "native command started before the bind proof"
            self.records[self.ids["peer"]]["State"]["Running"] = True
        return ExecResult(return_code=0, stdout=output, stderr="")


async def test_shell_less_sidecar_retains_its_native_startup():
    """B61-1 at 03ca4742: unrelated scratch/distroless services need no shell."""
    fixture = Project()
    overlay = await bootstrap.prepare(fixture.env, fixture.compose)
    assert set(overlay["services"]) == {"main"}
    fixture.proved = True
    fixture.records[fixture.ids["main"]]["State"]["Running"] = True
    await bootstrap.release(fixture.env, fixture.compose)
    assert fixture.records[fixture.ids["peer"]]["State"]["Running"]
    assert fixture.records[fixture.ids["peer"]]["Config"]["Entrypoint"] == [
        "/native-service"
    ]


@pytest.mark.parametrize(
    "relation", ["pid", "ipc", "network_mode", "volumes_from", "volume"]
)
@pytest.mark.parametrize("reverse", [False, True])
async def test_related_services_still_wait_for_bind_proof(relation, reverse):
    """B61-1 at 03ca4742: sharing is symmetric and keeps the CONF-9 barrier."""
    fixture = Project()
    source, target = ("main", "peer") if reverse else ("peer", "main")
    if relation == "volume":
        for record in fixture.records.values():
            record["Mounts"].append(
                {"Type": "volume", "Name": "shared", "Destination": "/shared"}
            )
    else:
        fixture.project["services"][source][relation] = (
            [target + ":ro"] if relation == "volumes_from" else "service:" + target
        )
    overlay = await bootstrap.prepare(fixture.env, fixture.compose)
    assert set(overlay["services"]) == {"main", "peer"}


async def test_protection_is_transitive_but_not_an_ordinary_dependency():
    """B61-1 at 03ca4742: resource sharing, not depends_on, needs the barrier."""
    fixture = Project()
    fixture.project["services"]["main"]["depends_on"] = {
        "peer": {"condition": "service_healthy"}
    }
    assert set((await bootstrap.prepare(fixture.env, fixture.compose))["services"]) == {
        "main"
    }
    fixture.ids["third"] = "c" * 64
    fixture.records["c" * 64] = {
        "State": {"Running": False},
        "Config": {
            "Cmd": ["/bin/true"],
            "Labels": {"com.docker.compose.service": "third"},
        },
        "Mounts": [],
    }
    fixture.project["services"]["main"].pop("depends_on")
    fixture.project["services"]["peer"]["pid"] = "service:main"
    fixture.project["services"]["third"] = {"ipc": "service:peer"}
    assert set(
        (await bootstrap.prepare(fixture.env, fixture.compose))["services"]
    ) == set(fixture.ids)


async def test_native_healthcheck_uses_daemon_state_without_env_or_shell():
    """B61-1 at 03ca4742: scratch healthchecks keep their exec-form command."""
    fixture = Project()
    peer = fixture.records[fixture.ids["peer"]]
    peer["Config"]["Healthcheck"] = {"Test": ["CMD", "/native-service", "--health"]}
    peer["State"]["Health"] = {"Status": "healthy"}
    fixture.project["services"]["main"]["depends_on"] = {
        "peer": {"condition": "service_healthy"}
    }
    await bootstrap.prepare(fixture.env, fixture.compose)
    fixture.proved = True
    fixture.records[fixture.ids["main"]]["State"]["Running"] = True
    await bootstrap.release(fixture.env, fixture.compose)
    assert fixture.env._docker_cli.call_count == 2  # main's handler and release only


async def test_bind_backed_named_volume_requires_barrier():
    """B61-1 at 03ca4742: local-driver bind volumes need the same proof as binds."""
    fixture = Project()
    fixture.records[fixture.ids["main"]]["Mounts"] = []
    fixture.records[fixture.ids["peer"]]["Mounts"] = [
        {"Type": "volume", "Name": "backed", "Destination": "/data"}
    ]
    fixture.env._docker_cli = AsyncMock(
        return_value=ExecResult(
            return_code=0,
            stdout='[{"Options": {"type": "none", "o": "bind", "device": "/tmp/task-data"}}]',
            stderr="",
        )
    )
    assert set((await bootstrap.prepare(fixture.env, fixture.compose))["services"]) == {
        "peer"
    }


@pytest.mark.parametrize("backend", ["docker", "dind"])
async def test_initial_up_starts_only_protected_services(backend):
    """B61-1 at 03ca4742: native prototypes stay stopped until the proof."""
    from benchflow.sandbox.daytona_dind import _DaytonaDinD
    from benchflow.sandbox.docker import DockerSandbox

    run = AsyncMock(return_value=ExecResult(return_code=0, stdout="", stderr=""))
    env = SimpleNamespace(
        _bootstrap_services={"main"},
        _bootstrap_override="overlay",
        _run_docker_compose_command=run,
        _compose_exec=run,
    )
    if backend == "docker":
        await DockerSandbox._run_docker_compose_up(env)
    else:
        await _DaytonaDinD._compose_up_with_retry(
            env, SimpleNamespace(task_env_config=SimpleNamespace(build_timeout_sec=120))
        )
    command = run.call_args.args[0]
    assert command[-2:] == ["--no-deps", "main"]
    assert "peer" not in command


@pytest.mark.parametrize("phase", ["prepare", "release", "pin", "replacement"])
async def test_compose_warning_is_never_a_container_id(phase):
    """B61-2 at 03ca4742: each startup/restore ID consumer excludes stderr."""
    fixture = Project(warning=True)
    fixture.env._bind_baseline = None
    if phase == "prepare":
        await bootstrap.prepare(fixture.env, fixture.compose)
        return
    # Use a single protected service so this tests warnings independently of B61-1.
    fixture.ids.pop("peer")
    fixture.project["services"].pop("peer")
    fixture.records[fixture.ids["main"]]["Mounts"] = []
    fixture.env._bootstrap_configs = {
        "main": fixture.records[fixture.ids["main"]]["Config"]
    }
    fixture.env._bootstrap_project = fixture.project
    fixture.env._bootstrap_services = {"main"}
    fixture.records[fixture.ids["main"]]["State"]["Running"] = True
    fixture.env._docker_cli = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="", stderr="")
    )
    fixture.env._run_docker_compose_command = fixture.compose
    if phase == "release":
        await bootstrap.release(fixture.env, fixture.compose)
    elif phase == "pin":
        await DockerSandbox._pin_or_verify_bind_targets(fixture.env)
    else:
        await DockerSandbox._verify_replacement_bind_targets(
            fixture.env, fixture.ids["main"], {"main": {}}
        )


async def test_dind_bind_proof_excludes_compose_warnings():
    """B61-2 at 03ca4742: the DinD sibling uses the same discovery helper."""
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    fixture = Project(warning=True)
    fixture.ids.pop("peer")
    fixture.records[fixture.ids["main"]]["Mounts"] = []
    fixture.env._compose_exec = fixture.compose
    fixture.env._bind_baseline = None
    await _DaytonaDinD._pin_or_verify_bind_targets(fixture.env)
    assert fixture.env._bind_baseline == {"main": {}}


@pytest.mark.parametrize("output", ["warning", "a" * 64 + "\nwarning", "a" * 11])
async def test_invalid_stdout_is_not_treated_as_container_identity(output):
    """B61-2 at 03ca4742: separating stderr does not weaken ID validation."""
    from benchflow.sandbox._compose import compose_container_ids

    with pytest.raises(RuntimeError, match="Cannot identify"):
        await compose_container_ids(
            AsyncMock(return_value=ExecResult(return_code=0, stdout=output, stderr=""))
        )


async def test_dind_transport_keeps_warning_on_its_stderr_channel():
    """B61-2 at 03ca4742: SDK stdout/stderr remain separate for discovery."""
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    env = _DaytonaDinD.__new__(_DaytonaDinD)
    env._compose_cmd = lambda args: "docker compose " + " ".join(args)
    env._compose_env_vars = lambda: {}
    env._vm_exec = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="a" * 64, stderr="warning")
    )
    result = await env._compose_exec(["ps", "--all", "-q"], merge_stderr=False)
    assert result.stdout == "a" * 64 and result.stderr == "warning"
    assert "2>&1" not in env._vm_exec.call_args.args[0]
