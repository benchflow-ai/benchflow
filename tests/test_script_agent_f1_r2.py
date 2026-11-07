"""Guards SCRIPT-AGENT-DOCKER-F1 round 2 at 5572a868a (reviews A and B).

- DinD's identity exec ran ``unset <every removed name>``, and a valid Docker
  name that is no shell identifier (``LD_ODD-NAME``) ended the shell with
  status 2 before docker ran (B M1, a regression of round 1).
- A non-bootstrapped service's valued loader switch (``LD_TRACE_LOADED_OBJECTS=1``)
  reached root's system shell, set only empty or unset after the loader had
  acted (B M3, A L-2).
- test.sh runs without the image's LD_LIBRARY_PATH and the other names V3's
  verifier environment clears; the trial now says so, by name only
 .
"""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "ORDINARY",
        "LD_NORMAL",
        "LD_ODD-NAME",
        "LD_ODD.NAME",
        "LD_ODD NAME",
        "MALLOC_ODD-NAME",
    ],
)
async def test_dind_identity_exec_takes_any_docker_env_name(tmp_path, name):
    """Review B's six cases: production identity_exec, a real /bin/sh for
    the VM's command, and a docker that prints its arguments. Every name
    Docker accepts reaches docker; a name no loader reads is set empty, not
    removed, and only shell identifiers are unset."""
    from benchflow.sandbox._base import SYSTEM_PATH, ExecRoute
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    strategy = object.__new__(_DaytonaDinD)
    strategy._env = SimpleNamespace()
    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nprintf "ARG %s\\n" "$@"\n')
    fake.chmod(0o755)
    config = {"Env": [f"{name}=honest"]}

    async def compose_exec(args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="container\n", stderr="")

    async def vm_exec(command, **kwargs):
        if command.startswith("docker inspect"):
            return SimpleNamespace(return_code=0, stdout=json.dumps(config), stderr="")
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            env={"PATH": f"{fake.parent}:/usr/bin:/bin"},
            check=False,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    strategy._compose_exec = compose_exec
    strategy._vm_exec = vm_exec
    route = ExecRoute(pin=SYSTEM_PATH, system_shell=True, task_root=False)
    with patch(
        "benchflow.sandbox.daytona_dind.system_shell_route",
        AsyncMock(return_value=route),
    ):
        result = await strategy.identity_exec("true")
    assert result.return_code == 0, (result.stdout, result.stderr)
    args = [line[4:] for line in result.stdout.splitlines() if line.startswith("ARG ")]
    assert args[:1] == ["exec"] and "container" in args
    if name.isidentifier() and name.startswith("LD_"):
        assert name in args and f"{name}=" not in args
    else:
        assert f"{name}=" in args and name not in args


def test_loader_variable_is_only_a_shell_identifier():
    from benchflow.sandbox._base import loader_variable

    assert loader_variable("LD_TRACE_LOADED_OBJECTS")
    assert loader_variable("MALLOC_ARENA_MAX")
    assert loader_variable("GLIBC_TUNABLES")
    for name in ("LD_ODD-NAME", "LD_ODD.NAME", "LD_ODD NAME", "MALLOC_ODD-NAME"):
        assert not loader_variable(name), name


def test_every_glibc_loader_switch_is_cleared_for_root():
    """glibc's ld.so switches (rtld.c process_envvars, the old releases'
    too) are in the list root's shells and S1's bootstrap clear; the ones
    that stop or replace a program's output are never given back."""
    from benchflow.sandbox import _inert_bootstrap
    from benchflow.sandbox._base import (
        IMAGE_UNRESTORED_VARS,
        SYSTEM_SHELL_UNSET_VARS,
    )

    switches = {
        "LD_TRACE_LOADED_OBJECTS",
        "LD_TRACE_PRELINKING",
        "LD_BIND_NOW",
        "LD_BIND_NOT",
        "LD_WARN",
        "LD_VERBOSE",
        "LD_SHOW_AUXV",
        "LD_DYNAMIC_WEAK",
        "LD_DEBUG",
        "LD_POINTER_GUARD",
        "LD_PREFER_MAP_32BIT_EXEC",
    }
    assert switches <= set(SYSTEM_SHELL_UNSET_VARS)
    assert switches <= set(_inert_bootstrap.EMPTY_ENV)
    assert "LD_TRACE_LOADED_OBJECTS" in _inert_bootstrap.BOOTSTRAP
    assert {"LD_TRACE_LOADED_OBJECTS", "LD_TRACE_PRELINKING"} <= IMAGE_UNRESTORED_VARS
    sandbox = SimpleNamespace(
        _bootstrap_configs={
            "main": {"Env": ["LD_TRACE_LOADED_OBJECTS=1", "LD_LIBRARY_PATH=/opt/l"]}
        }
    )
    assert _inert_bootstrap.restored_env(sandbox) == {"LD_LIBRARY_PATH": "/opt/l"}


SIDECAR_ENV = [
    "PATH=/usr/bin:/bin",
    "LD_TRACE_LOADED_OBJECTS=1",
    "LD_SHOW_AUXV=",
    "LD_LIBRARY_PATH=/opt/lib",
    "LD_ODD-NAME=1",
    "LD_DEBUG",
    "MALLOC_ARENA_MAX=2",
]


def test_loader_switches_are_the_valued_presence_names():
    """The names root's shell must lose before it loads: valued (an empty
    value counts for a switch), unset by the shell anyway, not file-naming
    (those stay empty) and not bare (not in Docker's exec environment)."""
    from benchflow.sandbox._base import loader_switches

    assert loader_switches(SIDECAR_ENV) == ["LD_SHOW_AUXV", "LD_TRACE_LOADED_OBJECTS"]
    assert loader_switches(None) == [] and loader_switches(["PATH=/bin"]) == []


def test_route_exec_argv_removes_the_given_switches():
    """A bare -e NAME for each switch the container sets (Docker removes
    it), empty values for the names that point at files, and no change at
    all for a container that sets none."""
    from benchflow.sandbox._base import SYSTEM_PATH, ExecRoute, loader_switches
    from benchflow.sandbox.docker import route_exec_argv

    route = ExecRoute(pin=SYSTEM_PATH, system_shell=True, task_root=False)
    plain = route_exec_argv(route, "true", container="side", user="0")
    argv = route_exec_argv(
        route, "true", container="side", user="0", remove=loader_switches(SIDECAR_ENV)
    )
    flags = [argv[i + 1] for i in range(len(argv) - 1) if argv[i] == "-e"]
    for name in ("LD_TRACE_LOADED_OBJECTS", "LD_SHOW_AUXV"):
        assert name in flags and f"{name}=" not in flags
    assert "LD_PRELOAD=" in flags and "LD_LIBRARY_PATH=" in flags
    # Without switches to remove the argv is what it was; with them, only
    # the bare pairs are added.
    assert not [
        f for f in plain[: plain.index("side")] if f.startswith("LD_") and "=" not in f
    ]
    stripped = list(argv)
    for name in ("LD_SHOW_AUXV", "LD_TRACE_LOADED_OBJECTS"):
        i = stripped.index(name)
        del stripped[i - 1 : i + 1]
    assert stripped == plain


def test_root_system_shell_on_a_traced_service_runs_its_command():
    """sh -c as root on a service with LD_TRACE_LOADED_OBJECTS=1 and
    LD_SHOW_AUXV in its environment runs its command: the route's flags
    applied to that environment, as Docker applies them, leave no switch."""
    from benchflow.sandbox._base import SYSTEM_PATH, ExecRoute, loader_switches
    from benchflow.sandbox.docker import route_exec_argv

    argv = route_exec_argv(
        ExecRoute(pin=SYSTEM_PATH, system_shell=True, task_root=False),
        "printf EXECUTED",
        container="side",
        user="0",
        remove=loader_switches(SIDECAR_ENV),
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LD_TRACE_LOADED_OBJECTS": "1",
        "LD_SHOW_AUXV": "",
    }
    for i in range(len(argv) - 1):
        if argv[i] == "-e":
            name, separator, value = argv[i + 1].partition("=")
            if separator:
                environment[name] = value
            else:
                environment.pop(name, None)
    shell = argv[argv.index("side") + 1 :]
    done = subprocess.run(shell, env=environment, capture_output=True, text=True)
    assert done.stdout == "EXECUTED", done.stdout


def test_only_a_service_s1_did_not_bootstrap_has_switches_to_remove():
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox.__new__(DockerSandbox)
    assert sandbox._service_loader_switches("side") == []
    sandbox._bootstrap_configs = {
        "main": {"Env": SIDECAR_ENV},
        "side": {"Env": SIDECAR_ENV},
        "db": {"Env": SIDECAR_ENV},
    }
    sandbox._bootstrap_services = {"main", "db"}
    assert sandbox._service_loader_switches("main") == []
    assert sandbox._service_loader_switches("db") == []
    assert sandbox._service_loader_switches("side") == [
        "LD_SHOW_AUXV",
        "LD_TRACE_LOADED_OBJECTS",
    ]


@pytest.mark.asyncio
async def test_docker_cli_exec_with_removals_runs_without_those_names(monkeypatch):
    """A bare -e NAME takes the docker client's own value when it has one:
    an exec that removes names runs without them and without any loader
    name; every other call keeps the inherited environment."""
    from benchflow.sandbox import docker as docker_module
    from benchflow.sandbox.docker import DockerSandbox

    seen = []

    async def fake_exec(*argv, env=None, **kwargs):
        seen.append(env)
        return SimpleNamespace(
            returncode=0, communicate=AsyncMock(return_value=(b"", b""))
        )

    monkeypatch.setenv("LD_SHOW_AUXV", "1")
    monkeypatch.setenv("LD_PRELOAD", "/host.so")
    monkeypatch.setattr(docker_module.asyncio, "create_subprocess_exec", fake_exec)
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._main_container = "c"
    await sandbox._docker_cli(
        ["exec", "-e", "LD_SHOW_AUXV", "c", "/bin/sh", "-c", "true"]
    )
    await sandbox._docker_cli(
        ["exec", "-e", "LD_PRELOAD=", "c", "/bin/sh", "-c", "true"]
    )
    assert seen[0] is not None and seen[0].get("PATH")
    assert "LD_SHOW_AUXV" not in seen[0] and "LD_PRELOAD" not in seen[0]
    assert seen[1] is None


def _verifier_task(env, service="main"):
    return SimpleNamespace(
        name="t",
        config=SimpleNamespace(verifier=SimpleNamespace(env=env, service=service)),
    )


def test_verifier_dropped_image_env_names_only():
    """The trial names each image name test.sh ran without,
    never its value; a task's own [verifier.env] value is no drop."""
    from benchflow._utils.trial_warnings import recording_trial_warnings
    from benchflow.rollout._setup import (
        _warn_verifier_dropped_image_env,
        verifier_dropped_image_env,
    )

    secret = "/opt/secret-lib-path"
    sandbox = SimpleNamespace(
        _bootstrap_configs={
            "main": {
                "Env": [
                    f"LD_LIBRARY_PATH={secret}",
                    "PYTHONSTARTUP=/opt/start.py",
                    "LD_PRELOAD",
                    "KEEP=1",
                ]
            },
            "side": {"Env": ["LD_PRELOAD=/opt/x.so"]},
        }
    )
    final = {"LD_LIBRARY_PATH": "", "PYTHONSTARTUP": "", "LD_PRELOAD": "", "KEEP": "1"}
    task = _verifier_task(final)
    assert verifier_dropped_image_env(sandbox, task) == [
        "LD_LIBRARY_PATH",
        "PYTHONSTARTUP",
    ]
    assert verifier_dropped_image_env(sandbox, _verifier_task(final, "side")) == [
        "LD_PRELOAD"
    ]
    # The task set its own value: nothing dropped.
    own = _verifier_task({**final, "LD_LIBRARY_PATH": "/opt/l", "PYTHONSTARTUP": "x"})
    assert verifier_dropped_image_env(sandbox, own) == []
    # No record of the service's own environment (another backend): no claim.
    assert verifier_dropped_image_env(SimpleNamespace(), task) == []
    with recording_trial_warnings() as warnings:
        _warn_verifier_dropped_image_env(sandbox, task)
        _warn_verifier_dropped_image_env(sandbox, task)
    assert len(warnings) == 1
    assert "LD_LIBRARY_PATH" in warnings[0] and "PYTHONSTARTUP" in warnings[0]
    assert secret not in warnings[0] and "/opt" not in warnings[0]


def test_result_json_lists_warnings_only_when_there_are_some(tmp_path):
    from datetime import datetime

    from benchflow.rollout._results import _build_rollout_result

    def build(folder, warnings):
        folder.mkdir()
        _build_rollout_result(
            folder,
            task_name="t",
            rollout_name="r",
            agent="oracle",
            agent_name="",
            model=None,
            n_tool_calls=0,
            prompts=[],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
            warnings=warnings,
        )
        return json.loads((folder / "result.json").read_text())

    text = "test.sh ran without the image's LD_LIBRARY_PATH: ..."
    assert build(tmp_path / "a", [text])["warnings"] == [text]
    assert "warnings" not in build(tmp_path / "b", [])
