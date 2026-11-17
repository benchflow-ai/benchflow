"""Root's execs in main start bounded once the agent's turn has started.

RCI's run on #234 (86fd158d6) replaced the image's ``cat``, ``rm``, ``mkdir``,
``chmod`` and the rest from a root oracle's solve.sh; every root step
BenchFlow ran in main afterwards ran them with NET_ADMIN. The exec layer now
puts ``bfx launch --`` ahead of the route's shell for every such exec
(benchflow.sandbox._bounded_exec), from main's first admitted ingress for
BenchFlow's own plumbing and from the end of the turn for the task's own
commands as root, wherever BenchFlow's helper adds main's rules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow._bfx.locate import SANDBOX_PATH
from benchflow.sandbox import _bounded_exec
from benchflow.sandbox._base import ROOT_EXEC_PATH, SYSTEM_SHELL, ExecRoute
from benchflow.sandbox._bounded_exec import (
    BFX_LAUNCH,
    bfx_sh,
    bounds,
    end_agent_turn,
    launch_for,
    shell_launch,
)
from benchflow.sandbox.docker import route_exec_argv
from benchflow.trust.state import TRUSTED_TRANSFER, IngressKind, admit, tainted
from tests.test_docker_direct_exec import (
    MAIN_ID,
    FakeDocker,
    _sandbox,
    _started,
    _without_client_options,
)

PLUMBING = ExecRoute(pin=ROOT_EXEC_PATH, task_root=False, explicit_root=True)
TASK_ROOT = ExecRoute(pin=None, task_root=True, explicit_root=True)
NOT_ROOT = ExecRoute(pin=None, task_root=False, root=False)


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(tmp_path / "trust"))


@pytest.fixture
def docker(monkeypatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake)
    return fake


async def _helper_sandbox(tmp_path):
    """A started sandbox whose helper adds main's rules (decided before the
    start, so the fake daemon builds no helper image)."""
    sandbox = _sandbox(tmp_path)
    sandbox._rules_by_helper = True
    await _started(sandbox)
    sandbox.bfx_delivery = "bind"
    return sandbox


def _env(**kw):
    return SimpleNamespace(**{"rules_by_helper": True, "bfx_delivery": "bind", **kw})


def test_nothing_is_bounded_before_main_admits_an_ingress():
    env = _env()
    assert not tainted(env)
    for route in (PLUMBING, TASK_ROOT, NOT_ROOT):
        assert not bounds(env, route)
    # A trusted transfer (the task's or BenchFlow's own bytes) is no ingress.
    admit(env, "main", TRUSTED_TRANSFER, reason="the test's own bytes")
    assert not bounds(env, PLUMBING)


@pytest.mark.parametrize("kind", list(IngressKind))
def test_every_ingress_kind_bounds_root_plumbing(kind):
    env = _env()
    admit(env, "main", kind)
    assert bounds(env, PLUMBING)
    assert launch_for(env, PLUMBING) == BFX_LAUNCH
    # During the turn the task route is the agent's own command (solve.sh).
    assert not bounds(env, TASK_ROOT)
    assert not bounds(env, NOT_ROOT)
    # Another service has no bfx and no bound here.
    assert not bounds(env, PLUMBING, "db")


def test_the_end_of_the_turn_bounds_the_task_route_too():
    env = _env()
    end_agent_turn(env)
    assert bounds(env, TASK_ROOT)
    assert bounds(env, PLUMBING)
    assert not bounds(env, NOT_ROOT)


def test_without_the_helper_main_rules_come_from_main_and_nothing_is_bounded():
    env = _env(rules_by_helper=False)
    admit(env, "main", IngressKind.AGENT_LAUNCH)
    end_agent_turn(env)
    assert launch_for(env, PLUMBING) is None
    assert launch_for(env, TASK_ROOT) is None


def test_without_bfx_the_shell_text_starts_the_launch():
    env = _env(bfx_delivery=None)
    admit(env, "main", IngressKind.ORACLE_LAUNCH)
    assert launch_for(env, PLUMBING) == ()
    text = shell_launch(SYSTEM_SHELL, "id")
    assert "--bounding-set=-net_admin,-net_raw,-sys_ptrace" in text
    assert "--no-new-privs" in text
    assert "exit 97" in text


def test_route_exec_argv_puts_bfx_launch_ahead_of_the_route_shell():
    argv = route_exec_argv(PLUMBING, "id", container="c", user="0", launch=BFX_LAUNCH)
    at = argv.index("c")
    assert argv[at + 1 : at + 6] == [SANDBOX_PATH, "launch", "--", SYSTEM_SHELL, "-c"]
    plain = route_exec_argv(PLUMBING, "id", container="c", user="0")
    assert plain[plain.index("c") + 1] == SYSTEM_SHELL
    fallback = route_exec_argv(PLUMBING, "id", container="c", user="0", launch=())
    assert fallback[-3:-1] == [SYSTEM_SHELL, "-c"]
    assert "--bounding-set=-net_admin,-net_raw,-sys_ptrace" in fallback[-1]


def test_bfx_sh_names_bfx_by_path_and_refuses_relative_paths():
    text = bfx_sh("mkdir", "/run/runtime", parents=True, mode=0o1777)
    assert text.startswith(f"{SANDBOX_PATH} mkdir --json ")
    assert '"path":"run/runtime"' in text and '"root":"/"' in text
    with pytest.raises(ValueError):
        bfx_sh("remove", "tmp/x")
    with pytest.raises(ValueError):
        bfx_sh("remove", "/")


def _after_container(argv: list[str]) -> list[str]:
    return argv[argv.index(MAIN_ID) + 1 :]


async def test_docker_exec_starts_every_post_agent_root_exec_bounded(tmp_path, docker):
    """The exec layer itself: the same sandbox, the same call, before and
    after the first ingress and the end of the turn."""
    sandbox = await _helper_sandbox(tmp_path)

    async def sent(command: str, **kw) -> list[str]:
        docker.calls.clear()
        await sandbox.exec(command, **kw)
        ((argv, _),) = docker.calls
        return _after_container(_without_client_options(argv))

    # Before the agent's turn: as today.
    assert (await sent("id", user="root"))[0] == SYSTEM_SHELL

    admit(sandbox, "main", IngressKind.AGENT_LAUNCH)
    for kw in (
        {"user": "root"},
        {"user": 0},
        {},
        {"user": "root", "system_shell": True},
    ):
        tail = await sent("id", **kw)
        assert tail[:5] == [SANDBOX_PATH, "launch", "--", SYSTEM_SHELL, "-c"], kw
    # The agent's own command as root during its turn, and any non-root user.
    with sandbox.task_command():
        assert (await sent("solve", user="root"))[0] == SYSTEM_SHELL
    assert (await sent("id", user="1000"))[0] == SYSTEM_SHELL

    end_agent_turn(sandbox)
    with sandbox.task_command():
        tail = await sent("bash /tests/test.sh", user="root")
    assert tail[:5] == [SANDBOX_PATH, "launch", "--", SYSTEM_SHELL, "-c"]

    # Without bfx the route's shell starts the launch in its own text.
    sandbox.bfx_delivery = None
    tail = await sent("id", user="root")
    assert tail[:2] == [SYSTEM_SHELL, "-c"]
    assert "--bounding-set=-net_admin,-net_raw,-sys_ptrace" in tail[2]

    # With the helper off, main's rules come from main: nothing is bounded.
    sandbox.bfx_delivery = "bind"
    sandbox._rules_by_helper = False
    assert (await sent("id", user="root"))[0] == SYSTEM_SHELL


async def test_a_long_command_still_starts_bounded(tmp_path, docker):
    """A command past the transport bound travels on stdin to a constant
    reader: the reader's exec starts bounded too."""
    sandbox = await _helper_sandbox(tmp_path)
    admit(sandbox, "main", IngressKind.ORACLE_LAUNCH)
    docker.calls.clear()
    await sandbox.exec("true; " + "x" * 300_000, user="root")
    ((argv, _),) = docker.calls
    tail = _after_container(_without_client_options(argv))
    assert tail[:5] == [SANDBOX_PATH, "launch", "--", SYSTEM_SHELL, "-c"]
    assert len(tail[5]) < 10_000


def test_the_module_names_its_attribute_on_the_sandbox():
    env = SimpleNamespace()
    end_agent_turn(env)
    assert getattr(env, _bounded_exec._LEVEL) == _bounded_exec.TURN_OVER
    end_agent_turn(None)


def test_the_protected_script_and_identity_execs_carry_the_launch():
    """The script agent's runner and the identity lookups build their own
    ``docker compose exec`` (``protected_exec_argv``): bfx launch goes
    ahead of their ``env -i``."""
    from benchflow.sandbox._identities import protected_exec_argv

    argv = protected_exec_argv({"Env": []}, "id", route=PLUMBING, launch=BFX_LAUNCH)
    at = argv.index("main")
    assert argv[at + 1 : at + 5] == [SANDBOX_PATH, "launch", "--", "/usr/bin/env"]
    plain = protected_exec_argv({"Env": []}, "id", route=PLUMBING)
    assert plain[plain.index("main") + 1] == "/usr/bin/env"
    fallback = protected_exec_argv({"Env": []}, "id", route=PLUMBING, launch=())
    assert fallback[fallback.index("main") + 1] == "/usr/bin/env"
    assert "--bounding-set=-net_admin,-net_raw,-sys_ptrace" in fallback[-1]
