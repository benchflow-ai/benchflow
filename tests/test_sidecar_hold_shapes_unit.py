"""The hold's start-up shapes without a sandbox (the Docker rows are in
``test_sidecar_hold_shapes_docker.py``): a held service's healthcheck counted
as Docker counts it, from its release; a service that can't wait refused with
the reason; a restart at start-up covered again and released; a restart
later in the trial recorded, and a failing test.sh after it unscored.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _inert_bootstrap as bootstrap


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    async def sleep(self, delay: float) -> None:
        self.now += max(0.0, delay)


def _fake_loop(monkeypatch, clock: _Clock) -> None:
    async def wait_for(awaitable, timeout):
        return await awaitable

    monkeypatch.setattr(
        bootstrap,
        "asyncio",
        SimpleNamespace(
            get_running_loop=lambda: SimpleNamespace(time=lambda: clock.now),
            sleep=clock.sleep,
            wait_for=wait_for,
            gather=__import__("asyncio").gather,
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "system_shell_route",
        AsyncMock(return_value=SimpleNamespace(kit=False)),
    )
    monkeypatch.setattr("benchflow.sandbox.docker.route_exec_argv", lambda *a, **k: [])


def _health(**extra) -> dict:
    return {
        "Healthcheck": {
            "Test": ["CMD", "probe"],
            "Interval": 1_000_000_000,
            "Retries": 1,
            **extra,
        },
        "Labels": {"com.docker.compose.service": "web"},
    }


# ------------------------------------------------- a fast healthcheck


async def test_a_strict_check_counts_from_docker_s_first_probe(monkeypatch):
    """A fast healthcheck (interval 1 s, 1 retry, no start period): the server is not
    listening at its release, and is by 1 s. Docker's first probe is at 1 s,
    so it passes there; the probe at 0 counts nothing here."""
    clock = _Clock()
    _fake_loop(monkeypatch, clock)
    times = []

    async def probe(*_a, **_k):
        times.append(clock.now)
        return SimpleNamespace(return_code=0 if clock.now >= 1 else 1)

    env = SimpleNamespace(_docker_cli=probe)
    await bootstrap.wait_healthy(env, "web", _health(), released_at=0.0)
    assert times == [0.0, 1.0], times


async def test_the_check_counts_from_the_release_not_from_the_wait(monkeypatch):
    """Released at 0, waited on from 5 (after a dependency): Docker's first
    probe is at 8 (its interval), so the probe at 5 counts nothing and the
    one at 8 is the one retry."""
    clock = _Clock(5.0)
    _fake_loop(monkeypatch, clock)
    times = []

    async def probe(*_a, **_k):
        times.append(clock.now)
        return SimpleNamespace(return_code=1)

    env = SimpleNamespace(_docker_cli=probe)
    with pytest.raises(RuntimeError, match=r"healthcheck failed .*'web'"):
        await bootstrap.wait_healthy(
            env, "web", _health(Interval=8_000_000_000), released_at=0.0
        )
    assert times == [5.0, 8.0], times


async def test_a_service_that_never_turns_healthy_still_fails(monkeypatch):
    """Guard: the retries still end the wait (2 retries, interval 1 s)."""
    clock = _Clock()
    _fake_loop(monkeypatch, clock)
    times = []

    async def probe(*_a, **_k):
        times.append(clock.now)
        return SimpleNamespace(return_code=1)

    env = SimpleNamespace(_docker_cli=probe)
    with pytest.raises(RuntimeError, match="healthcheck failed"):
        await bootstrap.wait_healthy(env, "web", _health(Retries=2), released_at=0.0)
    assert times == [0.0, 1.0, 2.0], times


# --------------------------------------------------------- a service with no shell


@pytest.mark.parametrize(
    ("every", "policy", "until"),
    [
        (True, "", "until its metadata rule is in"),
        (
            True,
            "no-network",
            "until the rules of the task's network policy (no-network) are in",
        ),
        (False, "", "until its host binds are proven"),
    ],
    ids=["open-network", "policy", "binds"],
)
async def test_a_service_that_cannot_wait_is_refused_with_why_it_is_held(
    every, policy, until
):
    from benchflow.sandbox.service_cover import ServiceCoverError

    env = SimpleNamespace(
        _network_policy=policy,
        _docker_cli=AsyncMock(return_value=SimpleNamespace(return_code=1)),
    )
    with pytest.raises(ServiceCoverError) as caught:
        await bootstrap._require_bootstrap_tools(
            env, {"web"}, {"web": {"Id": "c" * 64}}, every=every
        )
    said = str(caught.value)
    assert "service 'web': this sidecar's image has no shell" in said, said
    assert f"so BenchFlow can't hold it {until}" in said, said
    assert "main" not in said, said
    if every and not policy:
        assert "--allow-cloud-metadata" in said and "no cloud credentials" in said


# ------------------------------------------------------- a restart at start-up


class _Restarting(SimpleNamespace):
    """A sandbox that can start a held container again (Docker's can)."""

    async def _start_held_again(self, name, cid):
        self.started.append((name, cid))
        if self.refuse:
            from benchflow.sandbox._held_mounts import HeldStartRefused

            raise HeldStartRefused(self.refuse)


def _restarting(inspect, *, policy="on-failure", refuse=""):
    sent = []

    async def docker(argv, **_k):
        sent.append(argv)
        return SimpleNamespace(return_code=0)

    env = _Restarting(
        _inspect_container=inspect,
        _docker_cli=docker,
        _task_restart_policies={"side": {"Name": policy}},
        started=[],
        refuse=refuse,
    )
    return env, sent


async def test_a_held_service_that_exits_at_start_is_started_again_by_benchflow(
    monkeypatch,
):
    """It exits once just after its release. Docker never starts it again
    (its policy is no); its task's policy would, so BenchFlow starts the
    same container, covers it again, releases it again, and watches it until
    it has run the settle time."""
    clock = _Clock(10.0)
    _fake_loop(monkeypatch, clock)
    states = iter([{"Running": False, "ExitCode": 1}, {"Running": True}])

    async def inspect(cid):
        return {"State": next(states, {"Running": True}), "RestartCount": 0}

    env, sent = _restarting(inspect)
    monkeypatch.setattr(bootstrap, "wait_ready", AsyncMock())
    monkeypatch.setattr(
        "benchflow.sandbox.docker.route_exec_argv",
        lambda route, script, container, user: ["exec", container, script],
    )
    recover = AsyncMock()
    released = {"side": 10.0}
    starts = bootstrap._Starts(env, {"side": "cid"}, released, recover)
    await bootstrap._settle_restarts(env, {"side": "cid"}, {"side"}, starts)
    assert env.started == [("side", "cid")]
    recover.assert_awaited_once_with("side", "cid")
    # Released from inside its container, never by Docker's kill.
    assert sent == [["exec", "cid", bootstrap._RELEASE_SH]], sent
    assert clock.now - released["side"] >= bootstrap.RESTART_SETTLE_SEC


async def test_a_start_the_checks_refuse_stays_down(monkeypatch):
    """BenchFlow's checks refuse the start (a mount that would land on bfx):
    the service stays down, named with the reason, and is not watched."""
    clock = _Clock()
    _fake_loop(monkeypatch, clock)

    async def inspect(cid):
        return {"State": {"Running": False, "ExitCode": 1}, "RestartCount": 0}

    env, _ = _restarting(inspect, refuse="the mount at /data/bfx lands on /.bfx/bfx")
    recover = AsyncMock()
    starts = bootstrap._Starts(env, {"side": "cid"}, {"side": 0.0}, recover)
    await bootstrap._settle_restarts(env, {"side": "cid"}, {"side"}, starts)
    assert starts.refused == {"side"}
    assert "lands on /.bfx/bfx" in env._held_start_refusals["side"]
    recover.assert_not_awaited()


@pytest.mark.parametrize(
    ("name", "joined"), [("side", False), ("main", False), ("side", True)]
)
async def test_a_sidecar_is_released_from_inside_and_main_by_docker_s_kill(
    monkeypatch, name, joined
):
    """Docker's kill cancels a container's restart policy for its next exit:
    a sidecar's release must not go through it. main's stays Docker's kill
    (its restart policy is refused under the block), and so does a sidecar's
    in another container's pid namespace, whose PID 1 is not its own."""
    sent = []

    async def docker(argv, **_k):
        sent.append(argv)
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(
        bootstrap,
        "system_shell_route",
        AsyncMock(return_value=SimpleNamespace(kit=False)),
    )
    monkeypatch.setattr(
        "benchflow.sandbox.docker.route_exec_argv",
        lambda route, script, container, user: ["exec", container, user, script],
    )
    await bootstrap._signal_release(
        SimpleNamespace(_docker_cli=docker), "cid", name, joined=joined
    )
    if name == "main" or joined:
        assert sent == [["kill", "--signal", "USR1", "cid"]], sent
    else:
        assert sent == [["exec", "cid", "0", bootstrap._RELEASE_SH]], sent
        assert 'kill -USR1 "$pid"' in bootstrap._RELEASE_SH


async def test_a_main_that_never_started_is_not_called_stopped():
    """A start refused before main's release (a sidecar that can't wait)
    leaves main a stopped prototype: that is no "main stopped right after
    its release", and the start's own reason stands."""
    from benchflow.sandbox.docker import DockerSandbox

    class _Box:
        _main_container = "a" * 64
        _main_stopped_record = None

        async def _inspect_container(self, cid):
            return {
                "State": {
                    "Running": False,
                    "ExitCode": 0,
                    "StartedAt": "0001-01-01T00:00:00Z",
                }
            }

    assert await DockerSandbox.main_stopped(_Box()) is None


async def test_a_service_with_no_restart_policy_is_not_watched(monkeypatch):
    clock = _Clock()
    _fake_loop(monkeypatch, clock)
    env, _ = _restarting(AsyncMock(), policy="no")
    recover = AsyncMock()
    starts = bootstrap._Starts(env, {"side": "cid"}, {"side": 0.0}, recover)
    await bootstrap._settle_restarts(env, {"side": "cid"}, {"side"}, starts)
    env._inspect_container.assert_not_awaited()
    recover.assert_not_awaited()


async def test_a_crash_loop_at_start_fails_the_start(monkeypatch):
    clock = _Clock()
    _fake_loop(monkeypatch, clock)

    async def inspect(cid):
        return {"State": {"Running": False, "ExitCode": 1}, "RestartCount": 0}

    env, _ = _restarting(inspect, policy="always")
    monkeypatch.setattr(bootstrap, "wait_ready", AsyncMock())
    monkeypatch.setattr(
        "benchflow.sandbox.docker.route_exec_argv",
        lambda route, script, container, user: ["exec", container, script],
    )
    recover = AsyncMock()
    starts = bootstrap._Starts(env, {"side": "cid"}, {"side": 0.0}, recover)
    with pytest.raises(RuntimeError) as caught:
        await bootstrap._settle_restarts(env, {"side": "cid"}, {"side"}, starts)
    # The loop stops at the limit, and says so with the count.
    limit = bootstrap.RESTART_LIMIT
    assert str(caught.value) == (
        f"Task service 'side' restarted {limit + 1} times during startup; "
        "inspect the task logs"
    ), caught.value
    assert limit == 5 and recover.await_count == limit
    assert len(env.started) == limit


# ------------------------------------------------- a restart later in the trial


async def test_a_restart_after_the_start_is_named():
    async def inspect(cid):
        return {
            "State": {"Running": True, "StartedAt": "2026-11-20T23:00:00Z"},
            "RestartCount": 2 if cid == "side-id" else 0,
        }

    env = SimpleNamespace(
        _inspect_container=inspect,
        _held_restart_base={"side": ("side-id", 1), "db": ("db-id", 0)},
    )
    assert await bootstrap.held_restarts(env) == [
        {
            "service": "side",
            "restarts": 1,
            "restarted_at": "2026-11-20T23:00:00Z",
            "why": "held after a restart; its metadata rule can't be re-proven",
        }
    ]


def _restarted_env():
    async def inspect(cid):
        return {"State": {"StartedAt": "2026-11-20T23:00:00Z"}, "RestartCount": 1}

    return SimpleNamespace(
        _inspect_container=inspect, _held_restart_base={"side": ("side-id", 0)}
    )


@pytest.mark.parametrize(
    ("exit_code", "reward"),
    [(1, 0.0), (0, 0.5), (0, 1.0)],
    ids=["test-fails", "partial", "test-passes"],
)
async def test_a_held_restart_never_unscores_a_trial(exit_code, reward):
    """A cause inside the task's containers
    never turns a trial unscored. The score stands as test.sh gave it, and the
    service is named for result.json and the trial's warnings (its read
    flag)."""
    from benchflow.rollout._setup import (
        _held_after_restart,
        held_after_restart_warning,
    )

    env = _restarted_env()
    rewards, error = await _held_after_restart(
        env, SimpleNamespace(test_return_code=exit_code), {"reward": reward}, None
    )
    assert rewards == {"reward": reward} and error is None
    assert [h["service"] for h in env._held_after_restart] == ["side"]
    said = held_after_restart_warning(env._held_after_restart)
    assert "'side' (restarted at 2026-11-20T23:00:00Z" in said, said
    assert "scored as test.sh says: read it" in said, said


async def test_an_earlier_ending_stays_and_no_restart_changes_nothing():
    from benchflow.rollout._setup import _held_after_restart

    env = _restarted_env()
    assert await _held_after_restart(
        env, SimpleNamespace(test_return_code=1), None, "verifier crashed: x"
    ) == (None, "verifier crashed: x")
    quiet = SimpleNamespace()
    assert await _held_after_restart(
        quiet, SimpleNamespace(test_return_code=1), {"reward": 0.0}, None
    ) == ({"reward": 0.0}, None)


async def test_a_restart_s_exit_status_comes_from_docker_s_events():
    calls = []

    async def docker(argv, **_k):
        calls.append(argv)
        return SimpleNamespace(return_code=0, stdout="0\n137\n1\n")

    async def inspect(cid):
        return {"State": {"StartedAt": "t"}, "RestartCount": 1}

    env = SimpleNamespace(
        _inspect_container=inspect,
        _docker_cli=docker,
        _held_restart_base={"side": ("side-id", 0)},
        _held_restart_since=1_700_000_000,
    )
    held = await bootstrap.held_restarts(env)
    assert held[0]["exit_code"] == 1, held
    argv = calls[0]
    assert argv[0] == "events" and "container=side-id" in argv and "event=die" in argv


def test_the_trial_s_warnings_carry_the_read_flag_once():
    from benchflow.rollout import _with_held_after_restart

    held = [{"service": "side", "restarted_at": "t", "exit_code": 1}]
    once = _with_held_after_restart(["earlier"], held)
    assert once[0] == "earlier" and "held after a restart" in once[1], once
    assert _with_held_after_restart(once, held) == once
    assert _with_held_after_restart(["x"], None) == ["x"]


# ------------------------------------------- a compose file under another name


@pytest.mark.parametrize("name", ["docker-compose.yml", "compose.yaml", "compose.yml"])
def test_a_compose_file_under_another_name_is_named_as_not_read(tmp_path, name):
    """Docker stacks only environment/docker-compose.yaml; another name was
    ignored without a word (the task ran as main alone). 0.10.1 says so at
    start and in the trial's warnings; reading it is a later release's."""
    from benchflow._utils.trial_warnings import recording_trial_warnings
    from benchflow.sandbox._compose import unread_compose_file, unread_compose_warning

    env = tmp_path / "environment"
    env.mkdir()
    (env / name).write_text("services:\n  main: {}\n  side: {image: alpine}\n")
    found = unread_compose_file(env)
    assert found == env / name
    said = unread_compose_warning(found)
    assert f"environment/{name} is not read" in said, said
    assert "only environment/docker-compose.yaml" in said and "main alone" in said
    (env / "docker-compose.yaml").write_text("services:\n  main: {}\n")
    assert unread_compose_file(env) is None
    with recording_trial_warnings() as warnings:
        from benchflow._utils.trial_warnings import report_trial_warning

        report_trial_warning(said)
    assert warnings == [said]


def test_the_docker_factory_names_an_unread_compose_file(tmp_path, monkeypatch, caplog):
    """The sandbox factory, for Docker, warns and records it before the
    sandbox is made (the factory itself is stopped right after)."""
    import logging

    from benchflow._utils.trial_warnings import recording_trial_warnings
    from benchflow.sandbox import setup

    task = tmp_path / "task"
    for sub in ("environment", "tests", "solution"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\ndocker_image = "python:3.12-slim"\n'
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    (task / "environment" / "docker-compose.yml").write_text(
        "services:\n  main: {}\n  side: {image: alpine}\n"
    )

    class _Made(RuntimeError):
        pass

    def stop(*_a, **_k):
        raise _Made("the factory went on to stage the sandbox")

    from benchflow.task import RolloutPaths, Task

    monkeypatch.setattr(setup, "_stage_environment", stop)
    caplog.set_level(logging.WARNING)
    with recording_trial_warnings() as warnings, pytest.raises(_Made):
        setup._create_sandbox_environment(
            "docker",
            Task(task),
            task,
            "t",
            RolloutPaths(rollout_dir=tmp_path / "trial"),
        )
    assert any("docker-compose.yml is not read" in w for w in warnings), warnings
    assert "docker-compose.yml is not read" in caplog.text


async def test_the_end_of_turn_re_proof_lets_a_held_restart_be_scored():
    """A restart in the agent's turn: the held container's new namespace has
    none of its rules, so the re-proof at the turn's end can't prove them.
    None of its processes ran, and the cause is inside the task's containers:
    no ending (the trial is scored as test.sh says); the service is named,
    with what the re-proof found."""
    from benchflow.sandbox.policy_reproof import Reproof, with_held_restarts

    env = _restarted_env()
    proof = Reproof(
        at="the end of the agent's turn",
        checked_at="2026-11-20T23:01:00Z",
        breaches=["service side (start): 10 of 10 rules missing"],
    )
    assert await with_held_restarts(env, proof) is None
    held = env._held_after_restart
    assert [h["service"] for h in held] == ["side"]
    assert (
        "not proven (service side (start): 10 of 10 rules missing)"
        in held[0]["reproof"]
    ), held


async def test_a_breach_elsewhere_stays_a_breach_with_the_restart_named():
    from benchflow.sandbox.policy_reproof import Reproof, with_held_restarts
    from benchflow.task.verifier_errors import (
        NetworkPolicyBreachedError,
        NetworkPolicyUnprovenError,
    )

    proof = Reproof(
        at="the end of the agent's turn",
        checked_at="x",
        breaches=[
            "service side (start): 10 of 10 rules missing",
            "main (agent): 2 of 6 rules missing",
        ],
    )
    error = await with_held_restarts(_restarted_env(), proof)
    assert isinstance(error, NetworkPolicyBreachedError)
    assert not isinstance(error, NetworkPolicyUnprovenError)
    assert "'side'" in str(error) and "held after a restart" in str(error)
    plain = await with_held_restarts(SimpleNamespace(), proof)
    assert "held after a restart" not in str(plain)


def test_the_rollout_keeps_the_unread_compose_warning_for_result_json(
    tmp_path, monkeypatch
):
    """The warning reaches the trial's record, not only the log: the rollout
    keeps what the sandbox factory reports (on the first version it was
    logged and lost; react-performance-debugging's result.json had none)."""
    import json
    from pathlib import Path

    import benchflow as bf
    from benchflow.sandbox import setup

    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    task = tmp_path / "task"
    for sub in ("environment", "tests", "solution"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\ndocker_image = "python:3.12-slim"\n'
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    for path in (task / "tests" / "test.sh", task / "solution" / "solve.sh"):
        path.write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
        path.chmod(0o755)
    (task / "environment" / "docker-compose.yml").write_text(
        "services:\n  main: {}\n  side: {image: alpine}\n"
    )

    def stop(*_a, **_k):
        raise RuntimeError("stopped after the factory's checks")

    monkeypatch.setattr(setup, "_stage_environment", stop)
    result = bf.run_sync(
        bf.RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            jobs_dir=tmp_path / "jobs",
        )
    )
    recorded = json.loads((Path(result.rollout_dir) / "result.json").read_text())
    warnings = recorded.get("warnings") or []
    assert any("docker-compose.yml is not read" in w for w in warnings), recorded.get(
        "error"
    )


# ---------------------------------------------- BenchFlow's waiter (bfx hold)


class _Covering(SimpleNamespace):
    """A backend that puts rules into its services (Docker, remote Docker)."""

    async def metadata_service_targets(self, **_k):
        return []


def _prototype(name: str, *, entry=None, cmd=None, user="", binds=False):
    return {
        "Id": f"{name}-id",
        "State": {"Running": False},
        "Config": {
            "Labels": {"com.docker.compose.service": name},
            "Entrypoint": entry,
            "Cmd": cmd or ["python3", "-m", "http.server"],
            "Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "LD_PRELOAD=/x.so"],
            "User": user,
        },
        "Mounts": [],
    }


async def _overlay(monkeypatch, env, prototypes):
    async def inspect(cid):
        return prototypes[cid]

    env._inspect_container = inspect
    monkeypatch.setattr(bootstrap, "protected_services", AsyncMock(return_value=set()))
    project = {
        "services": {
            p["Config"]["Labels"]["com.docker.compose.service"]: {}
            for p in prototypes.values()
        }
    }
    return await bootstrap.overlay(env, project, list(prototypes))


async def test_every_held_service_waits_in_bfx_and_nothing_of_the_image_runs(
    monkeypatch,
):
    """Before, the hold ran the image's /bin/sh and looped on its
    /bin/sleep before the service's rule. Now each held service's PID 1 is
    BenchFlow's waiter (main included), mounted read-only into the sidecar.
    The held container's environment has the loader variables cleared (every
    docker exec BenchFlow makes into it inherits it); the service's own
    process gets its values back at the release, from bfx's --setenv."""
    from benchflow._bfx import delivery

    env = _Covering(
        bfx_delivery=delivery.VOLUME,
        _bfx_source="benchflow-bfx-abc",
        _hold_every_service=True,
        _main_metadata_rule=True,
    )
    prototypes = {
        "main-id": _prototype("main", cmd=["sh", "-c", "sleep infinity"]),
        "side-id": _prototype("side", entry=["/srv/run"], cmd=["--port", "8080"]),
    }
    document = await _overlay(monkeypatch, env, prototypes)
    side = document["services"]["side"]
    token = env._hold_tokens["side"]
    assert side["entrypoint"] == [
        "/.bfx/bfx",
        "hold",
        "--token",
        token,
        "--setenv",
        "LD_PRELOAD=/x.so",
        "--",
        "/srv/run",
        "--port",
        "8080",
    ], side
    assert side["command"] == [] and side["healthcheck"] == {"disable": True}
    assert side["environment"] == bootstrap.EMPTY_ENV, side
    assert side["volumes"] == [
        delivery.mount(delivery.VOLUME, source="benchflow-bfx-abc")
    ]
    main = document["services"]["main"]
    assert main["entrypoint"][:2] == ["/.bfx/bfx", "hold"] and "volumes" not in main
    assert env._hold_tokens["main"] != token
    assert all(
        "/bin/sh" not in arg
        for s in document["services"].values()
        for arg in s["entrypoint"][:5]
    )


async def test_a_held_service_without_loader_values_runs_its_own_argv(monkeypatch):
    """No loader or start-up value to give back (the usual image, a
    distroless one included): the waiter execs the service's own process,
    nothing of the image's in between."""
    from benchflow._bfx import delivery

    env = _Covering(
        bfx_delivery=delivery.VOLUME,
        _bfx_source="benchflow-bfx-abc",
        _hold_every_service=True,
    )
    clean = _prototype("side", entry=["/srv/run"])
    clean["Config"]["Env"] = ["PATH=/usr/local/bin:/usr/bin:/bin"]
    document = await _overlay(
        monkeypatch, env, {"main-id": _prototype("main"), "side-id": clean}
    )
    side = document["services"]["side"]
    assert side["entrypoint"][5:] == ["/srv/run", "python3", "-m", "http.server"]
    assert side["environment"] == bootstrap.EMPTY_ENV


async def test_a_covering_backend_without_bfx_refuses_rather_than_run_the_image(
    monkeypatch,
):
    from benchflow.sandbox.service_cover import ServiceCoverError

    env = _Covering(
        bfx_delivery=None, _hold_every_service=True, _main_metadata_rule=True
    )
    with pytest.raises(ServiceCoverError) as caught:
        await _overlay(
            monkeypatch,
            env,
            {"main-id": _prototype("main"), "side-id": _prototype("side")},
        )
    said = str(caught.value)
    assert (
        "only BenchFlow's own waiter (bfx)" in said
        and "never runs the image's own shell" in said
    ), said


@pytest.mark.parametrize(
    "rules",
    [
        {"_hold_every_service": True},
        {"_main_metadata_rule": True},
        {"_start_hold_mode": "no-network"},
    ],
    ids=["every-service", "main-metadata", "policy"],
)
async def test_any_wait_for_rules_without_bfx_is_refused(monkeypatch, rules):
    from benchflow.sandbox.service_cover import ServiceCoverError

    env = _Covering(bfx_delivery=None, **rules)
    with pytest.raises(ServiceCoverError, match="next: deliver bfx"):
        await _overlay(monkeypatch, env, {"main-id": _prototype("main")})


async def test_a_bind_proof_alone_keeps_the_image_shell_hold_without_bfx(monkeypatch):
    """An open network whose cloud metadata opt-out the run granted: main
    waits only for its bind proof, no rule, so without bfx it keeps the hold
    that predates the waiter, as on 0.10.0 (BENCHFLOW_BFX_DELIVERY=off)."""
    env = _Covering(
        bfx_delivery=None,
        _main_metadata_rule=False,
        _docker_cli=AsyncMock(return_value=SimpleNamespace(return_code=0)),
    )
    prototypes = {"main-id": _prototype("main")}

    async def inspect(cid):
        return prototypes[cid]

    env._inspect_container = inspect
    monkeypatch.setattr(
        bootstrap, "protected_services", AsyncMock(return_value={"main"})
    )
    document = await bootstrap.overlay(env, {"services": {"main": {}}}, ["main-id"])
    main = document["services"]["main"]
    assert main["entrypoint"][:3] == ["/bin/sh", "-c", bootstrap.BOOTSTRAP]
    assert env._hold_tokens == {}


async def test_a_backend_without_rules_or_bfx_keeps_the_image_shell_hold(monkeypatch):
    """Daytona's DinD (bfx opt-in there, and no rule goes into its services):
    the hold that predates the waiter, as on 0.10.0."""
    env = SimpleNamespace(
        bfx_delivery=None,
        _main_metadata_rule=True,
        _docker_cli=AsyncMock(return_value=SimpleNamespace(return_code=0)),
    )
    document = await _overlay(monkeypatch, env, {"main-id": _prototype("main")})
    main = document["services"]["main"]
    assert main["entrypoint"][:3] == ["/bin/sh", "-c", bootstrap.BOOTSTRAP]
    assert main["environment"] == bootstrap.EMPTY_ENV
    assert env._hold_tokens == {}


async def test_readiness_and_release_ask_bfx_by_the_waiter_s_token(monkeypatch):
    calls = []
    answers = iter([1, 1, 0])

    async def docker(argv, **_k):
        calls.append(argv)
        if argv[-2] == "--ready":
            return SimpleNamespace(return_code=next(answers))
        return SimpleNamespace(return_code=0)

    clock = _Clock()
    _fake_loop(monkeypatch, clock)
    env = SimpleNamespace(_docker_cli=docker, _hold_tokens={"side": "t" * 32})
    await bootstrap.wait_ready(env, "side-id", token="t" * 32)
    await bootstrap._signal_release(env, "side-id", "side")
    assert calls == [
        ["exec", "-u", "0", "side-id", "/.bfx/bfx", "hold", "--ready", "t" * 32],
        ["exec", "-u", "0", "side-id", "/.bfx/bfx", "hold", "--ready", "t" * 32],
        ["exec", "-u", "0", "side-id", "/.bfx/bfx", "hold", "--ready", "t" * 32],
        ["exec", "-u", "0", "side-id", "/.bfx/bfx", "hold", "--release", "t" * 32],
    ], calls
    assert not any(c[0] == "kill" for c in calls)


@pytest.mark.parametrize(
    ("token", "cid", "status", "waits"),
    [
        ("t" * 32, "side-id", 0, True),
        ("t" * 32, "side-id", 1, False),
        ("t" * 32, None, 0, False),
        (None, "side-id", 0, False),
    ],
    ids=["holds", "released", "no-container", "never-held"],
)
async def test_whether_a_service_waits_in_the_hold_is_bfx_s_answer(
    token, cid, status, waits
):
    """After a command in a held service exits as bfx does when it ran
    nothing, bfx itself is asked (by the container's id, on the waiter's
    route), never the service's image."""
    calls = []

    async def docker(argv, **_k):
        calls.append(argv)
        return SimpleNamespace(return_code=status)

    async def container_id(service):
        return cid

    env = SimpleNamespace(
        _docker_cli=docker,
        _service_container_id=container_id,
        _hold_tokens={"side": token} if token else {},
    )
    assert await bootstrap.waits_in_hold(env, "side") is waits
    asked = ["exec", "-u", "0", "side-id", "/.bfx/bfx", "hold", "--ready", token]
    assert calls == ([asked] if token and cid else []), calls


@pytest.mark.parametrize("status", [2, 3, 127])
async def test_a_waiter_that_answers_otherwise_fails_the_start(monkeypatch, status):
    async def docker(argv, **_k):
        return SimpleNamespace(return_code=status)

    clock = _Clock()
    _fake_loop(monkeypatch, clock)
    env = SimpleNamespace(_docker_cli=docker, _hold_tokens={"side": "t" * 32})
    with pytest.raises(RuntimeError, match="waiter did not become ready"):
        await bootstrap.wait_ready(env, "side-id", token="t" * 32)
    with pytest.raises(
        RuntimeError, match="Cannot release service 'side' from BenchFlow's waiter"
    ):
        await bootstrap._signal_release(env, "side-id", "side")


async def test_a_held_healthcheck_that_never_passes_stops_at_the_start_up_budget(
    monkeypatch,
):
    """A check that never passes (an hour's interval) would hold the
    start as long as Docker waits; it stops at the environment's own
    start-up budget, with a named error (no agent code has run)."""
    from benchflow.sandbox.protocol import EnvironmentStartTimeoutError

    clock = _Clock()
    _fake_loop(monkeypatch, clock)

    async def probe(*_a, **_k):
        return SimpleNamespace(return_code=1)

    env = SimpleNamespace(_docker_cli=probe)
    with pytest.raises(EnvironmentStartTimeoutError) as caught:
        await bootstrap.wait_healthy(
            env, "web", _health(Interval=3_600_000_000_000), released_at=0.0, cap=600.0
        )
    said = str(caught.value)
    assert "'web' did not pass within 600s of its release" in said, said
    assert "environment.build_timeout_sec" in said and clock.now <= 600.0
    assert (
        bootstrap._health_cap(
            SimpleNamespace(task_env_config=SimpleNamespace(build_timeout_sec=600.0))
        )
        == 600.0
    )
    assert bootstrap._health_cap(SimpleNamespace()) is None


def test_pid_sharing_the_operator_grants_is_named_in_the_trial(tmp_path, monkeypatch):
    """Pid sharing with host access: the operator granted it, so the factory names
    it in the trial's warnings."""
    from benchflow._utils.trial_warnings import recording_trial_warnings
    from benchflow.sandbox import setup
    from benchflow.task import RolloutPaths, Task

    task = tmp_path / "task"
    for sub in ("environment", "tests", "solution"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\ndocker_image = "python:3.12-slim"\n'
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main: {}\n  side: {image: alpine}\n"
        "  other: {image: alpine, pid: 'service:side'}\n"
    )
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")

    def stop(*_a, **_k):
        raise RuntimeError("stopped after the factory's checks")

    monkeypatch.setattr(setup, "_stage_environment", stop)
    with recording_trial_warnings() as warnings, pytest.raises(RuntimeError):
        setup._create_sandbox_environment(
            "docker",
            Task(task),
            task,
            "t",
            RolloutPaths(rollout_dir=tmp_path / "trial"),
        )
    assert any(
        "other joins side" in w and "BENCHFLOW_ALLOW_TASK_HOST_ACCESS" in w
        for w in warnings
    ), warnings


@pytest.mark.parametrize(
    ("running", "ready", "raises"),
    [
        (False, 1, False),
        (True, 1, False),
        (True, 0, True),
        (True, 2, True),
        (True, 3, True),
        (True, 127, True),
        (None, 1, True),
    ],
    ids=[
        "container-exited",
        "released",
        "still-waiting",
        "usage",
        "two-waiters",
        "no-bfx",
        "inspect-fails",
    ],
)
async def test_a_release_whose_exec_died_with_its_container_counts(
    monkeypatch, running, ready, raises
):
    """A program that exits at once after its release takes the container,
    and the release's own ``docker exec`` with it: that release happened.
    A waiter still there, unreleased, fails the start, and so does anything
    that can't tell: an answer other than bfx's "no waiter" (1), or an
    inspect that fails."""

    async def docker(argv, **_k):
        if argv[-2] == "--release":
            return SimpleNamespace(return_code=137)
        return SimpleNamespace(return_code=ready)

    async def inspect(cid):
        if running is None:
            raise RuntimeError("docker inspect 'main-id' failed: daemon busy")
        return {"State": {"Running": running}}

    env = SimpleNamespace(
        _docker_cli=docker, _inspect_container=inspect, _hold_tokens={"main": "t" * 32}
    )
    if raises:
        with pytest.raises(RuntimeError, match="Cannot release service 'main'"):
            await bootstrap._signal_release(env, "main-id", "main")
    else:
        await bootstrap._signal_release(env, "main-id", "main")


# ------------------------------------- a held service that exited, not restarting


def _exited_env(*, running=False, restarting=False, restarts=0, exit_code=1):
    async def inspect(cid):
        return {
            "State": {
                "Running": running,
                "Restarting": restarting,
                "ExitCode": exit_code,
                "StartedAt": "2026-11-21T07:00:00Z",
                "FinishedAt": "2026-11-21T07:00:05Z",
            },
            "RestartCount": restarts,
        }

    return SimpleNamespace(
        _inspect_container=inspect, _held_restart_base={"side": ("side-id", 0)}
    )


@pytest.mark.parametrize(
    ("state", "exited", "restarted"),
    [
        ({}, ["side"], []),
        ({"running": True}, [], []),
        ({"restarting": True}, [], ["side"]),
        ({"restarts": 1}, [], ["side"]),
    ],
    ids=["exited", "running", "restart-pending", "restarted"],
)
async def test_an_exited_service_is_told_from_a_restarting_one(
    state, exited, restarted
):
    """Exited and not starting again (no restart policy): held_exited. One
    Docker is waiting out its restart delay for, or has restarted, waits in
    the hold again: held_restarts."""
    env = _exited_env(**state)
    assert [e["service"] for e in await bootstrap.held_exited(env)] == exited
    held = await bootstrap.held_restarts(env)
    assert [h["service"] for h in held] == restarted
    if state.get("restarting"):
        assert held[0]["restart_pending"] is True


@pytest.mark.parametrize(
    ("before", "at", "phase"),
    [
        ({"side"}, "the end of the agent's turn", "start-up"),
        (set(), "the end of the agent's turn", "the agent's turn"),
        (set(), "the end of test.sh", "test.sh"),
    ],
    ids=["one-shot-at-start-up", "in-the-turn", "in-test-sh"],
)
async def test_an_exited_service_s_unproven_rules_never_unscore_the_trial(
    before, at, phase
):
    """It has no network namespace and runs nothing: its rows the re-proof
    could not check are no ending, at either re-proof. It is recorded with
    its exit status and when it exited; only an exit after start-up is
    flagged for a read."""
    from benchflow.rollout._setup import held_exited_warning
    from benchflow.sandbox.policy_reproof import Reproof, with_held_restarts

    env = _exited_env()
    env._held_exited_before_turn = before
    unproven = (
        "service side (start): not proven (the metadata firewall helper did not start)"
    )
    proof = Reproof(at=at, checked_at="x", unproven=[unproven])
    assert await with_held_restarts(env, proof) is None
    (record,) = env._held_exited
    assert (record["service"], record["exit_code"], record["phase"]) == (
        "side",
        1,
        phase,
    )
    assert unproven in record["reproof"]
    flag = held_exited_warning(env._held_exited)
    if phase == "start-up":
        assert flag is None
    else:
        assert flag and f"in {phase}, exit 1" in flag and "read it" in flag


async def test_an_exit_keeps_the_phase_it_was_first_seen_in():
    """Seen at the end of the agent's turn, it stays the turn's at the end of
    test.sh."""
    from benchflow.sandbox.policy_reproof import Reproof, with_held_restarts

    env = _exited_env()
    row = "service side (start): not proven (the helper did not start)"
    for at in ("the end of the agent's turn", "the end of test.sh"):
        proof = Reproof(at=at, checked_at="x", unproven=[row])
        assert await with_held_restarts(env, proof) is None
    assert [e["phase"] for e in env._held_exited] == ["the agent's turn"]


async def test_a_breach_elsewhere_stays_the_ending_with_the_exit_named():
    from benchflow.sandbox.policy_reproof import Reproof, with_held_restarts
    from benchflow.task.verifier_errors import NetworkPolicyBreachedError

    proof = Reproof(
        at="the end of the agent's turn",
        checked_at="x",
        unproven=["service side (start): not proven (the helper did not start)"],
        breaches=["main (agent): 2 of 6 rules missing"],
    )
    error = await with_held_restarts(_exited_env(), proof)
    assert isinstance(error, NetworkPolicyBreachedError)
    assert "'side' exited and did not start again" in str(error)
    assert error.record["held_exited"][0]["service"] == "side"


async def test_the_exits_before_the_turn_are_kept_once():
    env = _exited_env()
    await bootstrap.note_exits_before_turn(env)
    assert env._held_exited_before_turn == {"side"}
    env._held_restart_base = {"side": ("side-id", 0), "other": ("other-id", 0)}
    await bootstrap.note_exits_before_turn(env)
    assert env._held_exited_before_turn == {"side"}


def test_the_rollout_flags_an_exit_in_the_turn_and_only_records_one_at_start_up():
    from benchflow.rollout import _with_held_after_restart

    turn = [{"service": "side", "phase": "the agent's turn", "exit_code": 1}]
    start = [{"service": "side", "phase": "start-up", "exit_code": 0}]
    (line,) = _with_held_after_restart(["earlier"], None, turn)[1:]
    assert "'side' (in the agent's turn, exit 1) exited" in line
    assert _with_held_after_restart(["earlier"], None, start) == ["earlier"]


async def test_an_exit_with_a_restart_pending_follows_the_held_restart_path():
    """on-failure: Docker waits out its restart delay before the waiter starts
    again. The re-proof meets the container down; it is a held restart, not
    an exit, and the trial is scored."""
    from benchflow.sandbox.policy_reproof import Reproof, with_held_restarts

    env = _exited_env(restarting=True)
    row = "service side (start): not proven (the helper did not start)"
    proof = Reproof(at="the end of the agent's turn", checked_at="x", unproven=[row])
    assert await with_held_restarts(env, proof) is None
    assert [h["service"] for h in env._held_after_restart] == ["side"]
    assert env._held_after_restart[0]["restart_pending"] is True
    assert not getattr(env, "_held_exited", None)


def test_a_failed_helper_start_says_docker_s_reason_not_its_usage_hint():
    from benchflow.sandbox._metadata_helper import _docker_reason

    out = (
        "docker: Error response from daemon: cannot join network namespace of a "
        "non running container: container side-id is exited.\n"
        "See 'docker run --help'.\n"
    )
    assert _docker_reason(out).startswith("docker: Error response from daemon")
    assert _docker_reason("Run 'docker run --help' for more information\n") == ""


async def test_a_check_that_could_not_run_in_a_stopped_service_says_so():
    """The re-proof names a stopped container's state, never Docker's raw
    text: its network namespace is gone with it."""
    from benchflow.sandbox import policy_reproof

    async def install(container, command):
        return (
            None,
            "the metadata firewall helper did not start: See 'docker run --help'.",
        )

    async def inspect(cid):
        return {"State": {"Running": False, "Status": "exited", "ExitCode": 1}}

    env = SimpleNamespace(metadata_service_install=install, _inspect_container=inspect)
    out, why = await policy_reproof._in_namespace(env, "side-id", "true")
    assert out == ""
    assert why == (
        "its container is not running (exited, exit 1), so the helper could not join it"
    ), why
    assert "--help" not in why


async def test_the_interactive_shell_s_hooks_are_cleared_for_root_execs(monkeypatch):
    """PROMPT_COMMAND, PS0, PS1 and PS4 from the image or the task's Compose
    env: a held container's environment (every root exec's) does not carry
    them, and the service's own process gets them back at its release."""
    from benchflow._bfx import delivery

    env = _Covering(
        bfx_delivery=delivery.VOLUME,
        _bfx_source="benchflow-bfx-abc",
        _hold_every_service=True,
    )
    side = _prototype("side", entry=["/srv/run"])
    hooks = ["PROMPT_COMMAND=. /tmp/pc.sh", "PS0=$(id)", "PS1=$(id)", "PS4=$(id)"]
    side["Config"]["Env"] = ["PATH=/usr/bin:/bin", *hooks]
    document = await _overlay(
        monkeypatch, env, {"main-id": _prototype("main"), "side-id": side}
    )
    held = document["services"]["side"]
    for name in ("PROMPT_COMMAND", "PS0", "PS1", "PS4"):
        assert name in held["environment"] and held["environment"][name] is None
    assert bootstrap.given_back(side["Config"]) == sorted(hooks)


def _stopped_service(logs: str):
    async def inspect(cid):
        return {"State": {"Running": False, "ExitCode": 127}}

    async def docker(argv, **_k):
        assert argv[:3] == ["logs", "--tail", "20"]
        return SimpleNamespace(return_code=0, stdout="", stderr=logs)

    env = SimpleNamespace(
        _inspect_container=inspect, _docker_cli=docker, _hold_tokens={"side": "t" * 32}
    )
    config = {
        "Labels": {"com.docker.compose.service": "side"},
        "Entrypoint": ["/srv/run"],
        "Cmd": ["--port", "8080"],
        "Env": [],
    }
    return env, config


async def test_a_service_whose_program_did_not_start_says_so():
    """bfx's own line, naming the program BenchFlow gave the waiter, says
    which program did not start at the release, and why."""
    env, config = _stopped_service(
        "bfx hold: cannot start /srv/run: No such file or directory\n"
    )
    with pytest.raises(RuntimeError) as caught:
        await bootstrap.wait_healthy(env, "side-id", config)
    said = str(caught.value)
    assert "Task service 'side' exited during startup" in said
    assert "bfx could not start '/srv/run': No such file or directory" in said


async def test_a_spoofed_waiter_line_is_shown_as_the_container_s_own_output():
    """A program can print bfx's words: a line naming any other program is
    the container's output, labelled so, never bfx's reason."""
    env, config = _stopped_service(
        "bfx hold: cannot start /usr/bin/env: No such file or directory\n"
    )
    with pytest.raises(RuntimeError) as caught:
        await bootstrap.wait_healthy(env, "side-id", config)
    said = str(caught.value)
    assert "bfx could not start" not in said
    assert "exit 127" in said and "not BenchFlow's" in said
    assert "'bfx hold: cannot start /usr/bin/env: No such file or directory'" in said
