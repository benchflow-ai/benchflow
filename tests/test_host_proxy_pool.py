"""The job's warm host proxies (providers.litellm_runtime._HostProxyPool).

A trial asks for its host proxy when its sandbox starts and takes it when its
agent connects; the proxy is still the trial's own (its key, port, log), and
only a proxy started from exactly the trial's launch is handed out.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

import benchflow.rollout as rollout_mod
from benchflow._utils import timing_spans
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime
from tests.test_install_layer_eligibility import _task
from tests.test_proxy_layer_use import _DockerLike, _trial


class _HostProxyDocker(_DockerLike):
    """A Docker-like sandbox whose agents reach a host proxy."""

    starts_host_proxy_ahead = True


AGENT = "claude-agent-acp"
MODEL = "claude-sonnet-4-6"
ENV = {"ANTHROPIC_API_KEY": "sk-ant-test-marker"}


class FakeServer:
    def __init__(self, n: int, kwargs: dict[str, Any]) -> None:
        self.base_url = f"http://127.0.0.1:{30000 + n}"
        self.route = kwargs["route"]
        self.session_id = kwargs["session_id"]
        self.agent_name = kwargs["agent_name"]
        self.stopped = False
        self.trajectory = None

    async def is_running(self) -> bool:
        return not self.stopped

    async def stop(self) -> None:
        self.stopped = True

    def start_live_capture(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.fixture
def pool(monkeypatch):
    fresh = runtime_mod._HostProxyPool()
    monkeypatch.setattr(runtime_mod, "_HOST_PROXY_POOL", fresh)
    monkeypatch.delenv(runtime_mod.HOST_PROXY_WARM_ENV, raising=False)
    monkeypatch.delenv(runtime_mod.HOST_PROXY_AHEAD_ENV, raising=False)
    return fresh


@pytest.fixture
def starts(monkeypatch):
    started: list[dict[str, Any]] = []
    servers: list[FakeServer] = []

    async def fake_start(**kwargs):
        started.append(kwargs)
        server = FakeServer(len(started), kwargs)
        servers.append(server)
        return server

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    return started, servers


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _reserve(env: dict[str, str] | None = None):
    return runtime_mod.reserve_host_proxy(
        agent=AGENT,
        agent_env=dict(env or ENV),
        model=MODEL,
        environment="docker",
    )


async def _connect(reservation, env: dict[str, str] | None = None, session="run-a"):
    with runtime_mod.connecting_with(reservation):
        return await ensure_litellm_runtime(
            agent=AGENT,
            agent_env=dict(env or ENV),
            model=MODEL,
            runtime=None,
            environment="docker",
            session_id=session,
        )


@pytest.mark.asyncio
async def test_reserved_proxy_starts_before_the_connection_and_is_taken(pool, starts):
    started, servers = starts
    reservation = _reserve()
    assert reservation is not None
    await _settle()
    # Started when the sandbox started, before any connection.
    assert len(started) == 1

    updated, runtime = await _connect(reservation, session="run-7")

    assert runtime.server is servers[0]
    # The trial's own key is the one the proxy started with.
    assert runtime.master_key == started[0]["master_key"]
    assert runtime.master_key in updated.values()
    assert servers[0].session_id == "run-7"
    assert reservation.answered
    await _settle()
    # None kept beyond the trials that asked (BENCHFLOW_HOST_PROXY_WARM).
    assert len(started) == 1


@pytest.mark.asyncio
async def test_warm_keeps_the_next_trials_proxy_ready(pool, starts, monkeypatch):
    started, servers = starts
    monkeypatch.setenv(runtime_mod.HOST_PROXY_WARM_ENV, "1")
    reservation = _reserve()
    await _settle()

    _, runtime = await _connect(reservation)
    await _settle()

    assert runtime.server is servers[0]
    # The next trial's proxy, with a key of its own.
    assert len(started) == 2
    assert started[1]["master_key"] != started[0]["master_key"]
    assert not servers[1].stopped


@pytest.mark.asyncio
async def test_each_trial_gets_a_proxy_of_its_own(pool, starts):
    started, _ = starts
    first, second = _reserve(), _reserve()
    await _settle()
    assert len(started) == 2

    _, one = await _connect(first, session="run-a")
    _, two = await _connect(second, session="run-b")

    assert one.server is not two.server
    assert one.master_key != two.master_key
    assert {one.server.session_id, two.server.session_id} == {"run-a", "run-b"}


@pytest.mark.asyncio
async def test_another_launch_starts_fresh_and_later_trials_ask_for_it(pool, starts):
    started, servers = starts
    reservation = _reserve()
    await _settle()
    other = {"ANTHROPIC_API_KEY": "sk-ant-other-marker"}

    _, runtime = await _connect(reservation, env=other)

    # The reserved proxy holds the other key: never handed to this trial.
    assert runtime.server is not servers[0]
    fresh = started[1]
    assert fresh["agent_env"]["ANTHROPIC_API_KEY"] == "sk-ant-other-marker"
    assert runtime.master_key == fresh["master_key"]
    await _settle()
    before = len(started)
    _reserve()
    _reserve()
    await _settle()
    # The job learned that its trials connect with the other launch: what it
    # starts for them now is that one.
    new = started[before:]
    assert new
    assert all(
        s["agent_env"]["ANTHROPIC_API_KEY"] == "sk-ant-other-marker" for s in new
    )


@pytest.mark.asyncio
async def test_a_proxy_that_died_waiting_is_not_handed_out(pool, starts):
    started, servers = starts
    reservation = _reserve()
    await _settle()
    servers[0].stopped = True

    _, runtime = await _connect(reservation)

    assert runtime.server is not servers[0]
    assert runtime.master_key == started[1]["master_key"]


@pytest.mark.asyncio
async def test_unused_reservations_are_given_back_and_stopped(pool, starts):
    started, servers = starts
    first, second = _reserve(), _reserve()
    await _settle()
    assert len(started) == 2

    runtime_mod.release_host_proxy(first)
    runtime_mod.release_host_proxy(second)
    await _settle()

    assert all(s.stopped for s in servers)


@pytest.mark.asyncio
async def test_warm_keeps_one_of_the_unused_ones(pool, starts, monkeypatch):
    _, servers = starts
    monkeypatch.setenv(runtime_mod.HOST_PROXY_WARM_ENV, "1")
    first, second = _reserve(), _reserve()
    await _settle()

    runtime_mod.release_host_proxy(first)
    runtime_mod.release_host_proxy(second)
    await _settle()

    # One stays ready for the job's next trial; the other is stopped.
    assert sorted(s.stopped for s in servers) == [False, True]
    await runtime_mod.drain_host_proxies()
    assert all(s.stopped for s in servers)


@pytest.mark.asyncio
async def test_the_job_end_stops_every_proxy_no_trial_took(pool, starts, monkeypatch):
    _, servers = starts
    monkeypatch.setenv(runtime_mod.HOST_PROXY_WARM_ENV, "1")
    reservation = _reserve()
    await _settle()
    await _connect(reservation)
    await _settle()
    spare = servers[1]

    await runtime_mod.drain_host_proxies()

    assert spare.stopped
    # The trial's own proxy is the trial's to stop.
    assert not servers[0].stopped


@pytest.mark.asyncio
async def test_ahead_zero_turns_the_pool_off(pool, starts, monkeypatch):
    started, _ = starts
    monkeypatch.setenv(runtime_mod.HOST_PROXY_AHEAD_ENV, "0")

    assert _reserve() is None
    await _settle()
    assert started == []
    await _connect(None)
    await _settle()
    assert len(started) == 1


@pytest.mark.asyncio
async def test_no_reservation_starts_as_before(pool, starts):
    started, _ = starts
    await _connect(None)
    await _settle()
    # No reservation, no pool: one start, at the connection, nothing ahead.
    assert len(started) == 1


@pytest.mark.asyncio
async def test_a_key_of_the_callers_never_takes_a_pool_proxy(pool, starts):
    _, servers = starts
    reservation = _reserve()
    await _settle()
    keyed = {**ENV, runtime_mod.LITELLM_MASTER_KEY_ENV: "sk-callers-own"}

    assert _reserve(keyed) is None
    _, runtime = await _connect(reservation, env=keyed)

    assert runtime.master_key == "sk-callers-own"
    assert runtime.server is not servers[0]


@pytest.mark.asyncio
async def test_spans_go_to_the_trial_that_waits(pool, monkeypatch):
    async def slow_start(**kwargs):
        with timing_spans.span(timing_spans.PROXY_HEALTH_WAIT):
            await asyncio.sleep(0.05)
        return FakeServer(1, kwargs)

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", slow_start)
    asked: dict[str, Any] = {}
    took: dict[str, Any] = {}
    with timing_spans.recording_into(asked):
        reservation = _reserve()
    with timing_spans.recording_into(took):
        await _connect(reservation)

    # The start ran in the pool, not in the trial that asked; the trial that
    # took the proxy records what it waited for.
    assert timing_spans.PROXY_HEALTH_WAIT not in asked
    assert took[timing_spans.PROXY_HEALTH_WAIT] >= 0.0


def test_proxies_left_at_exit_are_stopped_and_their_folders_removed(tmp_path):
    folder = tmp_path / "benchflow-litellm-left"
    folder.mkdir()
    (folder / "config.yaml").write_text("general_settings: {master_key: sk-x}\n")

    class Left:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:
            self.terminated = True

    left = Left()
    processes = cast(dict[Any, Path], runtime_mod._HOST_PROXY_PROCESSES)
    processes[left] = folder
    runtime_mod._stop_host_proxies_at_exit()

    assert left.terminated
    assert not folder.exists()
    assert left not in processes


@pytest.mark.asyncio
async def test_rollout_starts_its_proxy_while_the_sandbox_starts(
    tmp_path: Path, monkeypatch, pool, starts
):
    """The trial's host proxy starts with its sandbox, not after its install."""
    started, _ = starts
    seen_at_sandbox_start: list[int] = []

    async def fake_sandbox_start(*args, **kwargs):
        await _settle()
        seen_at_sandbox_start.append(len(started))

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(rollout_mod, "_start_env_and_upload", fake_sandbox_start)
    monkeypatch.setattr(rollout_mod, "_run_environment_healthcheck", nothing)
    monkeypatch.setattr(rollout_mod, "_run_environment_setup_commands", nothing)
    trial = _trial(tmp_path, _HostProxyDocker(), location=None, environment="docker")
    trial._task = _task(tmp_path)
    trial._agent_env = dict(ENV)

    await trial.start()

    assert seen_at_sandbox_start == [1]
    assert trial._host_proxy_reservation is not None
    # Clean-up gives an untaken reservation back.
    trial._release_host_proxy()
    assert trial._host_proxy_reservation is None
    assert pool._wanted[next(iter(pool._spares))] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["connect", "connect_as"])
async def test_the_first_connection_takes_the_reserved_proxy(
    tmp_path: Path, monkeypatch, pool, starts, entry
):
    """Through connect() and through a scene role's connect_as(), which is
    how `bench eval run` connects its agent: the proxy reserved when the
    sandbox started is the one the agent gets."""
    from unittest.mock import AsyncMock, MagicMock

    from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        cloud_metadata_record,
    )

    started, servers = starts
    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            scenes=[Scene(roles=[Role(name="agent", agent=AGENT, model=MODEL)])],
            agent_env=dict(ENV),
            skip_agent_install=True,
            environment="docker",
        )
    )

    class DockerMock(MagicMock):
        starts_host_proxy_ahead = True

    trial._env = DockerMock()
    trial._rollout_dir = tmp_path
    trial._cloud_metadata = CloudMetadataPlan(True)
    trial._agent_env = rollout_mod._apply_web_policy(
        trial._planes.resolve_agent_env(AGENT, MODEL, dict(ENV)), disallow=False
    )
    for method in ("write_credential_files", "apply_web_tool_policy"):
        monkeypatch.setattr(trial._planes, method, AsyncMock())
    monkeypatch.setattr(
        rollout_mod,
        "enforce_codex_apps_policy",
        AsyncMock(side_effect=lambda env, **kw: kw["agent_env"]),
    )

    async def firewall(env, user, *, holds, can_drop=True):
        return {"state": "clear"}, None

    async def block(env, user):
        return cloud_metadata_record("on", "dummy proof")

    async def agent(**kw):
        return (None, None, None, AGENT)

    monkeypatch.setattr(rollout_mod, "check_firewall_power", firewall)
    monkeypatch.setattr(trial._planes, "block_cloud_metadata", block)
    monkeypatch.setattr(trial, "_connect_live_agent", agent)

    trial._reserve_host_proxy()
    await _settle()
    assert len(started) == 1
    if entry == "connect":
        await trial.connect()
    else:
        await trial.connect_as(Role(name="agent", agent=AGENT, model=MODEL))

    assert trial._usage_runtime.server is servers[0]
    assert trial._usage_runtime.master_key == started[0]["master_key"]
    assert trial._host_proxy_reservation is None
    await _settle()
    # None started at the connection, and none kept ready by default.
    assert len(started) == 1


@pytest.mark.asyncio
async def test_an_import_setting_tiktokens_cache_dir_keeps_the_reservation(
    pool, starts, monkeypatch
):
    """litellm's import (the prewarm thread runs it as the first proxy
    starts) sets TIKTOKEN_CACHE_DIR in this process; the proxy's own import
    sets the same value, so the reserved proxy is still this launch's."""
    _, servers = starts
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    reservation = _reserve()
    await _settle()
    monkeypatch.setenv(
        "TIKTOKEN_CACHE_DIR", "/venv/litellm/litellm_core_utils/tokenizers"
    )

    _, runtime = await _connect(reservation)

    assert runtime.server is servers[0]


@pytest.mark.asyncio
async def test_a_changed_environment_starts_fresh_and_says_what_changed(
    pool, starts, monkeypatch, caplog
):
    _, servers = starts
    reservation = _reserve()
    await _settle()
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")

    with caplog.at_level("DEBUG", logger="benchflow.providers.litellm_runtime"):
        _, runtime = await _connect(reservation)

    assert runtime.server is not servers[0]
    said = [
        r.getMessage() for r in caplog.records if "warm host proxy" in r.getMessage()
    ]
    assert said and "env HTTPS_PROXY" in said[0]
    assert "proxy.invalid" not in " ".join(said)


@pytest.mark.asyncio
async def test_a_sandbox_without_a_host_proxy_never_starts_one_ahead(
    tmp_path: Path, pool, starts
):
    """Daytona, Modal and the rest run the proxy in the sandbox; a mocked
    sandbox never grows the type's marker either."""
    from unittest.mock import MagicMock

    started, _ = starts
    for env in (MagicMock(), _DockerLike()):
        trial = _trial(tmp_path, env, location=None, environment="docker")
        trial._agent_env = dict(ENV)
        trial._reserve_host_proxy()
        assert trial._host_proxy_reservation is None
    await _settle()
    assert started == []


@pytest.mark.asyncio
async def test_an_unused_proxy_is_killed_without_waiting_on_its_log(tmp_path):
    """A pool proxy no trial took never served a request: nothing to settle."""
    import subprocess
    from types import SimpleNamespace

    folder = tmp_path / "benchflow-litellm-unused"
    folder.mkdir()
    process = subprocess.Popen(["sleep", "60"])
    server = SimpleNamespace(process=process, runtime_dir=folder)
    loop = asyncio.get_running_loop()
    started = loop.time()

    await runtime_mod._stop_pool_proxy(server)

    assert process.poll() is not None
    assert not folder.exists()
    # Well inside the log-stable wait (0.5 s) a used proxy's stop takes.
    assert loop.time() - started < 0.5


def test_a_forked_child_leaves_its_parents_proxies_alone(tmp_path, monkeypatch):
    """The registry is inherited by a fork; the proxies are not the child's."""
    folder = tmp_path / "benchflow-litellm-parents"
    folder.mkdir()

    class Parents:
        terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

    proxy = Parents()
    processes = cast(dict[Any, Path], runtime_mod._HOST_PROXY_PROCESSES)
    monkeypatch.setitem(processes, proxy, folder)
    monkeypatch.setattr(runtime_mod, "_HOST_PROXY_OWNER_PID", -1, raising=False)

    runtime_mod._stop_host_proxies_at_exit()

    assert not proxy.terminated
    assert folder.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ahead", "order"), [("1", ["import", "healthy"]), ("0", ["healthy"])]
)
async def test_ahead_zero_imports_litellm_after_the_health_wait(
    tmp_path, monkeypatch, ahead, order
):
    """``BENCHFLOW_HOST_PROXY_AHEAD=0`` also gives back the import's old place:
    after the health wait, where the agent's model metadata needs it."""
    from benchflow.providers.litellm_config import resolve_litellm_route

    monkeypatch.setenv(runtime_mod.HOST_PROXY_AHEAD_ENV, ahead)
    steps: list[str] = []
    monkeypatch.setattr(
        runtime_mod, "_prewarm_litellm_import", lambda: steps.append("import")
    )

    class Proxy:
        pid = 4242

        def poll(self):
            return None

    async def healthy(*args, **kwargs):
        steps.append("healthy")

    monkeypatch.setattr(runtime_mod.subprocess, "Popen", lambda *a, **k: Proxy())
    monkeypatch.setattr(runtime_mod, "_poll_host_health", healthy)
    folder = tmp_path / "runtime"
    folder.mkdir()
    monkeypatch.setattr(runtime_mod.tempfile, "mkdtemp", lambda **_: str(folder))
    env = {"OPENAI_API_KEY": "sk-test"}
    started = await runtime_mod._start_host_litellm(
        route=resolve_litellm_route("openai/gpt-4o-mini", env),
        master_key="sk-master",
        agent_env=env,
        environment="local",
        session_id="s",
        agent_name="ahead-test",
    )
    runtime_mod._HOST_PROXY_PROCESSES.pop(started.process, None)

    assert steps == order
