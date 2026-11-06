"""Timing spans in timing.json (round PY8, target T0).

Each span is recorded when its step runs and absent (never 0) when it does
not: nop and the oracle install no agent, the oracle starts no proxy, a host
proxy installs nothing. A fake clock makes every recorded value exact. The job
hands each trial the seconds it held its concurrency slot before starting
(``slot_wait``); ``tests/test_start_pacing.py`` checks it, and the
start pacer's ``start_pace_wait``, at --concurrency 32 and 64 (U9).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils import timing_spans
from benchflow.rollout import Role, Rollout, RolloutConfig
from tests._receipt_fakes import receipt_aware


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(timing_spans, "_clock", fake)
    return fake


def _ticking(clock: FakeClock, seconds: float, result=None):
    async def step(*_args, **_kwargs):
        clock.advance(seconds)
        return result

    return step


# --- the recorder --------------------------------------------------------------


def test_a_span_adds_its_seconds_and_a_step_is_a_sub_key(clock):
    timing: dict = {}
    for seconds in (2.0, 3.0):
        with timing_spans.span("acp_connect", into=timing):
            clock.advance(seconds)
    with timing_spans.span("harden_before_verify", "reclaim_disk", into=timing):
        clock.advance(0.25)
    assert timing == {
        "acp_connect": 5.0,
        "harden_before_verify": {"reclaim_disk": 0.25},
    }


def test_a_span_records_the_seconds_before_a_failure(clock):
    timing: dict = {}
    with (
        pytest.raises(RuntimeError),
        timing_spans.span("install_agent", into=timing),
    ):
        clock.advance(4.0)
        raise RuntimeError("install failed")
    assert timing == {"install_agent": 4.0}


def test_a_span_outside_a_trial_records_nothing(clock):
    timing: dict = {}
    with timing_spans.span("litellm_install"):
        clock.advance(1.0)
    with timing_spans.recording_into(timing), timing_spans.span("litellm_install"):
        clock.advance(2.0)
    with timing_spans.span("litellm_install"):
        clock.advance(1.0)
    assert timing == {"litellm_install": 2.0}


def test_the_job_hands_a_trial_its_waits_once():
    timing_spans.set_pending_trial_timing(slot_wait=1.5)
    first = Rollout(RolloutConfig(task_path=Path("/tmp/task-a"), agent="nop"))
    retry = Rollout(RolloutConfig(task_path=Path("/tmp/task-a"), agent="nop"))
    assert first.timing == {"slot_wait": 1.5}
    assert retry.timing == {}


def test_timing_json_keeps_phases_at_a_tenth_and_spans_at_a_millisecond():
    assert timing_spans.rounded(
        {
            "environment_setup": 2.04,
            "total": 9.96,
            "acp_connect": 1.23456,
            "harden_before_verify": {"total": 0.4567, "reclaim_disk": 0.0123},
        }
    ) == {
        "environment_setup": 2.0,
        "total": 10.0,
        "acp_connect": 1.235,
        "harden_before_verify": {"total": 0.457, "reclaim_disk": 0.012},
    }


# --- install_agent ------------------------------------------------------------------


def _install_trial(tmp_path: Path, agent: str, **kwargs) -> Rollout:
    config = RolloutConfig.from_legacy(
        task_path=tmp_path / "task",
        agent=agent,
        prompts=[None],
        sandbox_user="agent",
        **kwargs,
    )
    trial = Rollout(config)
    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=MagicMock(stdout="/workspace\n"))
    trial._rollout_dir = tmp_path / "trial"
    trial._rollout_dir.mkdir()
    trial._rollout_paths = MagicMock()
    trial._task = MagicMock()
    trial._effective_locked = []
    return trial


def _quiet_planes(monkeypatch, trial: Rollout, clock: FakeClock) -> AsyncMock:
    install = AsyncMock(side_effect=_ticking(clock, 7.0, MagicMock()))
    monkeypatch.setattr(trial._planes, "install_agent", install)
    for name in (
        "write_credential_files",
        "upload_subscription_auth",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "prepare_verifier_runtime",
        "deploy_skills",
        "lockdown_paths",
        "apply_web_tool_policy",
    ):
        monkeypatch.setattr(trial._planes, name, AsyncMock())
    monkeypatch.setattr(
        trial._planes, "setup_sandbox_user", AsyncMock(return_value="/home/agent")
    )
    return install


@pytest.mark.asyncio
async def test_install_agent_is_timed_for_a_model_agent(tmp_path, monkeypatch, clock):
    trial = _install_trial(tmp_path, "claude-agent-acp")
    install = _quiet_planes(monkeypatch, trial, clock)
    await trial.install_agent()
    install.assert_awaited_once()
    assert trial.timing == {"install_agent": 7.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["nop", "oracle"])
async def test_nop_and_the_oracle_install_nothing_and_have_no_span(
    tmp_path, monkeypatch, clock, agent
):
    trial = _install_trial(tmp_path, agent)
    install = _quiet_planes(monkeypatch, trial, clock)
    await trial.install_agent()
    install.assert_not_awaited()
    assert "install_agent" not in trial.timing


@pytest.mark.asyncio
async def test_a_skipped_install_has_no_span(tmp_path, monkeypatch, clock):
    trial = _install_trial(tmp_path, "claude-agent-acp", skip_agent_install=True)
    install = _quiet_planes(monkeypatch, trial, clock)
    await trial.install_agent()
    install.assert_not_awaited()
    assert "install_agent" not in trial.timing


# --- connect: acp_connect, and the proxy spans reach the rollout --------------------


@pytest.mark.asyncio
async def test_connect_times_the_acp_connect_and_records_the_proxy_spans(
    tmp_path, monkeypatch, clock
):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="codex-acp",
            model="openai/gpt-4.1-mini",
            environment="docker",
        )
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "rollout"
    rollout._agent_env = {"OPENAI_API_KEY": "sk-test"}
    rollout._agent_launch = "codex-acp"
    rollout._agent_cwd = "/workspace"
    # C33 prepares the managed launcher file; this fake operation adds no time.
    rollout._env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0))
    )
    rollout._reapply_ask_user_handler = lambda: None
    rollout._attach_trajectory_writer = lambda _rollout_dir: None

    async def fake_litellm(**kwargs):
        # Stands in for the proxy start, whose spans record into the trial.
        with timing_spans.span(timing_spans.LITELLM_INSTALL):
            clock.advance(5.0)
        with timing_spans.span(timing_spans.PROXY_HEALTH_WAIT):
            clock.advance(1.5)
        return dict(kwargs["agent_env"]), SimpleNamespace(kind="litellm")

    async def fake_apps_policy(_env, **kwargs):
        return kwargs["agent_env"]

    monkeypatch.setattr("benchflow.rollout.enforce_codex_apps_policy", fake_apps_policy)

    async def fake_connect_acp(**_kwargs):
        clock.advance(3.0)
        return (AsyncMock(), AsyncMock(), AsyncMock(), "codex-acp")

    rollout._planes = SimpleNamespace(
        ensure_litellm_runtime=fake_litellm,
        connect_acp=fake_connect_acp,
        # connect() stages the connection's credential files (C23/C24).
        write_credential_files=AsyncMock(return_value=[]),
        upload_subscription_auth=AsyncMock(return_value=[]),
        # ...and runs the web-tool policy setup after them (C30).
        apply_web_tool_policy=AsyncMock(),
        quiesce_agent=AsyncMock(),
    )

    await rollout.connect()

    assert rollout.timing["litellm_install"] == 5.0
    assert rollout.timing["proxy_health_wait"] == 1.5
    assert rollout.timing["acp_connect"] == 3.0
    # The spans sit inside agent_setup, which keeps its meaning.
    assert rollout.timing["agent_setup"] >= 0


@pytest.mark.asyncio
async def test_connect_as_times_a_role_agents_install_and_connect(tmp_path, clock):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="codex-acp",
            model="openai/gpt-4.1-mini",
            environment="docker",
            agent_env={"OPENAI_API_KEY": "sk-openai"},
        )
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "rollout"
    rollout._agent_cwd = "/workspace"
    rollout._env = SimpleNamespace()
    rollout._agent_cfg = SimpleNamespace()
    rollout._reapply_ask_user_handler = lambda: None
    rollout._attach_trajectory_writer = lambda _rollout_dir: None
    role = Role(
        name="reviewer",
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        env={"ANTHROPIC_API_KEY": "sk-ant"},
    )

    async def fake_litellm(**kwargs):
        return dict(kwargs["agent_env"]), SimpleNamespace(kind="litellm")

    async def fake_connect_acp(**_kwargs):
        clock.advance(2.0)
        return (AsyncMock(), AsyncMock(), AsyncMock(), "claude-agent-acp")

    rollout._planes = SimpleNamespace(
        agent_launch=lambda agent, disallow_web_tools, disallow_hosted_search=False: (
            agent
        ),
        resolve_agent_env=lambda agent, model, env: dict(env or {}),
        ensure_litellm_runtime=fake_litellm,
        install_agent=AsyncMock(side_effect=_ticking(clock, 6.0, SimpleNamespace())),
        write_credential_files=AsyncMock(),
        upload_subscription_auth=AsyncMock(),
        apply_web_tool_policy=AsyncMock(),
        connect_acp=fake_connect_acp,
    )

    await rollout.connect_as(role)

    assert rollout.timing["install_agent"] == 6.0
    assert rollout.timing["acp_connect"] == 2.0
    # This fake proxy did no install and no health wait: no span, not 0.
    assert "litellm_install" not in rollout.timing
    assert "proxy_health_wait" not in rollout.timing


# --- the proxy: in the sandbox, on the host, and none for the oracle ---------------


class _Result:
    def __init__(self, return_code: int = 0, stdout: str = "") -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = ""


class _FakeSandbox:
    """Answers the in-sandbox proxy's install, launch and health commands."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.started = False

    async def upload_file(
        self, local_path, remote_path, *, mode=None, **_kwargs
    ) -> None:
        return None

    @receipt_aware
    async def exec(self, command: str, timeout_sec=None, user=None) -> _Result:
        if "pip install" in command and "litellm" in command:
            self.clock.advance(5.0)
            return _Result()
        if "launcher.py" in command:
            self.clock.advance(1.0)
            self.started = True
            return _Result()
        if command.strip().startswith("cat") and "state.json" in command:
            self.clock.advance(0.5)
            return _Result(stdout=json.dumps({"pid": 7, "port": 45999}))
        if "urllib.request" in command:
            self.clock.advance(0.25)
            return _Result()
        return _Result()


@pytest.mark.asyncio
async def test_the_in_sandbox_proxy_records_its_install_and_health_wait(clock):
    from benchflow.providers import litellm_runtime as runtime_mod
    from benchflow.providers.litellm_config import resolve_litellm_route

    route = resolve_litellm_route("claude-haiku-4-5", {"ANTHROPIC_API_KEY": "k"})
    timing: dict = {}
    with timing_spans.recording_into(timing):
        await runtime_mod._start_sandbox_litellm(
            sandbox=_FakeSandbox(clock),
            route=route,
            master_key="sk-master",
            agent_env={"ANTHROPIC_API_KEY": "k"},
            session_id="s",
            agent_name="claude-agent-acp",
        )
    assert timing == {"litellm_install": 5.0, "proxy_health_wait": 1.75}


@pytest.mark.asyncio
async def test_the_host_proxy_waits_for_health_and_installs_nothing(monkeypatch, clock):
    from benchflow.providers import litellm_runtime as runtime_mod
    from benchflow.providers.litellm_config import resolve_litellm_route

    class _Process:
        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, _timeout=None) -> int:
            return 0

        def kill(self) -> None:
            return None

    monkeypatch.setattr(runtime_mod, "_host_litellm_executable", lambda: "litellm")
    monkeypatch.setattr(runtime_mod, "_host_bind_address", lambda _env: "127.0.0.1")
    monkeypatch.setattr(
        runtime_mod.subprocess, "Popen", lambda *_args, **_kwargs: _Process()
    )
    monkeypatch.setattr(runtime_mod, "_poll_host_health", _ticking(clock, 2.0))
    route = resolve_litellm_route("claude-haiku-4-5", {"ANTHROPIC_API_KEY": "k"})
    timing: dict = {}
    with timing_spans.recording_into(timing):
        runner = await runtime_mod._start_host_litellm(
            route=route,
            master_key="sk-master",
            agent_env={"ANTHROPIC_API_KEY": "k"},
            environment="docker",
            session_id="s",
            agent_name="claude-agent-acp",
        )
    shutil.rmtree(runner.runtime_dir, ignore_errors=True)
    assert timing == {"proxy_health_wait": 2.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["oracle", "nop"])
async def test_no_proxy_runs_for_the_oracle_or_nop(clock, agent):
    from benchflow.providers.litellm_runtime import ensure_litellm_runtime

    timing: dict = {}
    with timing_spans.recording_into(timing):
        _env, runtime = await ensure_litellm_runtime(
            agent=agent,
            agent_env={},
            model=None,
            runtime=None,
            environment="docker",
            sandbox=_FakeSandbox(clock),
        )
    assert runtime is None
    assert timing == {}


# --- harden_before_verify: one sub-key per step -------------------------------------


def _stub_hardening(monkeypatch, clock: FakeClock) -> None:
    from benchflow.sandbox import lockdown

    # V7's pre-freeze activation decision is independent of these step clocks.
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    for seconds, name in enumerate(
        [
            "_kill_sandbox_user_procs",
            "clear_verifier_output_dir",
            "ensure_legacy_app_dir",
            "_reclaim_disk",
            "_restore_workspace_state",
            "_freeze_workspace",
            "_build_verifier_env",
        ],
        start=1,
    ):
        monkeypatch.setattr(lockdown, name, _ticking(clock, float(seconds), {}))


def _hardening_env(clock: FakeClock):
    # Acknowledge credential cleanup; keep the existing timed conftest exec.
    async def execute(command, **kwargs):
        if ".config/muse/auth.json" in command:
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        if "/proc/self/mountinfo" in command:
            from tests.sandbox_mounts import ROOT_MOUNTINFO

            return SimpleNamespace(return_code=0, stdout=ROOT_MOUNTINFO, stderr="")
        if "printf 'complete\\n'" in command:
            return SimpleNamespace(return_code=0, stdout="complete\n", stderr="")
        return await _ticking(clock, 0.5)(command, **kwargs)

    return SimpleNamespace(
        exec=AsyncMock(side_effect=receipt_aware(execute)),
        _benchflow_path_inspectors={"main": "/etc/fake-inspector"},
    )


def _hardening_task():
    return SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(verifier=SimpleNamespace(env={}, service="main")),
    )


@pytest.mark.asyncio
async def test_every_hardening_step_has_its_sub_key(monkeypatch, clock):
    from benchflow.sandbox.lockdown import harden_before_verify

    _stub_hardening(monkeypatch, clock)
    timing: dict = {}
    with timing_spans.recording_into(timing):
        await harden_before_verify(
            _hardening_env(clock),
            _hardening_task(),
            "agent",
            workspace="/app",
            restore_workspace=True,
        )
    assert timing == {
        "harden_before_verify": {
            "kill_sandbox_user_procs": 1.0,
            "clear_verifier_output": 2.0,
            "prepare_app_dir": 3.0,
            "reclaim_disk": 4.0,
            "restore_workspace": 5.0,
            "freeze_workspace": 6.0,
            "cleanup_conftests": 0.5,
            "build_verifier_env": 7.0,
        }
    }
    assert tuple(timing["harden_before_verify"]) == timing_spans.HARDEN_STEPS


@pytest.mark.asyncio
async def test_hardening_steps_that_do_not_run_have_no_sub_key(monkeypatch, clock):
    from benchflow.sandbox.lockdown import harden_before_verify

    _stub_hardening(monkeypatch, clock)
    timing: dict = {}
    with timing_spans.recording_into(timing):
        # No sandbox user (nothing to kill), no workspace (nothing to freeze).
        await harden_before_verify(_hardening_env(clock), _hardening_task(), None)
    steps = timing["harden_before_verify"]
    assert "kill_sandbox_user_procs" not in steps
    assert "restore_workspace" not in steps
    assert "freeze_workspace" not in steps
    assert set(steps) == {
        "clear_verifier_output",
        "prepare_app_dir",
        "reclaim_disk",
        "cleanup_conftests",
        "build_verifier_env",
    }


@pytest.mark.asyncio
async def test_the_verifier_records_the_hardening_total_and_its_steps(tmp_path, clock):
    from benchflow.rollout._setup import _verify_rollout

    async def harden(*_args, **_kwargs):
        with timing_spans.span("harden_before_verify", "reclaim_disk"):
            clock.advance(1.25)
        clock.advance(0.5)
        raise RuntimeError("stop before the verifier")

    planes = SimpleNamespace(harden_before_verify=harden)
    paths = SimpleNamespace(verifier_dir=tmp_path / "verifier")
    task = SimpleNamespace(
        name="t", config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=60))
    )
    timing: dict = {}
    rewards, error, _ = await _verify_rollout(
        SimpleNamespace(), task, paths, timing, planes
    )
    assert rewards is None and error is not None
    assert timing["harden_before_verify"] == {"total": 1.75, "reclaim_disk": 1.25}
    assert "verifier" in timing


# --- cleanup ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_is_timed(tmp_path, monkeypatch, clock):
    rollout = Rollout(RolloutConfig(task_path=tmp_path / "task", agent="nop"))
    # The teardown's slow steps: closing the agent, collecting what it left.
    monkeypatch.setattr(rollout, "disconnect", _ticking(clock, 1.5))
    monkeypatch.setattr(
        "benchflow.rollout.collect_unverified_rollout_artifacts", _ticking(clock, 2.0)
    )
    monkeypatch.setattr("benchflow.rollout.price_from_session_log", _ticking(clock, 0))
    await rollout.cleanup()
    assert rollout.timing == {"cleanup": 3.5}
