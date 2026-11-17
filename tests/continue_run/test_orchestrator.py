"""Tests for the pure orchestration helpers (no sandbox / no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.continue_run.orchestrator import (
    LiteLLMLiveForwarder,
    _continue_run_with_sandbox_proxy,
    _safe_sandbox_continuation_teardown,
    _stop_continuation_proxies,
    build_agent_env,
    build_rollout_config,
    continued_rollout_name,
    resolve_task_path,
    select_proxy_mode,
    stitched_trajectory_lines,
    summarize_llm_trajectory_usage,
    update_continued_metadata,
    write_stitched_trajectory,
)
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder
from benchflow.rollout import Rollout

from ._helpers import completion, exchange, write_run_folder


def _load(tmp_path, **kw):
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))], **kw
    )
    return load_run_folder(folder)


def test_build_agent_env_points_at_proxy():
    env = build_agent_env("http://10.0.0.1:9000/v1")
    assert env["LLM_BASE_URL"] == "http://10.0.0.1:9000/v1"
    assert env["LLM_MODEL"].startswith("openai/")
    assert env["LLM_API_KEY"]


def test_select_proxy_mode_uses_sandbox_for_remote_environments():
    """Guards PR #648 and #936 against unreachable host-loopback replay."""
    assert select_proxy_mode("auto", "daytona") == "sandbox"
    assert select_proxy_mode("auto", "modal") == "sandbox"
    assert select_proxy_mode("auto", "apple-container") == "sandbox"
    assert select_proxy_mode("auto", "docker") == "host"
    assert select_proxy_mode("host", "daytona") == "host"


def test_continued_rollout_name_is_unique_to_source_folder(tmp_path):
    """Guards PR #648 follow-up against batch continuation directory collisions."""
    folder = write_run_folder(
        tmp_path / "demo-task__abc123",
        exchanges=[exchange(completion(content="a"))],
        task_name="demo-task",
    )
    run = load_run_folder(folder)
    assert continued_rollout_name(run) == "demo-task__abc123__continued"


def test_resolve_task_path_via_tasks_dir(tmp_path):
    run = _load(tmp_path)
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "demo-task").mkdir(parents=True)
    assert resolve_task_path(run, tasks_dir) == tasks_dir / "demo-task"


def test_resolve_task_path_missing_in_tasks_dir(tmp_path):
    run = _load(tmp_path)
    (tmp_path / "tasks").mkdir()
    with pytest.raises(RunFolderError, match="does not exist"):
        resolve_task_path(run, tmp_path / "tasks")


def test_resolve_task_path_no_source_errors(tmp_path):
    run = _load(tmp_path)  # recorded task_path is /tasks/demo-task (absent)
    with pytest.raises(RunFolderError, match="cannot locate task source"):
        resolve_task_path(run, None)


def test_resolve_task_path_falls_back_to_recorded(tmp_path):
    real_task = tmp_path / "real-task"
    real_task.mkdir()
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))]
    )
    # repoint config.task_path at an existing dir
    cfg = json.loads((folder / "config.json").read_text())
    cfg["task_path"] = str(real_task)
    (folder / "config.json").write_text(json.dumps(cfg))
    run = load_run_folder(folder)
    assert resolve_task_path(run, None) == real_task


def test_build_rollout_config_disables_litellm_and_points_at_proxy(tmp_path):
    run = _load(tmp_path, prompts=["go"])
    task = tmp_path / "real-task"
    task.mkdir()
    cfg = build_rollout_config(
        run,
        task_path=task,
        live_model="gemini-3.1-flash-lite-preview",
        agent_env=build_agent_env("http://host:1/v1"),
        timeout=123,
        output_dir=tmp_path / "out",
        rollout_name="demo-task__continued",
    )
    assert cfg.agent == "openhands"
    # the seam that stops benchflow starting its own gateway
    assert cfg.usage_tracking.mode == "off"
    assert cfg.agent_env["LLM_BASE_URL"] == "http://host:1/v1"
    assert cfg.timeout == 123
    # model is None so resolve_agent_env skips provider key validation; the live
    # model is carried in provenance and used only by the forwarder.
    assert cfg.model is None
    assert cfg.source_provenance["live_model"] == "gemini-3.1-flash-lite-preview"
    assert cfg.prompts == ["go"]
    assert cfg.source_provenance["continued_from"] == str(run.path)
    assert cfg.source_provenance["kind"] == "benchflow-continue"


def test_live_forwarder_build_kwargs_resolves_route_offline():
    fwd = LiteLLMLiveForwarder(
        "gemini-3.1-flash-lite-preview", env={"GEMINI_API_KEY": "x"}
    )
    assert fwd.upstream_model.startswith("gemini/")
    kwargs = fwd.build_kwargs(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "temperature": 0.5,
            "stream": True,  # must be forced False for the non-streamed capture
        }
    )
    assert kwargs["model"] == fwd.upstream_model
    assert kwargs["messages"][0]["content"] == "hi"
    assert kwargs["stream"] is False
    assert kwargs["tools"][0]["function"]["name"] == "bash"
    assert kwargs["temperature"] == 0.5


def test_stitched_trajectory_recorded_prefix_plus_live_suffix(tmp_path):
    original = tmp_path / "orig.jsonl"
    original.write_text('{"a": 1}\n{"b": 2}\n')
    live = [exchange(completion(content="LIVE"))]
    lines = stitched_trajectory_lines(original, live)
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"a": 1}
    last = json.loads(lines[2])
    assert last["response"]["body"]["choices"][0]["message"]["content"] == "LIVE"


def test_write_stitched_trajectory_creates_file(tmp_path):
    original = tmp_path / "orig.jsonl"
    original.write_text('{"a": 1}\n')
    rollout_dir = tmp_path / "rollout"
    out = write_stitched_trajectory(
        rollout_dir, original, [exchange(completion(content="L"))]
    )
    assert out == rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    assert len(out.read_text().strip().splitlines()) == 2


def test_summarize_llm_trajectory_usage_splits_recorded_and_live(tmp_path):
    """Guards the PR #648 continuation metadata fix for stitched token usage."""
    traj = tmp_path / "llm_trajectory.jsonl"
    rows = [
        {
            "response": {
                "body": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                        "cache_creation_input_tokens": 3,
                    }
                }
            }
        },
        {
            "response": {
                "body": {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                        "cache_read_input_tokens": 7,
                    }
                }
            }
        },
    ]
    traj.write_text("".join(json.dumps(row) + "\n" for row in rows))

    usage = summarize_llm_trajectory_usage(traj, n_recorded=1)

    assert usage.n_input_tokens == 30
    assert usage.n_output_tokens == 7
    assert usage.total_tokens == 37
    assert usage.recorded_total_tokens == 12
    assert usage.live_total_tokens == 25
    assert usage.n_cache_creation_tokens == 3
    assert usage.n_cache_read_tokens == 7
    assert usage.usage_source == "provider_response"


def test_update_continued_metadata_writes_model_and_usage(tmp_path):
    """Guards the PR #648 continuation metadata fix for HF-compatible results."""
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "config.json").write_text(json.dumps({"model": None, "source": {}}))
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "model": None,
                "agent_result": {
                    "total_tokens": 0,
                    "usage_source": "unavailable",
                    "cost_usd": None,
                },
                "final_metrics": {},
                "usage_tracking": {"requested": "off", "status": "off"},
            }
        )
    )
    traj = tmp_path / "llm_trajectory.jsonl"
    traj.write_text(
        json.dumps(
            {
                "response": {
                    "body": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        }
                    }
                }
            }
        )
        + "\n"
    )

    update_continued_metadata(
        rollout,
        live_model="aws-bedrock/us.anthropic.claude-opus-4-8",
        usage=summarize_llm_trajectory_usage(traj, n_recorded=0),
        environment="daytona",
    )

    config = json.loads((rollout / "config.json").read_text())
    result = json.loads((rollout / "result.json").read_text())
    assert config["model"] == "aws-bedrock/us.anthropic.claude-opus-4-8"
    assert result["model"] == "aws-bedrock/us.anthropic.claude-opus-4-8"
    assert result["agent_result"]["total_tokens"] == 12
    # vocabulary, the old name beside it.
    assert result["agent_result"]["usage_source"] == "proxy"
    assert result["agent_result"]["usage_source_legacy"] == "provider_response"
    assert result["usage_tracking"]["requested"] == "required"
    assert result["usage_tracking"]["endpoint_kind"] == "sandbox"
    assert result["usage_tracking"]["status"] == "captured_from_stitched_llm_trajectory"
    assert config["usage_tracking"]["requested"] == "required"


@pytest.mark.asyncio
async def test_sandbox_teardown_still_runs_rollout_cleanup_after_sidecar_failure():
    """Guards PR #648 follow-up: Daytona artifacts must survive stop failures."""

    class FailingProxy:
        async def stop(self):
            raise RuntimeError("proxy unavailable")

    class FakeRollout:
        _error = None

        def __init__(self):
            self.cleaned = False

        async def cleanup(self):
            self.cleaned = True

    async def stop_provider_runtime(runtime):
        raise RuntimeError(f"{runtime} refused stop")

    rollout = FakeRollout()
    events: list[str] = []

    async def before_cleanup():
        events.append("artifact")
        assert rollout._error is not None

    errors = await _safe_sandbox_continuation_teardown(
        rollout=rollout,
        replay_proxy=FailingProxy(),
        provider_runtime="provider",
        stop_provider_runtime=stop_provider_runtime,
        before_cleanup=before_cleanup,
    )

    assert rollout.cleaned is True
    assert events == ["artifact"]
    assert len(errors) == 2
    assert rollout._error is not None
    assert "proxy unavailable" in rollout._error
    assert "provider refused stop" in rollout._error


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_origin", ["cleanup", "stitched", "none"])
async def test_bare_timeout_waits_for_continuation_usage(
    tmp_path, monkeypatch, usage_origin
):
    """Guards PR #1131: pre-cleanup capture must not consume the timeout verdict."""
    original = completion(content="recorded")
    original.pop("usage")
    run = load_run_folder(
        write_run_folder(tmp_path / "source", exchanges=[exchange(original)])
    )
    output = tmp_path / "out" / "continued"
    output.mkdir(parents=True)
    events = []

    class Trial:
        _record_agent_timeout = Rollout._record_agent_timeout
        _maybe_classify_api_error = Rollout._maybe_classify_api_error

        def __init__(self):
            self._rollout_dir = output
            self._phase = "installed"
            self._error = None
            self._verifier_error = None
            self._rewards = None
            self._usage_metrics = {}
            self._n_tool_calls = 0
            self._api_failure_summary_cached = None
            self._executed_prompts = ["p"]
            self._resolved_prompts = ["p"]
            self._agent_env = {"BENCHFLOW_PROVIDER_NAME": "litellm"}
            self._config = SimpleNamespace(agent="openhands", model="test")
            self._diagnostics = SimpleNamespace(
                set=lambda d: None, capture_idle=lambda e: None
            )
            self._timeout = 14400
            self._started_at = datetime.now() - timedelta(seconds=303)
            self._planes = SimpleNamespace(resolve_agent_env=lambda *args: {})
            self._disallow_web_tools = False
            self._egress_denylist = None
            self.env = object()
            self.setup = AsyncMock()
            self.start = AsyncMock()
            self.install_agent = AsyncMock()
            self._run_steps = AsyncMock(side_effect=TimeoutError())
            (output / "config.json").write_text("{}")

        async def verify(self):
            self._phase = "verified"
            self._rewards = {"reward": 0.0}

        async def cleanup(self):
            # The evidence is durable, but classification has not happened.
            assert (output / "trajectory/llm_trajectory.jsonl").is_file()
            assert not (output / "result.json").exists()
            assert self._bare_timeout is True
            events.append("cleanup")
            if usage_origin == "cleanup":
                self._usage_metrics = {
                    "total_tokens": 123,
                    "usage_source": "provider_response",
                }
            self._phase = "cleaned"

        async def finalize(self):
            assert events == ["cleanup"]
            self._maybe_classify_api_error()
            payload = {
                "error": self._error,
                "rewards": self._rewards,
                "agent_result": dict(self._usage_metrics),
            }
            (output / "result.json").write_text(json.dumps(payload))
            return SimpleNamespace(error=self._error, rewards=self._rewards)

    trial = Trial()
    monkeypatch.setattr(Rollout, "create", AsyncMock(return_value=trial))
    monkeypatch.setattr(
        "benchflow.providers.runtime.ensure_litellm_runtime",
        AsyncMock(
            return_value=(
                {
                    "LLM_BASE_URL": "http://unused",
                    "LLM_API_KEY": "dummy",
                    "LLM_MODEL": "test",
                },
                object(),
            )
        ),
    )
    monkeypatch.setattr(
        "benchflow.providers.runtime.stop_provider_runtime", AsyncMock()
    )
    live = [exchange(completion(content="live"))] if usage_origin == "stitched" else []
    proxy = SimpleNamespace(stop=AsyncMock(), live_exchanges=live)
    monkeypatch.setattr(
        "benchflow.continue_run.orchestrator.SandboxReplayProxy.start",
        AsyncMock(return_value=proxy),
    )
    result = await _continue_run_with_sandbox_proxy(
        run,
        task_path=tmp_path / "task",
        live_model="test-model",
        timeout=None,
        output_dir=output.parent,
        rollout_name=output.name,
        strict_divergence=False,
    )
    persisted = json.loads((output / "result.json").read_text())
    assert result.error == persisted["error"]
    assert result.rewards == persisted["rewards"]
    assert persisted["model"] == "test-model"
    if usage_origin == "none":
        assert "suspected provider api error" in result.error
        assert result.rewards is None
    else:
        assert "elapsed in rollout" in result.error
        assert result.rewards == {"reward": 0.0}
        assert persisted["agent_result"]["total_tokens"] == (
            123 if usage_origin == "cleanup" else 2
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "where,step", [("install", "install"), ("replay proxy", "agent set-up")]
)
async def test_a_continued_runs_step_timeout_is_not_the_agents(
    tmp_path, monkeypatch, where, step
):
    """Review of PR #141, 2026-11-13 (S3.5): continue_run's own lifecycle
    recorded a timeout of its install or of the replay proxy it starts in the
    sandbox as the agent's, "Agent timed out after 0s elapsed in rollout
    (budget 0s)", category timeout, never retried. It follows run()'s rule."""
    from benchflow._utils.scoring import PHASE_TIMEOUT, PHASE_TIMEOUT_HEADER
    from benchflow.rollout import RolloutConfig

    run = _load(tmp_path)
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 900\n')
    (task / "instruction.md").write_text("Do the thing.\n")
    output = tmp_path / "out" / "continued"
    output.mkdir(parents=True)
    (output / "config.json").write_text("{}")
    rollout = Rollout(RolloutConfig(task_path=task, agent="openhands"))
    rollout._rollout_dir = output
    rollout._rollout_name = output.name
    rollout._started_at = datetime.now()
    rollout._timeout = 0
    rollout._env = SimpleNamespace()

    def entering(phase):
        async def step(*_args, **_kwargs):
            rollout._phase = phase

        return step

    rollout.setup = AsyncMock(side_effect=entering("setup"))
    rollout.start = AsyncMock(side_effect=entering("started"))
    rollout.install_agent = AsyncMock(
        side_effect=TimeoutError() if where == "install" else entering("installed")
    )
    rollout.cleanup = AsyncMock()
    monkeypatch.setattr(
        rollout._planes, "resolve_agent_env", lambda *args: {}, raising=False
    )
    monkeypatch.setattr(Rollout, "create", AsyncMock(return_value=rollout))
    monkeypatch.setattr(
        "benchflow.providers.runtime.ensure_litellm_runtime",
        AsyncMock(
            return_value=(
                {
                    "LLM_BASE_URL": "http://unused",
                    "LLM_API_KEY": "dummy",
                    "LLM_MODEL": "test",
                },
                object(),
            )
        ),
    )
    monkeypatch.setattr(
        "benchflow.providers.runtime.stop_provider_runtime", AsyncMock()
    )
    monkeypatch.setattr(
        "benchflow.continue_run.orchestrator.SandboxReplayProxy.start",
        AsyncMock(side_effect=TimeoutError()),
    )

    result = await _continue_run_with_sandbox_proxy(
        run,
        task_path=task,
        live_model="test-model",
        timeout=None,
        output_dir=output.parent,
        rollout_name=output.name,
        strict_divergence=False,
    )

    assert result.error.startswith(f"{PHASE_TIMEOUT_HEADER} [{step}] after "), (
        result.error
    )
    assert "Agent timed out" not in result.error
    saved = json.loads((output / "result.json").read_text())
    assert saved["error_category"] == PHASE_TIMEOUT
    assert saved["phase_timeout_info"]["step"] == step


@pytest.mark.asyncio
@pytest.mark.parametrize("later", ["exec timeout", "connection lost"])
async def test_a_continued_runs_later_failure_keeps_the_agents_timeout(
    tmp_path, monkeypatch, later
):
    """Review of PR #141, 2026-11-13 (F1, F3): continue_run follows run()'s
    rule. After the agent's prompt timeout, the verify step raised
    ExecTimeoutError or lost its connection, and continue_run's own handlers
    replaced the agent's timeout with it. The timeout stays the trial's error
    and category, is not retried, the failure is recorded beside it, and the
    trial is unscored."""
    import benchflow.rollout as rollout_module
    from benchflow._utils.scoring import TIMED_OUT
    from benchflow.diagnostics import (
        AgentPromptTimeoutDiagnostic,
        AgentPromptTimeoutError,
    )
    from benchflow.evaluation import RetryConfig
    from benchflow.rollout import RolloutConfig
    from benchflow.sandbox.protocol import ExecTimeoutError

    run = _load(tmp_path)
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 900\n')
    (task / "instruction.md").write_text("Do the thing.\n")
    output = tmp_path / "out" / "continued"
    output.mkdir(parents=True)
    (output / "config.json").write_text("{}")
    rollout = Rollout(RolloutConfig(task_path=task, agent="openhands"))
    rollout._rollout_dir = output
    rollout._rollout_name = output.name
    rollout._started_at = datetime.now()
    rollout._env = SimpleNamespace()
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())

    async def turn(_steps):
        rollout._n_tool_calls = 2
        raise AgentPromptTimeoutError(
            "Agent prompt timed out after 900s",
            trajectory=[],
            diagnostic=AgentPromptTimeoutDiagnostic(timeout_sec=900.0),
        )

    rollout._run_steps = turn

    async def probe(_rollout, _phase):
        return None

    monkeypatch.setattr(rollout_module, "_probe_integrity_access", probe)
    if later == "exec timeout":
        monkeypatch.setattr(
            rollout_module,
            "capture_terminal_workspace",
            AsyncMock(side_effect=ExecTimeoutError(600)),
        )
        expected = (
            "exec_timeout",
            "ExecTimeoutError: Command timed out after 600 seconds",
        )
    else:
        rollout._muse_credentials_staged = True
        rollout.disconnect = AsyncMock(side_effect=ConnectionError("transport lost"))
        expected = ("connection_lost", "ConnectionError: transport lost")
    monkeypatch.setattr(
        rollout._planes, "resolve_agent_env", lambda *args: {}, raising=False
    )
    monkeypatch.setattr(Rollout, "create", AsyncMock(return_value=rollout))
    monkeypatch.setattr(
        "benchflow.providers.runtime.ensure_litellm_runtime",
        AsyncMock(
            return_value=(
                {
                    "LLM_BASE_URL": "http://unused",
                    "LLM_API_KEY": "dummy",
                    "LLM_MODEL": "test",
                },
                object(),
            )
        ),
    )
    monkeypatch.setattr(
        "benchflow.providers.runtime.stop_provider_runtime", AsyncMock()
    )
    monkeypatch.setattr(
        "benchflow.continue_run.orchestrator.SandboxReplayProxy.start",
        AsyncMock(return_value=SimpleNamespace(live_exchanges=[], stop=AsyncMock())),
    )

    result = await _continue_run_with_sandbox_proxy(
        run,
        task_path=task,
        live_model="test-model",
        timeout=None,
        output_dir=output.parent,
        rollout_name=output.name,
        strict_divergence=False,
    )

    assert result.error == "Agent prompt timed out after 900s", result.error
    assert result.rewards is None
    saved = json.loads((output / "result.json").read_text())
    assert saved["error_category"] == TIMED_OUT
    assert not RetryConfig().should_retry(result.error, category=TIMED_OUT)
    later_info = saved["post_turn_failure_info"]
    assert (later_info["step"], later_info["kind"], later_info["detail"]) == (
        "verify",
        *expected,
    )


@pytest.mark.asyncio
async def test_the_continuation_s_proxies_stop_before_the_verifier():
    """the live suffix and the LiteLLM log are read before test.sh,
    which runs as root, can stop the proxies or change their logs; one that
    cannot be stopped there is left to the teardown."""
    order: list[str] = []

    async def replay_stop():
        order.append("replay proxy")

    async def provider_stop(runtime):
        order.append(f"provider {runtime}")
        raise RuntimeError("exec channel lost")

    await _stop_continuation_proxies(
        SimpleNamespace(stop=replay_stop), "runtime", provider_stop
    )
    assert order == ["replay proxy", "provider runtime"]
    await _stop_continuation_proxies(None, None, provider_stop)
    assert order == ["replay proxy", "provider runtime"]


@pytest.mark.asyncio
async def test_a_stopped_replay_proxy_keeps_the_live_suffix_it_read(tmp_path):
    from pathlib import Path

    """The teardown stops the replay proxy again: the second stop must not
    read the removed log as no live calls."""
    from benchflow.continue_run.sandbox_proxy import SandboxReplayProxy

    live = tmp_path / "live_exchanges.jsonl"
    live.write_text(exchange(completion(content="live")).model_dump_json() + "\n")

    class Sandbox:
        async def download_file(self, source, target):
            Path(target).write_bytes(Path(source).read_bytes())

        async def exec(self, command, **_kwargs):
            if command.startswith("rm -rf"):
                live.unlink(missing_ok=True)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    proxy = SandboxReplayProxy(
        sandbox=Sandbox(),
        runtime_dir=str(tmp_path),
        port=1,
        pid_path=str(tmp_path / "pid"),
        live_log_path=str(live),
        state_path=str(tmp_path / "state.json"),
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
    )
    await proxy.stop()
    assert len(proxy.live_exchanges) == 1 and not live.exists()
    await proxy.stop()
    assert len(proxy.live_exchanges) == 1
