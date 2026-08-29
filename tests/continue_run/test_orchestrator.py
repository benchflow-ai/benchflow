"""Tests for the pure orchestration helpers (no sandbox / no network)."""

from __future__ import annotations

import json

import pytest

from benchflow.continue_run.orchestrator import (
    LiteLLMLiveForwarder,
    _safe_sandbox_continuation_teardown,
    build_agent_env,
    build_rollout_config,
    continued_rollout_name,
    refresh_stitched_trajectory_manifest,
    resolve_task_path,
    select_proxy_mode,
    stitched_trajectory_lines,
    summarize_llm_trajectory_usage,
    update_continued_metadata,
    write_stitched_trajectory,
)
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder
from benchflow.trajectories.llm_capture_manifest import (
    AuthMode,
    CaptureFidelity,
    CaptureSource,
    CaptureStatus,
    LLMTrajectoryManifest,
    capture_manifest_allows_training,
    initialize_llm_trajectory_artifacts,
    write_llm_trajectory_manifest,
)

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


def test_refresh_stitched_manifest_replaces_pre_stitch_finalization(tmp_path):
    """Guards PR #1057 against stale continuation capture manifests."""

    model = "openai/gpt-5.5"
    source = write_run_folder(
        tmp_path / "source",
        exchanges=[exchange(completion(content="recorded"))],
        model=model,
    )
    source_manifest = LLMTrajectoryManifest(
        status=CaptureStatus.COMPLETE,
        capture_source=CaptureSource.LITELLM_PROXY,
        capture_fidelity=CaptureFidelity.PROVIDER_WIRE,
        auth_mode=AuthMode.API_KEY,
        agent="openhands",
        model=model,
        session_id="source",
        exchange_count=1,
        request_complete=True,
        response_complete=True,
        started_at="2026-08-29T00:00:00Z",
        finished_at="2026-08-29T00:01:00Z",
    )
    write_llm_trajectory_manifest(source, source_manifest)

    rollout = tmp_path / "continued"
    initialize_llm_trajectory_artifacts(
        rollout,
        agent="openhands",
        model=None,
        session_id="continued",
        started_at=source_manifest.finished_at,
    )
    live = [exchange(completion(content="live"))]
    out = write_stitched_trajectory(
        rollout,
        source / "trajectory" / "llm_trajectory.jsonl",
        live,
        live_model=model,
    )
    manifest = refresh_stitched_trajectory_manifest(
        rollout,
        source,
        original_model=model,
        live_model=model,
        n_recorded=1,
        n_live=1,
    )

    assert manifest.status is CaptureStatus.COMPLETE
    assert manifest.capture_fidelity is CaptureFidelity.PROVIDER_WIRE
    assert manifest.exchange_count == 2
    assert capture_manifest_allows_training(
        manifest.model_dump(mode="json"), exchange_count=2
    )
    live_row = json.loads(out.read_text().splitlines()[-1])
    assert live_row["metadata"]["capture_fidelity"] == "provider_wire"
    assert live_row["metadata"]["model"] == model


def test_refresh_stitched_manifest_keeps_lower_fidelity_prefix_partial(tmp_path):
    """Guards PR #1057 against promoting audit-only continuation prefixes."""

    source = write_run_folder(
        tmp_path / "source",
        exchanges=[exchange(completion(content="recorded"))],
        model="claude-sonnet-4-6",
    )
    source_manifest = LLMTrajectoryManifest(
        status=CaptureStatus.PARTIAL,
        capture_source=CaptureSource.CLAUDE_NATIVE_SESSION,
        capture_fidelity=CaptureFidelity.AGENT_SESSION,
        auth_mode=AuthMode.OAUTH_SUBSCRIPTION,
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        session_id="source",
        exchange_count=1,
        request_complete=False,
        response_complete=True,
        started_at="2026-08-29T00:00:00Z",
        finished_at="2026-08-29T00:01:00Z",
        missing_fields=["provider_request"],
    )
    write_llm_trajectory_manifest(source, source_manifest)

    rollout = tmp_path / "continued"
    initialize_llm_trajectory_artifacts(
        rollout,
        agent="openhands",
        model=None,
        session_id="continued",
        started_at=source_manifest.finished_at,
    )
    write_stitched_trajectory(
        rollout,
        source / "trajectory" / "llm_trajectory.jsonl",
        [exchange(completion(content="live"))],
        live_model="openai/gpt-5.5",
    )
    manifest = refresh_stitched_trajectory_manifest(
        rollout,
        source,
        original_model="claude-sonnet-4-6",
        live_model="openai/gpt-5.5",
        n_recorded=1,
        n_live=1,
    )

    assert manifest.status is CaptureStatus.PARTIAL
    assert manifest.capture_source is CaptureSource.MIXED
    assert manifest.capture_fidelity is CaptureFidelity.MIXED
    assert manifest.auth_mode is AuthMode.MIXED
    assert not capture_manifest_allows_training(
        manifest.model_dump(mode="json"), exchange_count=2
    )


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
    assert result["agent_result"]["usage_source"] == "provider_response"
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
