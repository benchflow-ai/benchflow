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
    resolve_task_path,
    select_proxy_mode,
    summarize_llm_trajectory_usage,
    update_continued_metadata,
)
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder
from benchflow.continue_run.trajectory_artifacts import (
    refresh_stitched_trajectory_manifest,
    stitched_trajectory_lines,
    write_stitched_trajectory,
)
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
    first = json.loads(lines[0])
    assert first["a"] == 1
    assert first["metadata"]["schema_version"] == 2
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
        live_attempt_count=1,
        live_errors=[],
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
    assert live_row["metadata"]["schema_version"] == 2


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
        live_attempt_count=1,
        live_errors=[],
    )

    assert manifest.status is CaptureStatus.PARTIAL
    assert manifest.capture_source is CaptureSource.MIXED
    assert manifest.capture_fidelity is CaptureFidelity.MIXED
    assert manifest.auth_mode is AuthMode.MIXED
    assert not capture_manifest_allows_training(
        manifest.model_dump(mode="json"), exchange_count=2
    )


def test_root_sandbox_live_suffix_is_retained_but_audit_only(tmp_path):
    """Guards PR #1057 against trusting root-writable continuation capture."""

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
    out = write_stitched_trajectory(
        rollout,
        source / "trajectory" / "llm_trajectory.jsonl",
        [exchange(completion(content="live"))],
        live_model=model,
        live_capture_trusted=False,
    )
    manifest = refresh_stitched_trajectory_manifest(
        rollout,
        source,
        original_model=model,
        live_model=model,
        n_recorded=1,
        n_live=1,
        live_attempt_count=1,
        live_errors=[],
        live_capture_trusted=False,
    )

    live_row = json.loads(out.read_text().splitlines()[-1])
    assert live_row["metadata"]["capture_fidelity"] == "agent_session"
    assert live_row["metadata"]["capture_custody"] == "agent_writable_sandbox"
    assert manifest.status is CaptureStatus.PARTIAL
    assert manifest.capture_fidelity is CaptureFidelity.MIXED
    assert any("shared root custody" in error for error in manifest.errors)
    assert not capture_manifest_allows_training(
        manifest.model_dump(mode="json"), exchange_count=2
    )


def test_refresh_stitched_manifest_rejects_missing_live_attempt(tmp_path):
    """Guards PR #1057 against completing a lost continuation exchange."""

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
    )
    write_llm_trajectory_manifest(source, source_manifest)
    rollout = tmp_path / "continued"
    initialize_llm_trajectory_artifacts(
        rollout,
        agent="openhands",
        model=None,
        session_id="continued",
        started_at=source_manifest.started_at,
    )
    stitched = write_stitched_trajectory(
        rollout,
        source / "trajectory" / "llm_trajectory.jsonl",
        [],
        live_model=model,
    )

    manifest = refresh_stitched_trajectory_manifest(
        rollout,
        source,
        original_model=model,
        live_model=model,
        n_recorded=1,
        n_live=0,
        live_attempt_count=1,
        live_errors=["live provider request failed before capture"],
    )

    assert manifest.status is CaptureStatus.PARTIAL
    assert manifest.request_complete is False
    assert manifest.response_complete is False
    assert "live_provider_exchange" in manifest.missing_fields
    assert any("count mismatch" in error for error in manifest.errors)
    recorded_row = json.loads(stitched.read_text())
    assert recorded_row["metadata"]["schema_version"] == 2


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

    async def before_cleanup(teardown_errors):
        events.append("artifact")
        assert rollout._error is not None
        assert len(teardown_errors) == 2

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
async def test_sandbox_teardown_passes_errors_when_agent_already_failed():
    """Guards PR #1057 against hiding capture teardown behind an agent error."""

    class FailingProxy:
        async def stop(self):
            raise RuntimeError("capture state unavailable")

    class FakeRollout:
        _error = "agent failed first"

        async def cleanup(self):
            return None

    observed: list[str] = []

    async def before_cleanup(teardown_errors):
        observed.extend(teardown_errors)

    async def stop_provider_runtime(_runtime):
        return None

    rollout = FakeRollout()
    errors = await _safe_sandbox_continuation_teardown(
        rollout=rollout,
        replay_proxy=FailingProxy(),
        provider_runtime=None,
        stop_provider_runtime=stop_provider_runtime,
        before_cleanup=before_cleanup,
    )

    assert errors == observed
    assert errors and "capture state unavailable" in errors[0]
    assert rollout._error == "agent failed first"


@pytest.mark.asyncio
async def test_sandbox_teardown_refinalizes_manifest_after_cleanup(tmp_path):
    """Guards PR #1057 against cleanup overwriting teardown capture errors."""
    manifest_path = tmp_path / "manifest.json"

    class FailingProxy:
        async def stop(self):
            raise RuntimeError("capture stop failed")

    class FakeRollout:
        _error = "agent failed first"

        async def cleanup(self):
            manifest_path.write_text(json.dumps({"status": "complete"}))

    async def stop_provider_runtime(_runtime):
        return None

    async def before_cleanup(teardown_errors):
        assert teardown_errors
        manifest_path.write_text(json.dumps({"status": "partial"}))

    async def after_cleanup(teardown_errors):
        assert teardown_errors
        manifest_path.write_text(json.dumps({"status": "partial"}))

    await _safe_sandbox_continuation_teardown(
        rollout=FakeRollout(),
        replay_proxy=FailingProxy(),
        provider_runtime=None,
        stop_provider_runtime=stop_provider_runtime,
        before_cleanup=before_cleanup,
        after_cleanup=after_cleanup,
    )

    assert json.loads(manifest_path.read_text())["status"] == "partial"


def test_update_continued_metadata_rebuilds_trainer_results(tmp_path):
    """Guards PR #1057 against retaining the pre-stitch trainer row."""
    rollout = tmp_path / "job" / "demo-task__continued"
    (rollout / "trajectory").mkdir(parents=True)
    model = "openai/gpt-5.5"
    row = exchange(completion(content="done")).model_dump(mode="json")
    row["request"]["body"]["messages"] = [{"role": "user", "content": "Do the task."}]
    row["metadata"] = {
        "schema_version": 2,
        "capture_source": "litellm_proxy",
        "capture_fidelity": "provider_wire",
        "auth_mode": "api_key",
        "request_complete": True,
        "response_complete": True,
        "payload_redacted": True,
    }
    trajectory_path = rollout / "trajectory" / "llm_trajectory.jsonl"
    trajectory_path.write_text(json.dumps(row) + "\n")
    manifest = LLMTrajectoryManifest(
        status=CaptureStatus.COMPLETE,
        capture_source=CaptureSource.LITELLM_PROXY,
        capture_fidelity=CaptureFidelity.PROVIDER_WIRE,
        auth_mode=AuthMode.API_KEY,
        agent="openhands",
        model=model,
        session_id="continued",
        exchange_count=1,
        request_complete=True,
        response_complete=True,
        started_at="2026-08-29T00:00:00Z",
        finished_at="2026-08-29T00:01:00Z",
    )
    write_llm_trajectory_manifest(rollout, manifest)
    (rollout / "config.json").write_text(json.dumps({"model": None, "source": {}}))
    (rollout / "prompts.json").write_text(json.dumps(["Do the task."]))
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "demo-task",
                "rollout_name": "demo-task__continued",
                "agent": "openhands",
                "agent_name": "OpenHands",
                "model": None,
                "n_tool_calls": 0,
                "partial_trajectory": False,
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "export_error": None,
                "timing": {},
                "agent_result": {"total_tokens": 0, "usage_source": "unavailable"},
                "usage_tracking": {"requested": "off", "status": "off"},
            }
        )
    )
    (rollout / "results.jsonl").write_text(
        json.dumps({"info": {"training_ready": False, "model": None}}) + "\n"
    )

    update_continued_metadata(
        rollout,
        live_model=model,
        usage=summarize_llm_trajectory_usage(trajectory_path, n_recorded=0),
        environment="docker",
    )

    refreshed = json.loads((rollout / "results.jsonl").read_text())
    aggregated = json.loads((rollout.parent / "results.jsonl").read_text())
    assert refreshed["info"]["model"] == model
    assert refreshed["info"]["training_ready"] is True
    assert refreshed["token_usage"]["total_tokens"] == 2
    assert len(refreshed["trajectory"]) == 1
    assert aggregated == refreshed


def test_stitching_structurally_redacts_escaped_secret(tmp_path):
    """Guards PR #1057 against string redaction corrupting stitched JSON."""
    secret = "ESCbearerSECRETtok123456"
    source = tmp_path / "source.jsonl"
    payload = exchange(completion(content="done")).model_dump(mode="json")
    payload["request"]["body"]["authorization"] = f'Bearer {secret}\\"tail'
    source.write_text(json.dumps(payload) + "\n")

    rendered = stitched_trajectory_lines(source, [])

    assert len(rendered) == 1
    restored = json.loads(rendered[0])
    assert secret not in rendered[0]
    assert "***REDACTED***" in restored["request"]["body"]["authorization"]
    assert restored["metadata"]["schema_version"] == 2


def test_refresh_stitched_manifest_rejects_malformed_row(tmp_path):
    """Guards PR #1057 against a complete sidecar for invalid stitched JSONL."""
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
    )
    write_llm_trajectory_manifest(source, source_manifest)
    rollout = tmp_path / "continued"
    initialize_llm_trajectory_artifacts(
        rollout,
        agent="openhands",
        model=None,
        session_id="continued",
        started_at=source_manifest.started_at,
    )
    stitched = rollout / "trajectory" / "llm_trajectory.jsonl"
    stitched.write_text('{"request": broken}\n')

    manifest = refresh_stitched_trajectory_manifest(
        rollout,
        source,
        original_model=model,
        live_model=model,
        n_recorded=1,
        n_live=0,
        live_attempt_count=0,
        live_errors=[],
    )

    assert manifest.status is CaptureStatus.PARTIAL
    assert manifest.request_complete is False
    assert manifest.response_complete is False
    assert "valid_provider_exchange" in manifest.missing_fields
    assert any("malformed row" in error for error in manifest.errors)
