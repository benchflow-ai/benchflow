"""Tests for the pure orchestration helpers (no sandbox / no network)."""

from __future__ import annotations

import json

import pytest

from benchflow.continue_run.orchestrator import (
    LiteLLMLiveForwarder,
    build_agent_env,
    build_rollout_config,
    resolve_task_path,
    stitched_trajectory_lines,
    write_stitched_trajectory,
)
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder

from ._helpers import completion, exchange, write_run_folder


def _load(tmp_path, **kw):
    folder = write_run_folder(tmp_path / "run", exchanges=[exchange(completion(content="a"))], **kw)
    return load_run_folder(folder)


def test_build_agent_env_points_at_proxy():
    env = build_agent_env("http://10.0.0.1:9000/v1")
    assert env["LLM_BASE_URL"] == "http://10.0.0.1:9000/v1"
    assert env["LLM_MODEL"].startswith("openai/")
    assert env["LLM_API_KEY"]


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
    out = write_stitched_trajectory(rollout_dir, original, [exchange(completion(content="L"))])
    assert out == rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    assert len(out.read_text().strip().splitlines()) == 2
