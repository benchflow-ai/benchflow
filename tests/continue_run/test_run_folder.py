"""Tests for loading/validating an original run folder."""

from __future__ import annotations

import json

import pytest

from benchflow.continue_run.run_folder import RunFolderError, load_run_folder

from ._helpers import completion, exchange, write_run_folder


def test_loads_valid_timeout_folder(tmp_path):
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a")), exchange(completion(content="b"))],
        prompts=["Solve it."],
    )
    run = load_run_folder(folder)

    assert run.agent == "openhands"
    assert run.task_name == "demo-task"
    assert run.environment == "docker"
    assert run.timeout_sec == 3600
    assert run.agent_idle_timeout_sec == 600
    assert run.is_timeout is True
    assert run.prompts == ["Solve it."]
    assert run.n_recorded_exchanges == 2
    # response bodies survive the round-trip
    assert run.exchanges[0].response.body["choices"][0]["message"]["content"] == "a"


def test_missing_config_is_error(tmp_path):
    folder = tmp_path / "run"
    (folder / "trajectory").mkdir(parents=True)
    (folder / "trajectory" / "llm_trajectory.jsonl").write_text("{}\n")
    with pytest.raises(RunFolderError, match="missing required artifact"):
        load_run_folder(folder)


def test_missing_llm_trajectory_is_error(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"agent": "openhands"}))
    with pytest.raises(RunFolderError, match="record-replay needs the LLM"):
        load_run_folder(folder)


def test_empty_trajectory_is_error(tmp_path):
    folder = write_run_folder(tmp_path / "run", exchanges=[])
    with pytest.raises(RunFolderError, match="no usable LLM exchanges"):
        load_run_folder(folder)


def test_non_openhands_agent_rejected(tmp_path):
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
    )
    with pytest.raises(RunFolderError, match="openhands"):
        load_run_folder(folder)


def test_non_timeout_warns_but_loads_by_default(tmp_path):
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        error_category="agent_error",
    )
    run = load_run_folder(folder)  # permissive default — warn only
    assert run.is_timeout is False


def test_require_timeout_rejects_non_timeout(tmp_path):
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        error_category="agent_error",
    )
    with pytest.raises(RunFolderError, match="not a"):
        load_run_folder(folder, require_timeout=True)


def test_malformed_line_skipped_not_fatal(tmp_path):
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
    )
    traj = folder / "trajectory" / "llm_trajectory.jsonl"
    traj.write_text(traj.read_text() + "this is not json\n")
    run = load_run_folder(folder)
    assert run.n_recorded_exchanges == 1  # bad line dropped, good one kept
