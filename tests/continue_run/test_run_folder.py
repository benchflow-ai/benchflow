"""Tests for loading/validating an original run folder."""

from __future__ import annotations

import json

import pytest

from benchflow.continue_run.run_folder import (
    CONTINUE_SUPPORTED_AGENTS,
    ContinueUnsupportedError,
    MissingRecordingError,
    RunFolderError,
    UnsupportedAgentError,
    load_run_folder,
)

from ._helpers import completion, exchange, write_run_folder


def test_loads_valid_timeout_folder(tmp_path):
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[
            exchange(completion(content="a")),
            exchange(completion(content="b")),
        ],
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


# ── #1083 step 1(b): the continue gate must explain itself ────────────────────


def test_supported_agents_come_from_one_named_constant():
    """A future replay ingress updates exactly one place."""
    assert set(CONTINUE_SUPPORTED_AGENTS) == {"openhands"}


def test_unsupported_agent_error_names_agent_and_reason(tmp_path):
    """The gate says which agent this run used, what is supported, and WHY."""
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
    )
    with pytest.raises(UnsupportedAgentError) as excinfo:
        load_run_folder(folder)

    exc = excinfo.value
    message = str(exc)
    # names *this* run's agent, not a generic complaint
    assert "claude-agent-acp" in message
    # names the supported set, and says the set is protocol-derived
    assert "openhands" in message
    assert "wire protocol" in message
    # the actual reason: the replay proxy's ingress + the env the agent reads
    assert "/v1/chat/completions" in message
    assert "LLM_BASE_URL" in message
    # typed, machine-readable triage verdict
    assert isinstance(exc, ContinueUnsupportedError)
    assert exc.reason_code == "unsupported_agent"
    assert exc.agent == "claude-agent-acp"
    assert exc.supported_agents == ("openhands",)


def test_unsupported_agent_with_recording_is_recoverable_in_principle(tmp_path):
    """A recorded run is blocked on an ingress, not on the recording."""
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
    )
    with pytest.raises(UnsupportedAgentError) as excinfo:
        load_run_folder(folder)

    exc = excinfo.value
    assert exc.has_recording is True
    assert exc.recoverable_in_principle is True
    message = str(exc)
    assert "does have" in message
    assert "never be continued" not in message
    # the alternative route is offered conditionally, never promised
    assert "if" in message.lower() and "snapshot" in message


def test_unsupported_agent_without_recording_is_never_recoverable(tmp_path):
    """A subscription-auth run must hear the permanent verdict, not the agent one."""
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
    )
    (folder / "trajectory" / "llm_trajectory.jsonl").unlink()

    with pytest.raises(UnsupportedAgentError) as excinfo:
        load_run_folder(folder)

    exc = excinfo.value
    assert exc.has_recording is False
    assert exc.recoverable_in_principle is False
    message = str(exc)
    assert "claude-agent-acp" in message
    assert "never be continued" in message
    assert "subscription-auth" in message


def test_missing_recording_is_distinguished_from_wrong_agent(tmp_path):
    """Supported agent + no recording is a *different*, permanent verdict."""
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        agent="openhands",
    )
    (folder / "trajectory" / "llm_trajectory.jsonl").unlink()

    with pytest.raises(MissingRecordingError) as excinfo:
        load_run_folder(folder)

    exc = excinfo.value
    assert not isinstance(exc, UnsupportedAgentError)
    assert exc.reason_code == "no_llm_recording"
    assert exc.recoverable_in_principle is False
    message = str(exc)
    assert "subscription-auth" in message
    assert "never be continued" in message
    # does not blame the agent — this run's agent *is* supported
    assert "only the OpenHands agent template" not in message


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
