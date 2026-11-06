"""The ACP agent's final reply where verifiers read it (#1182), and the
links the agent can leave at the names publishing writes.

Registry tasks that grade the reply read ``/logs/agent/output.txt`` (MMMLU's
``test.sh`` tries ``oracle.txt``, ``gemini-cli.txt``, ``codex.txt``, then
``output.txt``); before #1182 nothing wrote it for an ACP agent, so every
trial read an empty answer. The real-container half of these checks is in
``test_agent_sandbox_files_container.py``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout import Rollout
from benchflow.rollout._setup import (
    _publish_trajectory_for_verifier,
    final_agent_reply,
)
from tests.reachable_fixture import directory

ANSWER_TRAJECTORY = [
    {"type": "user_message", "text": "Which is right? Reply with Answer: X."},
    {"type": "agent_message", "text": "Let me think. Answer: B looks wrong."},
    {"type": "tool_call", "tool_call_id": "t1", "kind": "execute", "title": "ls"},
    {"type": "agent_thought", "text": "C fits."},
    {"type": "agent_message", "text": "Answer: C"},
]


class RecordingEnv:
    """A sandbox that records publishing: uploads by target, execs by user."""

    def __init__(self) -> None:
        self.execs: list[tuple[str, str | None]] = []
        self.uploads: dict[str, str] = {}

    async def exec(self, command: str, user: str | None = None, timeout_sec=None):
        self.execs.append((command, user))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source: Path | str, target: str, **_kwargs) -> None:
        self.uploads[target] = Path(source).read_text()


def _staged(env: RecordingEnv, name: str) -> str:
    (target,) = [t for t in env.uploads if t.endswith(f"/{name}")]
    assert target.startswith("/logs/.benchflow-publish-"), target
    return env.uploads[target]


# --- which text is the final reply ------------------------------------------


def test_final_reply_is_the_last_message_of_the_last_turn() -> None:
    assert final_agent_reply(ANSWER_TRAJECTORY) == "Answer: C"


def test_final_reply_joins_text_split_only_by_thoughts() -> None:
    trajectory = [
        {"type": "user_message", "text": "q"},
        {"type": "agent_message", "text": "Answer:"},
        {"type": "agent_thought", "text": "sure?"},
        {"type": "notice", "severity": "info", "title": "model fallback"},
        {"type": "agent_message", "text": " D"},
    ]
    assert final_agent_reply(trajectory) == "Answer: D"


def test_final_reply_keeps_the_text_before_a_closing_tool_call() -> None:
    trajectory = [
        {"type": "user_message", "text": "q"},
        {"type": "agent_message", "text": "Answer: A"},
        {"type": "tool_call", "tool_call_id": "t", "kind": "execute", "title": "x"},
    ]
    assert final_agent_reply(trajectory) == "Answer: A"


@pytest.mark.parametrize("blank", ["", "\n", "  "])
def test_final_reply_skips_a_blank_chunk_after_a_closing_tool_call(blank) -> None:
    """Guards V6-R1-A-1: the session records empty agent_message_chunk texts,
    so a blank message after the tool call must not hide the reply."""
    trajectory = [
        {"type": "user_message", "text": "q"},
        {"type": "agent_message", "text": "Answer: A"},
        {"type": "tool_call", "tool_call_id": "t", "kind": "execute", "title": "x"},
        {"type": "agent_message", "text": blank},
    ]
    assert final_agent_reply(trajectory) == "Answer: A"


def test_final_reply_never_comes_from_an_earlier_turn_or_a_subagent() -> None:
    trajectory = [
        {"type": "user_message", "text": "first"},
        {"type": "agent_message", "text": "Answer: A"},
        {"type": "user_message", "text": "second"},
        {"type": "agent_message", "text": "Answer: B", "parent_tool_call_id": "t"},
    ]
    assert final_agent_reply(trajectory) is None
    assert final_agent_reply([{"type": "oracle", "command": "solve.sh"}]) is None
    assert final_agent_reply([]) is None


# --- publishing ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_reply_is_published_to_logs_agent_output_txt(tmp_path) -> None:
    """Guards #1182: MMMLU-style verifiers read the reply from output.txt."""
    env = RecordingEnv()
    agent_dir = tmp_path / "agent"

    await _publish_trajectory_for_verifier(
        env, ANSWER_TRAJECTORY, directory(agent_dir), sandbox_user="agent"
    )

    assert _staged(env, "output.txt") == "Answer: C\n"
    assert (agent_dir / "output.txt").read_text() == "Answer: C\n"
    copies = [(c, u) for c, u in env.execs if "> /logs/agent/output.txt" in c]
    # Copied into place as the agent user, never written there by root.
    assert [u for _, u in copies] == ["agent"]
    assert "rm -f /logs/agent/output.txt" in copies[0][0]
    # The staging folder is removed afterwards.
    assert env.execs[-1][0].startswith("rm -rf /logs/.benchflow-publish-")
    assert env.execs[-1][1] == "root"


@pytest.mark.asyncio
async def test_no_reply_publishes_empty_output_and_the_trajectory(tmp_path) -> None:
    """Guards CONF-rollout-1: a no-text turn must replace 82e412f1's stale reply."""
    env = RecordingEnv()
    trajectory = [{"type": "user_message", "text": "q"}]

    await _publish_trajectory_for_verifier(
        env, trajectory, directory(tmp_path / "agent")
    )

    assert {t.rsplit("/", 1)[1] for t in env.uploads} == {
        "acp_trajectory.jsonl",
        "output.txt",
    }
    assert (tmp_path / "agent" / "output.txt").read_text() == ""


@pytest.mark.asyncio
async def test_published_reply_is_redacted_like_the_trajectory(tmp_path) -> None:
    env = RecordingEnv()
    trajectory = [
        {"type": "user_message", "text": "q"},
        {"type": "agent_message", "text": "GITHUB_TOKEN=abc123def456ghi\nAnswer: C"},
    ]

    await _publish_trajectory_for_verifier(
        env, trajectory, directory(tmp_path / "agent")
    )

    published = _staged(env, "output.txt")
    assert "abc123def456ghi" not in published
    assert published.endswith("Answer: C\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["acp_trajectory.jsonl", "output.txt"])
async def test_host_copy_never_follows_a_link_the_agent_left(tmp_path, name) -> None:
    """On Docker /logs/agent is the host agent/ folder, which the agent user
    owns: a link it leaves there must not send the host write elsewhere."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    victim = tmp_path / "victim-a"
    victim.write_text("original\n")
    (agent_dir / name).symlink_to(victim)

    await _publish_trajectory_for_verifier(
        RecordingEnv(), ANSWER_TRAJECTORY, directory(agent_dir), sandbox_user="agent"
    )

    assert victim.read_text() == "original\n"
    assert not (agent_dir / name).is_symlink()
    assert (agent_dir / name).read_text().strip()


@pytest.mark.asyncio
async def test_host_copy_that_cannot_be_written_does_not_fail_the_rollout(
    tmp_path,
) -> None:
    agent_dir = tmp_path / "agent"
    (agent_dir / "output.txt").mkdir(parents=True)
    env = RecordingEnv()

    await _publish_trajectory_for_verifier(env, ANSWER_TRAJECTORY, directory(agent_dir))

    assert _staged(env, "output.txt") == "Answer: C\n"


@pytest.mark.asyncio
async def test_host_copy_replaces_a_fifo_instead_of_waiting_on_it(tmp_path) -> None:
    """A FIFO at the name would block an O_WRONLY open with no reader."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    os.mkfifo(agent_dir / "output.txt")

    await asyncio.wait_for(
        _publish_trajectory_for_verifier(
            RecordingEnv(),
            ANSWER_TRAJECTORY,
            directory(agent_dir),
            sandbox_user="agent",
        ),
        timeout=10,
    )

    assert (agent_dir / "output.txt").read_text() == "Answer: C\n"


@pytest.mark.asyncio
async def test_each_file_is_copied_on_its_own(tmp_path) -> None:
    """One name the agent blocked (a folder) must not hold back the other."""
    env = RecordingEnv()

    await _publish_trajectory_for_verifier(
        env, ANSWER_TRAJECTORY, directory(tmp_path / "agent"), sandbox_user="agent"
    )

    (copy,) = [c for c, _ in env.execs if "> /logs/agent/" in c]
    assert copy.startswith("rc=0; ") and copy.endswith("; exit $rc")
    assert copy.count("|| rc=1") == 2


@pytest.mark.asyncio
async def test_rollout_verify_publishes_as_the_sandbox_user(monkeypatch) -> None:
    """The rollout passes its sandbox user, so the copy runs as that user."""
    import benchflow.rollout as rollout_mod

    publish = AsyncMock()
    monkeypatch.setattr(rollout_mod, "_publish_trajectory_for_verifier", publish)
    monkeypatch.setattr(
        rollout_mod,
        "_verify_rollout",
        AsyncMock(return_value=({"reward": 1}, None, None)),
    )
    monkeypatch.setattr(rollout_mod, "capture_terminal_workspace", AsyncMock())
    monkeypatch.setattr(rollout_mod, "collect_rollout_artifacts", AsyncMock())
    monkeypatch.setattr(rollout_mod, "require_safe_branch_world", lambda _r: None)
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent="opencode", sandbox_user="agent", task_path=Path("/nowhere")
    )
    rollout._env = object()
    rollout._task = SimpleNamespace(config=None)
    rollout._trajectory = list(ANSWER_TRAJECTORY)
    rollout._rollout_paths = SimpleNamespace(agent_dir=Path("/nowhere/agent"))
    # Host-side record writers use the trial's directory, including children.
    rollout._rollout_dir = Path("/nowhere")
    rollout._timing = {}
    rollout._planes = object()
    rollout._agent_cwd = "/app"
    rollout._diagnostics = MagicMock()
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _r: "not eligible",
    )
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.mark_solver_complete", lambda _r: None
    )

    await rollout.verify()

    assert publish.await_args.kwargs == {"sandbox_user": "agent", "publish_reply": True}
