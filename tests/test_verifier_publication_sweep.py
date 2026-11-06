"""Regressions introduced by V6's 26da710a publication path."""

import os
from types import SimpleNamespace

import pytest

from benchflow.rollout._setup import _publish_trajectory_for_verifier
from tests.reachable_fixture import directory
from tests.test_acp_final_reply import RecordingEnv


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Answer: A\ud800", "Answer: A\udfff", "Answer: A 🌟"])
async def test_reply_encoding_is_total_and_does_not_truncate_linked_evidence(
    tmp_path, text
):
    """Encode first, replace inodes, retain JSONL."""
    target = tmp_path / "agent"
    target.mkdir()
    witness = tmp_path / "witness"
    witness.write_text("original evidence")
    os.link(witness, target / "output.txt")
    os.link(witness, target / "acp_trajectory.jsonl")
    env = RecordingEnv()
    await _publish_trajectory_for_verifier(
        env, [{"type": "agent_message", "text": text}], directory(target)
    )
    expected = (text + "\n").encode("utf-8", errors="backslashreplace")
    assert (target / "output.txt").read_bytes() == expected
    assert witness.read_text() == "original evidence"
    assert '"type": "agent_message"' in (target / "acp_trajectory.jsonl").read_text()
    assert any(value.encode("utf-8") == expected for value in env.uploads.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["mkdir -m", "chmod 644", "rc=0;"])
@pytest.mark.parametrize("transient", [False, True])
async def test_required_publication_retries_or_refuses_grading(
    tmp_path, step, transient
):
    """Absent evidence is not a legitimate zero."""
    env = RecordingEnv()
    original = env.exec
    failures = []

    async def exec_(command, **kwargs):
        # ``in``, not ``startswith``: the stage mkdir follows the defensive
        # ``mkdir -p /logs/agent;`` in the same exec (PY8 U12).
        if step in command and (not transient or not failures):
            failures.append(command)
            return SimpleNamespace(
                return_code=1, stdout="", stderr="transient transport failure"
            )
        return await original(command, **kwargs)

    env.exec = exec_
    call = _publish_trajectory_for_verifier(
        env, [{"type": "agent_message", "text": "done"}], directory(tmp_path / "agent")
    )
    if transient:
        await call
        assert len(failures) == 1
        assert any(c.startswith("rc=0;") for c, _ in env.execs)
    else:
        with pytest.raises(RuntimeError, match=r"verifier.*publication"):
            await call
        assert len(failures) == 2


@pytest.mark.asyncio
async def test_failed_publication_is_a_verifier_error_without_running_grader(
    tmp_path, monkeypatch
):
    """Guards the fix at Rollout.verify, not only its upload helper."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock

    import benchflow.rollout as mod
    from benchflow.rollout import Rollout, _setup

    VerifierPublicationError = getattr(_setup, "VerifierPublicationError", RuntimeError)

    for name in ("capture_terminal_workspace", "collect_rollout_artifacts"):
        monkeypatch.setattr(mod, name, AsyncMock())
    monkeypatch.setattr(mod, "require_safe_branch_world", lambda _r: None)
    monkeypatch.setattr(
        mod,
        "_publish_trajectory_for_verifier",
        AsyncMock(
            side_effect=VerifierPublicationError("verifier evidence publication failed")
        ),
    )
    grader = AsyncMock()
    monkeypatch.setattr(mod, "_verify_rollout", grader)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _r: "not eligible",
    )
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.mark_solver_complete", lambda _r: None
    )
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent="opencode", sandbox_user="agent", task_path=Path("/tmp/task")
    )
    rollout._task = SimpleNamespace(config=None)
    rollout._trajectory = [{"type": "agent_message", "text": "done"}]
    rollout._env = object()
    rollout._rollout_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    rollout._timing = {}
    rollout._planes = object()
    rollout._agent_cwd = "/app"
    rollout._diagnostics = MagicMock()
    assert await rollout.verify() is None
    assert "publication failed" in rollout._verifier_error
    grader.assert_not_awaited()
