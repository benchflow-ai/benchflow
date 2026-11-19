"""A sandbox lost mid-turn keeps the calls its agent already logged.

On the ACP path a turn's tokens arrive with its end-of-turn report. A sandbox
lost before it (0.9.3's confirmed gap; X2 saw a lost sandbox remove both
measurements) took the agent's session log with it: Claude Code's
``~/.claude/projects`` and Codex's rollouts stay in the sandbox, and the ACP
adapter's updates carry the context's size, not each call's tokens. While a
turn runs, the log's new bytes are copied to the trial folder; at teardown,
when the log cannot be copied, that copy is counted.
"""

from __future__ import annotations

import asyncio
import json
import shlex

import pytest

from benchflow.acp.session import ACPSession
from benchflow.rollout._cost import INCOMPLETE_MIRROR
from benchflow.sandbox.protocol import ExecResult
from tests.test_codex_acp_tokens import THREE_CALLS, _codex_log, _summed
from tests.test_en374_salvage import (
    OPUS,
    OPUS_RATES,
    _claude_session_log,
    _rollout,
    _tokens,
    _usage,
)

LOST = "container is not running"


class Sandbox:
    """A local sandbox whose commands run in a shell, until it is lost."""

    def __init__(self) -> None:
        self.lost = False
        self.commands: list[str] = []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        if self.lost:
            raise RuntimeError(LOST)
        self.commands.append(cmd)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        assert process.returncode is not None
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def stop(self, *, delete: bool = True) -> None:
        pass


def _acp_rollout(tmp_path, monkeypatch, agent):
    from benchflow.rollout import _session_mirror

    monkeypatch.setattr(_session_mirror, "MIRROR_INTERVAL_SEC", 0.05)
    home = tmp_path / "sandbox-home"
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent=agent, harness="acp", home=home
    )
    sandbox = Sandbox()
    rollout._env = sandbox
    rollout._timeout = 60
    rollout._resolved_prompts = ["Build the site."]
    rollout._acp_client = object()
    rollout._session = ACPSession("s")
    return rollout, trial, sandbox, home


async def _turn_lost_after(rollout, monkeypatch, sandbox, before_loss, at_loss=None):
    """The agent logs *before_loss*, a few copies go by, then it logs
    *at_loss* and the sandbox is lost before the turn reports."""

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        before_loss()
        await asyncio.sleep(0.4)
        if at_loss is not None:
            at_loss()
        sandbox.lost = True
        raise RuntimeError(LOST)

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    with pytest.raises(RuntimeError, match=LOST):
        await rollout.execute()


@pytest.mark.asyncio
async def test_a_claude_turn_lost_mid_turn_keeps_its_logged_responses(
    tmp_path, monkeypatch
):
    rollout, trial, sandbox, home = _acp_rollout(
        tmp_path, monkeypatch, "claude-agent-acp"
    )
    first = _usage(1500, 0, 3000, 120)
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"

    def a_second_response_never_copied():
        with log.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "requestId": "req_late",
                        "message": {
                            "id": "msg_late",
                            "model": OPUS,
                            "usage": _usage(20, 3000, 400, 300),
                        },
                    }
                )
                + "\n"
            )

    await _turn_lost_after(
        rollout,
        monkeypatch,
        sandbox,
        lambda: _claude_session_log(home, [(OPUS, first)]),
        a_second_response_never_copied,
    )
    assert any("tail -c" in cmd for cmd in sandbox.commands)
    rollout._env = None  # the run lost its sandbox

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == {
        "n_input_tokens": 1500,
        "n_output_tokens": 120,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 3000,
        "total_tokens": 1500 + 120 + 3000,
    }
    assert usage["usage_source"] == "session_log"
    # The response logged after the last copy is missing: the count says so
    # (review of PR #122, M1), and so does the cost priced from it.
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == INCOMPLETE_MIRROR
    assert usage["cost_usd"] == pytest.approx(
        1500 * OPUS_RATES[0] + 120 * OPUS_RATES[1] + 3000 * OPUS_RATES[3]
    )
    assert usage["cost_source"] == "list_price_estimate"
    assert (trial / "agent" / "claude-sessions").is_dir()
    assert not (trial / ".session-mirror").exists()


@pytest.mark.asyncio
async def test_a_codex_turn_lost_mid_turn_keeps_its_logged_calls(tmp_path, monkeypatch):
    rollout, trial, sandbox, home = _acp_rollout(tmp_path, monkeypatch, "codex-acp")
    await _turn_lost_after(
        rollout, monkeypatch, sandbox, lambda: _codex_log(home, THREE_CALLS)
    )
    rollout._env = None

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == _summed(THREE_CALLS)
    assert usage["usage_source"] == "session_log"
    # The turn's last copy failed with the sandbox: a call after the last
    # tick would be missing, so the count is marked (review of PR #122, M1).
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == INCOMPLETE_MIRROR
    assert usage["cost_source"] == "list_price_estimate"
    assert (trial / "agent" / "codex-sessions").is_dir()
    model_calls = json.loads((trial / "model-calls.json").read_text())
    assert model_calls["calls"] == 3


@pytest.mark.asyncio
async def test_the_copy_is_redacted_and_gives_way_to_the_full_copy(
    tmp_path, monkeypatch
):
    """The copy holds no credential the agent was given, and a sandbox still
    up at teardown gives the full copy, the live one removed."""
    from benchflow.rollout._session_mirror import MIRROR_DIR, SessionMirror

    rollout, trial, sandbox, home = _acp_rollout(
        tmp_path, monkeypatch, "claude-agent-acp"
    )
    # A credential no pattern knows: only the run's own values find it.
    secret = "zq9-FAKE-plain-credential-value-77"
    rollout._agent_env = {"BENCHFLOW_PROVIDER_API_KEY": secret}
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps({"type": "user", "message": {"content": f"key {secret}"}})
        + "\n"
        + '{"type": "assistant", "partial'
    )
    mirror = SessionMirror(rollout, f"{home}/.claude/projects", "claude-sessions")
    await mirror.sync()
    copied = trial / MIRROR_DIR / "claude-sessions" / "-app" / "session-1.jsonl"
    text = copied.read_text()
    assert secret not in text and "[redacted]" in text
    # The line still being written waits for its end.
    assert text.count("\n") == 1 and "partial" not in text
    with log.open("a") as handle:
        handle.write(" : 1}\n")
    await mirror.sync()
    assert copied.read_text().count("\n") == 2
    assert shlex.quote(f"{home}/.claude/projects") in sandbox.commands[0]


@pytest.mark.asyncio
async def test_a_failed_session_factory_turn_is_still_salvaged(tmp_path, monkeypatch):
    """A session-factory turn that fails mid-prompt is typed
    AgentTurnFailedError (#142), not a timeout; it ended without its usage
    report all the same, so teardown counts the calls its agent logged."""
    from benchflow.agents.errors import AgentTurnFailedError

    rollout, _, _, home = _acp_rollout(tmp_path, monkeypatch, "claude-agent-acp")
    rollout._is_session_factory = True
    first = _usage(1500, 0, 3000, 120)

    async def prompts(session, prompt_list, *_args, **_kwargs):
        _claude_session_log(home, [(OPUS, first)])
        raise AgentTurnFailedError(
            "session factory error: provider returned HTTP 503",
            trajectory=[{"type": "user_message", "text": prompt_list[0]}],
            executed_prompts=prompt_list,
        )

    monkeypatch.setattr(rollout._planes, "execute_prompts_session_factory", prompts)
    with pytest.raises(AgentTurnFailedError):
        await rollout.execute()
    assert rollout._usage_report_missing is True

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["usage_source"] == "session_log"
    assert usage["n_output_tokens"] == 120
    assert usage["total_tokens"] == 1500 + 120 + 3000
