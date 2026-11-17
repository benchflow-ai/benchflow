"""Copying the agent's session log never holds up its turn.

Review of PR #122 (F2b-R122B, 2026-11-13, L-d; required by the lead): both
copies of the session log redacted on the event loop. One 4 MB tick of the
copy made while a turn runs held the loop 3.8 s, and the teardown copy of a
27 MB log 24-34 s: the agent's turn, its heartbeats and every other trial of
the run stood still meanwhile. The redaction now runs in a worker thread,
and what the loop still does (the reads, the appends) is small.

The turn here answers every 50 ms while the copy takes a 27 MB Claude Code
log, a credential and a token-shaped string in it; then the teardown copy
takes the same log with a heartbeat beside it. Every value is made up.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time

import pytest

from benchflow.rollout import _session_mirror
from benchflow.rollout._session_log import SESSION_DIR, _copy_out
from tests.test_en374_salvage import OPUS, _rollout
from tests.test_en374_session_mirror import _acp_rollout

SECRET = "zq9-FAKE-plain-credential-value-77"
TOKEN = "sk-ant-api03-" + "Zx9FAKEq" * 11 + "AA"
SIZE = 27 * 1024 * 1024
# The redaction never runs on the event loop's thread (checked below, which
# no load can make flaky), and the turn's answers never wait long for it.
# The review measured the gap (F2b-R122C, 2026-11-14, 280 runs): at most
# 0.37 s on a quiet machine; 0.65 s (mirror) and 0.83 s (teardown) with its
# cores' worth of other work; 0.92 s and 1.11 s with twice that. What is left
# is the GIL, held through one pattern scan of one string (moves the
# redaction to another process). On the loop, the copies held it 1.06 s
# (mirror) and 5.5 s (teardown) on this tree, and 1.9 s at 40c582e8c.
MAX_GAP_SEC = 1.5
PROSE = "read the file, run the tests and fix the failing case; " * 40


def _log() -> str:
    """A 27 MB Claude Code session log: assistant answers and tool results
    up to 1 MB, some carrying the credential or a token."""
    lines, size, i = [], 0, 0
    while size < SIZE:
        text = PROSE * (1 + (i * 7919) % 400)
        if i % 25 == 0:
            text = f"key {SECRET} and {TOKEN} then " + text
        kind = "assistant" if i % 2 else "user"
        line = json.dumps(
            {
                "type": kind,
                "uuid": f"u-{i}",
                "message": {
                    "id": f"msg_{i}",
                    "model": OPUS,
                    "role": kind,
                    "content": [{"type": "text", "text": text}],
                },
            }
        )
        lines.append(line)
        size += len(line) + 1
        i += 1
    return "\n".join(lines) + "\n"


def _check_redacted(copy: str, source: str) -> None:
    assert copy.count("\n") == source.count("\n")
    assert SECRET not in copy and TOKEN not in copy
    assert copy.count("[redacted]") == source.count(SECRET)
    first = json.loads(copy.split("\n", 1)[0])
    assert first["message"]["content"][0]["text"].startswith("key [redacted] and ")


@pytest.mark.asyncio
async def test_a_27_mb_log_is_copied_while_the_turn_keeps_answering(
    tmp_path, monkeypatch
):
    for part in ("mirror", "teardown"):
        (tmp_path / part).mkdir()
    rollout, trial, _sandbox, home = _acp_rollout(
        tmp_path / "mirror", monkeypatch, "claude-agent-acp"
    )
    rollout._agent_env = {"BENCHFLOW_PROVIDER_API_KEY": SECRET}
    object.__setattr__(rollout._config, "model", OPUS)
    source = _log()
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"
    copied = trial / _session_mirror.MIRROR_DIR / SESSION_DIR / "-app" / log.name
    answers: list[float] = []
    seen: list[tuple[int, bytes]] = []

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        log.parent.mkdir(parents=True)
        log.write_text(source)
        mirror = rollout._session_mirror
        deadline = time.monotonic() + 300
        # The turn streams its answer while the copy catches up.
        while mirror._offsets.get("./-app/session-1.jsonl", 0) < len(source):
            assert time.monotonic() < deadline, "the copy never caught up"
            answers.append(time.monotonic())
            if copied.exists():
                size = copied.stat().st_size
                with copied.open("rb") as handle:
                    handle.seek(max(0, size - 64))
                    seen.append((size, handle.read()))
            await asyncio.sleep(0.05)
        answers.append(time.monotonic())
        session.record_prompt_usage(
            {"inputTokens": 1500, "outputTokens": 120, "totalTokens": 1620}
        )
        return [], 0

    # Where each copy redacts: in a worker thread, never the loop's.
    from benchflow.rollout import _session_log

    loop_thread = threading.current_thread()
    ran: dict[str, list[bool]] = {"mirror": [], "teardown": []}

    def watched(name, func):
        def call(*args, **kwargs):
            ran[name].append(threading.current_thread() is loop_thread)
            return func(*args, **kwargs)

        return call

    monkeypatch.setattr(
        _session_mirror, "_render", watched("mirror", _session_mirror._render)
    )
    monkeypatch.setattr(
        _session_log, "_redact_files", watched("teardown", _session_log._redact_files)
    )
    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    await rollout.execute()

    gaps = [b - a for a, b in itertools.pairwise(answers)]
    print(f"mirror: {len(answers)} answers, max gap {max(gaps) * 1000:.0f} ms")
    assert ran["mirror"], "the mirror never redacted"
    assert not any(ran["mirror"]), "the mirror redacted on the event loop"
    assert len(answers) > 10
    assert max(gaps) < MAX_GAP_SEC
    mirrored = copied.read_bytes()
    # Appends only: every size the copy had is a prefix of the final copy.
    assert [size for size, _ in seen] == sorted(size for size, _ in seen)
    for size, tail in seen:
        assert mirrored[max(0, size - 64) : size] == tail
    _check_redacted(mirrored.decode(), source)
    assert rollout._session_mirror.current is True

    # The teardown copy of the same log, a heartbeat beside it.
    box_rollout, box_trial = _rollout(
        tmp_path / "teardown",
        monkeypatch,
        agent="claude-agent-acp",
        harness="acp",
        home=home,
    )
    target = box_rollout._rollout_paths.agent_dir / SESSION_DIR
    beats: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            beats.append(time.monotonic())
            await asyncio.sleep(0.05)

    beat = asyncio.create_task(heartbeat())
    copied_out = await _copy_out(
        box_rollout._env, f"{home}/.claude/projects", target, [SECRET]
    )
    stop.set()
    await beat
    beats.append(time.monotonic())
    gaps = [b - a for a, b in itertools.pairwise(beats)]
    print(f"teardown: {len(beats)} beats, max gap {max(gaps) * 1000:.0f} ms")
    assert copied_out is True
    assert ran["teardown"], "the teardown copy never redacted"
    assert not any(ran["teardown"]), "the teardown copy redacted on the event loop"
    assert max(gaps) < MAX_GAP_SEC
    full = (box_trial / "agent" / SESSION_DIR / "-app" / log.name).read_bytes()
    # The two copies redact alike, byte for byte.
    assert full == mirrored
