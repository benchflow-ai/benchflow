"""Per-agent instruction-file selection + sandbox write."""

from __future__ import annotations

import pytest

from benchflow.agents.registry import resolve_agent
from benchflow.arena.instructions import instruction_target, write_agent_instructions


def test_instruction_filename_by_agent_family():
    assert resolve_agent("claude-agent-acp").instruction_filename == "CLAUDE.md"
    assert resolve_agent("gemini").instruction_filename == "GEMINI.md"
    assert resolve_agent("codex-acp").instruction_filename == "AGENTS.md"  # default


def test_instruction_target_path():
    cfg = resolve_agent("claude-agent-acp")
    assert instruction_target("/work/claude-0", cfg) == "/work/claude-0/CLAUDE.md"


class _FakeSandbox:
    def __init__(self):
        self.execs: list[str] = []
        self.uploads: list[tuple[str, str]] = []

    async def exec(self, cmd, *, user="root", timeout_sec=30):
        self.execs.append(cmd)

    async def upload_file(self, src, dst):
        self.uploads.append((str(src), dst))


@pytest.mark.asyncio
async def test_write_uploads_to_right_filename(tmp_path):
    body = tmp_path / "codex.md"
    body.write_text("play to win")
    cfg = resolve_agent("codex-acp")
    sb = _FakeSandbox()

    target = await write_agent_instructions(sb, "/work/cx", cfg, body)

    assert target == "/work/cx/AGENTS.md"
    assert sb.uploads == [(str(body), "/work/cx/AGENTS.md")]
    assert any("mkdir -p" in c and "/work/cx" in c for c in sb.execs)


@pytest.mark.asyncio
async def test_no_instructions_is_noop():
    sb = _FakeSandbox()
    out = await write_agent_instructions(sb, "/work/x", resolve_agent("codex-acp"), None)
    assert out is None
    assert sb.uploads == []
