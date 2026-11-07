"""Guard 4a8b3c67's raw absolute-agent regression and legacy-name compatibility."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

SPELLINGS = [
    "agent",
    "agent-name",
    "agent_name",
    "/opt/dummy/agent",
    "/other/dummy/agent",
    "bin/agent",
    "./bin/agent",
    "../agent",
    "a/../../agent",
    "a//agent",
    "a/./agent",
    r"C:\dummy\agent.exe",
    r"..\agent",
    "agent --flag /tmp/dummy",
    "agent; $(echo dummy) | cat",
    " name\twith\nspaces ",
    "é机器人/模型",
    "e\u0301机器人/模型",
    "😀",
    ".",
    "..",
    "/",
    "",
    "-flag",
    "CON",
    "nul.txt",
    "a\x00b",
    "\udcff",
    "\ud800",
    "a" * 4096,
    "a" * 4096 + "b",
    "/" + "folder/" * 200 + "agent",
]
SPELLING_IDS = [f"spelling-{i:02}" for i in range(len(SPELLINGS))]


def _legacy(spelling):
    return f"{spelling.replace('-', '_')}.txt"


def test_accepted_legacy_paths_are_byte_for_byte_unchanged(tmp_path):
    """Determine old admission with the old writer, not the mapper."""
    from benchflow._utils.artifact_names import agent_log_name
    from benchflow._utils.reachable_write import ReachablePath

    root = ReachablePath.rollout(tmp_path, "agent")
    accepted = []
    refused = []
    for spelling in SPELLINGS:
        old = _legacy(spelling)
        try:
            destination = root / old
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("original marker\n")
        except (ValueError, OSError, UnicodeError):
            refused.append(spelling)
        else:
            accepted.append(spelling)
            assert agent_log_name(spelling) == old
            assert (root / agent_log_name(spelling)).read_text() == "original marker\n"
    assert accepted and refused
    generated = [agent_log_name(s) for s in refused]
    assert len(set(generated)) == len(generated)
    for name in generated:
        assert re.fullmatch(r"-raw-abs-[0-9a-f]{64}\.txt", name)
        assert len(name.encode()) < 100
        assert "-" not in _legacy(name)  # no old spelling can produce this namespace
        (root / name).write_text("new marker\n")


class _Launched(Exception):
    pass


@pytest.mark.parametrize("spelling", SPELLINGS, ids=SPELLING_IDS)
@pytest.mark.parametrize(
    "backend", ["docker", "remote-docker", "daytona", "modal", "apple", "agentcore"]
)
async def test_acp_spelling_reaches_transport_without_renaming_supported_logs(
    tmp_path, monkeypatch, spelling, backend
):
    """Guard 4a8b3c67 on every dispatch path, preserving the exact command."""
    from benchflow._utils.artifact_names import agent_log_name
    from benchflow.acp import runtime

    logs = []

    def transport(**kwargs):
        log = kwargs["agent_log_path"]
        assert log.relative_to(tmp_path / "agent") == Path(agent_log_name(spelling))
        assert kwargs["command"].endswith("/tmp/dummy-agent --honest")
        logs.append(log)
        raise _Launched

    monkeypatch.setattr(runtime, "ContainerTransport", transport)
    sandbox = SimpleNamespace(
        live_process=AsyncMock(return_value=object()),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="")),
    )
    for _ in range(2):
        with pytest.raises(_Launched):
            await runtime.connect_acp(
                env=sandbox,
                agent=spelling,
                agent_launch="/tmp/dummy-agent --honest",
                agent_env={},
                sandbox_user=None,
                model=None,
                rollout_dir=tmp_path,
                environment=backend,
                agent_cwd="/tmp/dummy-workspace",
            )
    assert logs[0].relative_to(tmp_path) == logs[1].relative_to(tmp_path)
    assert sandbox.live_process.await_count == 2


@pytest.mark.parametrize("spelling", SPELLINGS, ids=SPELLING_IDS)
def test_native_and_acp_share_log_names_and_append_on_reconnect(tmp_path, spelling):
    """Guard 4a8b3c67's native sibling without renaming trusted CLI streams."""
    from benchflow._utils.artifact_names import agent_log_name
    from benchflow.native_harness.client import NativeCLIClient
    from benchflow.native_harness.harnesses import CLAUDE_CODE

    for _ in range(2):
        client = NativeCLIClient(
            env=object(),
            harness=CLAUDE_CODE,
            agent=spelling,
            launch_env={},
            sandbox_user=None,
            cwd="/tmp/dummy-workspace",
            rollout_dir=tmp_path,
        )
        assert client._log_path.relative_to(tmp_path / "agent") == Path(
            agent_log_name(spelling)
        )
        assert client._stream_path.relative_to(tmp_path / "agent") == Path(
            f"{CLAUDE_CODE.cli}.jsonl"
        )
        # ACP creates intermediate log folders before passing the capability;
        # NativeCLIClient does so lazily through ReachableLog.
        client._log("dummy stderr")
        client._log_file.close()
    assert client._log_path.read_text().count("dummy stderr") == 2


def test_literal_fallback_looking_agent_cannot_claim_an_absolute_agents_log():
    """The two namespaces cannot meet, even for an exact hash."""
    from benchflow._utils.artifact_names import agent_log_name

    encoded = agent_log_name("/tmp/dummy-agent")
    literal = encoded.removesuffix(".txt")
    assert agent_log_name(literal) == literal.replace("-", "_") + ".txt"
    assert agent_log_name(literal) != encoded


@pytest.mark.parametrize(
    "spelling, preserved",
    [
        ("agent-name", True),
        ("bin/agent", True),
        ("", True),
        (".", True),
        (r"C:\dummy\agent.exe", False),
        (r"..\agent", False),
        ("CON", False),
        ("nul.txt", False),
        ("a:b", False),
        ("a" * 251, True),
        ("a" * 252, False),
        ("😀" * 125, True),
        ("😀" * 126, False),
        ("\ud800", True),
        ("\udcff", True),
    ],
)
def test_windows_path_policy_keeps_admitted_names_and_bounds_fallbacks(
    monkeypatch, spelling, preserved
):
    """Guard the fix for Windows' existing path grammar and UTF-16 name limit."""
    from pathlib import PureWindowsPath

    from benchflow._utils import artifact_names

    # Exercise only the pure name policy; this does not claim live Win32 I/O.
    monkeypatch.setattr(artifact_names, "Path", PureWindowsPath)
    monkeypatch.setattr(artifact_names, "os", SimpleNamespace(name="nt"))
    mapped = artifact_names.agent_log_name(spelling)
    if preserved:
        assert mapped == _legacy(spelling)
    else:
        assert re.fullmatch(r"-raw-abs-[0-9a-f]{64}\.txt", mapped)


@pytest.mark.parametrize("spelling", ["/tmp/dummy", "../dummy", "a/../../dummy"])
def test_reachable_path_still_refuses_escape(tmp_path, spelling):
    """Guard 4a8b3c67: mapping names never relaxes the no-follow writer."""
    from benchflow._utils.reachable_write import ReachablePath

    with pytest.raises(ValueError, match="relative and stay"):
        ReachablePath.rollout(tmp_path, "agent") / spelling
