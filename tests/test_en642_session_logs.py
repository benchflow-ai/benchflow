"""The lead's ruling D2: the agent's own logs are read at the end of
its turn, before hardening and the verifier.

A subscription trial (Claude Code or Codex on a login, the native harness)
is counted and priced from the agent's session log and, for a native Claude
turn cut off, from the stream log in agent/. Teardown read them after the
verifier, whose test.sh runs as root with code the agent wrote, and the
sandbox shares agent/: test.sh could rewrite a log, or plant one, and the
trial recorded it. The logs are now copied at the end of the agent's turn
(``snapshot_session_logs``) and teardown reads only those copies. Each test
plays test.sh on the real verify() and teardown, with the sandbox's home a
local folder (tests.test_en374_salvage).
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from tests.test_en374_salvage import (
    KILLED_MID_TURN,
    KILLED_TOKENS,
    OPUS,
    OPUS_RATES,
    _claude_session_log,
    _codex_session,
    _rollout,
    _stream_log,
    _tokens,
    _usage,
)

_TOKEN_FIELDS = re.compile(r'("(?:\w+_)?(?:input|output)(?:_\w+)?_tokens"\s*:\s*)\d+')


def _cheapen(path) -> None:
    """test.sh's rewrite: every token count in the log becomes 1."""
    path.write_text(_TOKEN_FIELDS.sub(r"\g<1>1", path.read_text()))


def _root_test_sh(monkeypatch, run):
    from benchflow.rollout import Rollout

    async def verifier(self):
        run()
        self._rewards = {"reward": 1.0}
        return self._rewards

    monkeypatch.setattr(Rollout, "_score_finished_solver", verifier)


CLAUDE_CALLS = [
    (OPUS, _usage(1500, 0, 3000, 120)),
    (OPUS, _usage(20, 3000, 400, 300)),
]
CLAUDE_TOKENS = {
    "n_input_tokens": 1520,
    "n_output_tokens": 420,
    "n_cache_read_tokens": 3000,
    "n_cache_creation_tokens": 3400,
    "total_tokens": 1520 + 420 + 3000 + 3400,
}
CLAUDE_USD = (
    1520 * OPUS_RATES[0]
    + 420 * OPUS_RATES[1]
    + 3000 * OPUS_RATES[2]
    + 3400 * OPUS_RATES[3]
)


@pytest.mark.asyncio
async def test_a_session_log_test_sh_rewrites_changes_no_count_or_cost(
    tmp_path, monkeypatch
):
    """Fails before: the trial was counted and priced from the rewritten log
    (one token per count)."""
    home = tmp_path / "sandbox-home"
    _claude_session_log(home, CLAUDE_CALLS)
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"
    original = log.read_text()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True
    _root_test_sh(monkeypatch, lambda: _cheapen(log))

    await rollout.verify()
    assert log.read_text() != original  # test.sh did rewrite it
    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == CLAUDE_TOKENS
    assert usage["cost_usd"] == pytest.approx(CLAUDE_USD)
    assert usage["price_source"] == "agent_session_log"
    kept = trial / "agent" / "claude-sessions" / "-app" / "session-1.jsonl"
    assert kept.read_text() == original


@pytest.mark.asyncio
async def test_a_session_log_test_sh_plants_is_never_read(tmp_path, monkeypatch):
    """No log at the end of the turn: one test.sh writes later is not the
    trial's, so the count stays the agent's report (none) and no cost."""
    home = tmp_path / "sandbox-home"
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True
    _root_test_sh(monkeypatch, lambda: _claude_session_log(home, CLAUDE_CALLS))

    await rollout.verify()
    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage.get("total_tokens") in (None, 0)
    assert usage.get("cost_usd") is None
    assert not (trial / "agent" / "claude-sessions").exists()


@pytest.mark.asyncio
async def test_a_codex_session_log_test_sh_rewrites_changes_no_count(
    tmp_path, monkeypatch
):
    from tests.test_codex_acp_tokens import THREE_CALLS

    home = tmp_path / "sandbox-home"
    _codex_session(home, THREE_CALLS)
    logs = sorted((home / ".codex" / "sessions").rglob("*.jsonl"))
    assert logs
    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="codex-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True

    def rewrite():
        for path in logs:
            _cheapen(path)

    _root_test_sh(monkeypatch, rewrite)
    await rollout.verify()
    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_input_tokens"] == 12000 + 896 + 828
    assert usage["n_cache_read_tokens"] == 11904 + 12672
    assert usage["n_output_tokens"] == 300 + 150 + 84
    assert usage["usage_source"] == "session_log"


@pytest.mark.asyncio
async def test_a_native_stream_log_test_sh_rewrites_in_agent_changes_no_count(
    tmp_path, monkeypatch
):
    """The native harness's stream log is in agent/, which the sandbox
    shares: a cut-off native Claude turn is counted from the copy made at
    the end of the turn."""
    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    stream = _stream_log(trial, [KILLED_MID_TURN])
    rollout._usage_report_missing = True
    _root_test_sh(monkeypatch, lambda: _cheapen(stream))

    await rollout.verify()
    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == KILLED_TOKENS
    assert usage["usage_source"] == "stream_log"


@pytest.mark.asyncio
async def test_session_folders_in_the_sandbox_are_cleared_as_root(
    tmp_path, monkeypatch
):
    """/logs/agent's session folders are cleared in the sandbox before
    teardown places its copies: on Docker, root's files there are root's on
    the host until the sandbox stops."""
    from types import SimpleNamespace

    from benchflow.rollout._session_log import clear_session_log_folders_in_sandbox

    home = tmp_path / "sandbox-home"
    _claude_session_log(home, CLAUDE_CALLS)
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    _root_test_sh(monkeypatch, lambda: None)
    await rollout.verify()
    sent = []

    class Mounted:
        is_mounted = True

        async def exec(self, command, **kwargs):
            sent.append((command, kwargs))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    rollout._env = Mounted()
    await clear_session_log_folders_in_sandbox(rollout)
    assert sent == []  # nothing there
    (trial / "agent" / "codex-sessions").mkdir()
    await clear_session_log_folders_in_sandbox(rollout)
    assert sent == [
        (
            "rm -rf -- /logs/agent/claude-sessions /logs/agent/codex-sessions",
            {"user": "root", "system_shell": True, "timeout_sec": 30},
        )
    ]
    # A folder teardown placed is never removed (on a mount it is the
    # host's copy).
    sent.clear()
    rollout._session_log_snapshot.published.add("claude-sessions")
    await clear_session_log_folders_in_sandbox(rollout)
    assert [command for command, _ in sent] == ["rm -rf -- /logs/agent/codex-sessions"]


@pytest.mark.asyncio
async def test_a_subscription_trial_keeps_its_log_on_a_mounted_sandbox(
    tmp_path, monkeypatch
):
    """The price places the kept copy first; the clear before it must not
    take it away. On Docker the sandbox's /logs/agent is the trial's agent/,
    so the stand-in's root removes what it is told to under agent/."""
    import shlex
    from types import SimpleNamespace

    home = tmp_path / "sandbox-home"
    _claude_session_log(home, CLAUDE_CALLS)
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True
    _root_test_sh(monkeypatch, lambda: None)
    await rollout.verify()
    box = rollout._env

    class Mounted:
        is_mounted = True

        async def exec(self, command, **kwargs):
            if kwargs.get("system_shell"):
                argv = shlex.split(command)
                assert argv[:3] == ["rm", "-rf", "--"]
                for path in argv[3:]:
                    name = path.removeprefix("/logs/agent/")
                    shutil.rmtree(trial / "agent" / name, ignore_errors=True)
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            return await box.exec(command, **kwargs)

        def __getattr__(self, name):
            return getattr(box, name)

    rollout._env = Mounted()
    await rollout._cleanup_before_teardown()
    kept = trial / "agent" / "claude-sessions" / "-app" / "session-1.jsonl"
    assert kept.is_file()
    assert rollout._usage_metrics["price_source"] == "agent_session_log"


def _state(path) -> object:
    """What is at ``path`` (followed): a file's bytes and mtime, a folder's
    entries, each the same way."""
    if path.is_dir():
        return sorted((child.name, _state(child)) for child in path.iterdir())
    return (path.read_bytes(), path.stat().st_mtime_ns)


@pytest.mark.asyncio
@pytest.mark.parametrize("planted", ["file", "folder"])
async def test_a_stream_log_link_in_agent_is_refused_and_recorded(
    tmp_path, monkeypatch, planted
):
    """MG's C4c scan finding: the native stream log in agent/, which the
    sandbox shares, was copied with shutil.copyfile, outside the reachable
    helper. A link there, to a file or to a folder outside the run folder, is
    never followed: nothing outside is read into the trial or written, and the
    trial's warnings say why. Fails on 4ec771e1: no warning."""
    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    outside = tmp_path / "outside"
    lines = "".join(json.dumps(line) + "\n" for line in KILLED_MID_TURN)
    if planted == "file":
        outside.write_text(lines)
    else:
        outside.mkdir()
        (outside / "claude-code.jsonl").write_text(lines)
    before = _state(outside)
    (trial / "agent" / "claude-code.jsonl").symlink_to(
        outside, target_is_directory=planted == "folder"
    )
    rollout._usage_report_missing = True
    _root_test_sh(monkeypatch, lambda: None)

    await rollout.verify()
    await rollout._cleanup_before_teardown()
    rollout._build_result()

    assert rollout._usage_metrics.get("usage_source") != "stream_log"
    assert _state(outside) == before
    warnings = json.loads((trial / "result.json").read_text()).get("warnings") or []
    assert [w for w in warnings if w.startswith("agent/claude-code.jsonl, ")] == [
        "agent/claude-code.jsonl, the native harness's stream log, is not a regular "
        "file (a link, a folder or a special file): BenchFlow did not read it, so a "
        "native Claude Code turn cut off before its report is not counted from it"
    ], warnings


@pytest.mark.asyncio
async def test_a_session_log_folder_link_in_agent_is_replaced_never_written_through(
    tmp_path, monkeypatch
):
    """test.sh plants agent/claude-sessions as a link to a folder outside the
    run folder: teardown places the trial's copy in its stead and writes
    nothing through it."""
    home = tmp_path / "sandbox-home"
    _claude_session_log(home, CLAUDE_CALLS)
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("untouched")
    before = _state(outside)

    def plant():
        (trial / "agent" / "claude-sessions").symlink_to(
            outside, target_is_directory=True
        )

    _root_test_sh(monkeypatch, plant)
    await rollout.verify()
    await rollout._cleanup_before_teardown()

    assert _state(outside) == before
    kept = trial / "agent" / "claude-sessions"
    assert kept.is_dir() and not kept.is_symlink()
    assert (kept / "-app" / "session-1.jsonl").is_file()


# What a root process could plant outside the run folder for the snapshot
# to read: one unmistakable line.
_CANARY = b'{"type":"canary","note":"outside the run folder"}\n'


def _held_canary(rollout) -> list[str]:
    """The files in the snapshot's private folder that hold the canary."""
    folder = rollout._session_log_snapshot.view.parent
    return [
        str(path)
        for path in folder.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and _CANARY.strip() in path.read_bytes()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("window", ["before the copier", "at the copier's open"])
async def test_a_stream_log_swapped_for_a_link_after_the_check_is_never_read(
    tmp_path, monkeypatch, window
):
    """RUSAGE's race on 4ec771e11: agent/claude-code.jsonl is a regular file
    when the snapshot checks it, and a link to a file outside the run folder
    by the time it copies. The hook swaps it in that window, just before the
    copy begins, whichever copier the snapshot uses (4ec771e11's
    shutil.copyfile; the helper's snapshot_tree since c3cc8e53). The copy is
    refused, the canary is neither read into the trial nor copied, and the
    warning is recorded. The second window is inside the helper's copier,
    between its own check and its open, where the open refuses the link (no
    follow, then fstat). Fails on 4ec771e11: it copied the canary."""
    from benchflow._utils.reachable_write import ReachablePath
    from benchflow.rollout import _session_log
    from benchflow.rollout._session_log import (
        close_session_log_snapshot,
        snapshot_session_logs,
    )

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    stream = _stream_log(trial, [KILLED_MID_TURN])
    canary = tmp_path / "outside" / "canary.jsonl"
    canary.parent.mkdir()
    canary.write_bytes(_CANARY)
    before = _state(canary)
    swapped: list[str] = []

    def in_the_window(copier):
        def hooked(source, *args, **kwargs):
            if not swapped and getattr(source, "name", "") == "claude-code.jsonl":
                stream.unlink()
                stream.symlink_to(canary)
                swapped.append(str(source))
            return copier(source, *args, **kwargs)

        return hooked

    monkeypatch.setattr(shutil, "copyfile", in_the_window(shutil.copyfile))
    if window == "before the copier" and hasattr(_session_log, "snapshot_tree"):
        monkeypatch.setattr(
            _session_log, "snapshot_tree", in_the_window(_session_log.snapshot_tree)
        )
    if window == "at the copier's open":
        monkeypatch.setattr(ReachablePath, "open", in_the_window(ReachablePath.open))

    await snapshot_session_logs(rollout)
    try:
        assert swapped, "the hook never ran between the check and the copy"
        assert rollout._session_log_snapshot.stream is None
        assert _held_canary(rollout) == []
        assert _state(canary) == before
        assert [
            w
            for w in vars(rollout).get("_warnings", [])
            if w.startswith("agent/claude-code.jsonl, ")
        ]
    finally:
        close_session_log_snapshot(rollout)


@pytest.mark.asyncio
async def test_a_symlinked_agent_folder_fails_closed_and_reads_nothing(
    tmp_path, monkeypatch
):
    """agent/ itself is a link to a folder outside the run folder, which
    holds a claude-code.jsonl canary. The helper anchors on agent/ without
    following it, so the snapshot fails closed (ReachableWriteError): it reads
    nothing and records the warning. Fails on 4ec771e11: it followed agent/
    and copied the canary."""
    from benchflow._utils.reachable_write import ReachableWriteError
    from benchflow.rollout._session_log import (
        close_session_log_snapshot,
        snapshot_session_logs,
    )

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    outside = tmp_path / "outside-agent"
    outside.mkdir()
    (outside / "claude-code.jsonl").write_bytes(_CANARY)
    before = _state(outside)
    agent = trial / "agent"
    shutil.rmtree(agent)
    agent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ReachableWriteError):
        (rollout._rollout_paths.agent_dir / "claude-code.jsonl").lstat()
    await snapshot_session_logs(rollout)
    try:
        assert rollout._session_log_snapshot is not None
        assert rollout._session_log_snapshot.stream is None
        assert _held_canary(rollout) == []
        assert _state(outside) == before
        assert [
            w
            for w in vars(rollout).get("_warnings", [])
            if w.startswith("agent/claude-code.jsonl, ")
        ]
    finally:
        close_session_log_snapshot(rollout)


def _compact(home) -> list[str]:
    """One compaction line in Claude Code's session log."""
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"
    line = json.dumps(
        {
            "type": "system",
            "subtype": "compact_boundary",
            "compactMetadata": {"trigger": "auto", "preTokens": 150000},
        }
    )
    with log.open("a") as handle:
        handle.write(line + "\n")
    return [line]


@pytest.mark.asyncio
async def test_a_codex_log_copy_test_sh_plants_in_agent_is_never_counted(
    tmp_path, monkeypatch
):
    """RUSAGE's repro: test.sh plants a cheapened agent/codex-sessions (agent/
    is /logs/agent, which root can write). Fails before: the Codex count read
    the planted copy (every count 1)."""
    from tests.test_codex_acp_tokens import THREE_CALLS

    home = tmp_path / "sandbox-home"
    _codex_session(home, THREE_CALLS)
    sessions = home / ".codex" / "sessions"
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="codex-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True

    def plant():
        planted = trial / "agent" / "codex-sessions"
        shutil.copytree(sessions, planted)
        for path in planted.rglob("*.jsonl"):
            _cheapen(path)

    _root_test_sh(monkeypatch, plant)
    await rollout.verify()
    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_input_tokens"] == 12000 + 896 + 828
    assert usage["n_cache_read_tokens"] == 11904 + 12672
    assert usage["n_output_tokens"] == 300 + 150 + 84
    kept = sorted((trial / "agent" / "codex-sessions").rglob("*.jsonl"))
    originals = sorted(sessions.rglob("*.jsonl"))
    assert [p.read_text() for p in kept] == [p.read_text() for p in originals]


@pytest.mark.asyncio
async def test_a_claude_log_copy_test_sh_plants_feeds_no_count(tmp_path, monkeypatch):
    """A proxied Claude trial keeps no copy of the log, so a planted
    agent/claude-sessions used to stay and feed the compaction count and the
    model calls. They come from the copy held since the end of the turn, and
    the planted folder is emptied."""
    from benchflow.rollout._session_log import compactions_in

    home = tmp_path / "sandbox-home"
    _claude_session_log(home, CLAUDE_CALLS)
    clean = _compact(home)
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_metrics = {
        "usage_source": "provider_response",
        "total_tokens": 9,
        "cost_usd": 0.01,
    }

    def plant():
        planted = trial / "agent" / "claude-sessions" / "-app"
        planted.mkdir(parents=True)
        lines = [json.dumps({"type": "system", "subtype": "compact_boundary"})] * 3 + [
            json.dumps(
                {
                    "type": "assistant",
                    "requestId": f"req_x{n}",
                    "message": {
                        "id": f"msg_x{n}",
                        "model": OPUS,
                        "usage": _usage(1, 0, 0, 1),
                    },
                }
            )
            for n in range(5)
        ]
        (planted / "planted.jsonl").write_text("".join(line + "\n" for line in lines))

    _root_test_sh(monkeypatch, plant)
    await rollout.verify()
    await rollout._cleanup_before_teardown()

    assert rollout._compactions == compactions_in(clean)
    calls = trial / "trajectory" / "model-calls.jsonl"
    assert len(calls.read_text().splitlines()) == len(CLAUDE_CALLS)
    left = trial / "agent" / "claude-sessions"
    assert not list(left.rglob("*.jsonl")) if left.exists() else True
