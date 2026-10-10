"""Streaming-write tests for ACP trajectory: TrajectoryWriter, non-destructive
snapshot, multi-scene cumulative semantics, unknown-update guard, stale .tmp
cleanup, and the late-scene partial-capture regression from PR #566 review."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.acp.session import ACPSession
from benchflow.acp.types import ToolCallStatus
from benchflow.rollout import Rollout
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _snapshot_session_trajectory,
    make_trajectory_sink,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestSnapshotSessionTrajectory:
    """Non-destructive snapshot preserves chunk streaming until prompt_end."""

    def test_snapshot_does_not_flush_pending(self) -> None:
        session = ACPSession("s1")
        session.record_user_prompt("Solve")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Hel"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "lo"},
            }
        )
        snapshot = _snapshot_session_trajectory(session)
        assert len(session._pending_text) == 2
        assert [e["type"] for e in snapshot] == ["user_message", "agent_message"]
        assert snapshot[1]["text"] == "Hello"
        session.mark_prompt_end()
        final = _capture_session_trajectory(session)
        assert [e["type"] for e in final] == ["user_message", "agent_message"]
        assert final[1]["text"] == "Hello"

    def test_snapshot_after_prompt_end_matches_capture(self) -> None:
        session = ACPSession("s1")
        session.record_user_prompt("Hi")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Done"},
            }
        )
        session.mark_prompt_end()
        assert _snapshot_session_trajectory(session) == _capture_session_trajectory(
            session
        )


class TestTrajectoryWriter:
    """Streams incremental snapshots to disk as the session evolves."""

    def test_writer_flushes_after_each_update(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "trajectory" / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        session = ACPSession("s1")
        session.on_change = writer
        assert not traj_path.exists()

        session.record_user_prompt("Solve the task")
        assert traj_path.exists()
        snap = _read_jsonl(traj_path)
        assert [e["type"] for e in snap] == ["user_message"]
        assert snap[0]["text"] == "Solve the task"

        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls /app",
                "kind": "bash",
            }
        )
        snap = _read_jsonl(traj_path)
        assert [e["type"] for e in snap] == ["user_message", "tool_call"]
        assert snap[1]["status"] == ToolCallStatus.PENDING.value

        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        snap = _read_jsonl(traj_path)
        assert len(snap) == 2
        assert snap[1]["status"] == ToolCallStatus.COMPLETED.value

        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "All "},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "done."},
            }
        )
        snap = _read_jsonl(traj_path)
        assert [e["type"] for e in snap] == [
            "user_message",
            "tool_call",
            "agent_message",
        ]
        assert snap[2]["text"] == "All done."

        session.mark_prompt_end()
        final = _read_jsonl(traj_path)
        assert final == _capture_session_trajectory(session)

    def test_writer_swallows_callback_errors(self, tmp_path: Path) -> None:
        session = ACPSession("s1")

        def boom(_session: ACPSession) -> None:
            raise RuntimeError("nope")

        session.on_change = boom
        session.record_user_prompt("Solve")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )

    def test_writer_atomic_rewrite_creates_no_torn_lines(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        session = ACPSession("s1")
        session.on_change = writer
        session.record_user_prompt("Solve")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        assert not traj_path.with_suffix(traj_path.suffix + ".tmp").exists()
        for line in traj_path.read_text().splitlines():
            json.loads(line)

    def test_write_final_overwrites_streamed_file(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        session = ACPSession("s1")
        session.on_change = writer
        session.record_user_prompt("Solve")
        assert traj_path.exists()
        writer.write_final(
            [{"type": "oracle", "command": "solve.sh", "return_code": 0}]
        )
        snapshot = _read_jsonl(traj_path)
        assert snapshot == [{"type": "oracle", "command": "solve.sh", "return_code": 0}]


class TestMultiSceneCumulativeStreaming:
    """The streaming writer must include events from prior scenes — not just
    the current session — so multi-scene rollouts don't lose history on
    disk between scene transitions.
    """

    def test_sink_includes_prior_trajectory(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        prior = [
            {"type": "user_message", "text": "scene 1 prompt"},
            {
                "type": "tool_call",
                "tool_call_id": "tc_s1",
                "kind": "bash",
                "title": "ls",
                "status": "completed",
                "content": [],
            },
            {"type": "agent_message", "text": "scene 1 done"},
        ]
        session = ACPSession("s2")
        session.on_change = make_trajectory_sink(writer, prior)

        session.record_user_prompt("scene 2 prompt")
        snapshot = _read_jsonl(traj_path)
        assert len(snapshot) == 4
        assert snapshot[0]["text"] == "scene 1 prompt"
        assert snapshot[1]["tool_call_id"] == "tc_s1"
        assert snapshot[2]["text"] == "scene 1 done"
        assert snapshot[3]["text"] == "scene 2 prompt"

    def test_empty_new_session_does_not_wipe_prior(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        prior = [{"type": "user_message", "text": "scene 1 only event"}]
        writer.write_final(prior)
        session = ACPSession("s2")
        session.on_change = make_trajectory_sink(writer, prior)
        session.on_change(session)
        snapshot = _read_jsonl(traj_path)
        assert snapshot == prior, "empty session must not wipe prior events"

    def test_sink_isolates_prior_snapshot_from_caller_mutation(
        self, tmp_path: Path
    ) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        prior: list[dict] = [{"type": "user_message", "text": "prior"}]
        session = ACPSession("s2")
        session.on_change = make_trajectory_sink(writer, prior)
        session.record_user_prompt("current")
        prior.append({"type": "user_message", "text": "current"})
        session.on_change(session)
        snapshot = _read_jsonl(traj_path)
        assert [e.get("text") for e in snapshot] == ["prior", "current"]


class _Clock:
    """Manually advanced clock for the live sink's write interval."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _chunk(kind: str, text: str) -> dict:
    return {"sessionUpdate": kind, "content": {"type": "text", "text": text}}


class _CountingWriter(TrajectoryWriter):
    """Counts the live sink's snapshot writes."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.writes = 0

    def write_events(self, events: list[dict]) -> None:
        self.writes += 1
        super().write_events(events)


class TestLiveSinkCoalescing:
    """The live sink writes event boundaries at once and coalesces the
    streamed growth of an event, so the cost of a long streamed thought does
    not grow with its chunk count."""

    def test_streamed_chunks_do_not_each_rewrite_the_file(self, tmp_path: Path) -> None:
        """Guards the live sink from PR #566 against rewriting the whole
        trajectory on every streamed chunk, which made a long thought
        quadratic and left the ACP read loop behind the agent."""
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = _CountingWriter(traj_path)
        session = ACPSession("s1")
        session.on_change = make_trajectory_sink(writer, [], clock=_Clock())

        session.record_user_prompt("Solve")
        for _ in range(1000):
            session.handle_update(_chunk("agent_thought_chunk", "x"))
        assert writer.writes == 2, "the prompt and the thought's first chunk only"
        assert _read_jsonl(traj_path)[1] == {"type": "agent_thought", "text": "x"}

        session.mark_prompt_end()
        assert writer.writes == 3
        assert _read_jsonl(traj_path) == [
            {"type": "user_message", "text": "Solve"},
            {"type": "agent_thought", "text": "x" * 1000},
        ]

    def test_boundaries_are_written_at_once_and_growth_per_interval(
        self, tmp_path: Path
    ) -> None:
        """Guards the live sink from PR #566: coalescing streamed growth must
        not delay a new event, a tool status change, or the end of a prompt."""
        traj_path = tmp_path / "acp_trajectory.jsonl"
        clock = _Clock()
        session = ACPSession("s1")
        session.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), [], min_interval_sec=2.0, clock=clock
        )

        session.record_user_prompt("Solve")
        session.handle_update(_chunk("agent_thought_chunk", "a"))
        session.handle_update(_chunk("agent_thought_chunk", "b"))
        assert _read_jsonl(traj_path)[-1] == {"type": "agent_thought", "text": "a"}
        clock.now = 2.0
        session.handle_update(_chunk("agent_thought_chunk", "c"))
        assert _read_jsonl(traj_path)[-1] == {"type": "agent_thought", "text": "abc"}
        session.handle_update(_chunk("agent_thought_chunk", "d"))

        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "make",
                "kind": "execute",
            }
        )
        snap = _read_jsonl(traj_path)
        assert snap[1] == {"type": "agent_thought", "text": "abcd"}
        assert snap[2]["status"] == "pending"

        def output(status: str, text: str) -> dict:
            return {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": status,
                "content": [
                    {"type": "content", "content": {"type": "text", "text": text}}
                ],
            }

        session.handle_update(output("in_progress", "line 1"))
        assert len(_read_jsonl(traj_path)[2]["content"]) == 1
        session.handle_update(output("in_progress", "line 2"))
        assert len(_read_jsonl(traj_path)[2]["content"]) == 1
        session.handle_update(output("completed", "line 3"))
        snap = _read_jsonl(traj_path)
        assert snap[2]["status"] == "completed"
        assert len(snap[2]["content"]) == 3

        session.handle_update(_chunk("agent_message_chunk", "Do"))
        session.handle_update(_chunk("agent_message_chunk", "ne"))
        assert _read_jsonl(traj_path)[-1] == {"type": "agent_message", "text": "Do"}
        session.mark_prompt_end()
        assert _read_jsonl(traj_path) == _capture_session_trajectory(session)

    def test_scene_sinks_keep_prior_scenes_once_while_coalescing(
        self, tmp_path: Path
    ) -> None:
        """Guards PR #566's cumulative multi-scene sink: with coalescing, the
        file holds the prior scenes' events exactly once, both mid-stream and
        after the prompt ends."""
        traj_path = tmp_path / "acp_trajectory.jsonl"
        clock = _Clock()
        scene1 = ACPSession("s1")
        scene1.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), [], clock=clock
        )
        scene1.record_user_prompt("scene 1")
        scene1.handle_update(_chunk("agent_message_chunk", "one"))
        scene1.mark_prompt_end()
        prior = _capture_session_trajectory(scene1)

        scene2 = ACPSession("s2")
        scene2.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), prior, clock=clock
        )
        scene2.record_user_prompt("scene 2")
        scene2.handle_update(_chunk("agent_message_chunk", "t"))
        scene2.handle_update(_chunk("agent_message_chunk", "wo"))
        assert _read_jsonl(traj_path) == [
            *prior,
            {"type": "user_message", "text": "scene 2"},
            {"type": "agent_message", "text": "t"},
        ]
        scene2.mark_prompt_end()
        assert _read_jsonl(traj_path) == prior + _capture_session_trajectory(scene2)

    def test_tool_kind_change_is_written_at_once(self, tmp_path: Path) -> None:
        """Guards the live sink from PR #566: an update that reclassifies a
        running tool call as a skill (issue #507) is a boundary, not growth."""
        traj_path = tmp_path / "acp_trajectory.jsonl"
        session = ACPSession("s1")
        session.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), [], clock=_Clock()
        )
        session.record_user_prompt("Solve")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Load PDF skill",
                "kind": "other",
                "status": "in_progress",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "in_progress",
                "content": [
                    {
                        "type": "content",
                        "content": {
                            "type": "text",
                            "text": "Tool: invoke_skill\nResult:\n[skill: pdf]",
                        },
                    }
                ],
            }
        )
        assert _read_jsonl(traj_path)[1]["kind"] == "skill"


class _NoConcatText(str):
    """Text chunk that fails when merged by pairwise concatenation."""

    def __add__(self, other: str) -> str:
        raise AssertionError("text chunks merged with + instead of one join")


class TestPendingTextMerge:
    def test_text_runs_are_merged_with_one_join(self) -> None:
        """Guards the pending-text merge from commit b588c9e8 and PR #566:
        growing a run with ``+=`` copied the whole run again on every chunk,
        quadratic in the chunk count of a long streamed thought."""
        session = ACPSession("s1")
        session.record_user_prompt("Solve")
        for text in ("Hel", "lo", " world"):
            session.handle_update(_chunk("agent_thought_chunk", _NoConcatText(text)))
        session.handle_update(_chunk("agent_message_chunk", _NoConcatText("o")))
        session.handle_update(_chunk("agent_message_chunk", _NoConcatText("k")))
        expected = [
            {"type": "user_message", "text": "Solve"},
            {"type": "agent_thought", "text": "Hello world"},
            {"type": "agent_message", "text": "ok"},
        ]

        assert _snapshot_session_trajectory(session) == expected
        assert len(session._pending_text) == 5
        session.mark_prompt_end()
        assert _capture_session_trajectory(session) == expected

    def test_lone_chunk_is_kept_as_sent(self) -> None:
        """Guards the pending-text merge from PR #566: a run of one chunk is
        copied as is, so a null text stays null instead of raising."""
        session = ACPSession("s1")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": None},
            }
        )
        assert _snapshot_session_trajectory(session) == [
            {"type": "agent_message", "text": None}
        ]


class TestHandleUpdateUnknownType:
    """Unknown sessionUpdate types should not trigger on_change — no state
    mutated, no reason to re-snapshot.
    """

    def test_unknown_update_type_skips_notify(self) -> None:
        session = ACPSession("s1")
        calls: list[int] = []
        session.on_change = lambda _s: calls.append(1)
        session.handle_update({"sessionUpdate": "unknown_future_type"})
        assert calls == [], "on_change must NOT fire for unrecognized update types"

    def test_known_update_type_still_notifies(self) -> None:
        session = ACPSession("s1")
        calls: list[int] = []
        session.on_change = lambda _s: calls.append(1)
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        assert calls == [1], "on_change must fire for recognized update types"


class TestTrajectoryWriterStaleTmpCleanup:
    """Stale .tmp file left by a previous crashed run must be swept on
    writer construction so a follow-up reader can't pick it up.
    """

    def test_init_unlinks_pre_existing_tmp(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        stale_tmp = traj_path.with_suffix(traj_path.suffix + ".tmp")
        stale_tmp.parent.mkdir(parents=True, exist_ok=True)
        stale_tmp.write_text('{"type":"user_message","text":"orphaned"}')
        assert stale_tmp.exists()
        TrajectoryWriter(traj_path)
        assert not stale_tmp.exists(), "stale .tmp must be cleaned up on init"

    def test_init_tolerates_no_pre_existing_tmp(self, tmp_path: Path) -> None:
        TrajectoryWriter(tmp_path / "acp_trajectory.jsonl")


class _StubACPClient:
    """Bare-minimum ACPClient stub for partial-capture tests — only exposes
    the .session attribute that ``_capture_partial_acp_trajectory`` reads."""

    def __init__(self, session: ACPSession | None) -> None:
        self.session = session


def _build_rollout(
    *,
    prior_trajectory: list[dict],
    active_session: ACPSession | None,
    session_traj_count: int,
) -> Rollout:
    """Build a Rollout instance bypassing __init__ so we can exercise the
    pure trajectory-capture logic without dragging the whole config /
    sandbox tree through the test."""
    r: Any = Rollout.__new__(Rollout)
    r._trajectory = list(prior_trajectory)
    r._acp_client = _StubACPClient(active_session)
    r._session_traj_count = session_traj_count
    r._partial_trajectory = False
    r._trajectory_source = "acp" if prior_trajectory else None
    r._n_tool_calls = 0
    return r


def _build_commit_rollout() -> Rollout:
    class _Tree:
        def __init__(self) -> None:
            self.steps = []

        def advance(self, _cursor, step):
            self.steps.append(step)
            return object()

        def populate(self, _node, step):
            self.steps.append(step)
            return object()

    r: Any = Rollout.__new__(Rollout)
    r._trajectory = []
    r._session_traj_count = 0
    r._session_tool_count = 0
    r._n_tool_calls = 0
    r._executed_prompts = []
    r._trajectory_source = None
    r._partial_trajectory = False
    r._timing = {}
    r._session = None
    r._native_usage_checkpoint = None
    r._tree = _Tree()
    r._cursor = object()
    return r


class TestMultiScenePartialCaptureFix:
    """Regression for PR #566 review finding #1.

    Scene 1 succeeds and extends ``self._trajectory``. Scene 2 connects to
    a fresh session that streams partial events, then ``execute_prompts``
    raises before returning — so the scene-2 events are NOT yet extended
    into ``self._trajectory``. The bug: ``_capture_partial_acp_trajectory``
    used to early-return whenever ``self._trajectory`` was non-empty,
    leaving scene 2's partial events stranded in the session and lost when
    the final write overwrote the file with stale cumulative state.

    Fix: capture the current session's delta beyond
    ``self._session_traj_count`` and extend ``self._trajectory`` with it.
    Mark ``partial_trajectory=True`` and ``trajectory_source='partial_acp'``
    when a delta was actually appended.
    """

    def test_partial_capture_appends_late_scene_delta(self) -> None:
        prior = [
            {"type": "user_message", "text": "scene 1 prompt"},
            {
                "type": "tool_call",
                "tool_call_id": "tc_s1",
                "kind": "bash",
                "title": "ls",
                "status": "completed",
                "content": [],
            },
            {"type": "agent_message", "text": "scene 1 done"},
        ]
        scene2 = ACPSession("s2")
        scene2.record_user_prompt("scene 2 prompt")
        scene2.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_s2",
                "title": "cat data",
                "kind": "bash",
            }
        )
        # `execute_prompts` would have updated _session_traj_count; here we
        # simulate the crash-before-return case where it's still 0.
        r = _build_rollout(
            prior_trajectory=prior, active_session=scene2, session_traj_count=0
        )
        r._capture_partial_acp_trajectory()
        types = [e.get("type") for e in r._trajectory]
        assert types == [
            "user_message",
            "tool_call",
            "agent_message",
            "user_message",
            "tool_call",
        ], "scene 1 events must be preserved AND scene 2 partial appended"
        assert r._partial_trajectory is True
        assert r._trajectory_source == "partial_acp"

    def test_partial_capture_noop_when_no_delta(self) -> None:
        """When the current session has nothing new beyond ``_session_traj_count``,
        leave ``_trajectory`` and metadata alone.
        """
        prior = [{"type": "user_message", "text": "scene 1 done"}]
        scene2 = ACPSession("s2")
        # Pretend execute_prompts already extended these into _trajectory.
        scene2.record_user_prompt("nothing new")
        r = _build_rollout(
            prior_trajectory=prior,
            active_session=scene2,
            session_traj_count=len(_capture_session_trajectory(scene2)),
        )
        r._capture_partial_acp_trajectory()
        assert r._trajectory == prior
        assert r._partial_trajectory is False
        assert r._trajectory_source == "acp"

    def test_clean_timeout_commit_does_not_mark_partial_acp(self) -> None:
        """Guards PR #638 follow-up: clean wall-clock timeout snapshots are complete."""

        r = _build_commit_rollout()

        r._commit_acp_execution(
            trajectory=[
                {"type": "user_message", "text": "solve"},
                {
                    "type": "tool_call",
                    "tool_call_id": "tc",
                    "kind": "bash",
                    "title": "run",
                    "status": "completed",
                    "content": [],
                },
                {
                    "type": "agent_timeout",
                    "reason": "wall_clock_timeout",
                    "timeout_sec": 900.0,
                    "pending_tool_call_ids": [],
                    "terminal_trajectory_complete": True,
                },
            ],
            n_tool_calls=1,
            prev_session_tools=0,
            effective_prompts=["solve"],
            started_at=datetime.now(),
            node=None,
        )

        assert r._trajectory_source == "acp"
        assert r._partial_trajectory is False
        assert r._n_tool_calls == 1
        assert r._executed_prompts == ["solve"]

    def test_pending_timeout_commit_stays_partial_acp(self) -> None:
        """Guards PR #640: pending tool-call timeouts stay quarantinable partials."""

        r = _build_commit_rollout()

        r._commit_acp_execution(
            trajectory=[
                {"type": "user_message", "text": "solve"},
                {
                    "type": "tool_call",
                    "tool_call_id": "tc",
                    "kind": "bash",
                    "title": "run",
                    "status": "pending",
                    "content": [],
                },
                {
                    "type": "agent_timeout",
                    "reason": "wall_clock_timeout",
                    "timeout_sec": 900.0,
                    "pending_tool_call_ids": ["tc"],
                    "terminal_trajectory_complete": False,
                },
            ],
            n_tool_calls=1,
            prev_session_tools=0,
            effective_prompts=["solve"],
            started_at=datetime.now(),
            node=None,
            partial_trajectory=True,
        )

        assert r._trajectory_source == "partial_acp"
        assert r._partial_trajectory is True
        assert r._n_tool_calls == 1

    def test_partial_capture_first_scene_still_works(self) -> None:
        """Single-scene crash path: prior empty, session has events.
        Must still capture (regression check against breaking the original
        early-empty behaviour).
        """
        scene1 = ACPSession("s1")
        scene1.record_user_prompt("solve")
        r = _build_rollout(
            prior_trajectory=[], active_session=scene1, session_traj_count=0
        )
        r._capture_partial_acp_trajectory()
        assert len(r._trajectory) == 1
        assert r._trajectory[0]["type"] == "user_message"
        assert r._partial_trajectory is True
        assert r._trajectory_source == "partial_acp"

    def test_partial_capture_is_idempotent_across_cleanup_disconnect(self) -> None:
        """Guards PR #566 against duplicating partial events when cleanup and
        disconnect both attempt late-session capture.
        """
        scene1 = ACPSession("s1")
        scene1.record_user_prompt("solve")
        scene1.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_s1",
                "title": "ls",
                "kind": "bash",
            }
        )
        r = _build_rollout(
            prior_trajectory=[], active_session=scene1, session_traj_count=0
        )
        r._capture_partial_acp_trajectory()
        r._capture_partial_acp_trajectory()
        assert [e["type"] for e in r._trajectory] == ["user_message", "tool_call"]
        assert r._n_tool_calls == 1

    def test_partial_capture_with_no_live_session_is_noop(self) -> None:
        prior = [{"type": "user_message", "text": "x"}]
        r = _build_rollout(
            prior_trajectory=prior, active_session=None, session_traj_count=0
        )
        r._capture_partial_acp_trajectory()
        assert r._trajectory == prior
        assert r._partial_trajectory is False

    def test_partial_capture_publishes_coalesced_stream(self, tmp_path: Path) -> None:
        """Guards PR #566's partial capture once the live sink coalesces
        streamed chunks: a cancelled run skips write_final, so teardown must
        publish what the sink held back."""
        traj_path = tmp_path / "acp_trajectory.jsonl"
        session = ACPSession("s1")
        session.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), [], clock=_Clock()
        )
        session.record_user_prompt("solve")
        for text in ("thinking", " about", " it"):
            session.handle_update(_chunk("agent_thought_chunk", text))
        assert _read_jsonl(traj_path)[-1]["text"] == "thinking"

        r = _build_rollout(
            prior_trajectory=[], active_session=session, session_traj_count=0
        )
        r._capture_partial_acp_trajectory()
        assert _read_jsonl(traj_path) == [
            {"type": "user_message", "text": "solve"},
            {"type": "agent_thought", "text": "thinking about it"},
        ]
        assert _read_jsonl(traj_path) == r._trajectory

    def test_partial_capture_without_a_prompt_writes_no_file(
        self, tmp_path: Path
    ) -> None:
        """Guards PR #566's partial capture: teardown of a session that never
        got a prompt must not create an empty live trajectory."""
        traj_path = tmp_path / "acp_trajectory.jsonl"
        session = ACPSession("s1")
        session.on_change = make_trajectory_sink(TrajectoryWriter(traj_path), [])
        r = _build_rollout(
            prior_trajectory=[], active_session=session, session_traj_count=0
        )
        r._capture_partial_acp_trajectory()
        assert not traj_path.exists()
