"""Conformance suite for the ACP-session capture-event JSON Schema.

Pins ``src/benchflow/trajectories/schemas/acp-capture-event-v1.schema.json``
against what the ACP-session capture emitter actually produces — the records
built by ``_events_to_trajectory`` and by the ``ACPSession`` legacy fallback.
The schema is a description of that emitter, not of everything
``TrajectoryWriter`` is willing to serialize: the writer validates nothing, so
a schema derived from its tolerance would describe nothing at all.

Its scope is narrower than the file. A complete ``acp_trajectory.jsonl`` may
also carry records from sources this suite does not cover — session-factory
``session.steps`` and the downstream ``oracle`` record — see
``docs/trace-interop.md`` §2.1 and §2.4.

The suite is falsifiable along two independent axes:

* **Record shape** — the C1 corpus below. Change the fields a record carries
  and validation fails, with no fixture to quietly adjust.
* **Event-type vocabulary** — ``test_schema_covers_exactly_the_emitted_event_types``,
  which reads the guards of ``_events_to_trajectory`` via AST. Add a branch to
  that function and the comparison against the schema fails.

Two corpora, both normative:

* **C1 — emitter-generated.** Real :class:`ACPSession` objects driven through
  ``handle_update`` / ``record_user_prompt`` / ``record_agent_timeout``,
  captured with the production functions and written to disk with the
  production writer. Every line of the resulting file must validate.
* **C2 — existing in-repo fixtures.** The ACP event lists already used as
  exporter inputs by the ATIF and ADP export tests, imported rather than copied
  so they cannot drift.

Deliberately NOT validated: the five ``acp_trajectory.jsonl`` files under
``.agents/skills/benchflow-experiment-review/evals/``. They are synthetic
eval-harness fixtures that reuse the filename without following the production
writer, and they are excluded from the conformance corpus rather than pinned —
see the "Known divergences" section of ``docs/trace-interop.md``.
"""

import ast
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchflow.acp.session import ACPSession, ToolCallRecord
from benchflow.acp.types import ToolCallStatus
from benchflow.trajectories import _capture
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
)
from tests.trajectories.test_export_adp import _sample_events as _adp_sample_events
from tests.trajectories.test_export_atif import _sample_events as _atif_sample_events

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "benchflow"
    / "trajectories"
    / "schemas"
    / "acp-capture-event-v1.schema.json"
)


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    return Draft202012Validator(_schema())


def _assert_valid(validator: Draft202012Validator, events: list[dict]) -> None:
    """Fail with the offending event and the concrete reason, not just a bool."""
    for index, event in enumerate(events):
        errors = sorted(validator.iter_errors(event), key=lambda e: e.path)
        assert not errors, (
            f"event[{index}] type={event.get('type')!r} failed schema:\n"
            + "\n".join(f"  - {e.json_path}: {e.message}" for e in errors)
            + f"\n  event = {json.dumps(event, default=str)}"
        )


def _round_trip_through_writer(tmp_path: Path, events: list[dict]) -> list[dict]:
    """Persist with the production writer, read back what landed on disk.

    Validating the parsed file rather than the in-memory list keeps redaction
    and serialization inside the tested path — a redaction that corrupted a
    record would surface here.

    Every line is parsed, blanks included: ``splitlines`` already absorbs the
    documented trailing newline, so a blank line left anywhere in the payload is
    a real defect and must raise here rather than be skipped.
    """
    path = tmp_path / "acp_trajectory.jsonl"
    TrajectoryWriter(path).write_final(events)
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# The schema document itself
# ---------------------------------------------------------------------------


def test_schema_is_a_valid_draft_2020_12_document():
    Draft202012Validator.check_schema(_schema())


def _schema_event_types() -> set[str]:
    """Event types the schema declares, gathered across every ``$defs`` branch."""
    declared: set[str] = set()
    for definition in _schema()["$defs"].values():
        type_prop = definition.get("properties", {}).get("type")
        if not type_prop:
            continue
        if "const" in type_prop:
            declared.add(type_prop["const"])
        declared.update(type_prop.get("enum", []))
    return declared


def _is_event_type_lookup(node: ast.expr) -> bool:
    """True for the expression ``event["type"]`` exactly."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "event"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "type"
    )


def _emitted_event_types() -> set[str]:
    """Event types accepted by ``_events_to_trajectory``'s guards, read from source.

    That function is a filter: an event is serialized if and only if its
    ``type`` matches one of its ``event["type"]`` comparisons, so the guard
    vocabulary *is* the emitted vocabulary. Only string literals compared
    against that exact expression are collected — ``== "x"`` via a single
    comparator, ``in ("x", "y")`` via a tuple/list/set of them.
    """
    tree = ast.parse(Path(_capture.__file__).read_text())
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_events_to_trajectory"
        ),
        None,
    )
    assert function is not None, (
        "_events_to_trajectory not found in benchflow.trajectories._capture. The "
        "conformance suite derives the emitted event-type vocabulary from its "
        "guards; if the function was renamed or moved, update this extraction "
        "rather than removing the check."
    )

    emitted: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or not _is_event_type_lookup(node.left):
            continue
        for comparator in node.comparators:
            candidates: list[ast.expr] = (
                list(comparator.elts)
                if isinstance(comparator, ast.Tuple | ast.List | ast.Set)
                else [comparator]
            )
            emitted.update(
                candidate.value
                for candidate in candidates
                if isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
            )
    return emitted


def test_schema_covers_exactly_the_emitted_event_types():
    """Schema vocabulary must equal the emitter's, with neither side hardcoded.

    The left side is parsed from the schema document; the right side is read
    out of ``benchflow.trajectories._capture._events_to_trajectory`` via AST.
    Adding a branch to that function therefore fails this test until the schema
    documents the new event type.

    **This guarantee depends on the current shape of those guards.** The
    function expresses its output vocabulary as ``event["type"]`` comparisons
    against string literals, and the extraction reads exactly that. If it is
    refactored so the vocabulary is no longer literal comparisons — a dispatch
    table, a module-level constant, a helper predicate — the extraction returns
    a different set and this test fails on purpose. There is deliberately no
    permissive fallback: a failure here means the extraction needs review, not
    that the check should be relaxed.
    """
    emitted = _emitted_event_types()
    assert emitted, (
        "no event types extracted from _events_to_trajectory. Its guards are no "
        'longer literal `event["type"]` comparisons, so this check can no longer '
        "see the emitted vocabulary — review the extraction above and re-derive "
        "it from whatever now expresses that vocabulary."
    )
    assert _schema_event_types() == emitted


# ---------------------------------------------------------------------------
# C1 — emitter-generated corpus
# ---------------------------------------------------------------------------


def _full_session() -> ACPSession:
    """One session exercising every emitted event type in one trajectory."""
    session = ACPSession("sess-conformance")
    session.record_user_prompt("List the files.")
    # Two chunks of the same type merge into a single agent_thought event.
    session.handle_update(
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "I should "},
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "run ls."},
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc1",
            "title": "ls -la",
            "kind": "execute",
            "status": "pending",
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc1",
            "status": "completed",
            "content": [
                {"type": "content", "content": {"type": "text", "text": "README.md"}}
            ],
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "One file: README.md."},
        }
    )
    session.record_agent_timeout(
        timeout_sec=1.5,
        pending_tool_call_ids=["tc9"],
        terminal_trajectory_complete=False,
    )
    return session


def test_c1_full_session_validates(validator, tmp_path):
    events = _round_trip_through_writer(
        tmp_path, _capture_session_trajectory(_full_session())
    )
    _assert_valid(validator, events)
    assert {event["type"] for event in events} == {
        "user_message",
        "agent_thought",
        "tool_call",
        "agent_message",
        "agent_timeout",
    }


def test_c1_openclaw_shim_updates_validate(validator, tmp_path):
    """``text_update`` / ``agent_thought`` are whole-text shim variants."""
    session = ACPSession("sess-shim")
    session.record_user_prompt("go")
    session.handle_update({"sessionUpdate": "text_update", "text": "shim message"})
    session.handle_update({"sessionUpdate": "agent_thought", "text": "shim thought"})
    events = _round_trip_through_writer(tmp_path, _capture_session_trajectory(session))
    _assert_valid(validator, events)


def test_c1_tool_call_update_for_unopened_id_validates(validator, tmp_path):
    """The fallback record carries ``kind: "tool"`` and empty title/content.

    ``"tool"`` is not a ``ToolKind`` member — this case is why the schema keeps
    ``kind`` an open string.
    """
    session = ACPSession("sess-orphan")
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "orphan",
            "status": "failed",
        }
    )
    events = _round_trip_through_writer(tmp_path, _capture_session_trajectory(session))
    _assert_valid(validator, events)
    assert events[0]["kind"] == "tool"
    assert events[0]["title"] == ""
    assert events[0]["content"] == []


def test_c1_legacy_capture_path_validates(validator, tmp_path):
    """Sessions with no event log fall back to flat tool_calls + message."""
    session = ACPSession("sess-legacy")
    record = ToolCallRecord("legacy1", "grep foo", "search")
    record.update_status(
        ToolCallStatus.COMPLETED,
        [{"type": "content", "content": {"type": "text", "text": "hit"}}],
    )
    session.tool_calls.append(record)
    session.message_chunks.append("done")
    events = _round_trip_through_writer(tmp_path, _capture_session_trajectory(session))
    _assert_valid(validator, events)
    assert [event["type"] for event in events] == ["tool_call", "agent_message"]


@pytest.mark.parametrize("status", [s.value for s in ToolCallStatus])
def test_c1_every_tool_call_status_validates(validator, tmp_path, status):
    session = ACPSession(f"sess-{status}")
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc1",
            "title": "t",
            "kind": "read",
            "status": "pending",
        }
    )
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "tc1", "status": status}
    )
    events = _round_trip_through_writer(tmp_path, _capture_session_trajectory(session))
    _assert_valid(validator, events)
    assert events[0]["status"] == status


def test_c1_empty_text_events_validate(validator, tmp_path):
    """Empty text is emitted, not filtered — the schema must accept it.

    ``record_user_prompt`` records unconditionally, and the chunk handlers
    append a chunk whose text is ``""`` (the ``text_update`` / ``agent_thought``
    shim handlers do skip empty text, which is why this differs by path).
    """
    session = ACPSession("sess-empty-text")
    session.record_user_prompt("")
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": ""},
        }
    )
    events = _round_trip_through_writer(tmp_path, _capture_session_trajectory(session))
    _assert_valid(validator, events)
    assert [event["text"] for event in events] == ["", ""]


def test_c1_empty_session_writes_no_lines(tmp_path):
    """An empty trajectory is an empty file, not a blank line."""
    events = _capture_session_trajectory(ACPSession("sess-empty"))
    assert events == []
    path = tmp_path / "acp_trajectory.jsonl"
    TrajectoryWriter(path).write_final(events)
    assert path.read_text() == ""


# ---------------------------------------------------------------------------
# C2 — existing in-repo fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample",
    [
        pytest.param(_atif_sample_events, id="test_export_atif._sample_events"),
        pytest.param(_adp_sample_events, id="test_export_adp._sample_events"),
    ],
)
def test_c2_exporter_input_fixtures_validate(validator, sample):
    _assert_valid(validator, sample())


# ---------------------------------------------------------------------------
# Falsifiability — the schema must reject, or it documents nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "why"),
    [
        pytest.param(
            {"type": "not_a_real_type", "text": "x"},
            "unknown event type",
            id="unknown-type",
        ),
        pytest.param({"text": "x"}, "no discriminator", id="missing-type"),
        pytest.param(
            {"type": "user_message"}, "text is required", id="text-event-missing-text"
        ),
        pytest.param(
            {"type": "agent_message", "text": "x", "timestamp": 1},
            "additionalProperties is false — timestamps are not emitted today",
            id="text-event-extra-field",
        ),
        pytest.param(
            {
                "type": "tool_call",
                "tool_call_id": "t",
                "kind": "read",
                "title": "t",
                "status": "completed",
            },
            "content is required",
            id="tool-call-missing-content",
        ),
        pytest.param(
            {
                "type": "tool_call",
                "tool_call_id": "t",
                "kind": "read",
                "title": "t",
                "status": "finished",
                "content": [],
            },
            "status is a closed enum",
            id="tool-call-bad-status",
        ),
        pytest.param(
            {
                "type": "tool_call",
                "tool_call_id": "t",
                "kind": "read",
                "title": "t",
                "status": "completed",
                "content": [],
                "rawInput": {"command": "ls"},
            },
            "rawInput is dropped by ACP-session capture and must not appear",
            id="tool-call-rawinput",
        ),
        pytest.param(
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": 1.0,
                "pending_tool_call_ids": [],
            },
            "terminal_trajectory_complete is required",
            id="timeout-missing-field",
        ),
        pytest.param(
            {
                "type": "agent_timeout",
                "reason": "some_other_reason",
                "timeout_sec": 1.0,
                "pending_tool_call_ids": [],
                "terminal_trajectory_complete": True,
            },
            "reason is a single-member enum today",
            id="timeout-unknown-reason",
        ),
        pytest.param(
            {
                "type": "tool_call",
                "tool_call_id": "t",
                "kind": "read",
                "title": "t",
                "status": "completed",
                "content": {"type": "content"},
            },
            "content must be an array",
            id="tool-call-content-not-array",
        ),
    ],
)
def test_schema_rejects(validator, event, why):
    assert not validator.is_valid(event), f"schema should reject: {why}"
