"""Preservation and loss characterization for the ``ACP → ATIF`` conversion.

``tests/trajectories/test_export_atif.py`` pins the *shape* ATIF export
produces. This suite pins something different and complementary: which
information survives the conversion, and which does not. Every claim in the
loss table of ``docs/trace-interop.md`` §5 that concerns ATIF has an executable
assertion here, so a future change to the converter either keeps the property
or fails a test that says in one line what changed.

Four classes are used throughout, and the distinction between the last two is
the point of the suite:

* **preserved** — the value reaches the ATIF document unchanged.
* **normalized** — it reaches the document, but relocated or reshaped, so a
  consumer must know BenchFlow's convention to read it.
* **dropped** — the ACP capture events carry it and the converter discards it.
  Fixable in ``export_atif.py`` alone.
* **unsupported** — the capture events never carried it in the first place.
  ``export_atif.py`` cannot fix these; the loss is upstream, at the ACP wire
  boundary (:meth:`ACPSession.handle_update`) or in
  ``_events_to_trajectory``.

Losses are asserted with sentinel values rather than with structural checks
wherever possible: a sentinel absent from ``json.dumps(document)`` is a claim
about the whole document, not about the one field the test happened to look
at, and it stays true if the converter later moves data around.

Scope: the ``ACP-session capture events`` → ATIF path only. ADP and
Verifiers/ORS share ``_export_common`` and most of these properties, but they
are deliberately out of scope here; nothing in this file needs to change to
cover them later.

No runtime module is imported for its side effects and nothing here writes
outside ``tmp_path``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from benchflow.acp.session import ACPSession
from benchflow.acp.types import StopReason
from benchflow.trajectories import export_atif
from benchflow.trajectories._capture import _capture_session_trajectory
from benchflow.trajectories.export_atif import (
    acp_events_to_atif_steps,
    trajectory_to_atif_record,
    write_rollout_atif_json,
)
from tests.integration.scenarios import atif_issues

# Sentinels. Distinctive enough that finding one anywhere in a serialized
# document is unambiguous evidence the value survived, and finding none is
# evidence it did not.
RAW_INPUT = "SENTINEL-rawInput-8f21"
RAW_OUTPUT = "SENTINEL-rawOutput-4c07"
LOCATION = "/SENTINEL-locations-1b93/main.py"
META = "SENTINEL-meta-77de"
PENDING_TOOL_CALL_ID = "SENTINEL-pending-tc-9f13"
TIMEOUT_SEC = 1337.75
NON_TEXT_BLOCK = "SENTINEL-diff-newText-a55c"
USAGE_TOKENS = 424242

# `datetime.now()` renders as `2026-08-14T06:43:29.280301`; the capture path
# builds such values on every ToolCallRecord, so their absence downstream is
# the observable form of "timestamps are not exported".
ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


# ---------------------------------------------------------------------------
# Corpora — built through the production capture path, not hand-written
# ---------------------------------------------------------------------------


def _rich_session() -> ACPSession:
    """One session carrying every emitted event type plus the dropped extras.

    The ``tool_call`` updates deliberately include the four ACP protocol
    fields BenchFlow does not read (``rawInput``, ``rawOutput``, ``locations``,
    ``_meta``) so the boundary tests below can assert where they stop.
    """
    session = ACPSession("sess-a2")
    session.record_user_prompt("List the files.")
    session.handle_update(
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "I should run ls."},
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc1",
            "title": "ls -la",
            "kind": "execute",
            "status": "pending",
            "rawInput": {"command": RAW_INPUT},
            "locations": [{"path": LOCATION}],
            "_meta": {"trace": META},
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc1",
            "status": "completed",
            "rawOutput": {"stdout": RAW_OUTPUT},
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
    # Session-level state that lives beside the event log and is never exported.
    session.stop_reason = next(iter(StopReason))
    session.usage_snapshots.append({"input_tokens": USAGE_TOKENS})
    session.record_agent_timeout(
        timeout_sec=TIMEOUT_SEC,
        pending_tool_call_ids=[PENDING_TOOL_CALL_ID],
        terminal_trajectory_complete=False,
    )
    return session


def _rich_events() -> list[dict[str, Any]]:
    return _capture_session_trajectory(_rich_session())


PROMPTS = ["Solve the task.", "Then stop."]


def _rich_document() -> dict[str, Any]:
    return trajectory_to_atif_record(
        session_id="sess-a2",
        agent_name="claude-code",
        events=_rich_events(),
        prompts=PROMPTS,
    )


def _walk_keys(obj: Any):
    """Yield every mapping key in a nested document."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_keys(item)


def _tool_calls(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [call for step in document["steps"] for call in step.get("tool_calls", [])]


# ---------------------------------------------------------------------------
# Source introspection — keeps the loss claims tied to the code, not to a copy
# ---------------------------------------------------------------------------


def _converter_handled_event_types() -> set[str]:
    """Event types ``acp_events_to_atif_steps`` branches on, read via AST.

    Mirrors the technique in ``test_acp_capture_event_schema.py``: asserting
    against the source rather than against a hand-maintained list means adding
    a branch to the converter fails a test here instead of silently
    invalidating this suite's premise.
    """
    tree = ast.parse(Path(export_atif.__file__).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "acp_events_to_atif_steps"
    )
    handled: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "etype"):
            continue
        # ast.Compare keeps ops and comparators in lockstep, so strict zipping
        # is a free assertion that the node is well-formed.
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if (
                isinstance(op, ast.Eq)
                and isinstance(comparator, ast.Constant)
                and isinstance(comparator.value, str)
            ):
                handled.add(comparator.value)
    return handled


def _validator_valid_sources() -> set[str]:
    """The ``source`` allowlist inside ``tests.integration.scenarios.atif_issues``.

    A local variable rather than a module constant, so it is read from source.
    Extracting it keeps the emitter and its only in-repo consumer from drifting
    apart without a test noticing.
    """
    import tests.integration.scenarios as scenarios

    tree = ast.parse(Path(scenarios.__file__).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "atif_issues"
    )
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Set):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "valid_sources"
            for target in node.targets
        ):
            return {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
            }
    raise AssertionError(
        "atif_issues no longer assigns a literal `valid_sources` set; re-derive "
        "the consumer's accepted sources from whatever replaced it."
    )


# ---------------------------------------------------------------------------
# Preservation invariants — these must survive any future refactor
# ---------------------------------------------------------------------------


def test_message_text_survives_verbatim():
    document = _rich_document()
    messages = [step.get("message", "") for step in document["steps"]]
    assert "List the files." in messages
    assert "One file: README.md." in messages


def test_every_prompt_becomes_a_leading_user_step():
    document = _rich_document()
    leading = document["steps"][: len(PROMPTS)]
    assert [step["message"] for step in leading] == PROMPTS
    assert {step["source"] for step in leading} == {"user"}


def test_tool_call_identity_is_preserved():
    (call,) = _tool_calls(_rich_document())
    assert call["tool_call_id"] == "tc1"


def test_textual_tool_output_reaches_the_observation():
    document = _rich_document()
    contents = [
        result["content"]
        for step in document["steps"]
        for result in step.get("observation", {}).get("results", [])
    ]
    assert contents == ["README.md"]


def test_no_thought_text_is_lost():
    document = _rich_document()
    reasoning = " ".join(
        step.get("reasoning_content", "") for step in document["steps"]
    )
    assert "I should run ls." in reasoning


def test_tool_kind_becomes_function_name_with_tool_fallback():
    (call,) = _tool_calls(_rich_document())
    assert call["function_name"] == "execute"

    # `handle_update` writes the literal "tool" as the kind when an update
    # arrives for an id that was never opened; the converter's own fallback
    # covers the case where even that is missing.
    (step,) = acp_events_to_atif_steps(
        [{"type": "tool_call", "tool_call_id": "x", "kind": "", "content": []}]
    )
    assert step["tool_calls"][0]["function_name"] == "tool"


def test_tool_status_and_title_survive_somewhere():
    """Location-agnostic on purpose — that they survive is the invariant."""
    (call,) = _tool_calls(_rich_document())
    serialized = json.dumps(call)
    assert "completed" in serialized
    assert "ls -la" in serialized


def test_step_ids_are_dense_and_match_total_steps():
    document = _rich_document()
    assert [step["step_id"] for step in document["steps"]] == list(
        range(1, len(document["steps"]) + 1)
    )
    assert document["final_metrics"]["total_steps"] == len(document["steps"])


def test_observation_results_resolve_within_their_own_step():
    """The one ATIF spec constraint the converter must not break.

    ``source_call_id`` resolves against the *same* step's ``tool_calls``, so a
    trajectory with several calls is the case that would catch a converter
    that started resolving trajectory-wide.
    """
    events = [
        {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "kind": "execute",
            "title": f"cmd {index}",
            "status": "completed",
            "content": [{"text": f"out {index}"}],
        }
        for index, tool_call_id in enumerate(["tc1", "", "tc3", ""])
    ]
    steps = acp_events_to_atif_steps(events)
    assert len(steps) == len(events)
    seen: set[str] = set()
    for step in steps:
        ids = {call["tool_call_id"] for call in step["tool_calls"]}
        for result in step["observation"]["results"]:
            assert result["source_call_id"] in ids
        seen |= ids
    # Synthesized ids must not collide with each other or with real ones.
    assert len(seen) == len(events)


def test_emitter_output_satisfies_the_in_repo_atif_validator(tmp_path):
    """The only consumer of ATIF inside this repository must accept the output."""
    write_rollout_atif_json(
        tmp_path,
        session_id="sess-a2",
        agent_name="claude-code",
        prompts=PROMPTS,
        trajectory=_rich_events(),
    )
    assert atif_issues(tmp_path) == []


# ---------------------------------------------------------------------------
# Loss characterization — pins today's behaviour so a change is deliberate
# ---------------------------------------------------------------------------


def test_tool_arguments_are_always_empty():
    """Loss #1. Empty even when the source event carries extra structure.

    If the capture layer is ever enriched (open question 3 in
    ``docs/trace-interop.md``), this test fails and points at the converter as
    the place that still has to be taught to read the new field.
    """
    for call in _tool_calls(_rich_document()):
        assert call["arguments"] == {}

    (step,) = acp_events_to_atif_steps(
        [
            {
                "type": "tool_call",
                "tool_call_id": "tc1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": [],
                "rawInput": {"command": RAW_INPUT},
            }
        ]
    )
    assert step["tool_calls"][0]["arguments"] == {}
    assert RAW_INPUT not in json.dumps(step)


def test_tool_status_and_title_live_in_the_non_standard_extra():
    """Loss #2, the normalized half of ``test_tool_status_and_title_survive``.

    ``extra`` was added in ATIF-v1.7; a consumer reading only standard fields
    sees neither the status nor the command.
    """
    (call,) = _tool_calls(_rich_document())
    assert call["extra"] == {"title": "ls -la", "status": "completed"}
    assert "status" not in {key for key in call if key != "extra"}


def test_no_timestamp_reaches_the_document():
    """Loss #3. ``ToolCallRecord`` stamps ``started_at``/``finished_at``.

    Those attributes exist on the record the capture path built above, and
    nothing serializes them, so no ISO-8601 value can appear downstream.
    """
    serialized = json.dumps(_rich_document())
    assert not ISO_DATETIME.search(serialized)


def test_agent_timeout_leaves_no_trace_in_the_document():
    """Loss #4. The marker is captured, then dropped by the converter.

    Asserted against the whole serialized document, not against the step list,
    so a future partial rendering of the timeout would still be visible here.
    """
    events = _rich_events()
    assert any(event["type"] == "agent_timeout" for event in events), (
        "the corpus no longer contains an agent_timeout event; this test would "
        "pass vacuously"
    )
    serialized = json.dumps(_rich_document())
    assert "wall_clock_timeout" not in serialized
    assert PENDING_TOOL_CALL_ID not in serialized
    assert str(TIMEOUT_SEC) not in serialized


def test_converter_handles_exactly_the_event_types_it_documents():
    """The structural counterpart of the loss above.

    Reads the converter's own branches. ``agent_timeout`` is emitted by
    ``_events_to_trajectory`` and deliberately absent here; ``oracle`` is
    handled although the ACP-session emitter never produces it (§2.4).
    """
    assert _converter_handled_event_types() == {
        "user_message",
        "agent_thought",
        "agent_message",
        "tool_call",
        "oracle",
    }


def test_non_text_content_blocks_are_dropped():
    """Loss #5. ``content_blocks_to_text`` renders text blocks and nothing else."""
    (step,) = acp_events_to_atif_steps(
        [
            {
                "type": "tool_call",
                "tool_call_id": "tc1",
                "kind": "edit",
                "title": "patch main.py",
                "status": "completed",
                "content": [
                    {
                        "type": "diff",
                        "path": "main.py",
                        "oldText": "a",
                        "newText": NON_TEXT_BLOCK,
                    }
                ],
            }
        ]
    )
    assert "observation" not in step
    assert NON_TEXT_BLOCK not in json.dumps(step)


def test_no_per_step_metrics_and_no_session_level_usage_or_stop_reason():
    """Losses #6, #8 and #9, asserted over the whole document.

    Per-step ``metrics`` is a valid ATIF field on agent steps and is never
    emitted; ``ACPSession.usage_snapshots`` and ``stop_reason`` were populated
    on the session that produced this corpus and reach nothing.
    """
    document = _rich_document()
    assert all("metrics" not in step for step in document["steps"])
    keys = set(_walk_keys(document))
    assert "metrics" not in keys
    assert "stop_reason" not in keys
    serialized = json.dumps(document)
    assert str(USAGE_TOKENS) not in serialized
    assert next(iter(StopReason)).value not in serialized


def test_agent_version_is_declared_unknown_rather_than_fabricated():
    """Loss #7, documented in the converter as deliberate."""
    assert _rich_document()["agent"]["version"] == "unknown"


def test_oracle_events_never_produce_an_oracle_source():
    """The emitter renders oracle activity as an ``agent`` step.

    The in-repo validator accepts ``source: "oracle"``; nothing produces it.
    Whether that third value should be produced or removed is open question 3
    in ``docs/trace-interop.md`` — this test states the divergence rather than
    resolving it, and fails if either side moves.
    """
    steps = acp_events_to_atif_steps(
        [
            {"type": "user_message", "text": "go"},
            {"type": "agent_message", "text": "ok"},
            {"type": "oracle", "command": "bash run.sh"},
        ]
    )
    produced = {step["source"] for step in steps}
    accepted = _validator_valid_sources()

    assert produced <= accepted
    assert "oracle" in accepted
    assert "oracle" not in produced
    assert steps[-1] == {
        "step_id": 3,
        "source": "agent",
        "message": "[oracle: bash run.sh]",
    }


def test_empty_text_events_produce_no_step():
    """Step count is not a function of event count."""
    assert (
        acp_events_to_atif_steps(
            [
                {"type": "user_message", "text": ""},
                {"type": "agent_message", "text": ""},
                {"type": "agent_thought", "text": ""},
            ]
        )
        == []
    )


def test_unknown_and_malformed_events_do_not_perturb_step_numbering():
    """Dropping is silent and leaves no gap in ``step_id``."""
    steps = acp_events_to_atif_steps(
        [
            {"type": "user_message", "text": "first"},
            {"type": "a_future_event_type", "text": "ignored"},
            "not-a-dict",
            {"type": "agent_timeout", "reason": "wall_clock_timeout"},
            {"type": "agent_message", "text": "second"},
        ]
    )
    assert [step["step_id"] for step in steps] == [1, 2]
    assert [step["message"] for step in steps] == ["first", "second"]


def test_consecutive_thoughts_are_indistinguishable_from_one_joined_thought():
    """The ``\\n\\n`` join is not reversible.

    ``ThoughtBuffer`` joins buffered thoughts with a blank line, so a single
    thought that already contains one produces the same ``reasoning_content``
    as two separate events. Reachable in production: the Gemini scrape path
    (``_capture._parse_gemini_trajectory``) appends one ``agent_thought`` event
    per entry of a message's ``thoughts`` list, so consecutive thought events
    are a shape the capture layer really emits.
    """
    two_events = acp_events_to_atif_steps(
        [
            {"type": "agent_thought", "text": "first"},
            {"type": "agent_thought", "text": "second"},
        ]
    )
    one_event = acp_events_to_atif_steps(
        [{"type": "agent_thought", "text": "first\n\nsecond"}]
    )
    assert two_events == one_event


# ---------------------------------------------------------------------------
# Producer boundary — where the tool-argument loss actually happens
# ---------------------------------------------------------------------------


def test_raw_input_family_never_reaches_the_capture_events():
    """``handle_update`` reads five fields and drops the rest.

    This is the load-bearing test of the suite: it places the tool-argument
    loss at the ACP wire boundary rather than in the converter, which is what
    makes ``arguments: {}`` unfixable in ``export_atif.py`` alone.
    """
    events = _rich_events()
    (tool_call,) = [event for event in events if event["type"] == "tool_call"]
    assert set(tool_call) == {
        "type",
        "tool_call_id",
        "kind",
        "title",
        "status",
        "content",
    }
    serialized = json.dumps(events)
    for sentinel in (RAW_INPUT, RAW_OUTPUT, LOCATION, META):
        assert sentinel not in serialized


def test_raw_input_family_never_reaches_the_atif_document():
    serialized = json.dumps(_rich_document())
    for sentinel in (RAW_INPUT, RAW_OUTPUT, LOCATION, META):
        assert sentinel not in serialized
