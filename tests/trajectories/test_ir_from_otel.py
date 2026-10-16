"""Conversion suite for ``OTLP/JSON → canonical Trace IR`` — the inbound OTel edge.

The three edges before this one could be checked against a producer in this
repository: `ACP → IR` against events a real `ACPSession` emitted, `IR → ATIF`
against the document the direct exporter writes, `ATIF → IR` against both. There
is no OpenTelemetry producer here at all (`docs/trace-interop.md` §4.2), so this
suite gets its ground truth from the two libraries the repository's ``uv.lock``
already pins:

- ``opentelemetry-proto==1.41.1`` — the wire shape;
- ``opentelemetry-semantic-conventions==0.62b1`` — the ``gen_ai.*`` vocabulary.

:data:`PRODUCER_PAYLOAD_JSON` below is **not hand-written**. It is the output of
``google.protobuf.json_format.MessageToJson`` over an ``ExportTraceServiceRequest``
built with ``opentelemetry-proto`` at the pinned version, so every encoding
choice in it — base64 identifiers, ``intValue`` as a JSON *string*, enum members
as names, absent fields where the value is the protobuf default — is the
library's, not this suite's. The generator is reproduced in
``_PRODUCER_PAYLOAD_RECIPE`` so anyone can rebuild it. Neither package is
imported here or added as a dependency.

What the rest of the suite is for:

- the **contract guards** — that no IR field quietly stops being filled and
  quietly stops being declared, that every declared path resolves in the
  canonical encoding, and that removing a declaration makes the guard fail;
- the **anti-invention** tests — no role, no synthesized id, no derived run
  extent, no time sorting, no parsing of a serialized argument string;
- the **adversarial** cases — payloads no conformant producer writes, which is
  where the ATIF edge found four real defects that the conformant path never
  touched.

Nothing here writes to disk and nothing imports a run path.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from benchflow.trajectories import _otlp_anyvalue as anyvalue_module
from benchflow.trajectories import ir_from_otel as ir_from_otel_module
from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlockKind,
    EventKind,
    LossClass,
    LossRecord,
    LossReport,
    PathSpace,
    ToolCall,
    TraceUsage,
    validate_trace,
)
from benchflow.trajectories.ir_from_otel import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    LOSS_DIRECTION,
    OPERATION_EXECUTE_TOOL,
    OTEL_SOURCE,
    OTLP_JSON_SOURCE,
    OTLP_PROTO_VERSION,
    SEMCONV_VERSION,
    USAGE_SOURCE,
    OtlpSpan,
    group_spans_by_trace_id,
    otel_spans_to_ir,
    otlp_json_spans,
    otlp_json_to_ir,
)
from tests.trajectories.test_trace_ir import resolve_ir_path

_PRODUCER_PAYLOAD_RECIPE = """
# pip install opentelemetry-proto==1.41.1   (the version uv.lock pins)
import json
from google.protobuf.json_format import MessageToJson
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as ts
from opentelemetry.proto.common.v1 import common_pb2 as c
from opentelemetry.proto.resource.v1 import resource_pb2 as r
from opentelemetry.proto.trace.v1 import trace_pb2 as t
# ... build one ResourceSpans holding five spans: an invoke_agent root, a chat
# child carrying usage and gen_ai.input.messages, two execute_tool children (one
# OK with arguments and a result, one ERROR with neither), and a plain HTTP
# client span with no gen_ai attribute at all ...
print(MessageToJson(ts.ExportTraceServiceRequest(resource_spans=[...])))
"""

PRODUCER_PAYLOAD_JSON = r"""
{
 "resourceSpans": [
  {
   "resource": {
    "attributes": [
     {
      "key": "service.name",
      "value": {
       "stringValue": "benchflow-rollout"
      }
     },
     {
      "key": "service.version",
      "value": {
       "stringValue": "0.6.10"
      }
     }
    ]
   },
   "scopeSpans": [
    {
     "scope": {
      "name": "benchflow.agent",
      "version": "0.1.0"
     },
     "spans": [
      {
       "traceId": "S/kvNXezTaajzpKdDg5HNg==",
       "spanId": "APBnqgupArc=",
       "name": "invoke_agent solver",
       "kind": "SPAN_KIND_INTERNAL",
       "startTimeUnixNano": "1755500000000000000",
       "endTimeUnixNano": "1755500004200000000",
       "attributes": [
        {
         "key": "gen_ai.operation.name",
         "value": {
          "stringValue": "invoke_agent"
         }
        },
        {
         "key": "gen_ai.agent.name",
         "value": {
          "stringValue": "solver"
         }
        },
        {
         "key": "gen_ai.agent.version",
         "value": {
          "stringValue": "2.1.0"
         }
        },
        {
         "key": "gen_ai.provider.name",
         "value": {
          "stringValue": "anthropic"
         }
        },
        {
         "key": "gen_ai.conversation.id",
         "value": {
          "stringValue": "conv-7731"
         }
        }
       ],
       "status": {
        "code": "STATUS_CODE_OK"
       }
      },
      {
       "traceId": "S/kvNXezTaajzpKdDg5HNg==",
       "spanId": "APBnqgupArg=",
       "parentSpanId": "APBnqgupArc=",
       "name": "chat claude-sonnet-5",
       "kind": "SPAN_KIND_CLIENT",
       "startTimeUnixNano": "1755500000010000000",
       "endTimeUnixNano": "1755500001900000375",
       "attributes": [
        {
         "key": "gen_ai.operation.name",
         "value": {
          "stringValue": "chat"
         }
        },
        {
         "key": "gen_ai.request.model",
         "value": {
          "stringValue": "claude-sonnet-5"
         }
        },
        {
         "key": "gen_ai.response.model",
         "value": {
          "stringValue": "claude-sonnet-5-20260101"
         }
        },
        {
         "key": "gen_ai.response.finish_reasons",
         "value": {
          "arrayValue": {
           "values": [
            {
             "stringValue": "tool_use"
            }
           ]
          }
         }
        },
        {
         "key": "gen_ai.usage.input_tokens",
         "value": {
          "intValue": "1204"
         }
        },
        {
         "key": "gen_ai.usage.output_tokens",
         "value": {
          "intValue": "87"
         }
        },
        {
         "key": "gen_ai.usage.cache_read.input_tokens",
         "value": {
          "intValue": "1024"
         }
        },
        {
         "key": "gen_ai.input.messages",
         "value": {
          "stringValue": "[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"list the files\"}]}]"
         }
        }
       ],
       "events": [
        {
         "timeUnixNano": "1755500001200000000",
         "name": "first_token"
        }
       ],
       "status": {
        "code": "STATUS_CODE_OK"
       }
      },
      {
       "traceId": "S/kvNXezTaajzpKdDg5HNg==",
       "spanId": "APBnqgupArk=",
       "parentSpanId": "APBnqgupArc=",
       "name": "execute_tool read_file",
       "kind": "SPAN_KIND_INTERNAL",
       "startTimeUnixNano": "1755500001950000000",
       "endTimeUnixNano": "1755500002100000000",
       "attributes": [
        {
         "key": "gen_ai.operation.name",
         "value": {
          "stringValue": "execute_tool"
         }
        },
        {
         "key": "gen_ai.tool.name",
         "value": {
          "stringValue": "read_file"
         }
        },
        {
         "key": "gen_ai.tool.type",
         "value": {
          "stringValue": "function"
         }
        },
        {
         "key": "gen_ai.tool.call.id",
         "value": {
          "stringValue": "toolu_01A"
         }
        },
        {
         "key": "gen_ai.tool.call.arguments",
         "value": {
          "kvlistValue": {
           "values": [
            {
             "key": "path",
             "value": {
              "stringValue": "/repo/README.md"
             }
            },
            {
             "key": "limit",
             "value": {
              "intValue": "200"
             }
            }
           ]
          }
         }
        },
        {
         "key": "gen_ai.tool.call.result",
         "value": {
          "stringValue": "# benchflow\n"
         }
        }
       ],
       "droppedAttributesCount": 2,
       "links": [
        {
         "traceId": "S/kvNXezTaajzpKdDg5HNg==",
         "spanId": "APBnqgupArg="
        }
       ],
       "status": {
        "code": "STATUS_CODE_OK"
       }
      },
      {
       "traceId": "S/kvNXezTaajzpKdDg5HNg==",
       "spanId": "APBnqgupAro=",
       "parentSpanId": "APBnqgupArc=",
       "name": "execute_tool write_file",
       "kind": "SPAN_KIND_INTERNAL",
       "startTimeUnixNano": "1755500002200000000",
       "endTimeUnixNano": "1755500002260000000",
       "attributes": [
        {
         "key": "gen_ai.operation.name",
         "value": {
          "stringValue": "execute_tool"
         }
        },
        {
         "key": "gen_ai.tool.name",
         "value": {
          "stringValue": "write_file"
         }
        },
        {
         "key": "gen_ai.tool.call.id",
         "value": {
          "stringValue": "toolu_01B"
         }
        },
        {
         "key": "error.type",
         "value": {
          "stringValue": "PermissionError"
         }
        }
       ],
       "status": {
        "message": "read-only filesystem",
        "code": "STATUS_CODE_ERROR"
       }
      },
      {
       "traceId": "S/kvNXezTaajzpKdDg5HNg==",
       "spanId": "APBnqgupArs=",
       "parentSpanId": "APBnqgupArg=",
       "name": "POST",
       "kind": "SPAN_KIND_CLIENT",
       "startTimeUnixNano": "1755500000012000000",
       "endTimeUnixNano": "1755500001890000000",
       "attributes": [
        {
         "key": "http.request.method",
         "value": {
          "stringValue": "POST"
         }
        },
        {
         "key": "url.full",
         "value": {
          "stringValue": "https://api.anthropic.com/v1/messages"
         }
        },
        {
         "key": "http.response.status_code",
         "value": {
          "intValue": "200"
         }
        }
       ]
      }
     ],
     "schemaUrl": "https://opentelemetry.io/schemas/1.40.0"
    }
   ],
   "schemaUrl": "https://opentelemetry.io/schemas/1.40.0"
  }
 ]
}
"""

TRACE_ID = "S/kvNXezTaajzpKdDg5HNg=="
"""The fixture's trace id, base64 — which is what the pinned library writes for
a 16-byte id, and the reason this edge never re-encodes one."""

ROOT_SPAN_ID = "APBnqgupArc="
CHAT_SPAN_ID = "APBnqgupArg="


def payload() -> dict[str, Any]:
    """A fresh copy of the producer-derived payload."""
    return json.loads(PRODUCER_PAYLOAD_JSON)


def only_trace(document: dict[str, Any] | None = None) -> CanonicalTrace:
    traces, envelope = otlp_json_to_ir(document if document is not None else payload())
    assert envelope.lossless, envelope.records
    assert len(traces) == 1
    return traces[0]


# ---------------------------------------------------------------------------
# Payload builders for the cases a real producer will not write
# ---------------------------------------------------------------------------


def S(value: str) -> dict[str, Any]:
    return {"stringValue": value}


def I(value: int | str) -> dict[str, Any]:  # noqa: E743 - mirrors the OTLP name
    return {"intValue": value}


def attrs(*pairs: tuple[str, Any]) -> list[dict[str, Any]]:
    return [{"key": key, "value": value} for key, value in pairs]


def span(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "traceId": TRACE_ID,
        "spanId": ROOT_SPAN_ID,
        "name": "span",
        "startTimeUnixNano": "1755500000000000000",
        "endTimeUnixNano": "1755500001000000000",
    }
    base.update(overrides)
    return base


def tool_span(**overrides: Any) -> dict[str, Any]:
    attributes = overrides.pop("attributes", [])
    return span(
        name="execute_tool read_file",
        attributes=attrs((GEN_AI_OPERATION_NAME, S(OPERATION_EXECUTE_TOOL)))
        + attributes,
        **overrides,
    )


def tool_span_without_timestamps(**overrides: Any) -> dict[str, Any]:
    """An ``execute_tool`` span a producer wrote with no readable instants.

    The degenerate shape the field-level contract guard used to walk straight
    past: the fixture fills ``tool_call.started_at``, so the guard was satisfied
    for that path on every payload, including this one where it is empty.
    """
    raw = tool_span(**overrides)
    raw.pop("startTimeUnixNano", None)
    raw.pop("endTimeUnixNano", None)
    return raw


def wrap(*spans: dict[str, Any], **envelope: Any) -> dict[str, Any]:
    scope_spans: dict[str, Any] = {"spans": list(spans)}
    scope_spans.update(envelope.pop("scope_spans", {}))
    resource_spans: dict[str, Any] = {"scopeSpans": [scope_spans]}
    resource_spans.update(envelope.pop("resource_spans", {}))
    assert not envelope, envelope
    return {"resourceSpans": [resource_spans]}


def convert(*spans: dict[str, Any]) -> CanonicalTrace:
    """One trace from bare spans, with no envelope in the way."""
    return otel_spans_to_ir([OtlpSpan(span=one) for one in spans])


def fields(trace: CanonicalTrace, loss_class: LossClass | None = None) -> set[str]:
    return {
        record.field
        for record in trace.losses.records
        if record.space is PathSpace.HUB
        and (loss_class is None or record.loss_class is loss_class)
    }


def record_for(trace: CanonicalTrace, field: str) -> LossRecord:
    matching = trace.losses.for_field(field)
    assert len(matching) == 1, (field, matching)
    return matching[0]


# ---------------------------------------------------------------------------
# The producer-derived payload
# ---------------------------------------------------------------------------


def test_the_pinned_versions_are_the_ones_this_suite_reads():
    """The constants naming the evidence are not decoration.

    Every ``gen_ai.*`` string and every wire-shape assumption in the module
    under test was copied from these two versions. If the module's idea of what
    it was written against drifts from this suite's, the mapping stops being
    checkable against anything.
    """
    assert OTLP_PROTO_VERSION == "1.41.1"
    assert SEMCONV_VERSION == "0.62b1"
    assert "opentelemetry-proto==1.41.1" in _PRODUCER_PAYLOAD_RECIPE


def test_the_attribute_names_are_the_pinned_spellings():
    """Spelled out once, because a near-miss reads exactly like a hit.

    Each right-hand side is the literal value of the same-named constant in
    ``opentelemetry.semconv._incubating.attributes.gen_ai_attributes`` at
    :data:`SEMCONV_VERSION`. The deleted `OTelCollector` is the cautionary case:
    it read ``gen_ai.usage.cache_read_input_tokens``, and the attribute is
    ``gen_ai.usage.cache_read.input_tokens`` — one dot apart, and never a match.
    """
    assert GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
    assert GEN_AI_TOOL_NAME == "gen_ai.tool.name"
    assert GEN_AI_TOOL_CALL_ID == "gen_ai.tool.call.id"
    assert GEN_AI_TOOL_CALL_ARGUMENTS == "gen_ai.tool.call.arguments"
    assert OPERATION_EXECUTE_TOOL == "execute_tool"
    assert ir_from_otel_module.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS == (
        "gen_ai.usage.cache_read.input_tokens"
    )
    assert ir_from_otel_module.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS == (
        "gen_ai.usage.cache_creation.input_tokens"
    )
    assert not hasattr(ir_from_otel_module, "GEN_AI_USAGE_TOTAL_TOKENS")


def test_the_fixture_is_shaped_the_way_the_pinned_library_writes():
    """Guards the fixture itself against being 'tidied' into a hand-written one.

    Each assertion is an encoding choice ``MessageToJson`` made and a
    hand-written payload would probably get wrong — and each is one this edge
    has to handle: a base64 id it must not re-encode, an int64 as a JSON string,
    an enum as its member name, and a default-valued field written as absence.
    """
    document = payload()
    spans = document["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 5

    assert spans[0]["traceId"] == TRACE_ID
    assert spans[0]["traceId"].endswith("==")  # base64 of 16 bytes, not 32 hex

    chat = spans[1]
    usage = {pair["key"]: pair["value"] for pair in chat["attributes"]}
    assert usage["gen_ai.usage.input_tokens"] == {"intValue": "1204"}
    assert isinstance(chat["startTimeUnixNano"], str)
    assert chat["kind"] == "SPAN_KIND_CLIENT"
    assert spans[2]["status"] == {"code": "STATUS_CODE_OK"}

    # The root span has no parent, and a zero-valued ``parentSpanId`` is absent
    # rather than empty — the presence limit this edge declares.
    assert "parentSpanId" not in spans[0]
    assert spans[1]["parentSpanId"] == ROOT_SPAN_ID


def test_a_producer_payload_reads_into_one_valid_trace():
    trace = only_trace()
    assert validate_trace(trace) == []
    assert trace.losses is not None
    assert trace.losses.direction == LOSS_DIRECTION
    assert trace.provenance.source_format == OTLP_JSON_SOURCE
    assert len(trace.events) == 5
    assert all(event.provenance.source_format == OTEL_SOURCE for event in trace.events)
    assert all(event.provenance.producer == "benchflow.agent" for event in trace.events)


def test_only_execute_tool_becomes_a_typed_event():
    """The mapping's whole reach, asserted as a set rather than described.

    ``invoke_agent``, ``chat`` and a plain HTTP span are all ``UNKNOWN``. That
    is not a gap this suite is waiting to close — it is the edge refusing to
    read agent semantics into spans that do not carry them, and the assertion
    exists so widening it is a deliberate edit.
    """
    trace = only_trace()
    assert [event.kind for event in trace.events] == [
        EventKind.UNKNOWN,
        EventKind.UNKNOWN,
        EventKind.TOOL_CALL,
        EventKind.TOOL_CALL,
        EventKind.UNKNOWN,
    ]
    assert [event.source_type for event in trace.events] == [
        "invoke_agent solver",
        "chat claude-sonnet-5",
        "execute_tool read_file",
        "execute_tool write_file",
        "POST",
    ]


def test_identity_and_parentage_survive_verbatim():
    """The property the IR has no field for, and therefore the one at risk.

    The IR models no parent link, so the span graph lives in ``extensions``.
    Preserved is preserved: every id is the payload's own string, byte for byte,
    and the edge set is reconstructible from the trace alone.
    """
    document = payload()
    source_spans = document["resourceSpans"][0]["scopeSpans"][0]["spans"]
    trace = only_trace(document)

    assert trace.trace_id == TRACE_ID
    for event, source in zip(trace.events, source_spans, strict=True):
        carried = event.extensions["otel"]["span"]
        assert carried["spanId"] == source["spanId"]
        assert carried.get("parentSpanId") == source.get("parentSpanId")
        assert carried["traceId"] == source["traceId"]

    edges = {
        event.extensions["otel"]["span"]["spanId"]: event.extensions["otel"][
            "span"
        ].get("parentSpanId")
        for event in trace.events
    }
    assert edges[ROOT_SPAN_ID] is None
    assert edges[CHAT_SPAN_ID] == ROOT_SPAN_ID
    assert sum(1 for parent in edges.values() if parent == ROOT_SPAN_ID) == 3


def test_links_and_span_events_are_carried():
    """Two more structures with no IR home, and both explicitly in scope."""
    trace = only_trace()
    chat = trace.events[1].extensions["otel"]["span"]
    assert chat["events"] == [
        {"timeUnixNano": "1755500001200000000", "name": "first_token"}
    ]
    tool = trace.events[2].extensions["otel"]["span"]
    assert tool["links"] == [{"traceId": TRACE_ID, "spanId": CHAT_SPAN_ID}]


def test_two_structurally_distinct_scope_spans_stay_distinguishable():
    """F-1: flattening the envelope must not flatten the partition.

    The producer batched these two spans under *separate* ``ScopeSpans``
    objects. Their ``scope`` payloads are equal, so once the envelope is
    flattened into one list of spans, nothing except the coordinates says they
    were ever apart — and "the producer grouped these separately" is a fact
    about the payload, not noise.
    """
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {"scope": {"name": "s"}, "spans": [span(name="a")]},
                    {"scope": {"name": "s"}, "spans": [span(name="b")]},
                ],
            }
        ]
    }
    trace = only_trace(payload)
    envelopes = [event.extensions["otel"]["envelope"] for event in trace.events]
    assert envelopes == [
        {"resource_spans_index": 0, "scope_spans_index": 0, "span_index": 0},
        {"resource_spans_index": 0, "scope_spans_index": 1, "span_index": 0},
    ]
    # The scopes themselves are indistinguishable, which is exactly why the
    # coordinates have to carry the difference.
    assert (
        trace.events[0].extensions["otel"]["scope"]
        == trace.events[1].extensions["otel"]["scope"]
    )


def test_two_spans_in_one_scope_spans_are_also_distinguishable():
    """The complement: same group, different position within it."""
    payload = wrap(span(name="a"), span(name="b"))
    trace = only_trace(payload)
    assert [
        event.extensions["otel"]["envelope"]["span_index"] for event in trace.events
    ] == [0, 1]
    assert {
        event.extensions["otel"]["envelope"]["scope_spans_index"]
        for event in trace.events
    } == {0}


def test_a_span_the_caller_built_carries_no_envelope_coordinates():
    """No payload, no partition — and no invented coordinates.

    ``otel_spans_to_ir`` accepts spans a caller assembled itself. Writing
    ``span_index: 0`` for those would claim an envelope position that never
    existed, which is the same class of invention as a synthesized id.
    """
    trace = convert(span())
    assert "envelope" not in trace.events[0].extensions["otel"]


def test_resource_and_scope_ride_with_every_event():
    """Per event, not per trace — one payload legitimately mixes them."""
    trace = only_trace()
    for event in trace.events:
        carried = event.extensions["otel"]
        assert carried["scope"] == {"name": "benchflow.agent", "version": "0.1.0"}
        assert carried["scope_schema_url"] == "https://opentelemetry.io/schemas/1.40.0"
        assert carried["resource_schema_url"] == (
            "https://opentelemetry.io/schemas/1.40.0"
        )
        assert carried["resource"]["attributes"][0]["key"] == "service.name"


def test_the_tool_call_reads_every_pinned_attribute():
    trace = only_trace()
    call = trace.events[2].tool_call
    assert call == ToolCall(
        call_id="toolu_01A",
        name="read_file",
        name_semantics=GEN_AI_TOOL_NAME,
        title=None,
        status=None,
        arguments={"path": "/repo/README.md", "limit": 200},
        content=call.content,
        started_at=trace.events[2].started_at,
        finished_at=trace.events[2].finished_at,
    )
    assert [(block.kind, block.text) for block in call.content] == [
        (ContentBlockKind.TEXT, "# benchflow\n")
    ]
    # An observed argument map is not a declared absence.
    assert trace.losses.for_field("events[2].tool_call.arguments") == []


def test_a_tool_span_is_the_only_source_that_has_ever_timed_a_tool_call():
    """`§5 loss #3` closed from one side.

    ACP tracks the two instants in memory and serializes neither; ATIF has no
    slot at all. An ``execute_tool`` span *is* the execution, so its extent is
    the call's without inference — the first inbound edge in this family that
    can fill these fields.
    """
    trace = only_trace()
    call = trace.events[2].tool_call
    assert call.started_at is not None and call.finished_at is not None
    assert call.started_at < call.finished_at


def test_usage_is_per_span_and_says_which_definition_it_used():
    trace = only_trace()
    assert trace.events[1].usage == TraceUsage(
        input_tokens=1204,
        output_tokens=87,
        cache_read_tokens=1024,
        source=USAGE_SOURCE,
    )
    assert all(event.usage is None for event in trace.events if event.index != 1)
    # And the run total is not computed from it.
    assert trace.usage is None
    assert record_for(trace, "usage").loss_class is LossClass.NORMALIZED


def test_agent_identity_is_gathered_from_the_spans_that_carry_it():
    trace = only_trace()
    assert trace.agent.agent_name == "solver"
    assert trace.agent.agent_version == "2.1.0"
    assert trace.agent.model == "claude-sonnet-5"
    assert trace.agent.provider == "anthropic"
    assert not [field for field in fields(trace) if field.startswith("agent.")]


def test_a_conversation_id_is_not_read_as_a_session_id():
    """The plausible mapping this edge declines to make.

    ``gen_ai.conversation.id`` is "a conversation (session, thread)" in the
    pinned package, which is close enough that reading it into ``session_id``
    would look right and would quietly settle a question nobody asked. The
    value is preserved; the mapping is left to a maintainer (§8.11).

    Found by mutation: making this edge read the attribute left the suite green
    until this test existed.
    """
    trace = only_trace()
    assert trace.session_id is None
    record = record_for(trace, "session_id")
    assert record.loss_class is LossClass.UNSUPPORTED
    assert "gen_ai.conversation.id" in record.detail
    assert (
        trace.events[0].extensions["otel"]["attributes"]["gen_ai.conversation.id"]
        == "conv-7731"
    )


def test_a_caller_supplied_session_id_is_used_and_declares_nothing():
    """The context an inbound edge is allowed to take: from its caller."""
    spans, _ = otlp_json_spans(payload())
    trace = otel_spans_to_ir(spans, session_id="rollout-42")
    assert trace.session_id == "rollout-42"
    assert trace.losses is not None
    assert trace.losses.for_field("session_id") == []


def test_sub_microsecond_precision_is_declared_and_kept():
    """The loss only a real OTLP timestamp produces.

    ``datetime`` resolves to microseconds and OTLP counts nanoseconds, so the
    IR field cannot hold the value. Declaring it is not enough on its own —
    what makes it a preserved value rather than a lost one is that the exact
    integer is still in the document.
    """
    trace = only_trace()
    record = record_for(trace, "events[1].finished_at")
    assert record.loss_class is LossClass.NORMALIZED
    assert "375 ns is truncated" in record.detail
    assert trace.events[1].extensions["otel"]["span"]["endTimeUnixNano"] == (
        "1755500001900000375"
    )
    assert trace.events[1].finished_at is not None
    assert trace.events[1].finished_at.microsecond == 900000


def test_the_producer_declaring_its_own_drop_is_unsupported_not_dropped():
    """``droppedAttributesCount`` is loss that happened before this reader.

    The class is the whole point: no converter can recover an attribute the
    emitting SDK discarded, so the fix is a producer-side limit rather than
    code here. Calling it ``DROPPED`` would point the reader at this file.
    """
    trace = only_trace()
    records = [
        record
        for record in trace.losses.for_field("events[2].extensions")
        if "droppedAttributesCount" in record.detail
    ]
    assert len(records) == 1
    assert records[0].loss_class is LossClass.UNSUPPORTED


# ---------------------------------------------------------------------------
# Ordering, and the causality it is not
# ---------------------------------------------------------------------------


def test_document_order_is_preserved_and_time_order_is_not_imposed():
    """The explicit instruction, made a property.

    A payload whose spans are out of chronological order stays out of order.
    Sorting would be the converter claiming a sequence OTLP does not carry —
    siblings overlap, and the real structure is the parent edge set, which is
    preserved regardless.
    """
    late = span(
        spanId="AAAAAAAAAAE=", name="late", startTimeUnixNano="1755500009000000000"
    )
    early = span(
        spanId="AAAAAAAAAAI=", name="early", startTimeUnixNano="1755500001000000000"
    )
    trace = convert(late, early)

    assert [event.source_type for event in trace.events] == ["late", "early"]
    assert [event.index for event in trace.events] == [0, 1]
    assert trace.events[0].started_at > trace.events[1].started_at


def test_an_unreadable_span_leaves_no_hole_in_the_index():
    """Invariant 2 under an envelope that carries something unreadable."""
    document = wrap(span(name="a"), "not a span", span(name="b"))
    traces, envelope = otlp_json_to_ir(document)
    assert [record.field for record in envelope.records] == [
        "resourceSpans[0].scopeSpans[0].spans[1]"
    ]
    assert envelope.records[0].space is PathSpace.SOURCE
    assert len(traces) == 1
    assert [event.index for event in traces[0].events] == [0, 1]
    assert [event.source_type for event in traces[0].events] == ["a", "b"]


def test_the_run_extent_is_never_derived_from_the_spans():
    """Even when every span is timed, the trace-level fields stay empty."""
    trace = convert(span(name="a"), span(name="b"))
    assert trace.started_at is None and trace.finished_at is None
    assert all(event.started_at is not None for event in trace.events)
    for field in ("started_at", "finished_at"):
        assert record_for(trace, field).loss_class is LossClass.NORMALIZED


def test_with_no_readable_timestamp_the_run_extent_is_unsupported_instead():
    """The class tracks the reason, not the outcome.

    Both cases leave the field ``None``. ``NORMALIZED`` says the information
    exists per span and was not aggregated; ``UNSUPPORTED`` says there was
    nothing to aggregate. Collapsing them would lose where a fix would land.
    """
    trace = convert(
        {"traceId": TRACE_ID, "spanId": ROOT_SPAN_ID, "name": "untimed"},
    )
    for field in ("started_at", "finished_at"):
        assert record_for(trace, field).loss_class is LossClass.UNSUPPORTED


# ---------------------------------------------------------------------------
# What this edge refuses to invent
# ---------------------------------------------------------------------------


def test_no_event_is_ever_attributed_to_a_speaker():
    trace = only_trace()
    assert all(event.role is None for event in trace.events)
    assert record_for(trace, "events[].role").loss_class is LossClass.UNSUPPORTED


def test_no_conversation_text_is_read_out_of_message_attributes():
    """`gen_ai.input.messages` is carried, not interpreted.

    The pinned package defines the message structure by reference to a JSON
    schema it does not ship, so reading it into ``text`` would be a mapping
    against a document nobody in this repository can check.
    """
    trace = only_trace()
    assert all(event.text is None for event in trace.events)
    assert all(event.reasoning is None for event in trace.events)
    record = record_for(trace, "events[1].text")
    assert record.loss_class is LossClass.NORMALIZED
    assert "gen_ai.input.messages" in record.detail
    assert (
        trace.events[1].extensions["otel"]["attributes"]["gen_ai.input.messages"]
        == '[{"role":"user","parts":[{"type":"text","content":"list the files"}]}]'
    )


def test_nothing_is_synthesized_on_an_inbound_edge():
    """An inbound edge has no target to satisfy, so it fabricates nothing."""
    trace = only_trace()
    assert trace.losses.by_class(LossClass.SYNTHESIZED) == []


def test_a_tool_span_with_no_id_and_no_name_gets_neither():
    trace = convert(tool_span())
    call = trace.events[0].tool_call
    assert call is not None
    assert call.call_id is None and call.name is None
    assert call.name_semantics is None
    assert (
        record_for(trace, "events[0].tool_call.call_id").loss_class
        is LossClass.UNSUPPORTED
    )
    record = record_for(trace, "events[0].tool_call.name")
    assert record.loss_class is LossClass.UNSUPPORTED
    # And the span name is not quietly used instead.
    assert "execute_tool read_file" not in str(call)


def test_a_serialized_argument_string_is_not_parsed():
    """The pinned text puts deserialization on the instrumentation, not here.

    Parsing would be the converter deciding a string that happens to be JSON
    was meant as structure. The string is preserved; the refusal is declared;
    invariant 7 is satisfied by that same record.
    """
    trace = convert(
        tool_span(
            attributes=attrs((GEN_AI_TOOL_CALL_ARGUMENTS, S('{"path": "/tmp/x"}')))
        )
    )
    call = trace.events[0].tool_call
    assert call is not None and call.arguments is None
    record = record_for(trace, "events[0].tool_call.arguments")
    assert record.loss_class is LossClass.NORMALIZED
    assert "serialized string" in record.detail
    assert (
        trace.events[0].extensions["otel"]["attributes"][GEN_AI_TOOL_CALL_ARGUMENTS]
        == '{"path": "/tmp/x"}'
    )
    assert validate_trace(trace) == []


def test_an_observed_empty_argument_map_is_not_an_absence():
    """The tri-state rule at the one place the IR makes it an invariant."""
    trace = convert(
        tool_span(
            attributes=attrs(
                (GEN_AI_TOOL_CALL_ARGUMENTS, {"kvlistValue": {"values": []}})
            )
        )
    )
    call = trace.events[0].tool_call
    assert call is not None
    assert call.arguments == {}
    assert trace.losses.for_field("events[0].tool_call.arguments") == []
    assert validate_trace(trace) == []


def test_no_trace_id_is_invented():
    trace = convert({"spanId": ROOT_SPAN_ID, "name": "orphan"})
    assert trace.trace_id is None
    assert record_for(trace, "trace_id").loss_class is LossClass.UNSUPPORTED


def test_identifiers_are_never_re_encoded():
    """Hex and base64 ids are not distinguishable, so neither is normalized.

    The pinned JSON parser accepts a 32-character hex trace id *as base64* and
    yields 24 bytes from it, so a reader that decided which encoding it was
    looking at would be guessing. Both survive as written.
    """
    hex_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    trace = convert(span(traceId=hex_id))
    assert trace.trace_id == hex_id
    assert trace.events[0].extensions["otel"]["span"]["traceId"] == hex_id

    base64_trace = convert(span())
    assert base64_trace.trace_id == TRACE_ID


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def test_one_payload_can_hold_several_traces():
    other = "AAAAAAAAAAAAAAAAAAAAAA=="
    document = wrap(
        span(name="a"),
        span(traceId=other, name="b"),
        span(name="c"),
    )
    traces, envelope = otlp_json_to_ir(document)
    assert envelope.lossless
    assert [trace.trace_id for trace in traces] == [TRACE_ID, other]
    assert [event.source_type for event in traces[0].events] == ["a", "c"]
    assert [event.source_type for event in traces[1].events] == ["b"]


def test_spans_with_no_trace_id_group_together_rather_than_joining_one():
    """ "May belong to that trace" is not a fact this reader gets to record."""
    groups = group_spans_by_trace_id(
        [
            OtlpSpan(span=span(name="a")),
            OtlpSpan(span={"name": "b"}),
            OtlpSpan(span=span(name="c")),
        ]
    )
    assert [key for key, _ in groups] == [TRACE_ID, None]
    assert [len(members) for _, members in groups] == [2, 1]


def test_a_mixed_group_keeps_the_first_id_and_lists_them_all():
    other = "AAAAAAAAAAAAAAAAAAAAAA=="
    trace = convert(span(name="a"), span(traceId=other, name="b"))
    assert trace.trace_id == TRACE_ID
    assert trace.extensions["otel"]["trace_ids"] == [TRACE_ID, other]
    assert record_for(trace, "trace_id").loss_class is LossClass.NORMALIZED


def test_a_non_string_trace_id_is_reported_against_the_source():
    trace = convert(span(traceId=17))
    assert trace.trace_id is None
    source = trace.losses.for_field("spans[0].traceId", PathSpace.SOURCE)
    assert len(source) == 1 and source[0].loss_class is LossClass.DROPPED


def test_an_envelope_report_belongs_to_the_payload_not_to_a_trace():
    """The one asymmetry in this family, asserted so it stays deliberate."""
    document = {"resourceSpans": [{"scopeSpans": "not a list"}, 7]}
    spans, envelope = otlp_json_spans(document)
    assert spans == []
    assert {record.field for record in envelope.records} == {
        "resourceSpans[0].scopeSpans",
        "resourceSpans[1]",
    }
    assert all(record.space is PathSpace.SOURCE for record in envelope.records)
    traces, _ = otlp_json_to_ir(document)
    assert traces == []


def test_an_empty_payload_reads_as_no_traces_rather_than_an_empty_one():
    traces, envelope = otlp_json_to_ir({})
    assert traces == []
    assert envelope.lossless


def test_a_non_mapping_payload_is_refused():
    with pytest.raises(TypeError):
        otlp_json_spans([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"resourceSpans": {}}, {"resourceSpans"}),
        ({"resourceSpans": [1]}, {"resourceSpans[0]"}),
        (
            {"resourceSpans": [{"scopeSpans": [{"spans": 3}]}]},
            {"resourceSpans[0].scopeSpans[0].spans"},
        ),
        (
            {"resourceSpans": [{"scopeSpans": [None]}]},
            {"resourceSpans[0].scopeSpans[0]"},
        ),
    ],
)
def test_every_envelope_level_declares_what_it_could_not_read(document, expected):
    spans, envelope = otlp_json_spans(document)
    assert spans == []
    assert {record.field for record in envelope.records} == expected


# ---------------------------------------------------------------------------
# AnyValue: the three states, and the ones a map cannot hold
# ---------------------------------------------------------------------------


def test_the_three_states_of_an_attribute_value_are_kept_apart():
    """Present-and-empty, present-but-typeless, and absent.

    ``{"stringValue": ""}`` is an observed empty string. ``{}`` is protobuf's
    "no oneof member set". A ``KeyValue`` with no ``value`` key is a third
    thing again. A plain map can hold the first and collapses the other two to
    ``None`` — so the map is not the record: the original list is kept beside
    it and the collapse is declared.
    """
    trace = convert(
        span(
            attributes=[
                {"key": "empty_string", "value": {"stringValue": ""}},
                {"key": "typeless", "value": {}},
                {"key": "valueless"},
            ]
        )
    )
    carried = trace.events[0].extensions["otel"]
    assert carried["attributes"] == {
        "empty_string": "",
        "typeless": None,
        "valueless": None,
    }
    assert carried["attributes_raw"] == [
        {"key": "empty_string", "value": {"stringValue": ""}},
        {"key": "typeless", "value": {}},
        {"key": "valueless"},
    ]
    record = record_for(trace, "events[0].extensions")
    assert record.loss_class is LossClass.NORMALIZED
    assert "typeless" in record.detail and "valueless" in record.detail


def test_faithful_decoding_is_semantic_not_wire_invertible():
    """F-2: the two int64 spellings collapse, and that is declared.

    The canonical protobuf JSON mapping writes an ``int64`` as a *string*; much
    of the ecosystem writes it as a number. Both are the value ``7``, so nothing
    semantic is lost and ``attributes_raw`` is correctly not kept — but the two
    payloads are indistinguishable afterwards, so the edge must not be described
    as preserving the wire form.
    """
    as_string = convert(span(attributes=attrs(("k", I("7")))))
    as_number = convert(span(attributes=attrs(("k", I(7)))))
    carried = [trace.events[0].extensions["otel"] for trace in (as_string, as_number)]
    assert carried[0]["attributes"] == carried[1]["attributes"] == {"k": 7}
    assert all("attributes_raw" not in one for one in carried)

    for trace in (as_string, as_number):
        record = record_for(trace, "events[].extensions")
        assert record.loss_class is LossClass.NORMALIZED
        assert "wire form is not" in record.detail


def test_the_wire_normalization_is_declared_once_not_per_span():
    """A property of the decoding, so declaring it per span would be noise.

    §8.6's affordability argument: a record that repeats one sentence per event
    multiplies the report by the trace length while adding nothing.
    """
    one = convert(span(attributes=attrs(("k", S("v")))))
    many = convert(*[span(attributes=attrs(("k", S("v")))) for _ in range(10)])
    assert len(one.losses.for_field("events[].extensions")) == 1
    assert len(many.losses.for_field("events[].extensions")) == 1


def test_a_span_with_no_attributes_declares_no_wire_normalization():
    """Nothing was decoded, so there is nothing to say about how."""
    trace = convert(span())
    assert trace.losses.for_field("events[].extensions") == []


def test_a_faithful_attribute_list_keeps_no_raw_copy():
    """The complement, so the guard above is not satisfied by always copying."""
    trace = convert(span(attributes=attrs(("a", S("x")), ("b", I("2")))))
    carried = trace.events[0].extensions["otel"]
    assert carried["attributes"] == {"a": "x", "b": 2}
    assert "attributes_raw" not in carried
    assert trace.losses.for_field("events[0].extensions") == []


def test_duplicate_attribute_keys_cannot_be_silently_deduplicated():
    """OTLP models attributes as a list, so duplicates are representable."""
    trace = convert(span(attributes=attrs(("k", S("first")), ("k", S("second")))))
    carried = trace.events[0].extensions["otel"]
    assert carried["attributes"] == {"k": "second"}
    assert len(carried["attributes_raw"]) == 2
    assert "repeats key" in record_for(trace, "events[0].extensions").detail


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"stringValue": "x"}, "x"),
        ({"boolValue": False}, False),
        ({"intValue": "42"}, 42),
        ({"intValue": 42}, 42),
        ({"intValue": "-1"}, -1),
        ({"doubleValue": 0.0}, 0.0),
        ({"doubleValue": 2}, 2.0),
        (
            {"arrayValue": {"values": [{"intValue": "1"}, {"stringValue": "a"}]}},
            [1, "a"],
        ),
        ({"arrayValue": {"values": []}}, []),
        (
            {"kvlistValue": {"values": [{"key": "k", "value": {"boolValue": True}}]}},
            {"k": True},
        ),
    ],
)
def test_every_any_value_member_this_reader_knows(value, expected):
    """Falsy values included — the deleted `OTelCollector` collapsed them.

    Its ``_parse_attributes`` read
    ``stringValue or intValue or doubleValue or boolValue``, which turns ``""``,
    ``0`` and ``False`` into "no value". They are values.
    """
    trace = convert(span(attributes=attrs(("k", value))))
    carried = trace.events[0].extensions["otel"]
    assert carried["attributes"]["k"] == expected
    assert "attributes_raw" not in carried


@pytest.mark.parametrize(
    "spelling",
    ["NaN", "Infinity", "-Infinity"],
)
def test_the_non_finite_doubles_are_read_from_their_json_spelling(spelling):
    trace = convert(span(attributes=attrs(("k", {"doubleValue": spelling}))))
    value = trace.events[0].extensions["otel"]["attributes"]["k"]
    assert isinstance(value, float)
    assert (value != value) if spelling == "NaN" else abs(value) == float("inf")


@pytest.mark.parametrize(
    "value",
    [
        {"bytesValue": "AQI="},
        {"stringValue": "x", "intValue": "1"},
        {"futureValue": 1},
        {"intValue": "not a number"},
        {"intValue": True},
        {"doubleValue": "3.5"},
        {"arrayValue": {"values": [{"futureValue": 1}]}},
        {"arrayValue": {"values": "no"}},
        {"kvlistValue": {"values": [{"key": "k"}]}},
        {
            "kvlistValue": {
                "values": [
                    {"key": "k", "value": {"stringValue": "a"}},
                    {"key": "k", "value": {"stringValue": "b"}},
                ]
            }
        },
        "not an object",
    ],
)
def test_an_any_value_a_map_cannot_hold_keeps_the_list_and_says_so(value):
    """Every refusal is declared; none of them loses the payload."""
    trace = convert(span(attributes=attrs(("k", value))))
    carried = trace.events[0].extensions["otel"]
    assert carried["attributes_raw"] == [{"key": "k", "value": value}]
    assert record_for(trace, "events[0].extensions").loss_class is LossClass.NORMALIZED
    assert validate_trace(trace) == []


def test_a_malformed_attribute_list_is_declared_rather_than_ignored():
    trace = convert(span(attributes={"k": "v"}))
    carried = trace.events[0].extensions["otel"]
    assert carried["attributes"] == {}
    assert carried["attributes_raw"] == {"k": "v"}
    assert "not a list" in record_for(trace, "events[0].extensions").detail


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "marker"),
    [
        (None, "is absent"),
        ("0", "protobuf default"),
        (0, "protobuf default"),
        ("-1", "unsigned"),
        ("later", "not an integer"),
        (True, "bool"),
        ([], "list"),
    ],
)
def test_an_unreadable_timestamp_is_declared_and_never_guessed(value, marker):
    raw = span()
    if value is None:
        raw.pop("startTimeUnixNano")
    else:
        raw["startTimeUnixNano"] = value
    trace = convert(raw)
    assert trace.events[0].started_at is None
    record = record_for(trace, "events[0].started_at")
    assert record.loss_class is LossClass.NORMALIZED
    assert marker in record.detail


def test_a_whole_microsecond_timestamp_declares_nothing():
    trace = convert(span())
    assert trace.losses.for_field("events[0].started_at") == []
    assert trace.events[0].started_at is not None
    assert trace.events[0].started_at.tzinfo is not None


def test_a_tool_calls_own_timestamps_are_declared_at_their_own_path():
    """F-3: both slots hold the value, so both absences must be addressable.

    A record at ``events[i].started_at`` does not answer for
    ``events[i].tool_call.started_at``: they are different paths, a reader
    checking the second one finds nothing, and "the parent said something" is
    not how the loss report is addressed. `ir_from_acp` and `ir_from_atif` both
    declare these two paths unconditionally; this edge declares them whenever it
    could not fill them.
    """
    trace = convert(tool_span_without_timestamps())
    call = trace.events[0].tool_call
    assert call is not None
    assert call.started_at is None and call.finished_at is None
    for field in ("started_at", "finished_at"):
        record = record_for(trace, f"events[0].tool_call.{field}")
        assert record.loss_class is LossClass.NORMALIZED
        assert "the same instant as its span" in record.detail
        # and the event-level path is still declared in its own right
        assert trace.losses.for_field(f"events[0].{field}")


def test_a_filled_tool_call_timestamp_declares_nothing():
    """The complement — otherwise the record above would be unconditional."""
    trace = convert(tool_span())
    call = trace.events[0].tool_call
    assert call.started_at is not None and call.finished_at is not None
    assert trace.losses.for_field("events[0].tool_call.started_at") == []
    assert trace.losses.for_field("events[0].tool_call.finished_at") == []


def test_the_tool_call_inherits_the_truncation_record_too():
    """The 375 ns leave both fields, so both say so."""
    trace = convert(tool_span(endTimeUnixNano="1755500001000000375"))
    record = record_for(trace, "events[0].tool_call.finished_at")
    assert record.loss_class is LossClass.NORMALIZED
    assert "375 ns is truncated" in record.detail
    assert trace.events[0].tool_call.finished_at == trace.events[0].finished_at


# ---------------------------------------------------------------------------
# Status, results, usage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        {"code": "STATUS_CODE_ERROR", "message": "boom"},
        {"code": 2},
        {"code": "STATUS_CODE_OK"},
        {"code": 1},
        {"message": "note"},
    ],
)
def test_a_span_status_is_carried_and_declared_but_never_mapped(status):
    """Both enum spellings the pinned mapping produces are recognized."""
    trace = convert(span(status=status))
    assert trace.events[0].outcome is None
    assert trace.events[0].extensions["otel"]["span"]["status"] == status
    assert record_for(trace, "events[0].outcome").loss_class is LossClass.NORMALIZED


@pytest.mark.parametrize(
    "status", [None, {}, {"code": 0}, {"code": "STATUS_CODE_UNSET"}]
)
def test_an_unset_status_declares_nothing(status):
    """``STATUS_CODE_UNSET`` is the protobuf default: the span said nothing."""
    raw = span()
    if status is not None:
        raw["status"] = status
    trace = convert(raw)
    assert trace.losses.for_field("events[0].outcome") == []


def test_a_tool_calls_status_records_which_question_was_left_open():
    with_status = convert(tool_span(status={"code": "STATUS_CODE_ERROR"}))
    without = convert(tool_span())
    assert (
        record_for(with_status, "events[0].tool_call.status").loss_class
        is LossClass.NORMALIZED
    )
    assert (
        record_for(without, "events[0].tool_call.status").loss_class
        is LossClass.UNSUPPORTED
    )
    assert with_status.events[0].tool_call is not None
    assert with_status.events[0].tool_call.status is None


def test_an_object_result_becomes_an_opaque_block():
    result = {"kvlistValue": {"values": [{"key": "rows", "value": {"intValue": "3"}}]}}
    trace = convert(tool_span(attributes=attrs(("gen_ai.tool.call.result", result))))
    call = trace.events[0].tool_call
    assert call is not None
    assert [(block.kind, block.raw) for block in call.content] == [
        (ContentBlockKind.OPAQUE, {"rows": 3})
    ]
    assert validate_trace(trace) == []


def test_a_result_that_is_neither_text_nor_an_object_is_not_wrapped():
    trace = convert(
        tool_span(
            attributes=attrs(
                ("gen_ai.tool.call.result", {"arrayValue": {"values": [S("a")]}})
            )
        )
    )
    call = trace.events[0].tool_call
    assert call is not None and call.content == []
    record = record_for(trace, "events[0].tool_call.content")
    assert record.loss_class is LossClass.NORMALIZED
    assert trace.events[0].extensions["otel"]["attributes"][
        "gen_ai.tool.call.result"
    ] == ["a"]


def test_a_deprecated_usage_spelling_is_read_and_the_reading_is_declared():
    """The pinned package states the replacement, so following it is supported.

    Declaring it anyway is what lets a consumer tell a ``prompt_tokens``
    document from an ``input_tokens`` one after the fact.
    """
    trace = convert(span(attributes=attrs(("gen_ai.usage.prompt_tokens", I("11")))))
    assert trace.events[0].usage is not None
    assert trace.events[0].usage.input_tokens == 11
    record = record_for(trace, "events[0].usage.input_tokens")
    assert record.loss_class is LossClass.NORMALIZED
    assert "gen_ai.usage.prompt_tokens" in record.detail


def test_the_preferred_spelling_wins_and_the_disagreement_is_declared():
    trace = convert(
        span(
            attributes=attrs(
                ("gen_ai.usage.input_tokens", I("11")),
                ("gen_ai.usage.prompt_tokens", I("99")),
            )
        )
    )
    assert trace.events[0].usage is not None
    assert trace.events[0].usage.input_tokens == 11
    classes = {
        record.loss_class
        for record in trace.losses.for_field("events[0].usage.input_tokens")
    }
    assert classes == {LossClass.DROPPED}


def test_a_token_count_that_is_not_a_count_is_declared_rather_than_coerced():
    trace = convert(span(attributes=attrs(("gen_ai.usage.input_tokens", S("many")))))
    assert trace.events[0].usage is None
    assert (
        record_for(trace, "events[0].usage.input_tokens").loss_class
        is LossClass.DROPPED
    )


def test_gen_ai_usage_total_tokens_is_not_in_the_pinned_vocabulary():
    """A concrete correction of the deleted collector, kept as a test.

    ``otel.py`` read ``gen_ai.usage.total_tokens``; no such attribute exists at
    :data:`SEMCONV_VERSION`, so a document carrying it is carried, not read.
    """
    trace = convert(span(attributes=attrs(("gen_ai.usage.total_tokens", I("300")))))
    assert trace.events[0].usage is None
    assert (
        trace.events[0].extensions["otel"]["attributes"]["gen_ai.usage.total_tokens"]
        == 300
    )


def test_the_deprecated_system_attribute_fills_provider_and_says_so():
    trace = convert(span(attributes=attrs(("gen_ai.system", S("anthropic")))))
    assert trace.agent.provider == "anthropic"
    record = record_for(trace, "agent.provider")
    assert record.loss_class is LossClass.NORMALIZED
    assert "gen_ai.system" in record.detail


def test_the_current_provider_attribute_is_preferred_over_the_deprecated_one():
    trace = convert(
        span(
            attributes=attrs(
                ("gen_ai.system", S("legacy")),
                ("gen_ai.provider.name", S("anthropic")),
            )
        )
    )
    assert trace.agent.provider == "anthropic"
    assert trace.losses.for_field("agent.provider") == []


def test_a_later_span_using_the_current_spelling_clears_the_deprecation():
    """The record has to describe where the value came from, not where it could.

    The first span offers the value through the deprecated attribute; a later
    span offers the same value through the current one. Keeping the
    ``NORMALIZED`` record would say the trace's provider is only readable with
    convention knowledge, which by then is false.
    """
    trace = convert(
        span(attributes=attrs(("gen_ai.system", S("anthropic")))),
        span(attributes=attrs(("gen_ai.provider.name", S("anthropic")))),
    )
    assert trace.agent.provider == "anthropic"
    assert trace.losses.for_field("agent.provider") == []


def test_spans_disagreeing_about_the_agent_declare_it():
    trace = convert(
        span(attributes=attrs(("gen_ai.request.model", S("a")))),
        span(attributes=attrs(("gen_ai.request.model", S("b")))),
    )
    assert trace.agent.model == "a"
    record = record_for(trace, "agent.model")
    assert record.loss_class is LossClass.DROPPED
    assert "'b'" in record.detail


def test_a_repeated_identical_agent_value_is_not_a_conflict():
    trace = convert(
        span(attributes=attrs(("gen_ai.request.model", S("a")))),
        span(attributes=attrs(("gen_ai.request.model", S("a")))),
    )
    assert trace.losses.for_field("agent.model") == []


def test_a_payload_with_no_gen_ai_attribute_still_converts():
    """The general-tracing case: nothing is a tool call, nothing is lost."""
    trace = convert(
        span(name="GET", attributes=attrs(("http.request.method", S("GET"))))
    )
    assert validate_trace(trace) == []
    assert trace.events[0].kind is EventKind.UNKNOWN
    assert trace.events[0].extensions["otel"]["attributes"] == {
        "http.request.method": "GET"
    }
    assert record_for(trace, "events[].usage").loss_class is LossClass.UNSUPPORTED


def test_a_span_with_no_name_reads_as_no_source_type():
    trace = convert({"traceId": TRACE_ID, "spanId": ROOT_SPAN_ID})
    assert trace.events[0].source_type is None
    record = record_for(trace, "events[].source_type")
    assert record.loss_class is LossClass.UNSUPPORTED
    assert "no presence" in record.detail


# ---------------------------------------------------------------------------
# Contract guards — the tests that make an undeclared loss fail the suite
# ---------------------------------------------------------------------------


def _nested_models(annotation: Any) -> list[type]:
    """The pydantic models reachable from a field annotation."""
    from pydantic import BaseModel

    found: list[type] = []
    stack = [annotation]
    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            found.append(current)
            continue
        stack.extend(getattr(current, "__args__", ()) or ())
    return found


def _is_list_of_models(annotation: Any) -> bool:
    """True for ``list[SomeModel]`` — the shape that needs an ``[]`` in its path."""
    import typing

    if typing.get_origin(annotation) is list:
        return bool(_nested_models(annotation))
    return any(_is_list_of_models(argument) for argument in typing.get_args(annotation))


def _model_paths(
    model: type, prefix: str = "", skip: frozenset[str] = frozenset()
) -> set[str]:
    """Every field of *model*, as the systemic loss path that would address it.

    Derived from the models rather than listed by hand, so a field added to the
    IR shows up here without anyone remembering to add it — which is the only
    reason the guards below are worth having.

    A ``list[Model]`` field contributes an ``[]`` segment, so ``events`` yields
    ``events[].text`` and ``tool_call.content`` yields
    ``tool_call.content[].raw``. That is the same spelling the converters use in
    their systemic records, and the reason the content blocks are inside the
    contract rather than quietly outside it.
    """
    paths: set[str] = set()
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        if path in skip:
            continue
        paths.add(path)
        nested = _nested_models(info.annotation)
        if not nested:
            continue
        child = f"{path}[]." if _is_list_of_models(info.annotation) else f"{path}."
        for model_type in nested:
            paths |= _model_paths(model_type, child, skip)
    return paths


#: Set by the converter on every trace and every event, so an absence is
#: impossible rather than undeclared. Everything else has to be filled or
#: declared, per instance.
STRUCTURAL = frozenset(
    {
        "ir_version",
        "extensions",
        "provenance",
        "provenance.source_format",
        "provenance.producer",
        "provenance.captured_at",
        "events",
        "events[].index",
        "events[].kind",
        "events[].provenance",
        "events[].provenance.source_format",
        "events[].provenance.producer",
        "events[].provenance.captured_at",
        # A required model field: a block with no kind cannot be constructed.
        "events[].tool_call.content[].kind",
    }
)


def _is_absent(value: Any) -> bool:
    """True only for ``None`` — the IR's "the source did not carry this".

    ``{}`` and ``[]`` are **not** absences. §8.2 makes that the whole point of
    the tri-state rule: ``arguments={}`` is "the source carried an empty
    argument map" and an ``OPAQUE`` block whose ``raw`` is ``{}`` carried an
    empty object. Treating them as missing would make the guard demand a
    declaration for a value that was observed.

    The known limit this inherits — stated when `ACP → IR` first met it — is
    that a converter can still dodge a declaration by writing ``{}`` where it
    means ``None``. Neither this guard nor invariant 7 can see that; only
    reading the produced document can.
    """
    return value is None


def _addressable_paths() -> set[str]:
    """Every path a hub loss record could legitimately name."""
    return _model_paths(CanonicalTrace, "", frozenset({"losses"})) | {"losses"}


def _contract_paths() -> set[str]:
    """The IR fields this edge owes an answer for, as systemic paths."""
    return _model_paths(CanonicalTrace, "", frozenset({"losses"})) - STRUCTURAL


def _systemic(field: str) -> str:
    """A concrete record's path, rewritten to the systemic form it answers."""
    import re

    return re.sub(r"\[\d+\]", "[]", field)


def _instances(path: str, trace: CanonicalTrace) -> list[str]:
    """*path* expanded to one concrete path per instance in *trace*.

    Exempts ``tool_call`` and its descendants on an event that is not a tool
    call: invariant 3 does not merely permit the absence, it **requires** it, so
    a declaration there would be asserting a loss the IR forbids having.
    """
    if "events[]" not in path:
        return [path]
    concrete: list[str] = []
    for index, event in enumerate(trace.events):
        tail = path[len("events[].") :]
        if tail.startswith("tool_call") and event.kind is not EventKind.TOOL_CALL:
            continue
        here = f"events[{index}].{tail}"
        if "content[]" not in here:
            concrete.append(here)
            continue
        blocks = event.tool_call.content if event.tool_call else []
        concrete.extend(
            here.replace("content[]", f"content[{position}]", 1)
            for position in range(len(blocks))
        )
    return concrete


def _covered(concrete: str, declared: set[str]) -> bool:
    """True when a record addresses *concrete*, its systemic form, or an ancestor.

    The ancestor rule is the IR's own "address the outermost absent node": a
    conversion with no usage at all declares ``usage``, not
    ``usage.input_tokens``, and the declaration covers everything beneath it.
    """
    for candidate in (concrete, _systemic(concrete)):
        parts = candidate.split(".")
        for cut in range(len(parts), 0, -1):
            if ".".join(parts[:cut]) in declared:
                return True
    return False


def _unanswered(trace: CanonicalTrace) -> list[str]:
    """Concrete paths that are neither filled nor declared — one per instance.

    This is the per-instance form of the contract. An earlier version asked only
    whether a *field* was ever filled or ever declared anywhere in the trace,
    which let a field that is filled on one event and silently empty on another
    pass; ``events[i].tool_call.started_at`` was exactly that hole.
    """
    declared = {
        record.field
        for record in (trace.losses.records if trace.losses else [])
        if record.space is PathSpace.HUB
    }
    document = trace.model_dump(mode="json")
    missing: list[str] = []
    for path in sorted(_contract_paths()):
        for concrete in _instances(path, trace):
            found, value = resolve_ir_path(document, concrete)
            if found and not _is_absent(value):
                continue
            if not _covered(concrete, declared):
                missing.append(concrete)
    return missing


def test_the_contract_path_list_is_not_empty_or_trivial():
    """The guard below is only worth anything if it enumerates something.

    Derived from the models, so a field added to the IR appears here without
    anyone remembering to add it — which is the point. The content-block paths
    are asserted explicitly because they were outside the contract until a
    review found them there.
    """
    paths = _contract_paths()
    assert len(paths) > 25
    assert {
        "trace_id",
        "session_id",
        "agent.agent_version",
        "outcome.stop_reason",
        "usage.cost_usd",
        "events[].role",
        "events[].tool_call.arguments",
        "events[].tool_call.started_at",
        "events[].tool_call.finished_at",
        "events[].tool_call.content[].text",
        "events[].tool_call.content[].raw",
        "events[].usage.total_tokens",
    } <= paths


@pytest.mark.parametrize(
    "trace_factory",
    [
        pytest.param(only_trace, id="producer-fixture"),
        pytest.param(lambda: convert(span()), id="bare-span"),
        pytest.param(lambda: convert(tool_span()), id="tool-span-with-nothing"),
        pytest.param(
            lambda: convert(
                tool_span(
                    status={"code": "STATUS_CODE_ERROR"},
                    attributes=attrs(
                        (GEN_AI_TOOL_NAME, S("read")),
                        (GEN_AI_TOOL_CALL_ID, S("t1")),
                        (GEN_AI_TOOL_CALL_ARGUMENTS, {"kvlistValue": {"values": []}}),
                        ("gen_ai.tool.call.result", S("out")),
                    ),
                )
            ),
            id="tool-span-with-everything",
        ),
        pytest.param(
            lambda: convert(
                tool_span(
                    attributes=attrs(
                        (
                            "gen_ai.tool.call.result",
                            {"kvlistValue": {"values": []}},
                        )
                    )
                )
            ),
            id="opaque-result-block",
        ),
        pytest.param(
            lambda: convert(
                span(attributes=attrs(("gen_ai.usage.input_tokens", I("5")))),
                span(),
            ),
            id="usage-on-one-span-only",
        ),
        pytest.param(
            lambda: convert(
                span(attributes=attrs(("gen_ai.usage.input_tokens", I("5")))),
                span(attributes=attrs(("gen_ai.usage.output_tokens", I("6")))),
            ),
            id="usage-fields-split-across-spans",
        ),
        pytest.param(
            lambda: convert({"traceId": TRACE_ID, "spanId": ROOT_SPAN_ID}),
            id="span-with-nothing-at-all",
        ),
    ],
)
def test_every_ir_field_is_filled_or_declared_on_every_instance(trace_factory):
    """The guard that makes an undeclared loss a test failure.

    **Per instance, not per field.** An earlier version asked only whether a
    field was filled *somewhere* or declared *somewhere*, which let a field that
    the fixture happens to fill pass forever — including on payloads where it is
    empty and nothing declares it. ``events[i].tool_call.started_at`` was
    exactly that hole, and the ``tool-span-with-nothing`` case is the payload
    that walked through it.

    The parameters matter as much as the assertion: a single trace can only
    exercise the instances it contains, so the corpus has to include the
    degenerate shapes — a span with no attributes, a tool call with no result,
    usage on some spans but not others.
    """
    assert _unanswered(trace_factory()) == []


def test_the_guard_catches_the_hole_it_was_strengthened_for():
    """Anti-tautology, aimed at the specific defect a review found.

    A tool span with no readable timestamps leaves ``tool_call.started_at`` and
    ``.finished_at`` empty. Before the fix nothing declared them and the
    field-level guard was satisfied by the fixture; now their absence is
    declared, and removing the declarations has to fail.
    """
    trace = convert(tool_span_without_timestamps())
    assert _unanswered(trace) == []
    assert trace.losses is not None

    stripped = trace.model_copy(
        update={
            "losses": LossReport(
                direction=trace.losses.direction,
                records=[
                    record
                    for record in trace.losses.records
                    if "tool_call.started_at" not in record.field
                    and "tool_call.finished_at" not in record.field
                ],
            )
        }
    )
    missing = _unanswered(stripped)
    assert "events[0].tool_call.started_at" in missing
    assert "events[0].tool_call.finished_at" in missing


def test_the_guard_fails_when_a_systemic_declaration_is_removed():
    """The other half: a systemic record covers every instance, so losing it
    must fail on every instance rather than on none."""
    trace = only_trace()
    assert trace.losses is not None
    stripped = trace.model_copy(
        update={
            "losses": LossReport(
                direction=trace.losses.direction,
                records=[
                    record
                    for record in trace.losses.records
                    if record.field != "events[].role"
                ],
            )
        }
    )
    missing = _unanswered(stripped)
    assert missing == [f"events[{index}].role" for index in range(len(trace.events))]


def test_a_content_block_declares_the_payload_field_it_does_not_carry():
    """F-8: the content descendants are inside the contract, not beside it.

    A ``TEXT`` block read from a string result has no ``raw`` — the attribute
    value *is* the text — and an ``OPAQUE`` block has no rendered ``text``.
    Neither absence is structural, so each is declared at its own concrete path,
    the way `ir_from_atif` declares ``content[].raw``.
    """
    text_block = convert(
        tool_span(attributes=attrs(("gen_ai.tool.call.result", S("out"))))
    )
    assert (
        record_for(text_block, "events[0].tool_call.content[0].raw").loss_class
        is LossClass.UNSUPPORTED
    )
    assert _unanswered(text_block) == []

    opaque = convert(
        tool_span(
            attributes=attrs(
                ("gen_ai.tool.call.result", {"kvlistValue": {"values": []}})
            )
        )
    )
    assert opaque.events[0].tool_call.content[0].kind is ContentBlockKind.OPAQUE
    assert (
        record_for(opaque, "events[0].tool_call.content[0].text").loss_class
        is LossClass.UNSUPPORTED
    )
    assert _unanswered(opaque) == []


def test_a_non_tool_event_owes_no_tool_call_declaration():
    """Invariant 3 forbids the payload, so its absence is not a loss.

    Without this exemption the guard would demand a declaration that the IR's
    own validator rejects the alternative to — and every plain span in every
    payload would need one.
    """
    trace = convert(span(name="plain"))
    assert trace.events[0].tool_call is None
    assert _unanswered(trace) == []
    assert not [
        record
        for record in trace.losses.records
        if record.field.startswith("events[0].tool_call")
    ]


def test_every_declared_path_names_a_field_that_exists():
    """The other direction: a typo in a path is a declaration nobody can check."""
    trace = only_trace()
    assert trace.losses is not None
    known = _addressable_paths()
    unknown = sorted(
        {
            _systemic(record.field)
            for record in trace.losses.records
            if record.space is PathSpace.HUB
        }
        - known
    )
    assert unknown == [], unknown


def test_every_concrete_loss_path_resolves_in_the_canonical_encoding():
    """A declaration a reader of the JSON cannot find is not a declaration.

    Same property the ACP and ATIF edges are held to, and the one whose
    violation `exclude_none=True` produced in §8.4.
    """
    trace = only_trace()
    assert trace.losses is not None
    canonical = trace.model_dump(mode="json")
    concrete = [
        record.field
        for record in trace.losses.records
        if record.space is PathSpace.HUB
        and record.field.startswith("events[")
        and not record.field.startswith("events[]")
    ]
    assert concrete
    unresolved = [
        field for field in concrete if not resolve_ir_path(canonical, field)[0]
    ]
    assert unresolved == [], unresolved

    lean = trace.model_dump(mode="json", exclude_none=True)
    assert any(not resolve_ir_path(lean, field)[0] for field in concrete)


def test_stripping_an_arguments_declaration_makes_the_trace_invalid():
    """Invariant 7, exercised against this edge rather than against a fixture.

    The IR promises that a converter which quietly fails to carry arguments
    produces an *invalid* trace. This is that promise, checked on the document
    this edge actually produces.
    """
    trace = convert(tool_span())
    assert validate_trace(trace) == []
    assert trace.losses is not None
    stripped = trace.model_copy(
        update={
            "losses": LossReport(
                direction=trace.losses.direction,
                records=[
                    record
                    for record in trace.losses.records
                    if record.field != "events[0].tool_call.arguments"
                ],
            )
        }
    )
    issues = validate_trace(stripped)
    assert len(issues) == 1
    assert "absence must be declared" in issues[0]


def test_a_target_space_record_cannot_satisfy_a_hub_invariant():
    """The space is part of the address, not decoration."""
    trace = convert(tool_span())
    assert trace.losses is not None
    relabelled = trace.model_copy(
        update={
            "losses": LossReport(
                direction=trace.losses.direction,
                records=[
                    record.model_copy(update={"space": PathSpace.TARGET})
                    if record.field == "events[0].tool_call.arguments"
                    else record
                    for record in trace.losses.records
                ],
            )
        }
    )
    assert validate_trace(relabelled) != []


def test_the_report_does_not_grow_with_the_trace():
    """Systemic losses are declared once, so declared absence stays affordable.

    Two spans of the same shape add per-span records only; the systemic set is
    identical. `ir_from_acp` measured the same property, and it is what keeps a
    50-tool-call trace from carrying 50 copies of one sentence.
    """
    one = convert(tool_span())
    many = convert(*[tool_span() for _ in range(20)])
    systemic = lambda trace: {  # noqa: E731 - local, and reads better inline
        record.field
        for record in trace.losses.records
        if record.field.startswith("events[]") or "[" not in record.field
    }
    assert systemic(one) == systemic(many)
    per_span = len(many.losses.records) - len(systemic(many))
    assert per_span == 20 * (len(one.losses.records) - len(systemic(one)))


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_this_edge_imports_nothing_but_the_hub():
    """The family rule, checked at the new member rather than only globally.

    ``tests/trajectories/test_trace_ir.py`` asserts that nothing outside the IR
    family imports it. This is the other half: the new edge does not reach into
    a run path either, so the proposal stays deletable.
    """
    source = Path(ir_from_otel_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    benchflow_imports = {name for name in imported if name.startswith("benchflow")}
    assert benchflow_imports == {
        "benchflow.trajectories.ir",
        "benchflow.trajectories._otlp_anyvalue",
    }, sorted(benchflow_imports)


def test_the_anyvalue_decoder_knows_nothing_about_the_hub():
    """The split is only worth making if it is a real boundary.

    `_otlp_anyvalue` turns protobuf JSON wrappers into Python values. If it ever
    imports the IR or the loss model, it has stopped being a decoder and the
    module has become a second place where mapping decisions live.
    """
    source = Path(anyvalue_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert [name for name in imported if name.startswith("benchflow")] == []
    assert [name for name in imported if name.startswith("opentelemetry")] == []


def test_the_edge_takes_no_opentelemetry_dependency():
    """The versions are pinned so the mapping is checkable, not so it can import.

    Adding ``opentelemetry`` as a runtime dependency would change ``uv.lock``,
    which nine CI jobs verify with ``uv sync --locked``. The whole point of
    reading dictionaries is that this slice costs the lock file nothing.
    """
    source = Path(ir_from_otel_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            assert not any(name.startswith("opentelemetry") for name in names), names


def test_no_input_dictionary_is_mutated():
    """The reader is a reader: the caller's payload comes back unchanged."""
    document = payload()
    before = json.dumps(document, sort_keys=True)
    otlp_json_to_ir(document)
    assert json.dumps(document, sort_keys=True) == before
