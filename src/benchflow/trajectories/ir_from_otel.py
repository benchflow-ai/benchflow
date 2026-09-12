"""OTLP/JSON spans → canonical Trace IR — the inbound OpenTelemetry edge.

> **PROVISIONAL.** Companion to :mod:`benchflow.trajectories.ir`, itself an
> unapproved proposal (`docs/trace-interop.md` §8). Nothing imports this module,
> nothing writes its output to disk, no run path changes because it exists, and
> **no `IR → OTel` emitter ships with it.** This is one direction of one edge.

`§4.2` is FACT: this repository has no OpenTelemetry code and no OTel runtime
dependency. What it *does* have is a lock file, and that is what this module is
built against rather than recollection.

## What is pinned, and what that buys

``uv.lock`` resolves ``opentelemetry-proto==1.41.1`` and
``opentelemetry-semantic-conventions==0.62b1`` as transitive dependencies of
``daytona`` (the ``sandbox-daytona`` extra). Neither is a dependency of this
module — nothing here imports ``opentelemetry`` and the lock is untouched — but
both are *artifacts this repository already names, at a version, with a hash*.
So the two questions an OTel reader has to answer have checkable answers:

- **the wire shape** — field names, JSON names and types — comes from the
  ``ExportTraceServiceRequest`` descriptor in ``opentelemetry-proto`` 1.41.1;
- **the attribute vocabulary** — every ``gen_ai.*`` name below — comes from
  ``opentelemetry.semconv._incubating.attributes.gen_ai_attributes`` in
  ``opentelemetry-semantic-conventions`` 0.62b1.

Both versions are recorded in :data:`OTLP_PROTO_VERSION` and
:data:`SEMCONV_VERSION` so a reader can re-derive every constant here. The
mapping is *not* written against the published specification text, which is not
vendored: where the specification says something the pinned artifacts do not,
this module does nothing and says so.

**The GenAI conventions are `_incubating` even at the pinned version** — the
package puts them under that namespace, which is its own statement that the
names are not stable. Several are already marked deprecated in 0.62b1
(``gen_ai.system``, ``gen_ai.usage.prompt_tokens``, ``gen_ai.prompt``,
``gen_ai.completion``). This module reads the deprecated spellings, because the
pinned package documents the replacement relationship itself, and declares each
such read :attr:`~benchflow.trajectories.ir.LossClass.NORMALIZED`.

## The rule this module is built on

**A span is evidence of an operation, not a statement about an agent.**

OTLP is a general tracing format. A payload of spans says what was instrumented,
when, and under what identifiers; it does not say who spoke, what a turn was, or
how a conversation was structured. So this edge maps exactly one span shape onto
a typed IR kind — a span whose ``gen_ai.operation.name`` is ``execute_tool``,
which the pinned vocabulary defines with that meaning — and every other span
becomes :attr:`~benchflow.trajectories.ir.EventKind.UNKNOWN` with its whole
content carried verbatim.

That is deliberately less than a plausible reading would allow. A ``chat`` span
*probably* corresponds to an agent turn; ``gen_ai.input.messages`` *probably*
holds the user text. Both are guesses about intent, both would put invented
agent semantics into the hub where every other format would inherit them, and
the second one additionally depends on a JSON schema the pinned package
references by relative path and does not ship. They are listed in
`docs/trace-interop.md` §8.11 as open maintainer decisions, not implemented.

## Order is order; it is not causality

The IR event list is ordered and dense (invariant 2), and this edge preserves
**document order** — the order the spans appear in the payload — and nothing
else. Spans are *not* sorted by start time.

Sorting would be the converter asserting a causal or logical sequence that OTLP
does not carry: sibling spans overlap, a partial batch may omit a parent, and
two spans with the same start instant have no defined order at all. The real
structure is the ``parentSpanId`` edge set, and it is preserved exactly, per
span, in ``extensions.otel.span`` — together with ``spanId``, ``traceId``,
``traceState``, ``flags`` and both timestamps. Nothing that expresses causality
is dropped, reordered, or turned into an ordering claim.

**The IR models no parent link**, so parentage lives in ``extensions`` rather
than in a canonical field. Adding one would be a change to the hub, which is a
maintainer's decision and not this slice's; §8.11 records it as such.

## Batches, and why the entry points are shaped as they are

An OTLP export request is a *batch*. It may carry spans from several traces, it
need not contain a trace's root span, and the same trace may arrive across
several requests. "One payload is one run" is therefore false, and an API that
implied it would be wrong at the boundary rather than merely imprecise.

So reading is split into the three things that actually happen:

- :func:`otlp_json_spans` walks the envelope and returns the spans it found,
  each with its resource and scope context, plus a report addressed to the
  *payload* — an envelope is not a trace, so its defects are not a trace's
  losses, and that report is returned rather than attached.
- :func:`group_spans_by_trace_id` groups by trace id, in first-appearance order.
- :func:`otel_spans_to_ir` converts one group into one
  :class:`~benchflow.trajectories.ir.CanonicalTrace` with its report attached,
  which is the inbound convention `ir.py` describes.

:func:`otlp_json_to_ir` composes the three for callers that want a payload in
and traces out.

## What is carried but not canonicalized

Everything OTel-specific with no stable IR meaning rides in
``extensions.otel``: the span verbatim (minus its attributes), the decoded
attribute map, the resource and scope with their schema URLs. That is the IR's
own contract for ``extensions`` — "source fields with no IR home, carried
verbatim so a conversion is not forced to choose between dropping a value and
growing the IR for it".

Attributes are decoded from OTLP's ``KeyValue``/``AnyValue`` wrappers into a
plain map because a map is what a consumer wants. Decoding can lose something —
duplicate keys, an empty ``AnyValue``, a value type this reader does not know —
and whenever it does, the original list is kept beside the map as
``attributes_raw`` and the collapse is declared. The map is a convenience; the
list is the record.

## Identifiers are never re-encoded

``traceId`` and ``spanId`` are ``bytes`` in the proto, and the protobuf canonical
JSON mapping encodes bytes as base64 — verified against the pinned package,
which round-trips a 16-byte id as a 24-character base64 string. Much of the OTel
ecosystem writes them as lowercase hex instead, and the two are **not reliably
distinguishable**: feeding a 32-character hex id to the pinned JSON parser is
accepted as base64 and silently yields 24 bytes.

This module therefore carries the id **exactly as written**, as a string, and
re-encodes nothing. Which encoding a BenchFlow trace id should canonically use
is a maintainer's decision (§8.11); guessing it here would make identity depend
on a heuristic.

## What OTLP/JSON cannot tell this reader

Protobuf scalar fields without presence — ``name``, ``startTimeUnixNano``,
``parentSpanId``, ``kind``, ``droppedAttributesCount`` — serialize to nothing
when they hold the default, so **absent and default are the same document**.
Verified against the pinned descriptors: ``has_presence`` is ``False`` for all
of them. The IR's tri-state rule (value / ``None`` / observed-empty) is real
here only for fields OTLP models with presence, and the limit is declared once
per conversion rather than left implicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from benchflow.trajectories._otlp_anyvalue import (
    Attributes,
    decode_attributes,
)
from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    LossReport,
    ModelInfo,
    PathSpace,
    Provenance,
    ToolCall,
    TraceEvent,
    TraceOutcome,
    TraceUsage,
)

LOSS_DIRECTION = "otel->ir"

OTEL_SOURCE = "otel"
"""Per-event provenance: the value the IR's :class:`Provenance` already names."""

OTLP_JSON_SOURCE = "otlp-json"
"""Trace-level provenance: the wire encoding actually read, since a trace's
spans may come from different instrumentation scopes."""

OTLP_PROTO_VERSION = "1.41.1"
"""``opentelemetry-proto`` version whose descriptors define the wire shape below.

Pinned in this repository's ``uv.lock`` (transitively, via ``daytona``). Not
imported: this module reads JSON dictionaries and takes no OTel dependency.
"""

SEMCONV_VERSION = "0.62b1"
"""``opentelemetry-semantic-conventions`` version every ``GEN_AI_*`` name below
was copied from, likewise pinned in ``uv.lock`` and likewise not imported.

The GenAI group is ``_incubating`` at this version — the package's own statement
that the names are unstable.
"""

USAGE_SOURCE = "otel_gen_ai_usage"
"""Value written to :attr:`~benchflow.trajectories.ir.TraceUsage.source`.

Counters read off ``gen_ai.usage.*`` are not the same measurement as BenchFlow's
``llm_proxy_normalized`` or ``acp_session_snapshot``; naming the definition is
how the IR keeps a consumer from comparing unlike numbers.
"""

TOOL_NAME_SEMANTICS = "gen_ai.tool.name"
"""What a tool call's :attr:`~benchflow.trajectories.ir.ToolCall.name` is when
it came from this edge — neither an ACP ``kind`` nor an ATIF ``function_name``.
"""

# --- semantic-convention attribute names, copied from the pinned package ------
# Every constant below is the literal value of the same-named symbol in
# ``opentelemetry.semconv._incubating.attributes.gen_ai_attributes`` at
# ``SEMCONV_VERSION``. The comments record what that package says about them.

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_VERSION = "gen_ai.agent.version"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_SYSTEM = "gen_ai.system"  # deprecated: replaced by gen_ai.provider.name
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"  # deprecated → input
GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"  # deprecated → output

GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_PROMPT = "gen_ai.prompt"  # deprecated: removed, no replacement
GEN_AI_COMPLETION = "gen_ai.completion"  # deprecated: removed, no replacement

OPERATION_EXECUTE_TOOL = "execute_tool"
"""``GenAiOperationNameValues.EXECUTE_TOOL`` — the one operation this edge maps
onto a typed IR kind, because the pinned vocabulary defines it as exactly that.
"""

_CONTENT_ATTRIBUTES: tuple[str, ...] = (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROMPT,
    GEN_AI_COMPLETION,
)
"""Attributes that carry conversation text. None is mapped onto ``text``; see
the module docstring and §8.11."""

# ``final_metrics``-style pairing, but for spans: the IR usage field each pinned
# attribute feeds, and whether reading it is a deprecated spelling.
_USAGE_ATTRIBUTES: tuple[tuple[str, str, str | None], ...] = (
    (GEN_AI_USAGE_INPUT_TOKENS, "input_tokens", None),
    (GEN_AI_USAGE_OUTPUT_TOKENS, "output_tokens", None),
    (GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, "cache_read_tokens", None),
    (GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, "cache_creation_tokens", None),
    (GEN_AI_USAGE_PROMPT_TOKENS, "input_tokens", GEN_AI_USAGE_INPUT_TOKENS),
    (GEN_AI_USAGE_COMPLETION_TOKENS, "output_tokens", GEN_AI_USAGE_OUTPUT_TOKENS),
)

_USAGE_UNSUPPORTED: tuple[str, ...] = (
    "total_tokens",
    "reasoning_tokens",
    "cost_usd",
    "price_source",
)
"""IR usage fields with no attribute in the pinned GenAI vocabulary.

``gen_ai.usage.total_tokens`` is worth naming explicitly: the deleted
``OTelCollector`` (§4.2) read it, and it does not exist in ``SEMCONV_VERSION``.
"""

_USAGE_PREFERRED: tuple[tuple[str, str], ...] = (
    ("input_tokens", GEN_AI_USAGE_INPUT_TOKENS),
    ("output_tokens", GEN_AI_USAGE_OUTPUT_TOKENS),
    ("cache_read_tokens", GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS),
    ("cache_creation_tokens", GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS),
)
"""IR usage fields this edge *can* fill, and the attribute each one reads.

Distinct from :data:`_USAGE_UNSUPPORTED`, and the difference is the point: a
field here that stays empty is a value this payload did not carry, while one
there is a value no OTLP payload can carry. Both are declared; only the second
is a property of the format.
"""

# ``AnyValue`` oneof members, by their protobuf JSON names.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_SPAN_ATTRIBUTES_KEY = "attributes"


@dataclass(frozen=True)
class OtlpSpan:
    """One span with the envelope context it was found under.

    A plain frozen dataclass rather than a model: nothing here validates,
    coerces or copies its input. The dictionaries are the caller's, verbatim,
    which is what lets :func:`otel_spans_to_ir` promise it carried them
    unchanged.

    ``resource`` and ``scope`` are kept per span, not per conversion, because
    one payload legitimately mixes them and a trace-level answer would be wrong
    for some of its own spans — the same reason the IR carries
    :class:`~benchflow.trajectories.ir.Provenance` per event.

    The three positions record **where in the envelope the span was found**.
    Flattening ``resourceSpans[] → scopeSpans[] → spans[]`` into one list
    otherwise destroys the partition: two spans under *different* ``ScopeSpans``
    objects that happen to carry an equal ``scope`` become indistinguishable,
    and the payload said they were separately batched. Carrying the coordinates
    keeps that recoverable without the IR growing a concept of an envelope.

    They are ``None`` when the caller built the span itself rather than reading
    it out of a payload — then there is no partition to preserve, and inventing
    coordinates would claim an envelope that never existed.
    """

    span: dict[str, Any]
    resource: dict[str, Any] | None = None
    resource_schema_url: str | None = None
    scope: dict[str, Any] | None = None
    scope_schema_url: str | None = None
    resource_index: int | None = None
    scope_index: int | None = None
    span_index: int | None = None


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def otlp_json_spans(payload: dict[str, Any]) -> tuple[list[OtlpSpan], LossReport]:
    """Flatten one OTLP/JSON export request into spans plus envelope context.

    The structure walked is ``resourceSpans[] → scopeSpans[] → spans[]``, the
    JSON names of ``ExportTraceServiceRequest`` at :data:`OTLP_PROTO_VERSION`.
    Nothing is interpreted here: a span is returned as the dictionary it was.

    The returned :class:`~benchflow.trajectories.ir.LossReport` is addressed to
    the **payload**, in :attr:`~benchflow.trajectories.ir.PathSpace.SOURCE`, and
    is *returned* rather than attached to anything. A malformed
    ``resourceSpans`` entry is not any trace's loss — the spans it would have
    contained are unidentifiable, so there is no trace to attach it to. This is
    the one place an inbound edge in this family hands a report back, and the
    reason is that an envelope is not a trace.

    Raises ``TypeError`` for a non-mapping payload: there is no partial reading
    of something that is not an export request, and returning an empty list
    would claim a payload was read.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"OTLP payload must be a mapping, got {type(payload).__name__}")

    losses = LossReport(direction=LOSS_DIRECTION)
    spans: list[OtlpSpan] = []

    for resource_position, resource_spans in enumerate(
        _entries(payload, "resourceSpans", "resourceSpans", losses)
    ):
        where_resource = f"resourceSpans[{resource_position}]"
        if not isinstance(resource_spans, dict):
            _not_an_object(losses, where_resource, resource_spans)
            continue
        resource = resource_spans.get("resource")
        resource = resource if isinstance(resource, dict) else None
        resource_schema_url = _schema_url(resource_spans)

        for scope_position, scope_spans in enumerate(
            _entries(
                resource_spans,
                "scopeSpans",
                f"{where_resource}.scopeSpans",
                losses,
            )
        ):
            where_scope = f"{where_resource}.scopeSpans[{scope_position}]"
            if not isinstance(scope_spans, dict):
                _not_an_object(losses, where_scope, scope_spans)
                continue
            scope = scope_spans.get("scope")
            scope = scope if isinstance(scope, dict) else None

            for span_position, span in enumerate(
                _entries(scope_spans, "spans", f"{where_scope}.spans", losses)
            ):
                if not isinstance(span, dict):
                    _not_an_object(
                        losses, f"{where_scope}.spans[{span_position}]", span
                    )
                    continue
                spans.append(
                    OtlpSpan(
                        span=span,
                        resource=resource,
                        resource_schema_url=resource_schema_url,
                        scope=scope,
                        scope_schema_url=_schema_url(scope_spans),
                        resource_index=resource_position,
                        scope_index=scope_position,
                        span_index=span_position,
                    )
                )

    return spans, losses


def _entries(
    holder: dict[str, Any], key: str, where: str, losses: LossReport
) -> list[Any]:
    """The list at *key*, declaring the case where it is present but not one."""
    raw = holder.get(key)
    if isinstance(raw, list):
        return raw
    if raw is not None:
        losses.add(
            where,
            LossClass.DROPPED,
            f"{key} is {type(raw).__name__}, not a list; nothing could be read from it",
            space=PathSpace.SOURCE,
        )
    return []


def _not_an_object(losses: LossReport, where: str, value: Any) -> None:
    losses.add(
        where,
        LossClass.DROPPED,
        f"entry is {type(value).__name__}, not a JSON object; it carries no span "
        "this reader can address",
        space=PathSpace.SOURCE,
    )


def _schema_url(holder: dict[str, Any]) -> str | None:
    value = holder.get("schemaUrl")
    return value if isinstance(value, str) else None


def group_spans_by_trace_id(
    spans: list[OtlpSpan],
) -> list[tuple[str | None, list[OtlpSpan]]]:
    """Group *spans* by trace id, preserving first-appearance order.

    The key is the ``traceId`` string **exactly as written** — see the module
    docstring on identifier encodings — or ``None`` for a span that carries no
    string trace id. Spans with no id are grouped together rather than merged
    into an identified trace: they may belong to it, and "may" is not a fact
    this reader gets to record.

    Order is preserved twice over: groups appear in the order their first span
    appeared, and each group's spans keep their relative document order.
    """
    groups: dict[str | None, list[OtlpSpan]] = {}
    for entry in spans:
        raw = entry.span.get("traceId")
        key = raw if isinstance(raw, str) else None
        groups.setdefault(key, []).append(entry)
    return list(groups.items())


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def _timestamp(span: dict[str, Any], key: str) -> tuple[datetime | None, str | None]:
    """Read a ``fixed64`` nanosecond field as a datetime, plus what it cost.

    Returns ``(value, note)``; *note* is ``None`` when nothing needs declaring.

    Two losses are real here and neither is avoidable:

    - **Absent is default.** ``startTimeUnixNano`` has no presence in the proto
      (verified against the pinned descriptors), so a producer that never set it
      and one that set it to 0 write the same document. Both read as ``None``.
    - **Nanoseconds do not fit.** ``datetime`` resolves to microseconds, so a
      timestamp that is not a whole number of microseconds is truncated. The
      exact integer stays in ``extensions.otel.span``, so the value is not lost
      — but the IR field no longer holds it, and that is a normalization.
    """
    if key not in span:
        return None, (
            f"{key} is absent; OTLP models it as a fixed64 with no presence, so "
            "an unset field and a 0 are the same document and no instant can be "
            "read from either"
        )
    raw = span[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return None, (
            f"{key} is {type(raw).__name__}; OTLP encodes fixed64 as a JSON "
            "string or number"
        )
    try:
        nanos = int(raw)
    except ValueError:
        return None, f"{key} is {raw!r}, which is not an integer nanosecond count"
    if nanos < 0:
        return None, (
            f"{key} is negative and fixed64 is unsigned, so no conformant "
            "producer wrote it"
        )
    if nanos == 0:
        return None, (
            f"{key} is 0, the protobuf default for a field with no presence; the "
            "document cannot say whether the producer set it"
        )
    micros, remainder = divmod(nanos, 1000)
    value = _EPOCH + timedelta(microseconds=micros)
    if remainder:
        return value, (
            f"{key} is {nanos} ns and datetime resolves to microseconds, so "
            f"{remainder} ns is truncated; the exact integer is kept in "
            "extensions.otel.span"
        )
    return value, None


# ---------------------------------------------------------------------------
# One trace
# ---------------------------------------------------------------------------


def otel_spans_to_ir(
    spans: list[OtlpSpan],
    *,
    session_id: str | None = None,
) -> CanonicalTrace:
    """Convert one group of OTLP spans into a :class:`CanonicalTrace`.

    *spans* is normally one trace's worth — see :func:`group_spans_by_trace_id`
    — but nothing requires it: a list mixing trace ids converts, and the
    disagreement is declared rather than silently resolved.

    *session_id* is context the caller has from elsewhere, exactly as in
    :func:`~benchflow.trajectories.ir_from_acp.acp_events_to_ir`. It is not read
    from the spans: ``gen_ai.conversation.id`` is a candidate and is preserved
    in the attribute map, but whether a conversation id *is* a BenchFlow session
    id is a maintainer's decision (§8.11), and answering it here would put the
    answer in the hub.

    One span becomes exactly one event, in document order, so indices stay dense
    (invariant 2) and no span is skipped. The returned trace carries its own
    report on :attr:`~benchflow.trajectories.ir.CanonicalTrace.losses` and
    satisfies :func:`~benchflow.trajectories.ir.validate_trace` for any input,
    including payloads no conformant producer would write.
    """
    losses = LossReport(direction=LOSS_DIRECTION)
    events: list[TraceEvent] = []
    agent = _AgentAccumulator()

    for entry in spans:
        events.append(_span_to_event(entry, len(events), agent, losses))

    trace_id = _trace_id(spans, losses)
    _declare_trace_losses(
        losses,
        events=events,
        session_id=session_id,
        has_tool_call=any(event.kind is EventKind.TOOL_CALL for event in events),
    )
    agent.declare(losses)

    extensions: dict[str, Any] = {}
    other_ids = _distinct_trace_ids(spans)
    if len(other_ids) > 1:
        extensions["otel"] = {"trace_ids": other_ids}

    return CanonicalTrace(
        trace_id=trace_id,
        session_id=session_id,
        agent=agent.model_info(),
        events=events,
        # Never derived. Every span's own extent is preserved on its event and
        # in extensions; computing a run's extent from them would be this
        # converter asserting that the payload holds the whole run, which a
        # batch does not promise. Declared below.
        started_at=None,
        finished_at=None,
        usage=None,
        # Always present, never populated: OTLP has no run-outcome concept, and
        # per-span status is preserved per event. A record addresses this
        # section by path, so the section has to exist.
        outcome=TraceOutcome(),
        provenance=Provenance(source_format=OTLP_JSON_SOURCE),
        extensions=extensions,
        losses=losses,
    )


def otlp_json_to_ir(
    payload: dict[str, Any],
) -> tuple[list[CanonicalTrace], LossReport]:
    """Read one OTLP/JSON export request into one trace per trace id.

    Returns the traces — each with its own report attached — and the
    **envelope** report from :func:`otlp_json_spans`, which belongs to the
    payload rather than to any trace.

    A payload carrying no readable span yields an empty list. That is a claim,
    not a failure: the envelope report says what was in the way.
    """
    spans, envelope = otlp_json_spans(payload)
    traces = [otel_spans_to_ir(group) for _, group in group_spans_by_trace_id(spans)]
    return traces, envelope


class _AgentAccumulator:
    """Trace-level agent identity, gathered from span attributes.

    OTel puts identity on spans; the IR puts it on the trace. The first
    observed value for each field wins, in document order, and a *different*
    later value is kept and declared rather than overwritten silently — a
    payload whose spans disagree about the model is saying something, and
    picking one quietly would erase it.
    """

    _FIELDS = ("agent_name", "agent_version", "model", "provider")

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.conflicts: dict[str, list[str]] = {}
        self.deprecated: set[str] = set()

    def offer(self, field: str, value: str | None, *, deprecated: bool = False) -> None:
        if value is None:
            return
        if field not in self.values:
            self.values[field] = value
            if deprecated:
                self.deprecated.add(field)
            return
        if self.values[field] != value:
            self.conflicts.setdefault(field, []).append(value)
            return
        if not deprecated:
            # A later span agreed, using the current spelling. The value did not
            # come only from the deprecated one after all, so the normalization
            # record would be false.
            self.deprecated.discard(field)

    def model_info(self) -> ModelInfo:
        return ModelInfo(**self.values)

    def declare(self, losses: LossReport) -> None:
        for field in self._FIELDS:
            if field in self.values:
                continue
            losses.add(
                f"agent.{field}",
                LossClass.UNSUPPORTED,
                "no span in this trace carries the attribute this field reads "
                f"(semantic conventions {SEMCONV_VERSION})",
            )
        if "provider" in self.deprecated:
            losses.add(
                "agent.provider",
                LossClass.NORMALIZED,
                f"read from the deprecated {GEN_AI_SYSTEM!r}; the pinned package "
                f"marks it replaced by {GEN_AI_PROVIDER_NAME!r}",
            )
        for field, extra in sorted(self.conflicts.items()):
            losses.add(
                f"agent.{field}",
                LossClass.DROPPED,
                f"spans disagree: the first value is kept and {extra!r} "
                "is not representable in a trace-level agent block",
            )


def _trace_id(spans: list[OtlpSpan], losses: LossReport) -> str | None:
    """The trace id, verbatim, or ``None`` with the reason declared."""
    ids = _distinct_trace_ids(spans)
    non_string = [
        position
        for position, entry in enumerate(spans)
        if "traceId" in entry.span and not isinstance(entry.span["traceId"], str)
    ]
    for position in non_string:
        losses.add(
            f"spans[{position}].traceId",
            LossClass.DROPPED,
            "traceId is not a string; OTLP/JSON encodes the id as text and this "
            "reader does not re-encode, so it cannot be read",
            space=PathSpace.SOURCE,
        )
    if not ids:
        losses.add(
            "trace_id",
            LossClass.UNSUPPORTED,
            "no span in this group carries a string traceId",
        )
        return None
    if len(ids) > 1:
        losses.add(
            "trace_id",
            LossClass.NORMALIZED,
            f"the group spans {len(ids)} trace ids; the first is kept and all of "
            "them are listed in extensions.otel.trace_ids",
        )
    return ids[0]


def _distinct_trace_ids(spans: list[OtlpSpan]) -> list[str]:
    seen: list[str] = []
    for entry in spans:
        raw = entry.span.get("traceId")
        if isinstance(raw, str) and raw not in seen:
            seen.append(raw)
    return seen


def _span_to_event(
    entry: OtlpSpan,
    index: int,
    agent: _AgentAccumulator,
    losses: LossReport,
) -> TraceEvent:
    """One span as one IR event."""
    span = entry.span
    where = f"events[{index}]"
    attributes = decode_attributes(span.get(_SPAN_ATTRIBUTES_KEY))

    agent.offer("agent_name", attributes.text(GEN_AI_AGENT_NAME))
    agent.offer("agent_version", attributes.text(GEN_AI_AGENT_VERSION))
    agent.offer("model", attributes.text(GEN_AI_REQUEST_MODEL))
    provider = attributes.text(GEN_AI_PROVIDER_NAME)
    if provider is not None:
        agent.offer("provider", provider)
    else:
        agent.offer("provider", attributes.text(GEN_AI_SYSTEM), deprecated=True)

    started_at, started_note = _timestamp(span, "startTimeUnixNano")
    finished_at, finished_note = _timestamp(span, "endTimeUnixNano")
    for note, field in ((started_note, "started_at"), (finished_note, "finished_at")):
        if note is not None:
            losses.add(f"{where}.{field}", LossClass.NORMALIZED, note)

    if not attributes.faithful:
        losses.add(
            f"{where}.extensions",
            LossClass.NORMALIZED,
            "the attribute list does not fit a map without loss ("
            + "; ".join(attributes.notes)
            + "); the original list is kept as extensions.otel.attributes_raw",
        )
    _declare_dropped_counts(span, where, losses)

    status = span.get("status")
    if _has_status(status):
        losses.add(
            f"{where}.outcome",
            LossClass.NORMALIZED,
            "the span carries a status; the IR outcome slot is free text with no "
            "vocabulary an OTLP status code maps into, so the status object is "
            "kept verbatim in extensions.otel.span.status",
        )

    present_content = [key for key in _CONTENT_ATTRIBUTES if key in attributes.values]
    if present_content:
        losses.add(
            f"{where}.text",
            LossClass.NORMALIZED,
            "conversation content is carried in extensions.otel.attributes ("
            + ", ".join(present_content)
            + ") rather than in text; the pinned package defines the message "
            "structure by reference to a JSON schema it does not ship",
        )

    usage = _read_usage(attributes, where, losses)
    tool_call = None
    kind = EventKind.UNKNOWN
    if attributes.text(GEN_AI_OPERATION_NAME) == OPERATION_EXECUTE_TOOL:
        kind = EventKind.TOOL_CALL
        tool_call = _read_tool_call(
            attributes,
            where=where,
            started_at=started_at,
            finished_at=finished_at,
            timestamp_notes={
                "started_at": started_note,
                "finished_at": finished_note,
            },
            has_status=_has_status(status),
            losses=losses,
        )

    name = span.get("name")
    return TraceEvent(
        index=index,
        kind=kind,
        # The span name, verbatim. It is the only thing a non-GenAI span says
        # about what it is, and ``gen_ai.operation.name`` — the typed answer
        # when there is one — stays readable in the attribute map.
        source_type=name if isinstance(name, str) else None,
        # Never set. OTLP attributes nothing to a speaker, and the IR's Role
        # members are the ones some source in this repository distinguishes.
        role=None,
        text=None,
        reasoning=None,
        tool_call=tool_call,
        started_at=started_at,
        finished_at=finished_at,
        outcome=None,
        usage=usage,
        provenance=Provenance(
            source_format=OTEL_SOURCE,
            producer=_scope_name(entry),
        ),
        extensions={"otel": _otel_extensions(entry, attributes)},
    )


def _scope_name(entry: OtlpSpan) -> str | None:
    """The instrumentation scope's name — the closest OTLP has to an emitter."""
    if entry.scope is None:
        return None
    name = entry.scope.get("name")
    return name if isinstance(name, str) else None


def _has_status(status: Any) -> bool:
    """True when the span carries a status other than the unset default.

    ``STATUS_CODE_UNSET`` is 0, the protobuf default, so a status object holding
    only that says nothing a conformant producer had to write. Both the enum
    name and the integer are accepted: the pinned JSON mapping emits names by
    default and integers under ``use_integers_for_enums``, and parses either.
    """
    if not isinstance(status, dict):
        return False
    code = status.get("code")
    if code in (None, 0, "STATUS_CODE_UNSET"):
        return bool(status.get("message"))
    return True


def _declare_dropped_counts(
    span: dict[str, Any], where: str, losses: LossReport
) -> None:
    """Declare what the *producer* dropped before this reader ever saw it.

    OTLP's ``dropped*Count`` fields are the SDK stating that it discarded data
    to stay inside a limit. That is the cleanest possible
    :attr:`~benchflow.trajectories.ir.LossClass.UNSUPPORTED`: the value never
    reached the document, so no converter can recover it and the fix is a
    producer-side limit, not code here.
    """
    for key, what in (
        ("droppedAttributesCount", "attributes"),
        ("droppedEventsCount", "span events"),
        ("droppedLinksCount", "links"),
    ):
        count = span.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            continue
        losses.add(
            f"{where}.extensions",
            LossClass.UNSUPPORTED,
            f"the producer reports {key}={count}: that many {what} were "
            "discarded by the emitting SDK and are not in the document",
        )


def _otel_extensions(entry: OtlpSpan, attributes: Attributes) -> dict[str, Any]:
    """Everything OTel-specific, carried verbatim under one key.

    The span goes in whole, minus its attribute list, so a field this reader has
    never heard of — a future ``Span`` member — rides along instead of being
    truncated to what the mapping happens to know. The decoded attribute map
    sits beside it, and the original list joins them whenever decoding was not
    faithful.

    ``envelope`` holds the span's coordinates in the payload it was read from.
    That is what keeps the ``resourceSpans``/``scopeSpans`` partition from being
    silently flattened: two spans whose ``resource`` and ``scope`` objects are
    equal but which the producer batched separately differ here and nowhere
    else. Absent when the caller did not read the span out of a payload.
    """
    carried: dict[str, Any] = {
        "span": {
            key: value
            for key, value in entry.span.items()
            if key != _SPAN_ATTRIBUTES_KEY
        },
        "attributes": attributes.values,
    }
    if not attributes.faithful:
        carried["attributes_raw"] = entry.span.get(_SPAN_ATTRIBUTES_KEY)
    envelope = {
        name: index
        for name, index in (
            ("resource_spans_index", entry.resource_index),
            ("scope_spans_index", entry.scope_index),
            ("span_index", entry.span_index),
        )
        if index is not None
    }
    if envelope:
        carried["envelope"] = envelope
    if entry.resource is not None:
        carried["resource"] = entry.resource
    if entry.resource_schema_url is not None:
        carried["resource_schema_url"] = entry.resource_schema_url
    if entry.scope is not None:
        carried["scope"] = entry.scope
    if entry.scope_schema_url is not None:
        carried["scope_schema_url"] = entry.scope_schema_url
    return carried


def _read_usage(
    attributes: Attributes, where: str, losses: LossReport
) -> TraceUsage | None:
    """Per-span token counters from the pinned ``gen_ai.usage.*`` attributes.

    Reading a deprecated spelling is declared: the pinned package states the
    replacement itself, so following it is supported rather than assumed, but a
    consumer should still know which name the number came from. A non-integer
    value is declared and dropped rather than coerced — a token count that is
    not a count is a malformed document, not a number to salvage.
    """
    values: dict[str, Any] = {}
    for attribute, ir_field, replaced_by in _USAGE_ATTRIBUTES:
        if attribute not in attributes.values:
            continue
        raw = attributes.values[attribute]
        if isinstance(raw, bool) or not isinstance(raw, int):
            losses.add(
                f"{where}.usage.{ir_field}",
                LossClass.DROPPED,
                f"{attribute} is {type(raw).__name__}, not an integer token count",
            )
            continue
        if ir_field in values:
            # Both the current and the deprecated spelling are present. The
            # current one is read first by the order of _USAGE_ATTRIBUTES.
            if values[ir_field] != raw:
                losses.add(
                    f"{where}.usage.{ir_field}",
                    LossClass.DROPPED,
                    f"{attribute} disagrees with the preferred attribute already "
                    f"read for this field ({raw} vs {values[ir_field]}); the "
                    "preferred one is kept",
                )
            continue
        values[ir_field] = raw
        if replaced_by is not None:
            losses.add(
                f"{where}.usage.{ir_field}",
                LossClass.NORMALIZED,
                f"read from the deprecated {attribute!r}; the pinned package "
                f"marks it replaced by {replaced_by!r}",
            )
    if not values:
        return None
    return TraceUsage(source=USAGE_SOURCE, **values)


def _read_tool_call(
    attributes: Attributes,
    *,
    where: str,
    started_at: datetime | None,
    finished_at: datetime | None,
    timestamp_notes: dict[str, str | None],
    has_status: bool,
    losses: LossReport,
) -> ToolCall:
    """The tool call of an ``execute_tool`` span.

    The span's own extent is the tool call's: an ``execute_tool`` span *is* the
    execution, so its start and end are not an inference. They are the only
    timestamps in this family that an inbound edge has ever been able to fill —
    both ACP and ATIF declare them unsupported outright.

    *Being* fillable is not the same as being filled, and the difference is
    exactly where a silent absence hides. ``timestamp_notes`` carries whatever
    :func:`_timestamp` had to say about each instant, and the same note is
    declared a second time against the tool call's own path: both fields hold
    that one value, so a reader checking ``events[i].tool_call.started_at``
    must find the declaration there rather than one level up. The two ACP and
    ATIF edges declare these paths unconditionally; this one declares them
    whenever it could not fill them, which is the same contract for a source
    that sometimes can.

    Nothing is synthesized. No id is invented when ``gen_ai.tool.call.id`` is
    absent, no name is taken from the span name (the recommended span name is
    ``execute_tool {name}``, so reading it as the tool name would be reading a
    convention as a value), and no status is derived from the span's.
    """
    for field, note in timestamp_notes.items():
        if note is None:
            continue
        losses.add(
            f"{where}.tool_call.{field}",
            LossClass.NORMALIZED,
            f"{note}; the tool call reads the same instant as its span, so the "
            "same thing happened to this field",
        )

    call_id = attributes.text(GEN_AI_TOOL_CALL_ID)
    if call_id is None:
        losses.add(
            f"{where}.tool_call.call_id",
            LossClass.UNSUPPORTED,
            f"the span carries no string {GEN_AI_TOOL_CALL_ID!r}; an id is not "
            "invented here, because a synthesized id is indistinguishable from "
            "an observed one once written",
        )

    name = attributes.text(GEN_AI_TOOL_NAME)
    if name is None:
        losses.add(
            f"{where}.tool_call.name",
            LossClass.UNSUPPORTED,
            f"the span carries no string {GEN_AI_TOOL_NAME!r}; the span name is "
            "not read as a substitute, since the recommended span name is "
            f"{OPERATION_EXECUTE_TOOL!r} followed by the tool name",
        )
        losses.add(
            f"{where}.tool_call.name_semantics",
            LossClass.UNSUPPORTED,
            "there is no name, so there is nothing to say about what kind of "
            "name it is; the field is not filled with the attribute this edge "
            "would have read",
        )

    arguments = _read_arguments(attributes, where, losses)
    content = _read_result(attributes, where, losses)

    losses.add(
        f"{where}.tool_call.status",
        LossClass.NORMALIZED if has_status else LossClass.UNSUPPORTED,
        (
            "the span's status is kept in extensions.otel.span.status; whether an "
            "OTLP status code maps onto ToolStatus is a semantic question this "
            "edge does not answer"
        )
        if has_status
        else (
            "the span carries no status, and the pinned vocabulary has no tool "
            "lifecycle attribute"
        ),
    )

    return ToolCall(
        call_id=call_id,
        name=name,
        name_semantics=TOOL_NAME_SEMANTICS if name is not None else None,
        # No pinned attribute is a human-readable label for the call. The ACP
        # edge fills this from a title the capture path records; OTel has none.
        title=None,
        status=None,
        arguments=arguments,
        content=content,
        started_at=started_at,
        finished_at=finished_at,
    )


def _read_arguments(
    attributes: Attributes, where: str, losses: LossReport
) -> dict[str, Any] | None:
    """``gen_ai.tool.call.arguments``, and every way it can fail to be a map.

    The pinned package says the attribute "is expected to be an object" and MAY
    be recorded as a JSON string where structured attributes are unsupported. A
    ``kvlistValue`` therefore decodes straight into
    :attr:`~benchflow.trajectories.ir.ToolCall.arguments`; **a JSON string is
    not parsed.** Parsing would be this converter deciding that a string that
    happens to be JSON was meant as structure, and the pinned text puts that
    obligation on the instrumentation, not on the reader. The string stays in
    the attribute map and the case is declared (§8.11).

    Every branch declares something, because invariant 7 requires an
    ``arguments is None`` to be paired with a hub record at exactly this path —
    the one invariant that makes the loss report a contract.
    """
    field = f"{where}.tool_call.arguments"
    if GEN_AI_TOOL_CALL_ARGUMENTS not in attributes.values:
        losses.add(
            field,
            LossClass.UNSUPPORTED,
            f"the span carries no {GEN_AI_TOOL_CALL_ARGUMENTS!r}",
        )
        return None
    raw = attributes.values[GEN_AI_TOOL_CALL_ARGUMENTS]
    if isinstance(raw, dict):
        # Including ``{}``: an empty kvlist is an observed empty argument map,
        # which the IR distinguishes from "the source carried no arguments".
        return raw
    losses.add(
        field,
        LossClass.NORMALIZED,
        f"{GEN_AI_TOOL_CALL_ARGUMENTS} is "
        f"{'a serialized string' if isinstance(raw, str) else type(raw).__name__}"
        ", not an object; it is kept verbatim in extensions.otel.attributes and "
        "is not parsed into a mapping here",
    )
    return None


def _read_result(
    attributes: Attributes, where: str, losses: LossReport
) -> list[ContentBlock]:
    """``gen_ai.tool.call.result`` as content blocks.

    A string result is the rendered output and becomes a ``TEXT`` block. An
    object result becomes an ``OPAQUE`` block carrying it verbatim, which is
    exactly what that kind exists for. Anything else — a bare number, a list —
    has no block shape it fits, and wrapping it would invent one, so it stays in
    the attribute map with a record.

    Both block kinds leave one of :class:`ContentBlock`'s two payload fields
    empty, and each is declared at its own concrete path. A ``TEXT`` block has
    no ``raw`` because the attribute value *is* the text — there was never a
    separate source block — and an ``OPAQUE`` block has no ``text`` because the
    document holds an object and rendering it would be this converter writing
    the string. Neither absence is structural, so neither is left to inference.
    """
    if GEN_AI_TOOL_CALL_RESULT not in attributes.values:
        losses.add(
            f"{where}.tool_call.content",
            LossClass.UNSUPPORTED,
            f"the span carries no {GEN_AI_TOOL_CALL_RESULT!r}, so there is no "
            "captured output to represent as content blocks",
        )
        return []
    raw = attributes.values[GEN_AI_TOOL_CALL_RESULT]
    if isinstance(raw, str):
        losses.add(
            f"{where}.tool_call.content[0].raw",
            LossClass.UNSUPPORTED,
            f"{GEN_AI_TOOL_CALL_RESULT} is the rendered text itself; OTLP has no "
            "separate source block behind it to carry",
        )
        return [ContentBlock(kind=ContentBlockKind.TEXT, text=raw)]
    if isinstance(raw, dict):
        losses.add(
            f"{where}.tool_call.content[0].text",
            LossClass.UNSUPPORTED,
            f"{GEN_AI_TOOL_CALL_RESULT} is an object and the document carries no "
            "rendering of it; producing one here would be this converter writing "
            "the text rather than reading it",
        )
        return [ContentBlock(kind=ContentBlockKind.OPAQUE, raw=raw)]
    losses.add(
        f"{where}.tool_call.content",
        LossClass.NORMALIZED,
        f"{GEN_AI_TOOL_CALL_RESULT} is {type(raw).__name__}; a content block is "
        "text or an object, and the value is kept in extensions.otel.attributes "
        "rather than wrapped in a shape the source did not have",
    )
    return []


def _declare_trace_losses(
    losses: LossReport,
    *,
    events: list[TraceEvent],
    session_id: str | None,
    has_tool_call: bool,
) -> None:
    """The records that describe the conversion rather than one span.

    Two kinds are mixed here, and the classes keep them apart:

    - **``UNSUPPORTED``** — OTLP carries nothing for the field. A run outcome, a
      session id, a speaker: no attribute in the pinned vocabulary expresses
      them, so no converter could fill them.
    - **``NORMALIZED``** — OTLP carries the information, *per span*, and the IR
      wants it per trace. A run's temporal extent and its total token usage are
      both of that kind. Deriving them would mean assuming this payload holds
      the whole run, which a batch does not promise; the per-span values are
      preserved, and the aggregate is simply not computed.

    The difference is the one that says where a fix would land — the same line
    `ir_from_acp` and `ir_from_atif` draw.
    """
    if session_id is None:
        losses.add(
            "session_id",
            LossClass.UNSUPPORTED,
            f"no span attribute is a BenchFlow session id; {GEN_AI_CONVERSATION_ID!r} "
            "is the candidate and is preserved in the attribute map, but reading "
            "it as one would settle a mapping this edge does not own",
        )

    has_span_times = any(
        event.started_at is not None or event.finished_at is not None
        for event in events
    )
    for field in ("started_at", "finished_at"):
        if has_span_times:
            losses.add(
                field,
                LossClass.NORMALIZED,
                "OTLP has no run-level timestamp; each span's own extent is "
                "preserved on its event, and deriving a run extent from them "
                "would assume this payload holds the whole trace",
            )
        else:
            losses.add(
                field,
                LossClass.UNSUPPORTED,
                "no span in this group carries a readable timestamp",
            )

    losses.add(
        "outcome",
        LossClass.UNSUPPORTED,
        "OTLP has no run-outcome section; a span status is per span and is "
        "preserved on each event, and reward and error category have no "
        "attribute in the pinned vocabulary",
    )

    with_usage = [event for event in events if event.usage is not None]
    if with_usage:
        losses.add(
            "usage",
            LossClass.NORMALIZED,
            "token counters are per span and are preserved on their events; "
            "summing them would be an aggregation this converter does not "
            "perform, and a batch need not hold every span of the run",
        )
        if len(with_usage) != len(events):
            # Declared once for the whole conversion rather than per bare span:
            # the fact is "not every event has usage", and repeating it per
            # event would multiply one sentence by the trace length.
            losses.add(
                "events[].usage",
                LossClass.UNSUPPORTED,
                f"{len(events) - len(with_usage)} of {len(events)} spans carry no "
                f"{GEN_AI_USAGE_INPUT_TOKENS!r}-family attribute, so their events "
                "carry no usage at all",
            )
        for field in _USAGE_UNSUPPORTED:
            losses.add(
                f"events[].usage.{field}",
                LossClass.UNSUPPORTED,
                f"no attribute in the pinned GenAI vocabulary ({SEMCONV_VERSION}) "
                "carries it",
            )
        for field, attribute in _USAGE_PREFERRED:
            # Declared unless *every* usage-bearing event filled it. "Some spans
            # have it" leaves the others empty, and that absence needs the same
            # declaration as "no span has it" — with a detail that says which.
            filled = [
                event for event in with_usage if getattr(event.usage, field) is not None
            ]
            if len(filled) == len(with_usage):
                continue
            losses.add(
                f"events[].usage.{field}",
                LossClass.UNSUPPORTED,
                f"{len(with_usage) - len(filled)} of {len(with_usage)} spans with "
                f"usage carry no {attribute!r}; the attribute is readable, this "
                "payload simply does not have it there",
            )
    else:
        losses.add(
            "usage",
            LossClass.UNSUPPORTED,
            f"no span carries a {GEN_AI_USAGE_INPUT_TOKENS!r}-family attribute",
        )
        losses.add(
            "events[].usage",
            LossClass.UNSUPPORTED,
            "no span carries token counters",
        )

    losses.add(
        "events[].role",
        LossClass.UNSUPPORTED,
        "OTLP attributes a span to an instrumentation scope, not to a speaker; "
        "the IR's roles are the ones a source in this repository distinguishes",
    )
    losses.add(
        "events[].text",
        LossClass.UNSUPPORTED,
        "no span becomes user-visible text: the attributes that carry "
        f"conversation content ({', '.join(_CONTENT_ATTRIBUTES)}) are defined by "
        "reference to a JSON schema the pinned package does not ship, so they "
        "are carried in extensions and not read. A per-event record names the "
        "ones a given span actually had",
    )
    losses.add(
        "events[].outcome",
        LossClass.UNSUPPORTED,
        "the IR outcome slot is free text with no vocabulary an OTLP status code "
        "maps into, so no span fills it; a per-event record names the spans that "
        "carried a status worth mapping",
    )
    for field in ("reasoning", "reasoning_segments"):
        losses.add(
            f"events[].{field}",
            LossClass.UNSUPPORTED,
            f"the pinned GenAI vocabulary ({SEMCONV_VERSION}) has no attribute for "
            "reasoning content, so there are neither thoughts nor boundaries "
            "between them to carry",
        )
    if any(event.extensions["otel"]["attributes"] for event in events):
        # Faithful decoding means no *value* was lost or collapsed. It does not
        # mean the wire form survives: the canonical protobuf JSON mapping
        # writes an int64 as a string and much of the ecosystem writes it as a
        # number, so ``{"intValue": "7"}`` and ``{"intValue": 7}`` both decode
        # to ``7`` and become indistinguishable, and the AnyValue wrapper type
        # goes the same way. Semantic preservation, wire normalization —
        # declared rather than implied, and declared once, because it is a
        # property of the decoding and not of any one span.
        losses.add(
            "events[].extensions",
            LossClass.NORMALIZED,
            "attribute values are decoded out of their AnyValue wrappers into a "
            "JSON map, so the wrapper type and the int64 spelling ('7' versus 7) "
            "are not recoverable from it; the values are preserved, their wire "
            "form is not. The original list is kept beside the map only when a "
            "value itself would have been lost, and a per-event record says so",
        )
    losses.add(
        "events[].source_type",
        LossClass.UNSUPPORTED,
        "OTLP models the span name as a scalar with no presence, so an absent "
        "name and an empty one are the same document and the IR's "
        "absent-versus-empty distinction cannot be recovered",
    )
    if has_tool_call:
        losses.add(
            "events[].tool_call.title",
            LossClass.UNSUPPORTED,
            "the pinned vocabulary has no human-readable label for a tool call",
        )
