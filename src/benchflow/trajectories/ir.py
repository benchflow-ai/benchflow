"""Canonical Trace IR — the provisional hub representation for trace interop.

> **PROVISIONAL — v0.** This module is a proposal implemented in code so it can
> be reviewed as code. No maintainer has approved this direction; see
> `docs/trace-interop.md` §8. Nothing imports it, nothing writes it to disk, and
> no existing format changes because it exists. It is deliberately reversible:
> deleting this file and its test restores the tree to its previous behaviour.

## Why a hub at all

BenchFlow has four trace-shaped representations in play — the ACP-session
capture events (`acp_trajectory.jsonl`), ATIF (`trainer/atif.json`), ADP
(`trainer/adp.jsonl`) and the Verifiers/ORS record — and OpenTelemetry is the
obvious fifth. Pairwise converters cost ``N*(N-1)`` edges and, worse, give every
edge its own private answer to the same questions: what happens to a tool call
with no arguments, how a thought boundary is preserved, whether a timeout is
representable. Those answers already diverge today (`docs/trace-interop.md` §5).

A hub makes each format's conversion one edge to a written contract, and makes
the information loss a *value* rather than a comment — see :class:`LossReport`.

## The rule this module is built on

**The IR is a pragmatic superset of what BenchFlow can actually observe, not a
model of what an agent trace could theoretically contain.** Every field below
exists because some source in this repository carries the value today, or
because an adjacent format has a required slot for it. Where a value is not
observable, the IR carries ``None`` *and requires a matching loss record* —
absence is declared, never silent (see :func:`validate_trace`).

Consequently the IR does **not** invent: tool arguments (the ACP capture path
never reads ``rawInput``), per-event timestamps for sources that carry none,
agent versions, synthetic tool-call ids, or OTel span/trace ids. Those are
target-side concerns and belong in converters, which record them as
:attr:`LossClass.SYNTHESIZED`.

## Tri-state fields

For an optional value, the IR distinguishes three states, and converters must
preserve the distinction:

- a value          — observed in the source;
- ``None``         — not available from this source (a loss record says why);
- an empty value   — observed *and* empty (``{}``, ``""``, ``[]``).

``arguments={}`` ("the source captured an empty argument map") and
``arguments=None`` ("the source never carried arguments") are different facts.
Today every ACP-derived tool call is the second; ATIF and ADP both serialize the
first, which is why their documents read as though the agent called every tool
with no arguments.

## Canonical JSON encoding

A trace serializes with **every null retained** — ``model_dump(mode="json")`` or
``model_dump_json()``. **``exclude_none=True`` is not a valid encoding of a
Trace IR document.**

This is a semantic rule, not a formatting preference. ``None`` here is a
positive statement — *the source did not carry this field* — and every such
statement is paired with a :class:`LossRecord` that addresses the field **by
path**. Drop the key and the record points at something a reader of the document
cannot find: the declaration that makes the absence legal becomes unverifiable
inside the very document that carries it, and "we looked and it was not there"
becomes indistinguishable from "this version has no such field".

A pydantic consumer is unaffected either way — both encodings re-validate to an
equal model — but the audience of an interchange format reads the JSON, and it
is the JSON that has to be self-describing.

No dedicated serializer ships with this module. There is no on-disk artifact
yet, and providing a writer would anticipate an interface this proposal has not
earned. The rule is enforced by
``test_every_concrete_loss_path_resolves_in_the_canonical_encoding`` rather than
by a function, so a future writer inherits it instead of redefining it.

**Corollary — address the outermost absent node.** A record may only name a path
that resolves, so when a whole section is missing the record names the section,
not a field inside it: a conversion with no usage at all declares ``usage``, not
``usage.input_tokens``. Sections that every conversion has an opinion about —
:attr:`CanonicalTrace.agent`, :attr:`CanonicalTrace.outcome` — are therefore
always present, with ``None`` fields inside them.

Both rules apply to :attr:`PathSpace.HUB` records only; see below.

## Path spaces

Not every record is about a node of the IR. An inbound edge can read an input
element that becomes no IR node at all, and an outbound edge can emit a value
the IR never held — a field its target format requires, or one supplied by the
conversion context rather than by the trace. Neither has an IR path, and
inventing one would produce an address that does not resolve.

:class:`PathSpace` states which document a record's path addresses, so the space
is a property of the record rather than something inferred from the string. See
that class for the three values and why three is enough for any format.

Two consequences worth stating plainly:

- **Only ``HUB`` records compose across edges.** The IR is the output of an
  inbound conversion and the input of an outbound one, so ``events[1].tool_call.arguments``
  denotes the same field in both reports and the two records join on it: the
  ACP edge declares it ``UNSUPPORTED`` (the source never carried arguments), the
  ATIF edge declares it ``SYNTHESIZED`` (the target required a value). Read
  together they are the whole history of one field along the pipeline.
- **``SOURCE`` and ``TARGET`` records are terminal.** They name objects in
  documents that only one edge ever sees, so joining them across edges would be
  meaningless.

## Which side owns the report

A report belongs to a *conversion*, not to a document — but for one direction
the distinction collapses, and that is why :attr:`CanonicalTrace.losses` exists:

- **Inbound** (``X -> IR``): a trace is built exactly once, by one conversion,
  so its report may be attached to it. A trace separated from the record of what
  building it cost is a trace whose absences cannot be checked.
- **Outbound** (``IR -> Y``): one trace may be converted to ATIF, to OTel and to
  ADP, so there are *N* reports and none of them describes how the trace came to
  exist. An outbound converter therefore **returns** its report alongside its
  document and leaves ``trace.losses`` untouched.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ``v0`` is a statement, not a placeholder: the shape is unreviewed and carries
# no compatibility promise. There is deliberately no migration machinery — a v1
# would be a new constant and a converter, not a mutation of this one. Named so
# it can never be confused with ``ATIF-v1.7`` or ADP's ``1.3.1``.
TRACE_IR_VERSION = "bf-trace-ir-v0"


class LossClass(StrEnum):
    """Why a value is not in the IR (or not in a target document).

    The distinction between the first two is the one that decides *where a fix
    would have to land*, and it is the same taxonomy `docs/trace-interop.md`
    §5.1 already uses for `ACP -> ATIF`.
    """

    UNSUPPORTED = "unsupported"
    """The source never carried the value. Not fixable in a converter."""

    DROPPED = "dropped"
    """The source carried it and the conversion discarded it. Fixable here."""

    NORMALIZED = "normalized"
    """Carried, but relocated or reshaped — readable only with convention
    knowledge (an ACP ``kind`` landing in an ATIF ``function_name`` slot)."""

    SYNTHESIZED = "synthesized"
    """The *target* required a value the source did not have, and the converter
    produced one. Recorded so a fabricated value is never mistaken for an
    observed one (ATIF's ``call_{n}`` ids, ADP's ``call_NNNNNN``)."""


class PathSpace(StrEnum):
    """Which document a :attr:`LossRecord.field` path addresses.

    Every conversion has the IR on exactly one side, so it has exactly one
    non-hub space: an inbound edge (``X -> IR``) can talk about its *source*, an
    outbound edge (``IR -> Y``) about its *target*. Three spaces therefore cover
    every direction, and a new format adds none.

    The space is a property of the record, never of the string: ``field`` is the
    path *inside* its space and carries no prefix announcing which one. Two
    records may legitimately hold the identical path in different spaces — for
    ``acp -> ir``, source ``events[3]`` and hub ``events[3]`` are different
    objects whenever an earlier entry produced no IR event.
    """

    HUB = "hub"
    """A node of the canonical IR. **The only composable space**: the IR is the
    output of an inbound edge and the input of an outbound one, so the same path
    means the same thing in both reports and their records join on it."""

    SOURCE = "source"
    """An element of an inbound edge's input with no corresponding IR node."""

    TARGET = "target"
    """A value an outbound edge produced with no IR antecedent — required by the
    target format, or supplied by the conversion context rather than by the
    trace."""


class LossRecord(BaseModel):
    """One declared information loss, addressed to a field."""

    model_config = ConfigDict(extra="forbid")

    field: str
    """Dotted path **within** :attr:`space`, e.g. ``events[3].tool_call.arguments``.

    For :attr:`PathSpace.HUB` — the default, and the only one the resolvability
    guard checks — this addresses the canonical IR, so one vocabulary covers
    every direction and records from different edges join on it."""

    space: PathSpace = PathSpace.HUB
    """Which document :attr:`field` addresses. Defaults to the hub, so every
    record written before this field existed keeps its meaning."""

    loss_class: LossClass
    detail: str
    """Why, in one sentence, naming the responsible symbol where possible."""

    doc_ref: str | None = None
    """Anchor into ``docs/trace-interop.md`` (e.g. ``"§5 loss #1"``)."""


class LossReport(BaseModel):
    """The explicit, typed result of one conversion.

    A conversion returns a trace *and* this. A converter that carries nothing
    across still produces a report; an empty report is a claim ("nothing was
    lost"), not a default.
    """

    model_config = ConfigDict(extra="forbid")

    direction: str
    """``"acp->ir"``, ``"ir->atif"``, … — free text, one edge of the hub."""

    ir_version: str = TRACE_IR_VERSION
    records: list[LossRecord] = Field(default_factory=list)

    def add(
        self,
        field: str,
        loss_class: LossClass,
        detail: str,
        doc_ref: str | None = None,
        space: PathSpace = PathSpace.HUB,
    ) -> None:
        self.records.append(
            LossRecord(
                field=field,
                space=space,
                loss_class=loss_class,
                detail=detail,
                doc_ref=doc_ref,
            )
        )

    def by_class(self, loss_class: LossClass) -> list[LossRecord]:
        return [r for r in self.records if r.loss_class is loss_class]

    def by_space(self, space: PathSpace) -> list[LossRecord]:
        return [r for r in self.records if r.space is space]

    def for_field(
        self, field: str, space: PathSpace = PathSpace.HUB
    ) -> list[LossRecord]:
        """Records addressing *field* in *space*.

        The space is part of the address: the same string in two spaces names
        two different objects, so it is not defaulted away silently — the
        default is the hub because that is the composable space.
        """
        return [r for r in self.records if r.field == field and r.space is space]

    @property
    def lossless(self) -> bool:
        """True only when the conversion declared no loss of any class."""
        return not self.records


class Provenance(BaseModel):
    """Where the values in a trace or event came from.

    Kept per-event as well as per-trace because a single `acp_trajectory.jsonl`
    can hold records from more than one producer (`docs/trace-interop.md` §2.4),
    so a trace-level answer would be wrong for some of its own events.
    """

    model_config = ConfigDict(extra="forbid")

    source_format: str
    """``"acp-capture-v1"``, ``"atif"``, ``"adp"``, ``"otel"``, ``"oracle"``…"""

    producer: str | None = None
    """The emitting symbol when known (``"_events_to_trajectory"``)."""

    captured_at: datetime | None = None


class Role(StrEnum):
    """Who a step is attributable to.

    Exactly the four values some source in this repository already
    distinguishes: ATIF's validator accepts ``user``/``agent``/``oracle``, ADP
    uses ``user``/``environment``. No ``system`` member — no producer here emits
    one, and inventing it would be inventing semantics.
    """

    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"
    ORACLE = "oracle"


class EventKind(StrEnum):
    """What an event *is*, normalized across sources.

    ``UNKNOWN`` is load-bearing: the ACP trajectory has an open type vocabulary
    (`docs/trace-interop.md` §2.4, §7) and today every exporter silently skips
    what it does not recognize. An unrecognized record becomes ``UNKNOWN`` with
    :attr:`TraceEvent.source_type` holding the original string, so it survives
    conversion instead of vanishing.
    """

    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    AGENT_REASONING = "agent_reasoning"
    TOOL_CALL = "tool_call"
    TIMEOUT = "timeout"
    ORACLE = "oracle"
    UNKNOWN = "unknown"


class ToolStatus(StrEnum):
    """Tool-call lifecycle status.

    Mirrors the ACP ``ToolCallStatus`` vocabulary, plus ``UNKNOWN`` for sources
    that carry no status at all (ADP drops it outright — §5 loss #2). The
    superset relationship is pinned by a test rather than by an import, so the
    IR does not take a runtime dependency on the ACP layer it is supposed to be
    neutral about.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ContentBlockKind(StrEnum):
    """Two kinds, because two are all BenchFlow interprets today.

    ``content_blocks_to_text`` renders text blocks and skips everything else,
    which is §5 loss #5. ``OPAQUE`` is how the IR stops that from being a loss:
    the block is not understood, but it is carried.
    """

    TEXT = "text"
    OPAQUE = "opaque"


class ContentBlock(BaseModel):
    """One block of captured tool output.

    ``raw`` holds the source block verbatim and is required for ``OPAQUE``. It
    is also kept for ``TEXT`` when available, so a round-trip can reproduce the
    original block rather than a re-serialization of its text.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ContentBlockKind
    text: str | None = None
    raw: dict[str, Any] | None = None


class ToolCall(BaseModel):
    """One tool invocation and its captured result."""

    model_config = ConfigDict(extra="forbid")

    call_id: str | None = None
    """The id as observed. ``""`` is a real captured value (``handle_update``
    defaults to it); ``None`` means the source carried no id field at all.
    Converters that need a unique id synthesize one and record
    :attr:`LossClass.SYNTHESIZED` — the IR does not."""

    name: str | None = None
    """The tool's name *as the source labels it*."""

    name_semantics: str | None = None
    """What :attr:`name` actually is — ``"acp_kind"``, ``"function_name"`` or
    ``"span_name"``. An ACP ``kind`` is a category (``execute``, ``read``), not
    a function name, and ATIF today puts it in a ``function_name`` slot. Naming
    the semantics keeps the IR from laundering that normalization into a fact."""

    title: str | None = None
    """Human-readable label. For ACP ``execute`` calls, conventionally the
    command line — which is the only place the command survives today."""

    status: ToolStatus | None = None

    arguments: dict[str, Any] | None = None
    """``None`` = the source never carried arguments (every ACP-derived call
    today); ``{}`` = the source carried an empty argument map. See the module
    docstring; :func:`validate_trace` requires a loss record for the ``None``
    case."""

    content: list[ContentBlock] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    """``ToolCallRecord`` tracks both in memory and ``_events_to_trajectory``
    serializes neither (§5 loss #3), so both are ``None`` for anything read off
    disk today. The slots exist because the values demonstrably exist upstream,
    not because a target format wants them."""


class TraceUsage(BaseModel):
    """Token accounting, with the definition it was computed under.

    Named :class:`TraceUsage` rather than ``TokenUsage`` to stay distinct from
    ``benchflow.trajectories.types.TokenUsage``, which is the proxy-capture
    dataclass.

    ``source`` exists because open question 4 in `docs/trace-interop.md` §6 is
    unanswered: ``_exchange_token_usage`` (cross-provider normalized, cache
    folded into input) and ``normalize_acp_usage`` (the raw ACP snapshot) do not
    mean the same thing. The IR refuses to pick one and instead records which
    was used — a consumer can then compare like with like. Documented values:
    ``"llm_proxy_normalized"``, ``"acp_session_snapshot"``.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    source: str | None = None


class TraceEvent(BaseModel):
    """One ordered step of a trace."""

    model_config = ConfigDict(extra="forbid")

    index: int
    """Position in the source sequence, dense from 0. Ordering is the one
    property every format in play preserves, so the IR makes it structural."""

    kind: EventKind

    source_type: str | None = None
    """The source's own type string, verbatim (``"agent_thought"``,
    ``"oracle"``, an unrecognized future value). Kept next to the normalized
    :attr:`kind` so normalization is never destructive."""

    role: Role | None = None

    text: str | None = None
    """User-visible text. ``""`` is meaningful: text-empty events exist and are
    dropped by both exporters today (§5.1)."""

    reasoning: str | None = None
    """Internal reasoning, kept in its own field rather than merged into
    :attr:`text` — ATIF and ADP both have a separate ``reasoning_content``."""

    reasoning_segments: list[str] | None = None
    """The individual thought events, when the source had boundaries.

    ``ThoughtBuffer.take`` joins thoughts with a blank line, so a thought
    containing a blank line and two consecutive thoughts serialize identically
    and the event count is unrecoverable (§5 loss #10). The segment list is how
    the IR keeps the boundary that the join destroys; :func:`validate_trace`
    checks the two stay consistent."""

    tool_call: ToolCall | None = None

    started_at: datetime | None = None
    finished_at: datetime | None = None

    outcome: str | None = None
    """Terminal signal carried by the event itself — e.g. the
    ``wall_clock_timeout`` reason of an ``agent_timeout`` record, which no
    exporter represents today (§5 loss #4)."""

    usage: TraceUsage | None = None
    """Per-event usage. ``None`` for everything ACP-derived: ACP events carry no
    usage and ``usage_snapshots`` is routed to ``result.json`` instead (§5
    losses #6, #8)."""

    provenance: Provenance
    extensions: dict[str, Any] = Field(default_factory=dict)
    """Source fields with no IR home, carried verbatim so a conversion is not
    forced to choose between dropping a value and growing the IR for it. Values
    must be JSON-serializable."""


class ModelInfo(BaseModel):
    """Agent / model identity, as observed.

    ``agent_version`` is ``None`` when unknown. It is *not* ``"unknown"``:
    ``trajectory_to_atif_record`` hardcodes that string because ATIF requires
    the field, which is a target-side obligation and belongs in that converter,
    recorded as :attr:`LossClass.SYNTHESIZED`.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str | None = None
    agent_version: str | None = None
    model: str | None = None
    provider: str | None = None


class OutcomeStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class TraceOutcome(BaseModel):
    """How the run ended.

    ``stop_reason`` is captured on ``ACPSession`` and exported nowhere (§5 loss
    #9); ``reward`` and ``error_category`` live in ``result.json``, outside the
    trajectory. All fields optional — a trace built from the capture file alone
    has none of them, and that is a declarable loss rather than a reason to fake
    a value.

    The *section* is not optional on :class:`CanonicalTrace`, unlike its fields;
    see :attr:`CanonicalTrace.outcome`.
    """

    model_config = ConfigDict(extra="forbid")

    status: OutcomeStatus | None = None
    stop_reason: str | None = None
    reward: float | None = None
    error_category: str | None = None


class CanonicalTrace(BaseModel):
    """One agent run, in the hub representation."""

    model_config = ConfigDict(extra="forbid")

    ir_version: str = TRACE_IR_VERSION
    trace_id: str | None = None
    """``None`` until a source supplies one. No OTel-shaped id is invented."""

    session_id: str | None = None
    agent: ModelInfo = Field(default_factory=ModelInfo)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    usage: TraceUsage | None = None
    """``None`` when no source measured usage at all. A conversion that has none
    declares the loss against ``usage``, the outermost absent node — see the
    addressing rule in the module docstring."""

    outcome: TraceOutcome = Field(default_factory=TraceOutcome)
    """Always present, like :attr:`agent`, even when every field inside it is
    ``None``.

    It was briefly optional, and that made ``outcome.stop_reason`` — a loss
    every ACP conversion declares — unresolvable in any trace that did not time
    out, because the parent object was ``null``. A section that loss records
    address by path has to exist for the path to land."""

    provenance: Provenance
    extensions: dict[str, Any] = Field(default_factory=dict)

    losses: LossReport | None = None
    """The report from the conversion that produced this trace. Attached rather
    than returned alongside so a trace cannot be passed around separated from
    the record of what building it cost."""


def validate_trace(trace: CanonicalTrace) -> list[str]:
    """Check the IR invariants, returning one string per violation.

    Returns a list rather than raising: a caller checking a batch wants every
    problem, and a test asserting an invariant wants to name it. An empty list
    means every invariant below holds.

    The invariants, and why each one is here:

    1. **Version.** ``ir_version`` equals :data:`TRACE_IR_VERSION`. v0 has no
       migration path, so a mismatch is an error rather than an upgrade.
    2. **Dense ordering.** ``events[i].index == i``. Ordering is the property
       every format preserves; making it structural means a converter cannot
       silently drop an event without leaving a hole.
    3. **Kind/payload agreement.** ``kind is TOOL_CALL`` iff ``tool_call`` is
       set. Without this the IR would have two ways to say the same thing.
    4. **Content-block integrity.** A ``TEXT`` block carries ``text``; an
       ``OPAQUE`` block carries ``raw``. An opaque block with no payload is a
       block that was dropped while claiming to have been preserved.
    5. **Reasoning consistency.** When both ``reasoning`` and
       ``reasoning_segments`` are set, joining the segments with a blank line
       reproduces ``reasoning`` exactly — the join `ThoughtBuffer` performs. The
       segments are then a strictly richer encoding of the same value, not a
       second, divergent one.
    6. **Reasoning presence.** An ``AGENT_REASONING`` event carries reasoning in
       one of the two fields.
    7. **No silent absence.** Every ``arguments is None`` has a matching
       :attr:`PathSpace.HUB` loss record at ``events[i].tool_call.arguments``.
       This is the invariant that makes the loss report a contract instead of
       documentation: a converter that quietly fails to carry arguments produces
       an invalid trace. The space is checked, not guessed from the string — a
       ``TARGET`` record that happens to hold the same path addresses another
       document and does not declare anything about this one. And because the
       record addresses the field *by path*, the canonical encoding has to keep
       that path resolvable; see the module docstring.
    8. **Role coherence.** A ``USER_MESSAGE`` is attributed to ``USER`` or to
       nobody. The IR checks only this direction; agent-side attribution is
       genuinely ambiguous today (§5, the ``oracle`` divergence) and the IR does
       not pretend to settle it.
    """
    issues: list[str] = []

    if trace.ir_version != TRACE_IR_VERSION:
        issues.append(
            f"ir_version {trace.ir_version!r} != {TRACE_IR_VERSION!r}; "
            "v0 defines no migration"
        )

    declared_losses = {
        record.field
        for record in (trace.losses.records if trace.losses else [])
        if record.space is PathSpace.HUB
    }

    for position, event in enumerate(trace.events):
        where = f"events[{position}]"

        if event.index != position:
            issues.append(f"{where}.index is {event.index}, expected {position}")

        is_tool = event.kind is EventKind.TOOL_CALL
        if is_tool and event.tool_call is None:
            issues.append(f"{where} is a tool_call event with no tool_call payload")
        if not is_tool and event.tool_call is not None:
            issues.append(
                f"{where} carries a tool_call payload but kind is {event.kind.value}"
            )

        for block_position, block in enumerate(
            event.tool_call.content if event.tool_call else []
        ):
            block_where = f"{where}.tool_call.content[{block_position}]"
            if block.kind is ContentBlockKind.TEXT and block.text is None:
                issues.append(f"{block_where} is text but carries no text")
            if block.kind is ContentBlockKind.OPAQUE and block.raw is None:
                issues.append(f"{block_where} is opaque but carries no raw block")

        if event.reasoning is not None and event.reasoning_segments is not None:
            joined = "\n\n".join(event.reasoning_segments)
            if joined != event.reasoning:
                issues.append(f"{where}.reasoning_segments do not join to .reasoning")
        if event.kind is EventKind.AGENT_REASONING and not (
            event.reasoning or event.reasoning_segments
        ):
            issues.append(f"{where} is agent_reasoning but carries no reasoning")

        if event.tool_call is not None and event.tool_call.arguments is None:
            field = f"{where}.tool_call.arguments"
            if field not in declared_losses:
                issues.append(
                    f"{field} is None with no loss record; absence must be declared"
                )

        if (
            event.kind is EventKind.USER_MESSAGE
            and event.role is not None
            and event.role is not Role.USER
        ):
            issues.append(f"{where} is a user_message attributed to {event.role.value}")

    return issues
