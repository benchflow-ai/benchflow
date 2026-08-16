"""Round-trip measurement: how much of a trace survives `ACP → IR → ATIF → IR′`.

> **PROVISIONAL.** Companion to :mod:`benchflow.trajectories.ir`, itself an
> unapproved proposal (`docs/trace-interop.md` §8). Nothing imports this module
> and no artifact changes because it exists.

Slice D established that the hub *reproduces* the direct ATIF exporter — the
document that comes out is the one that came out before. That answers "is the IR
sufficient to write ATIF?" and says nothing about the more useful question:

**how much of the trace is still there after a trip through the format?**

This module answers it as a measurement rather than an assertion. It runs the
loop, compares the trace that went in against the trace that came back, and
reports the difference per field.

## Why the answer is not a percentage

A single number would hide the two things worth knowing. The comparison
therefore separates *what happened to a value* from *whether ATIF could have
carried it*:

- :class:`RoundTripOutcome` is **observed** — it comes from comparing the two
  traces and nothing else.
- :class:`Representability` is **declared**, in :data:`ATIF_REPRESENTABILITY`,
  one entry per IR field.

Crossing them is what makes the report actionable. A value that is gone *and*
has no ATIF slot is a cost of the format: no converter can fix it, and only a
format change or a side channel would. A value that is gone and ATIF *does* have
a slot for it is a gap in our own edge — a bug report, not a fact of life. Both
read as "lost" in a percentage, and they have nothing to do with each other.

## The fourth outcome

The loop does not only subtract. :attr:`RoundTripOutcome.FABRICATED` marks a
field that had no value going in and has one coming back, and it is the finding
this measurement exists to surface:

`ir_to_atif` writes `arguments: {}` for a call the IR says has *no* arguments,
and declares it ``SYNTHESIZED``. `atif_to_ir` reads the document as written —
correctly, since nothing in it marks the value as invented — and produces a
trace asserting the call was observed with an empty argument map. The tri-state
contract that made `None` mean "the source never carried this" is intact in both
traces; it is the *round trip* that turns one state into the other.

The information was not lost so much as **overwritten with a plausible value of
the same shape**. A consumer reading only the second trace cannot tell. That is
the strongest available argument for the hub — not that it converts, but that it
carries a report the format cannot.

## What is deliberately not compared

``ir_version``, ``losses`` and ``provenance`` (trace- and event-level) describe
the *representation*, not the run: a trace read from ATIF is correctly labelled
as coming from ATIF, and calling that a loss would count a true statement as
damage. Slice D excludes the same fields from its outbound coverage rule, for
the same reason.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchflow.trajectories.ir import CanonicalTrace, LossReport, TraceUsage
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_from_atif import atif_to_ir
from benchflow.trajectories.ir_to_atif import ir_to_atif


class RoundTripOutcome(StrEnum):
    """What the comparison observed for one field. Measured, never declared."""

    PRESERVED = "preserved"
    """Same values, same number of them, in the same order."""

    TRANSFORMED = "transformed"
    """Values on both sides, but not the same ones — reshaped, renumbered,
    fused, or partially carried."""

    LOST = "lost"
    """Values going in, none coming back."""

    FABRICATED = "fabricated"
    """No values going in, values coming back. The round trip invented them."""


class Representability(StrEnum):
    """Whether ATIF has anywhere to put a field. Declared, never measured."""

    REPRESENTABLE = "representable"
    """ATIF has a slot. A loss here is a gap in a converter, and fixable."""

    NOT_IN_ATIF = "non_representable"
    """ATIF has no slot at all. A loss here is a cost of the format."""


# One entry per IR field, keyed by canonical path (list indices collapsed to
# ``[]``). Lookup is longest-prefix, so ``extensions`` covers every key under it.
#
# This table is the declared half of the measurement and the place to argue with
# it. A test derives the field list from the IR models and fails when a field has
# no entry, so the IR cannot grow a field that silently escapes the round trip.
ATIF_REPRESENTABILITY: dict[str, Representability] = {
    # -- trace level ---------------------------------------------------------
    "trace_id": Representability.NOT_IN_ATIF,
    "session_id": Representability.REPRESENTABLE,
    "started_at": Representability.NOT_IN_ATIF,
    "finished_at": Representability.NOT_IN_ATIF,
    "extensions": Representability.NOT_IN_ATIF,
    "agent.agent_name": Representability.REPRESENTABLE,
    "agent.agent_version": Representability.REPRESENTABLE,
    "agent.model": Representability.REPRESENTABLE,
    "agent.provider": Representability.NOT_IN_ATIF,
    # ATIF's final_metrics has four of the nine usage fields.
    "usage.input_tokens": Representability.REPRESENTABLE,
    "usage.output_tokens": Representability.REPRESENTABLE,
    "usage.cache_read_tokens": Representability.REPRESENTABLE,
    "usage.cost_usd": Representability.REPRESENTABLE,
    "usage.cache_creation_tokens": Representability.NOT_IN_ATIF,
    "usage.reasoning_tokens": Representability.NOT_IN_ATIF,
    "usage.total_tokens": Representability.NOT_IN_ATIF,
    "usage.source": Representability.NOT_IN_ATIF,
    "usage.price_source": Representability.NOT_IN_ATIF,
    "outcome.status": Representability.NOT_IN_ATIF,
    "outcome.stop_reason": Representability.NOT_IN_ATIF,
    "outcome.reward": Representability.NOT_IN_ATIF,
    "outcome.error_category": Representability.NOT_IN_ATIF,
    # -- event level ---------------------------------------------------------
    # step_id is dense over emitted steps, so the number survives but the
    # identity does not; that shows up as TRANSFORMED whenever an event is
    # dropped ahead of it.
    "events[].index": Representability.REPRESENTABLE,
    # The step shape encodes user/agent/oracle and a tool call. It has no shape
    # for a timeout or an unrecognized record: those kinds are unrepresentable
    # values of a representable field, which the per-value counts show.
    "events[].kind": Representability.REPRESENTABLE,
    "events[].role": Representability.REPRESENTABLE,
    "events[].text": Representability.REPRESENTABLE,
    "events[].reasoning": Representability.REPRESENTABLE,
    "events[].source_type": Representability.NOT_IN_ATIF,
    "events[].reasoning_segments": Representability.NOT_IN_ATIF,
    "events[].started_at": Representability.NOT_IN_ATIF,
    "events[].finished_at": Representability.NOT_IN_ATIF,
    "events[].outcome": Representability.NOT_IN_ATIF,
    "events[].extensions": Representability.NOT_IN_ATIF,
    # ATIF *does* have a per-step metrics slot; `ir_to_atif` writes none. So a
    # loss here is ours to close, unlike the timestamps beside it.
    "events[].usage": Representability.REPRESENTABLE,
    "events[].tool_call.call_id": Representability.REPRESENTABLE,
    "events[].tool_call.name": Representability.REPRESENTABLE,
    "events[].tool_call.title": Representability.REPRESENTABLE,
    "events[].tool_call.status": Representability.REPRESENTABLE,
    "events[].tool_call.arguments": Representability.REPRESENTABLE,
    "events[].tool_call.name_semantics": Representability.NOT_IN_ATIF,
    "events[].tool_call.started_at": Representability.NOT_IN_ATIF,
    "events[].tool_call.finished_at": Representability.NOT_IN_ATIF,
    "events[].tool_call.content[].kind": Representability.REPRESENTABLE,
    "events[].tool_call.content[].text": Representability.REPRESENTABLE,
    "events[].tool_call.content[].raw": Representability.NOT_IN_ATIF,
}

# Fields describing the representation rather than the run. See the module
# docstring: comparing them would count a true statement as damage.
EXCLUDED_PATHS: frozenset[str] = frozenset(
    {"ir_version", "losses", "provenance", "events[].provenance"}
)

# Carried verbatim from a source that has no schema here, so their internal
# shape is not a set of IR fields. Compared whole: a source block either comes
# back or it does not, and reporting three findings because it happened to have
# three keys would weight one loss by the shape of the data.
OPAQUE_PATHS: frozenset[str] = frozenset(
    {"events[].tool_call.content[].raw", "events[].tool_call.arguments"}
)
"""Both hold a mapping whose keys come from a tool or an agent, not from the IR.

``arguments`` is here for the same reason ``raw`` is, and for one more: without
it the field's path would depend on its contents — ``arguments.cmd`` when a call
carries arguments, ``arguments`` when it carries ``{}`` — so the same IR field
would be counted in different places depending on the data."""

# Fields whose default *is* the empty container. An empty value at one of these
# is the absence of values, not an observed empty one, so it is not counted —
# unlike ``arguments``, which defaults to ``None`` and where ``{}`` is the
# observation the fabrication finding turns on.
DEFAULT_EMPTY_PATHS: frozenset[str] = frozenset(
    {"events", "extensions", "events[].extensions", "events[].tool_call.content"}
)


class FieldComparison(BaseModel):
    """One canonical field path, before and after the round trip."""

    model_config = ConfigDict(extra="forbid")

    path: str
    outcome: RoundTripOutcome
    representability: Representability

    before_count: int
    """How many non-null values the input trace held at this path."""

    after_count: int
    matched_count: int
    """How many of the input's values came back unchanged, counted as a
    multiset — order-insensitive, so a renumbering that keeps every value is
    visible as a preserved set rather than as total loss.

    Read it together with :attr:`outcome`. ``TRANSFORMED`` with ``matched=0``
    means *replaced*, not partially carried: ``events[].source_type`` has a
    value on both sides and not one in common, because ATIF's step `source` sat
    down where the ACP type string used to be.

    For a positional field the multiset can match while the events do not
    correspond at all — ``events[].index`` is dense on both sides, so it matches
    whenever the two traces are the same length, and says nothing about whether
    event *i* is the same event. Event correspondence is what
    :attr:`RoundTripReport.kinds_before` / ``kinds_after`` are for."""

    sample_before: Any = None
    """One representative dropped or changed value, for reading the report."""

    sample_after: Any = None

    @property
    def is_loss(self) -> bool:
        return self.outcome in (RoundTripOutcome.LOST, RoundTripOutcome.TRANSFORMED)


class RoundTripReport(BaseModel):
    """The measured difference between a trace and its round trip."""

    model_config = ConfigDict(extra="forbid")

    comparisons: list[FieldComparison] = Field(default_factory=list)
    events_before: int = 0
    events_after: int = 0
    kinds_before: dict[str, int] = Field(default_factory=dict)
    kinds_after: dict[str, int] = Field(default_factory=dict)

    def by_outcome(self, outcome: RoundTripOutcome) -> list[FieldComparison]:
        return [c for c in self.comparisons if c.outcome is outcome]

    def summary(self) -> dict[str, int]:
        """Field counts per outcome, with lost split by representability.

        The split is the point of the report: ``lost`` is what a converter could
        still recover, ``non_representable`` is what the format cannot hold.
        """
        counts = {
            RoundTripOutcome.PRESERVED.value: 0,
            RoundTripOutcome.TRANSFORMED.value: 0,
            "lost": 0,
            "non_representable": 0,
            RoundTripOutcome.FABRICATED.value: 0,
        }
        for comparison in self.comparisons:
            if comparison.outcome is RoundTripOutcome.LOST:
                key = (
                    "non_representable"
                    if comparison.representability is Representability.NOT_IN_ATIF
                    else "lost"
                )
            else:
                key = comparison.outcome.value
            counts[key] += 1
        return counts

    def value_summary(self) -> dict[str, int]:
        """The same measurement counted in values rather than in fields.

        A field-level count treats ``events[].text`` as one thing whether the
        trace has two events or two hundred. This one weights by how much data
        each field actually held.
        """
        total = sum(c.before_count for c in self.comparisons)
        matched = sum(c.matched_count for c in self.comparisons)
        return {
            "values_before": total,
            "values_after": sum(c.after_count for c in self.comparisons),
            "values_preserved": matched,
            "values_not_preserved": total - matched,
        }


def _canonical_leaves(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Every leaf of a dumped trace as ``(canonical_path, value)``.

    List indices collapse to ``[]``, so ``events[0].text`` and ``events[7].text``
    are the same path with two values. That is what makes the comparison
    independent of event alignment — which is not recoverable after a conversion
    that fuses and drops events, and guessing at it would be the same kind of
    invention this whole slice refuses.

    Empty containers are leaves: ``{}`` is a value, and the difference between
    ``None`` and ``{}`` is the one the fabrication finding rests on — except at
    the paths whose default is the empty container, which
    :func:`_values_by_path` filters out.

    Paths in :data:`OPAQUE_PATHS` are leaves whatever they contain.
    """
    if path in OPAQUE_PATHS:
        return [(path, node)]
    if isinstance(node, dict) and node:
        leaves: list[tuple[str, Any]] = []
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            leaves.extend(_canonical_leaves(value, child))
        return leaves
    if isinstance(node, list) and node:
        leaves = []
        for item in node:
            leaves.extend(_canonical_leaves(item, f"{path}[]"))
        return leaves
    return [(path, node)]


def _is_excluded(path: str) -> bool:
    return any(
        path == excluded or path.startswith(f"{excluded}.")
        for excluded in EXCLUDED_PATHS
    )


def _values_by_path(trace: CanonicalTrace) -> dict[str, list[Any]]:
    """Non-null values of *trace*, grouped by canonical path."""
    dumped = trace.model_dump(mode="json")
    grouped: dict[str, list[Any]] = {}
    for path, value in _canonical_leaves(dumped):
        if value is None or _is_excluded(path):
            continue
        if path in DEFAULT_EMPTY_PATHS and value in ({}, []):
            continue
        grouped.setdefault(path, []).append(value)
    return grouped


def _lookup_candidates(path: str):
    """*path*, then its parents, each also tried without a trailing ``[]``.

    A list of scalars dumps to ``events[].reasoning_segments[]`` while the table
    names the field itself, so the bare form has to be tried too — otherwise the
    entry would be missed and the field would silently default to
    unrepresentable.
    """
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        prefix = ".".join(parts[:cut])
        yield prefix
        if prefix.endswith("[]"):
            yield prefix[:-2]


def declared_entry_for(path: str) -> str | None:
    """The table entry governing *path*, or ``None`` when nothing does.

    Separate from :func:`representability_of` because "no entry" and "an entry
    saying unrepresentable" are different states, and only the first is a gap in
    the table. The suite asserts there are none.
    """
    for candidate in _lookup_candidates(path):
        if candidate in ATIF_REPRESENTABILITY:
            return candidate
    return None


def representability_of(path: str) -> Representability:
    """The declared representability of *path*, by longest matching prefix.

    Unknown paths — every key a trace carries inside ``extensions`` — inherit
    from their parent, which is why the table needs no entry per extension key.
    Anything with no entry at all is treated as unrepresentable: a field ATIF was
    never shown to carry should not be credited as carriable by default.
    """
    entry = declared_entry_for(path)
    return ATIF_REPRESENTABILITY[entry] if entry else Representability.NOT_IN_ATIF


def _multiset_overlap(before: list[Any], after: list[Any]) -> int:
    """How many of *before*'s values appear in *after*, counted with duplicates.

    Values are compared by their JSON dump, since a trace dumps to plain JSON
    types and dicts are not hashable.
    """
    remaining = list(after)
    matched = 0
    for value in before:
        for position, candidate in enumerate(remaining):
            if candidate == value:
                remaining.pop(position)
                matched += 1
                break
    return matched


def compare_traces(before: CanonicalTrace, after: CanonicalTrace) -> RoundTripReport:
    """Compare a trace with its round trip, field by canonical field.

    Neither argument is modified and neither report is read: this is a
    measurement of the *documents*, deliberately independent of what the
    converters declared. A converter that lost something without declaring it
    shows up here all the same, which is the property that makes this worth
    running against the loss reports rather than instead of them.
    """
    before_values = _values_by_path(before)
    after_values = _values_by_path(after)

    comparisons: list[FieldComparison] = []
    for path in sorted(set(before_values) | set(after_values)):
        mine = before_values.get(path, [])
        theirs = after_values.get(path, [])
        matched = _multiset_overlap(mine, theirs)

        if not mine:
            outcome = RoundTripOutcome.FABRICATED
        elif not theirs:
            outcome = RoundTripOutcome.LOST
        elif matched == len(mine) == len(theirs):
            outcome = RoundTripOutcome.PRESERVED
        else:
            outcome = RoundTripOutcome.TRANSFORMED

        comparisons.append(
            FieldComparison(
                path=path,
                outcome=outcome,
                representability=representability_of(path),
                before_count=len(mine),
                after_count=len(theirs),
                matched_count=matched,
                sample_before=mine[0] if mine else None,
                sample_after=theirs[0] if theirs else None,
            )
        )

    return RoundTripReport(
        comparisons=comparisons,
        events_before=len(before.events),
        events_after=len(after.events),
        kinds_before=_kind_counts(before),
        kinds_after=_kind_counts(after),
    )


def _kind_counts(trace: CanonicalTrace) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in trace.events:
        counts[event.kind.value] = counts.get(event.kind.value, 0) + 1
    return counts


class RoundTrip(BaseModel):
    """One full `ACP → IR → ATIF → IR′` loop and everything it produced."""

    model_config = ConfigDict(extra="forbid")

    before: CanonicalTrace
    document: dict[str, Any]
    after: CanonicalTrace
    outbound: LossReport
    """`IR → ATIF`, returned by the outbound edge."""

    report: RoundTripReport

    @property
    def inbound(self) -> LossReport | None:
        """`ATIF → IR`, carried on the reconstructed trace."""
        return self.after.losses


def round_trip_through_atif(
    events: list[dict[str, Any]],
    *,
    prompts: list[str] | None = None,
    session_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    usage: TraceUsage | None = None,
) -> RoundTrip:
    """Run captured ACP events through the hub, out to ATIF, and back.

    *prompts* are passed to the outbound edge exactly as a real export does.
    They are worth including in at least one measurement: they are not trace
    data, they become `user` steps declared ``SYNTHESIZED`` in the target space,
    and they come back indistinguishable from captured user messages — the same
    laundering as `arguments`, one level up.

    *usage* is the token accounting a real export passes to
    `trajectory_to_atif_record`, which reads it from `result.json`. The capture
    events carry none, so without it the four representable usage fields never
    enter the measurement at all — the loop would report on a trace poorer than
    the one a rollout actually produces.
    """
    before = acp_events_to_ir(
        events, session_id=session_id, agent_name=agent_name, model=model
    )
    if usage is not None:
        before = before.model_copy(update={"usage": usage})
    document, outbound = ir_to_atif(before, prompts=prompts)
    after = atif_to_ir(document)
    return RoundTrip(
        before=before,
        document=document,
        after=after,
        outbound=outbound,
        report=compare_traces(before, after),
    )
