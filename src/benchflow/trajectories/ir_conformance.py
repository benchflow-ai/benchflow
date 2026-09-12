"""Loss-bounded conformance — is the observed divergence inside the declared contract?

> **PROVISIONAL.** Part of the unwired IR family (`docs/trace-interop.md` §8.7).
> Nothing imports it from a run path and it changes no artifact.

`ir_round_trip` measures **D**: what actually differs between a trace and its
round trip, read off the two traces and never off the reports. The converters
produce **L**: what each edge declared it could not carry, or had to invent.
Neither knows about the other, and that separation is load-bearing —
`test_the_harness_reads_traces_and_not_reports` exists to keep the measurement
from confirming the converters against themselves.

This module is the **third** thing: the join. It answers one question — *is every
observed divergence accounted for by something the contract declared?* — and it
is deliberately not folded into either half.

## Why the join is not a set comparison

The obvious form, ``D(T, R(T)) ⊆ L(T)``, does not typecheck against the model
as it stands, for two reasons that are worth stating rather than papering over.

**The two sides address different spaces.** D speaks in canonical *hub* paths of
the IR (``events[].text``, indices collapsed). L speaks in three spaces
(:class:`~benchflow.trajectories.ir.PathSpace`), and an outbound edge
legitimately declares a value in ``TARGET`` space using its *target's*
vocabulary. `ir_to_atif` writes ``schema_version`` into the ATIF document and
declares it there, because at that moment the value has no IR antecedent — a hub
path would address a node the trace being converted does not contain. Then
`ir_from_atif` reads it faithfully back into ``extensions.schema_version``, and
the round trip sees a value the input never had. **Neither edge lied; the
fabrication is a property of the composition.** :data:`TARGET_TO_HUB` is the
bridge, and every entry in it is verified by a test rather than asserted.

**Not every divergence is a field-level fact.** When an event is fused away, the
values it held stop appearing — at every path it populated. Those divergences
are caused by a change in the event sequence, not by any converter mishandling a
field. :func:`structure_explained` is how the join says so **without** a blanket
exemption for the paths involved: the values that went missing must be exactly
values held by events of a kind that lost instances. An arbitrary edit to a
surviving event produces a missing value no vanished event held, and fails.

## The rules

1. **Undeclared fabrication is always a violation.** A path with values coming
   back and none going in must have a ``SYNTHESIZED`` declaration at that path —
   directly, or through the bridge. There is no allowlist, and structural
   metadata is not exempt: ``schema_version`` and ``step_id`` are declared like
   anything else, with a detail that says what they are.
2. **A representable loss is a violation.** Values that went in, none that came
   back, at a path the target *does* have a slot for, is a gap in our own edge.
3. **A transformed path must account for both of its sides.** Values that
   disappeared: declared, or structure-explained. Values that appeared:
   declared. ``TRANSFORMED`` conflates the two, and a rule that checked only the
   first would let an arbitrary insertion through.

## What a clean run does not establish

That the contract is *right*. An edge that declares ``SYNTHESIZED`` on a field
and then invents it passes here — the gate checks that the declaration and the
observation agree, not that the declaration is true. It is a consistency check
between two independently produced artefacts, which is exactly as much as a
join can be.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchflow.trajectories.ir import CanonicalTrace, LossClass, LossReport, PathSpace
from benchflow.trajectories.ir_round_trip import (
    Representability,
    RoundTripOutcome,
    RoundTripReport,
    _values_by_path,
)

TARGET_TO_HUB: dict[str, str] = {
    "schema_version": "extensions.schema_version",
    "steps[].step_id": "events[].extensions.step_id",
    "steps[].message": "events[].text",
}
"""Where a value declared in ``TARGET`` space lands in the hub on the way back.

Three entries, all for the `ATIF` pair, and each one is a fact about
`ir_from_atif` rather than a convention: it stores the document's
``schema_version`` under the trace's ``extensions``, each step's ``step_id``
under its event's, and a step ``message`` becomes the event's ``text``.

**This is a bridge, not an exemption.** Without a ``SYNTHESIZED`` record at the
target path the gate still fails; the map only lets the join find a declaration
that was made honestly in the other space. ``test_every_bridge_entry_is_real``
runs a trip and checks each correspondence against the values, so an entry
cannot rot into a lie.
"""


class Rule(BaseModel):
    """Marker for the rule a violation broke, so failures are greppable."""

    model_config = ConfigDict(extra="forbid")

    name: str
    text: str


UNDECLARED_FABRICATION = Rule(
    name="undeclared-fabrication",
    text="values came back that never went in, and no SYNTHESIZED record "
    "declares them at that path",
)
REPRESENTABLE_LOSS = Rule(
    name="representable-loss",
    text="values went in and none came back, at a path the target has a slot "
    "for — a gap in our own edge rather than a cost of the format",
)
UNEXPLAINED_DISAPPEARANCE = Rule(
    name="unexplained-disappearance",
    text="values stopped appearing at a path that declares nothing, and they "
    "were not held by the events the round trip fused away",
)
UNDECLARED_INSERTION = Rule(
    name="undeclared-insertion",
    text="values appeared at a transformed path with no record declaring them",
)


class Violation(BaseModel):
    """One observed divergence the declared contract does not account for."""

    model_config = ConfigDict(extra="forbid")

    path: str
    outcome: RoundTripOutcome
    rule: str
    detail: str
    declared: list[str] = Field(default_factory=list)
    """The loss classes found at this path, if any — empty is the usual cause."""

    def __str__(self) -> str:  # pragma: no cover - diagnostics
        return f"[{self.rule}] {self.path} ({self.outcome.value}): {self.detail}"


def canonical(field: str) -> str:
    """A loss record's path in the shape :mod:`ir_round_trip` compares by.

    ``events[3].tool_call.arguments`` and the systemic ``events[].tool_call.
    arguments`` are the same path once indices collapse, which is the same
    normalization ``_canonical_leaves`` performs on the trace side.
    """
    return re.sub(r"\[\d+\]", "[]", field)


def declared_classes(path: str, *reports: LossReport | None) -> set[LossClass]:
    """Every loss class declared at *path*, across the edges of one round trip.

    Hub records match directly. Target records match through
    :data:`TARGET_TO_HUB`, which is what lets a synthesis declared honestly in
    the target's vocabulary answer for the hub path it later occupies.
    """
    found: set[LossClass] = set()
    for report in reports:
        if report is None:
            continue
        for record in report.records:
            here = canonical(record.field)
            if record.space is PathSpace.HUB:
                matches = here == path
            elif record.space is PathSpace.TARGET:
                matches = TARGET_TO_HUB.get(here) == path
            else:
                # SOURCE records describe the format a trace came *from*; they
                # say nothing about a hub path and must not answer for one.
                matches = False
            if matches:
                found.add(record.loss_class)
    return found


def vanished_kinds(report: RoundTripReport) -> dict[str, int]:
    """Event kinds that lost instances across the trip, with how many.

    This is the only structural signal :class:`RoundTripReport` carries, and it
    is enough: alignment between individual events is not recoverable after a
    conversion that fuses and drops them, and guessing at it would be the
    invention this family refuses.
    """
    before, after = Counter(report.kinds_before), Counter(report.kinds_after)
    return {
        kind: count - after.get(kind, 0)
        for kind, count in before.items()
        if count > after.get(kind, 0)
    }


def _values_held_by(trace: CanonicalTrace, kinds: set[str], path: str) -> Counter:
    """What the events of *kinds* contribute at *path*, as a multiset.

    Each event is measured on its own so its values are attributed to it rather
    than to the trace, which is what makes the structural rule specific to the
    events that actually vanished.
    """
    held: Counter = Counter()
    for event in trace.events:
        if event.kind.value not in kinds:
            continue
        one = trace.model_copy(update={"events": [event]})
        for value in _values_by_path(one).get(path, []):
            held[repr(value)] += 1
    return held


def structure_explained(
    before: CanonicalTrace,
    after: CanonicalTrace,
    report: RoundTripReport,
    path: str,
) -> bool:
    """True when everything that stopped appearing at *path* left with an event.

    The rule is deliberately narrow. It does not say "some events vanished, so
    changes at this path are fine" — that would excuse an arbitrary edit
    whenever any merge happened. It says: **every value missing from this path
    is a value an event of a vanished kind was holding.** A surviving event
    whose text was rewritten leaves a missing value nobody held, and fails.
    """
    kinds = set(vanished_kinds(report))
    if not kinds:
        return False
    missing = Counter(map(repr, _values_by_path(before).get(path, []))) - Counter(
        map(repr, _values_by_path(after).get(path, []))
    )
    if not missing:
        return True
    return not (missing - _values_held_by(before, kinds, path))


def _appeared(before: CanonicalTrace, after: CanonicalTrace, path: str) -> Counter:
    return Counter(map(repr, _values_by_path(after).get(path, []))) - Counter(
        map(repr, _values_by_path(before).get(path, []))
    )


def conformance_violations(
    before: CanonicalTrace,
    after: CanonicalTrace,
    report: RoundTripReport,
    *reports: LossReport | None,
) -> list[Violation]:
    """Every observed divergence the declared contract does not account for.

    An empty list is the gate passing. *reports* are the loss reports of the
    edges the trip went through, in any order — typically the outbound one the
    converter returned and the inbound one carried on ``after``.

    Rule 2 reads :attr:`FieldComparison.representability`, which
    `ir_round_trip` fills from its ATIF capability table. That column is
    meaningful for the ATIF pair and not for any other; a loop with no ``LOST``
    comparison is unaffected either way.
    """
    violations: list[Violation] = []

    for comparison in report.comparisons:
        path = comparison.path
        declared = declared_classes(path, *reports)
        names = sorted(loss.value for loss in declared)

        if comparison.outcome is RoundTripOutcome.FABRICATED:
            if LossClass.SYNTHESIZED not in declared:
                violations.append(
                    Violation(
                        path=path,
                        outcome=comparison.outcome,
                        rule=UNDECLARED_FABRICATION.name,
                        detail=(
                            f"{comparison.after_count} value(s) came back and none "
                            f"went in; no SYNTHESIZED record addresses this path, "
                            f"directly or through the target bridge"
                        ),
                        declared=names,
                    )
                )
            continue

        if comparison.outcome is RoundTripOutcome.LOST:
            if comparison.representability is Representability.REPRESENTABLE:
                violations.append(
                    Violation(
                        path=path,
                        outcome=comparison.outcome,
                        rule=REPRESENTABLE_LOSS.name,
                        detail=(
                            f"{comparison.before_count} value(s) went in and none "
                            "came back, at a path the target can hold"
                        ),
                        declared=names,
                    )
                )
            continue

        if comparison.outcome is not RoundTripOutcome.TRANSFORMED:
            continue

        # Both sides of a transformation have to be accounted for separately:
        # values leaving and values arriving are different events with different
        # explanations, and one rule covering both would hide the second.
        if not declared and not structure_explained(before, after, report, path):
            violations.append(
                Violation(
                    path=path,
                    outcome=comparison.outcome,
                    rule=UNEXPLAINED_DISAPPEARANCE.name,
                    detail=(
                        "values stopped appearing here, nothing declares the path, "
                        "and they were not held by the events that vanished "
                        f"(kinds: {sorted(vanished_kinds(report)) or 'none'})"
                    ),
                    declared=names,
                )
            )
            continue

        if _appeared(before, after, path) and not declared:
            violations.append(
                Violation(
                    path=path,
                    outcome=comparison.outcome,
                    rule=UNDECLARED_INSERTION.name,
                    detail=(
                        "values appeared at this path that were not in the input, "
                        "and no record declares them"
                    ),
                    declared=names,
                )
            )

    return violations


def conformance_summary(violations: list[Violation]) -> dict[str, int]:
    """Violation counts per rule — for reading a failure at a glance."""
    summary: dict[str, Any] = {}
    for violation in violations:
        summary[violation.rule] = summary.get(violation.rule, 0) + 1
    return summary
