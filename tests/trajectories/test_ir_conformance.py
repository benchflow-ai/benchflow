"""The conformance gate: every observed divergence is accounted for, and the
gate still fails when it should.

Half of this file is the gate passing on real rollouts. The other half is the
part that gives the first half its meaning — each rule is shown *failing* on an
input built to break exactly it. A gate that cannot be made to fail proves
nothing about the runs where it passes.

No corpus is built here. The rollouts are the ones already captured for the
`ATIF` human verification, and the rich fixture is Slice E's.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from benchflow.trajectories.ir import LossClass, LossReport, PathSpace, Role
from benchflow.trajectories.ir_conformance import (
    REPRESENTABLE_LOSS,
    TARGET_TO_HUB,
    UNDECLARED_FABRICATION,
    UNDECLARED_INSERTION,
    UNEXPLAINED_DISAPPEARANCE,
    canonical,
    conformance_summary,
    conformance_violations,
    declared_classes,
    structure_explained,
    vanished_kinds,
)
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_round_trip import (
    Representability,
    RoundTripOutcome,
    _values_by_path,
    compare_traces,
    round_trip_through_atif,
)
from benchflow.trajectories.ir_to_acp import ir_to_acp_capture_events
from tests.trajectories.test_atif_preservation import _rich_events

EVIDENCE = pathlib.Path(__file__).resolve().parents[2].parent / "e2e-a2" / "evidence"


def _rollout(name: str) -> list[dict[str, Any]] | None:
    """A captured `ACP` rollout, or ``None`` when the evidence tree is absent.

    The captures live outside the clone, next to the human-verification
    scaffolding. The suite must stay green without them, so every test that
    wants one skips rather than fails — and :func:`_rich_events` covers the
    same rules from inside the repo, so a skip never leaves a rule untested.
    """
    path = EVIDENCE / name / "acp_trajectory.jsonl"
    if not path.is_file():
        return None
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gate(events: list[dict[str, Any]]):
    """Run the `ATIF` loop and join the measurement to the declarations."""
    trip = round_trip_through_atif(events, session_id="s", agent_name="a")
    return trip, conformance_violations(
        trip.before, trip.after, trip.report, trip.outbound, trip.inbound
    )


def _captured(name: str):
    events = _rollout(name)
    if events is None:
        pytest.skip(f"captured rollout {name!r} is not in this tree")
    return _gate(events)


# --------------------------------------------------------------------------
# the gate passing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h1", "h2"])
def test_a_captured_rollout_stays_inside_its_declared_contract(name):
    """The whole point: on a real run, D is covered by L.

    Nothing here asserts *how many* divergences there are. The count is a
    property of the rollout and of every converter it passes through, and
    pinning it would turn an honest improvement into a test failure.
    """
    trip, violations = _captured(name)
    assert violations == [], "\n".join(str(v) for v in violations)
    # ...and the run has to be non-trivial, or the assertion above is vacuous.
    assert any(
        c.outcome is not RoundTripOutcome.PRESERVED for c in trip.report.comparisons
    )


def test_the_rich_fixture_stays_inside_its_declared_contract():
    """Same gate, on the in-repo fixture — so it holds with no evidence tree."""
    _, violations = _gate(_rich_events())
    assert violations == [], "\n".join(str(v) for v in violations)


def test_the_acp_loop_is_a_regression_guard():
    """`IR → ACP → IR′` declares what it changes too.

    Slice G's loop is included as a guard, not as a second contract: the
    representability column travels from the ATIF table and is not about ACP,
    so only the rules that do not consult it are meaningful here.
    """
    before = acp_events_to_ir(_rich_events())
    events, outbound = ir_to_acp_capture_events(before)
    after = acp_events_to_ir(events)
    report = compare_traces(before, after)
    violations = [
        v
        for v in conformance_violations(before, after, report, outbound, after.losses)
        if v.rule != REPRESENTABLE_LOSS.name
    ]
    assert violations == [], "\n".join(str(v) for v in violations)


def test_every_bridge_entry_is_real():
    """Each `TARGET_TO_HUB` entry is checked against values, not asserted.

    A bridge that drifted out of step with `ir_from_atif` would silently
    forgive fabrications at a path nobody declares, which is the one failure
    mode the map could introduce. So: the target value the document carries has
    to actually turn up at the hub path it claims to reach.
    """
    trip, _ = _gate(_rich_events())
    hub = _values_by_path(trip.after)

    assert trip.document["schema_version"] in hub["extensions.schema_version"]
    step_ids = {step["step_id"] for step in trip.document["steps"]}
    assert step_ids & set(hub["events[].extensions.step_id"])
    if any(step.get("message") == "" for step in trip.document["steps"]):
        assert "" in hub["events[].text"]

    assert set(TARGET_TO_HUB) == {
        "schema_version",
        "steps[].step_id",
        "steps[].message",
    }, "a new bridge entry needs a check above before it can be trusted"


def test_the_two_structural_values_are_declared_and_not_exempted():
    """`schema_version` and `step_id` pass the gate the ordinary way.

    They are fabrications — the round trip really does produce values the input
    never had — and they clear rule 1 only because `ir_to_atif` declares them
    ``SYNTHESIZED``. There is no branch in the gate that knows their names.
    """
    trip, _ = _gate(_rich_events())
    fabricated = {
        c.path
        for c in trip.report.comparisons
        if c.outcome is RoundTripOutcome.FABRICATED
    }
    assert {"extensions.schema_version", "events[].extensions.step_id"} <= fabricated

    for path in ("extensions.schema_version", "events[].extensions.step_id"):
        assert LossClass.SYNTHESIZED in declared_classes(
            path, trip.outbound, trip.inbound
        )


# --------------------------------------------------------------------------
# the gate failing — one input per rule
# --------------------------------------------------------------------------


def _without(report: LossReport, field: str) -> LossReport:
    """The same report with every record at *field* removed."""
    return report.model_copy(
        update={"records": [r for r in report.records if r.field != field]}
    )


def test_removing_a_synthesized_declaration_fails_the_gate():
    """Rule 1, on the exact path the user asked not to exempt.

    Drop `ir_to_atif`'s ``schema_version`` record and the fabrication it
    covered has nothing left to stand on — which is what makes the passing case
    above evidence of a declaration rather than of an allowlist.
    """
    trip, clean = _gate(_rich_events())
    assert clean == []

    violations = conformance_violations(
        trip.before,
        trip.after,
        trip.report,
        _without(trip.outbound, "schema_version"),
        trip.inbound,
    )
    assert [v.path for v in violations] == ["extensions.schema_version"]
    assert violations[0].rule == UNDECLARED_FABRICATION.name


def test_removing_the_step_id_declaration_fails_the_gate():
    """Rule 1 again, for the second structural value."""
    trip, _ = _gate(_rich_events())
    violations = conformance_violations(
        trip.before,
        trip.after,
        trip.report,
        _without(trip.outbound, "steps[].step_id"),
        trip.inbound,
    )
    assert [v.path for v in violations] == ["events[].extensions.step_id"]
    assert violations[0].rule == UNDECLARED_FABRICATION.name


def test_a_declaration_of_the_wrong_class_does_not_excuse_a_fabrication():
    """Rule 1 asks for ``SYNTHESIZED`` specifically, not for any record at all.

    ``NORMALIZED`` says a source value was reshaped. A value that was never in
    the input has no source value to reshape, so a record of that class at the
    path is not an account of the fabrication — it is a different claim that
    happens to share an address.

    Found by mutating the rule into "is anything declared here", which the
    suite did not notice until this test existed.
    """
    trip, _ = _gate(_rich_events())
    mislabelled = LossReport(direction=trip.outbound.direction)
    for record in trip.outbound.records:
        mislabelled.add(
            record.field,
            LossClass.NORMALIZED
            if record.field == "schema_version"
            else record.loss_class,
            record.detail,
            space=record.space,
        )

    assert declared_classes("extensions.schema_version", mislabelled) == {
        LossClass.NORMALIZED
    }, "the path must still be declared, or this tests an empty lookup"

    violations = conformance_violations(
        trip.before, trip.after, trip.report, mislabelled, trip.inbound
    )
    assert ("extensions.schema_version", UNDECLARED_FABRICATION.name) in {
        (v.path, v.rule) for v in violations
    }


def test_a_fabrication_at_a_path_nobody_declares_fails_the_gate():
    """Rule 1, for a value no edge has ever heard of.

    An undeclared field appears in the reconstructed trace, and the gate says
    so without needing to know what it means.
    """
    trip, _ = _gate(_rich_events())
    after = trip.after.model_copy(
        update={"extensions": {**trip.after.extensions, "invented": "no-one-said-so"}}
    )
    report = compare_traces(trip.before, after)
    violations = conformance_violations(
        trip.before, after, report, trip.outbound, trip.inbound
    )
    assert [v.path for v in violations if v.rule == UNDECLARED_FABRICATION.name] == [
        "extensions.invented"
    ]


def test_editing_a_surviving_event_fails_the_gate():
    """Rule 3, and the reason ``structure_explained`` is narrow.

    Some events genuinely vanish in this loop, so a rule of the form "changes
    are fine when a merge happened" would wave this through. The value edited
    here belongs to an event that *survived*, so no vanished event was holding
    it, and the gate catches it.
    """
    trip, clean = _gate(_rich_events())
    assert clean == []

    survivors = list(trip.after.events)
    edited = survivors[0].model_copy(update={"role": Role.ORACLE})
    after = trip.after.model_copy(update={"events": [edited, *survivors[1:]]})

    report = compare_traces(trip.before, after)
    violations = conformance_violations(
        trip.before, after, report, trip.outbound, trip.inbound
    )
    assert UNEXPLAINED_DISAPPEARANCE.name in {v.rule for v in violations}
    assert "events[].role" in {v.path for v in violations}


def test_losing_a_representable_field_fails_the_gate():
    """Rule 2: a gap in our own edge, not a cost of the format."""
    trip, _ = _gate(_rich_events())
    stripped = [e.model_copy(update={"text": None}) for e in trip.after.events]
    after = trip.after.model_copy(update={"events": stripped})

    report = compare_traces(trip.before, after)
    assert any(
        c.path == "events[].text"
        and c.outcome is RoundTripOutcome.LOST
        and c.representability is Representability.REPRESENTABLE
        for c in report.comparisons
    )
    violations = conformance_violations(
        trip.before, after, report, trip.outbound, trip.inbound
    )
    assert ("events[].text", REPRESENTABLE_LOSS.name) in {
        (v.path, v.rule) for v in violations
    }


def test_an_undeclared_insertion_at_a_transformed_path_fails_the_gate():
    """Rule 3's second half — the side a one-sided rule would miss.

    ``TRANSFORMED`` covers both values leaving and values arriving. Here the
    loss side is structure-explained, and a value is *added* on top; the gate
    has to object to the addition on its own.
    """
    trip, _ = _gate(_rich_events())
    events = list(trip.after.events)
    kinds = set(vanished_kinds(trip.report))
    assert kinds, "this fixture must lose at least one event kind"

    # `role` is carried by structure alone, so an insertion is the only fault.
    extra = events[0].model_copy(update={"index": 999, "role": Role.ORACLE})
    after = trip.after.model_copy(update={"events": [*events, extra]})
    report = compare_traces(trip.before, after)
    violations = conformance_violations(
        trip.before, after, report, trip.outbound, trip.inbound
    )
    assert ("events[].role", UNDECLARED_INSERTION.name) in {
        (v.path, v.rule) for v in violations
    }


def test_the_gate_is_not_satisfied_by_the_measurement_alone():
    """With no declarations at all, a real run is full of violations.

    This is the negative control for the whole join: the observation half does
    not justify itself, and every clean result above is the *pair* agreeing.
    """
    trip, _ = _gate(_rich_events())
    violations = conformance_violations(trip.before, trip.after, trip.report)
    assert len(violations) >= 3
    assert UNDECLARED_FABRICATION.name in conformance_summary(violations)


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------


def test_structure_explained_needs_a_vanished_kind():
    """No event lost, no structural excuse — even for a real transformation."""
    trip, _ = _gate(_rich_events())
    report = trip.report.model_copy(
        update={"kinds_before": dict(trip.report.kinds_after)}
    )
    assert not structure_explained(trip.before, trip.after, report, "events[].kind")


def test_structure_explained_does_not_cover_a_value_nobody_held():
    """The rule is per value, not per path.

    A kind vanishing does not license *any* change at a path the vanished
    events touched — only the disappearance of the values they were holding.
    """
    trip, _ = _gate(_rich_events())
    assert structure_explained(trip.before, trip.after, trip.report, "events[].kind")

    survivors = list(trip.after.events)
    edited = survivors[0].model_copy(update={"role": Role.ORACLE})
    after = trip.after.model_copy(update={"events": [edited, *survivors[1:]]})
    assert not structure_explained(trip.before, after, trip.report, "events[].role")


def test_canonical_collapses_indices_the_way_the_measurement_does():
    assert canonical("events[3].tool_call.arguments") == (
        "events[].tool_call.arguments"
    )
    assert canonical("events[].text") == "events[].text"
    assert canonical("extensions.schema_version") == "extensions.schema_version"


def test_a_target_record_only_answers_for_the_path_it_bridges_to():
    """The bridge is a map, not a wildcard over the target space."""
    report = LossReport(direction="ir->atif")
    report.add(
        "steps[].step_id", LossClass.SYNTHESIZED, "structural", space=PathSpace.TARGET
    )
    assert declared_classes("events[].extensions.step_id", report) == {
        LossClass.SYNTHESIZED
    }
    assert declared_classes("events[].index", report) == set()
    assert declared_classes("steps[].step_id", report) == set()


def test_a_hub_record_answers_only_at_its_own_path():
    report = LossReport(direction="ir->atif")
    report.add("events[].text", LossClass.NORMALIZED, "reshaped", space=PathSpace.HUB)
    assert declared_classes("events[].text", report) == {LossClass.NORMALIZED}
    assert declared_classes("steps[].message", report) == set()


def test_vanished_kinds_reports_only_net_losses():
    trip, _ = _gate(_rich_events())
    vanished = vanished_kinds(trip.report)
    assert vanished
    for kind, count in vanished.items():
        assert count == trip.report.kinds_before[kind] - trip.report.kinds_after.get(
            kind, 0
        )
        assert count > 0


def test_the_gate_does_not_mutate_what_it_reads():
    """It is a join over two artefacts, and it leaves both as it found them."""
    trip, _ = _gate(_rich_events())
    snapshot = (
        trip.before.model_dump_json(),
        trip.after.model_dump_json(),
        trip.report.model_dump_json(),
        trip.outbound.model_dump_json(),
    )
    conformance_violations(
        trip.before, trip.after, trip.report, trip.outbound, trip.inbound
    )
    assert snapshot == (
        trip.before.model_dump_json(),
        trip.after.model_dump_json(),
        trip.report.model_dump_json(),
        trip.outbound.model_dump_json(),
    )


def test_a_missing_report_is_tolerated_not_treated_as_a_declaration():
    """``None`` in the reports is absence of evidence, and it must not pass."""
    trip, _ = _gate(_rich_events())
    violations = conformance_violations(
        trip.before, trip.after, trip.report, None, None
    )
    assert violations, "no declarations must not read as everything declared"
