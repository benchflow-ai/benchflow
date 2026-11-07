"""``bench assess``: blind pairwise human assessments of saved outputs.

Each command is one call of :mod:`benchflow.assessment` on a sidecar
directory. Nothing else in ``bench`` reads or writes a sidecar, and nothing
here touches a result, a regrade, a rollout stream or a reward. Refusals
append nothing: exit 3 for a conflict (an idempotency key reused with another
payload, a second answer, a stale expectation), exit 1 for any other refusal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer

if TYPE_CHECKING:
    from benchflow.assessment import Appended, Assessment

ASSESSMENT_NEXT = "`bench assess show <sidecar>` prints the sidecar's state"
# The next step of a refusal whose error names none, by the error's code.
NEXT_STEPS = {
    "assessment.evidence_rejected": (
        "restore the named evidence, or hash the changed attempt again with "
        "`bench assess attempt-ref` and start a new assessment"
    ),
    "assessment.corrupt_sidecar": (
        "keep the sidecar as evidence; `bench assess verify <sidecar>` names the first bad line"
    ),
    "assessment.invalid_record": "fix the named field; `bench assess --help` lists every command's inputs",
}

Sidecar = Annotated[
    Path,
    typer.Argument(
        help="The assessment's sidecar directory, e.g. <completed-rollout>/assessment."
    ),
]
JsonOut = Annotated[
    bool, typer.Option("--json", help="Print JSON instead of one line.")
]
IdempotencyKey = Annotated[
    str | None,
    typer.Option(
        "--idempotency-key",
        help="Key for this write (default: derived from its inputs). A retry with "
        "the same key and payload returns the original event.",
    ),
]
Actor = Annotated[
    str,
    typer.Option(
        "--actor",
        help="Who records this (the reviewer's collector name, or an operator).",
    ),
]


def _refused(exc: Exception) -> NoReturn:
    from benchflow.assessment import Conflict
    from benchflow.cli._shared import print_error

    hint = getattr(exc, "hint", None)
    step = NEXT_STEPS.get(getattr(exc, "code", ""), ASSESSMENT_NEXT)
    print_error(str(exc), next_step=hint or step)
    raise typer.Exit(3 if isinstance(exc, Conflict) else 1)


def _read_json(path: Path, what: str) -> Any:
    from benchflow.assessment import InvalidRecord, loads_strict
    from benchflow.assessment.evidence import read_regular
    from benchflow.assessment.jcs import CanonicalizationError

    try:
        return loads_strict(read_regular(path, what))
    except CanonicalizationError as exc:
        raise InvalidRecord(f"{what} {path}: {exc}") from None


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


def _report(result: Appended, json_out: bool) -> None:
    from benchflow.cli._shared import err_console

    event = result.event
    if json_out:
        _print_json(
            {
                "replayed": result.replayed,
                "state": result.assessment.state,
                "event": event.to_dict(),
            }
        )
    else:
        verb = "already recorded" if result.replayed else "appended"
        typer.echo(
            f"{verb}: {event.kind} #{event.seq} {event.id} "
            f"(state: {result.assessment.state})"
        )
    if result.label_error:
        err_console.print(
            f"note: the event is recorded, but the label file was not rewritten "
            f"({result.label_error}); `bench assess label <sidecar> --write` rebuilds it",
            markup=False,
            highlight=False,
        )


def _label_line(label: dict[str, Any]) -> str:
    status = label["status"]
    if status == "ready":
        value = label["value"]
        meaning = {1: "A preferred", 0: "B preferred", 0.5: "tie"}[value]
        shown = f"{float(value):g} ({meaning})"
    elif status == "withheld":
        shown = f"withheld ({label['withheld_reason']})"
    else:
        shown = status
    return (
        f"label {label['label_id']} revision {label['revision']}: {shown}; "
        f"state {label['state']}; {'resolved' if label['resolved'] else 'consumer disposition required'}"
    )


def _summary_lines(assessment: Assessment, events: bool) -> list[str]:
    summary = assessment.summary()
    lines = [
        f"assessment {summary['assessment_id']}: {summary['state']}",
        f"events {summary['events']}, head #{summary['head_event_seq']} {summary['head_event_sha256']}",
    ]
    if summary["assignment_id"]:
        lines.append(f"collector assignment {summary['assignment_id']}")
    label = assessment.label()
    if label is not None:
        lines.append(_label_line(label))
    if summary["reduction_pending"]:
        lines.append("reduction pending: `bench assess reduce` records it")
    if events:
        lines.extend(
            f"  #{e.seq} {e.kind} {e.id} {e.sha256}" for e in assessment.events
        )
    return lines


def register_assessment(app: typer.Typer) -> None:
    group = typer.Typer(
        help=(
            "Blind pairwise human assessments of saved outputs, kept in an opt-in sidecar.\n\n"
            "Two completed attempts of one task, a frozen rubric and judge protocol, a "
            "recorded release of every runtime, the review collector's blind assignment "
            "and vote, then a deterministic label: 1 (A preferred), 0 (B preferred), 0.5 "
            "(tie) or withheld (abstention). Rewards, result.json, regrade.json and "
            "rollout streams are never read or changed. See docs/jobs/assessment.md."
        ),
        no_args_is_help=True,
    )
    app.add_typer(group, name="assess")

    @group.command("attempt-ref")
    def attempt_ref_cmd(
        rollout: Annotated[
            Path, typer.Argument(help="A completed rollout folder (holds result.json).")
        ],
        output: Annotated[
            Path,
            typer.Option(
                "--output", help="The output to compare: a folder or one file."
            ),
        ],
        review_artifact: Annotated[
            Path,
            typer.Option(
                "--review-artifact",
                help="The file a reviewer sees (e.g. a preview image).",
            ),
        ],
        sandbox_provider: Annotated[
            str | None,
            typer.Option(
                "--sandbox-provider",
                help="Provider of result.json's sandbox_id (daytona, docker, modal, ...).",
            ),
        ] = None,
        sandbox_role: Annotated[
            str,
            typer.Option(
                "--sandbox-role",
                help="solver, verifier or shared (one sandbox for both).",
            ),
        ] = "solver",
        runtime: Annotated[
            list[str] | None,
            typer.Option(
                "--runtime",
                help="Another allocation of this attempt: PROVIDER:RUNTIME_ID[:ROLE]. Repeatable.",
            ),
        ] = None,
        media_type: Annotated[
            str | None,
            typer.Option("--media-type", help="The review artifact's media type."),
        ] = None,
        out: Annotated[
            Path | None,
            typer.Option(
                "--out", help="Write the reference here (a new file) instead of stdout."
            ),
        ] = None,
    ) -> None:
        """Hash a completed rollout into a saved-attempt reference (JSON)."""
        from benchflow.assessment import AssessmentError, attempt_ref, parse_runtime_ref

        try:
            ref = attempt_ref(
                rollout,
                output,
                review_artifact,
                sandbox_provider=sandbox_provider,
                sandbox_role=sandbox_role,
                runtimes=[parse_runtime_ref(value) for value in runtime or []],
                media_type=media_type,
            )
        except AssessmentError as exc:
            _refused(exc)
        text = json.dumps(ref, indent=2, ensure_ascii=False) + "\n"
        if out is None:
            typer.echo(text, nl=False)
            return
        try:
            fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            from benchflow.cli._shared import print_error

            print_error(
                f"{out} already exists; a reference is never overwritten",
                next_step="pass a new --out path",
            )
            raise typer.Exit(1) from None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        typer.echo(f"wrote {out} (attempt {ref['attempt_id']})")

    @group.command("create")
    def create_cmd(
        sidecar: Sidecar,
        attempt: Annotated[
            list[Path],
            typer.Option(
                "--attempt", help="A saved-attempt reference file. Give exactly two."
            ),
        ],
        rubric: Annotated[
            Path, typer.Option("--rubric", help="The frozen rubric file.")
        ],
        rubric_id: Annotated[str, typer.Option("--rubric-id", help="The rubric's id.")],
        rubric_version: Annotated[
            str, typer.Option("--rubric-version", help="The rubric's frozen version.")
        ],
        protocol: Annotated[
            Path,
            typer.Option(
                "--protocol",
                help="The frozen judge protocol (benchflow.assessment-judge-protocol.v1).",
            ),
        ],
        idempotency_key: IdempotencyKey = None,
        json_out: JsonOut = False,
    ) -> None:
        """Start an assessment of two saved attempts of one task."""
        from benchflow.assessment import AssessmentError, create

        try:
            refs = [_read_json(path, "attempt reference") for path in attempt]
            result = create(
                sidecar,
                refs,
                rubric=rubric,
                rubric_id=rubric_id,
                rubric_version=rubric_version,
                protocol=protocol,
                idempotency_key=idempotency_key,
            )
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("release")
    def release_cmd(
        sidecar: Sidecar,
        proof: Annotated[
            Path,
            typer.Option(
                "--proof",
                help="Observations that every runtime is gone "
                "(benchflow.assessment-release-proof.v1).",
            ),
        ],
        idempotency_key: IdempotencyKey = None,
        json_out: JsonOut = False,
    ) -> None:
        """Record that every runtime of both attempts was observed released."""
        from benchflow.assessment import AssessmentError, release

        try:
            result = release(sidecar, proof, idempotency_key=idempotency_key)
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("assign")
    def assign_cmd(
        sidecar: Sidecar,
        collector_db: Annotated[
            Path,
            typer.Option(
                "--collector-db",
                help="The collector's workspace.sqlite3 (a `cli.py backup` copy), read-only.",
            ),
        ],
        collector_code: Annotated[
            Path,
            typer.Option(
                "--collector-code",
                help="The collector's code folder (design-bench/prototype).",
            ),
        ],
        assignment: Annotated[
            str, typer.Option("--assignment", help="The collector's assignment id.")
        ],
        idempotency_key: IdempotencyKey = None,
        json_out: JsonOut = False,
    ) -> None:
        """Bind the collector's persisted blind assignment (after the release)."""
        from benchflow.assessment import AssessmentError, assign

        try:
            result = assign(
                sidecar,
                collector_db=collector_db,
                collector_code=collector_code,
                assignment_id=assignment,
                idempotency_key=idempotency_key,
            )
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("import-vote")
    def import_vote_cmd(
        sidecar: Sidecar,
        collector_db: Annotated[
            Path,
            typer.Option(
                "--collector-db", help="The collector's workspace.sqlite3, read-only."
            ),
        ],
        collector_code: Annotated[
            Path, typer.Option("--collector-code", help="The collector's code folder.")
        ],
        json_out: JsonOut = False,
    ) -> None:
        """Bind the collector's vote on the assignment (a skip becomes abstain)."""
        from benchflow.assessment import AssessmentError, import_vote

        try:
            result = import_vote(
                sidecar, collector_db=collector_db, collector_code=collector_code
            )
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("correct")
    def correct_cmd(
        sidecar: Sidecar,
        supersedes: Annotated[
            str,
            typer.Option(
                "--supersedes", help="The current effective answer's event id."
            ),
        ],
        choice: Annotated[
            str, typer.Option("--choice", help="left (A), right (B), tie or abstain.")
        ],
        reason: Annotated[
            str, typer.Option("--reason", help="The new answer's rationale.")
        ],
        correction_reason: Annotated[
            str, typer.Option("--correction-reason", help="Why the answer changed.")
        ],
        actor: Actor,
        actor_role: Annotated[
            str,
            typer.Option(
                "--actor-role", help="reviewer (the answer's own) or operator."
            ),
        ] = "reviewer",
        tag: Annotated[
            list[str] | None, typer.Option("--tag", help="A feedback tag. Repeatable.")
        ] = None,
        idempotency_key: IdempotencyKey = None,
        json_out: JsonOut = False,
    ) -> None:
        """Replace the effective answer with a corrected one."""
        from benchflow.assessment import AssessmentError, correct

        try:
            result = correct(
                sidecar,
                supersedes_event_id=supersedes,
                choice=choice,
                reason=reason,
                correction_reason=correction_reason,
                tags=tag or [],
                actor={"role": actor_role, "subject": actor},
                idempotency_key=idempotency_key,
            )
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("retract")
    def retract_cmd(
        sidecar: Sidecar,
        target: Annotated[
            str,
            typer.Option("--target", help="The current effective answer's event id."),
        ],
        reason: Annotated[str, typer.Option("--reason", help="Why it is retracted.")],
        actor: Actor,
        actor_role: Annotated[
            str,
            typer.Option(
                "--actor-role", help="reviewer (the answer's own) or operator."
            ),
        ] = "reviewer",
        idempotency_key: IdempotencyKey = None,
        json_out: JsonOut = False,
    ) -> None:
        """Retract the effective answer; its history stays in the log."""
        from benchflow.assessment import AssessmentError, retract

        try:
            result = retract(
                sidecar,
                target_event_id=target,
                reason=reason,
                actor={"role": actor_role, "subject": actor},
                idempotency_key=idempotency_key,
            )
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("reduce")
    def reduce_cmd(sidecar: Sidecar, json_out: JsonOut = False) -> None:
        """Record the deterministic label the log calls for, and refresh the label file."""
        from benchflow.assessment import AssessmentError, reduce

        try:
            reduced = reduce(sidecar)
        except AssessmentError as exc:
            _refused(exc)
        if json_out:
            _print_json(reduced.label)
            return
        verb = "appended" if reduced.event is not None else "up to date"
        typer.echo(f"{verb}: {_label_line(reduced.label)}")

    @group.command("consume")
    def consume_cmd(
        sidecar: Sidecar,
        revision: Annotated[
            int, typer.Option("--revision", help="The label revision used.")
        ],
        consumer: Annotated[str, typer.Option("--consumer", help="The consumer's id.")],
        receipt: Annotated[
            str,
            typer.Option(
                "--receipt",
                help="The consumer's immutable receipt (e.g. a ledger row).",
            ),
        ],
        idempotency_key: IdempotencyKey = None,
        json_out: JsonOut = False,
    ) -> None:
        """Record that a consumer used one exact ready label revision."""
        from benchflow.assessment import AssessmentError, consume

        try:
            result = consume(
                sidecar,
                label_revision=revision,
                consumer_id=consumer,
                consumer_receipt=receipt,
                idempotency_key=idempotency_key,
            )
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("withdraw")
    def withdraw_cmd(
        sidecar: Sidecar,
        revision: Annotated[
            int, typer.Option("--revision", help="The superseded label revision.")
        ],
        reason: Annotated[str, typer.Option("--reason", help="Why it is withdrawn.")],
        disposition: Annotated[
            str,
            typer.Option(
                "--disposition",
                help="not_applicable, queued_compensation or compensated: what its consumers do.",
            ),
        ],
        actor: Actor,
        actor_role: Annotated[
            str, typer.Option("--actor-role", help="operator or reducer.")
        ] = "operator",
        idempotency_key: IdempotencyKey = None,
        json_out: JsonOut = False,
    ) -> None:
        """Bar a corrected or retracted label revision from future use."""
        from benchflow.assessment import AssessmentError, withdraw

        try:
            result = withdraw(
                sidecar,
                label_revision=revision,
                reason=reason,
                disposition=disposition,
                actor={"role": actor_role, "subject": actor},
                idempotency_key=idempotency_key,
            )
        except AssessmentError as exc:
            _refused(exc)
        _report(result, json_out)

    @group.command("show")
    def show_cmd(
        sidecar: Sidecar,
        events: Annotated[
            bool, typer.Option("--events", help="List every event.")
        ] = False,
        json_out: JsonOut = False,
    ) -> None:
        """Verify the sidecar's log and print its state."""
        from benchflow.assessment import AssessmentError, load

        try:
            assessment = load(sidecar)
        except AssessmentError as exc:
            _refused(exc)
        if json_out:
            document: dict[str, Any] = {
                "summary": assessment.summary(),
                "label": assessment.label(),
            }
            if events:
                document["events"] = [event.to_dict() for event in assessment.events]
            _print_json(document)
            return
        for line in _summary_lines(assessment, events):
            typer.echo(line)

    @group.command("verify")
    def verify_cmd(
        sidecar: Sidecar,
        evidence: Annotated[
            bool,
            typer.Option(
                "--evidence",
                help="Also hash both attempts, the rubric and the protocol again.",
            ),
        ] = False,
        expect_event: Annotated[
            str | None,
            typer.Option(
                "--expect-event",
                help="An event SHA-256 a receipt holds; the log must still contain it.",
            ),
        ] = None,
        json_out: JsonOut = False,
    ) -> None:
        """Verify the hash chain, the label file and, optionally, the evidence."""
        from benchflow.assessment import AssessmentError, verify

        try:
            report = verify(sidecar, evidence=evidence, expect_event=expect_event)
        except AssessmentError as exc:
            _refused(exc)
        if json_out:
            _print_json(report)
            return
        typer.echo(
            f"verified: {report['events']} events, head {report['head_event_sha256']}; "
            f"state {report['state']}; label file {report['label_file']}"
            + ("; evidence unchanged" if report["evidence_verified"] else "")
        )

    @group.command("label")
    def label_cmd(
        sidecar: Sidecar,
        write: Annotated[
            bool, typer.Option("--write", help="Rewrite the label file from the log.")
        ] = False,
        check: Annotated[
            bool,
            typer.Option(
                "--check", help="Exit 1 unless the label file equals the log's label."
            ),
        ] = False,
    ) -> None:
        """Print the label rebuilt from the sidecar's log alone (JSON)."""
        from benchflow.assessment import AssessmentError, Sidecar, canonicalize, load
        from benchflow.cli._shared import print_error

        try:
            assessment = load(sidecar)
        except AssessmentError as exc:
            _refused(exc)
        label = assessment.label()
        if label is None:
            print_error(
                f"{sidecar} has no label yet (state {assessment.state})",
                next_step="`bench assess reduce <sidecar>` records one once there is an answer",
            )
            raise typer.Exit(1)
        if check and Sidecar(sidecar).read_label_file() != canonicalize(label) + b"\n":
            print_error(
                f"{sidecar}'s label file differs from the label its log rebuilds",
                next_step="`bench assess label <sidecar> --write` rewrites it from the log",
            )
            raise typer.Exit(1)
        if write:
            Sidecar(sidecar).write_label()
        _print_json(label)
