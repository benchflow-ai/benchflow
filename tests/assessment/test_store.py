"""The sidecar log on disk: corruption is refused, never repaired; writers exclude each other.

Corruption controls change one thing in a valid log (a byte, a line, a
field) and require the whole log to be refused, with the file left exactly
as found. Concurrency controls race real processes and threads on one
sidecar: exactly one of several conflicting writes may land, identical
retries land once, and the chain stays valid.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from benchflow import assessment as a
from benchflow.assessment import records as r
from benchflow.assessment import store as store_module
from benchflow.assessment.jcs import canonicalize
from benchflow.assessment.store import EVENTS_FILE, LABEL_FILE, Sidecar, parse_log
from tests.assessment._fixtures import OPERATOR, REVIEWER, world


@pytest.fixture
def labelled(tmp_path: Path):
    """A sidecar with a consumed label: every event kind but correction/withdrawal."""
    w = world(tmp_path)
    w.answered("left", reason="clearer", tags=("hierarchy",))
    a.reduce(w.sidecar)
    a.consume(
        w.sidecar, label_revision=1, consumer_id="ledger-1", consumer_receipt="row-1"
    )
    return w


def _log(w) -> Path:
    return w.sidecar / EVENTS_FILE


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonicalize(rec) + b"\n" for rec in records))


def _rehash(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute every hash after an edit, as a forger would."""
    previous = None
    for record in records:
        record["previous_event_sha256"] = previous
        record["expected_previous_event_sha256"] = previous
        record["event_sha256"] = r.event_digest(record)
        previous = record["event_sha256"]
    return records


def _refused(path: Path, match: str) -> None:
    before = path.read_bytes()
    with pytest.raises(a.CorruptSidecar, match=match):
        Sidecar(path.parent).load()
    assert path.read_bytes() == before


def test_a_torn_last_line_is_refused_and_kept(labelled) -> None:
    path = _log(labelled)
    data = path.read_bytes()
    path.write_bytes(data[:-17])  # an append cut short
    _refused(path, "torn append")
    with pytest.raises(a.CorruptSidecar):
        a.reduce(labelled.sidecar)
    assert path.read_bytes() == data[:-17]


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (
            lambda recs: recs[3]["payload"].update(choice="right"),
            "event_sha256 does not match",
        ),
        (lambda recs: recs.pop(2), "event_seq is 4, expected 3"),
        (lambda recs: recs.insert(3, recs[3]), "event_seq is 4, expected 5"),
        (lambda recs: recs.reverse(), "event_seq"),
        (
            lambda recs: recs[1].update(previous_event_sha256="0" * 64),
            "event_sha256 does not match",
        ),
    ],
    ids=[
        "changed-answer",
        "removed-line",
        "duplicated-line",
        "reordered",
        "broken-link",
    ],
)
def test_edited_lines_break_the_chain(labelled, edit: Callable, match: str) -> None:
    path = _log(labelled)
    records = _records(path)
    edit(records)
    _write(path, records)
    _refused(path, match)


def test_a_rehashed_forgery_still_fails_the_state_machine(labelled) -> None:
    """Recomputed hashes cannot launder a changed answer.

    Changing only the answer breaks its binding to the collector's vote;
    changing the embedded vote too (and its export digest) leaves a label
    that no longer follows from the answer. Only a rewrite of every later
    event is self-consistent, and a receipt's event hash (``verify
    --expect-event``) catches that.
    """
    path = _log(labelled)
    records = _records(path)
    answer = next(rec for rec in records if rec["kind"] == "review.submitted")
    answer["payload"]["choice"] = "right"
    _write(path, _rehash(records))
    _refused(path, "collector vote")
    answer["payload"]["collector_export"]["vote"]["choice"] = "right"
    answer["payload"]["imported_from"]["export_sha256"] = r.jcs_sha256(
        answer["payload"]["collector_export"]
    )
    _write(path, _rehash(records))
    _refused(path, "deterministic reduction")


def test_a_rehashed_assignment_before_release_is_refused(labelled) -> None:
    path = _log(labelled)
    records = _records(path)
    records[1], records[2] = records[2], records[1]
    records[1]["event_seq"], records[2]["event_seq"] = 2, 3
    _write(path, _rehash(records))
    _refused(path, r"compute\.released first")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda line: line.replace(b'{"actor"', b'{"actor":{},"actor"', 1),
            "duplicate member",
        ),
        (lambda line: line.replace(b'"event_seq":1', b'"event_seq": 1'), "canonical"),
        (lambda line: b"\xef\xbb\xbf" + line, "byte order mark"),
        (
            lambda line: line.replace(
                b"benchflow.assessment-event.v1", b"benchflow.assessment-event.v2"
            ),
            "unknown assessment schema version",
        ),
        (
            lambda line: line.replace(
                b'"kind":"assessment.created"', b'"kind":"assessment.made"'
            ),
            "must be one of",
        ),
        (lambda line: line[:-1] + b',"extra":1}', "unknown field extra"),
        (
            lambda line: line.replace(b'"event_seq":1', b'"event_seq":NaN'),
            "not a JSON number",
        ),
        (lambda line: line + b"\r", "not JSON|canonical"),
    ],
    ids=[
        "duplicate-key",
        "whitespace",
        "bom",
        "unknown-version",
        "unknown-kind",
        "unknown-field",
        "nan",
        "crlf",
    ],
)
def test_first_line_defects_refuse_the_whole_log(
    labelled, mutate: Callable, match: str
) -> None:
    path = _log(labelled)
    lines = path.read_bytes().split(b"\n")
    lines[0] = mutate(lines[0])
    path.write_bytes(b"\n".join(lines))
    _refused(path, match)


def test_an_empty_line_and_trailing_garbage_are_refused(labelled) -> None:
    path = _log(labelled)
    data = path.read_bytes()
    path.write_bytes(data + b"\n")
    _refused(path, "empty")
    path.write_bytes(data + b"{}\n")
    _refused(path, "missing")


def test_a_symlinked_log_is_refused(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.create()
    real = tmp_path / "elsewhere.jsonl"
    os.replace(_log(w), real)
    os.symlink(real, _log(w))
    with pytest.raises(a.CorruptSidecar, match="symbolic link"):
        Sidecar(w.sidecar).load()
    with pytest.raises(a.CorruptSidecar, match="symbolic link"):
        w.release()


def test_a_failed_write_leaves_no_bytes(labelled, monkeypatch) -> None:
    """A write that fails midway truncates its own bytes, still under the lock."""
    path = _log(labelled)
    before = path.read_bytes()

    class FullDisk:
        """The store's view of ``os``: a write of an event line stops halfway."""

        def __getattr__(self, name: str):
            return getattr(os, name)

        @staticmethod
        def write(fd: int, data) -> int:
            if len(data) > 100:
                os.write(fd, bytes(data[:50]))
                raise OSError(28, "No space left on device")
            return os.write(fd, data)

    monkeypatch.setattr(store_module, "os", FullDisk())
    target = a.load(labelled.sidecar).effective
    with pytest.raises(OSError, match="No space"):
        a.correct(
            labelled.sidecar,
            supersedes_event_id=target,
            choice="tie",
            reason="",
            correction_reason="x",
            actor=REVIEWER,
        )
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert len(Sidecar(labelled.sidecar).load().events) == len(before.splitlines())


def test_the_label_file_is_derived_and_rebuilds(labelled) -> None:
    sidecar = Sidecar(labelled.sidecar)
    label_path = labelled.sidecar / LABEL_FILE
    expected = canonicalize(sidecar.load().label()) + b"\n"
    assert label_path.read_bytes() == expected
    label_path.write_bytes(b'{"tampered":true}\n')
    assert a.verify(labelled.sidecar)["label_file"] == "stale"
    label_path.unlink()
    assert a.verify(labelled.sidecar)["label_file"] == "missing"
    sidecar.write_label()
    assert label_path.read_bytes() == expected
    assert a.verify(labelled.sidecar)["label_file"] == "current"


def test_verify_detects_a_truncated_log_against_a_receipt(labelled) -> None:
    path = _log(labelled)
    head = Sidecar(labelled.sidecar).load().head_sha256
    assert a.verify(labelled.sidecar, expect_event=head)["chain_verified"]
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))  # a clean truncation is a valid prefix...
    with pytest.raises(a.CorruptSidecar, match="truncated or replaced"):
        a.verify(labelled.sidecar, expect_event=head)  # ...a receipt still catches it


def test_parse_is_a_pure_function_of_the_bytes(labelled) -> None:
    data = _log(labelled).read_bytes()
    first, second = parse_log(data), parse_log(data)
    assert canonicalize(first.label()) == canonicalize(second.label())
    assert [e.line for e in first.events] == [e.line for e in second.events]


# ── Concurrency: real processes and threads on one sidecar ───────────────


def _race_correction(directory: str, choice: str, barrier, results) -> None:
    state = a.load(directory)
    barrier.wait()
    try:
        a.Sidecar(directory).append(
            kind="review.corrected",
            actor=REVIEWER,
            payload={
                "supersedes_event_id": state.effective,
                "assignment_id": state.assignment.payload["collector"]["assignment_id"],
                "reviewer_subject": "held-out-reviewer-01",
                "choice": choice,
                "reason": "",
                "tags": [],
                "correction_reason": f"race {choice}",
                "corrected_at": r.utc_now(),
            },
            idempotency_key=f"race-{choice}",
            expected_event_seq=state.head_seq,
            expected_previous_event_sha256=state.head_sha256,
        )
        results.put(("ok", choice))
    except a.AssessmentError as exc:
        results.put((type(exc).__name__, choice))


def test_racing_processes_land_exactly_one_conflicting_write(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    results = context.Queue()
    choices = ["right", "tie", "abstain", "left"]
    workers = [
        context.Process(
            target=_race_correction, args=(str(w.sidecar), choice, barrier, results)
        )
        for choice in choices
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    outcomes = sorted(results.get(timeout=5) for _ in workers)
    assert [o for o, _ in outcomes].count("ok") == 1
    assert {o for o, _ in outcomes} <= {"ok", "StaleExpectation"}
    state = a.load(w.sidecar)
    assert [e.kind for e in state.events].count("review.corrected") == 1
    winner = next(c for o, c in outcomes if o == "ok")
    assert state.answers[state.effective].choice == winner


def _race_retry(directory: str, barrier, results) -> None:
    barrier.wait()
    reduced = a.reduce(directory)
    results.put(reduced.label["label_id"] + ":" + str(reduced.label["revision"]))


def test_racing_identical_reductions_append_once(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("tie")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    results = context.Queue()
    workers = [
        context.Process(target=_race_retry, args=(str(w.sidecar), barrier, results))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    labels = {results.get(timeout=5) for _ in workers}
    assert len(labels) == 1
    state = a.load(w.sidecar)
    assert [e.kind for e in state.events].count("label.reduced") == 1
    assert state.current.event.payload["value"] == 0.5


def test_racing_threads_keep_one_valid_chain(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    a.reduce(w.sidecar)
    errors: list[BaseException] = []

    def consumer(index: int) -> None:
        try:
            for _ in range(len(threads) + 2):
                try:
                    a.consume(
                        w.sidecar,
                        label_revision=1,
                        consumer_id=f"ledger-{index}",
                        consumer_receipt=f"row-{index}",
                    )
                    return
                except a.StaleExpectation:
                    continue
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=consumer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors
    state = a.load(w.sidecar)
    consumers = sorted(c.payload["consumer_id"] for c in state.current.consumptions)
    assert consumers == sorted(f"ledger-{i}" for i in range(8))
    assert [e.seq for e in state.events] == list(range(1, len(state.events) + 1))


def test_operator_withdrawal_is_monotonic(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.answered("left")
    a.reduce(w.sidecar)
    a.consume(
        w.sidecar, label_revision=1, consumer_id="ledger-1", consumer_receipt="row"
    )
    target = a.load(w.sidecar).effective
    a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="right",
        reason="",
        correction_reason="c",
        actor=REVIEWER,
    )
    a.reduce(w.sidecar)
    queued = a.withdraw(
        w.sidecar,
        label_revision=1,
        reason="superseded",
        disposition="queued_compensation",
        actor=OPERATOR,
    )
    assert queued.assessment.revisions[0].disposition == "queued_compensation"
    done = a.withdraw(
        w.sidecar,
        label_revision=1,
        reason="compensated",
        disposition="compensated",
        actor=OPERATOR,
    )
    assert done.assessment.revisions[0].disposition == "compensated"
    with pytest.raises(a.Conflict, match="only move queued_compensation"):
        a.withdraw(
            w.sidecar,
            label_revision=1,
            reason="undo",
            disposition="not_applicable",
            actor=OPERATOR,
        )
