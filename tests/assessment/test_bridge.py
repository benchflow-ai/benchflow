"""Evidence binding: saved attempts, frozen inputs and the collector's records.

A changed, missing, extra or cross-task file is refused before anything is
written, at creation, at assignment and at vote import. The collector's
assignment must show exactly the two attempts' review artifacts from the
build the frozen protocol pins; anything else stays collector history.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from benchflow import assessment as a
from benchflow.assessment import records as r
from benchflow.assessment.collector import collector_output_hash
from benchflow.assessment.store import EVENTS_FILE
from tests.assessment._fixtures import (
    OTHER_TASK_DIGEST,
    RUBRIC_ID,
    RUBRIC_VERSION,
    collector_connection,
    make_rollout,
    protocol_document,
    world,
    write_json,
)


def _count(w) -> int:
    path = w.sidecar / EVENTS_FILE
    return len(path.read_bytes().splitlines()) if path.exists() else 0


def test_an_attempt_ref_hashes_every_file(tmp_path: Path) -> None:
    w = world(tmp_path)
    ref = w.second.ref(media_type="image/jpeg")
    assert ref["output"]["kind"] == "directory"
    assert [m["path"] for m in ref["output"]["members"]] == [
        "assets/app.css",
        "index.html",
    ]
    assert ref["runtime_refs"] == [
        {
            "provider": "docker",
            "runtime_id": w.second.sandbox_id,
            "role": "solver",
            "source": "result.sandbox_id",
        }
    ]
    assert ref["attempt_id"] == r.attempt_id(
        ref["rollout_id"],
        ref["task_digest"],
        ref["result_sha256"],
        ref["output_manifest_sha256"],
    )
    # Locators are not identity: the same bytes elsewhere are the same attempt.
    moved = make_rollout(
        tmp_path / "moved",
        w.second.root.name,
        files={"index.html": b"<h1>second</h1>", "assets/app.css": b"body{}"},
    )
    moved.root.joinpath("result.json").write_bytes(
        w.second.root.joinpath("result.json").read_bytes()
    )
    assert moved.ref()["attempt_id"] == ref["attempt_id"]


def test_attempt_refs_need_a_finished_rollout_bound_to_a_task(tmp_path: Path) -> None:
    rollout = make_rollout(tmp_path, "t__1")
    result = json.loads((rollout.root / "result.json").read_text())
    for field, value, match in (
        ("finished_at", None, "not a completed rollout"),
        ("task_digest", None, "no task_digest"),
        ("task_digest", "sha256:XYZ", "not 'sha256:"),
        ("rollout_name", "", "names no rollout"),
    ):
        broken = dict(result, **{field: value})
        (rollout.root / "result.json").write_text(json.dumps(broken))
        with pytest.raises(a.EvidenceRejected, match=match):
            rollout.ref()
    (rollout.root / "result.json").write_text(
        '{"rollout_name": "a", "rollout_name": "b"}'
    )
    with pytest.raises(a.EvidenceRejected, match="repeats the member"):
        rollout.ref()


def test_runtimes_must_be_named(tmp_path: Path) -> None:
    rollout = make_rollout(tmp_path, "t__1")
    with pytest.raises(a.InvalidRecord, match="name its provider"):
        a.attempt_ref(rollout.root, rollout.output, rollout.preview)
    ref = rollout.ref(runtimes=[a.parse_runtime_ref("daytona:verifier-77:verifier")])
    assert [x["role"] for x in ref["runtime_refs"]] == ["solver", "verifier"]
    with pytest.raises(a.InvalidRecord, match="twice"):
        rollout.ref(runtimes=[a.parse_runtime_ref(f"docker:{rollout.sandbox_id}")])


def test_symlinks_in_an_output_are_refused(tmp_path: Path) -> None:
    rollout = make_rollout(tmp_path, "t__1")
    (rollout.output / "link.html").symlink_to(rollout.output / "index.html")
    with pytest.raises(a.EvidenceRejected, match="symbolic link"):
        rollout.ref()


def test_cross_task_attempts_are_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    other = make_rollout(
        tmp_path / "other", "other-task__cccc3333", task_digest=OTHER_TASK_DIGEST
    )
    with pytest.raises(a.EvidenceRejected, match="different tasks"):
        a.create(
            w.sidecar,
            [w.first.ref(), other.ref()],
            rubric=w.rubric,
            rubric_id=RUBRIC_ID,
            rubric_version=RUBRIC_VERSION,
            protocol=w.protocol,
        )
    assert not w.sidecar.exists()


@pytest.mark.parametrize("change", ["output", "missing", "extra", "review", "result"])
def test_changed_evidence_is_refused_at_creation(tmp_path: Path, change: str) -> None:
    w = world(tmp_path)
    refs = w.refs()
    if change == "output":
        (w.first.output / "index.html").write_bytes(b"<h1>changed</h1>")
    elif change == "missing":
        (w.second.output / "assets/app.css").unlink()
    elif change == "extra":
        (w.second.output / "late.txt").write_bytes(b"written after the ref")
    elif change == "review":
        w.first.preview.write_bytes(b"another preview")
    else:
        result = json.loads((w.first.root / "result.json").read_text())
        (w.first.root / "result.json").write_text(
            json.dumps(dict(result, rewards={"reward": 1.0}))
        )
    with pytest.raises(a.EvidenceRejected, match="changed"):
        a.create(
            w.sidecar,
            refs,
            rubric=w.rubric,
            rubric_id=RUBRIC_ID,
            rubric_version=RUBRIC_VERSION,
            protocol=w.protocol,
        )
    assert _count(w) == 0


def test_changed_evidence_is_refused_at_assignment_and_import(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.create()
    w.release()
    w.collector_assignment()
    original = w.first.preview.read_bytes()
    w.first.preview.write_bytes(b"swapped preview")
    with pytest.raises(a.EvidenceRejected, match="review artifact changed"):
        w.assign()
    w.first.preview.write_bytes(original)
    w.assign()
    w.vote("left")
    w.rubric.write_bytes(b"# a different rubric\n")
    with pytest.raises(a.EvidenceRejected, match="rubric"):
        w.import_vote()
    assert _count(w) == 3


def test_the_protocol_must_be_a_frozen_protocol_document(tmp_path: Path) -> None:
    w = world(tmp_path)
    for overrides, match in (
        (
            {
                "choices": {
                    "left": "right",
                    "right": "left",
                    "tie": "tie",
                    "skip": "abstain",
                }
            },
            "must map",
        ),
        ({"schema_version": "v0"}, "schema_version"),
        ({"rules": []}, "1 to 64"),
    ):
        write_json(w.protocol, protocol_document(w.collector.code, **overrides))
        with pytest.raises(a.InvalidRecord, match=match):
            w.create()
    assert _count(w) == 0


def _assign_expecting(w, match: str, **assign_kwargs) -> None:
    with pytest.raises(a.EvidenceRejected, match=match):
        w.assign(**assign_kwargs)
    assert _count(w) == 2


@pytest.fixture
def released(tmp_path: Path):
    w = world(tmp_path)
    w.create()
    w.release()
    return w


def test_a_foreign_artifact_is_not_bound(released) -> None:
    w = released
    stranger = make_rollout(
        w.base / "stranger", "design-task__dddd4444", preview=b"JPEG:stranger"
    )
    left = w.collector.add_artifact(w.first)
    right = w.collector.add_artifact(stranger)
    assignment = w.collector.assign(left, right)
    _assign_expecting(
        w, "exactly this assessment's two review artifacts", assignment_id=assignment
    )


def test_the_collector_build_must_be_the_pinned_one(released) -> None:
    w = released
    (w.collector.code / "model.py").write_bytes(b"# patched collector\n")
    w.collector_assignment()
    _assign_expecting(w, "differs from the one the frozen judge protocol pins")


def test_collector_files_must_be_the_attempts_outputs(released) -> None:
    w = released
    left = w.collector.add_artifact(w.first)
    right = w.collector.add_artifact(w.second)
    with collector_connection(w.collector.db) as connection:
        doc = json.loads(
            connection.execute(
                "SELECT doc FROM records WHERE id=?", (right,)
            ).fetchone()[0]
        )
        path = next(iter(doc["fileHashes"]))
        doc["fileHashes"][path] = "f" * 64
        doc["outputHash"] = collector_output_hash(doc["sha256"], doc["fileHashes"])
        connection.execute(
            "UPDATE records SET doc=? WHERE id=?", (json.dumps(doc), right)
        )
    assignment = w.collector.assign(left, right)
    _assign_expecting(w, "not the attempt's output files", assignment_id=assignment)


def test_a_collector_output_hash_must_match_its_own_files(released) -> None:
    w = released
    left = w.collector.add_artifact(w.first)
    right = w.collector.add_artifact(w.second)
    with collector_connection(w.collector.db) as connection:
        doc = json.loads(
            connection.execute(
                "SELECT doc FROM records WHERE id=?", (left,)
            ).fetchone()[0]
        )
        doc["outputHash"] = "0" * 64
        connection.execute(
            "UPDATE records SET doc=? WHERE id=?", (json.dumps(doc), left)
        )
    assignment = w.collector.assign(left, right)
    _assign_expecting(w, "its output hash does not match", assignment_id=assignment)


def test_a_legacy_artifact_with_a_frozen_rubric_binds_by_its_preview(released) -> None:
    """Preview binding does not waive the frozen-rubric requirement."""
    w = released
    task_id = "task-preview"
    rubric_sha = a.load(w.sidecar).created.payload["rubric"]["sha256"]
    w.collector.add_task(task_id, rubric_hash=rubric_sha)
    left = w.collector.add_artifact(w.first, with_files=False, task_id=task_id)
    right = w.collector.add_artifact(w.second, prefix="data/runs/2/", task_id=task_id)
    assignment = w.collector.assign(left, right, task_id=task_id)
    assigned = w.assign(assignment)
    export = assigned.event.payload["collector_export"]
    assert export["artifacts"]["left"]["file_hashes"] is None
    assert export["artifacts"]["right"]["file_hashes"] is not None


def test_an_invalidated_assignment_is_not_bound(released) -> None:
    w = released
    assignment = w.collector_assignment()
    w.collector.invalidate(assignment)
    _assign_expecting(w, "marked invalid")


def test_an_evaluation_task_must_carry_the_rubric_digest(released) -> None:
    w = released
    state = a.load(w.sidecar)
    rubric_sha = state.created.payload["rubric"]["sha256"]
    for rubric_hash, ok in (("e" * 64, False), (rubric_sha, True)):
        task_id = f"task-{rubric_hash[:4]}"
        w.collector.add_task(task_id, rubric_hash=rubric_hash)
        left = w.collector.add_artifact(w.first, task_id=task_id)
        right = w.collector.add_artifact(w.second, task_id=task_id)
        assignment = w.collector.assign(
            left, right, phase="evaluation", task_id=task_id
        )
        if ok:
            assert (
                w.assign(assignment).event.payload["collector_export"]["task"][
                    "rubric_hash"
                ]
                == rubric_sha
            )
        else:
            _assign_expecting(w, "frozen rubric digest", assignment_id=assignment)


def test_a_restarted_collector_must_return_the_same_assignment(released) -> None:
    w = released
    first = w.assign()
    again = w.assign(w.assignment_id)
    assert again.replayed and again.event.id == first.event.id
    other = w.collector.assign(w.artifact_ids["second"], w.artifact_ids["first"])
    with pytest.raises(a.Conflict, match="already has collector assignment"):
        w.assign(other)
    assert _count(w) == 3


def test_a_vote_is_bound_to_its_assignment_and_reviewer(released) -> None:
    w = released
    w.assign()
    with pytest.raises(a.EvidenceRejected, match="no vote"):
        w.import_vote()
    w.vote("right", reviewer="someone-else")
    with pytest.raises(
        a.EvidenceRejected, match="answer names reviewer 'someone-else'"
    ):
        w.import_vote()
    assert _count(w) == 3


def test_a_skip_is_an_abstention(released) -> None:
    w = released
    w.assign()
    w.vote("skip", reason="cannot judge from the preview")
    answered = w.import_vote()
    assert answered.event.payload["choice"] == "abstain"
    assert answered.event.payload["collector_export"]["vote"]["choice"] == "skip"
    reduced = a.reduce(w.sidecar)
    assert reduced.label["status"] == "withheld"
    assert reduced.label["withheld_reason"] == "human_abstention"
    assert "value" not in reduced.label


def test_vote_times_and_tags_come_from_the_collector(released) -> None:
    w = released
    w.assign()
    voted = datetime.now(UTC) + timedelta(seconds=2)
    w.vote("tie", tags=("layout", "color"), created=voted)
    answered = w.import_vote()
    assert answered.event.payload["tags"] == ["color", "layout"]
    assert answered.event.payload["submitted_at"] == r.format_timestamp(voted)
    assert answered.event.payload["imported_from"]["export_sha256"] == r.jcs_sha256(
        answered.event.payload["collector_export"]
    )


def test_the_collector_database_is_opened_read_only(released) -> None:
    w = released
    w.collector_assignment()
    before = w.collector.db.read_bytes()
    w.assign()
    assert w.collector.db.read_bytes() == before
    w.collector.db.chmod(0o444)
    try:
        w.assign()  # an identical retry still reads a read-only database
    finally:
        w.collector.db.chmod(0o644)


def test_a_backup_copy_is_read_without_side_files(released) -> None:
    """The documented path: a copy like the collector's ``cli.py backup`` makes.

    A live WAL database gets SQLite's own -wal/-shm files from any reader; a
    rollback-journal copy is read with nothing created beside it.
    """
    w = released
    assignment = w.collector_assignment()
    backup = w.base / "backups" / "workspace-copy.sqlite3"
    backup.parent.mkdir()
    with (
        collector_connection(w.collector.db) as source,
        closing(sqlite3.connect(backup)) as target,
    ):
        source.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")
    before = sorted(p.name for p in backup.parent.iterdir())
    data = backup.read_bytes()
    a.assign(
        w.sidecar,
        collector_db=backup,
        collector_code=w.collector.code,
        assignment_id=assignment,
    )
    assert (
        sorted(p.name for p in backup.parent.iterdir())
        == before
        == ["workspace-copy.sqlite3"]
    )
    assert backup.read_bytes() == data
