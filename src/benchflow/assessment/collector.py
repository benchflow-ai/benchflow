"""The bridge to the review collector: export one record, then bind it.

The collector (``design-bench/prototype`` in the lab repository) persists
blind pairwise assignments and votes in its ``workspace.sqlite3``. Its blind
review payload deliberately omits hashes, so the bridge reads the collector's
own rows instead, read-only, and never writes the collector's database:

* ``assignments``: id, case, phase, reviewer, left/right artifact ids and the
  preview SHA-256 each side showed, task version, creation time;
* ``votes``: the one vote of an assignment (choice ``left``/``right``/
  ``tie``/``skip``, reason, tags, creation time);
* ``records``: the two artifacts (preview SHA-256, file manifest and output
  hash when imported through the collector's output bridge), the task
  version, and any ``invalid-assignment`` mark.

:func:`export_record` turns those rows into a
``benchflow.assessment-collector-export.v1`` document, adding the SHA-256 of
each collector source file the frozen judge protocol pins. The sidecar embeds
the export and its JCS digest, so a later reader can check the binding from
the sidecar alone. :func:`check_assignment_binding` and
:func:`check_vote_binding` are those checks; the state machine runs them on
append and on every replay.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from benchflow.assessment import records as r
from benchflow.assessment.errors import EvidenceRejected

if TYPE_CHECKING:
    from benchflow.assessment.state import Assessment, Event

__all__ = [
    "check_assignment_binding",
    "check_vote_binding",
    "collector_output_hash",
    "export_record",
    "hash_sources",
]


def hash_sources(code_dir: str | Path, pin: Mapping[str, Any]) -> dict[str, str]:
    """SHA-256 of each source file ``pin["sources"]`` names, under ``code_dir``."""
    root = Path(code_dir)
    hashes: dict[str, str] = {}
    for relative in sorted(pin["sources"], key=r.utf8_sort_key):
        path = root / relative
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise EvidenceRejected(
                f"collector source {relative} is missing under {root}"
            ) from None
        if not stat.S_ISREG(info.st_mode):
            raise EvidenceRejected(f"collector source {relative} is not a regular file")
        with path.open("rb") as handle:
            hashes[relative] = hashlib.file_digest(handle, "sha256").hexdigest()
    return hashes


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise EvidenceRejected(f"collector database {path} does not exist")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=1")
    return connection


def _doc(
    connection: sqlite3.Connection, kind: str, record_id: str
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT doc FROM records WHERE kind=? AND id=?", (kind, record_id)
    ).fetchone()
    if row is None:
        return None
    try:
        document = json.loads(row["doc"])
    except ValueError:
        raise EvidenceRejected(
            f"collector {kind} {record_id} is not readable JSON"
        ) from None
    if not isinstance(document, dict):
        raise EvidenceRejected(f"collector {kind} {record_id} is not a JSON object")
    return document


def export_record(
    db_path: str | Path,
    code_dir: str | Path,
    assignment_id: str,
    *,
    pin: Mapping[str, Any],
    record: str,
) -> dict[str, Any]:
    """Read one assignment (``record="assignment"``) or its vote (``"vote"``).

    Opens the collector database read-only (``mode=ro``, ``query_only``);
    prefer a copy made by the collector's own ``cli.py backup``. The export
    names the implementation and commit the protocol pins and the SHA-256 of
    each pinned source file found under ``code_dir``; the binding check, not
    this reader, decides whether they match.
    """
    if record not in ("assignment", "vote"):
        raise ValueError(f"record must be assignment or vote, not {record!r}")
    sources = hash_sources(code_dir, pin)
    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM assignments WHERE id=?", (assignment_id,)
        ).fetchone()
        if row is None:
            raise EvidenceRejected(f"the collector has no assignment {assignment_id!r}")
        assignment = {
            key: row[key]
            for key in (
                "id",
                "case_id",
                "phase",
                "reviewer",
                "left_id",
                "right_id",
                "left_hash",
                "right_hash",
                "task_id",
                "created",
            )
        }
        artifacts = {}
        for side in ("left", "right"):
            document = _doc(connection, "artifact", row[f"{side}_id"])
            if document is None:
                raise EvidenceRejected(
                    f"the collector has no artifact {row[f'{side}_id']!r}"
                )
            artifacts[side] = {
                "id": document.get("id"),
                "case_id": document.get("caseId"),
                "sha256": document.get("sha256"),
                "task_id": document.get("taskId"),
                "output_hash": document.get("outputHash"),
                "file_hashes": document.get("fileHashes"),
            }
        task = None
        if row["task_id"] is not None:
            document = _doc(connection, "task", row["task_id"])
            if document is None:
                raise EvidenceRejected(f"the collector has no task {row['task_id']!r}")
            task = {
                "id": document.get("id"),
                "version": document.get("version"),
                "case_id": document.get("caseId"),
                "rubric_id": document.get("rubricId"),
                "rubric_hash": document.get("rubricHash"),
                "status": document.get("status"),
            }
        vote = None
        if record == "vote":
            found = connection.execute(
                "SELECT * FROM votes WHERE assignment_id=?", (assignment_id,)
            ).fetchone()
            if found is None:
                raise EvidenceRejected(
                    f"assignment {assignment_id!r} has no vote in the collector yet"
                )
            try:
                tags = json.loads(found["tags"])
            except ValueError:
                raise EvidenceRejected(
                    "the collector vote's tags are not JSON"
                ) from None
            vote = {
                "id": found["id"],
                "assignment_id": found["assignment_id"],
                "reviewer": found["reviewer"],
                "choice": found["choice"],
                "reason": found["reason"],
                "tags": tags,
                "created": found["created"],
            }
        invalidated = (
            connection.execute(
                "SELECT 1 FROM records WHERE kind='invalid-assignment' AND id=?",
                (assignment_id,),
            ).fetchone()
            is not None
        )
    except sqlite3.Error as exc:
        raise EvidenceRejected(f"cannot read the collector database: {exc}") from None
    finally:
        connection.close()
    export = {
        "schema_version": r.COLLECTOR_EXPORT_SCHEMA,
        "record": record,
        "collector": {
            "implementation": pin["implementation"],
            "commit": pin["commit"],
            "sources": sources,
        },
        "assignment": assignment,
        "artifacts": artifacts,
        "task": task,
        "vote": vote,
        "invalidated": invalidated,
    }
    try:
        r.check_collector_export(export)
    except r.InvalidRecord as exc:
        raise EvidenceRejected(
            f"the collector record does not fit the export schema: {exc}"
        ) from None
    return export


def collector_output_hash(preview_sha256: str, file_hashes: Mapping[str, str]) -> str:
    """The collector's own output digest (``Store.output_digest``).

    It hashes Python's sorted-key JSON, not JCS; reproduced exactly so the
    bridge can check the collector's claim, never used for a sidecar digest.
    """
    document = {"previewHash": preview_sha256, "fileHashes": dict(file_hashes)}
    text = json.dumps(
        document, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _bind_files(
    side: str, artifact: Mapping[str, Any], attempt: Mapping[str, Any]
) -> None:
    """A collector file manifest must be the attempt's output, under one prefix."""
    hashes = artifact["file_hashes"]
    if hashes is None:
        return  # a legacy collector artifact is bound by its preview alone
    if collector_output_hash(artifact["sha256"], hashes) != artifact["output_hash"]:
        raise EvidenceRejected(
            f"collector artifact {artifact['id']} ({side}): its output hash does not "
            "match its own preview and file hashes"
        )
    members = attempt["output"]["members"]
    if len(hashes) != len(members):
        raise EvidenceRejected(
            f"collector artifact {artifact['id']} ({side}) lists {len(hashes)} files; "
            f"the attempt's output has {len(members)}"
        )
    first = members[0]["path"]
    prefixes = {
        path[: -len(first)]
        for path in hashes
        if path == first or path.endswith("/" + first)
    }
    for prefix in sorted(prefixes):
        if all(hashes.get(prefix + m["path"]) == m["sha256"] for m in members):
            return
    raise EvidenceRejected(
        f"collector artifact {artifact['id']} ({side}): its files are not the "
        "attempt's output files (path or hash differs)"
    )


def check_assignment_binding(
    assessment: Assessment, payload: Mapping[str, Any]
) -> None:
    """Bind a collector assignment to the created, released assessment.

    The collector must be the build the frozen protocol pins (implementation,
    commit and every pinned source hash); its two sides must show exactly the
    two attempts' review artifacts, in the recorded A/B order; file manifests,
    when the collector has them, must be the attempts' outputs; every assignment
    must name a task version carrying the assessment's rubric digest; and the assignment must have
    been created after the release was recorded, so a legacy assignment made
    before this assessment can never be bound.
    """
    export = payload["collector_export"]
    pin = assessment.protocol_collector
    if dict(export["collector"]) != dict(pin):
        raise EvidenceRejected(
            "the collector build differs from the one the frozen judge protocol pins "
            "(implementation, commit or a source file hash)",
            hint="run the collector at the pinned commit, or freeze a new protocol",
        )
    collector = payload["collector"]
    if (collector["implementation"], collector["commit"]) != (
        pin["implementation"],
        pin["commit"],
    ):
        raise EvidenceRejected(
            "payload.collector does not name the pinned collector build"
        )
    row = export["assignment"]
    if collector["assignment_id"] != row["id"]:
        raise EvidenceRejected(
            "payload.collector.assignment_id is not the exported assignment"
        )
    if export["invalidated"]:
        raise EvidenceRejected(
            f"collector assignment {row['id']} is marked invalid (its evidence changed)"
        )
    artifacts = export["artifacts"]
    for side in ("left", "right"):
        if row[f"{side}_hash"] != artifacts[side]["sha256"]:
            raise EvidenceRejected(
                f"the assignment's {side} hash is not its artifact's"
            )
        if row[f"{side}_id"] != artifacts[side]["id"]:
            raise EvidenceRejected(
                f"the assignment's {side} artifact id differs from the export's"
            )
        if artifacts[side]["case_id"] != row["case_id"]:
            raise EvidenceRejected(
                f"the {side} artifact belongs to another collector case"
            )
    by_review = {a["review_artifact"]["sha256"]: a for a in assessment.attempts}
    if {row["left_hash"], row["right_hash"]} != set(by_review) or row[
        "left_hash"
    ] == row["right_hash"]:
        raise EvidenceRejected(
            "the collector assignment does not show exactly this assessment's two review "
            "artifacts (a changed, missing or foreign artifact)"
        )
    if dict(payload["presentation"]) != {"A": row["left_hash"], "B": row["right_hash"]}:
        raise EvidenceRejected(
            "presentation must be A = the collector's left, B = its right"
        )
    if payload["reviewer_subject"] != row["reviewer"]:
        raise EvidenceRejected(
            "reviewer_subject must be the collector's reviewer, exactly"
        )
    for side in ("left", "right"):
        _bind_files(side, artifacts[side], by_review[row[f"{side}_hash"]])
    task = export["task"]
    if row["task_id"] is None:
        raise EvidenceRejected(
            "the collector assignment has no task version binding the frozen rubric; "
            "unbound prototype records stay research history",
            hint="issue a new assignment for a task version carrying the frozen rubric digest",
        )
    else:
        if (
            task is None
            or task["id"] != row["task_id"]
            or task["case_id"] != row["case_id"]
        ):
            raise EvidenceRejected(
                "the export's task is not the assignment's task version"
            )
        for side in ("left", "right"):
            if artifacts[side]["task_id"] != row["task_id"]:
                raise EvidenceRejected(
                    f"the {side} output belongs to another task version"
                )
        rubric = assessment.created.payload["rubric"] if assessment.created else None
        if (
            task["rubric_hash"] is None
            or rubric is None
            or task["rubric_hash"] != rubric["sha256"]
        ):
            raise EvidenceRejected(
                "the collector task's frozen rubric digest is not the assessment's rubric"
            )
    assert assessment.release is not None
    released = r.timestamp(assessment.release.created_at, "release.created_at")
    if r.parse_collector_time(row["created"], "assignment.created") < released:
        raise EvidenceRejected(
            "the collector assignment predates the recorded release; an assignment made "
            "before this assessment stays research history and is not bound"
        )


def check_vote_binding(assessment: Assessment, event: Event) -> None:
    """Bind an imported collector vote to the assessment's assignment."""
    payload = event.payload
    assert assessment.assignment is not None
    issued = assessment.assignment.payload["collector_export"]
    export = payload["collector_export"]
    for field in ("collector", "assignment", "artifacts", "task"):
        if export[field] != issued[field]:
            raise EvidenceRejected(
                f"the vote's collector {field} differs from the assignment's export"
            )
    if export["invalidated"]:
        raise EvidenceRejected(
            "the collector marked this assignment invalid; its vote is not bound"
        )
    vote = export["vote"]
    row = export["assignment"]
    if vote["assignment_id"] != row["id"] or vote["reviewer"] != row["reviewer"]:
        raise EvidenceRejected("the vote is not this assignment's reviewer's vote")
    expected = {
        "choice": r.COLLECTOR_CHOICES[vote["choice"]],
        "reason": vote["reason"],
        "tags": vote["tags"],
        "submitted_at": r.format_timestamp(
            r.parse_collector_time(vote["created"], "vote.created")
        ),
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise EvidenceRejected(f"{field} must be the collector vote's ({value!r})")
    source = payload["imported_from"]
    if (
        source["collector"] != export["collector"]["implementation"]
        or source["record_id"] != vote["id"]
    ):
        raise EvidenceRejected("imported_from must name the collector and its vote id")
    if dict(event.actor) != {
        "role": "collector",
        "subject": export["collector"]["implementation"],
    }:
        raise EvidenceRejected(
            "an imported vote is written by its collector (actor role collector)"
        )
    if r.parse_collector_time(vote["created"], "vote.created") < r.parse_collector_time(
        row["created"], "assignment.created"
    ):
        raise EvidenceRejected("the vote predates its assignment")
