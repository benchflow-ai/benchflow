"""Synthetic saved attempts and a synthetic collector for assessment tests.

The rollouts are small completed-rollout folders (``result.json``, an output
tree, a preview image) written into ``tmp_path``. The collector is a SQLite
file with the prototype's own schema and record shapes
(``design-bench/prototype/model.py`` at lab commit f2650b05), filled the way
its ``Store.next_pair`` and ``Store.vote`` fill it, so the bridge's
read-only export is tested against the real layout. Votes here are synthetic
fixtures that exercise mechanics only: they are not human preferences.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from benchflow import assessment as a

TASK_DIGEST = "sha256:" + "11" * 32
OTHER_TASK_DIGEST = "sha256:" + "22" * 32
IMPLEMENTATION = "en161-design-bench-prototype"
COMMIT = "f2650b0542ac788401e8f5f8e48af121ac320249"
RUBRIC_ID = "design-quality"
RUBRIC_VERSION = "2026-10-26.1"
# The prototype's DDL, verbatim from model.py's Store.__init__.
COLLECTOR_DDL = """
  PRAGMA journal_mode=WAL;
  PRAGMA foreign_keys=ON;
  CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT, doc TEXT NOT NULL,
    PRIMARY KEY(kind,id));
  CREATE TABLE IF NOT EXISTS assignments(id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
    phase TEXT NOT NULL, reviewer TEXT NOT NULL, left_id TEXT NOT NULL,
    right_id TEXT NOT NULL, left_hash TEXT NOT NULL, right_hash TEXT NOT NULL,
    task_id TEXT, created TEXT NOT NULL);
  CREATE TABLE IF NOT EXISTS votes(id TEXT PRIMARY KEY, assignment_id TEXT UNIQUE NOT NULL,
    reviewer TEXT NOT NULL, choice TEXT NOT NULL, reason TEXT NOT NULL,
    tags TEXT NOT NULL, created TEXT NOT NULL,
    FOREIGN KEY(assignment_id) REFERENCES assignments(id));
  CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, kind TEXT NOT NULL,
    doc TEXT NOT NULL, created TEXT NOT NULL);
"""
COLLECTOR_SOURCES = {
    "model.py": b"# synthetic collector persistence\n",
    "app.py": b"# synthetic collector http\n",
    "web/app.js": b"// synthetic collector ui\n",
}


@contextmanager
def collector_connection(db: Path) -> Iterator[sqlite3.Connection]:
    """One committed transaction on a connection that is closed at once.

    Closing the last connection checkpoints a WAL database into its main
    file; leaving that to garbage collection would let it land later, in the
    middle of a test that checks nothing else writes the file.
    """
    with closing(sqlite3.connect(db)) as connection, connection:
        yield connection


def collector_canonical(value: Any) -> str:
    """The prototype's ``canonical``: Python's sorted-key JSON."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Rollout:
    root: Path
    output: Path
    preview: Path
    sandbox_id: str

    def ref(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("sandbox_provider", "docker")
        return a.attempt_ref(self.root, self.output, self.preview, **kwargs)


def make_rollout(
    base: Path,
    name: str,
    *,
    task_digest: str = TASK_DIGEST,
    files: dict[str, bytes] | None = None,
    preview: bytes | None = None,
    sandbox_id: str | None = None,
    reward: float | None = 0.0,
) -> Rollout:
    """A completed rollout folder holding result.json, out/ and a preview."""
    root = base / "jobs" / "job-1" / name
    output = root / "workspace" / "out"
    output.mkdir(parents=True)
    for relative, data in (
        files or {"index.html": f"<h1>{name}</h1>".encode()}
    ).items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    sandbox = sandbox_id or f"sbx-{name}"
    result = {
        "task_name": "design-task",
        "rollout_name": name,
        "rewards": {"reward": reward} if reward is not None else None,
        "agent": "fake-agent",
        "model": "fake/model",
        "error": None,
        "started_at": "2026-10-20 02:04:59.487056",
        "finished_at": "2026-10-20 02:40:08.606086",
        "task_digest": task_digest,
        "sandbox_id": sandbox,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    (root / "config.json").write_text(json.dumps({"agent": "fake-agent"}))
    image = base / "previews" / f"{name}.jpg"  # rendered outside the job folder
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(preview if preview is not None else f"JPEG:{name}".encode())
    return Rollout(root, output, image, sandbox)


def write_collector_code(base: Path) -> Path:
    code = base / "collector-code"
    for relative, data in COLLECTOR_SOURCES.items():
        path = code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return code


def protocol_document(code: Path, **overrides: Any) -> dict[str, Any]:
    sources = {rel: sha((code / rel).read_bytes()) for rel in COLLECTOR_SOURCES}
    document = {
        "schema_version": "benchflow.assessment-judge-protocol.v1",
        "id": "en161-blind-pair",
        "version": "1",
        "collector": {
            "implementation": IMPLEMENTATION,
            "commit": COMMIT,
            "sources": sources,
        },
        "choices": {"left": "left", "right": "right", "tie": "tie", "skip": "abstain"},
        "rules": [
            "Two outputs of one task, shown blind as A and B; the collector assigns sides.",
            "No producer, model, score or source is shown.",
        ],
    }
    document.update(overrides)
    return document


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


def release_proof(
    *rollouts: Rollout, provider: str = "docker", at: datetime | None = None
) -> dict:
    moment = (at or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return {
        "schema_version": "benchflow.assessment-release-proof.v1",
        "observations": [
            {
                "provider": provider,
                "runtime_id": rollout.sandbox_id,
                "state": "absent",
                "observed_at": moment,
                "observer": "test-operator",
                "method": "docker inspect: no such container",
            }
            for rollout in rollouts
        ],
    }


@dataclass
class Collector:
    """A collector database laid out exactly as the prototype lays it out."""

    data: Path
    code: Path
    db: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.db = self.data / "workspace.sqlite3"
        with collector_connection(self.db) as connection:
            connection.executescript(COLLECTOR_DDL)

    def _put(self, kind: str, doc: dict[str, Any]) -> None:
        with collector_connection(self.db) as connection:
            connection.execute(
                "INSERT INTO records VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET doc=excluded.doc",
                (kind, doc["id"], collector_canonical(doc)),
            )

    def add_artifact(
        self,
        rollout: Rollout,
        *,
        case_id: str = "case-1",
        with_files: bool = True,
        task_id: str | None = None,
        prefix: str | None = None,
    ) -> str:
        artifact_id = uuid.uuid4().hex
        preview_sha = sha(rollout.preview.read_bytes())
        doc: dict[str, Any] = {
            "id": artifact_id,
            "caseId": case_id,
            "title": rollout.root.name,
            "origin": "model",
            "status": "candidate",
            "preview": f"previews/{rollout.root.name}.jpg",
            "sha256": preview_sha,
        }
        if task_id is not None:
            doc["taskId"] = task_id
        if with_files:
            base = prefix if prefix is not None else f"runs/{rollout.root.name}/out/"
            file_hashes = {
                base + p.relative_to(rollout.output).as_posix(): sha(p.read_bytes())
                for p in sorted(rollout.output.rglob("*"))
                if p.is_file()
            }
            doc["fileHashes"] = file_hashes
            doc["outputHash"] = sha(
                collector_canonical(
                    {"previewHash": preview_sha, "fileHashes": file_hashes}
                ).encode()
            )
        self._put("artifact", doc)
        return artifact_id

    def add_task(
        self, task_id: str, *, rubric_hash: str | None, case_id: str = "case-1"
    ) -> None:
        self._put(
            "task",
            {
                "id": task_id,
                "version": 1,
                "caseId": case_id,
                "rubricId": "rubric-1",
                "rubricHash": rubric_hash,
                "status": "frozen",
            },
        )

    def assign(
        self,
        left: str,
        right: str,
        *,
        reviewer: str = "held-out-reviewer-01",
        case_id: str = "case-1",
        phase: str = "distillation",
        task_id: str | None = None,
        created: datetime | None = None,
    ) -> str:
        with collector_connection(self.db) as connection:
            connection.row_factory = sqlite3.Row
            docs = {
                row["id"]: json.loads(row["doc"])
                for row in connection.execute(
                    "SELECT id, doc FROM records WHERE kind='artifact'"
                )
            }
            assignment_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO assignments VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    assignment_id,
                    case_id,
                    phase,
                    reviewer,
                    left,
                    right,
                    docs[left]["sha256"],
                    docs[right]["sha256"],
                    task_id,
                    (created or datetime.now(UTC)).isoformat(),
                ),
            )
        return assignment_id

    def vote(
        self,
        assignment_id: str,
        choice: str,
        *,
        reason: str = "",
        tags: tuple[str, ...] = (),
        reviewer: str = "held-out-reviewer-01",
        created: datetime | None = None,
    ) -> str:
        vote_id = uuid.uuid4().hex
        with collector_connection(self.db) as connection:
            connection.execute(
                "INSERT INTO votes VALUES(?,?,?,?,?,?,?)",
                (
                    vote_id,
                    assignment_id,
                    reviewer,
                    choice,
                    reason,
                    collector_canonical(sorted(set(tags))),
                    (
                        created or datetime.now(UTC) + timedelta(milliseconds=1)
                    ).isoformat(),
                ),
            )
        return vote_id

    def invalidate(self, assignment_id: str) -> None:
        self._put("invalid-assignment", {"id": assignment_id, "reason": "changed"})


@dataclass
class World:
    """Two saved attempts, frozen inputs, a collector and a sidecar directory."""

    base: Path
    first: Rollout
    second: Rollout
    rubric: Path
    protocol: Path
    collector: Collector
    sidecar: Path
    proof: Path
    artifact_ids: dict[str, str] = field(default_factory=dict)
    assignment_id: str | None = None

    def refs(self) -> list[dict[str, Any]]:
        return [self.first.ref(), self.second.ref()]

    def create(self, **kwargs: Any) -> a.Appended:
        kwargs.setdefault("rubric", self.rubric)
        kwargs.setdefault("rubric_id", RUBRIC_ID)
        kwargs.setdefault("rubric_version", RUBRIC_VERSION)
        kwargs.setdefault("protocol", self.protocol)
        return a.create(self.sidecar, self.refs(), **kwargs)

    def release(self) -> a.Appended:
        return a.release(self.sidecar, self.proof)

    def collector_assignment(self, **kwargs: Any) -> str:
        # Every eligible assignment is bound to its frozen rubric.
        task_id = kwargs.setdefault("task_id", "task-frozen-1")
        if task_id is not None:
            self.collector.add_task(task_id, rubric_hash=sha(self.rubric.read_bytes()))
        if not self.artifact_ids:
            self.artifact_ids = {
                "first": self.collector.add_artifact(self.first, task_id=task_id),
                "second": self.collector.add_artifact(self.second, task_id=task_id),
            }
        self.assignment_id = self.collector.assign(
            self.artifact_ids["first"], self.artifact_ids["second"], **kwargs
        )
        return self.assignment_id

    def assign(self, assignment_id: str | None = None) -> a.Appended:
        return a.assign(
            self.sidecar,
            collector_db=self.collector.db,
            collector_code=self.collector.code,
            assignment_id=assignment_id
            or self.assignment_id
            or self.collector_assignment(),
        )

    def vote(self, choice: str, **kwargs: Any) -> str:
        assert self.assignment_id is not None
        return self.collector.vote(self.assignment_id, choice, **kwargs)

    def import_vote(self) -> a.Appended:
        return a.import_vote(
            self.sidecar,
            collector_db=self.collector.db,
            collector_code=self.collector.code,
        )

    def answered(self, choice: str = "left", **kwargs: Any) -> a.Appended:
        """Created, released, assigned and answered with ``choice``."""
        self.create()
        self.release()
        self.assign()
        self.vote(choice, **kwargs)
        return self.import_vote()


def world(base: Path, **rollout_kwargs: Any) -> World:
    first = make_rollout(base, "design-task__aaaa1111", **rollout_kwargs)
    second = make_rollout(
        base,
        "design-task__bbbb2222",
        files={"index.html": b"<h1>second</h1>", "assets/app.css": b"body{}"},
    )
    rubric = base / "frozen" / "rubric.md"
    rubric.parent.mkdir(parents=True, exist_ok=True)
    rubric.write_bytes(b"# Design quality rubric\n\nFrozen 2026-10-26.\n")
    code = write_collector_code(base)
    protocol = write_json(base / "frozen" / "protocol.json", protocol_document(code))
    collector = Collector(base / "collector-data", code)
    proof = write_json(
        base / "frozen" / "release-proof.json", release_proof(first, second)
    )
    return World(
        base=base,
        first=first,
        second=second,
        rubric=rubric,
        protocol=protocol,
        collector=collector,
        sidecar=first.root / "assessment",
        proof=proof,
    )


REVIEWER = {"role": "reviewer", "subject": "held-out-reviewer-01"}
OPERATOR = {"role": "operator", "subject": "operator-01"}
