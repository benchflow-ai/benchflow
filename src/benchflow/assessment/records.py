"""Strict shapes, identities and digests of the assessment sidecar's records.

Every record a sidecar reads or writes is checked here, field by field: the
event envelope (``benchflow.assessment-event.v1``, the frozen contract's
schema), each event kind's payload, saved-attempt references, output
manifests, the judge protocol, release proofs and collector exports. A
missing field, an unknown field, a wrong type, an unknown version or an
unknown enum value is an :class:`~benchflow.assessment.errors.InvalidRecord`;
nothing is coerced or defaulted.

Digests are SHA-256 over RFC 8785 canonical JSON (:mod:`.jcs`), written as 64
lowercase hex digits; ``task_digest`` keeps the ``sha256:`` prefix the saved
``result.json`` uses.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, NoReturn

from benchflow.assessment.errors import InvalidRecord
from benchflow.assessment.jcs import canonicalize

EVENT_SCHEMA = "benchflow.assessment-event.v1"
LABEL_SCHEMA = "benchflow.assessment-label.v1"
ATTEMPT_SCHEMA = "benchflow.saved-attempt-ref.v1"
OUTPUT_MANIFEST_SCHEMA = "benchflow.output-manifest.v1"
PROTOCOL_SCHEMA = "benchflow.assessment-judge-protocol.v1"
RELEASE_PROOF_SCHEMA = "benchflow.assessment-release-proof.v1"
COLLECTOR_EXPORT_SCHEMA = "benchflow.assessment-collector-export.v1"

KINDS = (
    "assessment.created",
    "compute.released",
    "assignment.issued",
    "review.submitted",
    "review.corrected",
    "review.retracted",
    "label.reduced",
    "label.consumed",
    "label.withdrawn",
)
ROLES = ("bridge", "collector", "reviewer", "reducer", "consumer", "operator")
# Which actor roles may write each kind.
KIND_ROLES: Mapping[str, tuple[str, ...]] = {
    "assessment.created": ("bridge", "operator"),
    "compute.released": ("bridge", "operator"),
    "assignment.issued": ("bridge", "collector"),
    # Answers come only from the collector (the one review
    # collector in this slice); corrections and retractions are SDK events.
    "review.submitted": ("collector",),
    "review.corrected": ("reviewer", "operator"),
    "review.retracted": ("reviewer", "operator"),
    "label.reduced": ("reducer",),
    "label.consumed": ("consumer",),
    "label.withdrawn": ("operator", "reducer"),
}
CHOICES = ("left", "right", "tie", "abstain")
# The collector's vote choices and what each means here: a skip is an
# explicit abstention, never a tie, a zero or a pending review.
COLLECTOR_CHOICES: Mapping[str, str] = {
    "left": "left",
    "right": "right",
    "tie": "tie",
    "skip": "abstain",
}
# Reduction of an effective answer (choice left = artifact A preferred).
CHOICE_VALUES: Mapping[str, float] = {"left": 1.0, "right": 0.0, "tie": 0.5}
LABEL_STATUSES = ("ready", "withheld")
WITHHELD_REASONS = ("human_abstention", "answer_retracted")
DISPOSITIONS = ("not_applicable", "queued_compensation", "compensated")
RUNTIME_ROLES = ("solver", "verifier", "shared")
RELEASED_STATES = ("deleted", "absent")
COLLECTOR_PHASES = ("distillation", "evaluation")
REDUCER = {"id": "benchflow.assessment.reduce", "version": "1"}

ENVELOPE_FIELDS = (
    "schema_version",
    "assessment_id",
    "event_id",
    "event_seq",
    "created_at",
    "kind",
    "actor",
    "idempotency_key",
    "expected_previous_event_sha256",
    "previous_event_sha256",
    "payload",
    "event_sha256",
)
# Writer clock fields a retry may legitimately change: left out of the
# semantic payload that idempotency compares (the frozen contract's
# "receipt timestamps").
RECEIPT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "compute.released": ("released_at",),
    "review.submitted": ("submitted_at",),
    "review.corrected": ("corrected_at",),
    "review.retracted": ("retracted_at",),
    "label.consumed": ("consumed_at",),
    "label.withdrawn": ("withdrawn_at",),
}

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_CONTROL = re.compile("[\x00-\x1f\x7f]")
MAX_TEXT = 4000
MAX_LABEL = 512
MAX_PATH = 4096
MAX_MEMBERS = 100_000


# ---------------------------------------------------------------------------
# Field checks
# ---------------------------------------------------------------------------


def fail(path: str, message: str) -> NoReturn:
    raise InvalidRecord(f"{path}: {message}")


def obj(
    value: Any, path: str, required: Iterable[str], optional: Iterable[str] = ()
) -> dict[str, Any]:
    """``value`` as an object holding every ``required`` name and no unknown one."""
    if not isinstance(value, dict):
        fail(path, "must be an object")
    required = tuple(required)
    allowed = set(required) | set(optional)
    missing = [name for name in required if name not in value]
    if missing:
        fail(path, f"missing {', '.join(missing)}")
    unknown = sorted(name for name in value if name not in allowed)
    if unknown:
        fail(path, f"unknown field {', '.join(unknown)}")
    return value


def text(
    value: Any,
    path: str,
    *,
    min_length: int = 1,
    max_length: int = MAX_LABEL,
    single_line: bool = True,
) -> str:
    if not isinstance(value, str):
        fail(path, "must be a string")
    if not min_length <= len(value) <= max_length:
        fail(path, f"must be {min_length} to {max_length} characters")
    if single_line and _CONTROL.search(value):
        fail(path, "must not contain control characters")
    if not single_line and re.search("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
        fail(path, "must not contain control characters other than tab and newline")
    if min_length and not value.strip():
        fail(path, "must not be blank")
    return value


def name(value: Any, path: str) -> str:
    """A lowercase machine name (an implementation or provider id)."""
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        fail(path, "must be a lowercase name: [a-z0-9][a-z0-9._-]{0,127}")
    return value


def digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        fail(path, "must be a SHA-256 digest: 64 lowercase hex digits")
    return value


def prefixed_digest(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or not _HEX64.fullmatch(value[7:])
    ):
        fail(path, "must be 'sha256:' and 64 lowercase hex digits")
    return value


def commit(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _HEX40.fullmatch(value):
        fail(path, "must be a full 40-hex git commit")
    return value


def uuid(value: Any, path: str, *, v7: bool = False) -> str:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        fail(path, "must be a lowercase canonical UUID")
    if v7 and (value[14] != "7" or value[19] not in "89ab"):
        fail(path, "must be a UUIDv7")
    return value


def timestamp(value: Any, path: str) -> datetime:
    """An RFC 3339 UTC timestamp ending in ``Z``."""
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        fail(path, "must be an RFC 3339 UTC timestamp like 2026-10-27T23:00:00.000000Z")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        fail(path, "is not a real date and time")


def integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(path, "must be an integer")
    if value < minimum:
        fail(path, f"must be at least {minimum}")
    return value


def enum(value: Any, path: str, options: Iterable[str]) -> str:
    options = tuple(options)
    if not isinstance(value, str) or value not in options:
        fail(path, f"must be one of {', '.join(options)}")
    return value


def array(value: Any, path: str, *, min_items: int = 0, max_items: int) -> list[Any]:
    if not isinstance(value, list):
        fail(path, "must be an array")
    if not min_items <= len(value) <= max_items:
        fail(path, f"must hold {min_items} to {max_items} items")
    return value


def relative_path(value: Any, path: str) -> str:
    """A relative POSIX path with no empty, '.' or '..' segment."""
    text(value, path, max_length=MAX_PATH)
    if value.startswith("/") or "\\" in value:
        fail(path, "must be a relative POSIX path")
    if any(part in ("", ".", "..") for part in value.split("/")):
        fail(path, "must not contain empty, '.' or '..' segments")
    return value


def locator(value: Any, path: str) -> str:
    """Where evidence was found: an absolute path, a retrieval hint only."""
    text(value, path, max_length=MAX_PATH)
    if not value.startswith("/"):
        fail(path, "must be an absolute path")
    return value


def utf8_sort_key(value: str) -> bytes:
    return value.encode("utf-8")


# ---------------------------------------------------------------------------
# Digests and identities
# ---------------------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jcs_sha256(value: Any) -> str:
    """SHA-256 of the RFC 8785 canonical form of ``value``."""
    return sha256_hex(canonicalize(value))


def event_digest(record: Mapping[str, Any]) -> str:
    """``SHA256(JCS(record with event_sha256 removed))``."""
    return jcs_sha256({k: v for k, v in record.items() if k != "event_sha256"})


def attempt_id(
    rollout_id: str, task_digest: str, result_sha256: str, output_manifest_sha256: str
) -> str:
    """The frozen contract's attempt identity; locators are not part of it."""
    return jcs_sha256(
        {
            "rollout_id": rollout_id,
            "task_digest": task_digest,
            "result_sha256": result_sha256,
            "output_manifest_sha256": output_manifest_sha256,
        }
    )


def output_manifest(kind: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": OUTPUT_MANIFEST_SCHEMA, "kind": kind, "members": members}


def output_manifest_sha256(kind: str, members: list[dict[str, Any]]) -> str:
    return jcs_sha256(output_manifest(kind, members))


def semantic_payload(
    kind: str, actor: Mapping[str, Any], payload: Mapping[str, Any]
) -> bytes:
    """What idempotency compares: kind, actor and payload, minus receipt clocks."""
    skip = RECEIPT_FIELDS.get(kind, ())
    return canonicalize(
        {
            "kind": kind,
            "actor": dict(actor),
            "payload": {k: v for k, v in payload.items() if k not in skip},
        }
    )


def uuid7() -> str:
    """A random UUIDv7 (RFC 9562): 48 bits of Unix milliseconds, then random."""
    millis = time.time_ns() // 1_000_000
    random = int.from_bytes(os.urandom(10), "big")
    value = (millis & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= (random >> 68) << 64  # 12 random bits
    value |= 0b10 << 62
    value |= random & ((1 << 62) - 1)
    hexed = f"{value:032x}"
    return f"{hexed[:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:]}"


def utc_now() -> str:
    return format_timestamp(datetime.now(UTC))


def format_timestamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_collector_time(value: Any, path: str) -> datetime:
    """A collector's ``datetime.isoformat()`` timestamp (UTC offset required)."""
    if not isinstance(value, str) or not 19 <= len(value) <= 40:
        fail(path, "must be an ISO 8601 timestamp")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        fail(path, "must be an ISO 8601 timestamp")
    if moment.tzinfo is None or moment.utcoffset() is None:
        fail(path, "must carry a UTC offset")
    return moment


# ---------------------------------------------------------------------------
# Input documents
# ---------------------------------------------------------------------------


def check_members(kind: str, members: Any, path: str) -> list[dict[str, Any]]:
    """Output members: unique relative paths in ascending UTF-8 byte order."""
    array(members, path, min_items=1, max_items=MAX_MEMBERS)
    previous: bytes | None = None
    for index, member in enumerate(members):
        where = f"{path}[{index}]"
        obj(member, where, ("path", "sha256", "bytes"))
        relative_path(member["path"], f"{where}.path")
        digest(member["sha256"], f"{where}.sha256")
        integer(member["bytes"], f"{where}.bytes")
        key = utf8_sort_key(member["path"])
        if previous is not None and key <= previous:
            fail(where, "members must be unique and sorted by path (UTF-8 bytes)")
        previous = key
    if kind == "file" and (len(members) != 1 or "/" in members[0]["path"]):
        fail(path, "a file output has exactly one member, named by the file")
    return members


def check_attempt(value: Any, path: str) -> dict[str, Any]:
    """A ``benchflow.saved-attempt-ref.v1`` whose identities recompute."""
    ref = obj(
        value,
        path,
        (
            "schema_version",
            "attempt_id",
            "rollout_id",
            "task_digest",
            "result_sha256",
            "output_manifest_sha256",
            "result",
            "output",
            "review_artifact",
            "runtime_refs",
        ),
    )
    if ref["schema_version"] != ATTEMPT_SCHEMA:
        fail(f"{path}.schema_version", f"must be {ATTEMPT_SCHEMA}")
    digest(ref["attempt_id"], f"{path}.attempt_id")
    text(ref["rollout_id"], f"{path}.rollout_id")
    prefixed_digest(ref["task_digest"], f"{path}.task_digest")
    digest(ref["result_sha256"], f"{path}.result_sha256")
    digest(ref["output_manifest_sha256"], f"{path}.output_manifest_sha256")
    result = obj(ref["result"], f"{path}.result", ("locator", "bytes"))
    locator(result["locator"], f"{path}.result.locator")
    integer(result["bytes"], f"{path}.result.bytes", minimum=1)
    output = obj(ref["output"], f"{path}.output", ("kind", "locator", "members"))
    enum(output["kind"], f"{path}.output.kind", ("directory", "file"))
    locator(output["locator"], f"{path}.output.locator")
    check_members(output["kind"], output["members"], f"{path}.output.members")
    review = obj(
        ref["review_artifact"],
        f"{path}.review_artifact",
        ("sha256", "bytes", "locator"),
        ("media_type",),
    )
    digest(review["sha256"], f"{path}.review_artifact.sha256")
    integer(review["bytes"], f"{path}.review_artifact.bytes", minimum=1)
    locator(review["locator"], f"{path}.review_artifact.locator")
    if "media_type" in review:
        text(review["media_type"], f"{path}.review_artifact.media_type", max_length=128)
    refs = array(ref["runtime_refs"], f"{path}.runtime_refs", min_items=1, max_items=64)
    seen: set[tuple[str, str]] = set()
    for index, runtime in enumerate(refs):
        where = f"{path}.runtime_refs[{index}]"
        obj(runtime, where, ("provider", "runtime_id", "role", "source"))
        name(runtime["provider"], f"{where}.provider")
        text(runtime["runtime_id"], f"{where}.runtime_id")
        enum(runtime["role"], f"{where}.role", RUNTIME_ROLES)
        enum(runtime["source"], f"{where}.source", ("result.sandbox_id", "declared"))
        key = (runtime["provider"], runtime["runtime_id"])
        if key in seen:
            fail(where, "names a runtime twice")
        seen.add(key)
    manifest = output_manifest_sha256(output["kind"], output["members"])
    if manifest != ref["output_manifest_sha256"]:
        fail(f"{path}.output_manifest_sha256", "does not match the output members")
    identity = attempt_id(
        ref["rollout_id"],
        ref["task_digest"],
        ref["result_sha256"],
        ref["output_manifest_sha256"],
    )
    if identity != ref["attempt_id"]:
        fail(f"{path}.attempt_id", "does not match the attempt's identities")
    return ref


def check_frozen_file(
    value: Any, path: str, *, extra: Iterable[str] = ()
) -> dict[str, Any]:
    ref = obj(value, path, ("id", "version", "sha256", "bytes", "locator", *extra))
    text(ref["id"], f"{path}.id", max_length=256)
    text(ref["version"], f"{path}.version", max_length=256)
    digest(ref["sha256"], f"{path}.sha256")
    integer(ref["bytes"], f"{path}.bytes", minimum=1)
    locator(ref["locator"], f"{path}.locator")
    return ref


def check_collector_pin(value: Any, path: str) -> dict[str, Any]:
    pin = obj(value, path, ("implementation", "commit", "sources"))
    name(pin["implementation"], f"{path}.implementation")
    commit(pin["commit"], f"{path}.commit")
    sources = pin["sources"]
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 64:
        fail(f"{path}.sources", "must map 1 to 64 source paths to SHA-256 digests")
    for source, value_digest in sources.items():
        relative_path(source, f"{path}.sources key")
        digest(value_digest, f"{path}.sources[{source!r}]")
    return pin


def check_protocol_document(value: Any, path: str = "protocol") -> dict[str, Any]:
    """A ``benchflow.assessment-judge-protocol.v1`` document."""
    document = obj(
        value,
        path,
        ("schema_version", "id", "version", "collector", "choices", "rules"),
    )
    if document["schema_version"] != PROTOCOL_SCHEMA:
        fail(f"{path}.schema_version", f"must be {PROTOCOL_SCHEMA}")
    text(document["id"], f"{path}.id", max_length=256)
    text(document["version"], f"{path}.version", max_length=256)
    check_collector_pin(document["collector"], f"{path}.collector")
    if document["choices"] != dict(COLLECTOR_CHOICES):
        fail(
            f"{path}.choices",
            "must map the collector's left/right/tie/skip to left/right/tie/abstain",
        )
    rules = array(document["rules"], f"{path}.rules", min_items=1, max_items=64)
    for index, rule in enumerate(rules):
        text(rule, f"{path}.rules[{index}]", max_length=MAX_TEXT)
    return document


def check_release_proof(value: Any, path: str = "release_proof") -> dict[str, Any]:
    """A ``benchflow.assessment-release-proof.v1`` document from an operator."""
    proof = obj(value, path, ("schema_version", "observations"))
    if proof["schema_version"] != RELEASE_PROOF_SCHEMA:
        fail(f"{path}.schema_version", f"must be {RELEASE_PROOF_SCHEMA}")
    check_observations(proof["observations"], f"{path}.observations")
    return proof


def check_observations(value: Any, path: str) -> list[dict[str, Any]]:
    observations = array(value, path, min_items=1, max_items=256)
    for index, observation in enumerate(observations):
        where = f"{path}[{index}]"
        obj(
            observation,
            where,
            ("provider", "runtime_id", "state", "observed_at", "observer", "method"),
        )
        name(observation["provider"], f"{where}.provider")
        text(observation["runtime_id"], f"{where}.runtime_id")
        enum(observation["state"], f"{where}.state", RELEASED_STATES)
        timestamp(observation["observed_at"], f"{where}.observed_at")
        text(observation["observer"], f"{where}.observer")
        text(observation["method"], f"{where}.method")
    return observations


def check_collector_export(
    value: Any, path: str = "collector_export"
) -> dict[str, Any]:
    """A ``benchflow.assessment-collector-export.v1``: one collector record, as read."""
    export = obj(
        value,
        path,
        (
            "schema_version",
            "record",
            "collector",
            "assignment",
            "artifacts",
            "task",
            "vote",
            "invalidated",
        ),
    )
    if export["schema_version"] != COLLECTOR_EXPORT_SCHEMA:
        fail(f"{path}.schema_version", f"must be {COLLECTOR_EXPORT_SCHEMA}")
    enum(export["record"], f"{path}.record", ("assignment", "vote"))
    check_collector_pin(export["collector"], f"{path}.collector")
    row = obj(
        export["assignment"],
        f"{path}.assignment",
        (
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
        ),
    )
    for field in ("id", "case_id", "left_id", "right_id"):
        text(row[field], f"{path}.assignment.{field}")
    enum(row["phase"], f"{path}.assignment.phase", COLLECTOR_PHASES)
    text(row["reviewer"], f"{path}.assignment.reviewer", max_length=80)
    digest(row["left_hash"], f"{path}.assignment.left_hash")
    digest(row["right_hash"], f"{path}.assignment.right_hash")
    if row["task_id"] is not None:
        text(row["task_id"], f"{path}.assignment.task_id")
    parse_collector_time(row["created"], f"{path}.assignment.created")
    artifacts = obj(export["artifacts"], f"{path}.artifacts", ("left", "right"))
    for side in ("left", "right"):
        where = f"{path}.artifacts.{side}"
        artifact = obj(
            artifacts[side],
            where,
            ("id", "case_id", "sha256", "task_id", "output_hash", "file_hashes"),
        )
        text(artifact["id"], f"{where}.id")
        text(artifact["case_id"], f"{where}.case_id")
        digest(artifact["sha256"], f"{where}.sha256")
        if artifact["task_id"] is not None:
            text(artifact["task_id"], f"{where}.task_id")
        if (artifact["output_hash"] is None) != (artifact["file_hashes"] is None):
            fail(where, "output_hash and file_hashes are both present or both null")
        if artifact["output_hash"] is not None:
            digest(artifact["output_hash"], f"{where}.output_hash")
            hashes = artifact["file_hashes"]
            if not isinstance(hashes, dict) or not 1 <= len(hashes) <= MAX_MEMBERS:
                fail(f"{where}.file_hashes", "must map file paths to SHA-256 digests")
            for file_path, file_digest in hashes.items():
                relative_path(file_path, f"{where}.file_hashes key")
                digest(file_digest, f"{where}.file_hashes[{file_path!r}]")
    if export["task"] is not None:
        where = f"{path}.task"
        task = obj(
            export["task"],
            where,
            ("id", "version", "case_id", "rubric_id", "rubric_hash", "status"),
        )
        text(task["id"], f"{where}.id")
        integer(task["version"], f"{where}.version")
        text(task["case_id"], f"{where}.case_id")
        if task["rubric_id"] is not None:
            text(task["rubric_id"], f"{where}.rubric_id")
        if task["rubric_hash"] is not None:
            digest(task["rubric_hash"], f"{where}.rubric_hash")
        text(task["status"], f"{where}.status", max_length=64)
    if export["vote"] is not None:
        where = f"{path}.vote"
        vote = obj(
            export["vote"],
            where,
            ("id", "assignment_id", "reviewer", "choice", "reason", "tags", "created"),
        )
        text(vote["id"], f"{where}.id")
        text(vote["assignment_id"], f"{where}.assignment_id")
        text(vote["reviewer"], f"{where}.reviewer", max_length=80)
        enum(vote["choice"], f"{where}.choice", COLLECTOR_CHOICES)
        text(
            vote["reason"],
            f"{where}.reason",
            min_length=0,
            max_length=MAX_TEXT,
            single_line=False,
        )
        check_tags(vote["tags"], f"{where}.tags")
        parse_collector_time(vote["created"], f"{where}.created")
    if (export["record"] == "vote") != (export["vote"] is not None):
        fail(
            f"{path}.vote", "a vote record carries its vote; an assignment record none"
        )
    if not isinstance(export["invalidated"], bool):
        fail(f"{path}.invalidated", "must be true or false")
    return export


def check_tags(value: Any, path: str) -> list[str]:
    tags = array(value, path, max_items=32)
    for index, tag in enumerate(tags):
        name(tag, f"{path}[{index}]")
    if tags != sorted(set(tags)):
        fail(path, "must be unique and sorted")
    return tags


# ---------------------------------------------------------------------------
# Envelope and payloads
# ---------------------------------------------------------------------------


def check_envelope(record: Any, path: str = "event") -> dict[str, Any]:
    """The frozen ``benchflow.assessment-event.v1`` envelope, then the payload."""
    obj(record, path, ENVELOPE_FIELDS)
    if record["schema_version"] != EVENT_SCHEMA:
        fail(
            f"{path}.schema_version",
            f"unknown assessment schema version {record['schema_version']!r}; "
            f"this reader knows only {EVENT_SCHEMA}",
        )
    uuid(record["assessment_id"], f"{path}.assessment_id", v7=True)
    uuid(record["event_id"], f"{path}.event_id", v7=True)
    integer(record["event_seq"], f"{path}.event_seq", minimum=1)
    timestamp(record["created_at"], f"{path}.created_at")
    kind = enum(record["kind"], f"{path}.kind", KINDS)
    actor = obj(record["actor"], f"{path}.actor", ("role", "subject"))
    enum(actor["role"], f"{path}.actor.role", ROLES)
    text(actor["subject"], f"{path}.actor.subject")
    if actor["role"] not in KIND_ROLES[kind]:
        fail(
            f"{path}.actor.role",
            f"{kind} is written by {' or '.join(KIND_ROLES[kind])}, not {actor['role']}",
        )
    text(record["idempotency_key"], f"{path}.idempotency_key")
    for field in ("expected_previous_event_sha256", "previous_event_sha256"):
        if record[field] is not None:
            digest(record[field], f"{path}.{field}")
    digest(record["event_sha256"], f"{path}.event_sha256")
    check_payload(kind, record["payload"], f"{path}.payload")
    return record


def check_payload(kind: str, payload: Any, path: str) -> dict[str, Any]:
    _PAYLOADS[kind](payload, path)
    return payload


def _created(payload: Any, path: str) -> None:
    obj(payload, path, ("attempts", "rubric", "judge_protocol"))
    attempts = array(payload["attempts"], f"{path}.attempts", min_items=2, max_items=2)
    for index, attempt in enumerate(attempts):
        check_attempt(attempt, f"{path}.attempts[{index}]")
    first, second = attempts
    if first["attempt_id"] == second["attempt_id"]:
        fail(f"{path}.attempts", "names one attempt twice")
    if first["task_digest"] != second["task_digest"]:
        fail(f"{path}.attempts", "come from different tasks (task_digest differs)")
    if first["review_artifact"]["sha256"] == second["review_artifact"]["sha256"]:
        fail(f"{path}.attempts", "present byte-identical review artifacts")
    check_frozen_file(payload["rubric"], f"{path}.rubric")
    protocol = check_frozen_file(
        payload["judge_protocol"], f"{path}.judge_protocol", extra=("collector",)
    )
    check_collector_pin(protocol["collector"], f"{path}.judge_protocol.collector")


def _released(payload: Any, path: str) -> None:
    obj(payload, path, ("runtime_refs", "released_at", "release_proof"))
    refs = array(
        payload["runtime_refs"], f"{path}.runtime_refs", min_items=1, max_items=128
    )
    for index, runtime in enumerate(refs):
        where = f"{path}.runtime_refs[{index}]"
        obj(runtime, where, ("attempt_id", "provider", "runtime_id", "role"))
        digest(runtime["attempt_id"], f"{where}.attempt_id")
        name(runtime["provider"], f"{where}.provider")
        text(runtime["runtime_id"], f"{where}.runtime_id")
        enum(runtime["role"], f"{where}.role", RUNTIME_ROLES)
    timestamp(payload["released_at"], f"{path}.released_at")
    proof = obj(
        payload["release_proof"], f"{path}.release_proof", ("sha256", "observations")
    )
    digest(proof["sha256"], f"{path}.release_proof.sha256")
    check_observations(proof["observations"], f"{path}.release_proof.observations")


def _assigned(payload: Any, path: str) -> None:
    obj(
        payload,
        path,
        ("collector", "presentation", "reviewer_subject", "collector_export"),
    )
    collector = obj(
        payload["collector"],
        f"{path}.collector",
        ("implementation", "commit", "assignment_id", "export_sha256"),
    )
    name(collector["implementation"], f"{path}.collector.implementation")
    commit(collector["commit"], f"{path}.collector.commit")
    text(collector["assignment_id"], f"{path}.collector.assignment_id")
    digest(collector["export_sha256"], f"{path}.collector.export_sha256")
    presentation = obj(payload["presentation"], f"{path}.presentation", ("A", "B"))
    digest(presentation["A"], f"{path}.presentation.A")
    digest(presentation["B"], f"{path}.presentation.B")
    text(payload["reviewer_subject"], f"{path}.reviewer_subject", max_length=80)
    export = check_collector_export(
        payload["collector_export"], f"{path}.collector_export"
    )
    if export["record"] != "assignment":
        fail(f"{path}.collector_export.record", "must be an assignment record")
    if jcs_sha256(export) != collector["export_sha256"]:
        fail(f"{path}.collector.export_sha256", "does not match the embedded export")


def _answer_fields(payload: dict[str, Any], path: str) -> None:
    text(payload["assignment_id"], f"{path}.assignment_id")
    text(payload["reviewer_subject"], f"{path}.reviewer_subject", max_length=80)
    enum(payload["choice"], f"{path}.choice", CHOICES)
    text(
        payload["reason"],
        f"{path}.reason",
        min_length=0,
        max_length=MAX_TEXT,
        single_line=False,
    )
    check_tags(payload["tags"], f"{path}.tags")


def _submitted(payload: Any, path: str) -> None:
    obj(
        payload,
        path,
        (
            "assignment_id",
            "reviewer_subject",
            "choice",
            "reason",
            "tags",
            "submitted_at",
            "imported_from",
            "collector_export",
        ),
    )
    _answer_fields(payload, path)
    timestamp(payload["submitted_at"], f"{path}.submitted_at")
    source = obj(
        payload["imported_from"],
        f"{path}.imported_from",
        ("collector", "record_id", "export_sha256"),
    )
    name(source["collector"], f"{path}.imported_from.collector")
    text(source["record_id"], f"{path}.imported_from.record_id")
    digest(source["export_sha256"], f"{path}.imported_from.export_sha256")
    export = check_collector_export(
        payload["collector_export"], f"{path}.collector_export"
    )
    if export["record"] != "vote":
        fail(f"{path}.collector_export.record", "must be a vote record")
    if jcs_sha256(export) != source["export_sha256"]:
        fail(
            f"{path}.imported_from.export_sha256", "does not match the embedded export"
        )


def _corrected(payload: Any, path: str) -> None:
    obj(
        payload,
        path,
        (
            "supersedes_event_id",
            "assignment_id",
            "reviewer_subject",
            "choice",
            "reason",
            "tags",
            "correction_reason",
            "corrected_at",
        ),
    )
    uuid(payload["supersedes_event_id"], f"{path}.supersedes_event_id", v7=True)
    _answer_fields(payload, path)
    text(
        payload["correction_reason"],
        f"{path}.correction_reason",
        max_length=MAX_TEXT,
        single_line=False,
    )
    timestamp(payload["corrected_at"], f"{path}.corrected_at")


def _retracted(payload: Any, path: str) -> None:
    obj(
        payload,
        path,
        (
            "target_event_id",
            "assignment_id",
            "reviewer_subject",
            "reason",
            "retracted_at",
        ),
    )
    uuid(payload["target_event_id"], f"{path}.target_event_id", v7=True)
    text(payload["assignment_id"], f"{path}.assignment_id")
    text(payload["reviewer_subject"], f"{path}.reviewer_subject", max_length=80)
    text(payload["reason"], f"{path}.reason", max_length=MAX_TEXT, single_line=False)
    timestamp(payload["retracted_at"], f"{path}.retracted_at")


def _reduced(payload: Any, path: str) -> None:
    status = payload.get("status") if isinstance(payload, dict) else None
    revision = payload.get("revision") if isinstance(payload, dict) else None
    optional = ["supersedes_revision"] if revision != 1 else []
    optional.append("value" if status == "ready" else "withheld_reason")
    obj(
        payload,
        path,
        (
            "label_id",
            "revision",
            "source_event_id",
            "source_event_sha256",
            "status",
            "reducer",
        ),
        optional,
    )
    uuid(payload["label_id"], f"{path}.label_id", v7=True)
    integer(payload["revision"], f"{path}.revision", minimum=1)
    if (
        payload["revision"] > 1
        and payload.get("supersedes_revision") != payload["revision"] - 1
    ):
        fail(f"{path}.supersedes_revision", "must name the previous revision")
    uuid(payload["source_event_id"], f"{path}.source_event_id", v7=True)
    digest(payload["source_event_sha256"], f"{path}.source_event_sha256")
    enum(payload["status"], f"{path}.status", LABEL_STATUSES)
    if payload["status"] == "ready":
        value = payload.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or value not in CHOICE_VALUES.values()
        ):
            fail(f"{path}.value", "a ready label's value is 1, 0 or 0.5")
    else:
        enum(
            payload.get("withheld_reason"), f"{path}.withheld_reason", WITHHELD_REASONS
        )
    if payload["reducer"] != REDUCER:
        fail(f"{path}.reducer", f"must be {REDUCER}")


def _consumed(payload: Any, path: str) -> None:
    obj(
        payload,
        path,
        (
            "label_id",
            "label_revision",
            "consumer_id",
            "consumer_receipt",
            "consumed_at",
        ),
    )
    uuid(payload["label_id"], f"{path}.label_id", v7=True)
    integer(payload["label_revision"], f"{path}.label_revision", minimum=1)
    text(payload["consumer_id"], f"{path}.consumer_id")
    text(payload["consumer_receipt"], f"{path}.consumer_receipt", max_length=MAX_TEXT)
    timestamp(payload["consumed_at"], f"{path}.consumed_at")


def _withdrawn(payload: Any, path: str) -> None:
    obj(
        payload,
        path,
        (
            "label_id",
            "label_revision",
            "reason",
            "consumer_disposition",
            "consumption_event_ids",
            "withdrawn_at",
        ),
    )
    uuid(payload["label_id"], f"{path}.label_id", v7=True)
    integer(payload["label_revision"], f"{path}.label_revision", minimum=1)
    text(payload["reason"], f"{path}.reason", max_length=MAX_TEXT, single_line=False)
    enum(payload["consumer_disposition"], f"{path}.consumer_disposition", DISPOSITIONS)
    ids = array(
        payload["consumption_event_ids"],
        f"{path}.consumption_event_ids",
        max_items=1024,
    )
    for index, event_id in enumerate(ids):
        uuid(event_id, f"{path}.consumption_event_ids[{index}]", v7=True)
    timestamp(payload["withdrawn_at"], f"{path}.withdrawn_at")


_PAYLOADS = {
    "assessment.created": _created,
    "compute.released": _released,
    "assignment.issued": _assigned,
    "review.submitted": _submitted,
    "review.corrected": _corrected,
    "review.retracted": _retracted,
    "label.reduced": _reduced,
    "label.consumed": _consumed,
    "label.withdrawn": _withdrawn,
}
