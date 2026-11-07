"""JSON Schemas of the assessment event and label, for docs/reference/schemas/.

``EVENT_ENVELOPE`` is the frozen contract's ``assessment-event.v1.schema.json``
(lab commit 92341d97, ``sdk/rounds/py8/release-readiness-20261024/
assessment-sidecar-v1``), unchanged. :func:`event_schema` adds one
``if kind / then payload`` rule per event kind and the payload shapes they
name, so every event valid here is valid under the frozen schema too.
:func:`label_schema` is ``benchflow.assessment-label.v1``.

The committed files are generated, never edited::

    python -m benchflow.assessment.schema docs/reference/schemas

``tests/assessment/test_schema.py`` keeps them current, checks the envelope
still equals the frozen file, and validates real events and labels against
them with an independent validator. The Python checks in
:mod:`benchflow.assessment.records` are stricter where JSON Schema cannot
reach (recomputed digests, sort order, cross-field rules).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from benchflow.assessment import records as r

__all__ = ["EVENT_ENVELOPE", "event_schema", "label_schema", "write_schemas"]

EVENT_FILE = "benchflow-assessment-event.v1.schema.json"
LABEL_FILE = "benchflow-assessment-label.v1.schema.json"

# The frozen contract's schema, verbatim (key order included).
EVENT_ENVELOPE: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://benchflow.ai/schemas/benchflow-assessment-event.v1.schema.json",
    "title": "One append-only event for a hash-bound blinded pairwise assessment",
    "type": "object",
    "additionalProperties": False,
    "required": [
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
    ],
    "properties": {
        "schema_version": {"const": "benchflow.assessment-event.v1"},
        "assessment_id": {"type": "string", "format": "uuid"},
        "event_id": {"type": "string", "format": "uuid"},
        "event_seq": {"type": "integer", "minimum": 1},
        "created_at": {"type": "string", "format": "date-time"},
        "kind": {
            "enum": [
                "assessment.created",
                "compute.released",
                "assignment.issued",
                "review.submitted",
                "review.corrected",
                "review.retracted",
                "label.reduced",
                "label.consumed",
                "label.withdrawn",
            ]
        },
        "actor": {
            "type": "object",
            "additionalProperties": False,
            "required": ["role", "subject"],
            "properties": {
                "role": {
                    "enum": [
                        "bridge",
                        "collector",
                        "reviewer",
                        "reducer",
                        "consumer",
                        "operator",
                    ]
                },
                "subject": {"type": "string", "minLength": 1, "maxLength": 512},
            },
        },
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 512},
        "expected_previous_event_sha256": {"$ref": "#/$defs/optionalDigest"},
        "previous_event_sha256": {"$ref": "#/$defs/optionalDigest"},
        "payload": {"type": "object"},
        "event_sha256": {"$ref": "#/$defs/digest"},
    },
    "$defs": {
        "digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "optionalDigest": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
    },
}

_CONTROL = "\\u0000-\\u001f\\u007f"
_TEXT_CONTROL = "\\u0000-\\u0008\\u000b\\u000c\\u000e-\\u001f\\u007f"
_SEGMENT = f"[^/\\\\{_CONTROL}]+"


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def _object(
    required: dict[str, Any], optional: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": {**required, **(optional or {})},
    }


def _text(max_length: int = r.MAX_LABEL) -> dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": max_length,
        "pattern": f"^[^{_CONTROL}]*\\S[^{_CONTROL}]*$",
    }


def _free_text(*, required: bool) -> dict[str, Any]:
    if required:
        pattern = f"^[^{_TEXT_CONTROL}]*\\S[^{_TEXT_CONTROL}]*$"
    else:
        pattern = f"^[^{_TEXT_CONTROL}]*$"
    return {
        "type": "string",
        "minLength": 1 if required else 0,
        "maxLength": r.MAX_TEXT,
        "pattern": pattern,
    }


def _payload_defs() -> dict[str, Any]:
    nullable_text = {"anyOf": [_text(), {"type": "null"}]}
    defs: dict[str, Any] = {
        "prefixedDigest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "uuid7": {
            "type": "string",
            "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        },
        "timestamp": {
            "type": "string",
            "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]{1,6})?Z$",
        },
        "name": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,127}$"},
        "text": _text(),
        "reviewer": _text(80),
        "relativePath": {
            "type": "string",
            "minLength": 1,
            "maxLength": r.MAX_PATH,
            "pattern": f"^(?!\\.\\.?(/|$))(?!.*/\\.\\.?(/|$)){_SEGMENT}(/{_SEGMENT})*$",
        },
        "locator": {
            "type": "string",
            "minLength": 1,
            "maxLength": r.MAX_PATH,
            "pattern": f"^/[^{_CONTROL}]*$",
        },
        "count": {"type": "integer", "minimum": 0},
        "tags": {
            "type": "array",
            "maxItems": 32,
            "uniqueItems": True,
            "items": _ref("name"),
        },
        "member": _object(
            {
                "path": _ref("relativePath"),
                "sha256": _ref("digest"),
                "bytes": _ref("count"),
            }
        ),
        "runtime": _object(
            {
                "provider": _ref("name"),
                "runtime_id": _ref("text"),
                "role": {"enum": list(r.RUNTIME_ROLES)},
                "source": {"enum": ["result.sandbox_id", "declared"]},
            }
        ),
        "savedAttemptRef": _object(
            {
                "schema_version": {"const": r.ATTEMPT_SCHEMA},
                "attempt_id": _ref("digest"),
                "rollout_id": _ref("text"),
                "task_digest": _ref("prefixedDigest"),
                "result_sha256": _ref("digest"),
                "output_manifest_sha256": _ref("digest"),
                "result": _object(
                    {
                        "locator": _ref("locator"),
                        "bytes": {"type": "integer", "minimum": 1},
                    }
                ),
                "output": _object(
                    {
                        "kind": {"enum": ["directory", "file"]},
                        "locator": _ref("locator"),
                        "members": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": r.MAX_MEMBERS,
                            "items": _ref("member"),
                        },
                    }
                ),
                "review_artifact": _object(
                    {
                        "sha256": _ref("digest"),
                        "bytes": {"type": "integer", "minimum": 1},
                        "locator": _ref("locator"),
                    },
                    {"media_type": _text(128)},
                ),
                "runtime_refs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 64,
                    "items": _ref("runtime"),
                },
            }
        ),
        "frozenFile": _object(
            {
                "id": _text(256),
                "version": _text(256),
                "sha256": _ref("digest"),
                "bytes": {"type": "integer", "minimum": 1},
                "locator": _ref("locator"),
            }
        ),
        "collectorPin": _object(
            {
                "implementation": _ref("name"),
                "commit": _ref("commit"),
                "sources": {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 64,
                    "propertyNames": _ref("relativePath"),
                    "additionalProperties": _ref("digest"),
                },
            }
        ),
        "observation": _object(
            {
                "provider": _ref("name"),
                "runtime_id": _ref("text"),
                "state": {"enum": list(r.RELEASED_STATES)},
                "observed_at": _ref("timestamp"),
                "observer": _ref("text"),
                "method": _ref("text"),
            }
        ),
        "collectorArtifact": _object(
            {
                "id": _ref("text"),
                "case_id": _ref("text"),
                "sha256": _ref("digest"),
                "task_id": nullable_text,
                "output_hash": {"anyOf": [_ref("digest"), {"type": "null"}]},
                "file_hashes": {
                    "anyOf": [
                        {
                            "type": "object",
                            "minProperties": 1,
                            "propertyNames": _ref("relativePath"),
                            "additionalProperties": _ref("digest"),
                        },
                        {"type": "null"},
                    ]
                },
            }
        ),
        "collectorExport": _object(
            {
                "schema_version": {"const": r.COLLECTOR_EXPORT_SCHEMA},
                "record": {"enum": ["assignment", "vote"]},
                "collector": _ref("collectorPin"),
                "assignment": _object(
                    {
                        "id": _ref("text"),
                        "case_id": _ref("text"),
                        "phase": {"enum": list(r.COLLECTOR_PHASES)},
                        "reviewer": _ref("reviewer"),
                        "left_id": _ref("text"),
                        "right_id": _ref("text"),
                        "left_hash": _ref("digest"),
                        "right_hash": _ref("digest"),
                        "task_id": nullable_text,
                        "created": {"type": "string", "minLength": 19, "maxLength": 40},
                    }
                ),
                "artifacts": _object(
                    {
                        "left": _ref("collectorArtifact"),
                        "right": _ref("collectorArtifact"),
                    }
                ),
                "task": {
                    "anyOf": [
                        _object(
                            {
                                "id": _ref("text"),
                                "version": _ref("count"),
                                "case_id": _ref("text"),
                                "rubric_id": nullable_text,
                                "rubric_hash": {
                                    "anyOf": [_ref("digest"), {"type": "null"}]
                                },
                                "status": _text(64),
                            }
                        ),
                        {"type": "null"},
                    ]
                },
                "vote": {
                    "anyOf": [
                        _object(
                            {
                                "id": _ref("text"),
                                "assignment_id": _ref("text"),
                                "reviewer": _ref("reviewer"),
                                "choice": {"enum": list(r.COLLECTOR_CHOICES)},
                                "reason": _free_text(required=False),
                                "tags": _ref("tags"),
                                "created": {
                                    "type": "string",
                                    "minLength": 19,
                                    "maxLength": 40,
                                },
                            }
                        ),
                        {"type": "null"},
                    ]
                },
                "invalidated": {"type": "boolean"},
            }
        ),
    }
    answer = {
        "assignment_id": _ref("text"),
        "reviewer_subject": _ref("reviewer"),
        "choice": {"enum": list(r.CHOICES)},
        "reason": _free_text(required=False),
        "tags": _ref("tags"),
    }
    defs.update(
        {
            "createdPayload": _object(
                {
                    "attempts": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": _ref("savedAttemptRef"),
                    },
                    "rubric": _ref("frozenFile"),
                    "judge_protocol": _object(
                        {
                            "id": _text(256),
                            "version": _text(256),
                            "sha256": _ref("digest"),
                            "bytes": {"type": "integer", "minimum": 1},
                            "locator": _ref("locator"),
                            "collector": _ref("collectorPin"),
                        }
                    ),
                }
            ),
            "releasedPayload": _object(
                {
                    "runtime_refs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 128,
                        "items": _object(
                            {
                                "attempt_id": _ref("digest"),
                                "provider": _ref("name"),
                                "runtime_id": _ref("text"),
                                "role": {"enum": list(r.RUNTIME_ROLES)},
                            }
                        ),
                    },
                    "released_at": _ref("timestamp"),
                    "release_proof": _object(
                        {
                            "sha256": _ref("digest"),
                            "observations": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 256,
                                "items": _ref("observation"),
                            },
                        }
                    ),
                }
            ),
            "assignedPayload": _object(
                {
                    "collector": _object(
                        {
                            "implementation": _ref("name"),
                            "commit": _ref("commit"),
                            "assignment_id": _ref("text"),
                            "export_sha256": _ref("digest"),
                        }
                    ),
                    "presentation": _object({"A": _ref("digest"), "B": _ref("digest")}),
                    "reviewer_subject": _ref("reviewer"),
                    "collector_export": {
                        "allOf": [
                            _ref("collectorExport"),
                            {"properties": {"record": {"const": "assignment"}}},
                        ]
                    },
                }
            ),
            "submittedPayload": _object(
                {
                    **answer,
                    "submitted_at": _ref("timestamp"),
                    "imported_from": _object(
                        {
                            "collector": _ref("name"),
                            "record_id": _ref("text"),
                            "export_sha256": _ref("digest"),
                        }
                    ),
                    "collector_export": {
                        "allOf": [
                            _ref("collectorExport"),
                            {"properties": {"record": {"const": "vote"}}},
                        ]
                    },
                }
            ),
            "correctedPayload": _object(
                {
                    "supersedes_event_id": _ref("uuid7"),
                    **answer,
                    "correction_reason": _free_text(required=True),
                    "corrected_at": _ref("timestamp"),
                }
            ),
            "retractedPayload": _object(
                {
                    "target_event_id": _ref("uuid7"),
                    "assignment_id": _ref("text"),
                    "reviewer_subject": _ref("reviewer"),
                    "reason": _free_text(required=True),
                    "retracted_at": _ref("timestamp"),
                }
            ),
            "reducedPayload": {
                "oneOf": [
                    _reduced(ready=True, first=True),
                    _reduced(ready=True, first=False),
                    _reduced(ready=False, first=True),
                    _reduced(ready=False, first=False),
                ]
            },
            "consumedPayload": _object(
                {
                    "label_id": _ref("uuid7"),
                    "label_revision": {"type": "integer", "minimum": 1},
                    "consumer_id": _ref("text"),
                    "consumer_receipt": _text(r.MAX_TEXT),
                    "consumed_at": _ref("timestamp"),
                }
            ),
            "withdrawnPayload": _object(
                {
                    "label_id": _ref("uuid7"),
                    "label_revision": {"type": "integer", "minimum": 1},
                    "reason": _free_text(required=True),
                    "consumer_disposition": {"enum": list(r.DISPOSITIONS)},
                    "consumption_event_ids": {
                        "type": "array",
                        "maxItems": 1024,
                        "uniqueItems": True,
                        "items": _ref("uuid7"),
                    },
                    "withdrawn_at": _ref("timestamp"),
                }
            ),
        }
    )
    return defs


def _reduced(*, ready: bool, first: bool) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "label_id": _ref("uuid7"),
        "revision": {"const": 1} if first else {"type": "integer", "minimum": 2},
    }
    if not first:
        fields["supersedes_revision"] = {"type": "integer", "minimum": 1}
    fields.update(
        {
            "source_event_id": _ref("uuid7"),
            "source_event_sha256": _ref("digest"),
            "status": {"const": "ready" if ready else "withheld"},
        }
    )
    if ready:
        fields["value"] = {"enum": [1, 0, 0.5]}
    else:
        fields["withheld_reason"] = {"enum": list(r.WITHHELD_REASONS)}
    fields["reducer"] = {"const": dict(r.REDUCER)}
    return _object(fields)


_PAYLOAD_DEF = {
    "assessment.created": "createdPayload",
    "compute.released": "releasedPayload",
    "assignment.issued": "assignedPayload",
    "review.submitted": "submittedPayload",
    "review.corrected": "correctedPayload",
    "review.retracted": "retractedPayload",
    "label.reduced": "reducedPayload",
    "label.consumed": "consumedPayload",
    "label.withdrawn": "withdrawnPayload",
}


def event_schema() -> dict[str, Any]:
    """The frozen envelope plus one payload rule per kind."""
    schema = copy.deepcopy(EVENT_ENVELOPE)
    schema["$defs"].update(_payload_defs())
    schema["allOf"] = [
        {
            "if": {"properties": {"kind": {"const": kind}}, "required": ["kind"]},
            "then": {"properties": {"payload": _ref(definition)}},
        }
        for kind, definition in _PAYLOAD_DEF.items()
    ]
    return schema


def label_schema() -> dict[str, Any]:
    """``benchflow.assessment-label.v1``: the label the log reduces to."""
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    uuid7 = {
        "type": "string",
        "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    }
    side = _object(
        {
            "attempt_id": digest,
            "rollout_id": {"type": "string", "minLength": 1},
            "result_sha256": digest,
            "output_manifest_sha256": digest,
            "review_artifact_sha256": digest,
        }
    )
    identity = _object(
        {"id": {"type": "string"}, "version": {"type": "string"}, "sha256": digest}
    )
    event_ref = _object({"event_id": uuid7, "event_sha256": digest})
    revision = _object(
        {
            "revision": {"type": "integer", "minimum": 1},
            "status": {"enum": list(r.LABEL_STATUSES)},
            "source_event_id": uuid7,
            "source_event_sha256": digest,
            "reduced_event_id": uuid7,
            "reduced_event_sha256": digest,
            "consumptions": {
                "type": "array",
                "items": _object(
                    {
                        "event_id": uuid7,
                        "event_sha256": digest,
                        "consumer_id": {"type": "string"},
                        "consumer_receipt": {"type": "string"},
                    }
                ),
            },
            "withdrawals": {
                "type": "array",
                "items": _object(
                    {
                        "event_id": uuid7,
                        "event_sha256": digest,
                        "consumer_disposition": {"enum": list(r.DISPOSITIONS)},
                        "reason": {"type": "string"},
                    }
                ),
            },
        },
        {
            "value": {"enum": [1, 0, 0.5]},
            "withheld_reason": {"enum": list(r.WITHHELD_REASONS)},
            "supersedes_revision": {"type": "integer", "minimum": 1},
        },
    )
    schema = _object(
        {
            "schema_version": {"const": r.LABEL_SCHEMA},
            "assessment_id": uuid7,
            "label_id": uuid7,
            "revision": {"type": "integer", "minimum": 1},
            "status": {"enum": ["ready", "withheld", "withdrawn", "stale"]},
            "state": {
                "enum": [
                    "answered",
                    "retracted",
                    "label_ready",
                    "consumed",
                    "withheld",
                    "withdrawn_for_future_use",
                ]
            },
            "consumable": {"type": "boolean"},
            "resolved": {"type": "boolean"},
            "unresolved_revisions": {
                "type": "array",
                "items": _object(
                    {
                        "label_revision": {"type": "integer", "minimum": 1},
                        "consumption_event_ids": {"type": "array", "items": uuid7},
                    }
                ),
            },
            "task_digest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "presentation": _object({"A": digest, "B": digest}),
            "attempts": _object({"A": side, "B": side}),
            "rubric": identity,
            "judge_protocol": identity,
            "collector": _object(
                {
                    "implementation": {"type": "string"},
                    "commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                    "assignment_id": {"type": "string"},
                    "assignment_export_sha256": digest,
                }
            ),
            "reviewer_subject": {"type": "string"},
            "source": _object(
                {**event_ref["properties"], "kind": {"enum": list(r.KINDS)}}
            ),
            "revisions": {"type": "array", "minItems": 1, "items": revision},
            "event_chain": _object(
                {
                    "length": {"type": "integer", "minimum": 1},
                    "head_event_seq": {"type": "integer", "minimum": 1},
                    "head_event_sha256": digest,
                }
            ),
            "reducer": {"const": dict(r.REDUCER)},
        },
        {
            "value": {"enum": [1, 0, 0.5]},
            "withheld_reason": {"enum": list(r.WITHHELD_REASONS)},
        },
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://benchflow.ai/schemas/benchflow-assessment-label.v1.schema.json",
        "title": "The deterministic label an assessment sidecar's events reduce to",
        **schema,
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "ready"}}},
                "then": {
                    "required": ["value"],
                    "not": {"required": ["withheld_reason"]},
                },
                "else": {"not": {"required": ["value"]}},
            },
            {
                "if": {"properties": {"status": {"const": "withheld"}}},
                "then": {"required": ["withheld_reason"]},
                "else": {"not": {"required": ["withheld_reason"]}},
            },
        ],
    }
    return schema


def write_schemas(directory: str | Path) -> list[Path]:
    """Write both schema files into ``directory`` (the docs' schemas folder)."""
    written = []
    for name, schema in ((EVENT_FILE, event_schema()), (LABEL_FILE, label_schema())):
        path = Path(directory) / name
        path.write_text(json.dumps(schema, indent=2) + "\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write_schemas(sys.argv[1] if len(sys.argv) > 1 else "."):
        print(path)
