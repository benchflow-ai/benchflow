"""No raw request target is kept; live and re-audit decide from the same relations.

A fetch's relation to every declared route is computed once, from the
complete request, when it is observed. Only those relations (bound to the
digest of the route table) and the redacted preview are stored, so a
re-audit reaches the live verdict on any trajectory, and no credential in
a URL reaches a stored record, an event payload or the event chain's hash
input. Relations computed against a different route table are unknown.
"""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from benchflow.integrity.contract import TaskBinding
from benchflow.integrity.emit import INTEGRITY_DIRNAME, emit_integrity
from benchflow.integrity.events import BenchGuardEvent, canonical_event_payload
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_text,
    stamp_observation_ids,
)
from tests.test_integrity import _call, _evidence, _tool

SEED = 20261018
CASES = 160
HOST = "http://grading-service:9005"


def _binding(*resources: tuple[str, str, str]) -> TaskBinding:
    return TaskBinding.model_validate(
        {
            "resources": [
                {"id": rid, "selector": sel, "class": cls, "reason": "x"}
                for rid, sel, cls in resources
            ]
        }
    )


def _stored(trajectory):
    return [
        json.loads(line)
        for line in redact_acp_trajectory_jsonl(trajectory).splitlines()
        if line.strip()
    ]


def _events(evidence) -> list[dict]:
    path = evidence.trial_dir / INTEGRITY_DIRNAME / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _decision(evidence, claim) -> dict:
    """What a verdict rests on: the claim and every agent event's resource."""
    return {
        "core_verdict": claim["core_verdict"],
        "exploited": claim["exploited"],
        "final_flags": claim["final_flags"],
        "events": [
            (
                event["event_type"],
                event["resource"],
                event["resource_class"],
                bool(event["payload"].get("unknown_reward_relevant_object")),
            )
            for event in _events(evidence)
            if event["actor_class"] == "Agent"
        ],
    }


def _live_then_reaudit(tmp_path, trajectory, binding, *, reaudit_binding=None):
    stamp_observation_ids(trajectory)  # as the rollout does, before any write
    evidence = _evidence(tmp_path, trajectory, binding=binding)
    live = _decision(evidence, emit_integrity(evidence))
    audit = evidence
    if reaudit_binding is not None:
        audit = _evidence(tmp_path, trajectory, binding=reaudit_binding)
    audit = dataclasses.replace(
        audit, trajectory=_stored(trajectory), source="audit_trial"
    )
    return evidence, live, _decision(audit, emit_integrity(audit))


# --- R15-1: a credential in the URL does not change the re-audit -------------


@pytest.mark.parametrize(
    ("kind", "path", "blamed"),
    [
        ("navigate", "/public/api_key=dummy", False),
        ("fetch", "/public/api_key=dummy", False),
        ("navigate", "/_admin/api_key=dummy", True),
        ("fetch", "/_admin/api_key=dummy", True),
        ("navigate", "/_admin/api_key=dummy/../../public", False),
    ],
)
def test_a_url_credential_leaves_the_reaudit_verdict_unchanged(
    tmp_path, kind, path, blamed
):
    binding = _binding(("admin", f"{HOST}/_admin", "Hidden"))
    trajectory = [_tool(kind, url=HOST + path)]
    assert "dummy" not in redact_acp_trajectory_jsonl(trajectory)
    _, live, again = _live_then_reaudit(tmp_path, trajectory, binding)
    assert again == live
    assert live["exploited"] is blamed
    assert not live["final_flags"]["unknown_event"]


# --- generated trajectories ---------------------------------------------------

_HOSTS = (
    "http://grading-service:9005",
    "HTTP://Grading-Service:9005",
    "https://grading-service:9005",
    "http://grading-service:9006",
    "http://localhost:9005",
    "http://127.0.0.1:9005",
    "http://other:9005",
    "http:\\\\grading-service:9005",
)
_SEGMENTS = ("_admin", "api", "public", "admin", "x", "..", ".", "a\\b", "")
_SECRETS = (
    "api_key={s}",
    "token={s}",
    "access_token={s}",
    "password={s}",
    "sk-ant-api03-{s}",
)
_ROUTES = (
    ("admin", f"{HOST}/_admin", "Hidden"),
    ("api", f"{HOST}/api", "Hidden"),
    ("api-public", f"{HOST}/api/public", "Public"),
    ("api-admin", f"{HOST}/api/admin", "VerifierOnly"),
    ("local", "http://localhost:9005/_admin", "Hidden"),
)


def _call_id(rng: random.Random, index: int, n: int, secrets: list[str]) -> str:
    """A plain id, one redaction rewrites, or one that collides with another
    call's after redaction (both carry a credential form)."""
    roll = rng.random()
    if roll < 0.4:
        return f"c{n}"
    secret = f"zi{index}y{n}" + "k" * 24
    secrets.append(secret)
    return rng.choice(_SECRETS).format(s=secret)


def _case(rng: random.Random, index: int):
    routes = rng.sample(_ROUTES, rng.randint(1, 4))
    calls, secrets = [], []
    for n in range(rng.randint(1, 4)):
        parts = [rng.choice(_SEGMENTS) for _ in range(rng.randint(1, 4))]
        if rng.random() < 0.6:
            secret = f"zq{index}x{n}" + "k" * 24
            secrets.append(secret)
            spot = rng.randrange(len(parts) + 1)
            parts.insert(spot, rng.choice(_SECRETS).format(s=secret))
        url = rng.choice(_HOSTS) + "/" + "/".join(parts)
        kind = rng.choice(("fetch", "navigate", "WebFetch"))
        calls.append((kind, url, _call_id(rng, index, n, secrets)))
    return routes, calls, secrets


def _generated():
    rng = random.Random(SEED)
    return [_case(rng, index) for index in range(CASES)]


def test_live_and_reaudit_agree_on_generated_trajectories(tmp_path):
    differ = []
    for index, (routes, calls, _) in enumerate(_generated()):
        trajectory = [_tool(kind, call_id=cid, url=url) for kind, url, cid in calls]
        _, live, again = _live_then_reaudit(
            tmp_path / str(index), trajectory, _binding(*routes)
        )
        if again != live:
            differ.append((routes, calls, live, again))
    assert not differ, differ[:2]


# A shell command's file-path annotations keep the paths as the command text
# names them (integ, pre-existing; not a URL target): when the shell parser
# reads a URL it cannot recognise as a path, it lands here. Reported, not
# scanned.
PATH_ANNOTATIONS = frozenset(
    {"referenced_paths", "derived_targets", "possible_targets"}
)


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in PATH_ANNOTATIONS}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _documents(path):
    text = path.read_text(errors="replace")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.suffix == ".json":
        return [json.loads(text)]
    return [text]


def _scan(evidence, needles: list[str]) -> list[str]:
    """Every stored file under integrity/ and every event's hash input."""
    found = []
    out = evidence.trial_dir / INTEGRITY_DIRNAME
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        for document in _documents(path):
            text = json.dumps(_strip(document))
            found += [f"{path.name}: {n}" for n in needles if n in text]
    for event in _events(evidence):
        canonical = canonical_event_payload(BenchGuardEvent.model_validate(event))
        canonical = json.dumps(_strip(json.loads(canonical)))
        found += [
            f"hash input {event['event_id']}: {n}" for n in needles if n in canonical
        ]
    return found


def test_no_raw_target_or_credential_is_stored_or_hashed(tmp_path):
    raw_keys = ['"match_target"', '"whatwg"', '"literal"']
    leaks = []
    for index, (routes, calls, secrets) in enumerate(_generated()):
        if not secrets:
            continue
        trajectory = [_tool(kind, call_id=cid, url=url) for kind, url, cid in calls]
        # The same URLs in shell commands (their loopback and network lists).
        trajectory += [
            _call(f"curl -s '{url}'", call_id=f"s{cid}") for _, url, cid in calls
        ]
        stamp_observation_ids(trajectory)  # as the rollout does
        stored = redact_acp_trajectory_jsonl(trajectory)
        assert not any(secret in stored for secret in secrets), "not a credential"
        for source in ("rollout", "audit_trial"):
            case = tmp_path / f"{index}-{source}"
            evidence = _evidence(case, trajectory, binding=_binding(*routes))
            if source == "audit_trial":
                evidence = dataclasses.replace(
                    evidence, trajectory=_stored(trajectory), source=source
                )
            emit_integrity(evidence)
            leaks += _scan(evidence, [*secrets, *raw_keys])
    assert not leaks, leaks[:5]


def test_the_redactor_treats_each_generated_form_as_a_credential():
    for form in _SECRETS:
        text = "/" + form.format(s="zq0x0" + "k" * 24)
        assert "zq0x0" not in redact_trajectory_text(text), form


# --- relations against a different route table are unknown -------------------


@pytest.mark.parametrize(
    "changed",
    [
        (("admin", f"{HOST}/_admin", "VerifierOnly"),),
        (("admin", f"{HOST}/_admin", "Hidden"), ("api", f"{HOST}/api", "Public")),
        (("admin", f"{HOST}/_adm", "Hidden"),),
        (("staff", f"{HOST}/_admin", "Hidden"),),
    ],
)
@pytest.mark.parametrize("path", ["/_admin/x", "/public/y"])
def test_a_changed_route_table_makes_the_fetch_unknown(tmp_path, changed, path):
    binding = _binding(("admin", f"{HOST}/_admin", "Hidden"))
    trajectory = [_tool("fetch", url=HOST + path)]
    _, live, again = _live_then_reaudit(
        tmp_path, trajectory, binding, reaudit_binding=_binding(*changed)
    )
    assert not live["final_flags"]["unknown_event"]
    assert again["final_flags"]["unknown_event"] and not again["exploited"]


def test_the_same_route_table_reuses_the_relations(tmp_path):
    binding = _binding(("admin", f"{HOST}/_admin", "Hidden"))
    trajectory = [_tool("fetch", url=f"{HOST}/_admin/x")]
    _, live, again = _live_then_reaudit(
        tmp_path, trajectory, binding, reaudit_binding=binding
    )
    assert again == live and live["exploited"]
