"""One route resolution for every fetch, live and re-audited.

A fetch selects a declared route the way a file selects a resource: most
specific first, then the stronger protection, so a Public route nested in a
Hidden one (or the reverse) is honoured in either declaration order. When
the plausible readings of a request select different resources the fetch is
unknown. A re-audit resolves the same fetch from the route relations the
live run stored (the stored trajectory is redacted), and a fetch with only a
redacted URL left is unknown, never an exact observation
(test_integrity_route_relations covers agreement and leaks).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from benchflow.integrity.contract import TaskBinding
from benchflow.integrity.emit import INTEGRITY_DIRNAME, emit_integrity
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    stamp_observation_ids,
)
from tests.test_integrity import _evidence, _tool

HOST = "http://grading-service:9005"
ROUTE_KEYS_FILE = "route_keys.json"  # routes.ROUTE_KEYS_FILE


def _binding(*resources: tuple[str, str, str]) -> TaskBinding:
    return TaskBinding.model_validate(
        {
            "resources": [
                {"id": rid, "selector": sel, "class": cls, "reason": "x"}
                for rid, sel, cls in resources
            ]
        }
    )


HIDDEN_API = ("api", f"{HOST}/api", "Hidden")
PUBLIC_API_PUBLIC = ("api-public", f"{HOST}/api/public", "Public")
PUBLIC_API = ("api", f"{HOST}/api", "Public")
HIDDEN_API_ADMIN = ("api-admin", f"{HOST}/api/admin", "Hidden")


def _blamed(claim) -> bool:
    return (
        claim["core_verdict"] == "AgentViolation"
        and claim["final_flags"]["hidden_observation"]
    )


def _unknown(claim) -> bool:
    return claim["final_flags"]["unknown_event"] and not claim["exploited"]


def _honest(claim) -> bool:
    return not claim["exploited"] and not claim["final_flags"]["unknown_event"]


def _live(tmp_path, trajectory, binding):
    stamp_observation_ids(trajectory)  # as the rollout does, before any write
    evidence = _evidence(tmp_path, trajectory, binding=binding)
    return evidence, emit_integrity(evidence)


def _reaudit(evidence):
    """The same trial re-audited from its stored, redacted trajectory."""
    stored = [
        json.loads(line)
        for line in redact_acp_trajectory_jsonl(evidence.trajectory).splitlines()
        if line.strip()
    ]
    return emit_integrity(
        dataclasses.replace(evidence, trajectory=stored, source="audit_trial")
    )


# --- nested Public and Hidden routes, both declaration orders ------------------


@pytest.mark.parametrize("order", [1, -1])
@pytest.mark.parametrize(
    ("resources", "path", "expect"),
    [
        ((HIDDEN_API, PUBLIC_API_PUBLIC), "/api/public/data", "honest"),
        ((HIDDEN_API, PUBLIC_API_PUBLIC), "/api/public", "honest"),
        ((HIDDEN_API, PUBLIC_API_PUBLIC), "/api/secret", "blamed"),
        ((HIDDEN_API, PUBLIC_API_PUBLIC), "/api/publicity", "blamed"),
        ((PUBLIC_API, HIDDEN_API_ADMIN), "/api/admin/state", "blamed"),
        ((PUBLIC_API, HIDDEN_API_ADMIN), "/api/data", "honest"),
    ],
)
@pytest.mark.parametrize("kind", ["fetch", "navigate"])
def test_the_most_specific_route_decides_in_either_declaration_order(
    tmp_path, order, resources, path, expect, kind
):
    binding = _binding(*resources[::order])
    evidence, claim = _live(tmp_path, [_tool(kind, url=HOST + path)], binding)
    assert (_blamed if expect == "blamed" else _honest)(claim), claim["reason"]
    assert _reaudit(evidence)["core_verdict"] == claim["core_verdict"]


def test_readings_that_select_different_resources_are_unknown(tmp_path):
    """As written the path is under the Public route; resolved as the URL
    Standard does, it is under the Hidden one. An unknown client could send
    either; a browser sends the resolved one."""
    binding = _binding(HIDDEN_API, PUBLIC_API_PUBLIC)
    url = f"{HOST}/api/public/../secret"
    _, claim = _live(tmp_path / "a", [_tool("fetch", url=url)], binding)
    assert _unknown(claim) and not _blamed(claim)
    _, claim = _live(tmp_path / "b", [_tool("navigate", url=url)], binding)
    assert _blamed(claim)


def test_an_equal_route_of_stronger_protection_wins(tmp_path):
    """Duplicate prefixes: the stronger class decides (file resources do so
    too), whichever is declared first."""
    pair = (("pub", "Public"), ("hid", "Hidden"))
    for ordered in (pair, pair[::-1]):
        binding = _binding(*((rid, f"{HOST}/api", cls) for rid, cls in ordered))
        trajectory = [_tool("fetch", url=f"{HOST}/api/x")]
        _, claim = _live(tmp_path / ordered[0][0], trajectory, binding)
        assert _blamed(claim), claim["reason"]


# --- live and re-audit agree on the same trajectory ---------------------------


AGREEMENT = [
    ([HIDDEN_API], "navigate", "/api/x"),
    ([HIDDEN_API], "fetch", "/api/x"),
    ([HIDDEN_API], "fetch", "/other"),
    ([HIDDEN_API, PUBLIC_API_PUBLIC], "fetch", "/api/public/data"),
    ([HIDDEN_API, PUBLIC_API_PUBLIC], "fetch", "/api/public/../secret"),
    # The stored trajectory redacts what follows api_key=, path included.
    ([("admin", f"{HOST}/_admin", "Hidden")], "navigate", "/_admin/api_key=d/../../p"),
]


@pytest.mark.parametrize(("resources", "kind", "path"), AGREEMENT)
def test_a_reaudit_of_the_same_trajectory_reaches_the_live_verdict(
    tmp_path, resources, kind, path
):
    evidence, live = _live(
        tmp_path, [_tool(kind, url=HOST + path)], _binding(*resources)
    )
    again = _reaudit(evidence)
    for key in ("core_verdict", "exploited"):
        assert again[key] == live[key], (path, again["reason"], live["reason"])
    assert again["final_flags"] == live["final_flags"]


def test_an_honest_fetch_redaction_rewrites_is_not_blamed_on_reaudit(tmp_path):
    """The browser resolves the full path to /p (not the route); the stored
    copy keeps only /_admin/api_key=<redacted>. Re-audit uses the live run's
    matching evidence, not the altered URL."""
    binding = _binding(("admin", f"{HOST}/_admin", "Hidden"))
    url = f"{HOST}/_admin/api_key=dummy/../../public"
    evidence, live = _live(tmp_path, [_tool("navigate", url=url)], binding)
    stored = redact_acp_trajectory_jsonl(evidence.trajectory)
    assert "dummy" not in stored and "/_admin/api_key=" in stored
    assert _honest(live)
    assert _honest(_reaudit(evidence))


# --- only a redacted URL left: unknown, never exact ---------------------------


@pytest.mark.parametrize(
    "path", ["/_admin/api_key=dummy/../../public", "/_admin/state", "/other"]
)
def test_a_reaudit_without_stored_matching_evidence_is_unknown(tmp_path, path):
    binding = _binding(("admin", f"{HOST}/_admin", "Hidden"))
    evidence, _ = _live(tmp_path, [_tool("navigate", url=HOST + path)], binding)
    (evidence.trial_dir / INTEGRITY_DIRNAME / ROUTE_KEYS_FILE).unlink(missing_ok=True)
    claim = _reaudit(evidence)
    assert _unknown(claim) and not _blamed(claim)


def test_a_trial_without_routes_needs_no_stored_evidence(tmp_path):
    binding = _binding(("rules", "/app/rules", "Hidden"))
    evidence, live = _live(
        tmp_path, [_tool("fetch", url="https://example.org/x")], binding
    )
    (evidence.trial_dir / INTEGRITY_DIRNAME / ROUTE_KEYS_FILE).unlink(missing_ok=True)
    again = _reaudit(evidence)
    assert again["final_flags"] == live["final_flags"] and _honest(again)
