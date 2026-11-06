"""Which declared route an observed fetch read: one resolution, live or re-audit.

No raw request target is ever kept. When a fetch is observed (in
:mod:`benchflow.integrity.agent_actions`, from the complete request before
any redaction or truncation), :func:`observe_relations` computes its relation
to every route the contract declares: ``yes``, ``no`` or ``ambiguous`` for
each plausible reading of the request (the URL Standard's; as written,
unless the client is a known browser), bound to :func:`routes_digest` of the
route table it was computed against. Only those relations (route ids and
those words) and the redacted display preview are stored.

:func:`resolve_route` decides from the relations alone, the way file
resources are selected: the most specific route, then the stronger
protection (:func:`benchflow.integrity.classify.resource_specificity`).
When the readings select different resources, or a route a reading may
address (two loopback names) could change the selection, the fetch is
``unknown``: never a blamed read, never certified.

The live run stores the relations beside the trajectory
(``integrity/route_keys.json``), keyed on each call's observation identity:
opaque, unique and stamped into the trajectory before any redaction
(:func:`benchflow.trajectories.types.stamp_observation_ids`). A re-audit reads the
redacted trajectory, so it reuses those relations when the digest matches its
own contract's routes. Missing relations, or a different digest, make the
fetch ``unknown``. Live and re-audit therefore decide from the same evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from benchflow.integrity.classify import resource_specificity
from benchflow.integrity.urls import (
    Target,
    literal_target,
    route_relation,
    route_target,
    whatwg_target,
)

ROUTE_KEYS_FILE = "route_keys.json"
ROUTE_KEYS_VERSION = 3

Outcome = Literal["read", "unknown", "no"]


@dataclass(frozen=True)
class Route:
    resource: str
    resource_class: str
    selector: str
    target: Target | None  # None: a selector this matcher cannot type


@dataclass(frozen=True)
class Resolution:
    outcome: Outcome
    resource: str | None = None


_UNKNOWN = Resolution("unknown")
_NO = Resolution("no")


def contract_routes(contract: Any) -> list[Route]:
    return [
        Route(item.id, item.resource_class, item.path, route_target(item.path))
        for item in contract.resources
        if item.kind == "service" and item.path
    ]


def routes_digest(routes: list[Route]) -> str:
    """The route table relations are computed against: ids, classes and
    typed selectors, in a canonical order."""
    table = sorted([r.resource, r.resource_class, r.selector] for r in routes)
    text = json.dumps(table, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def observe_relations(url: str, *, client: str, routes: list[Route]) -> dict:
    """A fetch's relation to every route, from its complete URL.

    ``readings`` holds one ``{route id: yes|no|ambiguous|unknown}`` map per
    plausible reading of the request; it is empty when no reading parses.
    Nothing from the URL itself is kept.
    """
    targets = [whatwg_target(url)]
    if client != "whatwg":
        targets.append(literal_target(url))
    readings: list[dict[str, str]] = []
    if any(target is not None for target in targets):
        for target in targets:
            readings.append(
                {
                    route.resource: "unknown"
                    if route.target is None
                    else route_relation(target, route.target)
                    for route in routes
                }
            )
    return {"routes": routes_digest(routes), "readings": readings}


def _rank(route: Route) -> tuple[int, int]:
    assert route.target is not None
    return resource_specificity(route.target[3].rstrip("/"), route.resource_class)


def _select(reading: Any, routes: list[Route]) -> Resolution:
    """The resource one reading of the request addresses, if any."""
    if not isinstance(reading, dict):
        return _UNKNOWN
    best: Route | None = None
    doubt: tuple[int, int] | None = None
    for route in routes:
        relation = reading.get(route.resource)
        if route.target is None or relation not in {"yes", "no", "ambiguous"}:
            return _UNKNOWN
        if relation == "yes" and (best is None or _rank(route) > _rank(best)):
            best = route
        elif relation == "ambiguous":
            doubt = max(doubt or _rank(route), _rank(route))
    if doubt is not None and (best is None or doubt >= _rank(best)):
        return _UNKNOWN  # a route it may address could outrank the selection
    return Resolution("read", best.resource) if best is not None else _NO


def resolve_route(relations: Any, routes: list[Route]) -> Resolution:
    """``read`` (with the resource), ``unknown`` or ``no`` for one fetch."""
    if not routes:
        return _NO
    if not isinstance(relations, dict) or relations.get("routes") != routes_digest(
        routes
    ):
        return _UNKNOWN  # no relations computed against these routes
    readings = relations.get("readings")
    if not isinstance(readings, list) or not readings:
        return _UNKNOWN  # no reading of the URL at all
    selections = {_select(reading, routes) for reading in readings}
    if len(selections) == 1:
        return selections.pop()
    return _UNKNOWN  # plausible readings select different resources


# --- storing the relations beside the trajectory --------------------------------


def _keys(records: list[dict[str, Any]]) -> list[str | None]:
    """Each fetch's key, aligned with ``records``: its observation identity
    (opaque and unique, stamped before any redaction and stored in the
    trajectory too) and its ordinal among that call's fetches. None for a
    record that is not a fetch, or a fetch without an identity, whose
    relations are then never stored or found."""
    seen: dict[str, int] = {}
    keys: list[str | None] = []
    for record in records:
        identity = record.get("observation_id")
        if "route_relations" not in record or not (
            isinstance(identity, str) and identity
        ):
            keys.append(None)
            continue
        ordinal = seen.get(identity, 0)
        seen[identity] = ordinal + 1
        keys.append(f"{identity}#{ordinal}")
    return keys


def route_keys(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The live run's relations for every fetch, to store."""
    return {
        "version": ROUTE_KEYS_VERSION,
        "observations": {
            key: record["route_relations"]
            for key, record in zip(_keys(records), records, strict=True)
            if key is not None
        },
    }


def restore_route_keys(
    records: list[dict[str, Any]], stored: Any
) -> list[dict[str, Any]]:
    """Re-audit: each fetch takes the live run's relations; the ones derived
    from the redacted trajectory are never used. A fetch without stored
    relations gets none, so it is unknown."""
    observations = (
        stored.get("observations")
        if isinstance(stored, dict) and stored.get("version") == ROUTE_KEYS_VERSION
        else None
    )
    restored = []
    for key, record in zip(_keys(records), records, strict=True):
        if "route_relations" not in record:
            restored.append(record)
            continue
        relations = (
            observations.get(key)
            if key is not None and isinstance(observations, dict)
            else None
        )
        restored.append(
            {
                **record,
                "route_relations": relations
                if isinstance(relations, dict)
                else {"missing": True},
            }
        )
    return restored


__all__ = [
    "ROUTE_KEYS_FILE",
    "Resolution",
    "Route",
    "contract_routes",
    "observe_relations",
    "resolve_route",
    "restore_route_keys",
    "route_keys",
    "routes_digest",
]
