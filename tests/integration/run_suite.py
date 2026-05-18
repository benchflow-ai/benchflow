#!/usr/bin/env python3
"""Plan integration suite lanes from a declarative manifest.

This runner intentionally starts in dry-run mode only. It validates a suite
manifest, expands named lane axes, and prints an auditable execution plan
without launching benchmark jobs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SUITE = Path(__file__).with_name("suites") / "release.yaml"
REQUIRED_TOP_LEVEL = {"suite", "version", "policy", "axes", "lanes"}


class SuiteError(ValueError):
    """Raised when the suite manifest is invalid."""


def load_suite(path: Path) -> dict[str, Any]:
    """Load and validate a suite manifest."""
    if not path.exists():
        raise SuiteError(f"suite file not found: {path}")

    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SuiteError(f"suite file must contain a mapping: {path}")

    missing = REQUIRED_TOP_LEVEL - set(loaded)
    if missing:
        raise SuiteError(f"suite file missing top-level keys: {sorted(missing)}")

    if not isinstance(loaded["axes"], dict):
        raise SuiteError("suite axes must be a mapping")
    if not isinstance(loaded["lanes"], list) or not loaded["lanes"]:
        raise SuiteError("suite lanes must be a non-empty list")

    seen: set[str] = set()
    for lane in loaded["lanes"]:
        if not isinstance(lane, dict):
            raise SuiteError("each lane must be a mapping")
        lane_id = lane.get("id")
        if not isinstance(lane_id, str) or not lane_id:
            raise SuiteError("each lane must have a non-empty string id")
        if lane_id in seen:
            raise SuiteError(f"duplicate lane id: {lane_id}")
        seen.add(lane_id)

    profiles = loaded.get("execution_profiles")
    if profiles is not None:
        if not isinstance(profiles, Mapping):
            raise SuiteError("suite execution_profiles must be a mapping")
        for profile_id, profile in profiles.items():
            if not isinstance(profile_id, str) or not profile_id:
                raise SuiteError(
                    "each execution profile must have a non-empty string id"
                )
            if not isinstance(profile, Mapping):
                raise SuiteError(f"execution profile {profile_id} must be a mapping")
            profile_lanes = profile.get("lanes", [])
            if profile_lanes == "all":
                continue
            if not isinstance(profile_lanes, list) or not profile_lanes:
                raise SuiteError(
                    f"execution profile {profile_id} lanes must be a non-empty list or all"
                )
            unknown = sorted(set(profile_lanes) - seen)
            if unknown:
                raise SuiteError(
                    f"execution profile {profile_id} references unknown lane id(s): "
                    f"{', '.join(unknown)}"
                )

    return loaded


def select_lanes(suite: Mapping[str, Any], lane_ids: list[str] | None) -> list[dict]:
    """Return selected lane mappings, preserving manifest order."""
    lanes = suite["lanes"]
    if not lane_ids:
        return lanes

    requested = set(lane_ids)
    selected = [lane for lane in lanes if lane["id"] in requested]
    found = {lane["id"] for lane in selected}
    missing = sorted(requested - found)
    if missing:
        raise SuiteError(f"unknown lane id(s): {', '.join(missing)}")
    return selected


def select_profile_lane_ids(
    suite: Mapping[str, Any], profile_id: str
) -> list[str] | None:
    """Return lane ids for a named execution profile."""
    profiles = suite.get("execution_profiles", {})
    if not isinstance(profiles, Mapping) or profile_id not in profiles:
        raise SuiteError(f"unknown execution profile: {profile_id}")

    profile = profiles[profile_id]
    lanes = profile.get("lanes", [])
    if lanes == "all":
        return None
    return list(lanes)


def resolve_axis_value(
    axes: Mapping[str, Any], axis_name: str, spec: Any
) -> tuple[list[Any], list[str]]:
    """Resolve a lane matrix entry against the suite axes.

    Returns resolved values plus TODO notes discovered in referenced task sets.
    """
    axis_defs = axes.get(axis_name)
    todos: list[str] = []

    if isinstance(spec, str):
        if isinstance(axis_defs, Mapping) and spec in axis_defs:
            return _normalize_values(axis_defs[spec], todos), todos
        return [spec], todos

    if isinstance(spec, list):
        values: list[Any] = []
        for item in spec:
            if (
                isinstance(axis_defs, Mapping)
                and isinstance(item, str)
                and item in axis_defs
            ):
                values.extend(_normalize_values(axis_defs[item], todos))
            else:
                values.append(item)
        return values, todos

    if isinstance(spec, Mapping):
        return [dict(spec)], todos

    return [spec], todos


def _normalize_values(value: Any, todos: list[str]) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        todo = value.get("todo")
        if isinstance(todo, str):
            todos.append(todo)
        return [dict(value)]
    return [value]


def expand_lane(suite: Mapping[str, Any], lane: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve all matrix axes for a lane into printable values."""
    axes = suite["axes"]
    matrix = lane.get("matrix", {})
    if not isinstance(matrix, Mapping):
        raise SuiteError(f"lane {lane['id']} matrix must be a mapping")

    expanded: dict[str, Any] = {}
    todos: list[str] = []
    for axis_name, spec in matrix.items():
        values, axis_todos = resolve_axis_value(axes, axis_name, spec)
        expanded[axis_name] = values
        todos.extend(axis_todos)

    return {"matrix": expanded, "todos": todos}


def iter_lane_ids(suite: Mapping[str, Any]) -> Iterable[str]:
    for lane in suite["lanes"]:
        yield lane["id"]


def print_lane_list(suite: Mapping[str, Any]) -> None:
    print(f"Suite: {suite['suite']} v{suite['version']}")
    print("Lanes:")
    for lane in suite["lanes"]:
        marker = "blocker" if lane.get("release_blocker") else "non-blocker"
        print(f"  - {lane['id']} ({marker})")


def print_profile_list(suite: Mapping[str, Any]) -> None:
    print(f"Suite: {suite['suite']} v{suite['version']}")
    print("Execution profiles:")
    profiles = suite.get("execution_profiles", {})
    if not isinstance(profiles, Mapping) or not profiles:
        print("  (none)")
        return

    for profile_id, profile in profiles.items():
        lanes = profile.get("lanes", [])
        lane_text = "all lanes" if lanes == "all" else ", ".join(lanes)
        print(f"  - {profile_id}: {lane_text}")


def print_plan(
    suite: Mapping[str, Any],
    lanes: list[Mapping[str, Any]],
    profile_id: str | None = None,
    profile: Mapping[str, Any] | None = None,
) -> None:
    print(f"Suite: {suite['suite']} v{suite['version']}")
    print(f"Policy: {suite.get('policy', {}).get('on_failure', 'unspecified')}")
    tracking = suite.get("run_tracking", {})
    if isinstance(tracking, Mapping) and tracking.get("future_system"):
        print(f"Future tracker: {tracking['future_system']}")
    if profile_id:
        print(f"Profile: {profile_id}")
    if profile:
        if purpose := profile.get("purpose"):
            print(f"Profile purpose: {purpose}")
        benchmark_suites = profile.get("benchmark_suites", [])
        if benchmark_suites:
            print(f"Benchmark suites: {', '.join(benchmark_suites)}")
        sandboxes = profile.get("preferred_sandboxes", [])
        if sandboxes:
            print(f"Preferred sandboxes: {', '.join(sandboxes)}")
    print(f"Lanes selected: {len(lanes)}")
    print()

    for lane in lanes:
        expanded = expand_lane(suite, lane)
        print(f"## {lane['id']}")
        print(f"Release blocker: {bool(lane.get('release_blocker'))}")
        if purpose := lane.get("purpose"):
            print(f"Purpose: {purpose}")

        matrix = expanded["matrix"]
        if matrix:
            print("Matrix:")
            for axis_name, values in matrix.items():
                rendered = ", ".join(_render_value(v) for v in values)
                print(f"  {axis_name}: {rendered}")

        checks = lane.get("checks")
        if checks:
            print("Checks:")
            for check in checks:
                print(f"  - {check}")

        sources = lane.get("sources")
        if sources:
            print("Sources:")
            for source in sources:
                print(f"  - {source}")

        task_budget = lane.get("task_budget")
        if isinstance(task_budget, Mapping):
            print("Task budget:")
            for key, value in task_budget.items():
                print(f"  {key}: {value}")

        acceptance = lane.get("acceptance", [])
        if acceptance:
            print("Acceptance:")
            for item in acceptance:
                print(f"  - {item}")

        if expanded["todos"]:
            print("TODOs:")
            for todo in expanded["todos"]:
                print(f"  - {todo}")
        print()


def _render_value(value: Any) -> str:
    if isinstance(value, Mapping):
        if "name" in value and "pr" in value:
            return f"{value['name']} (PR #{value['pr']})"
        if "source" in value and isinstance(value["source"], Mapping):
            source = _render_value(value["source"])
            include = value.get("include")
            if isinstance(include, list):
                return f"{source} ({len(include)} tasks)"
            return source
        if "repo" in value and "path" in value:
            ref = value.get("ref", "default")
            return f"{value['repo']}/{value['path']}@{ref}"
        if "todo" in value:
            return f"TODO: {value['todo']}"
        if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            return ", ".join(f"{k}={v}" for k, v in value.items())
        return str(dict(value))
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan BenchFlow integration suite lanes from a manifest."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE,
        help=f"Suite manifest path (default: {DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--lane",
        action="append",
        dest="lanes",
        help="Lane id to plan. May be repeated. Defaults to all lanes.",
    )
    parser.add_argument(
        "--profile",
        help="Execution profile to plan, such as near-term or full-release.",
    )
    parser.add_argument(
        "--list-lanes",
        action="store_true",
        help="List lane ids and exit.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List execution profile ids and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded lane plan without executing jobs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        suite = load_suite(args.suite)
        if args.list_profiles:
            print_profile_list(suite)
            return 0

        if args.list_lanes:
            print_lane_list(suite)
            return 0

        if args.profile and args.lanes:
            parser.error("use either --profile or --lane, not both")

        if not args.dry_run:
            parser.error("execution is not implemented yet; pass --dry-run")

        profile = None
        lane_ids = args.lanes
        if args.profile:
            lane_ids = select_profile_lane_ids(suite, args.profile)
            profile = suite["execution_profiles"][args.profile]

        lanes = select_lanes(suite, lane_ids)
        print_plan(suite, lanes, profile_id=args.profile, profile=profile)
    except SuiteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
