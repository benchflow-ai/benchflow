#!/usr/bin/env python3
"""Plan and run supported integration suite lanes from a declarative manifest.

Most lanes are still plan-only. The adapter-release-set lane has an executable
evidence checker because the v0.4 release audit already produces parity/smoke
artifacts for adapters and open adapter PRs. The trace-to-task-e2e lane is also
executable so ENG-93 evidence can be regenerated from the release manifest. The
hosted-env lane can validate env_uid/hub_url metadata and regenerate Harbor
inventory evidence.
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

    _validate_benchmark_axis(loaded["axes"].get("benchmarks"))

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


def _validate_benchmark_axis(benchmarks: Any) -> None:
    if benchmarks is None:
        return
    if not isinstance(benchmarks, Mapping):
        raise SuiteError("suite axes.benchmarks must be a mapping")

    seen_uids: set[str] = set()
    for group_name, entries in benchmarks.items():
        if not isinstance(entries, list) or not entries:
            raise SuiteError(f"benchmark group {group_name} must be a non-empty list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise SuiteError(
                    f"benchmark group {group_name} entries must be mappings"
                )

            name = entry.get("name")
            uid = entry.get("uid")
            source = entry.get("source")
            if not isinstance(name, str) or not name:
                raise SuiteError(f"benchmark group {group_name} entry missing name")
            if not isinstance(uid, str) or not uid:
                raise SuiteError(f"benchmark {name} missing uid")
            if uid in seen_uids:
                raise SuiteError(f"duplicate benchmark uid: {uid}")
            seen_uids.add(uid)

            if not isinstance(source, Mapping):
                raise SuiteError(f"benchmark {name} missing source mapping")
            for key in ("repo", "path", "ref"):
                if not isinstance(source.get(key), str) or not source[key]:
                    raise SuiteError(f"benchmark {name} source missing {key}")
            expected_uid = _benchmark_uid_from_source(source)
            if uid != expected_uid:
                raise SuiteError(
                    f"benchmark {name} uid must match source repo/path/ref: "
                    f"expected {expected_uid}, got {uid}"
                )


def _benchmark_uid_from_source(source: Mapping[str, Any]) -> str:
    return f"{source['repo']}:{source['path']}@{source['ref']}"


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
        for item in value:
            if isinstance(item, Mapping):
                todo = item.get("todo")
                if isinstance(todo, str):
                    todos.append(todo)
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


def collect_lane_todos(
    suite: Mapping[str, Any], lanes: list[Mapping[str, Any]]
) -> dict[str, list[str]]:
    """Return unresolved TODOs by lane id for selected lanes."""
    lane_todos: dict[str, list[str]] = {}
    for lane in lanes:
        expanded = expand_lane(suite, lane)
        todos = expanded["todos"]
        if todos:
            lane_todos[lane["id"]] = todos
    return lane_todos


def collect_lane_blockers(lanes: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return explicit lane blockers by lane id for selected lanes."""
    lane_blockers: dict[str, list[str]] = {}
    for lane in lanes:
        blocked_by = lane.get("blocked_by")
        if isinstance(blocked_by, list):
            blockers = [item for item in blocked_by if isinstance(item, str)]
            if blockers:
                lane_blockers[lane["id"]] = blockers
    return lane_blockers


def iter_lane_ids(suite: Mapping[str, Any]) -> Iterable[str]:
    for lane in suite["lanes"]:
        yield lane["id"]


def print_lane_list(suite: Mapping[str, Any]) -> None:
    print(f"Suite: {suite['suite']} v{suite['version']}")
    print("Lanes:")
    for lane in suite["lanes"]:
        if lane.get("status") == "backlog":
            marker = "backlog"
        else:
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
        if status := lane.get("status"):
            print(f"Status: {status}")
        if purpose := lane.get("purpose"):
            print(f"Purpose: {purpose}")
        if backlog_reason := lane.get("backlog_reason"):
            print(f"Backlog reason: {backlog_reason}")
        if evidence_dir := lane.get("evidence_dir"):
            print(f"Evidence dir: {evidence_dir}")
        if evidence_command := lane.get("evidence_command"):
            print(f"Evidence command: {evidence_command}")

        blocked_by = lane.get("blocked_by", [])
        if blocked_by:
            print("Blocked by:")
            for blocker in blocked_by:
                print(f"  - {blocker}")

        activation_criteria = lane.get("activation_criteria", [])
        if activation_criteria:
            print("Activation criteria:")
            for criterion in activation_criteria:
                print(f"  - {criterion}")

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
                print(f"  - {_render_value(source)}")

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
        if "uid" in value and "name" in value:
            suffix = f" (PR #{value['pr']})" if value.get("pr") else ""
            return f"{value['name']} [{value['uid']}]{suffix}"
        if "hub_url" in value and "name" in value:
            pattern = value.get("env_uid_pattern")
            suffixes = []
            if pattern:
                suffixes.append(f"env_uid={pattern}")
            selected = _render_selected_envs(value.get("selected_envs"))
            if selected:
                suffixes.append(f"selected={selected}")
            suffix = f" {'; '.join(suffixes)}" if suffixes else ""
            return f"{value['name']} [{value['hub_url']}]{suffix}"
        if "name" in value and "pr" in value:
            return f"{value['name']} (PR #{value['pr']})"
        if value.get("kind") == "local_task" and "path" in value:
            return f"local:{value['path']}"
        if value.get("kind") == "jsonl_trace" and "path" in value:
            suffix = f" ({value['format']})" if value.get("format") else ""
            return f"jsonl:{value['path']}{suffix}"
        if value.get("kind") == "opentraces_jsonl" and "path" in value:
            suffix = f" ({value['format']})" if value.get("format") else ""
            return f"opentraces:{value['path']}{suffix}"
        if value.get("kind") == "hf_dataset" and "alias" in value:
            repo = f" repo={value['repo']}" if value.get("repo") else ""
            status = f" status={value['status']}" if value.get("status") else ""
            return f"hf:{value['alias']}{repo}{status}"
        if value.get("kind") == "generated_trace_tasks":
            sources = value.get("sources", [])
            evidence_dir = value.get("evidence_dir")
            parts = [f"generated_trace_tasks ({len(sources)} sources)"]
            if evidence_dir:
                parts.append(f"evidence={evidence_dir}")
            return "; ".join(parts)
        if value.get("kind") == "harbor_skillsbench_baseline":
            repo = value.get("repo")
            ref = value.get("ref")
            path = value.get("path")
            return f"harbor-skillsbench:{repo}/{path}@{ref}"
        if value.get("kind") == "benchflow_results":
            path = value.get("path")
            return f"benchflow-results:{path}"
        if "env_uid" in value:
            hub_url = value.get("hub_url")
            suffix = f" [{hub_url}]" if hub_url else ""
            return f"{value['env_uid']}{suffix}"
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


def _render_selected_envs(selected_envs: Any) -> str:
    if not isinstance(selected_envs, list) or not selected_envs:
        return ""

    env_uids = []
    for selected in selected_envs:
        if isinstance(selected, Mapping) and isinstance(selected.get("env_uid"), str):
            env_uids.append(selected["env_uid"])

    if not env_uids:
        return ""
    if len(env_uids) <= 3:
        return ", ".join(env_uids)
    return ", ".join(env_uids[:3]) + f", +{len(env_uids) - 3} more"


def _run_adapter_evidence_checker(argv: list[str]) -> int:
    try:
        from tests.integration.check_adapter_evidence import main as evidence_main
    except ModuleNotFoundError:
        from check_adapter_evidence import main as evidence_main

    return evidence_main(argv)


def _run_trace_evidence_checker(argv: list[str]) -> int:
    try:
        from tests.integration.check_trace_to_task_evidence import (
            main as evidence_main,
        )
    except ModuleNotFoundError:
        from check_trace_to_task_evidence import main as evidence_main

    return evidence_main(argv)


def _run_hosted_env_evidence_checker(argv: list[str]) -> int:
    try:
        from tests.integration.check_hosted_env_evidence import main as evidence_main
    except ModuleNotFoundError:
        from check_hosted_env_evidence import main as evidence_main

    return evidence_main(argv)


def _run_skillsbench_harbor_parity_checker(argv: list[str]) -> int:
    try:
        from tests.integration.check_skillsbench_harbor_parity import (
            main as evidence_main,
        )
    except ModuleNotFoundError:
        from check_skillsbench_harbor_parity import main as evidence_main

    return evidence_main(argv)


def run_adapter_evidence(
    lanes: list[Mapping[str, Any]], args: argparse.Namespace
) -> int:
    """Run adapter-release-set evidence validation for selected lanes."""
    lane_ids = [lane["id"] for lane in lanes]
    unsupported = [lane_id for lane_id in lane_ids if lane_id != "adapter-release-set"]
    if unsupported:
        raise SuiteError(
            "--execute-adapter-evidence only supports adapter-release-set; "
            f"unsupported selected lane(s): {', '.join(unsupported)}"
        )
    if "adapter-release-set" not in lane_ids:
        raise SuiteError("--execute-adapter-evidence requires lane adapter-release-set")

    checker_args = ["--repo-root", str(args.adapter_evidence_repo_root)]
    if args.skillsbench_result:
        checker_args.extend(["--skillsbench-result", str(args.skillsbench_result)])
    for root in args.open_pr_root or []:
        checker_args.extend(["--open-pr-root", root])
    if args.allow_blocked:
        checker_args.append("--allow-blocked")

    return _run_adapter_evidence_checker(checker_args)


def run_trace_evidence(lanes: list[Mapping[str, Any]], args: argparse.Namespace) -> int:
    """Run trace-to-task-e2e evidence validation for selected lanes."""
    lane_ids = [lane["id"] for lane in lanes]
    unsupported = [lane_id for lane_id in lane_ids if lane_id != "trace-to-task-e2e"]
    if unsupported:
        raise SuiteError(
            "--execute-trace-evidence only supports trace-to-task-e2e; "
            f"unsupported selected lane(s): {', '.join(unsupported)}"
        )
    if "trace-to-task-e2e" not in lane_ids:
        raise SuiteError("--execute-trace-evidence requires lane trace-to-task-e2e")

    checker_args = [
        "--suite",
        str(args.suite),
        "--repo-root",
        str(args.trace_evidence_repo_root),
        "--sandbox",
        args.trace_evidence_sandbox,
    ]
    if args.trace_evidence_dir:
        checker_args.extend(["--evidence-dir", str(args.trace_evidence_dir)])
    if args.run_trace_eval:
        checker_args.append("--run-eval")

    return _run_trace_evidence_checker(checker_args)


def run_hosted_env_evidence(
    lanes: list[Mapping[str, Any]], args: argparse.Namespace
) -> int:
    """Run hosted-env compatibility-board evidence validation."""
    lane_ids = [lane["id"] for lane in lanes]
    unsupported = [
        lane_id for lane_id in lane_ids if lane_id != "hosted-env-compatibility-board"
    ]
    if unsupported:
        raise SuiteError(
            "--execute-hosted-env-evidence only supports hosted-env-compatibility-board; "
            f"unsupported selected lane(s): {', '.join(unsupported)}"
        )
    if "hosted-env-compatibility-board" not in lane_ids:
        raise SuiteError(
            "--execute-hosted-env-evidence requires lane hosted-env-compatibility-board"
        )

    checker_args = ["--suite", str(args.suite)]
    if args.hosted_env_evidence_dir:
        checker_args.extend(["--evidence-dir", str(args.hosted_env_evidence_dir)])
    if args.harbor_inventory_limit is not None:
        checker_args.extend(
            ["--harbor-inventory-limit", str(args.harbor_inventory_limit)]
        )

    return _run_hosted_env_evidence_checker(checker_args)


def run_skillsbench_harbor_parity(
    lanes: list[Mapping[str, Any]], args: argparse.Namespace
) -> int:
    """Run SkillsBench-vs-Harbor parity validation."""
    lane_ids = [lane["id"] for lane in lanes]
    unsupported = [
        lane_id for lane_id in lane_ids if lane_id != "skillsbench-harbor-parity"
    ]
    if unsupported:
        raise SuiteError(
            "--execute-skillsbench-harbor-parity only supports "
            "skillsbench-harbor-parity; unsupported selected lane(s): "
            f"{', '.join(unsupported)}"
        )
    if "skillsbench-harbor-parity" not in lane_ids:
        raise SuiteError(
            "--execute-skillsbench-harbor-parity requires lane "
            "skillsbench-harbor-parity"
        )
    if args.skillsbench_harbor_benchflow_root is None:
        raise SuiteError("--skillsbench-harbor-benchflow-root is required")
    if args.skillsbench_harbor_baseline_root is None:
        raise SuiteError("--skillsbench-harbor-baseline-root is required")

    checker_args = [
        "--benchflow-root",
        str(args.skillsbench_harbor_benchflow_root),
        "--harbor-baseline-root",
        str(args.skillsbench_harbor_baseline_root),
        "--harbor-baseline-ref",
        args.skillsbench_harbor_baseline_ref,
        "--max-outcome-rate-delta",
        str(args.skillsbench_harbor_max_outcome_rate_delta),
        "--max-mean-reward-delta",
        str(args.skillsbench_harbor_max_mean_reward_delta),
        "--max-task-reward-delta",
        str(args.skillsbench_harbor_max_task_reward_delta),
        "--expected-benchflow-source-repo",
        args.skillsbench_harbor_expected_source_repo,
        "--expected-benchflow-source-path-prefix",
        args.skillsbench_harbor_expected_source_path_prefix,
        "--expected-benchflow-task-entrypoint",
        args.skillsbench_harbor_expected_task_entrypoint,
        "--expected-skill-mode",
        args.skillsbench_harbor_expected_skill_mode,
        "--expected-skill-source",
        args.skillsbench_harbor_expected_skill_source,
    ]
    if args.skillsbench_harbor_expected_source_sha:
        checker_args.extend(
            [
                "--expected-benchflow-source-sha",
                args.skillsbench_harbor_expected_source_sha,
            ]
        )
    for task in args.skillsbench_harbor_task or []:
        checker_args.extend(["--task", task])
    if args.skillsbench_harbor_no_require_trajectories:
        checker_args.append("--no-require-trajectories")
    return _run_skillsbench_harbor_parity_checker(checker_args)


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
        help="Execution profile to plan, such as near-term, release-gated-cli, or full-release.",
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
    parser.add_argument(
        "--fail-on-todo",
        action="store_true",
        help=(
            "With --dry-run, exit non-zero when selected lanes contain unresolved "
            "TODOs or explicit blocked_by entries."
        ),
    )
    parser.add_argument(
        "--execute-adapter-evidence",
        action="store_true",
        help="Execute adapter-release-set evidence validation.",
    )
    parser.add_argument(
        "--execute-trace-evidence",
        action="store_true",
        help="Execute trace-to-task-e2e evidence generation and validation.",
    )
    parser.add_argument(
        "--execute-hosted-env-evidence",
        action="store_true",
        help="Execute hosted-env compatibility-board evidence validation.",
    )
    parser.add_argument(
        "--execute-skillsbench-harbor-parity",
        action="store_true",
        help="Execute SkillsBench-vs-Harbor parity validation.",
    )
    parser.add_argument(
        "--adapter-evidence-repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root for merged adapter evidence (default: current directory).",
    )
    parser.add_argument(
        "--skillsbench-result",
        type=Path,
        help="Representative SkillsBench result.json for adapter evidence.",
    )
    parser.add_argument(
        "--open-pr-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Open adapter PR worktree root for adapter evidence.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="For adapter evidence mode, exit zero when evidence is blocked but otherwise valid.",
    )
    parser.add_argument(
        "--trace-evidence-repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root for trace-to-task evidence (default: current directory).",
    )
    parser.add_argument(
        "--trace-evidence-dir",
        type=Path,
        help="Override generated_trace_tasks.evidence_dir for trace evidence.",
    )
    parser.add_argument(
        "--run-trace-eval",
        action="store_true",
        help="For trace evidence mode, run oracle eval on generated tasks.",
    )
    parser.add_argument(
        "--trace-evidence-sandbox",
        default="docker",
        help="Sandbox for trace evidence oracle eval (default: docker).",
    )
    parser.add_argument(
        "--hosted-env-evidence-dir",
        type=Path,
        help="Directory for hosted-env evidence artifacts.",
    )
    parser.add_argument(
        "--harbor-inventory-limit",
        type=int,
        default=2,
        help="Number of Harbor registry task refs to inventory.",
    )
    parser.add_argument(
        "--skillsbench-harbor-benchflow-root",
        type=Path,
        help="BenchFlow result root to compare against the Harbor baseline.",
    )
    parser.add_argument(
        "--skillsbench-harbor-baseline-root",
        type=Path,
        help="Local pinned benchflow-ai/skillsbench-trajectories baseline root.",
    )
    parser.add_argument(
        "--skillsbench-harbor-baseline-ref",
        default="2d86fe82f6a06f7c7b3a22a3ae90d554d0e9655c",
        help="Pinned skillsbench-trajectories ref expected at the baseline root.",
    )
    parser.add_argument(
        "--skillsbench-harbor-task",
        action="append",
        default=[],
        help="SkillsBench task to compare. May be repeated.",
    )
    parser.add_argument(
        "--skillsbench-harbor-no-require-trajectories",
        action="store_true",
        help="Do not require trajectory artifacts during Harbor parity.",
    )
    parser.add_argument(
        "--skillsbench-harbor-max-outcome-rate-delta",
        type=float,
        default=0.25,
        help="Allowed outcome-rate drift for SkillsBench-vs-Harbor parity.",
    )
    parser.add_argument(
        "--skillsbench-harbor-max-mean-reward-delta",
        type=float,
        default=0.25,
        help="Allowed mean-reward drift for SkillsBench-vs-Harbor parity.",
    )
    parser.add_argument(
        "--skillsbench-harbor-max-task-reward-delta",
        type=float,
        default=0.0,
        help="Allowed per-task reward movement outside Harbor observed range.",
    )
    parser.add_argument(
        "--skillsbench-harbor-expected-source-repo",
        default="benchflow-ai/skillsbench",
        help="Expected BenchFlow source.repo for SkillsBench parity artifacts.",
    )
    parser.add_argument(
        "--skillsbench-harbor-expected-source-path-prefix",
        default="tasks",
        help="Expected prefix for BenchFlow source.path values.",
    )
    parser.add_argument(
        "--skillsbench-harbor-expected-task-entrypoint",
        default="task.md",
        help="Required native task entrypoint in BenchFlow source.file_hashes.",
    )
    parser.add_argument(
        "--skillsbench-harbor-expected-source-sha",
        help="Optional exact BenchFlow source.resolved_sha for all compared results.",
    )
    parser.add_argument(
        "--skillsbench-harbor-expected-skill-mode",
        default="no-skill",
        help="Expected canonical BenchFlow skill_mode for the Harbor parity lane.",
    )
    parser.add_argument(
        "--skillsbench-harbor-expected-skill-source",
        default="none",
        help="Expected canonical BenchFlow skill_source for the Harbor parity lane.",
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
        execution_modes = [
            args.execute_adapter_evidence,
            args.execute_trace_evidence,
            args.execute_hosted_env_evidence,
            args.execute_skillsbench_harbor_parity,
        ]
        if args.dry_run and any(execution_modes):
            parser.error("use either --dry-run or an execution mode, not both")
        if sum(bool(mode) for mode in execution_modes) > 1:
            parser.error("select only one execution mode")

        profile = None
        lane_ids = args.lanes
        if args.profile:
            lane_ids = select_profile_lane_ids(suite, args.profile)
            profile = suite["execution_profiles"][args.profile]

        lanes = select_lanes(suite, lane_ids)
        if args.dry_run:
            print_plan(suite, lanes, profile_id=args.profile, profile=profile)
            if args.fail_on_todo:
                lane_todos = collect_lane_todos(suite, lanes)
                lane_blockers = collect_lane_blockers(lanes)
                if lane_todos or lane_blockers:
                    details = _format_unresolved_release_gate(lane_todos, lane_blockers)
                    print(
                        "unresolved TODOs or blocked lanes in selected lane(s): "
                        f"{details}",
                        file=sys.stderr,
                    )
                    return 1
            return 0

        if args.execute_adapter_evidence:
            return run_adapter_evidence(lanes, args)

        if args.execute_trace_evidence:
            return run_trace_evidence(lanes, args)

        if args.execute_hosted_env_evidence:
            return run_hosted_env_evidence(lanes, args)

        if args.execute_skillsbench_harbor_parity:
            return run_skillsbench_harbor_parity(lanes, args)

        parser.error(
            "execution is not implemented yet; pass --dry-run or "
            "--execute-adapter-evidence, --execute-trace-evidence, or "
            "--execute-hosted-env-evidence, or "
            "--execute-skillsbench-harbor-parity"
        )
    except SuiteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


def _format_unresolved_release_gate(
    lane_todos: Mapping[str, list[str]],
    lane_blockers: Mapping[str, list[str]],
) -> str:
    lane_ids = list(dict.fromkeys([*lane_todos.keys(), *lane_blockers.keys()]))
    details = []
    for lane_id in lane_ids:
        parts = []
        if todos := lane_todos.get(lane_id):
            parts.append(f"TODOs: {len(todos)}")
        if blockers := lane_blockers.get(lane_id):
            parts.append(f"blocked: {len(blockers)}")
        details.append(f"{lane_id} ({', '.join(parts)})")
    return ", ".join(details)


if __name__ == "__main__":
    raise SystemExit(main())
