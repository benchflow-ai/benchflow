#!/usr/bin/env python3
"""Validate adapter-release-set evidence artifacts.

The release suite manifest declares adapter coverage as a release blocker, but
the suite planner intentionally does not launch benchmark jobs yet. This checker
turns the current manual parity/smoke artifacts into an executable gate:

- Harvey LAB and ProgramBench use checked-in parity_experiment.json files.
- SkillsBench uses a representative result.json from a BenchFlow run.
- Open adapter PRs can be supplied as NAME=/path/to/worktree roots.

Usage::

    python tests/integration/check_adapter_evidence.py \
      --skillsbench-result dogfood/.../result.json

    python tests/integration/check_adapter_evidence.py \
      --skillsbench-result dogfood/.../result.json \
      --open-pr-root HILBench=/tmp/benchflow-pr279-triage \
      --open-pr-root OpaqueToolsBench=/tmp/benchflow-pr280-triage \
      --open-pr-root CLBench=/tmp/benchflow-pr283-triage

Guards: ENG-89 adapter release-set grooming.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    adapter: str
    status: str
    message: str
    path: Path


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _missing(adapter: str, path: Path) -> Finding:
    return Finding(adapter, "fail", f"missing evidence file: {path}", path)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def check_harvey_lab(repo_root: Path) -> Finding:
    adapter = "Harvey LAB"
    path = repo_root / "benchmarks" / "harvey-lab" / "parity_experiment.json"
    if not path.exists():
        return _missing(adapter, path)

    data = _load_json(path)
    if not isinstance(data, list):
        return Finding(adapter, "fail", "expected a list of parity experiments", path)

    e2e = next(
        (
            item
            for item in data
            if isinstance(item, dict)
            and item.get("benchmark") == "harvey-lab"
            and item.get("experiment") == "end-to-end"
        ),
        None,
    )
    if not isinstance(e2e, dict):
        return Finding(adapter, "fail", "missing end-to-end parity experiment", path)

    parity_tasks = _number(e2e.get("parity_tasks"))
    trials = _number(e2e.get("trials"))
    metrics = e2e.get("metrics")
    if not parity_tasks or parity_tasks <= 0:
        return Finding(adapter, "fail", "end-to-end parity has no tasks", path)
    if not trials or trials <= 0:
        return Finding(adapter, "fail", "end-to-end parity has no trials", path)
    if not isinstance(metrics, list) or not metrics:
        return Finding(adapter, "fail", "end-to-end parity has no metrics", path)

    return Finding(
        adapter,
        "pass",
        f"end-to-end parity recorded for {int(parity_tasks)} tasks x {int(trials)} trials",
        path,
    )


def check_programbench(repo_root: Path) -> Finding:
    adapter = "ProgramBench"
    path = repo_root / "benchmarks" / "programbench" / "parity_experiment.json"
    if not path.exists():
        return _missing(adapter, path)

    data = _load_json(path)
    if not isinstance(data, dict):
        return Finding(adapter, "fail", "expected a parity experiment mapping", path)

    pipeline = data.get("pipeline_parity")
    agent = data.get("agent_parity")
    if not isinstance(pipeline, dict) or not isinstance(agent, dict):
        return Finding(
            adapter, "fail", "missing pipeline or agent parity section", path
        )

    for label, section in (("pipeline", pipeline), ("agent", agent)):
        tasks = _number(section.get("tasks_tested"))
        summary = section.get("summary")
        if not tasks or tasks <= 0:
            return Finding(adapter, "fail", f"{label} parity has no tasks", path)
        if not isinstance(summary, dict):
            return Finding(adapter, "fail", f"{label} parity has no summary", path)
        if summary.get("mismatch") != 0:
            return Finding(adapter, "fail", f"{label} parity reports mismatches", path)

    return Finding(
        adapter,
        "pass",
        f"pipeline parity {int(pipeline['tasks_tested'])} tasks; agent parity {int(agent['tasks_tested'])} tasks",
        path,
    )


def check_skillsbench_result(path: Path | None) -> Finding:
    adapter = "SkillsBench"
    if path is None:
        return Finding(
            adapter,
            "fail",
            "representative result.json is required via --skillsbench-result",
            Path("<missing>"),
        )
    if not path.exists():
        return _missing(adapter, path)

    data = _load_json(path)
    if not isinstance(data, dict):
        return Finding(adapter, "fail", "result.json is not a JSON object", path)

    reward = _number((data.get("rewards") or {}).get("reward"))
    if data.get("error"):
        return Finding(adapter, "fail", f"run has error: {data['error']}", path)
    if data.get("verifier_error"):
        return Finding(
            adapter, "fail", f"run has verifier_error: {data['verifier_error']}", path
        )
    if reward is None:
        return Finding(adapter, "fail", "run has no numeric rewards.reward", path)
    if not data.get("task_name"):
        return Finding(adapter, "fail", "run has no task_name", path)

    return Finding(
        adapter,
        "pass",
        f"{data['task_name']} reward={reward:g} agent={data.get('agent', '?')}",
        path,
    )


def check_hilbench(root: Path) -> Finding:
    adapter = "HILBench"
    path = root / "benchmarks" / "hilbench" / "parity_experiment.json"
    if not path.exists():
        return _missing(adapter, path)

    data = _load_json(path)
    structural = data.get("structural_parity") if isinstance(data, dict) else None
    if not isinstance(structural, dict):
        return Finding(adapter, "fail", "missing structural_parity section", path)

    summary = structural.get("results_summary")
    if not isinstance(summary, dict):
        return Finding(adapter, "fail", "missing structural parity summary", path)
    if _number(summary.get("failed")) not in (0, 0.0):
        return Finding(adapter, "fail", "structural parity reports failures", path)
    passed = _number(summary.get("passed"))
    if not passed or passed <= 0:
        return Finding(adapter, "fail", "structural parity has no passing tasks", path)

    eval_parity = data.get("eval_parity")
    if isinstance(eval_parity, dict) and eval_parity.get("status") == "blocked":
        return Finding(
            adapter,
            "blocked",
            f"structural parity passed {int(passed)} tasks; eval parity blocked: {eval_parity.get('blocker')}",
            path,
        )

    return Finding(
        adapter,
        "pass",
        f"structural parity passed {int(passed)} tasks",
        path,
    )


def check_opaquetoolsbench(root: Path) -> Finding:
    adapter = "OpaqueToolsBench"
    path = root / "benchmarks" / "opaquetoolsbench" / "parity_experiment.json"
    if not path.exists():
        return _missing(adapter, path)

    data = _load_json(path)
    if not isinstance(data, dict):
        return Finding(adapter, "fail", "expected parity experiment mapping", path)
    results = data.get("results")
    eval_parity = data.get("eval_parity")
    security = data.get("security_parity")
    if not isinstance(results, dict):
        return Finding(adapter, "fail", "missing structural results", path)
    total = _number(results.get("total_tasks"))
    failed = _number(results.get("failed"))
    if not total or total <= 0 or failed not in (0, 0.0):
        return Finding(adapter, "fail", "structural parity is not clean", path)
    if not isinstance(eval_parity, dict) or eval_parity.get("status") != "passed":
        return Finding(adapter, "fail", "eval parity did not pass", path)
    if not isinstance(security, dict) or security.get("status") != "passed":
        return Finding(adapter, "fail", "security parity did not pass", path)

    return Finding(
        adapter,
        "pass",
        f"structural {int(total)}/{int(total)} plus eval/security parity passed",
        path,
    )


def check_clbench(root: Path) -> Finding:
    adapter = "CLBench"
    path = root / "benchmarks" / "clbench" / "parity_experiment.json"
    if not path.exists():
        return _missing(adapter, path)

    data = _load_json(path)
    if not isinstance(data, dict):
        return Finding(adapter, "fail", "expected parity experiment mapping", path)

    structural = data.get("structural_parity")
    eval_parity = data.get("eval_parity")
    e2e = data.get("e2e_parity")
    dogfood = data.get("dogfooding")
    for label, section in (
        ("structural", structural),
        ("eval", eval_parity),
        ("e2e", e2e),
    ):
        if not isinstance(section, dict):
            return Finding(adapter, "fail", f"missing {label} parity section", path)
        tested = _number(section.get("tasks_tested"))
        passed = _number(section.get("passed"))
        if not tested or not passed or passed < tested:
            return Finding(adapter, "fail", f"{label} parity did not pass", path)

    if not isinstance(dogfood, dict):
        return Finding(adapter, "fail", "missing dogfooding section", path)
    docker_build = dogfood.get("docker_build")
    docker_eval = dogfood.get("docker_eval")
    if not isinstance(docker_build, dict) or docker_build.get("result") != "success":
        return Finding(adapter, "fail", "docker build dogfood did not pass", path)
    if not isinstance(docker_eval, dict) or docker_eval.get("result") != "success":
        return Finding(adapter, "fail", "docker eval dogfood did not pass", path)

    return Finding(
        adapter,
        "pass",
        f"e2e parity passed {int(e2e['passed'])}/{int(e2e['tasks_tested'])} plus Docker dogfood",
        path,
    )


OPEN_PR_CHECKS = {
    "hilbench": check_hilbench,
    "opaquetoolsbench": check_opaquetoolsbench,
    "clbench": check_clbench,
}


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/worktree")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected NAME=/path/to/worktree")
    return name.strip(), Path(path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate adapter-release-set parity and smoke evidence."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="BenchFlow repo root for merged adapter evidence.",
    )
    parser.add_argument(
        "--skillsbench-result",
        type=Path,
        help="Representative SkillsBench result.json from a BenchFlow run.",
    )
    parser.add_argument(
        "--open-pr-root",
        action="append",
        type=parse_root,
        default=[],
        metavar="NAME=PATH",
        help="Optional open adapter PR worktree root. Names: HILBench, OpaqueToolsBench, CLBench.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit zero when evidence is blocked but otherwise valid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()

    findings = [
        check_harvey_lab(repo_root),
        check_programbench(repo_root),
        check_skillsbench_result(args.skillsbench_result),
    ]

    for name, root in args.open_pr_root:
        key = name.lower().replace("-", "").replace("_", "")
        check = OPEN_PR_CHECKS.get(key)
        if check is None:
            valid = ", ".join(sorted(OPEN_PR_CHECKS))
            findings.append(
                Finding(
                    name, "fail", f"unknown open PR adapter name; valid: {valid}", root
                )
            )
            continue
        findings.append(check(root.resolve()))

    width = max(len(f.adapter) for f in findings)
    print("Adapter release evidence")
    print("-" * 80)
    for finding in findings:
        print(
            f"{finding.adapter:<{width}}  {finding.status.upper():<7}  {finding.message}"
        )
        print(f"{'':<{width}}           {finding.path}")
    print("-" * 80)

    has_fail = any(f.status == "fail" for f in findings)
    has_blocked = any(f.status == "blocked" for f in findings)
    if has_fail:
        return 1
    if has_blocked and not args.allow_blocked:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
