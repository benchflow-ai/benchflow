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
      --open-pr-root ContinualLearningBench=/tmp/benchflow-pr283-triage

Guards: ENG-89 adapter release-set grooming.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
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
    if not isinstance(eval_parity, dict):
        return Finding(adapter, "fail", "missing eval_parity section", path)

    status = eval_parity.get("status")
    if status == "blocked":
        blocker = str(eval_parity.get("blocker") or "")
        if (
            "hil-bench-swe-images" in blocker
            or "HUGGINGFACE_TOKEN" in blocker
            or "gated" in blocker.lower()
        ):
            return Finding(
                adapter,
                "fail",
                "structural parity passed "
                f"{int(passed)} tasks; eval parity evidence is stale: "
                "HILBench images are Hugging Face bucket objects. Update the "
                "adapter to translate hf://buckets/ScaleAI/hil-bench-swe-images/"
                "images/<uid>.tar.zst into "
                "https://huggingface.co/buckets/ScaleAI/hil-bench-swe-images/"
                "resolve/images/<uid>.tar.zst instead of using dataset "
                "hf_hub_download.",
                path,
            )
        return Finding(
            adapter,
            "blocked",
            f"structural parity passed {int(passed)} tasks; eval parity blocked: {blocker}",
            path,
        )
    if status != "passed":
        return Finding(
            adapter,
            "fail",
            f"structural parity passed {int(passed)} tasks; eval parity status is {status!r}, expected 'passed'",
            path,
        )

    return Finding(
        adapter,
        "pass",
        f"structural parity passed {int(passed)} tasks; eval parity passed",
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


def check_continuallearningbench(root: Path) -> Finding:
    adapter = "ContinualLearningBench"
    path = root / "benchmarks" / "continuallearningbench" / "parity_experiment.json"
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


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _cleanup_empty(cleanup: Mapping[str, Any]) -> bool:
    for key in (
        "docker_containers",
        "docker_networks",
        "cua_containers",
        "cloud_vms",
    ):
        value = cleanup.get(key)
        if value is None:
            continue
        number = _number(value)
        if number is None or number != 0:
            return False
    return True


def _check_environment_slice(
    adapter: str,
    path: Path,
    item: Mapping[str, Any],
    *,
    require_screenshot: bool,
) -> str | None:
    slice_id = _string(item.get("id")) or "<unnamed>"
    if not _string(item.get("task_id")):
        return f"{slice_id} missing task_id"
    if not _string(item.get("agent_adapter")):
        return f"{slice_id} missing agent_adapter"
    if not _string(item.get("sandbox_provider")):
        return f"{slice_id} missing sandbox_provider"
    sandbox_provider = str(item.get("sandbox_provider"))
    sandbox_provider_mode = _string(item.get("sandbox_provider_mode"))
    if sandbox_provider == "cua":
        if sandbox_provider_mode not in {"local", "cloud-probed"}:
            return (
                f"{slice_id} Cua evidence must declare sandbox_provider_mode "
                "as local or cloud-probed"
            )
        if sandbox_provider_mode == "local":
            cleanup = _mapping(item.get("cleanup"))
            if "cloud_vms" in cleanup:
                return f"{slice_id} local Cua evidence must not claim cloud VM cleanup"
        if sandbox_provider_mode == "cloud-probed":
            cleanup = _mapping(item.get("cleanup"))
            if "cloud_vms" not in cleanup:
                return f"{slice_id} cloud Cua evidence must include cloud_vms cleanup"
    if not _string(item.get("environment_adapter")):
        return f"{slice_id} missing environment_adapter"
    if not _string(item.get("benchmark_adapter")):
        return f"{slice_id} missing benchmark_adapter"

    parity = _mapping(item.get("parity"))
    compared = _number(parity.get("criteria_compared"))
    agreed = _number(parity.get("criteria_agreed"))
    reward_delta = _number(parity.get("reward_delta_max"))
    if not compared or compared <= 0:
        return f"{slice_id} has no compared parity criteria"
    if agreed != compared:
        return f"{slice_id} parity criteria did not all agree"
    if reward_delta is None or reward_delta > 0.02:
        return f"{slice_id} reward delta exceeds tolerance"
    if parity.get("eval_run_summary") is not True:
        return f"{slice_id} missing eval-run summary evidence"
    if parity.get("artifact_manifest") is not True:
        return f"{slice_id} missing artifact manifest evidence"
    if parity.get("adoption_report") is not True:
        return f"{slice_id} missing scrubbed adoption report evidence"
    if parity.get("loop_state") is not True:
        return f"{slice_id} missing adapter-adoption loop state evidence"
    if (
        item.get("environment_adapter") == "browser"
        and item.get("benchmark_adapter") == "browser-use"
    ):
        if parity.get("environment_readiness") is not True:
            return f"{slice_id} missing browser environment readiness evidence"
        if parity.get("runtime_trace_schema") is not True:
            return f"{slice_id} missing browser runtime trace schema evidence"
    if (
        item.get("environment_adapter") == "desktop"
        and parity.get("runtime_trace_schema") is not True
    ):
        return f"{slice_id} missing desktop runtime trace schema evidence"

    original = _mapping(item.get("original_runner"))
    benchflow = _mapping(item.get("benchflow_run"))
    if _number(original.get("score")) is None:
        return f"{slice_id} original runner has no score"
    if _number(benchflow.get("reward")) is None:
        return f"{slice_id} BenchFlow run has no reward"
    if not _number(benchflow.get("trajectory_steps_min")):
        return f"{slice_id} BenchFlow run has no trajectory steps"
    if not _number(benchflow.get("tool_calls_min")):
        return f"{slice_id} BenchFlow run has no tool calls"
    screenshot_min = _number(benchflow.get("screenshots_min"))
    if require_screenshot and (screenshot_min is None or screenshot_min <= 0):
        return f"{slice_id} BenchFlow run has no screenshot evidence"

    cleanup = _mapping(item.get("cleanup"))
    if not cleanup or not _cleanup_empty(cleanup):
        return f"{slice_id} cleanup evidence is missing or non-empty"

    commands = _list(item.get("commands"))
    if not commands:
        return f"{slice_id} has no reproduction commands"
    if not all(isinstance(command, str) and command for command in commands):
        return f"{slice_id} has an invalid reproduction command"

    return None


def _check_environment_evidence(
    *,
    adapter: str,
    path: Path,
    expected_benchmark: str,
    expected_status: str,
    min_slices: int,
    require_screenshot: bool,
    require_cua_cloud_failure_probe: bool = False,
    require_browser_use_official_probe: bool = False,
    require_cookbook_support_report: bool = False,
    require_stagehand_official_parity: bool = False,
) -> Finding:
    if not path.exists():
        return _missing(adapter, path)
    data = _load_json(path)
    if not isinstance(data, dict):
        return Finding(adapter, "fail", "expected evidence mapping", path)
    if data.get("schema") != "benchflow.environment-adapter-evidence.v1":
        return Finding(adapter, "fail", "unexpected evidence schema", path)
    if data.get("benchmark") != expected_benchmark:
        return Finding(adapter, "fail", "evidence benchmark name mismatch", path)
    if data.get("status") != expected_status:
        return Finding(adapter, "fail", f"expected status {expected_status!r}", path)

    slices = _list(data.get("slices"))
    if len(slices) < min_slices:
        return Finding(adapter, "fail", "not enough verified slices", path)
    for item in slices:
        if not isinstance(item, dict):
            return Finding(adapter, "fail", "slice entry is not a mapping", path)
        issue = _check_environment_slice(
            adapter, path, item, require_screenshot=require_screenshot
        )
        if issue:
            return Finding(adapter, "fail", issue, path)

    gaps = _list(data.get("gaps"))
    if not gaps:
        return Finding(adapter, "fail", "evidence must name remaining gaps", path)
    if require_cua_cloud_failure_probe:
        cloud_issue = _check_cua_cloud_failure_probe(
            _mapping(data.get("cloud_failure_probe"))
        )
        if cloud_issue:
            return Finding(adapter, "fail", cloud_issue, path)
    if require_browser_use_official_probe:
        browser_use_issue = _check_browser_use_official_probe(
            _list(data.get("additional_evidence"))
        )
        if browser_use_issue:
            return Finding(adapter, "fail", browser_use_issue, path)
    if require_cookbook_support_report:
        support_issue = _check_cookbook_support_report(
            _mapping(data.get("unsupported_summary"))
        )
        if support_issue:
            return Finding(adapter, "fail", support_issue, path)
    if require_stagehand_official_parity:
        stagehand_issue = _check_stagehand_official_parity(
            _list(data.get("additional_evidence"))
        )
        if stagehand_issue:
            return Finding(adapter, "fail", stagehand_issue, path)

    return Finding(
        adapter,
        "pass",
        f"{len(slices)} slice(s) have parity, artifact, eval summary, and cleanup evidence",
        path,
    )


def _check_cua_cloud_failure_probe(probe: Mapping[str, Any]) -> str | None:
    if not probe:
        return "missing Cua cloud failure probe evidence"
    if probe.get("status") != "not-ready":
        return "Cua cloud failure probe must be not-ready until runtime works"
    if probe.get("failure_class") != "cloud-computer-server-cmd-404":
        return "Cua cloud failure probe has unexpected failure_class"
    failed = _list(probe.get("failed_capabilities"))
    required = {"startup", "shell", "file_transfer", "dimensions", "screenshot"}
    if not required.issubset({str(item) for item in failed}):
        return "Cua cloud failure probe missing failed runtime capabilities"
    sdk = _mapping(probe.get("sdk"))
    if not _string(sdk.get("package")) or not _string(sdk.get("version")):
        return "Cua cloud failure probe missing SDK package/version"
    request = _mapping(probe.get("request"))
    if not _string(request.get("linux_kind")):
        return "Cua cloud failure probe missing request linux_kind"
    cleanup = _mapping(probe.get("cleanup"))
    if _number(cleanup.get("cloud_vms")) != 0:
        return "Cua cloud failure probe must prove zero cloud VMs after cleanup"
    return None


def _check_browser_use_official_probe(items: list[Any]) -> str | None:
    record = _additional_evidence_item(items, "official-browser-use-encrypted-task")
    if not isinstance(record, dict):
        return "missing official Browser Use original-runner probe evidence"
    if record.get("status") not in {
        "blocked-original-runner-benchflow-passed",
        "blocked-original-runner-benchflow-completed",
    }:
        return "official Browser Use evidence must declare blocked original runner"
    probe = _mapping(record.get("original_runner_probe"))
    if probe.get("schema") != "benchflow.browser-use-original-runner-probe.v1":
        return "official Browser Use probe has unexpected schema"
    if probe.get("status") != "blocked":
        return "official Browser Use probe must be blocked until original parity works"
    allowed_failures = {
        "host-local-browser-startup-timeout",
        "host-local-browser-startup-failure",
        "original-runner-process-timeout",
        "runner-produced-no-agent-trace",
    }
    if probe.get("failure_class") not in allowed_failures:
        return "official Browser Use probe has unexpected failure_class"
    checks = _mapping(probe.get("checks"))
    if checks.get("trace_complete") is not False:
        return "official Browser Use blocked probe must show trace is incomplete"
    if _number(checks.get("expected_result_count")) != 1.0:
        return "official Browser Use probe must target one selected task"
    artifacts = _mapping(probe.get("artifacts"))
    if not _string(artifacts.get("raw_trace_policy")):
        return "official Browser Use probe must document raw trace policy"
    benchflow = _mapping(record.get("benchflow_run"))
    if _number(benchflow.get("reward")) is None:
        return "official Browser Use BenchFlow run must record reward"
    if not _number(benchflow.get("trajectory_steps_min")):
        return "official Browser Use BenchFlow run missing trajectory evidence"
    if not _number(benchflow.get("tool_calls_min")):
        return "official Browser Use BenchFlow run missing tool-call evidence"
    if not _number(benchflow.get("screenshots_min")):
        return "official Browser Use BenchFlow run missing screenshot evidence"
    parity = _mapping(record.get("parity"))
    if parity.get("comparable") is not False:
        return "official Browser Use evidence must not claim parity while blocked"
    if not (
        parity.get("benchflow_completed_same_selected_task") is True
        or parity.get("benchflow_passed_same_selected_task") is True
    ):
        return (
            "official Browser Use evidence must prove BenchFlow ran same selected task"
        )
    cleanup = _mapping(record.get("cleanup"))
    if not cleanup or not _cleanup_empty(cleanup):
        return "official Browser Use cleanup evidence is missing or non-empty"
    commands = _list(record.get("commands"))
    if len(commands) < 3:
        return "official Browser Use evidence must include import, BenchFlow, and probe commands"
    return None


def _check_cookbook_support_report(summary: Mapping[str, Any]) -> str | None:
    if not summary:
        return "use-computer cookbook evidence missing unsupported_summary"
    if not _number(summary.get("known_supported_raw_cuagym_tasks")):
        return "use-computer cookbook evidence missing supported raw CUA-Gym count"
    if not _number(summary.get("raw_cuagym_total_tasks")):
        return "use-computer cookbook evidence missing raw CUA-Gym total"
    support = _mapping(summary.get("support_report"))
    if support.get("schema") != "benchflow.cuagym-import-support-report.v1":
        return "use-computer cookbook support report has unexpected schema"
    if support.get("unsupported_records_persisted") is not True:
        return "use-computer cookbook support report must persist unsupported records"
    fields = {str(item) for item in _list(support.get("record_fields"))}
    required = {"task_id", "status", "app_type", "difficulty", "reason", "code"}
    if not required.issubset(fields):
        return "use-computer cookbook support report missing required record fields"
    if not _string(support.get("plaintext_policy")):
        return "use-computer cookbook support report missing plaintext policy"
    commands = _list(support.get("commands"))
    if not commands:
        return "use-computer cookbook support report missing reproduction command"
    blockers = _list(summary.get("top_remaining_blockers"))
    if not blockers:
        return "use-computer cookbook evidence missing top remaining blockers"
    return None


def _check_stagehand_official_parity(items: list[Any]) -> str | None:
    record = _additional_evidence_item(items, "official-stagehand-agent-sign-in")
    if not isinstance(record, dict):
        return "missing official Stagehand agent/sign_in parity evidence"
    issue = _check_stagehand_parity_record(
        record,
        task_id="agent/sign_in",
        require_exact_final_url=True,
    )
    if issue:
        return issue

    steam = _additional_evidence_item(items, "official-stagehand-agent-steam-games")
    if not isinstance(steam, dict):
        return "missing official Stagehand agent/steam_games parity evidence"
    issue = _check_stagehand_parity_record(
        steam,
        task_id="agent/steam_games",
        require_exact_final_url=False,
    )
    if issue:
        return issue

    inventory = _additional_evidence_item(items, "official-stagehand-support-inventory")
    if not isinstance(inventory, dict):
        return "missing official Stagehand unsupported inventory evidence"
    if inventory.get("status") != "unsupported-confirmed":
        return "official Stagehand inventory must be unsupported-confirmed"
    if not _number(inventory.get("task_count")):
        return "official Stagehand inventory missing task_count"
    if not _number(inventory.get("supported_count")):
        return "official Stagehand inventory missing supported_count"
    if not _number(inventory.get("unsupported_count")):
        return "official Stagehand inventory missing unsupported_count"
    supported = {str(item) for item in _list(inventory.get("supported_tasks"))}
    if not {"agent/sign_in", "agent/steam_games"}.issubset(supported):
        return "official Stagehand inventory missing verified supported tasks"
    issues = _mapping(inventory.get("unsupported_issues"))
    if not issues:
        return "official Stagehand inventory missing unsupported issue counts"
    if "stagehand-verifier-not-mapped" not in issues:
        return "official Stagehand inventory missing verifier unsupported reason"
    if "stagehand-expected-answer-verifier-not-mapped" not in issues:
        return "official Stagehand inventory missing expected-answer unsupported reason"
    return None


def _additional_evidence_item(
    items: list[Any], item_id: str
) -> Mapping[str, Any] | None:
    return next(
        (
            item
            for item in items
            if isinstance(item, Mapping) and item.get("id") == item_id
        ),
        None,
    )


def _check_stagehand_parity_record(
    record: Mapping[str, Any],
    *,
    task_id: str,
    require_exact_final_url: bool,
) -> str | None:
    label = f"official Stagehand {task_id}"
    if record.get("status") != "parity-confirmed":
        return f"{label} evidence is not parity-confirmed"
    if record.get("task_id") != task_id:
        return f"{label} evidence has wrong task_id"
    original = _mapping(record.get("original_runner"))
    benchflow = _mapping(record.get("benchflow_run"))
    parity = _mapping(record.get("parity"))
    if _number(original.get("score")) != 1.0:
        return f"{label} original runner score must be 1.0"
    if _number(benchflow.get("reward")) != 1.0:
        return f"{label} BenchFlow reward must be 1.0"
    if require_exact_final_url and original.get("final_url") != benchflow.get(
        "final_url"
    ):
        return f"{label} final URL parity is missing or divergent"
    compared = _number(parity.get("criteria_compared"))
    agreed = _number(parity.get("criteria_agreed"))
    if not compared or agreed != compared:
        return f"{label} parity criteria did not all agree"
    if _number(parity.get("reward_delta_max")) != 0.0:
        return f"{label} reward delta must be 0.0"
    if parity.get("loop_state") is not True:
        return f"{label} missing adapter-adoption loop state evidence"
    screenshots_min = _number(benchflow.get("screenshots_min"))
    if screenshots_min is None or screenshots_min <= 0:
        return f"{label} BenchFlow artifact is missing screenshot evidence"
    cleanup = _mapping(record.get("cleanup"))
    if not cleanup or not _cleanup_empty(cleanup):
        return f"{label} cleanup evidence is missing or non-empty"
    commands = _list(record.get("commands"))
    if len(commands) < 2:
        return f"{label} parity evidence must include reproduction commands"
    return None


def check_browser_use_smoke(repo_root: Path) -> Finding:
    return _check_environment_evidence(
        adapter="Browser Use / Stagehand",
        path=repo_root / "benchmarks" / "browser-use-smoke" / "adapter_evidence.json",
        expected_benchmark="browser-use-smoke",
        expected_status="parity-confirmed",
        min_slices=3,
        require_screenshot=True,
        require_browser_use_official_probe=True,
        require_stagehand_official_parity=True,
    )


def check_computer_use_smoke(repo_root: Path) -> Finding:
    return _check_environment_evidence(
        adapter="Computer Use Cua smoke",
        path=repo_root / "benchmarks" / "computer-use-smoke" / "adapter_evidence.json",
        expected_benchmark="computer-use-smoke",
        expected_status="parity-confirmed",
        min_slices=1,
        require_screenshot=True,
        require_cua_cloud_failure_probe=True,
    )


def check_use_computer_cookbook_smoke(repo_root: Path) -> Finding:
    return _check_environment_evidence(
        adapter="use-computer cookbook",
        path=repo_root
        / "benchmarks"
        / "use-computer-cookbook-smoke"
        / "adapter_evidence.json",
        expected_benchmark="use-computer-cookbook-smoke",
        expected_status="parity-confirmed",
        min_slices=2,
        require_screenshot=True,
        require_cookbook_support_report=True,
    )


def check_iosworld_smoke(repo_root: Path) -> Finding:
    adapter = "iOSWorld smoke"
    path = repo_root / "benchmarks" / "iosworld-smoke" / "adapter_evidence.json"
    if not path.exists():
        return _missing(adapter, path)
    data = _load_json(path)
    if not isinstance(data, dict):
        return Finding(adapter, "fail", "expected evidence mapping", path)
    if data.get("schema") != "benchflow.environment-adapter-evidence.v1":
        return Finding(adapter, "fail", "unexpected evidence schema", path)
    if data.get("benchmark") != "iosworld-smoke":
        return Finding(adapter, "fail", "evidence benchmark name mismatch", path)
    if data.get("status") != "unsupported-confirmed":
        return Finding(adapter, "fail", "expected unsupported-confirmed status", path)

    unsupported = _list(data.get("unsupported_reports"))
    if len(unsupported) < 2:
        return Finding(
            adapter, "fail", "missing task/environment unsupported reports", path
        )
    for item in unsupported:
        if not isinstance(item, dict):
            return Finding(adapter, "fail", "unsupported report is not a mapping", path)
        if item.get("status") != "unsupported-adapter-task":
            return Finding(adapter, "fail", "unsupported report has wrong status", path)
        if item.get("adapter") != "iosworld":
            return Finding(
                adapter, "fail", "unsupported report has wrong adapter", path
            )
        details = _mapping(item.get("details"))
        if details.get("required_provider") != "macos-ios-simulator":
            return Finding(
                adapter,
                "fail",
                "unsupported report missing macos-ios-simulator provider",
                path,
            )
        commands = _list(item.get("commands"))
        if not commands:
            return Finding(adapter, "fail", "unsupported report has no command", path)

    return Finding(
        adapter,
        "pass",
        "iOSWorld reports provider-honest macOS/iOS Simulator requirements",
        path,
    )


OPEN_PR_CHECKS = {
    "hilbench": check_hilbench,
    "opaquetoolsbench": check_opaquetoolsbench,
    "continuallearningbench": check_continuallearningbench,
}

REQUIRED_OPEN_PR_ADAPTERS = {
    "HILBench": "hilbench",
    "OpaqueToolsBench": "opaquetoolsbench",
    "ContinualLearningBench": "continuallearningbench",
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
        help="Open adapter PR worktree root. Names: HILBench, OpaqueToolsBench, ContinualLearningBench.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit zero when evidence is blocked but otherwise valid.",
    )
    parser.add_argument(
        "--only-universal-environment-adapters",
        action="store_true",
        help=(
            "Validate only the 0.7 browser/computer-use/iOSWorld environment "
            "adapter evidence manifests."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()

    universal_environment_findings = [
        check_browser_use_smoke(repo_root),
        check_computer_use_smoke(repo_root),
        check_use_computer_cookbook_smoke(repo_root),
        check_iosworld_smoke(repo_root),
    ]
    if args.only_universal_environment_adapters:
        findings = universal_environment_findings
    else:
        findings = [
            check_harvey_lab(repo_root),
            check_programbench(repo_root),
            *universal_environment_findings,
            check_skillsbench_result(args.skillsbench_result),
        ]

        supplied_roots = {
            name.lower().replace("-", "").replace("_", ""): (name, root)
            for name, root in args.open_pr_root
        }

        unknown_keys = sorted(
            set(supplied_roots) - set(REQUIRED_OPEN_PR_ADAPTERS.values())
        )
        for key in unknown_keys:
            name, root = supplied_roots[key]
            valid = ", ".join(REQUIRED_OPEN_PR_ADAPTERS)
            findings.append(
                Finding(
                    name,
                    "fail",
                    f"unknown open PR adapter name; valid: {valid}",
                    root,
                )
            )

        for display_name, key in REQUIRED_OPEN_PR_ADAPTERS.items():
            supplied = supplied_roots.get(key)
            if supplied is None:
                findings.append(
                    Finding(
                        display_name,
                        "fail",
                        "open adapter PR evidence root is required via "
                        f"--open-pr-root {display_name}=/path/to/worktree",
                        Path("<missing>"),
                    )
                )
                continue
            name, root = supplied
            check = OPEN_PR_CHECKS[key]
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
