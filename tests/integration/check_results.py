"""Review integration test results.

Reads jobs/<agent>/*/result.json and summary.json files produced by
``bench eval create`` and validates:

1. Every expected agent produced a jobs directory
2. Each trial has a valid result.json (schema check)
3. summary.json exists and has required fields
4. No infrastructure errors (agent_install, timeout, pipe)
5. Score table printed for human review

Usage::

    python tests/integration/check_results.py jobs/integration gemini pi-acp
    python tests/integration/check_results.py jobs/integration   # all subdirs

Guards: ENG-6 integration test plan (PR #255).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from benchflow._utils.config import normalize_agent_idle_timeout
from benchflow._utils.reward_events import memory_score_from_result
from benchflow._utils.scoring import (
    classify_error,
    classify_verifier_error,
    count_result_outcomes,
)
from benchflow._utils.source_provenance import source_issues, source_matches_parent

EXPECTED: dict[str, Any] = {}
EXPECTED_FIELDS = {
    "agent",
    "model",
    "environment",
    "concurrency",
    "agent_idle_timeout",
    "agent_idle_timeout_sec",
}
REMOTE_REACHABILITY: dict[tuple[str, str, str | None], bool] = {}
RESULT_REQUIRED = {"task_name", "agent", "rewards", "error", "verifier_error"}
SUMMARY_REQUIRED = {
    "total",
    "passed",
    "failed",
    "errored",
    "verifier_errored",
    "score",
}
INFRA_ERROR_CATEGORIES = {
    "install_failure",
    "timeout",
    "idle_timeout",
    "pipe_closed",
    "sandbox_setup",
    "infra_failure",
}


def load_results(agent_dir: Path) -> tuple[list[tuple[Path, dict]], list[str]]:
    """Load every result.json from an agent's jobs directory."""
    results: list[tuple[Path, dict]] = []
    issues: list[str] = []
    for rfile in sorted(agent_dir.rglob("result.json")):
        try:
            result = json.loads(rfile.read_text())
        except (json.JSONDecodeError, OSError) as e:
            issues.append(f"bad result file {rfile}: {e}")
            continue
        results.append((rfile, result))
    return results, issues


_MISSING = object()


def _parse_expected_value(value: str) -> Any:
    if value == "null":
        return None
    return value


def _expected(agent: str, field: str) -> Any:
    agent_value = EXPECTED.get(f"{agent}.{field}", _MISSING)
    if agent_value is not _MISSING:
        return agent_value
    return EXPECTED.get(field, _MISSING)


def _expected_agent_idle_timeout(agent: str) -> Any:
    expected = _expected(agent, "agent_idle_timeout_sec")
    if expected is not _MISSING:
        return expected
    return _expected(agent, "agent_idle_timeout")


def _artifact_agent_idle_timeout(payload: dict[str, Any]) -> Any:
    if "agent_idle_timeout_sec" in payload:
        return payload["agent_idle_timeout_sec"]
    if "agent_idle_timeout" in payload:
        return payload["agent_idle_timeout"]
    return _MISSING


def _parse_agent_idle_timeout_value(
    value: Any, label: str
) -> tuple[int | None, str | None]:
    if value is _MISSING:
        return None, f"{label} missing agent_idle_timeout_sec"
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{label} agent_idle_timeout_sec must be null or integer seconds"
    if isinstance(value, int):
        try:
            return normalize_agent_idle_timeout(value), None
        except ValueError as e:
            return None, f"{label} agent_idle_timeout_sec invalid: {e}"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None, None
        if not text.isdecimal():
            return (
                None,
                f"{label} agent_idle_timeout_sec must be null or integer seconds",
            )
        try:
            return normalize_agent_idle_timeout(text), None
        except ValueError as e:
            return None, f"{label} agent_idle_timeout_sec invalid: {e}"
    return None, f"{label} agent_idle_timeout_sec must be null or integer seconds"


def _compare_agent_idle_timeout(actual: Any, expected: Any, label: str) -> str | None:
    actual_value, actual_error = _parse_agent_idle_timeout_value(actual, label)
    if actual_error:
        return actual_error
    expected_value, expected_error = _parse_agent_idle_timeout_value(
        expected, "expected"
    )
    if expected_error:
        return expected_error
    if actual_value != expected_value:
        return (
            f"{label} agent_idle_timeout_sec {_expected_label(actual)} "
            f"does not match expected {_expected_label(expected)}"
        )
    return None


def _latest_result_entries_by_task(
    result_entries: list[tuple[Path, dict]],
) -> list[tuple[Path, dict]]:
    """Return the latest result entry per task for summary/artifact reconciliation."""
    latest_by_task: dict[str, tuple[float, Path, dict]] = {}
    for rfile, result in result_entries:
        task_name = result.get("task_name") or rfile.parent.name.rsplit("__", 1)[0]
        mtime = rfile.stat().st_mtime
        previous = latest_by_task.get(task_name)
        if previous is None or (mtime, str(rfile)) >= (previous[0], str(previous[1])):
            latest_by_task[task_name] = (mtime, rfile, result)
    return [(rfile, result) for _, rfile, result in latest_by_task.values()]


def _has_expected(agent: str, field: str) -> bool:
    return _expected(agent, field) is not _MISSING


def _expected_label(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    return "null" if value is None else repr(value)


def _latest_results_by_task(result_entries: list[tuple[Path, dict]]) -> list[dict]:
    """Return the latest result per task for summary/outcome reconciliation."""
    return [result for _, result in _latest_result_entries_by_task(result_entries)]


def _find_result_root(agent_dir: Path) -> Path:
    """Return the directory whose tree should be audited for result.json files."""
    if _is_rollout_artifact_root(agent_dir):
        return agent_dir
    run_dirs = sorted(
        (d for d in agent_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name,
    )
    if not run_dirs:
        return agent_dir
    return run_dirs[-1]


def _load_config(result_file: Path) -> tuple[Path, dict | None, str | None]:
    config_path = result_file.parent / "config.json"
    if not config_path.exists():
        return config_path, None, "missing config.json"
    try:
        return config_path, json.loads(config_path.read_text()), None
    except json.JSONDecodeError:
        return config_path, None, "config.json: invalid JSON"


def _source_hash_truth_issues(source: object, label: str) -> list[str]:
    if not isinstance(source, dict):
        return []
    source_dict = cast(dict[str, Any], source)
    local_path_raw = source_dict.get("local_path")
    file_hashes = source_dict.get("file_hashes")
    if not isinstance(local_path_raw, str):
        return [f"{label}: source.local_path must be a string for hash audit"]
    if not isinstance(file_hashes, dict):
        return []
    local_path = Path(local_path_raw)
    if not local_path.exists():
        return [f"{label}: source.local_path does not exist for hash audit"]
    try:
        from benchflow._utils.benchmark_repos import task_file_hashes

        actual = task_file_hashes(local_path)
    except Exception as e:
        return [f"{label}: source.file_hashes could not be recomputed: {e}"]
    if actual != file_hashes:
        return [f"{label}: source.file_hashes do not match local_path"]
    return _source_git_truth_issues(source_dict, local_path, label)


def _git_stdout(cwd: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _repo_slug_from_remote(remote: str) -> str | None:
    normalized = remote.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        return normalized.removeprefix("git@github.com:")
    marker = "github.com/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return None


def _git_ls_remote_contains(repo: str, sha: str, requested_ref: str | None) -> bool:
    key = (repo, sha, requested_ref)
    if key in REMOTE_REACHABILITY:
        return REMOTE_REACHABILITY[key]
    url = f"https://github.com/{repo}.git"
    ref_args: list[str] = []
    if requested_ref:
        ref_args = [
            requested_ref,
            f"refs/heads/{requested_ref}",
            f"refs/tags/{requested_ref}",
        ]
    completed = subprocess.run(
        ["git", "ls-remote", url, *ref_args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0 and ref_args:
        completed = subprocess.run(
            ["git", "ls-remote", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
    reachable = completed.returncode == 0 and any(
        line.split(maxsplit=1)[0] == sha
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    REMOTE_REACHABILITY[key] = reachable
    return reachable


def _verify_remote_reachability(
    repo: str,
    sha: str,
    requested_ref: str | None,
    label: str,
) -> tuple[bool | None, list[str]]:
    """Return (reachable, issues). ``None`` means verification itself failed."""
    try:
        return _git_ls_remote_contains(repo, sha, requested_ref), []
    except (OSError, subprocess.SubprocessError) as e:
        return None, [
            f"{label}: source.resolved_sha remote reachability could not be verified: {e}"
        ]


def _source_git_truth_issues(
    source: dict[str, Any], local_path: Path, label: str
) -> list[str]:
    issues: list[str] = []
    git_root_raw = _git_stdout(local_path, "rev-parse", "--show-toplevel")
    if not git_root_raw:
        return [f"{label}: source.local_path is not inside a git worktree"]
    git_root = Path(git_root_raw).resolve(strict=True)
    local_resolved = local_path.resolve(strict=True)
    if local_resolved != git_root and not local_resolved.is_relative_to(git_root):
        issues.append(f"{label}: source.local_path is outside its git worktree")
    repo = source.get("repo")
    remote_reachable: bool | None = None

    head = _git_stdout(local_path, "rev-parse", "HEAD")
    if head != source.get("resolved_sha"):
        issues.append(
            f"{label}: source.resolved_sha does not match source.local_path git HEAD"
        )
    elif isinstance(repo, str) and isinstance(head, str):
        requested_ref_raw = source.get("requested_ref")
        requested_ref = (
            requested_ref_raw if isinstance(requested_ref_raw, str) else None
        )
        if remote_reachable is None:
            remote_reachable, reachability_issues = _verify_remote_reachability(
                repo, head, requested_ref, label
            )
            issues.extend(reachability_issues)
        if remote_reachable is False:
            issues.append(
                f"{label}: source.resolved_sha is not reachable from source.repo/requested_ref"
            )

    status_args = ["status", "--porcelain"]
    try:
        rel_for_status = local_resolved.relative_to(git_root).as_posix()
        status_args.extend(["--", rel_for_status])
    except ValueError:
        pass
    status = _git_stdout(git_root, *status_args)
    if status is None:
        issues.append(f"{label}: source.local_path git status could not be read")
    elif bool(status) != bool(source.get("dirty")):
        issues.append(
            f"{label}: source.dirty does not match source.local_path git status"
        )
    elif status and source.get("dirty") is False:
        issues.append(f"{label}: source.local_path git worktree is dirty")

    remote = _git_stdout(local_path, "config", "--get", "remote.origin.url")
    if not remote:
        issues.append(f"{label}: source.local_path git remote.origin.url is missing")
    elif _repo_slug_from_remote(remote) != source.get("repo"):
        issues.append(
            f"{label}: source.repo does not match source.local_path git remote"
        )

    expected_path = str(source.get("path") or "").strip("/")
    if expected_path:
        actual_path = local_resolved.relative_to(git_root).as_posix()
        if actual_path != expected_path:
            issues.append(
                f"{label}: source.path does not match source.local_path repo-relative path"
            )
    return issues


def check_agent(agent_dir: Path) -> dict:
    """Validate one agent's output. Returns a summary dict."""
    agent = agent_dir.name
    findings: dict = {"agent": agent, "ok": True, "issues": []}
    is_artifact_root = _is_rollout_artifact_root(agent_dir)
    expected_agent = _expected(agent, "agent")
    if expected_agent is _MISSING and not is_artifact_root:
        expected_agent = agent

    result_root = _find_result_root(agent_dir)
    if result_root == agent_dir and not is_artifact_root:
        findings["ok"] = False
        findings["issues"].append("no run directory found")
        return findings

    result_entries, result_load_issues = load_results(result_root)
    if result_load_issues:
        findings["ok"] = False
        findings["issues"].extend(result_load_issues)
    results = [result for _, result in result_entries]
    latest_results = _latest_results_by_task(result_entries)
    latest_result_entries = _latest_result_entries_by_task(result_entries)

    if not results:
        findings["ok"] = False
        findings["issues"].append("no result.json files")
        return findings

    # Schema check
    for result_file, r in result_entries:
        missing = RESULT_REQUIRED - set(r.keys())
        if missing:
            findings["issues"].append(f"{r.get('task_name', '?')}: missing {missing}")
            findings["ok"] = False
        found_source_issues = source_issues(
            r.get("source"),
            f"{r.get('task_name', '?')}: result.json",
            require_file_hashes=True,
        )
        if found_source_issues:
            findings["issues"].extend(found_source_issues)
            findings["ok"] = False
        source_truth_issues = _source_hash_truth_issues(
            r.get("source"), f"{r.get('task_name', '?')}: result.json"
        )
        if source_truth_issues:
            findings["issues"].extend(source_truth_issues)
            findings["ok"] = False
        _config_path, config, config_error = _load_config(result_file)
        if config_error:
            findings["issues"].append(f"{r.get('task_name', '?')}: {config_error}")
            findings["ok"] = False
        else:
            assert config is not None
            expected_model = _expected(agent, "model")
            expected_environment = _expected(agent, "environment")
            expected_concurrency = _expected(agent, "concurrency")
            expected_agent_idle_timeout = _expected_agent_idle_timeout(agent)
            if r.get("agent") != config.get("agent"):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: result.json agent does not match config.json"
                )
                findings["ok"] = False
            if "model" in r and r.get("model") != config.get("model"):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: result.json model does not match config.json"
                )
                findings["ok"] = False
            if expected_agent is not _MISSING and expected_agent != config.get("agent"):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: config.json agent {config.get('agent')!r} does not match expected {_expected_label(expected_agent)}"
                )
                findings["ok"] = False
            if expected_model is not _MISSING and r.get("model") != expected_model:
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: result.json model {r.get('model')!r} does not match expected {_expected_label(expected_model)}"
                )
                findings["ok"] = False
            if expected_model is not _MISSING and config.get("model") != expected_model:
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: config.json model {config.get('model')!r} does not match expected {_expected_label(expected_model)}"
                )
                findings["ok"] = False
            if (
                expected_environment is not _MISSING
                and config.get("environment") != expected_environment
            ):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: config.json environment {config.get('environment')!r} does not match expected {_expected_label(expected_environment)}"
                )
                findings["ok"] = False
            if expected_concurrency is not _MISSING and str(
                config.get("concurrency")
            ) != str(expected_concurrency):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: config.json concurrency {config.get('concurrency')!r} does not match expected {_expected_label(expected_concurrency)}"
                )
                findings["ok"] = False
            config_agent_idle_timeout = _artifact_agent_idle_timeout(config)
            if expected_agent_idle_timeout is not _MISSING:
                idle_timeout_issue = _compare_agent_idle_timeout(
                    config_agent_idle_timeout,
                    expected_agent_idle_timeout,
                    f"{r.get('task_name', '?')}: config.json",
                )
            elif config_agent_idle_timeout is not _MISSING:
                idle_timeout_issue = _parse_agent_idle_timeout_value(
                    config_agent_idle_timeout,
                    f"{r.get('task_name', '?')}: config.json",
                )[1]
            else:
                idle_timeout_issue = None
            if idle_timeout_issue:
                findings["issues"].append(idle_timeout_issue)
                findings["ok"] = False
            config_source_issues = source_issues(
                config.get("source"),
                f"{r.get('task_name', '?')}: config.json",
                require_file_hashes=True,
            )
            if config_source_issues:
                findings["issues"].extend(config_source_issues)
                findings["ok"] = False
            config_truth_issues = _source_hash_truth_issues(
                config.get("source"), f"{r.get('task_name', '?')}: config.json"
            )
            if config_truth_issues:
                findings["issues"].extend(config_truth_issues)
                findings["ok"] = False
            if config.get("source") != r.get("source"):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: config.json source does not match result.json"
                )
                findings["ok"] = False

    # Infrastructure errors
    infra_errors = []
    idle_timeout_tasks: list[str] = []
    sandbox_startup_tasks: list[str] = []
    transport_error_tasks: list[str] = []
    for r in results:
        err = r.get("error")
        cat = classify_error(str(err)) if err else None
        if cat and cat in INFRA_ERROR_CATEGORIES:
            task = r.get("task_name", "?")
            if cat == "idle_timeout":
                info = r.get("idle_timeout_info")
                if info:
                    infra_errors.append(
                        f"{task}: idle timeout after "
                        f"{info.get('idle_duration_sec', '?')}s idle "
                        f"({info.get('n_tool_calls', '?')} tool calls, "
                        f"{info.get('wall_clock_elapsed_sec', '?')}s wall)"
                    )
                else:
                    infra_errors.append(f"{task}: {err}")
                idle_timeout_tasks.append(task)
            elif cat == "sandbox_setup":
                sinfo = r.get("sandbox_startup_info")
                if sinfo:
                    sid = sinfo.get("sandbox_id", "?")
                    state = sinfo.get("sandbox_state", "?")
                    attempts = sinfo.get("attempts", "?")
                    build_to = sinfo.get("build_timeout_sec", "?")
                    infra_errors.append(
                        f"{task}: sandbox startup failed (sandbox_id={sid}, "
                        f"state={state}, attempts={attempts}, "
                        f"build_timeout_sec={build_to})"
                    )
                else:
                    infra_errors.append(f"{task}: {err}")
                sandbox_startup_tasks.append(task)
            elif cat == "pipe_closed":
                tinfo = r.get("transport_error_info")
                if tinfo:
                    rc = tinfo.get("process_exit_code", "?")
                    diag = tinfo.get("transport_diagnosis", "?")
                    reachable = tinfo.get("sandbox_reachable", "?")
                    infra_errors.append(
                        f"{task}: transport closed (rc={rc}, "
                        f"diagnosis={diag}, sandbox_reachable={reachable})"
                    )
                else:
                    infra_errors.append(f"{task}: {err}")
                transport_error_tasks.append(task)
            else:
                infra_errors.append(f"{task}: {err}")
    if infra_errors:
        findings["issues"].extend(infra_errors)
        findings["ok"] = False
    if idle_timeout_tasks:
        findings["issues"].append(
            f"INVALIDATED: {len(idle_timeout_tasks)} task(s) hit idle timeout "
            f"and should be rerun: {', '.join(idle_timeout_tasks)}"
        )
    if sandbox_startup_tasks:
        findings["issues"].append(
            f"INVALIDATED: {len(sandbox_startup_tasks)} task(s) failed during "
            f"sandbox startup and should be rerun: "
            f"{', '.join(sandbox_startup_tasks)}"
        )
    if transport_error_tasks:
        findings["issues"].append(
            f"INVALIDATED: {len(transport_error_tasks)} task(s) lost ACP transport "
            f"(pipe closed / rc=255) and should be rerun: "
            f"{', '.join(transport_error_tasks)}"
        )

    # Verifier dependency install failures (ENG-151)
    dep_install_tasks: list[str] = []
    for r in results:
        verifier_err = r.get("verifier_error")
        vcat = classify_verifier_error(verifier_err) if verifier_err else None
        if vcat == "verifier_dep_install":
            task = r.get("task_name", "?")
            dep_install_tasks.append(task)
            findings["issues"].append(
                f"{task}: verifier dependency install failed — "
                f"measurement invalid (verifier never reached tests)"
            )
            findings["ok"] = False
    if dep_install_tasks:
        findings["issues"].append(
            f"INVALIDATED: {len(dep_install_tasks)} task(s) failed during "
            f"verifier dependency install and should be rerun after "
            f"fixing the index policy: {', '.join(dep_install_tasks)}"
        )

    # Verifier timeout failures (ENG-152)
    verifier_timeout_tasks: list[str] = []
    for r in results:
        verifier_err = r.get("verifier_error")
        vcat = classify_verifier_error(verifier_err) if verifier_err else None
        if vcat == "verifier_timeout":
            task = r.get("task_name", "?")
            vti = r.get("verifier_timeout_info")
            budget = vti.get("timeout_budget_sec", "?") if vti else "?"
            elapsed = vti.get("elapsed_sec", "?") if vti else "?"
            verifier_timeout_tasks.append(task)
            findings["issues"].append(
                f"{task}: verifier timed out (budget={budget}s, elapsed={elapsed}s) — "
                f"measurement invalid (verifier never produced reward)"
            )
            findings["ok"] = False
    if verifier_timeout_tasks:
        findings["issues"].append(
            f"INVALIDATED: {len(verifier_timeout_tasks)} task(s) had verifier "
            f"timeouts — increase timeout_sec or reduce verifier cost: "
            f"{', '.join(verifier_timeout_tasks)}"
        )

    # Summary.json — bench eval create writes it at the agent_dir root
    summary_path = agent_dir / "summary.json"
    if not summary_path.exists():
        summary_path = result_root / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            missing = SUMMARY_REQUIRED - set(summary.keys())
            if missing:
                findings["issues"].append(f"summary.json missing: {missing}")
                findings["ok"] = False
            if summary.get("source_mismatch_tasks"):
                findings["issues"].append(
                    f"summary.json source mismatch tasks: {summary['source_mismatch_tasks']}"
                )
                findings["ok"] = False
            if summary.get("source") is not None:
                found_source_issues = source_issues(
                    summary.get("source"),
                    "summary.json",
                    require_file_hashes=False,
                )
                if found_source_issues:
                    findings["issues"].extend(found_source_issues)
                    findings["ok"] = False
            elif any(not isinstance(r.get("source"), dict) for r in results):
                findings["issues"].append(
                    "summary.json missing source provenance and not all results have source"
                )
                findings["ok"] = False
            expected_model = _expected(agent, "model")
            expected_environment = _expected(agent, "environment")
            expected_concurrency = _expected(agent, "concurrency")
            expected_agent_idle_timeout = _expected_agent_idle_timeout(agent)
            summary_agent = summary.get("agent")
            if (
                expected_agent is not _MISSING
                and summary_agent is not None
                and summary_agent != expected_agent
            ):
                findings["issues"].append(
                    f"summary.json agent {summary_agent!r} does not match expected {_expected_label(expected_agent)}"
                )
                findings["ok"] = False
            if summary_agent is not None:
                for result in results:
                    if result.get("agent") != summary_agent:
                        findings["issues"].append(
                            f"summary.json agent {summary_agent!r} does not match result.json agent {result.get('agent')!r}"
                        )
                        findings["ok"] = False
            if (
                expected_model is not _MISSING
                and summary.get("model") != expected_model
            ):
                findings["issues"].append(
                    f"summary.json model {summary.get('model')!r} does not match expected {_expected_label(expected_model)}"
                )
                findings["ok"] = False
            if (
                expected_environment is not _MISSING
                and summary.get("environment") != expected_environment
            ):
                findings["issues"].append(
                    f"summary.json environment {summary.get('environment')!r} does not match expected {_expected_label(expected_environment)}"
                )
                findings["ok"] = False
            if expected_concurrency is not _MISSING and str(
                summary.get("concurrency")
            ) != str(expected_concurrency):
                findings["issues"].append(
                    f"summary.json concurrency {summary.get('concurrency')!r} does not match expected {_expected_label(expected_concurrency)}"
                )
                findings["ok"] = False
            summary_agent_idle_timeout = _artifact_agent_idle_timeout(summary)
            if expected_agent_idle_timeout is not _MISSING:
                idle_timeout_issue = _compare_agent_idle_timeout(
                    summary_agent_idle_timeout,
                    expected_agent_idle_timeout,
                    "summary.json",
                )
            elif summary_agent_idle_timeout is not _MISSING:
                idle_timeout_issue = _parse_agent_idle_timeout_value(
                    summary_agent_idle_timeout,
                    "summary.json",
                )[1]
            else:
                idle_timeout_issue = None
            if idle_timeout_issue:
                findings["issues"].append(idle_timeout_issue)
                findings["ok"] = False
            summary_source = summary.get("source")
            for result_file, r in latest_result_entries:
                _path, config, config_error = _load_config(result_file)
                if not config_error and config is not None:
                    config_agent_idle_timeout = _artifact_agent_idle_timeout(config)
                    if (
                        config_agent_idle_timeout is not _MISSING
                        and summary_agent_idle_timeout is not _MISSING
                    ):
                        consistency_issue = _compare_agent_idle_timeout(
                            config_agent_idle_timeout,
                            summary_agent_idle_timeout,
                            f"{r.get('task_name', '?')}: config.json",
                        )
                        if consistency_issue:
                            findings["issues"].append(
                                f"{consistency_issue} from summary.json"
                            )
                            findings["ok"] = False
            if isinstance(summary_source, dict):
                for r in results:
                    if not source_matches_parent(r.get("source"), summary_source):
                        findings["issues"].append(
                            f"summary source does not cover {r.get('task_name', '?')}"
                        )
                        findings["ok"] = False
            findings["summary"] = summary
        except json.JSONDecodeError:
            findings["issues"].append("summary.json: invalid JSON")
            findings["ok"] = False
    else:
        findings["issues"].append("summary.json not found")
        findings["ok"] = False

    # Stats
    findings["total"] = len(latest_results)
    outcome_counts = count_result_outcomes(latest_results)
    findings["passed"] = outcome_counts["passed"]
    findings["failed"] = outcome_counts["failed"]
    findings["errored"] = outcome_counts["errored"]
    findings["verifier_errored"] = outcome_counts["verifier_errored"]

    if summary := findings.get("summary"):
        for key in ("total", "passed", "failed", "errored", "verifier_errored"):
            if key in summary and summary[key] != findings[key]:
                findings["issues"].append(
                    f"summary.json {key}={summary[key]} but results imply {findings[key]}"
                )
                findings["ok"] = False
        memory_scores = [
            score
            for result in latest_results
            if (score := memory_score_from_result(result)) is not None
        ]
        if memory_scores and isinstance(summary.get("memory_score"), int | float):
            implied = sum(memory_scores) / len(memory_scores)
            if abs(float(summary["memory_score"]) - implied) > 1e-9:
                findings["issues"].append(
                    f"summary.json memory_score={summary['memory_score']} "
                    f"but results imply {implied}"
                )
                findings["ok"] = False

    return findings


def _is_rollout_artifact_root(path: Path) -> bool:
    """Return True when *path* is one completed bench eval artifact root."""
    if not (path / "summary.json").is_file():
        return False
    return any(path.rglob("result.json"))


def discover_agent_dirs(jobs_root: Path, agents: list[str] | None) -> list[Path]:
    """Discover result roots accepted by the audit CLI."""
    if agents:
        agent_dirs = [jobs_root / a for a in agents if (jobs_root / a).is_dir()]
        missing = sorted(a for a in agents if not (jobs_root / a).is_dir())
        if missing:
            print(f"ERROR: missing requested agent directories: {', '.join(missing)}")
            return []
        return agent_dirs
    if _is_rollout_artifact_root(jobs_root):
        return [jobs_root]
    return sorted(
        d for d in jobs_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def _expected_key_issues(agents: list[str] | None) -> list[str]:
    issues: list[str] = []
    agent_set = set(agents) if agents is not None else None
    for key in sorted(EXPECTED):
        parts = key.split(".", 1)
        if len(parts) == 1:
            agent_name = None
            field = parts[0]
        else:
            agent_name, field = parts
        if field not in EXPECTED_FIELDS:
            issues.append(
                f"unknown expected field {field!r}; use one of {sorted(EXPECTED_FIELDS)}"
            )
        if agent_name and agent_set is not None and agent_name not in agent_set:
            issues.append(
                f"expected key {key!r} names agent {agent_name!r}, which was not requested"
            )
    return issues


def _identity_expectation_issues(
    jobs_root: Path, agent_dirs: list[Path], agents: list[str] | None
) -> list[str]:
    issues: list[str] = []
    if _is_rollout_artifact_root(jobs_root):
        required = ("agent", "model", "environment", "concurrency")
        missing = [
            field for field in required if not _has_expected(jobs_root.name, field)
        ]
        if missing:
            issues.append(
                "direct artifact-root audits require expected identity args: "
                + ", ".join(missing)
            )
        return issues

    required = ("model", "environment", "concurrency")
    for agent_dir in agent_dirs:
        missing = [
            field for field in required if not _has_expected(agent_dir.name, field)
        ]
        if missing:
            label = (
                "requested agent"
                if agents is not None and agent_dir.name in agents
                else "discovered agent"
            )
            issues.append(
                f"{label} {agent_dir.name!r} audit requires expected identity args: "
                + ", ".join(missing)
            )
    return issues


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: check_results.py <jobs_root> [agent ...]")
        sys.exit(1)

    jobs_root = Path(sys.argv[1])
    raw_args = sys.argv[2:] if len(sys.argv) > 2 else []
    agents = []
    for arg in raw_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            EXPECTED[key] = _parse_expected_value(value)
        else:
            agents.append(arg)
    agents_arg = agents if agents else None
    expected_key_issues = _expected_key_issues(agents_arg)
    if expected_key_issues:
        for issue in expected_key_issues:
            print(f"ERROR: {issue}")
        sys.exit(1)

    if not jobs_root.exists():
        print(f"ERROR: {jobs_root} does not exist")
        sys.exit(1)

    agent_dirs = discover_agent_dirs(jobs_root, agents_arg)

    if not agent_dirs:
        print("ERROR: no agent directories found")
        sys.exit(1)

    identity_issues = _identity_expectation_issues(jobs_root, agent_dirs, agents_arg)
    if identity_issues:
        for issue in identity_issues:
            print(f"ERROR: {issue}")
        sys.exit(1)

    all_findings = []
    for d in agent_dirs:
        findings = check_agent(d)
        all_findings.append(findings)

    # Print table
    print()
    print(
        f"{'Agent':<25} {'Score':>8} {'Pass':>5} {'Fail':>5} {'Err':>5} {'Status':>8}"
    )
    print("-" * 62)
    any_fail = False
    for f in all_findings:
        status = "OK" if f["ok"] else "FAIL"
        if not f["ok"]:
            any_fail = True
        total = f.get("total", 0)
        passed = f.get("passed", 0)
        score = f"{passed / total:.1%}" if total else "N/A"
        print(
            f"{f['agent']:<25} {score:>8} "
            f"{passed:>5} {f.get('failed', 0):>5} "
            f"{f.get('errored', 0):>5} {status:>8}"
        )
        for issue in f.get("issues", []):
            print(f"  ⚠ {issue}")

    print("-" * 62)

    if any_fail:
        print("\nSome agents had issues. See warnings above.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
