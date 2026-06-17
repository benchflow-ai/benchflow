"""Review integration test results.

Reads jobs/<agent>/*/result.json and summary.json files produced by
``bench eval run`` and validates:

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
import re
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
from benchflow._utils.source_provenance import (
    is_sha256_digest,
    source_issues,
    source_matches_parent,
)
from benchflow.trajectories.metrics import (
    count_skill_invocations,
    result_skill_invocations,
)

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


def _nonnegative_int_issue(value: Any, label: str) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return f"{label} must be a non-negative integer"
    return None


def _load_acp_trajectory(path: Path) -> list[dict[str, Any]] | None:
    trajectory_path = path.parent / "trajectory" / "acp_trajectory.jsonl"
    if not trajectory_path.exists():
        return None
    trajectory: list[dict[str, Any]] = []
    try:
        for line in trajectory_path.read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                if isinstance(event, dict):
                    trajectory.append(event)
    except (json.JSONDecodeError, OSError):
        return None
    return trajectory


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


def _summary_path(agent_dir: Path, result_root: Path) -> Path:
    summary_path = agent_dir / "summary.json"
    if summary_path.exists():
        return summary_path
    return result_root / "summary.json"


def _load_summary(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, "summary.json not found"
    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None, "summary.json: invalid JSON"
    if not isinstance(summary, dict):
        return None, "summary.json must be a JSON object"
    return summary, None


def _is_worker_sharded_summary(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    worker_count = summary.get("worker_count")
    worker_concurrency = summary.get("worker_concurrency")
    return (
        isinstance(worker_count, int)
        and not isinstance(worker_count, bool)
        and worker_count > 0
        and isinstance(worker_concurrency, int)
        and not isinstance(worker_concurrency, bool)
        and worker_concurrency > 0
    )


def _expected_config_concurrency(
    agent: str,
    summary: dict[str, Any] | None,
) -> Any:
    expected_concurrency = _expected(agent, "concurrency")
    if expected_concurrency is _MISSING:
        return _MISSING
    if _is_worker_sharded_summary(summary):
        assert summary is not None
        return summary["worker_concurrency"]
    return expected_concurrency


# Skill modes that make skills available to the agent (canonical ``skill_mode``
# field). Only an explicit no-skill sentinel means no skills; any other mode
# (with-skill, self-gen, or a future/unknown mode) is treated as provisioned so
# the no-skill invariant never false-flags a legitimate with-skill trial.
_NO_SKILL_MODES = frozenset({"no-skill", "no-skills", "none", "without-skill"})
_SKILL_PATH_MARKERS = (
    ".agents/skills",
    ".codex/skills",
    ".claude/skills",
    "/skills/",
    "skills/",
)


def _config_provisions_skills(config: dict[str, Any] | None) -> bool:
    """Return whether a rollout config made any skills available to the agent.

    A "no-skill" trial has no single-agent ``skills_dir``, no task-bundled
    skills, no scene/role ``skills_dir``, and no skill-providing ``skill_mode``.
    Any of those present means skills were provisioned. Used to enforce the
    experiment-health invariant that no-skill trials must not record skill
    invocations.

    Handles both the current config schema (``skills_dir`` /
    ``include_task_skills``) and the canonical ``skill_mode`` field
    (``no-skill`` / ``with-skill`` / ``self-gen``) so the invariant stays
    correct as skill provisioning is recorded by mode.
    """
    if not isinstance(config, dict):
        return False
    if config.get("skills_dir"):
        return True
    if config.get("include_task_skills"):
        return True
    skill_mode = config.get("skill_mode")
    if isinstance(skill_mode, str):
        normalized = skill_mode.strip().lower().replace("_", "-")
        if normalized and normalized not in _NO_SKILL_MODES:
            return True
    scenes = config.get("scenes")
    if isinstance(scenes, list):
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            if scene.get("skills_dir"):
                return True
            roles = scene.get("roles")
            if isinstance(roles, list):
                for role in roles:
                    if isinstance(role, dict) and role.get("skills_dir"):
                        return True
    return False


def _task_skill_names(config: dict[str, Any], result: dict[str, Any]) -> set[str]:
    """Return bundled task skill names from the recorded source checkout."""
    for source in (config.get("source"), result.get("source")):
        if not isinstance(source, dict):
            continue
        local_path = source.get("local_path")
        if not isinstance(local_path, str) or not local_path:
            continue
        skills_root = Path(local_path) / "environment" / "skills"
        if not skills_root.is_dir():
            continue
        return {
            child.name
            for child in skills_root.iterdir()
            if child.is_dir() and (child / "SKILL.md").is_file()
        }
    return set()


def _skill_name_variants(name: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return {
        name.strip().lower(),
        normalized,
        normalized.replace("-", "_"),
        normalized.replace("-", " "),
    }


def _iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_iter_strings(item))
        return strings
    if isinstance(value, list | tuple):
        strings = []
        for item in value:
            strings.extend(_iter_strings(item))
        return strings
    return []


def _no_skill_task_file_access_evidence(
    trajectory: list[dict[str, Any]] | None,
    *,
    config: dict[str, Any],
    result: dict[str, Any],
) -> list[str]:
    if not trajectory:
        return []
    variants = {
        variant
        for name in _task_skill_names(config, result)
        for variant in _skill_name_variants(name)
        if variant
    }
    if not variants:
        return []

    evidence: list[str] = []
    for event in trajectory:
        for text in _iter_strings(event):
            lowered = text.lower()
            if not any(marker in lowered for marker in _SKILL_PATH_MARKERS):
                continue
            if any(variant in lowered for variant in variants):
                evidence.append(text)
                break
    return evidence


def _source_hash_truth_issues(source: object, label: str) -> list[str]:
    if not isinstance(source, dict):
        return []
    source_dict = cast(dict[str, Any], source)
    local_path_raw = source_dict.get("local_path")
    file_hashes = source_dict.get("file_hashes")
    if not isinstance(file_hashes, dict):
        return []
    if local_path_raw is None:
        return []
    if not isinstance(local_path_raw, str):
        return [f"{label}: source.local_path must be a string for hash audit"]
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


# Dataset identity stamp (dataset-versioning plan: benchflow #690 + dev-run
# digest stamping follow-up #691). dataset_name/dataset_version mark registry
# dataset runs; task_digest pins the exact task content for every run.
_DATASET_STAMP_KEYS = ("dataset_name", "dataset_version", "task_digest")


def _dataset_stamp_issues(artifact: object, label: str) -> list[str]:
    """Audit the dataset identity stamp on one result.json/config.json.

    Rules: dataset_name and dataset_version travel together; a dataset run
    must carry a task_digest; any task_digest must be ``sha256:<64 hex>``.
    A bare task_digest without dataset fields is a dev run — allowed.
    """
    if not isinstance(artifact, dict):
        return []
    artifact_dict = cast(dict[str, Any], artifact)
    name = artifact_dict.get("dataset_name")
    version = artifact_dict.get("dataset_version")
    digest = artifact_dict.get("task_digest")
    if name is None and version is None and digest is None:
        return []
    issues: list[str] = []
    if (name is None) != (version is None):
        issues.append(f"{label}: dataset_name and dataset_version must travel together")
    for field_name, value in (("dataset_name", name), ("dataset_version", version)):
        if value is not None and (not isinstance(value, str) or not value):
            issues.append(f"{label}: {field_name} must be a non-empty string")
    if name is not None and digest is None:
        issues.append(f"{label}: dataset-stamped artifact missing task_digest")
    if digest is not None and not is_sha256_digest(digest):
        issues.append(f"{label}: task_digest must be 'sha256:<64 hex>'")
    return issues


def _task_digest_truth_issues(result: object, label: str) -> list[str]:
    """Recompute the task content digest when the task dir is still on disk.

    Complements ``_source_hash_truth_issues``: ``source.file_hashes`` audits
    per-file hashes, this audits the single registry-algorithm digest the
    leaderboard pipeline keys on.
    """
    if not isinstance(result, dict):
        return []
    result_dict = cast(dict[str, Any], result)
    digest = result_dict.get("task_digest")
    if not is_sha256_digest(digest):
        return []  # absent or malformed — format issues reported elsewhere
    source = result_dict.get("source")
    if not isinstance(source, dict):
        return []
    local_path_raw = source.get("local_path")
    if not isinstance(local_path_raw, str):
        return []
    local_path = Path(local_path_raw)
    if not (local_path / "task.toml").exists():
        return []
    try:
        from benchflow._utils.task_authoring import task_digest

        actual = task_digest(local_path)
    except Exception as e:
        return [f"{label}: task_digest could not be recomputed: {e}"]
    if actual != digest:
        return [f"{label}: task_digest does not match local_path content"]
    return []


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
    summary_path = _summary_path(agent_dir, result_root)
    summary, summary_error = _load_summary(summary_path)

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
        skill_count = r.get("n_skill_invocations")
        agent_result = r.get("agent_result") or {}
        agent_skill_count = agent_result.get("n_skill_invocations")
        trajectory = _load_acp_trajectory(result_file)
        trajectory_skill_count = (
            count_skill_invocations(trajectory) if trajectory is not None else None
        )
        if skill_count is not None:
            issue = _nonnegative_int_issue(
                skill_count, f"{r.get('task_name', '?')}: n_skill_invocations"
            )
            if issue:
                findings["issues"].append(issue)
                findings["ok"] = False
            if agent_skill_count is not None and agent_skill_count != skill_count:
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: agent_result.n_skill_invocations "
                    "does not match result.json n_skill_invocations"
                )
                findings["ok"] = False
            if (
                trajectory_skill_count is not None
                and skill_count != trajectory_skill_count
            ):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: n_skill_invocations={skill_count} "
                    f"but trajectory implies {trajectory_skill_count}"
                )
                findings["ok"] = False
        # Effective count for the no-skill invariant: prefer the recorded value,
        # fall back to the trajectory so artifacts that omit the field are still
        # held to the invariant.
        effective_skill_count = (
            skill_count if skill_count is not None else trajectory_skill_count
        )
        if agent_skill_count is not None:
            issue = _nonnegative_int_issue(
                agent_skill_count,
                f"{r.get('task_name', '?')}: agent_result.n_skill_invocations",
            )
            if issue:
                findings["issues"].append(issue)
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
        dataset_stamp_issues = _dataset_stamp_issues(
            r, f"{r.get('task_name', '?')}: result.json"
        )
        if dataset_stamp_issues:
            findings["issues"].extend(dataset_stamp_issues)
            findings["ok"] = False
        digest_truth_issues = _task_digest_truth_issues(
            r, f"{r.get('task_name', '?')}: result.json"
        )
        if digest_truth_issues:
            findings["issues"].extend(digest_truth_issues)
            findings["ok"] = False
        _config_path, config, config_error = _load_config(result_file)
        if config_error:
            findings["issues"].append(f"{r.get('task_name', '?')}: {config_error}")
            findings["ok"] = False
        else:
            assert config is not None
            # Experiment-health invariant: a trial that provisioned no skills
            # must not record skill invocations. A positive count here is a
            # detector false-positive, agent-native skill leakage, or an
            # experiment-fidelity failure — quarantine it for human review.
            if (
                isinstance(effective_skill_count, int)
                and not isinstance(effective_skill_count, bool)
                and effective_skill_count > 0
                and not _config_provisions_skills(config)
            ):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: n_skill_invocations="
                    f"{effective_skill_count} but config provisions no skills "
                    "(no-skill trial must not invoke skills — quarantine: detector "
                    "false-positive, agent-native skill leakage, or experiment "
                    "fidelity failure)"
                )
                findings["ok"] = False
            skill_file_evidence = _no_skill_task_file_access_evidence(
                trajectory,
                config=config,
                result=r,
            )
            if skill_file_evidence and not _config_provisions_skills(config):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: trajectory accessed task skill "
                    "files but config provisions no skills (no-skill trial must "
                    f"not access task skill files): {skill_file_evidence[0]}"
                )
                findings["ok"] = False
            expected_model = _expected(agent, "model")
            expected_environment = _expected(agent, "environment")
            expected_concurrency = _expected_config_concurrency(agent, summary)
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
            config_dataset_issues = _dataset_stamp_issues(
                config, f"{r.get('task_name', '?')}: config.json"
            )
            if config_dataset_issues:
                findings["issues"].extend(config_dataset_issues)
                findings["ok"] = False
            if any(config.get(k) != r.get(k) for k in _DATASET_STAMP_KEYS):
                findings["issues"].append(
                    f"{r.get('task_name', '?')}: config.json dataset stamp "
                    f"does not match result.json"
                )
                findings["ok"] = False

    # Infrastructure errors — driven from the central diagnostic registry
    # so adding a new diagnostic kind doesn't require editing this file
    # (issue #503).
    from benchflow.diagnostics import DIAGNOSTIC_REGISTRY, format_issue_for_field

    _CATEGORY_TO_DIAGNOSTIC = {
        d.category: d
        for d in DIAGNOSTIC_REGISTRY
        if d.category and d.channel == "error"
    }
    _INVALIDATION_DESCRIPTIONS = {
        "idle_timeout": ("hit idle timeout", ""),
        "sandbox_setup": ("failed during sandbox startup", ""),
        "pipe_closed": (
            "lost ACP transport (pipe closed / rc=255)",
            "",
        ),
    }

    infra_errors: list[str] = []
    invalidated_by_category: dict[str, list[str]] = {}
    for r in latest_results:
        err = r.get("error")
        cat = r.get("error_category") or (classify_error(str(err)) if err else None)
        if cat and cat in INFRA_ERROR_CATEGORIES:
            task = r.get("task_name", "?")
            diag_cls = _CATEGORY_TO_DIAGNOSTIC.get(cat)
            info = r.get(diag_cls.field) if diag_cls else None
            if diag_cls and info:
                infra_errors.append(format_issue_for_field(diag_cls.field, task, info))
            else:
                infra_errors.append(f"{task}: {err}")
            if cat in _INVALIDATION_DESCRIPTIONS:
                invalidated_by_category.setdefault(cat, []).append(task)
    if infra_errors:
        findings["issues"].extend(infra_errors)
        findings["ok"] = False
    for cat, tasks in invalidated_by_category.items():
        desc, _ = _INVALIDATION_DESCRIPTIONS[cat]
        findings["issues"].append(
            f"INVALIDATED: {len(tasks)} task(s) {desc} "
            f"and should be rerun: {', '.join(tasks)}"
        )

    # Verifier dependency install failures (ENG-151)
    dep_install_tasks: list[str] = []
    for r in latest_results:
        verifier_err = r.get("verifier_error")
        vcat = r.get("verifier_error_category") or (
            classify_verifier_error(verifier_err) if verifier_err else None
        )
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

    # Verifier timeout failures (ENG-152) — rendered from the diagnostic
    # registry; falls back to a plain message when the structured
    # verifier_timeout_info dict is absent.
    _VERIFIER_DIAGS = {
        d.category: d for d in DIAGNOSTIC_REGISTRY if d.channel == "verifier_error"
    }
    verifier_timeout_tasks: list[str] = []
    for r in latest_results:
        verifier_err = r.get("verifier_error")
        vcat = classify_verifier_error(verifier_err) if verifier_err else None
        if vcat == "verifier_timeout":
            task = r.get("task_name", "?")
            diag_cls = _VERIFIER_DIAGS.get(vcat)
            vti = r.get(diag_cls.field) if diag_cls else None
            verifier_timeout_tasks.append(task)
            if diag_cls and vti:
                findings["issues"].append(
                    format_issue_for_field(diag_cls.field, task, vti)
                )
            else:
                findings["issues"].append(
                    f"{task}: verifier timed out — measurement invalid "
                    f"(verifier never produced reward)"
                )
            findings["ok"] = False
    if verifier_timeout_tasks:
        findings["issues"].append(
            f"INVALIDATED: {len(verifier_timeout_tasks)} task(s) had verifier "
            f"timeouts — increase timeout_sec or reduce verifier cost: "
            f"{', '.join(verifier_timeout_tasks)}"
        )

    # Summary.json — bench eval run writes it at the agent_dir root
    if summary_error is None:
        assert summary is not None
        worker_sharded_summary = _is_worker_sharded_summary(summary)
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
        # Dataset identity must agree across summary.json and every
        # result.json — one leaderboard namespace per dataset version (#691).
        for dataset_key in ("dataset_name", "dataset_version"):
            dataset_values = {r.get(dataset_key) for r in results}
            dataset_values.add(summary.get(dataset_key))
            if len(dataset_values) > 1:
                findings["issues"].append(
                    f"summary.json {dataset_key} does not agree across "
                    f"summary and results: {sorted(map(repr, dataset_values))}"
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
        summary_model_present = "model" in summary
        if expected_model is not _MISSING and (
            summary.get("model") != expected_model
            and (summary_model_present or not worker_sharded_summary)
        ):
            findings["issues"].append(
                f"summary.json model {summary.get('model')!r} does not match expected {_expected_label(expected_model)}"
            )
            findings["ok"] = False
        summary_environment_present = "environment" in summary
        if expected_environment is not _MISSING and (
            summary.get("environment") != expected_environment
            and (summary_environment_present or not worker_sharded_summary)
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
        for field in ("total_skill_invocations", "avg_skill_invocations"):
            if field in summary:
                value = summary[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or value < 0
                ):
                    findings["issues"].append(
                        f"summary.json {field} must be a non-negative number"
                    )
                    findings["ok"] = False
        findings["summary"] = summary
    else:
        findings["issues"].append(summary_error)
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
        if "total_skill_invocations" in summary:
            implied = sum(result_skill_invocations(result) for result in latest_results)
            if summary["total_skill_invocations"] != implied:
                findings["issues"].append(
                    "summary.json total_skill_invocations="
                    f"{summary['total_skill_invocations']} but results imply {implied}"
                )
                findings["ok"] = False
        if "avg_skill_invocations" in summary:
            total = sum(result_skill_invocations(result) for result in latest_results)
            implied = round(total / len(latest_results), 1) if latest_results else 0.0
            if abs(float(summary["avg_skill_invocations"]) - implied) > 1e-9:
                findings["issues"].append(
                    "summary.json avg_skill_invocations="
                    f"{summary['avg_skill_invocations']} but results imply {implied}"
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
