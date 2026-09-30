"""Glue between trials and the integrity pipeline.

- :func:`write_rollout_integrity`: called by the rollout at the end of scoring
  when ``integrity`` is ``audit`` or ``strict``.
- :func:`audit_trial`: re-verdict a stored trial folder offline (older jobs,
  replayed corpus trajectories, a hill-climb's trials). The checker is a pure
  function of the stored evidence.
- :func:`strict_launch_issues` and :func:`force_separate_verifier`: strict
  mode runs the verifier in BenchFlow's own separate verifier sandbox.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from benchflow.integrity.constants import INTEGRITY_MODES
from benchflow.integrity.contract import (
    ContractFacts,
    IntegrityContract,
    derive_contract,
    dockerfile_workdir,
    find_binding,
    load_binding,
)
from benchflow.integrity.emit import (
    INTEGRITY_DIRNAME,
    TrialEvidence,
    emit_integrity,
    emit_integrity_error,
)
from benchflow.integrity.verdict import IntegrityVerdict

logger = logging.getLogger(__name__)

SEPARATE_VERIFIER_RECORD = Path("verifier-sandbox") / "verifier-sandbox.json"
EGRESS_LOG = Path("trajectory") / "egress_denylist.jsonl"


def normalize_integrity_mode(value: Any) -> str:
    """``off``, ``audit`` or ``strict``; None and False mean ``off``."""

    if value is None or value is False:
        return "off"
    if value is True:
        return "audit"
    text = str(value).strip().lower()
    if text in INTEGRITY_MODES:
        return text
    raise ValueError(
        f"integrity must be one of {', '.join(INTEGRITY_MODES)}, got {value!r}"
    )


# --- strict mode ---------------------------------------------------------------


def strict_launch_issues(
    task_config: Any, *, sandbox: str, task_dir: Path | None
) -> list[str]:
    """Why strict mode cannot run this task here, from 0.8's own launch gate.

    Strict mode is the separate verifier sandbox, so the refusals are exactly
    the ones BenchFlow gives a task that declares ``environment_mode =
    "separate"`` itself: the backend, the verifier strategy and service, and
    whether a verifier image can be planned.
    """

    from benchflow.task.config import VerifierSandboxMode
    from benchflow.task.runtime_capabilities import validate_task_runtime_support

    forced = _with_separate_verifier(task_config, VerifierSandboxMode.SEPARATE)
    issues = validate_task_runtime_support(forced, sandbox=sandbox, task_dir=task_dir)
    return [
        issue.format()
        for issue in issues
        if issue.path.startswith("verifier.sandbox") or issue.path == "steps"
    ]


def force_separate_verifier(task_config: Any) -> Any:
    """``task_config`` with its verifier moved to a separate sandbox.

    The run-time config overlay refuses verifier keys so that nothing can
    weaken scoring per run; this only moves the verifier to where the agent
    never was.
    """

    from benchflow.task.config import VerifierSandboxMode

    return _with_separate_verifier(task_config, VerifierSandboxMode.SEPARATE)


def _with_separate_verifier(task_config: Any, mode: Any) -> Any:
    verifier = task_config.verifier.model_copy(update={"sandbox_mode": mode})
    return task_config.model_copy(update={"verifier": verifier})


# --- live rollouts -------------------------------------------------------------


def write_rollout_integrity(rollout: Any, result: Any) -> IntegrityVerdict | None:
    """Audit a finished rollout; never raises into scoring, never touches rewards."""

    cfg = rollout._config
    mode = normalize_integrity_mode(getattr(cfg, "integrity", "off"))
    trial_dir = getattr(rollout, "_rollout_dir", None)
    if mode == "off" or trial_dir is None:
        return None
    trial_dir = Path(trial_dir)
    try:
        evidence = evidence_from_rollout(rollout, result, mode=mode)
        claim = emit_integrity(evidence)
    except Exception as exc:  # the audit must never fail the trial
        logger.warning(
            "Integrity audit failed for %s: %s", trial_dir.name, exc, exc_info=True
        )
        claim = emit_integrity_error(trial_dir, mode=mode, error=exc)
    verdict = IntegrityVerdict.from_claim(
        claim, path=trial_dir / INTEGRITY_DIRNAME / "claim_verdict.json"
    )
    if verdict.exploited:
        logger.warning("Integrity (%s) %s: %s", mode, trial_dir.name, verdict.reason)
    return verdict


def evidence_from_rollout(rollout: Any, result: Any, *, mode: str) -> TrialEvidence:
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    cfg = rollout._config
    trial_dir = Path(rollout._rollout_dir)
    task = rollout._task
    config = getattr(task, "config", None)
    task_dir = Path(cfg.task_path)
    separate = separate_verifier_requested(config)
    facts = facts_from_task_config(
        config,
        task_id=getattr(task, "name", None) or task_dir.name,
        workspace=getattr(rollout, "_agent_cwd", None)
        or _config_workdir(config)
        or "/app",
        skills_dir=getattr(rollout, "_effective_skills_sandbox_dir", None)
        if getattr(rollout, "_effective_skills_dir", None) is not None
        else None,
        environment_manifest=getattr(cfg, "environment_manifest", None),
    )
    binding_path = find_binding(task_dir)
    contract = derive_contract(
        facts, binding=load_binding(binding_path) if binding_path else None
    )
    return TrialEvidence(
        trial_dir=trial_dir,
        run_id=getattr(rollout, "_rollout_name", None) or trial_dir.name,
        mode=mode,
        contract=contract,
        agent=str(cfg.primary_agent),
        trajectory=list(getattr(result, "trajectory", None) or []),
        trajectory_source=getattr(result, "trajectory_source", None),
        rewards=getattr(result, "rewards", None),
        verifier_error=getattr(result, "verifier_error", None),
        verifier_error_category=getattr(result, "verifier_error_category", None),
        error=getattr(result, "error", None),
        sandbox_user=cfg.sandbox_user,
        environment=str(cfg.environment),
        separate_verifier=separate,
        separate_verifier_record=_read_json(trial_dir / SEPARATE_VERIFIER_RECORD),
        egress_denials=_read_jsonl(trial_dir / EGRESS_LOG),
        source="rollout",
    )


def facts_from_task_config(
    config: Any,
    *,
    task_id: str,
    workspace: str,
    skills_dir: str | None = None,
    environment_manifest: Any = None,
) -> ContractFacts:
    """Contract facts from a parsed 0.8 ``TaskConfig`` (None gives defaults)."""

    if config is None:
        return ContractFacts(task_id=task_id, workspace=workspace)
    from benchflow.sandbox.egress_denylist import agent_network_sandbox_config

    network = agent_network_sandbox_config(config)
    raw_mode = getattr(network, "network_mode", None)
    mode = str(getattr(raw_mode, "value", raw_mode) or "public")
    if getattr(getattr(config, "sandbox", None), "allow_internet", True) is False:
        mode = "no-network"
    artifact_paths: list[str] = []
    for item in getattr(config, "artifacts", None) or []:
        source = item if isinstance(item, str) else getattr(item, "source", None)
        if isinstance(source, str) and source:
            artifact_paths.append(source)
    state = getattr(environment_manifest, "state", None)
    reward_range = getattr(getattr(config, "verifier", None), "reward_range", None)
    return ContractFacts(
        task_id=task_id,
        workspace=workspace,
        agent_network_mode=mode,
        allowed_hosts=list(getattr(network, "allowed_hosts", None) or []),
        skills_dir=skills_dir,
        artifact_paths=artifact_paths,
        state_paths=list(getattr(state, "paths", None) or []),
        reward_range=tuple(reward_range) if reward_range else None,
        metadata=dict(getattr(config, "metadata", None) or {}),
    )


# --- stored trials ---------------------------------------------------------------


def audit_trial(
    trial_dir: str | Path,
    *,
    task_path: str | Path | None = None,
    mode: str | None = None,
    workspace: str | None = None,
) -> IntegrityVerdict:
    """Re-verdict a stored trial folder from what the host recorded.

    Reads ``result.json``, ``config.json``, ``trajectory/acp_trajectory.jsonl``,
    the separate verifier's record and the egress log; derives the contract
    from ``task_path`` (or the task folder ``config.json`` names, when it still
    exists). Writes ``integrity/`` like a live run and returns the verdict.
    Nothing about the trial's reward or result changes.
    """

    trial_dir = Path(trial_dir)
    result = _read_json(trial_dir / "result.json") or {}
    config = _read_json(trial_dir / "config.json") or {}
    stored_manifest = _read_json(trial_dir / INTEGRITY_DIRNAME / "manifest.json") or {}
    resolved_mode = normalize_integrity_mode(mode or config.get("integrity") or "audit")
    if resolved_mode == "off":
        resolved_mode = "audit"

    task_dir = _task_dir(task_path, config)
    task_config = _load_task_config(task_dir) if task_dir is not None else None
    task_name = str(
        result.get("task_name") or (task_dir.name if task_dir else trial_dir.name)
    )
    resolved_workspace = (
        workspace
        or stored_manifest.get("workspace")
        or _config_workdir(task_config)
        or dockerfile_workdir(task_dir)
        or "/app"
    )
    facts = facts_from_task_config(
        task_config, task_id=task_name, workspace=resolved_workspace
    )
    binding_path = find_binding(task_dir)
    contract: IntegrityContract = derive_contract(
        facts, binding=load_binding(binding_path) if binding_path else None
    )
    if task_dir is None:
        contract.violations.append("task_package_missing")
    separate_record = _read_json(trial_dir / SEPARATE_VERIFIER_RECORD)
    evidence = TrialEvidence(
        trial_dir=trial_dir,
        run_id=str(result.get("rollout_name") or trial_dir.name),
        mode=resolved_mode,
        contract=contract,
        agent=str(result.get("agent") or config.get("agent") or ""),
        trajectory=_read_jsonl(trial_dir / "trajectory" / "acp_trajectory.jsonl"),
        trajectory_source=result.get("trajectory_source"),
        rewards=result.get("rewards")
        if isinstance(result.get("rewards"), dict)
        else None,
        verifier_error=result.get("verifier_error"),
        verifier_error_category=result.get("verifier_error_category"),
        error=result.get("error"),
        sandbox_user=config.get("sandbox_user", "agent"),
        environment=str(config.get("environment") or ""),
        separate_verifier=separate_record is not None,
        separate_verifier_record=separate_record,
        egress_denials=_read_jsonl(trial_dir / EGRESS_LOG),
        source="audit_trial",
    )
    claim = emit_integrity(evidence)
    return IntegrityVerdict.from_claim(
        claim, path=trial_dir / INTEGRITY_DIRNAME / "claim_verdict.json"
    )


def _task_dir(task_path: str | Path | None, config: dict[str, Any]) -> Path | None:
    candidates = [task_path]
    recorded = config.get("task_path")
    if isinstance(recorded, str) and Path(recorded).is_absolute():
        candidates.append(recorded)
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.is_dir():
            return path
    return None


def _load_task_config(task_dir: Path) -> Any:
    try:
        from benchflow.task import Task

        return Task(task_dir).config
    except Exception as exc:
        logger.warning(
            "Could not load task %s for the integrity contract: %s", task_dir, exc
        )
        return None


def _config_workdir(config: Any) -> str | None:
    workdir = getattr(getattr(config, "sandbox", None), "workdir", None)
    return workdir if isinstance(workdir, str) and workdir.startswith("/") else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


__all__ = [
    "audit_trial",
    "evidence_from_rollout",
    "facts_from_task_config",
    "force_separate_verifier",
    "normalize_integrity_mode",
    "strict_launch_issues",
    "write_rollout_integrity",
]
