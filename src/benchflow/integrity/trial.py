"""Glue between trials and the integrity pipeline.

- :func:`write_rollout_integrity`: called by the rollout at the end of scoring
  when ``integrity`` is ``audit`` or ``strict``.
- :func:`audit_trial`: re-verdict a stored trial folder offline (older jobs,
  replayed corpus trajectories, a hill-climb's trials). The checker is a pure
  function of the stored evidence.
- :func:`separate_launch_issues` (``strict_launch_issues``),
  :func:`check_separate_plan`, :func:`resolve_verifier_isolation` and
  :func:`force_separate_verifier`: strict mode, and the run-level
  ``--verifier-sandbox separate`` switch, run the verifier in BenchFlow's own
  separate verifier sandbox.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from benchflow.errors import UserError
from benchflow.integrity.access import (
    ACCESS_PROBE_FILE,
    probe_paths,
    run_access_probe,
)
from benchflow.integrity.constants import normalize_integrity_mode
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


# --- strict mode ---------------------------------------------------------------


def separate_launch_issues(
    task_config: Any,
    *,
    sandbox: str,
    task_dir: Path | None,
    environment_manifest: Any = None,
) -> list[str]:
    """Why the separate verifier sandbox cannot run this task here.

    From 0.8's own launch gate: the refusals are exactly the ones BenchFlow
    gives a task that declares ``environment_mode = "separate"`` itself: the
    backend, the verifier strategy and service, the verifier sandbox's
    network, whether a verifier image can be planned (and whether it would
    start compose services), multi-step tasks, services the task keeps for
    grading (``x-keep-services``: the verifier sandbox has none), and declared transfer roots
    that overlap a verifier-owned path (``verifier.sandbox.transfer``).
    ``--verifier-sandbox separate`` and ``--integrity strict`` both check
    every task with it before a job folder exists, and ``bench tasks check
    --verifier-sandbox separate`` prints the same lines.

    ``environment_manifest`` is the manifest the agent's sandbox gets (the
    run's ``--environment-manifest``/``--state``, or the task document's).
    When the manifest runs the agent's sandbox from its own ``image``, a
    verifier image that only mirrors the agent's environment (the
    ``environment/Dockerfile`` fallback, or a copy of ``[sandbox]
    docker_image``) is another image, so the tests would run elsewhere than
    the agent did: refused, with the next step. The manifest's services
    and state are like any other answer outside the transfer (a service, a
    database): not refused here, documented as the switch's limit.
    """

    from benchflow.task.config import VerifierSandboxMode
    from benchflow.task.runtime_capabilities import (
        KEPT_SERVICES_ISSUE_PATH,
        validate_task_runtime_support,
    )

    forced = _with_separate_verifier(task_config, VerifierSandboxMode.SEPARATE)
    issues = validate_task_runtime_support(forced, sandbox=sandbox, task_dir=task_dir)
    lines = [
        issue.format()
        for issue in issues
        if issue.path.startswith("verifier.sandbox")
        or issue.path in ("steps", KEPT_SERVICES_ISSUE_PATH)
    ]
    if environment_manifest is not None and task_dir is not None:
        manifest_issue = _manifest_image_issue(
            forced, Path(task_dir), environment_manifest, sandbox=sandbox
        )
        if manifest_issue is not None:
            lines.append(manifest_issue)
    return lines


#: Verifier images that stand for "the agent's environment, fresh": they are
#: only right when the agent's sandbox is exactly what the task declares.
_MIRRORED_IMAGE_SOURCES = ("environment/Dockerfile", "sandbox.docker_image")


def _manifest_image_issue(
    forced_config: Any, task_dir: Path, manifest: Any, *, sandbox: str
) -> str | None:
    from benchflow.environment.manifest import resolve_manifest_image
    from benchflow.task.runtime_capabilities import UnsupportedTaskFeature
    from benchflow.task.verifier_sandbox import (
        SeparateVerifierError,
        plan_verifier_image,
    )

    image = resolve_manifest_image(manifest)
    if not image:
        return None  # base_image only: the agent's image is the task's
    try:
        plan = plan_verifier_image(forced_config, task_dir)
    except SeparateVerifierError:
        return None  # already refused as "no verifier image"
    if plan.source not in _MIRRORED_IMAGE_SOURCES:
        return None
    name = getattr(manifest, "name", None) or "the environment manifest"
    return UnsupportedTaskFeature(
        path="verifier.sandbox",
        reason=(
            f"environment manifest {name!r} runs the agent's sandbox from "
            f"image {image!r}, but the verifier image would come from "
            f"{plan.source}, so the tests would run in another image than the "
            "agent's; add tests/Dockerfile or a [verifier.sandbox] "
            "docker_image, or run without the manifest"
        ),
        sandbox=sandbox,
    ).format()


def _task_document_manifest(task_dir: Path) -> Any:
    """The manifest a task.md task declares for its agent's sandbox, as
    Evaluation gives it to the trial when the run names none; None when it
    declares none or it does not load (the trial reports that itself)."""
    from benchflow.evaluation import _environment_manifest_from_task_document

    try:
        return _environment_manifest_from_task_document(task_dir)
    except Exception:
        return None


#: The name strict mode's gate had before the run-level switch shared it.
strict_launch_issues = separate_launch_issues

VerifierIsolation = tuple[str, str]


def run_forces_separate(run_setting: Any, integrity: Any) -> bool:
    """Whether the run moves every verifier into the separate sandbox:
    ``verifier_sandbox = separate``, or ``integrity = strict`` (which implies it)."""
    from benchflow.task.verifier_sandbox import normalize_verifier_sandbox

    return (
        normalize_verifier_sandbox(run_setting) == "separate"
        or normalize_integrity_mode(integrity) == "strict"
    )


def resolve_verifier_isolation(
    task_config: Any, run_setting: Any, integrity: Any
) -> VerifierIsolation:
    """Where this task's verifier runs in this run, and why.

    Returns ``(mode, source)``: ``mode`` is ``separate`` or ``shared``;
    ``source`` is ``task`` when the task's own declaration decides (a task
    that declares separate stays separate whatever the run says),
    ``integrity-strict`` when ``--integrity strict`` moves it, and
    ``run-flag`` when ``--verifier-sandbox separate`` does. Strict wins over
    the switch as the source because it also brings the audit.
    """
    from benchflow.task.verifier_sandbox import (
        normalize_verifier_sandbox,
        separate_verifier_requested,
    )

    setting = normalize_verifier_sandbox(run_setting)
    if separate_verifier_requested(task_config):
        return "separate", "task"
    if normalize_integrity_mode(integrity) == "strict":
        return "separate", "integrity-strict"
    if setting == "separate":
        return "separate", "run-flag"
    return "shared", "task"


def _run_switch_phrase(run_setting: Any, integrity: Any) -> tuple[str, str]:
    """(what moved the verifiers, how to run without it) for a refusal."""
    if normalize_integrity_mode(integrity) == "strict":
        return "--integrity strict", "run with --integrity audit"
    return "--verifier-sandbox separate", "run without --verifier-sandbox separate"


def separate_refusal_lines(task_name: str, issues: list[str]) -> list[str]:
    """One line per refusal, ``<task>: <issue>``, as the plan gate and
    ``bench tasks check --verifier-sandbox separate`` both print them."""
    return [f"{task_name}: {issue}" for issue in issues]


class SeparateVerifierRefusedError(ValueError, UserError):
    """The run moves every verifier into the separate verifier sandbox, and
    some selected tasks cannot use it there. Raised before any job or shard
    folder exists and before any sandbox starts. The message ends with the
    next step."""

    fault = "task"

    def __init__(
        self,
        refused: dict[str, list[str]],
        *,
        total: int,
        sandbox: str,
        run_setting: Any,
        integrity: Any,
    ) -> None:
        self.refused = dict(refused)
        switch, without = _run_switch_phrase(run_setting, integrity)
        names = sorted(self.refused)
        lines = [
            line
            for name in names
            for line in separate_refusal_lines(name, self.refused[name])
        ]
        self.lines = lines
        excludes = " ".join(f"--exclude {name}" for name in names)
        message = (
            f"{switch} runs every task's verifier in a separate verifier "
            f"sandbox, which {len(names)} of {total} selected task(s) cannot use "
            f"on {sandbox}; nothing ran:\n- " + "\n- ".join(lines) + "\n"
            f"next: drop those tasks ({excludes}), or {without}"
        )
        super().__init__(message)


def check_separate_plan(
    task_dirs: list[Path],
    *,
    sandbox: str,
    run_setting: Any,
    integrity: Any,
    config_override: dict | None = None,
    environment_manifest: Any = None,
) -> None:
    """Refuse a run that forces the separate verifier onto tasks that cannot
    use it, before any job folder exists.

    A no-op unless the run forces separate (:func:`run_forces_separate`).
    Every task is checked with :func:`separate_launch_issues` after the run's
    overlay is applied, as Rollout.setup applies it, and with the manifest
    its agent's sandbox gets (``environment_manifest``, else the task
    document's); every refused task is named in one
    :class:`SeparateVerifierRefusedError`. A task that does not load is left
    to its trial, which reports it as today.
    """
    if not run_forces_separate(run_setting, integrity):
        return
    refused: dict[str, list[str]] = {}
    for task_dir in task_dirs:
        config = _load_task_config(Path(task_dir))
        if config is None:
            continue
        if config_override:
            from benchflow._utils.config_override import apply_config_override

            try:
                config = apply_config_override(config, config_override)
            except Exception:
                continue  # the trial refuses the overlay with its own message
        manifest = environment_manifest
        if manifest is None:
            manifest = _task_document_manifest(Path(task_dir))
        issues = separate_launch_issues(
            config,
            sandbox=sandbox,
            task_dir=task_dir,
            environment_manifest=manifest,
        )
        if issues:
            refused[Path(task_dir).name] = issues
    if refused:
        raise SeparateVerifierRefusedError(
            refused,
            total=len(task_dirs),
            sandbox=sandbox,
            run_setting=run_setting,
            integrity=integrity,
        )


def separate_task_check(
    task_dir: Path, *, sandbox: str
) -> tuple[SeparateVerifierRefusedError | None, str | None]:
    """``bench tasks check --verifier-sandbox separate`` for one task folder.

    Returns the refusal the plan gate would raise for this task alone (None
    when it would run), and, when it would run, where its verifier image
    comes from (``plan_verifier_image``'s source: a ``docker_image``,
    ``tests/Dockerfile`` or the ``environment/Dockerfile`` fallback).
    """
    try:
        check_separate_plan(
            [task_dir], sandbox=sandbox, run_setting="separate", integrity="off"
        )
    except SeparateVerifierRefusedError as refusal:
        return refusal, None
    config = _load_task_config(Path(task_dir))
    if config is None:
        return None, None
    from benchflow.task.verifier_sandbox import (
        SeparateVerifierError,
        plan_verifier_image,
    )

    try:
        plan = plan_verifier_image(force_separate_verifier(config), Path(task_dir))
    except SeparateVerifierError:
        return None, None
    return None, plan.source


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
    """Audit a finished rollout; never raises into scoring, never touches rewards.

    The outer guard is what makes that promise hold: the fallback below writes
    a Rejected verdict to the trial folder, and the failure that made the audit
    fail (a full or read-only disk, a permission error) is exactly the failure
    that can make the fallback's own write fail. An audit that cannot even
    record its own failure is logged and dropped; scoring never sees it.
    """

    try:
        return _audit_finished_rollout(rollout, result)
    except Exception as exc:  # the audit must never fail the trial
        logger.warning(
            "Integrity audit could not be recorded for %s: %s",
            getattr(rollout, "_rollout_dir", "<unknown trial>"),
            exc,
            exc_info=True,
        )
        return None


def _audit_finished_rollout(rollout: Any, result: Any) -> IntegrityVerdict | None:
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
        # One line at the default level; the traceback is for --verbose.
        logger.warning(
            "Integrity audit failed for %s (%s: %s); its verdict is Rejected, "
            "in %s/claim_verdict.json",
            trial_dir.name,
            type(exc).__name__,
            exc,
            trial_dir / INTEGRITY_DIRNAME,
        )
        logger.debug("Integrity audit traceback for %s", trial_dir.name, exc_info=True)
        claim = emit_integrity_error(trial_dir, mode=mode, error=exc)
    verdict = IntegrityVerdict.from_claim(
        claim, path=trial_dir / INTEGRITY_DIRNAME / "claim_verdict.json"
    )
    if verdict.exploited:
        logger.warning("Integrity (%s) %s: %s", mode, trial_dir.name, verdict.reason)
    return verdict


def _rollout_contract(rollout: Any) -> IntegrityContract:
    cfg = rollout._config
    task = rollout._task
    config = getattr(task, "config", None)
    task_dir = Path(cfg.task_path)
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
    return derive_contract(
        facts, binding=load_binding(binding_path) if binding_path else None
    )


async def record_access_probe(rollout: Any, phase: str) -> None:
    """Probe the protected paths as root, before (``pre``) or after (``post``)
    the agent's turn, into ``integrity/access_probe.json``. Never raises.

    The scripted controls are recorded as such: the oracle reads the solution
    as root by design, and nop runs nothing.
    """

    from benchflow.agents.registry import is_scripted_agent

    try:
        cfg = rollout._config
        record = getattr(rollout, "_integrity_access", None)
        if record is None:
            record = {}
            rollout._integrity_access = record
        if is_scripted_agent(cfg.primary_agent):
            record[phase] = {"status": "scripted"}
        else:
            paths, _ = probe_paths(_rollout_contract(rollout))
            record[phase] = await run_access_probe(
                rollout._env, cfg.sandbox_user, paths
            )
        out = Path(rollout._rollout_dir) / INTEGRITY_DIRNAME
        out.mkdir(parents=True, exist_ok=True)
        (out / ACCESS_PROBE_FILE).write_text(json.dumps(record, indent=2))
    except Exception as exc:  # the probe must never fail the trial
        logger.warning("Integrity access probe (%s) failed: %s", phase, exc)


def evidence_from_rollout(rollout: Any, result: Any, *, mode: str) -> TrialEvidence:
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    cfg = rollout._config
    trial_dir = Path(rollout._rollout_dir)
    separate = separate_verifier_requested(getattr(rollout._task, "config", None))
    contract = _rollout_contract(rollout)
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
        egress_denials=_read_egress_denials(trial_dir),
        access_probe=getattr(rollout, "_integrity_access", None)
        or _read_json(trial_dir / INTEGRITY_DIRNAME / ACCESS_PROBE_FILE),
        source=rollout_evidence_source(result),
    )


def rollout_evidence_source(result: Any) -> str:
    """``rollout`` when the result holds the run's own trajectory, before
    redaction; ``rollout_stored`` when it was read back from the stored,
    redacted copy (resumed scoring), which is then re-audited, not live."""
    return (
        "rollout_stored" if getattr(result, "_trajectory_stored", False) else "rollout"
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
    # Environment backing stores come from the resolved run environment,
    # not task.toml. Keep those host-recorded restrictions when re-auditing.
    if stored_manifest:
        recorded_contract = IntegrityContract.model_validate(stored_manifest)
        facts.state_paths = [
            resource.path
            for resource in recorded_contract.resources
            if resource.id.startswith("resource.environment_state_")
            and resource.resource_class == "Hidden"
            and resource.kind == "path"
            and resource.path is not None
        ]
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
        egress_denials=_read_egress_denials(trial_dir),
        access_probe=_read_json(trial_dir / INTEGRITY_DIRNAME / ACCESS_PROBE_FILE),
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
    return _jsonl_rows(text)


def _read_egress_denials(trial_dir: Path) -> list[dict[str, Any]]:
    """Read the current transport and the host's retained reconnect epochs.

    Different transports remain distinct even when their rows match: the
    proxy timestamps have only second precision, not unique request IDs.
    Only the same underlying file reached twice is counted once. Original
    archive and canonical logs are never rewritten or combined on disk.
    """
    from benchflow._utils.reachable_write import (
        ReachableWriteError,
        destination,
        source_capability,
    )

    try:
        # A live trial reuses its pre-launch held root. An offline audit may
        # choose a stored trial root; descendants are still read no-follow.
        try:
            root = destination(
                Path(os.path.abspath(trial_dir)) / "network-transports"
            ).parent
        except ReachableWriteError:
            root = source_capability(trial_dir)
        logs = [root / EGRESS_LOG]
        with contextlib.suppress(OSError):
            logs.extend(
                archive / EGRESS_LOG
                for archive in sorted((root / "network-transports").iterdir())
                if re.fullmatch(r"[0-9a-f]{32}", archive.name)
            )
    except OSError:
        return []

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for path in logs:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                info = os.fstat(stream.fileno())
                identity = (info.st_dev, info.st_ino)
                if identity in seen:
                    continue
                rows.extend(_jsonl_rows(stream.read()))
                seen.add(identity)
        except OSError:
            # A symlink, special file or replaced ancestor is not evidence.
            continue
    return rows


def _jsonl_rows(text: str) -> list[dict[str, Any]]:
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
    "SeparateVerifierRefusedError",
    "audit_trial",
    "check_separate_plan",
    "evidence_from_rollout",
    "facts_from_task_config",
    "force_separate_verifier",
    "normalize_integrity_mode",
    "resolve_verifier_isolation",
    "run_forces_separate",
    "separate_launch_issues",
    "separate_refusal_lines",
    "separate_task_check",
    "strict_launch_issues",
    "write_rollout_integrity",
]
