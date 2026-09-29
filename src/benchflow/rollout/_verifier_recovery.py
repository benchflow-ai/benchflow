"""Verifier-only recovery from immutable workspace evidence (#1136).

This is filesystem recovery, not a VM checkpoint. Tasks with additional mutable
services or external artifacts are not eligible; their evidence remains available
for operator recovery without silently replaying a completed solver.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from benchflow._types import Scene
from benchflow._utils.config_override import apply_config_override
from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow._utils.task_authoring import task_digest
from benchflow._utils.text import describe_exception
from benchflow.contracts import default_rollout_planes
from benchflow.models import RolloutResult
from benchflow.review.evidence import (
    EvidenceError,
    EvidenceManifest,
    install_uploaded_workspace,
    prepare_workspace_upload,
)
from benchflow.review.evidence_runtime import ensure_evidence_python
from benchflow.review.persistence import scoring_lock, write_json_atomic
from benchflow.rollout._recovery_submission import (
    admitted_submission,
    restore_submission,
)
from benchflow.sandbox._recovery_baseline import (
    DockerRecoveryBaseline,
    config_digest,
    release_baseline,
    validate_task_identity,
)
from benchflow.sandbox._recovery_submission import CONTRACT_KIND
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.egress_denylist import (
    agent_network_sandbox_config,
    egress_denylist_for,
)
from benchflow.sandbox.recovery_network import prepare_recovery_network
from benchflow.task import Task

logger = logging.getLogger(__name__)

PRESERVED_SOLVER = "[solver-preserved]"


def recovery_ineligible_reason(
    rollout: Any, *, require_baseline: bool = True
) -> str | None:
    cfg = rollout._config
    if getattr(rollout, "_branch_child_active", False):
        return (
            "branch children do not own the parent solver completion/recovery lifecycle"
        )
    task = getattr(rollout, "_task", None)
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    submission_files = getattr(verifier, "submission_files", None)
    if task is None or (
        getattr(verifier, "workspace_recovery", False) is not True
        and not submission_files
    ):
        return "task must declare a validated workspace_recovery=true or submission_files contract"
    if submission_files and (cfg.environment != "docker" or not cfg.sandbox_user):
        return "explicit submission recovery requires owned Docker with a sandbox user"
    persisted_reason = getattr(rollout, "_recovery_ineligible_reason", None)
    if persisted_reason is not None:
        return persisted_reason
    if cfg.planes is not None:
        return "custom rollout planes cannot be reconstructed for verifier resume"
    if cfg.context_root is not None or cfg.base_image_override is not None:
        return "external build context/base overrides have no pinned recovery baseline"
    denylist = (
        agent_network_sandbox_config(task.config).network_mode
        in ("denylist", "allowlist")
        or getattr(rollout, "_egress_denylist", None) is not None
    )
    if denylist and (
        cfg.environment != "docker"
        or (
            require_baseline and not getattr(rollout, "_docker_recovery_baseline", None)
        )
    ):
        return "denylist policy bootstrap requires a captured original Docker runtime policy"
    image = task.config.sandbox.docker_image or ""
    immutable_image = isinstance(image, str) and re.fullmatch(
        r"[^\s]+@sha256:[0-9a-f]{64}", image
    )
    if not immutable_image:
        if cfg.environment != "docker":
            return "built-image recovery requires an original local Docker image lease"
        if require_baseline and not getattr(rollout, "_docker_recovery_baseline", None):
            return "original pre-agent Docker image lease was not captured"
    if task.config.sandbox.env:
        return "runtime environment values have no captured recovery identity"
    if getattr(cfg, "skills_dir", None) is not None:
        return "external skills have no immutable recovery input contract"
    if (task.paths.environment_dir / "benchflow-pre-compose.sh").exists():
        return "pre-compose hooks have no immutable recovery runtime contract"
    if task.config.sandbox.setup_commands:
        return "mutable setup commands have no pinned recovery baseline"
    if cfg.task_path.is_symlink() or any(
        path.is_symlink() for path in cfg.task_path.rglob("*")
    ):
        return "symlinked task inputs are outside the task digest recovery contract"
    if cfg.purpose != "task" or cfg.skip_verify:
        return "this is not a scored task rollout"
    if cfg.environment not in {"docker", "daytona"} or getattr(
        rollout, "_env_externally_owned", False
    ):
        return "recovery requires an owned Docker or Daytona sandbox, not an external runtime or physical system"
    if cfg.environment_manifest is not None or cfg.services:
        return "mutable environment/services require a full environment snapshot"
    if cfg.pre_agent_hooks or cfg.uploads:
        return "custom hooks/uploads have no reproducible recovery contract"
    if task.config.artifacts and not submission_files:
        return "external artifact restoration is not supported"
    if task.config.sandbox.mcp_servers:
        return "MCP state is not captured in workspace evidence"
    if any(task.paths.environment_dir.glob("*compose*.y*ml")):
        return "compose service state is not captured in workspace evidence"
    if task.config.verifier.type != "test-script":
        return "only deterministic script verifiers support workspace recovery"
    return None


def recorded_recovery_ineligible_reason(root: Path) -> str | None:
    """The ``recovery_ineligible_reason`` verdict saved in a trial's config.json."""
    path = root / "config.json"
    config = json.loads(path.read_text()) if path.is_file() else {}
    recorded = config.get("verifier_recovery") or {}
    if recorded.get("eligible") is True:
        return None
    return recorded.get("reason") or "no eligible recovery contract was recorded"


async def capture_original_docker_baseline(rollout: Any) -> None:
    """Retain the resolved image before any task uploads or agent installation."""
    cfg = rollout._config
    if getattr(rollout, "_docker_recovery_baseline", None) is not None:
        return
    if cfg.environment != "docker" or recovery_ineligible_reason(
        rollout, require_baseline=False
    ):
        return
    if not isinstance(rollout._env, DockerSandbox):
        return
    root = rollout._require_rollout_dir()
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    try:
        baseline = await rollout._env.capture_recovery_baseline(
            cfg.task_digest, config_digest(rollout._task.config)
        )
        # Record ownership first so teardown releases the lease on any failure.
        rollout._docker_recovery_baseline = baseline
        rollout._recovery_lease_owned = True
        write_json_atomic(root / "docker-recovery-baseline.json", baseline.model_dump())
        reason = recovery_ineligible_reason(rollout)
    except Exception as exc:
        reason = f"pre-agent Docker baseline unavailable: {type(exc).__name__}"
        rollout._recovery_ineligible_reason = reason
    config["verifier_recovery"] = {"eligible": reason is None, "reason": reason}
    write_json_atomic(config_path, config)


async def release_recovery_lease(rollout: Any) -> None:
    """Release the rollout's image lease; repeated calls are no-ops."""
    baseline = getattr(rollout, "_docker_recovery_baseline", None)
    if baseline is None or getattr(rollout, "_recovery_lease_released", False):
        return
    if await release_baseline(baseline):
        rollout._recovery_lease_released = True


async def release_lease_at_teardown(rollout: Any) -> None:
    """Release an owned lease at teardown unless recovery may still use it.

    A recovery child only borrows its parent's lease. When verifier-only
    recovery may follow, the attempt releases the lease once it finishes.
    """
    if not getattr(rollout, "_recovery_lease_owned", False):
        return
    if (
        needs_verifier_recovery(getattr(rollout, "_verifier_error", None))
        and recovery_ineligible_reason(rollout) is None
    ):
        return
    await release_recovery_lease(rollout)


def mark_solver_complete(rollout: Any) -> None:
    """Publish completion before capture, hardening or teardown can wedge.

    This is a stage checkpoint, not finalized telemetry or a terminal result.
    Evaluation resume recognizes it and cannot spend on a second solver.
    Tasks without an eligible recovery contract keep main's behaviour: later
    failures stay on their original, retryable channels.
    """
    if (
        getattr(rollout, "_solver_execution_complete", False)
        or getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_error", None)
        or getattr(rollout, "_task", None) is None
        or getattr(rollout, "_rollout_dir", None) is None
        or recovery_ineligible_reason(rollout) is not None
    ):
        return
    rollout._solver_execution_complete = True
    saved_error, saved_rewards = rollout._verifier_error, rollout._rewards
    rollout._verifier_error = (
        f"{PRESERVED_SOLVER} verifier pending after completed solver"
    )
    rollout._rewards = None
    try:
        rollout._solver_completion_result = rollout._build_result(
            result_filename="solver-complete.json"
        )
        root = rollout._require_rollout_dir()
        source = json.loads((root / "solver-complete.json").read_text())
        source["execution_stage"] = "solver_complete"
        source["telemetry_finalized"] = False
        write_json_atomic(root / "solver-complete.json", source)
    finally:
        rollout._verifier_error, rollout._rewards = saved_error, saved_rewards


def interrupted_solver_result(rollout: Any, detail: str) -> Any:
    from benchflow.rollout._review import read_admitted_result

    with scoring_lock(rollout._require_rollout_dir()):
        admitted = read_admitted_result(rollout)
        if admitted is not None:
            return admitted
        return _interrupted_solver_result_locked(rollout, detail)


def _interrupted_solver_result_locked(rollout: Any, detail: str) -> Any:
    """Keep post-solver deadline failure on the scoring channel, never replay."""
    message = f"{PRESERVED_SOLVER} post-solver finalization interrupted: {detail}"
    result = getattr(rollout, "_solver_completion_result", None)
    if result is None:
        result = RolloutResult(
            task_name=rollout._config.task_path.name,
            rollout_name=rollout._rollout_name or "",
        )
    result.error = None
    result.error_category = None
    result.rewards = None
    result.verifier_error = message
    result.verifier_error_category = VERIFIER_INFRA
    result.export_error = "Post-solver telemetry/evidence finalization incomplete"
    root = rollout._require_rollout_dir()
    checkpoint = root / "solver-complete.json"
    if checkpoint.is_file():
        source = json.loads(checkpoint.read_text())
        source.update(
            error=None,
            error_category=None,
            rewards=None,
            verifier_error=message,
            verifier_error_category=VERIFIER_INFRA,
            export_error=result.export_error,
        )
        if not (root / "solver.json").exists():
            write_json_atomic(root / "solver.json", source)
        # A reviewed trial's result.json is written only when its scoring
        # commits (see prepare_terminal_result).
        if getattr(rollout, "_review_plan", None) is None:
            write_json_atomic(root / "result.json", source)
    return result


def needs_verifier_recovery(error: str | None) -> bool:
    return bool(error) and (
        error.startswith("verifier_wedge:")
        or classify_verifier_error(error) == VERIFIER_INFRA
    )


@dataclass
class _RecoveryAdmission:
    active: bool = True


async def recover_verifier(rollout: Any) -> tuple[dict | None, str | None]:
    """Bound recovery separately from the already-finished solver lifecycle."""
    from benchflow.rollout._deadline import _cancel_and_abandon

    task_config = rollout._task.config
    budget = (
        task_config.sandbox.build_timeout_sec
        + task_config.verifier.timeout_sec
        + sum(command.timeout_sec for command in task_config.sandbox.setup_commands)
        + rollout._config.sandbox_setup_timeout
        + 1800
    )
    admission = _RecoveryAdmission()
    attempt = asyncio.create_task(_recover_verifier(rollout, admission))
    try:
        done, _ = await asyncio.wait({attempt}, timeout=budget)
    except asyncio.CancelledError:
        admission.active = False
        await _cancel_and_abandon(attempt)
        raise
    if done:
        return attempt.result()
    admission.active = False
    await _cancel_and_abandon(attempt)
    return (
        None,
        f"{PRESERVED_SOLVER} verifier recovery exceeded {budget}s; use bench eval score to resume",
    )


async def _recover_verifier(
    rollout: Any, admission: _RecoveryAdmission
) -> tuple[dict | None, str | None]:
    """Try one fresh-sandbox verifier execution, never starting an agent.

    The original solver/evidence and each failed verifier attempt are retained.
    A durable receipt makes failures explicit even if the process is interrupted.
    """
    from benchflow.rollout import Rollout
    from benchflow.rollout._setup import (
        _publish_trajectory_for_verifier,
        _verify_rollout,
    )

    root = rollout._require_rollout_dir()
    reason = recovery_ineligible_reason(rollout)
    if reason:
        # No contract, no attempt: leave neither a receipt nor a pointer.
        logger.info("Verifier recovery unavailable for %s: %s", root, reason)
        return None, f"{PRESERVED_SOLVER} verifier recovery unavailable: {reason}"
    attempt = root / "verifier-recovery" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    submission_files = rollout._task.config.verifier.submission_files
    record: dict[str, Any] = {
        "attempt": str(attempt.relative_to(root)),
        "status": "pending",
        "original_error": rollout._verifier_error,
        "task_digest": rollout._config.task_digest,
        "evidence": CONTRACT_KIND if submission_files else "workspace-only",
        "solver_replayed": False,
    }
    receipt = attempt / "recovery.json"
    write_json_atomic(receipt, record)
    child = None
    try:
        if (
            not rollout._config.task_digest
            or task_digest(rollout._config.task_path) != rollout._config.task_digest
        ):
            raise EvidenceError("task changed since solver setup")
        workspace = None
        if submission_files:
            await asyncio.to_thread(admitted_submission, rollout)
        else:
            bundle = root / "evidence"
            manifest = EvidenceManifest.model_validate_json(
                (bundle / "manifest.json").read_text()
            )
            workspace = PurePosixPath(manifest.workspace)
            if (
                not workspace.is_absolute()
                or len(workspace.parts) < 2
                or str(workspace)
                in {
                    "/",
                    "/home",
                    "/root",
                    "/etc",
                    "/usr",
                    "/var",
                    "/tmp",
                    "/opt",
                    "/logs",
                }
            ):
                raise EvidenceError(
                    "workspace is not a dedicated recoverable task directory"
                )
            if manifest.exclusions:
                raise EvidenceError(
                    "workspace contains excluded files; a complete restore cannot be established"
                )
            archive = attempt / "workspace.tar"
            await asyncio.to_thread(prepare_workspace_upload, bundle, archive)
        cfg = rollout._config
        task_snapshot = attempt / "task-input"
        await asyncio.to_thread(shutil.copytree, cfg.task_path, task_snapshot)
        if task_digest(task_snapshot) != cfg.task_digest:
            raise EvidenceError("task inputs changed while staging verifier recovery")
        child = Rollout(
            replace(
                cfg,
                task_path=task_snapshot,
                # Retain task/runtime/network policy; replace only solver
                # identity, interactive execution and output location.
                scenes=[Scene.single(agent="oracle")],
                agent="oracle",
                model=None,
                agent_env={},
                reasoning_effort=None,
                user=None,
                loop_strategy=None,
                allow_document_user=False,
                skip_verify=True,
                jobs_dir=attempt,
                job_name="runtime",
                rollout_name=f"verifier-{attempt.name}",
                planes=rollout._planes,
                export_generated_skills_to=None,
            )
        )
        await child.setup()
        # Docker binds /logs to the child's rollout directory. The verifier
        # must read from that same location; attempt-level paths are receipts,
        # not the running sandbox's mount source.
        paths = child._rollout_paths
        baseline = getattr(rollout, "_docker_recovery_baseline", None)
        if baseline is not None:
            validate_task_identity(baseline, cfg.task_digest, child._task.config)
            if not isinstance(child._env, DockerSandbox):
                raise EvidenceError(
                    "Original Docker image lease requires the Docker backend"
                )
            child._env.use_recovery_baseline(baseline)
            child._docker_recovery_baseline = baseline
        await child.start()
        await ensure_evidence_python(
            child._env, timeout_sec=cfg.sandbox_setup_timeout, allow_install=False
        )
        if submission_files:
            await child.install_agent()
            await restore_submission(rollout, child._env)
            workspace = child._agent_cwd
        else:
            remote = "/tmp/benchflow-verifier-recovery-" + uuid.uuid4().hex
            await child._env.upload_file(archive, remote + ".tar")
            await child._env.upload_file(bundle / "manifest.json", remote + ".json")
            await install_uploaded_workspace(
                child._env,
                remote + ".tar",
                remote,
                remote + ".json",
                restore_modes=True,
            )
            # Oracle installation performs trusted user/lockdown/build-baseline
            # setup only; no solver command or model is executed.
            await child.install_agent()
            if child._agent_cwd != str(workspace):
                raise EvidenceError(
                    "fresh sandbox workspace differs from captured workspace"
                )
            # Replace only the dedicated workspace in the newly created sandbox.
            # Admission above validates every restored byte before it is installed.
            script = (
                "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); "
                "assert not p.is_symlink(); "
                "shutil.rmtree(p) if p.exists() else None; "
                "shutil.move(sys.argv[2],p)"
            )
            moved = await child._env.exec(
                shlex.join(["python3", "-c", script, str(workspace), remote]),
                user="root",
                timeout_sec=60,
            )
            if moved.return_code:
                raise EvidenceError("could not install workspace in recovery sandbox")
            child._agent_cwd = str(workspace)
            if cfg.sandbox_user:
                await rollout._planes.setup_sandbox_user(
                    child._env,
                    cfg.sandbox_user,
                    workspace=str(workspace),
                    timeout_sec=cfg.sandbox_setup_timeout,
                )
        recovery_sandbox = agent_network_sandbox_config(child._task.config)
        child._egress_denylist = egress_denylist_for(recovery_sandbox)
        await prepare_recovery_network(child._env, recovery_sandbox, cfg.sandbox_user)
        await _publish_trajectory_for_verifier(
            child._env, rollout._trajectory, attempt / "agent"
        )
        rewards, error, _ = await _verify_rollout(
            child._env,
            child._task,
            paths,
            record.setdefault("timing", {}),
            rollout._planes,
            sandbox_user=cfg.sandbox_user,
            workspace=str(workspace),
            recovery_eligible=True,
        )
        if paths.verifier_dir.is_dir():
            shutil.copytree(paths.verifier_dir, attempt / "verifier")
        record.update(
            status="complete" if error is None else "failed",
            rewards=rewards,
            error=error,
        )
        if error:
            return None, f"{PRESERVED_SOLVER} verifier recovery failed: {error}"
        return rewards, None
    except asyncio.CancelledError:
        record.update(status="interrupted")
        raise
    except Exception as exc:
        record.update(status="unavailable", error=str(exc))
        return None, f"{PRESERVED_SOLVER} verifier recovery unavailable: {exc}"
    finally:
        # Attempt-local receipts remain writable after revocation. Canonical
        # files/pointer are a single non-awaiting publication under admission.
        try:
            write_json_atomic(receipt, record)
            if child is not None:
                cleanup = asyncio.create_task(child.cleanup())
                done, _ = await asyncio.wait({cleanup}, timeout=30)
                if done:
                    if not cleanup.cancelled():
                        error = cleanup.exception()
                        if error is not None:
                            record["cleanup_error"] = str(error)
                else:
                    cleanup.cancel()
                    from benchflow.rollout._deadline import _swallow_abandoned_outcome

                    cleanup.add_done_callback(_swallow_abandoned_outcome)
                    record["cleanup_error"] = "cleanup exceeded 30s; abandoned"
        finally:
            if not admission.active:
                record.update(status="interrupted", admission="revoked")
            write_json_atomic(receipt, record)
            if admission.active and record["status"] != "interrupted":
                if record["status"] == "complete" and (attempt / "verifier").is_dir():
                    try:
                        _publish_verifier_dir(root, attempt)
                    except Exception as exc:
                        # The verdict lives in the receipt; keep it admitted
                        # while the canonical folder keeps its original files.
                        record["publication_error"] = describe_exception(exc)
                        write_json_atomic(receipt, record)
                write_json_atomic(
                    root / "verification.json",
                    {"attempt": str(attempt.relative_to(root))},
                )
                # The attempt finished; an interrupted one keeps the lease
                # for `bench eval score` resume.
                await release_recovery_lease(rollout)


def _publish_verifier_dir(root: Path, attempt: Path) -> None:
    """Swap recovered outputs into ``verifier/``; on failure keep the original.

    Stage a full copy beside the canonical folder, then swap with two renames
    on the same filesystem, so a failed copy never leaves a partial folder.
    """
    canonical = root / "verifier"
    staged = root / f".verifier-{attempt.name}.staging"
    previous = attempt / "previous-verifier"
    shutil.rmtree(staged, ignore_errors=True)
    try:
        shutil.copytree(attempt / "verifier", staged)
        if canonical.is_dir():
            canonical.rename(previous)
        try:
            staged.rename(canonical)
        except BaseException:
            if previous.is_dir() and not canonical.exists():
                previous.rename(canonical)
            raise
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def verification_source(root: Path) -> dict:
    """Read immutable solver evidence plus its latest admitted verifier revision."""
    source = json.loads((root / "solver.json").read_text())
    pointer = root / "verification.json"
    if not pointer.is_file():
        return source
    relative = json.loads(pointer.read_text())["attempt"]
    path = Path(relative)
    if (
        len(path.parts) != 2
        or path.parts[0] != "verifier-recovery"
        or not path.parts[1].isalnum()
    ):
        raise ValueError("Invalid verifier recovery reference")
    record = json.loads((root / path / "recovery.json").read_text())
    if record.get("task_digest") != source.get("task_digest"):
        raise ValueError("Verifier recovery task digest differs from solver")
    source["verification"] = {"attempt": relative, "status": record["status"]}
    if record["status"] == "complete":
        source["rewards"] = record["rewards"]
        source["verifier_error"] = None
        source["verifier_error_category"] = None
    else:
        source["rewards"] = None
        source["verifier_error"] = (
            f"{PRESERVED_SOLVER} verifier recovery {record['status']}: {record.get('error', 'no verdict')}"
        )
        source["verifier_error_category"] = VERIFIER_INFRA
    source["timing"] = dict(source.get("timing") or {})
    source["timing"]["verifier_recovery"] = (record.get("timing") or {}).get(
        "verifier", 0
    )
    return source


async def resume_verification(root: Path, task_path: Path) -> dict:
    """Execute only deterministic verification using a trusted exact task.

    Called under the existing scoring lock. Credentials are re-resolved from
    task verifier env in the current process; none are read from artifacts.
    """
    from benchflow.rollout import RolloutConfig

    source = verification_source(root)
    if source.get("rewards") is not None and not source.get("verifier_error"):
        return source
    original = json.loads((root / "solver.json").read_text())
    if not needs_verifier_recovery(original.get("verifier_error")):
        return source
    reason = recorded_recovery_ineligible_reason(root)
    if reason is not None:
        # Never gains a contract on resume; the original error stays retryable.
        logger.info("Verifier recovery unavailable for %s: %s", root, reason)
        return source
    config = json.loads((root / "config.json").read_text())
    if task_digest(task_path) != original.get("task_digest"):
        raise ValueError("Task digest mismatch before verifier recovery")
    cfg = RolloutConfig(
        task_path=task_path,
        agent="oracle",
        task_digest=original["task_digest"],
        environment=config.get("environment", "docker"),
        sandbox_user=config.get("sandbox_user", "agent"),
        sandbox_locked_paths=config.get("sandbox_locked_paths"),
        sandbox_setup_timeout=config.get("sandbox_setup_timeout", 120),
        context_root=config.get("context_root"),
        base_image_override=config.get("base_image_override"),
        config_override=(config.get("config_override") or {}).get("patch"),
    )
    task = Task(task_path)
    task.config = apply_config_override(task.config, cfg.config_override)
    trial = SimpleNamespace(
        _config=cfg,
        _task=task,
        _planes=default_rollout_planes(),
        _require_rollout_dir=lambda: root,
        _trajectory=[
            json.loads(line)
            for line in (root / "trajectory/acp_trajectory.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ],
        _verifier_error=original.get("verifier_error"),
    )
    baseline_path = root / "docker-recovery-baseline.json"
    if baseline_path.exists():
        trial._docker_recovery_baseline = DockerRecoveryBaseline.model_validate_json(
            baseline_path.read_text()
        )
    await recover_verifier(trial)
    return verification_source(root)
