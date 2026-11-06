"""Regrade stored trials with a changed verifier.

``bench eval regrade <job|trial> [--tasks-dir DIR]`` (Python: ``bf.regrade``)
re-runs the task's *current* verifier against each trial's frozen final
workspace, in a fresh sandbox, and writes the new score next to the original.
Nothing the original run wrote is changed: ``result.json`` keeps its rewards.
A trial the separate verifier scored (or whose task asks for one) is scored
by that verifier's own path: a sandbox from the task's verifier image gets
the transfer the run packed, from the trial folder, and no sandbox is built
from the agent's environment.

Inputs, per trial:

- ``evidence/`` — the agent's final workspace (and declared artifacts that
  lived outside it), frozen before verifier hardening. Rubric review and
  verifier recovery write it; ``bench eval run --freeze-workspace`` writes it
  for any run. A trial without it is *not regradable*: the verifier would
  otherwise see a reconstructed or empty workspace, so it is never guessed.
- ``artifacts/`` + ``artifacts-manifest.json`` — files collected from
  ``/logs/artifacts``; restored there (hash-checked) when present.
- ``trajectory/acp_trajectory.jsonl`` — republished for trajectory-reading
  verifiers.

Outputs, per trial: ``regrade/<id>/`` (the new verifier's outputs, a copy of
the verifier files, ``regrade.json``, and a separate verifier's
``verifier-sandbox.json``) and ``regrade.json`` beside
``result.json`` (the original score plus every regrade block: verifier
digest, original and new reward, reason). The job gets ``regrade-summary.json``
listing changed verdicts and every not-regradable trial with its reason.

With ``review`` the trial's rubric review runs again too, on the same saved
outputs and gated by the new test, and the combined score is compared
(:func:`_review_again`; its files are under ``regrade/<id>/review/``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shlex
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from benchflow._root_python import root_python_command
from benchflow._utils.reachable_write import ReachablePath, pin_destination
from benchflow.sandbox._exec_output import exec_raw
from benchflow.trust.state import IngressKind, admit

if TYPE_CHECKING:
    from benchflow.review.automatic import PreparedReview
    from benchflow.review.options import ReviewerConfig
    from benchflow.review.outcome import ScoringResult

logger = logging.getLogger(__name__)

REGRADE_FILE = "regrade.json"
SUMMARY_FILE = "regrade-summary.json"
_UNSAFE_WORKSPACES = {
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

Status = Literal["regraded", "failed", "not_regradable"]

UNCHANGED_TASK_FLIP = (
    "the task is unchanged, so the verifier read state outside the frozen "
    "workspace (packages installed system-wide, services, files elsewhere) "
    "that a regrade does not restore, or it is nondeterministic"
)


@dataclass
class TrialRegrade:
    """One trial's regrade outcome (a row of the summary)."""

    trial: str
    task: str
    status: Status
    original_reward: float | None = None
    new_reward: float | None = None
    change: str | None = None
    reason: str | None = None
    regrade_id: str | None = None
    task_changed: bool | None = None
    # "verifier_reward" for a trial with a rubric review: a regrade reruns only
    # the test, so it compares the test's own reward, not the reviewed one.
    compared: str = "reward"

    @property
    def changed(self) -> bool:
        return self.status == "regraded" and self.change not in (None, "same")


@dataclass
class RegradeSummary:
    """What ``bf.regrade`` did to a job or trial."""

    path: str
    regrade_id: str
    tasks_dir: str | None
    reason: str | None
    trials: list[TrialRegrade] = field(default_factory=list)

    @property
    def regraded(self) -> list[TrialRegrade]:
        """Trials whose stored workspace was scored again."""
        return [t for t in self.trials if t.status == "regraded"]

    @property
    def changed(self) -> list[TrialRegrade]:
        """Regraded trials whose verdict changed."""
        return [t for t in self.trials if t.changed]

    @property
    def not_regradable(self) -> list[TrialRegrade]:
        """Trials that kept no frozen workspace to score again."""
        return [t for t in self.trials if t.status == "not_regradable"]

    @property
    def failed(self) -> list[TrialRegrade]:
        """Trials whose regrade failed."""
        return [t for t in self.trials if t.status == "failed"]

    def counts(self) -> dict[str, int]:
        """How many trials were regraded, changed (and which way), failed or not regradable."""
        changes = [t.change for t in self.changed]
        return {
            "trials": len(self.trials),
            "regraded": len(self.regraded),
            "changed": len(changes),
            "fail_to_pass": changes.count("fail->pass"),
            "pass_to_fail": changes.count("pass->fail"),
            "failed": len(self.failed),
            "not_regradable": len(self.not_regradable),
        }

    def to_dict(self) -> dict[str, Any]:
        """The summary as the JSON document ``bench eval regrade --json`` prints."""
        return {
            "path": self.path,
            "regrade_id": self.regrade_id,
            "tasks_dir": self.tasks_dir,
            "reason": self.reason,
            "counts": self.counts(),
            "changed": [asdict(t) for t in self.changed],
            "trials": [asdict(t) for t in self.trials],
        }


# --- discovery -------------------------------------------------------------


def _is_trial(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "result.json").is_file() or (path / "solver.json").is_file()
    )


def find_trials(path: Path) -> list[Path]:
    """``path`` itself when it is a trial, else its trial subfolders."""
    path = Path(path)
    if _is_trial(path):
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Not a job or trial folder: {path}")
    trials = sorted(p for p in path.iterdir() if p.is_dir() and _is_trial(p))
    if not trials:
        raise FileNotFoundError(f"No trials under {path}")
    return _without_replaced_attempts(trials)


def _result_or_empty(trial: Path) -> dict[str, Any]:
    try:
        result = _read_json(trial / "result.json")
    except (OSError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def _attempt_key(trial: Path) -> tuple[Any, Any, Any]:
    result = _result_or_empty(trial)
    return (
        result.get("task_name") or trial.name,
        result.get("agent_name") or result.get("agent"),
        result.get("model"),
    )


def _without_replaced_attempts(trials: list[Path]) -> list[Path]:
    """Drop unscored attempts a scored retry of the same task replaced.

    A trial whose sandbox failed to start (or that errored before scoring)
    and was then retried leaves its own folder; the job's summary, ``inspect``
    and ``load_job`` count the retry, so regrade must not list the replaced
    attempt as another trial.
    """
    scored = {
        _attempt_key(t)
        for t in trials
        if isinstance(_result_or_empty(t).get("rewards"), dict)
    }
    return [
        t
        for t in trials
        if isinstance(_result_or_empty(t).get("rewards"), dict)
        or (t / "evidence").is_dir()
        or _attempt_key(t) not in scored
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _task_dirname(trial: Path, config: dict, result: dict) -> str:
    recorded = config.get("task_path") or result.get("task_name") or trial.name
    return PurePosixPath(str(recorded)).name


def _is_task_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).is_file() for name in ("task.toml", "task.md")
    )


def resolve_task(trial: Path, tasks_dir: Path | None) -> tuple[Path | None, str | None]:
    """The task folder whose current verifier should grade ``trial``."""
    config = _read_json(trial / "config.json")
    result = _read_json(trial / "result.json")
    name = _task_dirname(trial, config, result)
    if tasks_dir is not None:
        tasks_dir = Path(tasks_dir)
        if _is_task_dir(tasks_dir) and tasks_dir.name == name:
            return tasks_dir, None
        candidate = tasks_dir / name
        if _is_task_dir(candidate):
            return candidate, None
        return None, f"task {name!r} not found in {tasks_dir}"
    recorded = Path(str(config.get("task_path") or ""))
    if recorded.is_absolute() and _is_task_dir(recorded):
        return recorded, None
    # A batch trial records only the task folder's name; the job records the
    # tasks folder it ran (evaluation.json beside the trials).
    job_tasks = _read_json(trial.parent / "evaluation.json").get("tasks_dir")
    if isinstance(job_tasks, str) and job_tasks:
        job_tasks_dir = Path(job_tasks)
        if _is_task_dir(job_tasks_dir) and job_tasks_dir.name == name:
            return job_tasks_dir, None
        if _is_task_dir(job_tasks_dir / name):
            return job_tasks_dir / name, None
    return None, f"task folder for {name!r} unknown; pass --tasks-dir"


def _restorable_task(trial: Path, task_dir: Path) -> tuple[Path, str | None]:
    """The task's native package, and why its world cannot be restored for a
    regrade, if it cannot.

    A regrade restores the frozen workspace into a fresh sandbox; an embodied
    task's world (the simulator or the robot) is not in that workspace, so the
    verifier would judge an episode the agent never touched. A folder in a task
    format is materialized first (evaluation.json records the source folder).
    """
    from benchflow.embodied.spec import RestoreRefused
    from benchflow.embodied.trials import (
        task_dir_restore_boundary,
        trial_restore_boundary,
    )

    try:
        native, declared = task_dir_restore_boundary(task_dir)
        trial_restore_boundary(trial, declared).require_world_restore("regrade")
    except (RestoreRefused, ValueError, RuntimeError) as exc:
        return task_dir, str(exc)
    return native, None


def original_score(trial: Path) -> dict[str, Any]:
    """The score the run itself recorded (never a previous regrade)."""
    result = _read_json(trial / "result.json") or _read_json(trial / "solver.json")
    rewards = result.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    # A rubric-reviewed trial's reward is the review's, gated by the test
    # (result.json's scoring block); its test's own reward is verifier_reward.
    verifier_reward = (
        rewards.get("verifier_reward")
        if isinstance(rewards, dict) and isinstance(result.get("scoring"), dict)
        else None
    )
    return {
        "rewards": rewards,
        "reward": reward,
        "verifier_reward": verifier_reward,
        "verifier_error": result.get("verifier_error"),
        "task_digest": result.get("task_digest")
        or _read_json(trial / "config.json").get("task_digest"),
    }


def ran_separate_verifier(trial: Path) -> bool:
    """Whether the run scored ``trial`` in the separate verifier sandbox.

    It wrote the separate verifier's record, or put every verifier there
    (``--verifier-sandbox separate``, strict integrity). The regrade then
    runs there too, whatever the task now declares: like a run, a regrade
    never moves a verifier out of the separate verifier sandbox. A task that
    asks for one is regraded there as well (the runner reads its config).
    """
    from benchflow.rollout._separate_verifier import RECORD_DIR, RECORD_FILE

    config = _read_json(trial / "config.json")
    return (
        (trial / RECORD_DIR / RECORD_FILE).is_file()
        or config.get("verifier_sandbox") == "separate"
        or config.get("integrity") == "strict"
    )


def regradable_reason(trial: Path) -> str | None:
    """Why ``trial`` cannot be regraded, or ``None`` when it can."""
    from benchflow.review.evidence import (
        EvidenceError,
        EvidenceManifest,
        validate_workspace,
    )

    config = _read_json(trial / "config.json")
    if config.get("purpose", "task") != "task":
        return "reviewer rollouts are not scored by a task verifier"
    if (trial / "steps.json").is_file():
        return (
            "a multi-step trial's reward comes from each step's tests "
            "(steps.json), and a regrade runs one verifier on the final workspace"
        )
    bundle = trial / "evidence"
    if not (bundle / "manifest.json").is_file():
        return (
            "no frozen workspace (evidence/); run with --freeze-workspace or a "
            "rubric reviewer to make trials regradable"
        )
    try:
        manifest = EvidenceManifest.model_validate_json(
            (bundle / "manifest.json").read_text()
        )
        validate_workspace(bundle / "workspace", manifest)
    except (EvidenceError, OSError, ValueError) as exc:
        return f"frozen workspace does not match its manifest: {exc}"
    workspace = PurePosixPath(manifest.workspace)
    if not workspace.is_absolute() or str(workspace) in _UNSAFE_WORKSPACES:
        return f"frozen workspace {workspace} is not a dedicated task directory"
    return None


# --- digests and verdicts --------------------------------------------------


def verifier_digest(task_dir: Path) -> str:
    """sha256 over the verifier files and the task's [verifier] settings."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.task import Task

    task = Task(task_dir)
    digest = hashlib.sha256()
    tests = task.paths.tests_dir
    digest.update((task_digest(tests) if tests.is_dir() else "none").encode())
    digest.update(
        json.dumps(
            task.config.verifier.model_dump(mode="json"), sort_keys=True
        ).encode()
    )
    return f"sha256:{digest.hexdigest()}"


def _passed(reward: float | None) -> bool | None:
    return None if reward is None else reward >= 1.0


def verdict_change(original: float | None, new: float | None) -> str:
    """``same``, ``fail->pass``, ``pass->fail``, ``reward <a> -> <b>``, or
    ``scored``/``unscored`` when only one side has a reward."""
    if original is None and new is None:
        return "same"
    if original is None:
        return "scored"
    if new is None:
        return "unscored"
    before, after = _passed(original), _passed(new)
    if before != after:
        return "fail->pass" if after else "pass->fail"
    if abs(original - new) > 1e-9:
        return f"reward {original:g} -> {new:g}"
    return "same"


# --- persistence -----------------------------------------------------------


def _write_block(trial: Path, attempt: Path, block: dict[str, Any]) -> None:
    from benchflow.review.persistence import write_json_atomic

    write_json_atomic(attempt / REGRADE_FILE, block)
    record = _read_json(trial / REGRADE_FILE)
    regrades: list[dict[str, Any]] = list(record.get("regrades") or [])
    regrades.append(block)
    record = {
        "original": record.get("original") or original_score(trial),
        "regrades": regrades,
    }
    record["latest"] = block["id"]
    write_json_atomic(trial / REGRADE_FILE, record)


# --- sandbox run -----------------------------------------------------------

Runner = Callable[..., Awaitable[tuple[dict | None, str | None]]]


async def _restore_logs_artifacts(env: Any, trial: Path, staging: Path) -> int:
    """Upload hash-checked /logs/artifacts files; return how many."""
    from benchflow.review.evidence import (
        EvidenceEntry,
        EvidenceError,
        EvidenceManifest,
        install_uploaded_workspace,
        prepare_workspace_upload,
    )
    from benchflow.rollout._artifacts import MANIFEST_NAME

    manifest = _read_json(trial / MANIFEST_NAME)
    files = [
        f
        for f in manifest.get("files", [])
        if f.get("collection") == 0 and f.get("kind") == "file"
    ]
    if not files:
        return 0
    bundle = staging / "logs-artifacts"
    tree = bundle / "workspace"
    entries: dict[str, EvidenceEntry] = {}
    for item in files:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError(f"unsafe artifact path {item['path']!r}")
        source = ReachablePath.rollout(trial, "artifacts") / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise EvidenceError(f"artifact {relative} changed since collection")
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source.copy_to(target)
        for parent in list(relative.parents)[:-1]:
            entries.setdefault(
                str(parent),
                EvidenceEntry(
                    path=str(parent),
                    original_path=f"/logs/artifacts/{parent}",
                    kind="directory",
                    size=0,
                ),
            )
        entries[str(relative)] = EvidenceEntry(
            path=str(relative),
            original_path=f"/logs/artifacts/{relative}",
            kind="file",
            size=item["size"],
            sha256=item["sha256"],
        )
    manifest_obj = EvidenceManifest(
        workspace="/logs/artifacts",
        archive_sha256="0" * 64,
        entries=tuple(sorted(entries.values(), key=lambda e: e.path)),
    )
    (bundle / "manifest.json").write_text(manifest_obj.model_dump_json())
    archive = staging / "logs-artifacts.tar"
    prepare_workspace_upload(bundle, archive)
    remote = "/tmp/runtime-regrade-logs-" + uuid.uuid4().hex
    admission = admit(env, "main", IngressKind.REGRADE_TRANSFER)
    await env.upload_file(archive, remote + ".tar", admission=admission)
    await env.upload_file(
        bundle / "manifest.json", remote + ".json", admission=admission
    )
    await install_uploaded_workspace(env, remote + ".tar", remote, remote + ".json")
    moved = await exec_raw(
        env.exec,
        f"mkdir -p /logs/artifacts && cp -a {shlex.quote(remote)}/. /logs/artifacts/",
        user="root",
        timeout_sec=120,
    )
    if moved.return_code:
        raise EvidenceError("could not restore /logs/artifacts")
    return len(files)


async def _install_bundle(env: Any, bundle: Path, staging: Path, name: str) -> str:
    """Upload one evidence bundle; return the sandbox folder holding it."""
    from benchflow.review.evidence import (
        install_uploaded_workspace,
        prepare_workspace_upload,
    )

    archive = staging / f"{name}.tar"
    await asyncio.to_thread(prepare_workspace_upload, bundle, archive)
    remote = f"/tmp/runtime-regrade-{name}-" + uuid.uuid4().hex
    admission = admit(env, "main", IngressKind.REGRADE_TRANSFER)
    await env.upload_file(archive, remote + ".tar", admission=admission)
    await env.upload_file(
        bundle / "manifest.json", remote + ".json", admission=admission
    )
    await install_uploaded_workspace(
        env, remote + ".tar", remote, remote + ".json", restore_modes=True
    )
    return remote


# Replace ``argv[1]`` with ``argv[2]``. An existing directory keeps its inode
# (its children are swapped), so shells whose cwd is the workspace stay valid.
_MOVE_SCRIPT = r"""
import pathlib, shutil, sys
dest, src = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
assert not dest.is_symlink()
if dest.is_dir() and src.is_dir():
    for child in list(dest.iterdir()):
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    for child in list(src.iterdir()):
        shutil.move(str(child), dest / child.name)
    src.rmdir()
else:
    if dest.is_dir():
        shutil.rmtree(dest)
    elif dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), dest)
"""


async def run_verifier_on_frozen_trial(
    trial: Path,
    task_dir: Path,
    attempt: Path,
    *,
    sandbox: str | None = None,
) -> tuple[dict | None, str | None]:
    """Fresh sandbox + frozen workspace + current verifier -> (rewards, error).

    Mirrors verifier-only recovery (``rollout/_verifier_recovery.py``) but
    for a *changed* task: the sandbox is built from ``task_dir``, the oracle
    install only sets up users and lockdown (no solver runs), and the frozen
    workspace replaces the fresh one byte for byte before hardening.

    A trial the separate verifier scored, or whose task asks for one, takes
    the separate verifier's own path instead (:func:`_verify_in_verifier_sandbox`):
    no sandbox is built from the task's agent environment.
    """
    from benchflow._types import Scene
    from benchflow.agents.registry import is_scripted_agent
    from benchflow.review.evidence import EvidenceError, EvidenceManifest
    from benchflow.review.evidence_runtime import ensure_evidence_python
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.rollout._setup import (
        _publish_trajectory_for_verifier,
        _verify_rollout,
    )
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    config = _read_json(trial / "config.json")
    bundle = trial / "evidence"
    manifest = EvidenceManifest.model_validate_json(
        (bundle / "manifest.json").read_text()
    )
    workspace = str(manifest.workspace)
    # TASKMD-R14 B-M2: the attempt (its verifier receipt and staging) is a
    # destination root of its own, held before any sandbox starts; a fresh
    # regrade process on another jobs filesystem inherits no other root.
    attempt_root = pin_destination(attempt)
    task_input = attempt / "task-input"
    await asyncio.to_thread(shutil.copytree, task_dir, task_input, symlinks=False)
    override = (config.get("config_override") or {}).get("patch")
    sandbox_user = config.get("sandbox_user", "agent")
    child = Rollout(
        RolloutConfig(
            task_path=task_input,
            scenes=[Scene.single(agent="oracle")],
            agent="oracle",
            environment=sandbox or config.get("environment") or "docker",
            sandbox_user=sandbox_user,
            sandbox_locked_paths=config.get("sandbox_locked_paths"),
            sandbox_setup_timeout=config.get("sandbox_setup_timeout") or 120,
            context_root=config.get("context_root"),
            base_image_override=config.get("base_image_override"),
            config_override=override,
            verifier_sandbox="separate" if ran_separate_verifier(trial) else "task",
            separate_verifier_include=tuple(config.get("verifier_include") or ()),
            skip_verify=True,
            jobs_dir=attempt,
            job_name="runtime",
            rollout_name="verifier",
        )
    )
    staging = Path(tempfile.mkdtemp(prefix="regrade-", dir=attempt))
    try:
        await child.setup(verifier_only=True)
        # The task config as the run's rollout sees it (overlay applied).
        if separate_verifier_requested(child._task.config):
            return await _verify_in_verifier_sandbox(
                child, trial, workspace, attempt_root
            )
        await child.start()
        env = child._env
        await ensure_evidence_python(
            env, timeout_sec=child._config.sandbox_setup_timeout, allow_install=True
        )
        remote = await _install_bundle(env, bundle, staging, "workspace")
        await child.install_agent()
        if child._agent_cwd != workspace:
            raise EvidenceError(
                f"the task's workspace is now {child._agent_cwd}, the frozen one "
                f"was {workspace}"
            )
        moved = await exec_raw(
            env.exec,
            await root_python_command(env, _MOVE_SCRIPT, workspace, remote),
            user="root",
            timeout_sec=120,
        )
        if moved.return_code:
            raise EvidenceError("could not install the frozen workspace")
        for artifact in manifest.artifacts:
            if artifact.bundle_path is None:
                continue
            sub = bundle / artifact.bundle_path
            sub_manifest = EvidenceManifest.model_validate_json(
                (sub / "manifest.json").read_text()
            )
            installed = await _install_bundle(
                env, sub, staging, artifact.bundle_path.replace("/", "-")
            )
            if sub_manifest.workspace != artifact.source:
                # A single file: its capture root is the file's parent.
                installed = f"{installed}/{PurePosixPath(artifact.source).name}"
            moved = await exec_raw(
                env.exec,
                await root_python_command(
                    env, _MOVE_SCRIPT, artifact.source, installed
                ),
                user="root",
                timeout_sec=120,
            )
            if moved.return_code:
                raise EvidenceError(f"could not restore artifact {artifact.source}")
        await _restore_logs_artifacts(env, trial, staging)
        if sandbox_user:
            # No agent runs here (only the verifier, as root), so no agent-UID
            # firewall and no sandbox-daemon guard: both only confine the
            # agent user, and installing iptables needs a package manager that
            # no-network sandboxes cannot reach. The provider still applies the
            # task's sandbox network mode to the verifier.
            await child._planes.setup_sandbox_user(
                env,
                sandbox_user,
                workspace=workspace,
                timeout_sec=child._config.sandbox_setup_timeout,
                guard_daemon=False,
            )
        trajectory = _stored_trajectory(trial)
        if trajectory is not None:
            await _publish_trajectory_for_verifier(
                env,
                trajectory,
                ReachablePath.rollout(attempt, "agent"),
                publish_reply=not is_scripted_agent(_trial_agent(trial)),
            )
        paths = child._rollout_paths
        timing: dict[str, Any] = {}
        rewards, error, _ = await _verify_rollout(
            env,
            child._task,
            paths,
            timing,
            child._planes,
            sandbox_user=sandbox_user,
            workspace=workspace,
        )
        if paths.verifier_dir.is_dir():
            paths.verifier_dir.copy_to(attempt_root / "verifier")
        return rewards, error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            await asyncio.wait_for(child.cleanup(), timeout=120)
        except Exception:
            logger.warning("Regrade sandbox cleanup failed", exc_info=True)
        shutil.rmtree(task_input, ignore_errors=True)


def _stored_trajectory(trial: Path) -> list[dict] | None:
    """The trial's saved ACP trajectory, for verifiers that read /logs/agent."""
    path = trial / "trajectory" / "acp_trajectory.jsonl"
    if not path.is_file():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trial_agent(trial: Path) -> str | None:
    return _read_json(trial / "result.json").get("agent") or _read_json(
        trial / "config.json"
    ).get("agent")


async def _verify_in_verifier_sandbox(
    child: Any, trial: Path, workspace: str, attempt_root: ReachablePath
) -> tuple[dict | None, str | None]:
    """Score ``trial`` by the separate verifier's own path, from its saved outputs.

    What the run's verify() does once the agent has stopped
    (``run_separate_verifier``), with no agent sandbox at all: a fresh
    sandbox from the task's verifier image (``plan_verifier_image``) gets one
    transfer packed from the trial folder (the frozen workspace, the declared
    artifacts and ``--separate-verifier-include`` paths, and
    ``/logs/artifacts``, each checked against its manifest), and the
    verifier runs there. The run's clean control and workspace come from its
    record, so the transfer is judged as the run judged it. The new record
    is copied beside the new ``verifier/``.
    """
    from benchflow.rollout._separate_verifier import (
        RECORD_DIR,
        RECORD_FILE,
        SERIAL,
        VerifierSandboxRun,
        _default_verify,
        read_record,
    )
    from benchflow.rollout._setup import _configured_task_workdir

    record = read_record(trial) or {}
    pristine = record.get("pristine")
    child._pristine_outputs = pristine
    child._outside_workspace_writes = record.get("outside_workspace_writes")
    # The workspace the task declares, else the one the run measured.
    child._agent_cwd = (
        _configured_task_workdir(child._task)
        or (pristine or {}).get("workspace")
        or workspace
    )
    # The run collected them; this rollout's own sandbox never starts.
    child._artifacts_collected = True
    run = VerifierSandboxRun(child, order=SERIAL, source_dir=trial)
    # A regrade has no agent sandbox: none runs beside the verifier.
    run.agent_stopped(None)
    rewards, error = await run.finish(
        _default_verify(
            child, _stored_trajectory(trial) or [], agent=_trial_agent(trial)
        )
    )
    paths = child._rollout_paths
    if paths.verifier_dir.is_dir():
        paths.verifier_dir.copy_to(attempt_root / "verifier")
    written = Path(child._rollout_dir) / RECORD_DIR / RECORD_FILE
    if written.is_file():
        (attempt_root / RECORD_FILE).write_bytes(written.read_bytes())
    return rewards, error


# --- the rubric review, again ------------------------------------------------

# What a trial's reviewer never saw, or a regrade replaces: the records written
# when its review committed, its reviews and regrades, its test stage
# (solver.json, verifier/, the separate verifier's folder, recovery
# revisions), and the frozen bundle, which the view links instead.
_NOT_REVIEWED = frozenset(
    {
        "evidence",
        "regrade",
        REGRADE_FILE,
        "result.json",
        "timing.json",
        "rewards.jsonl",
        "results.jsonl",
        "trainer",
        "reviews",
        "scoring",
        "solver.json",
        "verifier",
        "verifier-sandbox",
        "verification.json",
        "verifier-recovery",
        ".scoring.lock",
    }
)


def _review_plan(
    trial: Path, task_dir: Path, reviewer: ReviewerConfig | None
) -> PreparedReview:
    """The fixed task's rubric and the trial's reviewer settings
    (config.json ``review.reviewer``) with ``reviewer``'s on top.

    Raises ``ValueError`` before any sandbox when the task has no review
    rubric or the reviewer cannot run (no model, no key, no backend).
    """
    from benchflow.review.automatic import prepare_review
    from benchflow.review.resume import _recorded_setup_timeout, _reviewer_options

    plan = prepare_review(
        task_dir,
        _reviewer_options(trial, reviewer),
        sandbox_setup_timeout=_recorded_setup_timeout(trial),
    )
    if plan is None:
        raise ValueError(f"task {task_dir.name!r} has no review rubric")
    return plan


def _review_view(
    trial: Path, attempt: Path, view: Path, rewards: dict | None, task_digest: str
) -> None:
    """The trial folder as its reviewer saw it, with this regrade's test stage.

    Everything in ``_NOT_REVIEWED`` is left out; the regrade's ``verifier/``
    and separate verifier record take the run's places, and ``solver.json``
    carries the new rewards and the fixed task's digest.
    """
    from benchflow._utils.reachable_write import snapshot_tree
    from benchflow.review.persistence import write_json_atomic
    from benchflow.rollout._separate_verifier import RECORD_DIR, RECORD_FILE

    def ignore(folder: str, names: list[str]) -> set[str]:
        if Path(folder) != trial:
            return set()
        return {name for name in names if name in _NOT_REVIEWED}

    shutil.copytree(trial, view, symlinks=True, ignore=ignore)
    (view / "evidence").symlink_to(trial / "evidence", target_is_directory=True)
    if (attempt / "verifier").is_dir():
        # The regrade's verifier/ was the verifier sandbox's /logs/verifier:
        # copied through the helper, links kept as entries, never followed.
        snapshot_tree(attempt / "verifier", view / "verifier")
    if (attempt / RECORD_FILE).is_file():
        (view / RECORD_DIR).mkdir()
        shutil.copyfile(attempt / RECORD_FILE, view / RECORD_DIR / RECORD_FILE)
    solver = _read_json(trial / "solver.json") or _read_json(trial / "result.json")
    solver.pop("scoring", None)
    solver.update(rewards=rewards, verifier_error=None, task_digest=task_digest)
    if "verifier_error_category" in solver:
        solver["verifier_error_category"] = None
    write_json_atomic(view / "solver.json", solver)


async def _review_again(
    plan: PreparedReview,
    trial: Path,
    attempt: Path,
    rewards: dict | None,
    task_digest: str,
) -> ScoringResult:
    """The trial's rubric review, run again on its saved outputs.

    ``finish_review``, the run's own review step, over a view of the trial
    (:func:`_review_view`) in ``regrade/<id>/review/<trial>/``: the reviewer
    gets the same frozen workspace and records, the new verifier's outputs,
    and the fixed task; the new test gates the verdict. Every failure is a
    scoring error, as in the run.
    """
    from benchflow._utils.text import describe_exception
    from benchflow.review.automatic import finish_review
    from benchflow.review.outcome import scoring_error

    view = attempt / "review" / trial.name
    try:
        await asyncio.to_thread(
            _review_view, trial, attempt, view, rewards, task_digest
        )
    except OSError as exc:
        return scoring_error(f"review staging failed: {describe_exception(exc)}")
    return await finish_review(plan, view)


# --- orchestration ---------------------------------------------------------


def _new_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


async def _regrade_one(
    trial: Path,
    *,
    tasks_dir: Path | None,
    sandbox: str | None,
    reason: str | None,
    regrade_id: str,
    runner: Runner,
    review: bool = False,
    reviewer: ReviewerConfig | None = None,
) -> TrialRegrade:
    config = _read_json(trial / "config.json")
    result = _read_json(trial / "result.json")
    task_name = _task_dirname(trial, config, result)
    original = original_score(trial)
    # The regrade reruns the test, and the rubric review only with ``review``.
    # Without it, a reviewed trial's recorded reward is the review's, so the
    # test is compared with the test.
    compared = (
        "verifier_reward"
        if original["verifier_reward"] is not None and not review
        else "reward"
    )
    before = original[compared]
    row = TrialRegrade(
        trial=trial.name,
        task=task_name,
        status="not_regradable",
        original_reward=before,
        compared=compared,
    )
    why = regradable_reason(trial)
    task_dir = None
    if why is None:
        task_dir, why = resolve_task(trial, tasks_dir)
    if why is None and task_dir is not None:
        task_dir, why = _restorable_task(trial, task_dir)
    if why is not None or task_dir is None:
        row.reason = why
        return row
    from benchflow._utils.task_authoring import task_digest

    attempt = trial / "regrade" / regrade_id
    attempt.mkdir(parents=True, exist_ok=False)
    tests = _test_files_dir(task_dir)
    if tests is not None:
        shutil.copytree(tests, attempt / "verifier-files")
    block: dict[str, Any] = {
        "id": regrade_id,
        "created_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "task_dir": str(task_dir),
        "sandbox": sandbox or config.get("environment"),
        "original_task_digest": original["task_digest"],
        "task_digest": task_digest(task_dir),
        "verifier_digest": verifier_digest(task_dir),
        "original_rewards": original["rewards"],
        "original_reward": before,
        "compared": compared,
        "original_verifier_error": original["verifier_error"],
    }
    block["task_changed"] = block["original_task_digest"] != block["task_digest"]
    rewards: dict | None = None
    error: str | None = None
    plan = scoring = None
    try:
        # A missing rubric or reviewer key fails here, before any sandbox.
        plan = _review_plan(trial, task_dir, reviewer) if review else None
        rewards, error = await runner(trial, task_dir, attempt, sandbox=sandbox)
        if plan is not None and not error:
            scoring = await _review_again(
                plan, trial, attempt, rewards, block["task_digest"]
            )
    except asyncio.CancelledError:
        block.update(status="interrupted", new_rewards=rewards)
        _write_block(trial, attempt, block)
        raise
    except Exception as exc:
        logger.exception("Regrade of %s failed", trial)
        error = f"{type(exc).__name__}: {exc}"
    new_reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if plan is not None and scoring is not None:
        # The combined score (tests, blockers, quality), as result.json has it.
        combined = scoring.numeric_rewards()
        new_reward = combined["reward"] if combined is not None else None
        if combined is None:
            error = f"review failed: {scoring.error}"
        block["review"] = {
            "rubric_sha256": plan.rubric_sha256,
            "original_rubric_sha256": (config.get("review") or {}).get("rubric_sha256"),
            "reviewer": plan.config.to_config_artifact(),
            "rewards": combined,
            "scoring": scoring.to_dict(),
            "view": f"regrade/{regrade_id}/review/{trial.name}",
        }
    # A failed sandbox run is no verdict, not an "unscored" one.
    change = None if error else verdict_change(before, new_reward)
    block.update(
        status="failed" if error else "complete",
        new_rewards=rewards,
        new_reward=new_reward,
        verifier_error=error,
        change=change,
    )
    from benchflow.rollout._separate_verifier import RECORD_FILE

    separate = _read_json(attempt / RECORD_FILE)
    if separate:
        # The separate verifier scored it; its record lies beside verifier/.
        block["verifier_sandbox"] = {
            "image_source": separate.get("image_source"),
            "status": separate.get("status"),
        }
    _write_block(trial, attempt, block)
    row.regrade_id = regrade_id
    row.new_reward = new_reward
    row.task_changed = block["task_changed"]
    if error:
        row.status, row.reason = "failed", error
    else:
        row.status, row.change = "regraded", change
        if change not in (None, "same") and not block["task_changed"]:
            row.reason = UNCHANGED_TASK_FLIP
    return row


def _test_files_dir(task_dir: Path) -> Path | None:
    from benchflow.task import Task

    try:
        tests = Task(task_dir).paths.tests_dir
    except Exception:
        return None
    return tests if tests.is_dir() else None


async def aregrade(
    path: str | Path,
    *,
    tasks_dir: str | Path | None = None,
    sandbox: str | None = None,
    concurrency: int = 4,
    reason: str | None = None,
    runner: Runner | None = None,
    review: bool = False,
    reviewer: ReviewerConfig | None = None,
) -> RegradeSummary:
    """Async :func:`regrade`."""
    from benchflow.review.persistence import write_json_atomic

    root = Path(path).resolve()
    trials = find_trials(root)
    regrade_id = _new_id()
    tasks = Path(tasks_dir).resolve() if tasks_dir is not None else None
    gate = asyncio.Semaphore(max(1, concurrency))
    run = runner or run_verifier_on_frozen_trial

    async def one(trial: Path) -> TrialRegrade:
        async with gate:
            return await _regrade_one(
                trial,
                tasks_dir=tasks,
                sandbox=sandbox,
                reason=reason,
                regrade_id=regrade_id,
                runner=run,
                review=review,
                reviewer=reviewer,
            )

    rows = await asyncio.gather(*(one(t) for t in trials))
    summary = RegradeSummary(
        path=str(root),
        regrade_id=regrade_id,
        tasks_dir=str(tasks) if tasks else None,
        reason=reason,
        trials=list(rows),
    )
    write_json_atomic(root / SUMMARY_FILE, summary.to_dict())
    return summary


def regrade(
    path: str | Path,
    *,
    tasks_dir: str | Path | None = None,
    sandbox: str | None = None,
    concurrency: int = 4,
    reason: str | None = None,
    runner: Runner | None = None,
    review: bool = False,
    reviewer: ReviewerConfig | None = None,
) -> RegradeSummary:
    """Re-run each trial's task verifier (from ``tasks_dir`` when given) on
    the trial's frozen final workspace, in a fresh sandbox.

    ``path`` is a job folder or one trial folder. ``sandbox`` defaults to the
    backend each trial ran on. ``review`` also reruns each trial's rubric
    review on the same saved outputs, gated by the new test, with the
    reviewer settings the trial recorded under ``reviewer``'s; the combined
    score is then what is compared. The original ``result.json`` is never
    touched; see the module docstring for what is written. Blocking form of
    :func:`aregrade`, with the same arguments; it also works inside a running
    event loop, such as a Jupyter cell.
    """
    from benchflow.batch import run_blocking

    return run_blocking(
        lambda: aregrade(
            path,
            tasks_dir=tasks_dir,
            sandbox=sandbox,
            concurrency=concurrency,
            reason=reason,
            runner=runner,
            review=review,
            reviewer=reviewer,
        )
    )
