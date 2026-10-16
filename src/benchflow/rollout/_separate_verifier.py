"""Run a task's verifier in its own sandbox (Harbor ``environment_mode = "separate"``).

A task opts in with ``[verifier] sandbox_mode = "separate"`` (Harbor:
``environment_mode``) or by declaring ``[verifier.sandbox]``
(``[verifier.environment]``); ``--separate-verifier`` forces it for a run
without editing the task. The verifier then never runs where the agent ran:

1. After the agent stops, the rollout freezes the workspace into
   ``evidence/`` — the same capture the rubric review uses.
2. The host packs exactly those bytes, hash-checked against their manifest,
   into one tar. Nothing else from the agent's sandbox is read: files planted
   in ``/tests``, ``/logs/verifier``, site-packages, ``/usr/local/bin`` or
   anywhere outside the workspace do not exist in the verifier sandbox, and
   neither does anything a process with root changed there.
3. A fresh sandbox starts from the verifier image (see
   :func:`plan_verifier_image`), the tar is unpacked at the original absolute
   paths, and the normal verifier path runs there: hardening (which still
   removes workspace ``conftest.py`` / ``.pth`` / ``sitecustomize.py``),
   ``tests/`` upload, ``test.sh``, reward parsing.

Any failure before the verifier produces a reward — no frozen workspace, a
manifest mismatch, an image that does not build, an upload that does not
unpack — is an assessment error: ``rewards`` stays ``None`` and
``verifier_error`` starts with ``separate verifier``. It is never scored 0,
and never a pass.

Per-phase timing lands in the rollout's ``timing`` (``verifier_sandbox_setup``,
``verifier_transfer``, ``verifier``, ``verifier_sandbox_teardown``,
``verifier_sandbox_total``) and ``verifier-sandbox/verifier-sandbox.json``
records the image source, sandbox id, transfer inventory, status and
sandbox-seconds. Both sandboxes are billed while the verifier runs.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import posixpath
import shlex
import shutil
import stat
import tarfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier_sandbox import (
    SeparateVerifierError,
    VerifierImage,
    plan_verifier_image,
    separate_verifier_requested,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ERROR_PREFIX",
    "SeparateVerifierError",
    "build_transfer_payload",
    "run_separate_verifier",
    "separate_verifier_requested",
]

ERROR_PREFIX = "separate verifier"
RECORD_DIR = "verifier-sandbox"
RECORD_FILE = "verifier-sandbox.json"
# The frozen workspace replaces the contents of a dedicated task directory
# (/app, /testbed, /workspace, /home/agent/project), so what the agent deleted
# or renamed stays gone. Emptying one of these would wipe the verifier image,
# so the workspace is laid over it instead.
_SHARED_DIRS = frozenset(
    {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/lib",
        "/lib32",
        "/lib64",
        "/libx32",
        "/logs",
        "/media",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
    }
)
_STOP_TIMEOUT_SEC = 120


# --- host side: what crosses over ------------------------------------------


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SeparateVerifierError(f"unsafe transfer path {value!r}")
    return path


def _absolute(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise SeparateVerifierError(f"unsafe sandbox path {value!r}")
    return path


class _Payload:
    """One tar whose member names are sandbox paths without the leading ``/``."""

    def __init__(self, archive: tarfile.TarFile) -> None:
        self.archive = archive
        self.names: set[str] = set()
        self.files = 0
        self.bytes = 0

    def _info(
        self, target: PurePosixPath, kind: bytes, mode: int
    ) -> tarfile.TarInfo | None:
        name = str(target).lstrip("/")
        if not name or name in self.names:
            return None
        self.names.add(name)
        info = tarfile.TarInfo(name)
        info.type, info.mode, info.uid, info.gid = kind, mode, 0, 0
        info.mtime = int(time.time())
        return info

    def directory(self, target: PurePosixPath) -> None:
        info = self._info(target, tarfile.DIRTYPE, 0o755)
        if info is not None:
            self.archive.addfile(info)

    def file(self, target: PurePosixPath, source: Path) -> None:
        # The manifest records no permission bits; keep the executable bit the
        # captured copy has, as a regrade restores it.
        mode = 0o755 if source.stat().st_mode & stat.S_IXUSR else 0o644
        info = self._info(target, tarfile.REGTYPE, mode)
        if info is None:
            raise SeparateVerifierError(f"two transfers write {target}")
        info.size = source.stat().st_size
        with source.open("rb") as stream:
            self.archive.addfile(info, stream)
        self.files += 1
        self.bytes += info.size

    def symlink(self, target: PurePosixPath, link_target: str) -> None:
        info = self._info(target, tarfile.SYMTYPE, 0o777)
        if info is not None:
            info.linkname = link_target
            self.archive.addfile(info)


def _add_bundle(payload: _Payload, bundle: Path, manifest: Any) -> None:
    root = _absolute(manifest.workspace)
    tree = bundle / "workspace"
    entries = sorted(
        manifest.entries,
        key=lambda e: (e.kind == "symlink", e.kind == "file", len(e.path)),
    )
    # No member for the root itself: an existing /tmp or /root keeps its
    # mode, and the install script creates a missing workspace.
    for entry in entries:
        target = root / _safe_relative(entry.path)
        if entry.kind == "directory":
            payload.directory(target)
        elif entry.kind == "file":
            payload.file(target, tree / entry.path)
        elif entry.link_target is not None:
            payload.symlink(target, entry.link_target)


def build_transfer_payload(rollout_dir: Path, dest: Path) -> dict[str, Any]:
    """Pack the frozen workspace (and any captured declared artifacts).

    Every byte is checked against the manifest written when it was captured.
    Returns the transfer inventory; raises :class:`SeparateVerifierError`
    when anything the verifier needs is missing or does not match.
    """
    from benchflow.review.evidence import (
        EvidenceError,
        EvidenceManifest,
        validate_workspace,
    )

    bundle = Path(rollout_dir) / "evidence"
    if not (bundle / "manifest.json").is_file():
        raise SeparateVerifierError("no frozen workspace (evidence/manifest.json)")
    try:
        manifest = EvidenceManifest.model_validate_json(
            (bundle / "manifest.json").read_text()
        )
        validate_workspace(bundle / "workspace", manifest)
        subs = []
        for artifact in manifest.artifacts:
            if artifact.bundle_path is None:
                continue
            sub = bundle / artifact.bundle_path
            sub_manifest = EvidenceManifest.model_validate_json(
                (sub / "manifest.json").read_text()
            )
            validate_workspace(sub / "workspace", sub_manifest)
            subs.append((sub, sub_manifest))
    except (EvidenceError, OSError, ValueError) as exc:
        raise SeparateVerifierError(
            f"frozen workspace does not match its manifest: {exc}"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    with tarfile.open(dest, "x", format=tarfile.PAX_FORMAT) as archive:
        payload = _Payload(archive)
        _add_bundle(payload, bundle, manifest)
        for sub, sub_manifest in subs:
            _add_bundle(payload, sub, sub_manifest)
    with dest.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "workspace": manifest.workspace,
        "files": payload.files,
        "bytes": payload.bytes,
        "declared_artifacts": len(subs),
        "excluded": len(manifest.exclusions),
        "sha256": sha256,
    }


# --- sandbox side ------------------------------------------------------------


def _install_script(remote: str, workspace: str, sha256: str, root: str = "/") -> str:
    """Unpack the transfer at ``root`` (the sandbox's ``/``; a test's folder).

    Shell and tar only: a verifier image may have no Python.
    """
    quoted = shlex.quote(remote)
    # Directories this creates (a missing workspace) get the verifier's modes
    # whatever the runtime's mask; tar restores its own.
    lines = ["set -e", "umask 022"]
    lines.append(
        "if command -v sha256sum >/dev/null 2>&1; then "
        f"echo {shlex.quote(sha256 + '  ' + remote)} | sha256sum -c - >/dev/null; fi"
    )
    path = PurePosixPath(workspace)
    target = shlex.quote(posixpath.join(root, workspace.lstrip("/")))
    if str(path) not in _SHARED_DIRS:
        # A dedicated task directory: the agent's final state replaces what
        # the verifier image put there. The directory itself stays (a shell
        # whose cwd it is keeps a valid cwd, as in a regrade); its contents,
        # dotfiles included, go.
        lines.append(f"if [ -L {target} ]; then rm -f -- {target}; fi")
        lines.append(
            f"if [ -d {target} ]; then rm -rf -- {target}/* {target}/.[!.]* "
            f"{target}/..?*; else rm -f -- {target}; fi"
        )
    lines.append(f"mkdir -p -- {target}")
    lines.append(f"tar -xf {quoted} -C {shlex.quote(root)}")
    lines.append(f"rm -f -- {quoted}")
    return "\n".join(lines)


async def _install_payload(
    env: Any, archive: Path, summary: dict[str, Any], timeout_sec: int
) -> None:
    remote = f"/tmp/benchflow-verifier-transfer-{uuid.uuid4().hex}.tar"
    try:
        await env.upload_file(archive, remote)
        script = _install_script(remote, summary["workspace"], summary["sha256"])
        result = await env.exec(
            shlex.join(["sh", "-c", script]), user="root", timeout_sec=timeout_sec
        )
    except SeparateVerifierError:
        raise
    except Exception as exc:
        raise SeparateVerifierError(
            f"upload failed: {type(exc).__name__}: {exc}"
        ) from exc
    return_code = getattr(result, "return_code", None)
    if return_code is None:
        return_code = getattr(result, "exit_code", 0)
    if return_code:
        detail = (result.stderr or result.stdout or "").strip()[-500:]
        raise SeparateVerifierError(f"unpack failed (exit {return_code}): {detail}")


def _verifier_task(task: Any, plan: VerifierImage) -> Any:
    """``task`` as the verifier sandbox sees it: its image, a shared verifier."""
    view = copy.copy(task)
    config = task.config.model_copy(deep=True)
    config.sandbox = plan.sandbox.model_copy(deep=True)
    config.verifier.sandbox_mode = None
    config.verifier.sandbox = None
    # The agent's network policy binds the agent uid in the agent sandbox; it
    # must not be folded into the verifier sandbox.
    config.agent.network_mode = None
    config.agent.allowed_hosts = None
    config.artifacts = []
    view.config = config
    if getattr(view, "document", None) is not None:
        # The launch gate reads a task document before the config; this view
        # has no document, so the gate checks the shared verifier above.
        view.document = None
    return view


def _stage_context(root: Path, task_name: str, plan: VerifierImage) -> Path:
    """A task path whose ``environment/`` is the verifier image's build context.

    Named ``<task>__verifier`` so a Docker image tag never collides with the
    agent image built from the task's own ``environment/``.
    """
    context = root / "context" / f"{task_name}__verifier"
    if context.exists():
        shutil.rmtree(context)
    environment = context / "environment"
    if plan.context_dir is not None:
        shutil.copytree(plan.context_dir, environment, symlinks=False)
    else:
        environment.mkdir(parents=True)
    return context


CreateEnvironment = Callable[[Any, Path, RolloutPaths], Any]
Verify = Callable[..., Awaitable[tuple[dict | None, str | None]]]


def _default_create(rollout: Any) -> CreateEnvironment:
    cfg = rollout._config

    def create(task: Any, context_path: Path, paths: RolloutPaths) -> Any:
        return rollout._planes.create_environment(
            cfg.environment,
            task,
            context_path,
            f"{rollout._rollout_name or 'rollout'}-verifier",
            paths,
            preserve_agent_network=False,
            environment_manifest=None,
        )

    return create


def _default_verify(rollout: Any) -> Verify:
    async def verify(
        env: Any,
        task: Any,
        paths: RolloutPaths,
        timing: dict,
        *,
        workspace: str,
    ) -> tuple[dict | None, str | None]:
        from benchflow.rollout._setup import (
            _publish_trajectory_for_verifier,
            _verify_rollout,
        )

        await _publish_trajectory_for_verifier(
            env, rollout._trajectory, paths.agent_dir
        )
        rewards, error, timeout_diag = await _verify_rollout(
            env,
            task,
            paths,
            timing,
            rollout._planes,
            # No agent user exists here and no agent process ever ran: only
            # what the transfer wrote came from the agent.
            sandbox_user=None,
            workspace=workspace,
        )
        diagnostics = getattr(rollout, "_diagnostics", None)
        if timeout_diag is not None and diagnostics is not None:
            diagnostics.set(timeout_diag)
        return rewards, error

    return verify


def _publish_verifier_outputs(source: Path, dest: Path) -> None:
    if not source.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True, symlinks=True)


async def _stop(env: Any) -> str | None:
    try:
        await asyncio.wait_for(env.stop(delete=True), timeout=_STOP_TIMEOUT_SEC)
    except Exception as exc:
        logger.warning("Verifier sandbox cleanup failed", exc_info=True)
        return f"{type(exc).__name__}: {exc}"
    return None


def _write_record(path: Path, record: dict[str, Any]) -> None:
    from benchflow.review.persistence import write_json_atomic

    try:
        write_json_atomic(path, record)
    except Exception:
        # The record is diagnostics; it must never replace the verifier result.
        logger.warning("Could not write %s", path, exc_info=True)


async def run_separate_verifier(
    rollout: Any,
    *,
    create_environment: CreateEnvironment | None = None,
    verify: Verify | None = None,
) -> tuple[dict | None, str | None]:
    """Score ``rollout`` in a fresh verifier sandbox; return ``(rewards, error)``."""
    from benchflow.sandbox.metadata import persist_sandbox_info

    rollout_dir = Path(rollout._rollout_dir)
    root = rollout_dir / RECORD_DIR
    root.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = rollout._timing
    task = rollout._task
    record: dict[str, Any] = {
        "mode": "separate",
        "backend": rollout._config.environment,
        "status": "pending",
        "image_source": None,
        "sandbox_id": None,
        "transfer": None,
        "error": None,
    }
    archive = root / "transfer.tar"
    env = None
    rewards: dict | None = None
    error: str | None = None
    started = time.monotonic()
    try:
        plan = plan_verifier_image(task.config, Path(task.paths.task_dir))
        record["image_source"] = plan.source
        try:
            summary = await asyncio.to_thread(
                build_transfer_payload, rollout_dir, archive
            )
        except SeparateVerifierError as exc:
            capture = getattr(rollout, "_export_error", None)
            detail = f" ({capture})" if capture else ""
            raise SeparateVerifierError(f"transfer failed: {exc}{detail}") from exc
        record["transfer"] = {k: v for k, v in summary.items() if k != "sha256"}
        paths = RolloutPaths(rollout_dir=root)
        paths.mkdir()
        vtask = _verifier_task(task, plan)
        context = await asyncio.to_thread(_stage_context, root, task.name, plan)
        create = create_environment or _default_create(rollout)
        env = create(vtask, context, paths)
        t0 = time.monotonic()
        try:
            await env.start(force_build=False)
        except Exception as exc:
            record["status"] = "sandbox_failed"
            raise SeparateVerifierError(
                f"sandbox failed to start: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            timing["verifier_sandbox_setup"] = time.monotonic() - t0
        sandbox_id = getattr(env, "sandbox_id", None)
        record["sandbox_id"] = sandbox_id if isinstance(sandbox_id, str) else None
        persist_sandbox_info(env, root)
        t0 = time.monotonic()
        try:
            await _install_payload(
                env, archive, summary, int(rollout._config.sandbox_setup_timeout or 600)
            )
        except SeparateVerifierError as exc:
            raise SeparateVerifierError(f"transfer failed: {exc}") from exc
        finally:
            timing["verifier_transfer"] = time.monotonic() - t0
        run = verify or _default_verify(rollout)
        rewards, error = await run(
            env, vtask, paths, timing, workspace=summary["workspace"]
        )
        _publish_verifier_outputs(
            paths.verifier_dir, rollout._rollout_paths.verifier_dir
        )
        record["status"] = "complete" if error is None else "verifier_failed"
    except SeparateVerifierError as exc:
        rewards, error = None, f"{ERROR_PREFIX} {exc}"
        if record["status"] == "pending":
            record["status"] = (
                "transfer_failed" if "transfer failed" in str(exc) else "sandbox_failed"
            )
    except asyncio.CancelledError:
        record["status"] = "interrupted"
        raise
    except Exception as exc:
        logger.exception("Separate verifier sandbox failed")
        rewards, error = (
            None,
            f"{ERROR_PREFIX} sandbox failed: {type(exc).__name__}: {exc}",
        )
        record["status"] = "sandbox_failed"
    finally:
        if env is not None:
            t0 = time.monotonic()
            cleanup_error = await _stop(env)
            timing["verifier_sandbox_teardown"] = time.monotonic() - t0
            if cleanup_error:
                record["cleanup_error"] = cleanup_error
            total = time.monotonic() - started
            timing["verifier_sandbox_total"] = total
            record["sandbox_seconds"] = round(total, 3)
        else:
            record["sandbox_seconds"] = 0.0
        archive.unlink(missing_ok=True)
        record["error"] = error
        record["timing"] = {
            k: round(v, 3) for k, v in timing.items() if k.startswith("verifier")
        }
        _write_record(root / RECORD_FILE, record)
    return rewards, error


def read_record(rollout_dir: Path) -> dict[str, Any] | None:
    """The separate verifier record of a finished rollout, if it ran one."""
    path = Path(rollout_dir) / RECORD_DIR / RECORD_FILE
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
