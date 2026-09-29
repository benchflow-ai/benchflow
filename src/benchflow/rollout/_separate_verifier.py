"""Run a task's verifier in its own sandbox (Harbor ``environment_mode = "separate"``).

A task opts in with ``[verifier] environment_mode = "separate"`` (BenchFlow:
``sandbox_mode``) or by declaring ``[verifier.environment]``
(``[verifier.sandbox]``). The verifier then never runs where the agent ran:

1. After the agent stops, the rollout freezes the workspace into
   ``evidence/`` (with the task's declared artifacts) and collects
   ``/logs/artifacts`` into ``artifacts/`` — the same captures regrade uses.
2. The host packs exactly those bytes, hash-checked against their manifests,
   into one tar. Nothing else from the agent's sandbox is read: files planted
   in ``/tests``, ``/logs/verifier``, site-packages or anywhere outside the
   workspace and the declared artifacts do not exist in the verifier sandbox.
3. A fresh sandbox starts from the verifier image (see
   :func:`plan_verifier_image`), the tar is unpacked at the original absolute
   paths, and the normal verifier path runs there: hardening (which still
   removes workspace ``conftest.py`` / ``.pth`` / ``sitecustomize.py``),
   ``tests/`` upload, ``test.sh``, reward parsing.

Any failure before the verifier produces a reward — no frozen workspace, a
manifest mismatch, a failed artifact collection, an image that does not
build, an upload that does not unpack — is an assessment error: ``rewards``
stays ``None`` and ``verifier_error`` starts with ``separate verifier``. It
is never scored 0, with one exception: when the solution's own files are why
its outputs cannot cross (the workspace or ``/logs/artifacts`` over the
capture limits, a declared artifact that is a symlink, a ``/logs/artifacts``
file taking a declared artifact's place) and the same paths were within
bounds before the agent ran, the trial scores 0 and the record says why
(status ``refused``). The measurement before the agent is the clean control:
a failure it would also have hit stays unscored.

Per-phase timing lands in the rollout's ``timing`` (``verifier_sandbox_setup``,
``verifier_transfer``, ``verifier``, ``verifier_sandbox_teardown``,
``verifier_sandbox_total``) and ``verifier-sandbox/verifier-sandbox.json``
records the image source, sandbox id, transfer inventory, status and
sandbox-seconds. Job budgets add ``verifier_sandbox_total`` to the trial's
wall clock, since both sandboxes are billed while the verifier runs.
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
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.review.evidence import (
    WORKSPACE_MAX_BYTES,
    WORKSPACE_MAX_ENTRIES,
    python_missing,
)
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier_sandbox import (
    SeparateVerifierError,
    VerifierImage,
    plan_verifier_image,
    separate_verifier_requested,
    verifier_image_issue,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ERROR_PREFIX",
    "SeparateVerifierError",
    "SolutionTransferRefused",
    "VerifierImage",
    "build_transfer_payload",
    "measure_outputs",
    "plan_verifier_image",
    "record_pristine_outputs",
    "run_separate_verifier",
    "separate_verifier_requested",
    "verifier_image_issue",
]

ERROR_PREFIX = "separate verifier"
RECORD_DIR = "verifier-sandbox"
RECORD_FILE = "verifier-sandbox.json"
# Collection statuses that mean the bytes did not reach the host intact.
_FAILED_COLLECTIONS = frozenset({"error", "over_limit", "refused"})
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


class SolutionTransferRefused(SeparateVerifierError):
    """The solution's own files keep its outputs from the verifier; scored 0."""


# --- before the agent: the clean control -------------------------------------

# For each path: whether it exists, whether it is a symlink, and the entries
# and regular-file bytes under it (following only a top-level link, as capture
# does for the workspace). Excluded paths are counted too, so the count never
# undershoots what capture would have kept.
_PRISTINE_PROBE = r"""
import json, os, stat, sys
out = {}
for path in json.loads(sys.argv[1]):
    info = {"exists": os.path.lexists(path), "link": os.path.islink(path), "entries": 0, "bytes": 0}
    if os.path.isdir(path):
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                info["entries"] += 1
                try:
                    item = os.lstat(os.path.join(directory, name))
                except OSError:
                    continue
                if stat.S_ISREG(item.st_mode):
                    info["bytes"] += item.st_size
    elif os.path.isfile(path):
        info["entries"], info["bytes"] = 1, os.stat(path).st_size
    out[path] = info
print(json.dumps(out))
"""

# The same for images without python3: "exists link entries bytes" per path,
# with "?" for a count this image has no tool for.
_PRISTINE_SHELL_PROBE = r"""
for p in "$@"; do
  e=0; l=0; n=0; b=0
  if [ -e "$p" ] || [ -L "$p" ]; then e=1; fi
  if [ -L "$p" ]; then l=1; fi
  if [ -d "$p" ]; then
    if command -v find >/dev/null 2>&1 && command -v stat >/dev/null 2>&1; then
      n=$(find "$p/" -mindepth 1 | wc -l | tr -d ' ')
      b=$(find "$p/" -type f -exec stat -c %s {} + | awk '{s += $1} END {printf "%.0f", s}')
    else
      n='?'; b='?'
    fi
  elif [ -f "$p" ]; then
    n=1; b=$(stat -L -c %s "$p" 2>/dev/null || echo '?')
  fi
  echo "$e $l $n $b"
done
"""


def _count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _measured(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "exists": info.get("exists") in (True, "1"),
        "link": info.get("link") in (True, "1"),
        "entries": _count(info.get("entries")),
        "bytes": _count(info.get("bytes")),
    }


async def measure_outputs(
    env: Any,
    *,
    workspace: str,
    artifacts: Sequence[Any] = (),
    logs_source: str | None = None,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Measure the paths separate-verifier capture reads.

    For the workspace, ``/logs/artifacts`` and each declared artifact source:
    whether it exists, whether it is a symlink, and its entries and bytes
    (None when the image has no tool to count them). Raises when the probe
    cannot run.
    """
    from benchflow.rollout._artifacts import LOGS_ARTIFACTS
    from benchflow.task.artifacts import as_artifact_config

    logs = logs_source or LOGS_ARTIFACTS
    sources = [
        str(PurePosixPath(workspace) / as_artifact_config(item).source)
        for item in artifacts
    ]
    paths = list(dict.fromkeys([workspace, logs, *sources]))
    result = await env.exec(
        shlex.join(["python3", "-I", "-c", _PRISTINE_PROBE, json.dumps(paths)]),
        user="root",
        timeout_sec=timeout_sec,
    )
    if result.return_code and python_missing(result):
        result = await env.exec(
            shlex.join(["sh", "-c", _PRISTINE_SHELL_PROBE, "probe", *paths]),
            user="root",
            timeout_sec=timeout_sec,
        )
        if result.return_code:
            raise RuntimeError((result.stderr or result.stdout or "")[-500:])
        lines = (result.stdout or "").splitlines()
        if len(lines) != len(paths):
            raise RuntimeError("the probe returned an incomplete listing")
        raw = {
            # A short line leaves its counts unknown.
            path: dict(
                zip(("exists", "link", "entries", "bytes"), line.split(), strict=False)
            )
            for path, line in zip(paths, lines, strict=True)
        }
    elif result.return_code:
        raise RuntimeError((result.stderr or result.stdout or "")[-500:])
    else:
        raw = json.loads(result.stdout or "")
    return {
        "workspace": workspace,
        "logs": logs,
        "paths": {path: _measured(raw[path]) for path in paths},
    }


async def record_pristine_outputs(rollout: Any) -> None:
    """Measure what capture will read before the agent runs: the clean control.

    Stores :func:`measure_outputs` as ``rollout._pristine_outputs``, or None
    when the probe could not run, which leaves every capture failure unscored.
    """
    rollout._pristine_outputs = None
    try:
        rollout._pristine_outputs = await measure_outputs(
            rollout._env,
            workspace=str(rollout._agent_cwd),
            artifacts=rollout._task.config.artifacts,
            timeout_sec=int(
                getattr(rollout._config, "sandbox_setup_timeout", 0) or 120
            ),
        )
    except Exception as exc:
        logger.warning(
            "Could not measure the workspace before the agent ran (%s); a "
            "separate verifier leaves any capture failure unscored",
            exc,
        )


def _pristine_totals(
    pristine: dict[str, Any] | None, *, skip: str
) -> tuple[int, int] | None:
    """Entries and bytes over the measured paths other than ``skip``."""
    if not pristine:
        return None
    entries = size = 0
    for path, info in pristine["paths"].items():
        if path == skip:
            continue
        if info["entries"] is None or info["bytes"] is None:
            return None
        entries += info["entries"]
        size += info["bytes"]
    return entries, size


def _workspace_refusal(pristine: dict[str, Any] | None) -> str | None:
    """Why an over-limit workspace capture is the solution's doing, if it is."""
    if not pristine:
        return None
    totals = _pristine_totals(pristine, skip=pristine["logs"])
    if totals is None:
        return None
    entries, size = totals
    if entries > WORKSPACE_MAX_ENTRIES or size > WORKSPACE_MAX_BYTES:
        return None
    return (
        "the solution's workspace and declared artifacts exceed the capture "
        f"limits ({WORKSPACE_MAX_ENTRIES} entries, {WORKSPACE_MAX_BYTES} bytes); "
        f"before the agent ran they held {entries} entries and {size} bytes"
    )


def _collection_attributable(
    record: dict[str, Any], limits: dict[str, Any], pristine: dict[str, Any] | None
) -> bool:
    """Whether a failed collection is the solution's doing (see ``cause``)."""
    if not pristine:
        return False
    cause = record.get("cause")
    if cause == "limits":
        totals = _pristine_totals(pristine, skip=pristine["workspace"])
        max_files, max_bytes = limits.get("max_files"), limits.get("max_bytes")
        return (
            totals is not None
            and isinstance(max_files, int)
            and isinstance(max_bytes, int)
            and totals[0] <= max_files
            and totals[1] <= max_bytes
        )
    if cause == "symlink":
        before = pristine["paths"].get(record.get("source"))
        return before is not None and not before["link"]
    if cause == "clash":
        logs = pristine["paths"].get(pristine["logs"])
        return logs is not None and logs["entries"] == 0
    return False


def _collection_error(record: dict[str, Any]) -> str:
    return (
        f"artifact collection of {record.get('source')} "
        f"{record.get('status')}: {record.get('reason')}"
    )


# --- host side: what crosses over ------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


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

    def directory(self, target: PurePosixPath, mode: int | None = None) -> None:
        info = self._info(target, tarfile.DIRTYPE, 0o755 if mode is None else mode)
        if info is not None:
            self.archive.addfile(info)

    def file(
        self, target: PurePosixPath, source: Path, mode: int | None = None
    ) -> None:
        if mode is None:
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
    # mode, and the install script creates a missing workspace. Files and
    # directories get the permission bits the solver left, as a regrade
    # restores them (older manifests without modes keep the executable bit).
    for entry in entries:
        target = root / _safe_relative(entry.path)
        mode = None if entry.mode is None else entry.mode & 0o777
        if entry.kind == "directory":
            payload.directory(target, mode)
        elif entry.kind == "file":
            payload.file(target, tree / entry.path, mode)
        elif entry.link_target is not None:
            payload.symlink(target, entry.link_target)


def build_transfer_payload(
    rollout_dir: Path,
    dest: Path,
    *,
    pristine: dict[str, Any] | None = None,
    capture_over_limit: bool = False,
) -> dict[str, Any]:
    """Pack the frozen workspace, declared artifacts and ``/logs/artifacts``.

    Every byte is checked against the manifest written when it was captured.
    Returns the transfer inventory; raises :class:`SeparateVerifierError`
    when anything the verifier needs is missing or does not match, or
    :class:`SolutionTransferRefused` when the solution's own files are why,
    judged against ``pristine`` (:func:`record_pristine_outputs`).
    """
    from benchflow.review.evidence import (
        EvidenceError,
        EvidenceManifest,
        validate_workspace,
    )
    from benchflow.rollout._artifacts import LOGS_ARTIFACTS, MANIFEST_NAME

    bundle = Path(rollout_dir) / "evidence"
    if not (bundle / "manifest.json").is_file():
        refusal = _workspace_refusal(pristine) if capture_over_limit else None
        if refusal is not None:
            raise SolutionTransferRefused(refusal)
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

    collected = _read_json(Path(rollout_dir) / MANIFEST_NAME)
    failed = [
        record
        for record in collected.get("collections", [])
        if record.get("status") in _FAILED_COLLECTIONS
    ]
    if failed:
        limits = collected.get("limits") or {}
        blocking = [
            record
            for record in failed
            if not _collection_attributable(record, limits, pristine)
        ]
        if blocking:
            raise SeparateVerifierError(_collection_error(blocking[0]))
        raise SolutionTransferRefused(
            "; ".join(_collection_error(record) for record in failed)
        )
    logs_files = [
        f
        for f in collected.get("files", [])
        if f.get("collection") == 0 and f.get("kind") == "file"
    ]

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    with tarfile.open(dest, "x", format=tarfile.PAX_FORMAT) as archive:
        payload = _Payload(archive)
        _add_bundle(payload, bundle, manifest)
        for sub, sub_manifest in subs:
            _add_bundle(payload, sub, sub_manifest)
        logs_root = PurePosixPath(LOGS_ARTIFACTS)
        for item in sorted(logs_files, key=lambda f: f["path"]):
            relative = _safe_relative(item["path"])
            source = Path(rollout_dir) / "artifacts" / relative
            try:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError as exc:
                raise SeparateVerifierError(
                    f"/logs/artifacts/{relative} is gone: {exc}"
                ) from exc
            if digest != item.get("sha256"):
                raise SeparateVerifierError(
                    f"/logs/artifacts/{relative} changed since collection"
                )
            for parent in reversed(list(relative.parents)[:-1]):
                payload.directory(logs_root / parent)
            payload.file(logs_root / relative, source)
    with dest.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "workspace": manifest.workspace,
        # The roots this transfer writes under in the verifier sandbox (a
        # declared file's bundle root is its directory). There the plugin guard
        # distrusts these alone (with /logs): nothing else in that sandbox came
        # from the agent's run.
        "agent_paths": [
            manifest.workspace,
            *(sub_manifest.workspace for _, sub_manifest in subs),
            LOGS_ARTIFACTS,
        ],
        "files": payload.files,
        "bytes": payload.bytes,
        "declared_artifacts": len(subs),
        "missing_artifacts": [
            a.source for a in manifest.artifacts if a.status == "missing"
        ],
        "logs_artifacts": len(logs_files),
        "excluded": len(manifest.exclusions),
        "sha256": sha256,
    }


# --- sandbox side ------------------------------------------------------------


def _install_script(remote: str, workspace: str, sha256: str, root: str = "/") -> str:
    """Unpack the transfer at ``root`` (the sandbox's ``/``; a test's folder).

    Shell and tar only: a verifier image may have no Python.
    """
    quoted = shlex.quote(remote)
    # Directories this creates (a missing workspace, /logs/artifacts) get the
    # verifier's modes whatever the runtime's mask; tar restores its own. The
    # mask is lockdown.VERIFIER_UMASK, which the rollout kernel must not import.
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
    logs = shlex.quote(posixpath.join(root, "logs/artifacts"))
    lines.append(f"mkdir -p -- {target} {logs}")
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
    if result.return_code:
        detail = (result.stderr or result.stdout or "").strip()[-500:]
        raise SeparateVerifierError(
            f"unpack failed (exit {result.return_code}): {detail}"
        )


def _verifier_task(task: Any, plan: VerifierImage) -> Any:
    """``task`` as the verifier sandbox sees it: its image, a shared verifier."""
    view = copy.copy(task)
    config = task.config.model_copy(deep=True)
    config.sandbox = plan.sandbox.model_copy(deep=True)
    config.verifier.sandbox_mode = None
    config.verifier.sandbox = None
    # The agent's network policy binds the agent uid in the agent sandbox; it
    # must not be folded into the verifier sandbox (agent_network_sandbox_config).
    config.agent.network_mode = None
    config.agent.allowed_hosts = None
    config.artifacts = []
    view.config = config
    if getattr(view, "document", None) is not None:
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
        agent_paths: tuple[str, ...] = (),
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
            agent_paths=(workspace, *agent_paths),
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


async def run_separate_verifier(
    rollout: Any,
    *,
    create_environment: CreateEnvironment | None = None,
    verify: Verify | None = None,
) -> tuple[dict | None, str | None]:
    """Score ``rollout`` in a fresh verifier sandbox; return ``(rewards, error)``."""
    from benchflow.review.persistence import write_json_atomic
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
        record["pristine"] = getattr(rollout, "_pristine_outputs", None)
        try:
            summary = await asyncio.to_thread(
                build_transfer_payload,
                rollout_dir,
                archive,
                pristine=record["pristine"],
                capture_over_limit=getattr(rollout, "_capture_over_limit", False),
            )
        except SolutionTransferRefused:
            raise
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
            env,
            vtask,
            paths,
            timing,
            workspace=summary["workspace"],
            agent_paths=tuple(summary["agent_paths"]),
        )
        _publish_verifier_outputs(
            paths.verifier_dir, rollout._rollout_paths.verifier_dir
        )
        record["status"] = "complete" if error is None else "verifier_failed"
    except SolutionTransferRefused as exc:
        # The solution's own files: its result, not a failure to assess it.
        # No verifier error, so the retry loop cannot resample it away.
        rewards, error = {"reward": 0.0}, None
        record["status"] = "refused"
        record["refusal"] = f"the solution's outputs cannot reach the verifier: {exc}"
        logger.warning("%s; scored 0", record["refusal"])
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
        write_json_atomic(root / RECORD_FILE, record)
    return rewards, error
