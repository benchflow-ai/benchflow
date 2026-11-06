"""A task's ``[[verifier.collect]]`` hooks and its sidecar artifacts (task.toml format).

A task snapshots runtime state into files once the agent's turn is over. Its
``[[verifier.collect]]`` commands run in the task's compose services: the
``main`` ones after the agent's processes are stopped and before the workspace
and declared artifacts are frozen (:func:`run_collect_hooks` from
``capture_terminal_workspace``), the other services' ones before those
services' artifacts are read. A ``main`` hook that names no ``user`` runs as
the account the agent ran as (:func:`agent_account`), never as root over the
agent's files, and that account's processes are stopped again after it. A
hook is best effort, as the format says: a failure is logged and recorded in
``collect-hooks.json``, and the trial goes on.

An artifact whose ``service`` names another compose service is read from that
service's own filesystem, which the agent's container does not share: it is
copied there with `cp -a` (links stay links) into a fresh folder, downloaded into
the trial's ``artifacts/services/<service>/<source>``, and listed with its
digests in ``artifacts/services/manifest.json``. In separate verifier mode
each listed file crosses into the verifier sandbox at its source path once
its bytes match the list (:func:`install_sidecar_artifacts`), as the format's
verifier reads them; a source under a verifier-owned folder never crosses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow._utils.reachable_write import ReachablePath
from benchflow.review.persistence import write_json_atomic
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._task_route import as_task_command
from benchflow.task.config import MAIN_SERVICE, ArtifactConfig, VerifierCollectConfig

logger = logging.getLogger(__name__)

HOOKS_RECORD = "collect-hooks.json"
SIDECAR_DIR = "services"
SIDECAR_MANIFEST = "manifest.json"
_OUTPUT_TAIL = 2000


def _hooks(task: Any, *, main: bool) -> list[VerifierCollectConfig]:
    # A rollout rebuilt without its task (a regrade's) has no hooks to run.
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    hooks = list(getattr(verifier, "collect", None) or [])
    return [hook for hook in hooks if (hook.service == MAIN_SERVICE) == main]


def sidecar_artifacts(task: Any) -> list[ArtifactConfig]:
    """The task's artifacts read from a compose service other than main."""
    config = getattr(task, "config", None)
    return [
        artifact
        for artifact in getattr(config, "artifacts", None) or []
        if isinstance(artifact, ArtifactConfig)
        and artifact.service not in (None, MAIN_SERVICE)
    ]


def _append_record(rollout_dir: Path, entry: dict[str, Any]) -> None:
    path = rollout_dir / HOOKS_RECORD
    try:
        records = json.loads(path.read_text())
    except (OSError, ValueError):
        records = []
    records.append(entry)
    write_json_atomic(path, records)


def agent_account(rollout: Any) -> str | None:
    """The account a ``main`` hook that names no ``user`` runs as.

    That is the account the agent ran as, which owns the workspace: the
    sandbox user. Root would read the agent's repositories' own git config
    (``core.fsmonitor``, a filter driver) and run what it names as root,
    after the agent's turn; git's ownership check refuses those repositories
    to root, and BenchFlow grants root no trust over them. None, the
    service's own user, where the agent ran as root: with no sandbox user,
    and for the oracle (solve.sh runs as root, or as the task's
    ``[agent].user``).
    """
    config = getattr(rollout, "_config", None)
    user = getattr(config, "sandbox_user", None)
    if not user or getattr(config, "primary_agent", None) == "oracle":
        return None
    return user


async def run_collect_hooks(rollout: Any, *, main: bool) -> bool:
    """Run the task's collect hooks for main (or for the other services), once.

    True when it ran at least one hook: a ``main`` hook can run as the
    agent's account (:func:`agent_account`), so the caller stops that
    account's processes again before anything is captured.
    """
    flag = "_collect_hooks_main_ran" if main else "_collect_hooks_sidecars_ran"
    if getattr(rollout, flag, False) or rollout._env is None:
        return False
    setattr(rollout, flag, True)
    hooks = _hooks(rollout._task, main=main)
    for hook in hooks:
        kwargs: dict[str, Any] = {
            "service": hook.service,
            "timeout_sec": max(1, int(hook.timeout_sec)),
        }
        user = (
            hook.user if hook.user is not None or not main else agent_account(rollout)
        )
        if user is not None:
            kwargs["user"] = user
        entry: dict[str, Any] = {"service": hook.service, "command": hook.command}
        if user is not None:
            entry["user"] = user
        try:
            with as_task_command(rollout._env):
                result = await exec_raw(rollout._env.exec, hook.command, **kwargs)
            entry["return_code"] = getattr(result, "return_code", None)
            entry["stdout_tail"] = (getattr(result, "stdout", "") or "")[-_OUTPUT_TAIL:]
            entry["stderr_tail"] = (getattr(result, "stderr", "") or "")[-_OUTPUT_TAIL:]
            if entry["return_code"]:
                logger.warning(
                    "Collect hook in service %r exited %s: %s",
                    hook.service,
                    entry["return_code"],
                    hook.command[:200],
                )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Collect hook in service %r failed: %s", hook.service, exc)
        _append_record(Path(rollout._require_rollout_dir()), entry)
    return bool(hooks)


def _sha256(rollout_dir: Path, path: Path) -> str:
    """The file's SHA-256, read through the reachable-folder guard: the trial's
    artifacts folder is one the sandbox can write."""
    artifacts = Path(rollout_dir) / "artifacts"
    handle = ReachablePath.rollout(Path(rollout_dir), "artifacts") / path.relative_to(
        artifacts
    )
    digest = hashlib.sha256()
    with handle.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def collect_sidecar_artifacts(rollout: Any) -> list[dict[str, Any]]:
    """Copy each sidecar artifact into the trial and list it; never raises."""
    artifacts = sidecar_artifacts(rollout._task)
    if not artifacts or rollout._env is None:
        return []
    root = Path(rollout._require_rollout_dir()) / "artifacts" / SIDECAR_DIR
    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for artifact in artifacts:
        service = artifact.service
        if service is None:  # sidecar_artifacts lists named services only
            continue
        source = PurePosixPath(artifact.source)
        entry: dict[str, Any] = {
            "service": service,
            "source": str(source),
            "files": [],
        }
        q = shlex.quote
        try:
            staged = await exec_raw(
                rollout._env.exec,
                f"[ -e {q(str(source))} ] || {{ echo missing; exit 3; }}; "
                f'd=$(mktemp -d) && cp -a {q(str(source))} "$d"/ && echo "$d"',
                service=service,
                user="root",
                timeout_sec=120,
            )
            if getattr(staged, "return_code", 1) == 3:
                entry["status"] = "missing"
                entries.append(entry)
                continue
            if getattr(staged, "return_code", 1) != 0:
                raise RuntimeError(
                    (getattr(staged, "stderr", "") or "staging failed")[-300:]
                )
            remote = (getattr(staged, "stdout", "") or "").strip().splitlines()[-1]
            target = root / service / str(source).lstrip("/")
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                await rollout._env.download_dir(remote, tmp, service=service)
                landed = Path(tmp) / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(landed, target)
            await exec_raw(
                rollout._env.exec,
                f"rm -rf {q(remote)}",
                service=service,
                user="root",
                timeout_sec=60,
            )
            files = (
                [target]
                if target.is_file()
                else sorted(p for p in target.rglob("*") if p.is_file())
            )
            for path in files:
                if path.is_symlink():
                    continue
                relative = (
                    PurePosixPath(path.relative_to(target).as_posix())
                    if path != target
                    else None
                )
                entry["files"].append(
                    {
                        "path": str(source / relative) if relative else str(source),
                        "local": path.relative_to(root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": _sha256(rollout._require_rollout_dir(), path),
                        "mode": path.stat().st_mode & 0o777,
                    }
                )
            entry["status"] = "collected"
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Sidecar artifact %s from %r not collected: %s",
                source,
                service,
                exc,
            )
        entries.append(entry)
    write_json_atomic(root / SIDECAR_MANIFEST, {"version": 1, "artifacts": entries})
    return entries


async def install_sidecar_artifacts(
    env: Any, rollout_dir: Path, record: dict[str, Any]
) -> None:
    """Place each listed sidecar file in the verifier sandbox at its source path.

    A file whose bytes no longer match the list, or whose path lies under a
    verifier-owned folder, does not cross; the record says which.
    """
    from benchflow.task.verifier_sandbox import verifier_owned_overlap
    from benchflow.trust.state import IngressKind, admit

    root = Path(rollout_dir) / "artifacts" / SIDECAR_DIR
    try:
        manifest = json.loads((root / SIDECAR_MANIFEST).read_text())
    except (OSError, ValueError):
        return
    crossed, refused = [], []
    for entry in manifest.get("artifacts") or []:
        for item in entry.get("files") or []:
            path = PurePosixPath(item["path"])
            local = root / item["local"]
            if verifier_owned_overlap(str(path)) is not None:
                refused.append({"path": str(path), "reason": "verifier-owned"})
                continue
            if (
                not local.is_file()
                or local.is_symlink()
                or _sha256(Path(rollout_dir), local) != item["sha256"]
            ):
                refused.append(
                    {"path": str(path), "reason": "changed since collection"}
                )
                continue
            await exec_raw(
                env.exec,
                f"mkdir -p {shlex.quote(str(path.parent))}",
                user="root",
                timeout_sec=30,
            )
            await env.upload_file(
                local,
                str(path),
                admission=admit(env, "main", IngressKind.SEPARATE_VERIFIER_TRANSFER),
            )
            await exec_raw(
                env.exec,
                f"chmod {item['mode'] & 0o777:o} {shlex.quote(str(path))}",
                user="root",
                timeout_sec=30,
            )
            crossed.append(str(path))
    record["sidecar_artifacts"] = {"crossed": crossed, "refused": refused}
