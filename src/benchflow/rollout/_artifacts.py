"""Collect ``/logs/artifacts`` and declared task artifacts into the trial folder.

Harbor tasks declare ``artifacts = ["/app/report.xlsx", {source, destination,
exclude}]``; everything an agent leaves in ``/logs/artifacts`` is collected
too. Files land in ``<trial>/artifacts/`` and ``<trial>/artifacts-manifest.json``
records every collection (status, reason) and every file (size, sha256).

Transfers reuse the evidence tarball (``benchflow.review.evidence``): the
sandbox enumerates files without following links, leaves out symlinks that
leave the collected tree (listed in the collection's ``exclusions``), and
stops at the byte and file limits, so a collection is all or nothing. Unsafe
destinations (absolute, ``..``) and destination collisions are refused.
Collection never fails the rollout: every problem is a manifest status, not
an exception. A failed collection whose own files caused it records a
``cause`` (``limits``, ``symlink``, ``clash``), which lets a separate
verifier score it as the solution's result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.review.evidence import (
    EvidenceError,
    EvidenceLimitError,
    capture_workspace,
    python_missing,
)
from benchflow.review.persistence import write_json_atomic
from benchflow.task.artifacts import (
    artifact_destination,
    artifact_spec_issue,
    as_artifact_config,
)
from benchflow.task.config import ArtifactConfig

logger = logging.getLogger(__name__)

LOGS_ARTIFACTS = "/logs/artifacts"
MANIFEST_NAME = "artifacts-manifest.json"
DEFAULT_MAX_BYTES = 1024**3
DEFAULT_MAX_FILES = 10_000

# One exec for every source: where it resolves, whether it exists, whether it
# is a directory with content, and whether the declared path is itself a link.
_PROBE_SCRIPT = r"""
import json, os, pathlib, sys
workspace = pathlib.Path(sys.argv[1])
for raw in json.loads(sys.argv[2]):
    path = pathlib.Path(raw)
    path = path if path.is_absolute() else workspace / path
    info = {"source": str(path), "exists": os.path.lexists(path), "link": path.is_symlink()}
    info["dir"] = path.is_dir() and not info["link"]
    info["empty"] = info["dir"] and not any(path.iterdir())
    print(json.dumps(info))
"""


# The same probe for images without python3: four 0/1 flags per source
# (exists, link, dir, empty), in order.
_SHELL_PROBE = r"""
ws=$1; shift
for s in "$@"; do
  case "$s" in /*) p=$s;; *) p=$ws/$s;; esac
  e=0; l=0; d=0; m=0
  if [ -e "$p" ] || [ -L "$p" ]; then e=1; fi
  if [ -L "$p" ]; then l=1; fi
  if [ $l = 0 ] && [ -d "$p" ]; then
    d=1
    if [ -z "$(ls -A "$p" 2>/dev/null)" ]; then m=1; fi
  fi
  echo "$e $l $d $m"
done
"""


def _shell_probe_result(result: Any, workspace: str, sources: list[str]) -> Any:
    """Turn the shell probe's flags into the Python probe's JSON lines."""
    from benchflow.sandbox.protocol import ExecResult

    lines = []
    for source, flags in zip(sources, (result.stdout or "").splitlines(), strict=False):
        exists, link, is_dir, empty = (flag == "1" for flag in flags.split())
        path = PurePosixPath(source)
        path = path if path.is_absolute() else PurePosixPath(workspace) / path
        lines.append(
            json.dumps(
                {
                    "source": str(path),
                    "exists": exists,
                    "link": link,
                    "dir": is_dir,
                    "empty": empty,
                }
            )
        )
    return ExecResult(0, "\n".join(lines) + "\n", "")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _scan_local(root: Path, index: int) -> list[dict[str, Any]]:
    """Inventory a host folder without following links (mounted /logs)."""
    files: list[dict[str, Any]] = []
    real_root = root.resolve()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                escapes = not path.resolve().is_relative_to(real_root)
                files.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "target": target,
                        "escapes": escapes,
                        "collection": index,
                    }
                )
            elif path.is_file():
                files.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "size": path.stat().st_size,
                        "sha256": _sha256(path),
                        "collection": index,
                    }
                )
    return files


def _entries(
    manifest: Any, prefix: PurePosixPath | None, index: int, single: str | None
):
    """Evidence manifest entries as artifact files under ``prefix``."""
    files: list[dict[str, Any]] = []
    for entry in manifest.entries:
        if entry.kind == "directory":
            continue
        if single is not None:
            relative = prefix
        else:
            relative = (
                prefix / entry.path if prefix is not None else PurePosixPath(entry.path)
            )
        record: dict[str, Any] = {"path": str(relative), "kind": entry.kind}
        if entry.kind == "file":
            record.update(size=entry.size, sha256=entry.sha256)
        else:
            record.update(target=entry.link_target, escapes=False)
        record["collection"] = index
        files.append(record)
    return files


def _over_limits(max_files: int, max_bytes: int) -> str:
    return (
        f"the collected files exceed the collection limits ({max_files} files, "
        f"{max_bytes} bytes in all)"
    )


def _budget(files: list[dict[str, Any]]) -> tuple[int, int]:
    regular = [f for f in files if f["kind"] == "file"]
    return sum(f["size"] for f in regular), len(files)


async def collect_artifacts(
    env: Any,
    *,
    artifacts: Sequence[str | ArtifactConfig],
    workspace: str,
    artifacts_dir: Path,
    manifest_path: Path,
    mounted: bool,
    logs_source: str = LOGS_ARTIFACTS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    excluded_paths: Sequence[str] = (),
    timeout_sec: int = 300,
) -> dict[str, Any]:
    """Collect ``logs_source`` and every declared artifact; write the manifest.

    ``mounted`` means ``logs_source`` is bind-mounted onto ``artifacts_dir``
    (Docker): its files are already on the host and are only inventoried.
    Returns the manifest that was written to ``manifest_path``.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    configs = [as_artifact_config(item) for item in artifacts]
    collections: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    sources = [logs_source, *(c.source for c in configs)]
    probes: list[dict[str, Any]] | None = None
    probe_error = None
    try:
        result = await env.exec(
            shlex.join(
                ["python3", "-c", _PROBE_SCRIPT, workspace, json.dumps(sources)]
            ),
            user="root",
            timeout_sec=60,
        )
        if result.return_code and python_missing(result):
            result = await env.exec(
                shlex.join(["sh", "-c", _SHELL_PROBE, "probe", workspace, *sources]),
                user="root",
                timeout_sec=60,
            )
            if not result.return_code:
                result = _shell_probe_result(result, workspace, sources)
        if result.return_code:
            probe_error = (result.stderr or result.stdout or "")[-500:]
        else:
            probes = [json.loads(line) for line in (result.stdout or "").splitlines()]
            if len(probes) != len(sources):
                probes, probe_error = None, "probe returned an incomplete listing"
    except Exception as exc:  # the sandbox may already be gone
        probe_error = f"{type(exc).__name__}: {exc}"

    def used() -> tuple[int, int]:
        return _budget(files)

    # 1. /logs/artifacts, collected into the root of artifacts/.
    record: dict[str, Any] = {
        "kind": "logs",
        "source": logs_source,
        "destination": ".",
    }
    if mounted:
        scanned = _scan_local(artifacts_dir, 0)
        size, count = _budget(scanned)
        files.extend(scanned)
        if size > max_bytes or count > max_files:
            record.update(
                status="over_limit",
                cause="limits",
                reason="bind-mounted files exceed the collection limits; they stay "
                "in place but count against no further collection",
            )
        else:
            record["status"] = "collected" if scanned else "empty"
    elif probes is None:
        record.update(status="error", reason=f"artifact probe failed: {probe_error}")
    elif not probes[0]["exists"]:
        record["status"] = "missing"
    elif probes[0]["empty"]:
        record["status"] = "empty"
    else:
        try:
            captured, excluded = await _capture(
                env,
                logs_source,
                artifacts_dir,
                None,
                index=0,
                exclude=(),
                max_bytes=max_bytes,
                max_files=max_files,
                excluded_paths=excluded_paths,
                timeout_sec=timeout_sec,
                single=None,
            )
            files.extend(captured)
            record["status"] = "collected"
            if excluded:
                record["exclusions"] = excluded
        except (EvidenceError, OSError, ValueError) as exc:
            record.update(status="error", reason=str(exc)[-500:])
            if isinstance(exc, EvidenceLimitError):
                record.update(cause="limits", reason=_over_limits(max_files, max_bytes))
    collections.append(record)

    # 2. Declared artifacts, each at artifacts/<destination or basename>.
    for offset, config in enumerate(configs, start=1):
        record = {
            "kind": "declared",
            "source": config.source,
            "destination": config.destination,
            "exclude": list(config.exclude),
        }
        collections.append(record)
        problem = artifact_spec_issue(config)
        if problem is not None:
            record.update(status="refused", reason=problem)
            continue
        destination = artifact_destination(config)
        record["destination"] = str(destination)
        target = artifacts_dir / destination
        if probes is None:
            record.update(
                status="error", reason=f"artifact probe failed: {probe_error}"
            )
            continue
        probe = probes[offset]
        record["source"] = probe["source"]
        if not probe["exists"]:
            record["status"] = "missing"
            continue
        if probe["link"]:
            record.update(
                status="refused",
                cause="symlink",
                reason="declared source is a symlink; not followed",
            )
            continue
        if target.exists() or target.is_symlink():
            record.update(
                status="refused",
                reason=f"destination {destination} already exists in artifacts/",
            )
            name = str(destination)
            if any(
                f["collection"] == 0
                and (f["path"] == name or f["path"].startswith(name + "/"))
                for f in files
            ):
                # A file the agent left in /logs/artifacts took the place.
                record["cause"] = "clash"
            continue
        size, count = used()
        if size >= max_bytes or count >= max_files:
            record.update(
                status="error",
                cause="limits",
                reason="collection limits already reached",
            )
            continue
        try:
            captured, excluded = await _capture(
                env,
                probe["source"],
                target,
                destination,
                index=offset,
                exclude=config.exclude,
                max_bytes=max_bytes - size,
                max_files=max_files - count,
                excluded_paths=excluded_paths,
                timeout_sec=timeout_sec,
                single=None if probe["dir"] else PurePosixPath(probe["source"]).name,
            )
            files.extend(captured)
            record["status"] = "collected"
            if excluded:
                record["exclusions"] = excluded
        except (EvidenceError, OSError, ValueError) as exc:
            record.update(status="error", reason=str(exc)[-500:])
            if isinstance(exc, EvidenceLimitError):
                record.update(cause="limits", reason=_over_limits(max_files, max_bytes))

    total_bytes, _ = used()
    manifest = {
        "version": 1,
        "limits": {"max_bytes": max_bytes, "max_files": max_files},
        "collections": collections,
        "files": files,
        "total_bytes": total_bytes,
        "total_files": sum(1 for f in files if f["kind"] == "file"),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


async def _capture(
    env: Any,
    source: str,
    target: Path,
    prefix: PurePosixPath | None,
    *,
    index: int,
    exclude: Sequence[str],
    max_bytes: int,
    max_files: int,
    excluded_paths: Sequence[str],
    timeout_sec: int,
    single: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tar ``source`` out of the sandbox, then publish it at ``target``.

    ``target`` is the folder to fill (logs: the existing artifacts/ root) or
    the path to create (declared artifact). Nothing is published on failure.
    Returns the collected files and the capture's exclusions.
    """
    parent = target.parent if prefix is not None else target
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".artifact-", dir=parent) as temp:
        bundle = Path(temp) / "bundle"
        manifest = await capture_workspace(
            env,
            source,
            bundle,
            max_bytes=max(max_bytes, 1),
            max_entries=max(max_files, 1),
            timeout_sec=timeout_sec,
            exclude=exclude,
            excluded_paths=excluded_paths,
        )
        tree = bundle / "workspace"
        if prefix is None:
            children = sorted(tree.iterdir())
            clashes = [c.name for c in children if (target / c.name).exists()]
            if clashes:
                raise EvidenceError(f"artifacts/ already holds {clashes[:3]}")
            for child in children:
                shutil.move(str(child), target / child.name)
        elif single is not None:
            shutil.move(str(tree / single), target)
        else:
            tree.rename(target)
    exclusions = [x.model_dump(exclude_none=True) for x in manifest.exclusions]
    return _entries(manifest, prefix, index, single), exclusions


async def collect_rollout_artifacts(rollout: Any) -> None:
    """Collect a finished rollout's artifacts before verifier hardening."""
    if getattr(rollout, "_branch_child_active", False) or rollout._env is None:
        return
    rollout._artifacts_collected = True
    from benchflow.agents.credentials import credential_evidence_overrides

    cfg = rollout._config
    paths = rollout._rollout_paths
    try:
        excluded = credential_evidence_overrides(
            rollout._agent_env or {},
            workspace=rollout._agent_cwd,
            cred_home=f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root",
        )
        await collect_artifacts(
            rollout._env,
            artifacts=rollout._task.config.artifacts,
            workspace=rollout._agent_cwd,
            artifacts_dir=paths.artifacts_dir,
            manifest_path=paths.rollout_dir / MANIFEST_NAME,
            mounted=bool(getattr(rollout._env, "is_mounted", False)),
            excluded_paths=excluded,
        )
    except Exception:
        logger.exception("Artifact collection failed")


# How long cleanup may spend collecting an unverified rollout's artifacts.
UNVERIFIED_COLLECTION_TIMEOUT_SEC = 180.0


async def collect_unverified_rollout_artifacts(rollout: Any) -> None:
    """Collect the artifacts of a rollout that never reached its verifier.

    The verifier phase collects artifacts; an agent error (or any failure
    before verification) skipped it, so the trial kept nothing the agent
    left, such as its session log or partial outputs. Cleanup calls this
    before the sandbox stops. It is bounded, and skipped when the sandbox is
    known to be unreachable.
    """
    if getattr(rollout, "_artifacts_collected", False):
        return
    if rollout._env is None or getattr(rollout, "_rollout_paths", None) is None:
        return
    transport = getattr(getattr(rollout, "_diagnostics", None), "transport_closed", None)
    if transport is not None and getattr(transport, "sandbox_reachable", None) is False:
        return
    try:
        await asyncio.wait_for(
            collect_rollout_artifacts(rollout),
            timeout=UNVERIFIED_COLLECTION_TIMEOUT_SEC,
        )
    except TimeoutError:
        logger.warning(
            "Collecting the artifacts of an unverified rollout timed out after %.0f s",
            UNVERIFIED_COLLECTION_TIMEOUT_SEC,
        )
