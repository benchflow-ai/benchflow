"""Collect ``/logs/artifacts`` and declared task artifacts into the trial folder.

task.toml tasks declare ``artifacts = ["/app/report.xlsx", {source, destination,
exclude}]``; everything an agent leaves in ``/logs/artifacts`` is collected
too. Files land in ``<trial>/artifacts/`` and ``<trial>/artifacts-manifest.json``
records every collection (status, reason) and every file (size, sha256).

Transfers reuse the evidence tarball (``benchflow.review.evidence``): the
sandbox enumerates files without following links, admits only unchanged
plain relative links (others are listed in the collection's ``exclusions``), and
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
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow._root_python import root_python_command
from benchflow._utils.disk_space import is_disk_full
from benchflow._utils.reachable_write import (
    ReachablePath,
    SnapshotLimitError,
    private_folder,
    snapshot_file,
)
from benchflow.review.evidence import (
    EvidenceError,
    EvidenceLimitError,
    capture_workspace,
    python_missing,
)
from benchflow.review.persistence import write_json_atomic
from benchflow.rollout._artifact_links import plain_link_targets
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
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

# The least time a copy out of the sandbox gets.
COPY_OUT_MIN_DEADLINE_SEC = 180
# The slowest a copy may run and still finish: one second for each MiB of its
# cap. Measured on Daytona (Rust round P1.6), one 1 GiB copy took 71 to 84 s,
# but four at once ran at about 3.8 MB/s each and three passed a fixed 180 s,
# so an honest trial lost its collection (and, with a separate verifier, its
# grade). At this rate the slowest of them has about four times what it took.
COPY_OUT_BYTES_PER_SEC = 1024 * 1024


def copy_out_deadline(max_bytes: int) -> int:
    """Seconds a copy of up to ``max_bytes`` gets out of the sandbox.

    The larger of :data:`COPY_OUT_MIN_DEADLINE_SEC` and a second per MiB of
    the cap (1,024 s for the default 1 GiB cap), as Rust 0.8.2 does. It stays
    bounded, so code in the sandbox that never lets a copy finish still ends
    it, at the longer deadline.
    """
    return max(COPY_OUT_MIN_DEADLINE_SEC, max(max_bytes, 0) // COPY_OUT_BYTES_PER_SEC)


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


def _sha256(path: ReachablePath, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    try:
        with (
            snapshot_file(path, max_bytes=max_bytes) as snapshot,
            snapshot.open("rb") as stream,
        ):
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except SnapshotLimitError as exc:
        raise EvidenceLimitError(
            "artifact grew beyond the collection byte limit"
        ) from exc


def _scan_local(
    root: ReachablePath,
    index: int,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> list[dict[str, Any]]:
    """Admit the bounded inventory before hashing through no-follow handles."""
    files: list[dict[str, Any]] = []
    size = 0
    for count, path in enumerate(root.rglob("*"), start=1):
        if count > max_files:
            raise EvidenceLimitError("artifact entry limit exceeded")
        relative = path.relative_to(root).as_posix()
        info = path.stat()
        if stat.S_ISLNK(info.st_mode):
            files.append(
                dict(
                    path=relative,
                    kind="symlink",
                    target=path.readlink_text(),
                    collection=index,
                )
            )
        elif stat.S_ISDIR(info.st_mode):
            files.append(dict(path=relative, kind="directory", collection=index))
        elif stat.S_ISREG(info.st_mode):
            size += info.st_size
            if size > max_bytes:
                raise EvidenceLimitError("artifact byte limit exceeded")
            files.append(
                dict(path=relative, kind="file", size=info.st_size, collection=index)
            )
    links = {f["path"]: f["target"] for f in files if f["kind"] == "symlink"}
    _, excluded = plain_link_targets(links)
    remaining = max_bytes
    for record in files:
        if record["kind"] == "symlink":
            record["escapes"] = record["path"] in excluded
        elif record["kind"] == "file":
            path = root / record["path"]
            record["sha256"] = _sha256(path, min(record["size"], remaining))
            if path.stat().st_size != record["size"]:
                raise EvidenceError("artifact changed size during collection")
            remaining -= record["size"]
    return files


def _publish(source: Path, target: ReachablePath) -> None:
    if target.exists():
        raise EvidenceError(f"artifact destination {target} already exists")
    target.publish(source)


def _entries(
    manifest: Any, prefix: PurePosixPath | None, index: int, single: str | None
):
    """Evidence manifest entries as artifact files under ``prefix``."""
    files: list[dict[str, Any]] = []
    for entry in manifest.entries:
        if single is not None:
            relative = prefix
        else:
            relative = (
                prefix / entry.path if prefix is not None else PurePosixPath(entry.path)
            )
        record: dict[str, Any] = {"path": str(relative), "kind": entry.kind}
        if entry.kind == "file":
            record.update(size=entry.size, sha256=entry.sha256)
        elif entry.kind == "symlink":
            record.update(target=entry.original_link_target, escapes=False)
        record["collection"] = index
        files.append(record)
    return files


def _link_refusal(path: str, target: str | None) -> dict[str, Any]:
    return {"path": path, "reason": "symlink_refused", "link_target": target}


def _filter_captured_links(
    manifest: Any, tree: Path
) -> tuple[set[str], list[dict[str, Any]]]:
    """Refuse lossy or nonplain links before publishing the private capture.

    Evidence extraction may have rewritten a target. Only its original text
    can authorize delivery; absent provenance is a refusal too. Include omitted
    links in the check so an otherwise plain path cannot pass through one.
    """
    links = {
        e.path: e.original_link_target for e in manifest.entries if e.kind == "symlink"
    }
    exclusions = []
    for item in manifest.exclusions:
        if item.reason != "symlink_escape":
            exclusions.append(item.model_dump(exclude_none=True))
            continue
        name = str(PurePosixPath(item.original_path).relative_to(manifest.workspace))
        links[name] = item.link_target
        exclusions.append(
            dict(
                _link_refusal(name, item.link_target), original_path=item.original_path
            )
        )
    admitted, _ = plain_link_targets({name: text or "" for name, text in links.items()})
    refused = set()
    for entry in manifest.entries:
        if entry.kind != "symlink":
            continue
        if (
            entry.path not in admitted
            or entry.link_target != admitted[entry.path]
            or os.readlink(tree / entry.path) != admitted[entry.path]
        ):
            # The extractor already forbids symlink parents. This unlinks only
            # the link itself in our private staging tree, never its target.
            (tree / entry.path).unlink()
            refused.add(entry.path)
            exclusions.append(
                dict(
                    _link_refusal(entry.path, entry.original_link_target),
                    original_path=entry.original_path,
                )
            )
    return refused, exclusions


def _over_limits(max_files: int, max_bytes: int) -> str:
    return (
        f"the collected files exceed the collection limits ({max_files} files, "
        f"{max_bytes} bytes in all)"
    )


def _failure(exc: Exception) -> str:
    """A failed copy's reason for the manifest.

    Every backend raises ``RuntimeError("Command timed out after N seconds")``
    for a copy past its deadline, and a gone sandbox raises its own errors.
    Those escaped the collection, so no manifest was written: the copies that
    had finished went unrecorded, and a separate verifier, finding no failed
    collection, graded without ``/logs/artifacts`` instead of ending the
    trial as an error. Every failure is now the collection's ``error``.
    """
    known = isinstance(exc, EvidenceError | OSError | ValueError)
    text = str(exc) if known else f"{type(exc).__name__}: {exc}"
    return text[-500:]


def _budget(files: list[dict[str, Any]]) -> tuple[int, int]:
    regular = [f for f in files if f["kind"] == "file"]
    return sum(f["size"] for f in regular), len(files)


async def collect_artifacts(
    env: Any,
    *,
    artifacts: Sequence[str | ArtifactConfig],
    workspace: str,
    artifacts_dir: ReachablePath,
    manifest_path: Path,
    mounted: bool,
    logs_source: str = LOGS_ARTIFACTS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    excluded_paths: Sequence[str] = (),
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Collect ``logs_source`` and every declared artifact; write the manifest.

    ``mounted`` means ``logs_source`` is bind-mounted onto ``artifacts_dir``
    (Docker): its files are already on the host and are only inventoried.
    Each copy gets :func:`copy_out_deadline` of the bytes it may still bring
    (the whole cap for the first); ``timeout_sec``, when given, is a floor.
    Returns the manifest that was written to ``manifest_path``.
    """

    def deadline(cap: int) -> int:
        return max(timeout_sec or 0, copy_out_deadline(cap))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    configs = [as_artifact_config(item) for item in artifacts]
    collections: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    sources = [logs_source, *(c.source for c in configs)]
    probes: list[dict[str, Any]] | None = None
    probe_error = None
    try:
        result = await exec_raw(
            env.exec,
            await root_python_command(
                env, _PROBE_SCRIPT, workspace, json.dumps(sources)
            ),
            user="root",
            timeout_sec=60,
        )
        if result.return_code and python_missing(result):
            result = await exec_raw(
                env.exec,
                shlex.join(["sh", "-c", _SHELL_PROBE, "probe", workspace, *sources]),
                user="root",
                timeout_sec=60,
            )
            if not result.return_code:
                result = _shell_probe_result(result, workspace, sources)
        if result.return_code:
            probe_error = (
                diagnostic_result(result).stderr.render()
                or diagnostic_result(result).stdout.render()
                or ""
            )[-500:]
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
        try:
            scanned = await asyncio.to_thread(
                _scan_local,
                artifacts_dir,
                0,
                max_bytes=max_bytes,
                max_files=max_files,
            )
            files.extend(scanned)
            record["status"] = "collected" if scanned else "empty"
            invalid = [f for f in scanned if f["kind"] == "symlink" and f["escapes"]]
            if invalid:
                record["exclusions"] = [
                    _link_refusal(f["path"], f["target"]) for f in invalid
                ]
        except Exception as exc:
            record.update(status="error", reason=_failure(exc))
            if isinstance(exc, EvidenceLimitError):
                record.update(
                    status="over_limit",
                    cause="limits",
                    reason=_over_limits(max_files, max_bytes),
                )
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
                timeout_sec=deadline(max_bytes),
                single=None,
                root=artifacts_dir,
            )
            files.extend(captured)
            record["status"] = "collected"
            if excluded:
                record["exclusions"] = excluded
        except Exception as exc:  # a copy past its deadline raises RuntimeError
            record.update(status="error", reason=_failure(exc))
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
        if mounted and collections[0]["status"] in {"error", "over_limit"}:
            record.update(
                status="error", reason="logs artifact inventory did not complete"
            )
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
        try:
            exists = target.exists()
        except OSError as exc:
            record.update(status="refused", cause="symlink", reason=_failure(exc))
            continue
        if exists:
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
                timeout_sec=deadline(max_bytes - size),
                single=None if probe["dir"] else PurePosixPath(probe["source"]).name,
                root=artifacts_dir,
            )
            files.extend(captured)
            record["status"] = "collected"
            if excluded:
                record["exclusions"] = excluded
        except Exception as exc:  # a copy past its deadline raises RuntimeError
            record.update(status="error", reason=_failure(exc))
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
    target: ReachablePath,
    prefix: PurePosixPath | None,
    *,
    index: int,
    exclude: Sequence[str],
    max_bytes: int,
    max_files: int,
    excluded_paths: Sequence[str],
    timeout_sec: int,
    single: str | None,
    root: ReachablePath,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tar ``source`` out of the sandbox, then publish it at ``target``.

    ``target`` is the folder to fill (logs: the existing artifacts/ root) or
    the path to create (declared artifact). Nothing is published on failure.
    Returns the collected files and the capture's exclusions.

    ``root`` (artifacts/) is /logs/artifacts in a Docker sandbox, and an agent
    process can still run here: the capture lands in private custody and is
    placed through held folders, never written by a path under ``root`` (R4-1).
    """
    root.mkdir(parents=True, exist_ok=True)
    with private_folder(root) as temp:
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
        refused, exclusions = _filter_captured_links(manifest, tree)
        files = _entries(manifest, prefix, index, single)
        for entry, record in zip(manifest.entries, files, strict=True):
            if entry.path in refused:
                record["escapes"] = True
        if prefix is None:
            children = sorted(tree.iterdir())
            clashes = [c.name for c in children if (target / c.name).exists()]
            if clashes:
                raise EvidenceError(f"artifacts/ already holds {clashes[:3]}")
            for child in children:
                _publish(child, target / child.name)
        elif single is not None:
            _publish(tree / single, target)
        else:
            _publish(tree, target)
    return files, exclusions


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
    except Exception as exc:
        if is_disk_full(exc):
            # A full disk is the run's to report, once and with no traceback
            # (DX-a); an error manifest could not be written there either. The
            # trajectory published beside the collection (PERF-G) can be what
            # fills it first (merge wave batch 35).
            raise
        logger.exception("Artifact collection failed")
        write_json_atomic(
            paths.rollout_dir / MANIFEST_NAME,
            {
                "version": 1,
                "collections": [
                    {
                        "kind": "logs",
                        "source": LOGS_ARTIFACTS,
                        "destination": ".",
                        "status": "error",
                        "reason": _failure(exc),
                    }
                ],
                "files": [],
                "total_bytes": 0,
                "total_files": 0,
            },
        )


# How long cleanup may spend collecting an unverified rollout's artifacts:
# the copy deadline of the whole cap (1,024 s), since every copy of one
# collection shares that cap, plus a minute for the probe that comes first.
# It was a fixed 180 s, which a 1 GiB copy on a busy Daytona runner passes.
UNVERIFIED_COLLECTION_TIMEOUT_SEC = float(copy_out_deadline(DEFAULT_MAX_BYTES) + 60)


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
    transport = getattr(
        getattr(rollout, "_diagnostics", None), "transport_closed", None
    )
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
