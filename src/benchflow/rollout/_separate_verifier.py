"""Run a task's verifier in its own sandbox (task.toml ``environment_mode = "separate"``).

A task opts in with ``[verifier] environment_mode = "separate"`` (BenchFlow:
``sandbox_mode``) or by declaring ``[verifier.environment]``
(``[verifier.sandbox]``); ``--separate-verifier`` (or ``--integrity strict``)
forces it for a run without editing the task. The verifier then never runs
where the agent ran:

1. After the agent stops, the rollout freezes the workspace into
   ``evidence/`` (with the task's declared artifacts) and collects
   ``/logs/artifacts`` into ``artifacts/`` — the same captures regrade uses.
   Then it stops the agent's sandbox (``Rollout.verify``), so nothing in
   it can change those bytes or write into a mount while they cross.
2. The host packs exactly those bytes, hash-checked against their manifests,
   into one tar. Nothing else from the agent's sandbox is read: files planted
   in ``/tests``, ``/logs/verifier``, site-packages or anywhere outside the
   workspace and the declared artifacts do not exist in the verifier sandbox.
   A ``/logs/artifacts`` file is read once, without following links, into a
   private copy that is both hashed and packed. A workspace or declared
   artifact at, under or above a verifier-owned path (``VERIFIER_OWNED_PATHS``)
   never crosses.
3. A fresh sandbox starts from the verifier image (see
   :func:`plan_verifier_image`); it needs only task files, so on Docker it
   starts while step 1 runs (:class:`VerifierSandboxRun`). The tar is unpacked at the original absolute
   paths, and the normal verifier path runs there: hardening (which still
   removes workspace ``conftest.py`` / ``.pth`` / ``sitecustomize.py``),
   ``tests/`` upload, ``test.sh``, reward parsing.
4. Its outputs replace the trial's ``verifier/``: the folder the agent's
   sandbox could write (on Docker its live ``/logs/verifier`` mount) is moved
   to ``verifier-sandbox/agent-verifier-dir/`` and listed, never merged into.

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

Per-phase timing lands in the rollout's ``timing`` (``verifier_pack``,
``verifier_sandbox_setup``, ``verifier_transfer``, ``verifier``,
``verifier_sandbox_teardown``, ``verifier_sandbox_total``, and in the overlap
order ``verifier_sandbox_ready_wait`` and ``verifier_sandbox_overlap``) and
``verifier-sandbox/verifier-sandbox.json`` records the image source, sandbox
id, transfer inventory, the custody of what crossed (the tar's sha256, its
members per root, whether the sandbox checked the sha256, the trajectory),
what was moved out of ``verifier/``, what was published there, status and
sandbox-seconds (the verifier sandbox's life, from its start to its
deletion). Job budgets add ``verifier_sandbox_overlap`` to the trial's wall
clock: the seconds both sandboxes were billed at once. A record without it
(the serial order, older trials) adds ``verifier_sandbox_total``.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import errno
import hashlib
import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from benchflow._root_python import root_python_command
from benchflow._runtime_tree import RUNTIME_ROOT
from benchflow._utils import timing_spans
from benchflow._utils.reachable_write import (
    ReachablePath,
    ReachableWriteError,
    destination,
    note_reachable_move,
    private_folder,
    snapshot_file,
    snapshot_tree,
)
from benchflow._utils.text import describe_exception
from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.review.evidence import (
    WORKSPACE_MAX_BYTES,
    WORKSPACE_MAX_ENTRIES,
    python_missing,
)
from benchflow.rollout._artifact_links import plain_link_targets
from benchflow.rollout._collect_hooks import install_sidecar_artifacts
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier_errors import (
    verifier_failure_reason,
    verifier_infrastructure_failure,
)
from benchflow.task.verifier_sandbox import (
    SeparateVerifierError,
    VerifierImage,
    plan_verifier_image,
    separate_verifier_requested,
    verifier_context_services,
    verifier_image_issue,
    verifier_owned_overlap,
)
from benchflow.trust.state import IngressKind, admit

logger = logging.getLogger(__name__)

__all__ = [
    "ERROR_PREFIX",
    "VERIFIER_PREPARE_ENV",
    "SeparateVerifierError",
    "SolutionTransferRefused",
    "VerifierImage",
    "VerifierSandboxRun",
    "build_transfer_payload",
    "measure_outputs",
    "plan_verifier_image",
    "record_pristine_outputs",
    "read_record",
    "run_separate_verifier",
    "scan_outside_workspace_writes",
    "separate_verifier_requested",
    "serial_order_requested",
    "warmup_requested",
    "verifier_image_issue",
]

ERROR_PREFIX = "separate verifier"
RECORD_DIR = "verifier-sandbox"
RECORD_FILE = "verifier-sandbox.json"
# Where publish moves whatever the agent's run left in the trial's verifier/.
AGENT_VERIFIER_DIR = "agent-verifier-dir"
# The install script's report on the in-sandbox checksum.
_SHA256_CHECKED = "benchflow-transfer-sha256=checked"
_SHA256_SKIPPED = "benchflow-transfer-sha256=skipped"
# Listings in the record keep this many entries, plus the full count.
_LIST_LIMIT = 1000
# A listing stops counting here, and does not descend deeper than this.
_WALK_LIMIT = 100_000
_WALK_DEPTH = 64
# A listed file over this size gets its size but no sha256.
_HASH_LIMIT = 64 * 1024**2
# A listing reads at most this many bytes in all to hash the files it keeps
# (the agent can fill its /logs/verifier with large sparse files); files past
# it get their size only.
_HASH_BUDGET = 256 * 1024**2
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
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
# Where the verifier's programs, libraries and settings live. Agent bytes land
# here only as the one file a task declares: a folder the agent pointed here
# (``/out`` -> ``/usr``) would replace the tools test.sh trusts.
_SYSTEM_DIRS = (
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/proc",
    "/run",
    "/sbin",
    "/sys",
    "/usr/bin",
    "/usr/lib",
    "/usr/lib32",
    "/usr/lib64",
    "/usr/libexec",
    "/usr/libx32",
    "/usr/local/bin",
    "/usr/local/lib",
    "/usr/local/sbin",
    "/usr/sbin",
)
_STOP_TIMEOUT_SEC = 120


class SolutionTransferRefused(SeparateVerifierError):
    """The solution's own files keep its outputs from the verifier; scored 0."""


# --- before the agent: the clean control -------------------------------------

# For each path: whether it exists, whether it is a symlink, and the entries
# and regular-file bytes under it (following only a top-level link, as capture
# does for the workspace). Excluded paths are counted too, so the count never
# undershoots what capture would have kept. Also where the path resolves, as
# capture resolves it: the clean control for a transfer root that the agent
# pointed into a verifier-owned folder.
_PRISTINE_PROBE = r"""
import json, os, stat, sys
out = {}
for path in json.loads(sys.argv[1]):
    info = {"exists": os.path.lexists(path), "link": os.path.islink(path), "dir": os.path.isdir(path), "entries": 0, "bytes": 0}
    info["resolved"] = os.path.realpath(path)
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

# The same for images without python3: "exists link entries bytes dir
# resolved" per path, with "?" for a value this image has no tool for.
_PRISTINE_SHELL_PROBE = r"""
for p in "$@"; do
  e=0; l=0; n=0; b=0; d=0
  # -m (GNU) resolves a path whose folders do not exist yet, as realpath
  # does; busybox has only -f.
  r=$(readlink -m -- "$p" 2>/dev/null || readlink -f -- "$p" 2>/dev/null) || r='?'
  [ -n "$r" ] || r='?'
  if [ -e "$p" ] || [ -L "$p" ]; then e=1; fi
  if [ -L "$p" ]; then l=1; fi
  if [ -d "$p" ]; then
    d=1
    if command -v find >/dev/null 2>&1 && command -v stat >/dev/null 2>&1; then
      n=$(find "$p/" -mindepth 1 | wc -l | tr -d ' ')
      b=$(find "$p/" -type f -exec stat -c %s {} + | awk '{s += $1} END {printf "%.0f", s}')
    else
      n='?'; b='?'
    fi
  elif [ -f "$p" ]; then
    n=1; b=$(stat -L -c %s "$p" 2>/dev/null || echo '?')
  fi
  echo "$e $l $n $b $d $r"
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
        # Whether it was a folder (or a link to one); None when not measured.
        "dir": info["dir"] in (True, "1") if "dir" in info else None,
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
    result = await exec_raw(
        env.exec,
        await root_python_command(env, _PRISTINE_PROBE, json.dumps(paths)),
        user="root",
        timeout_sec=timeout_sec,
    )
    if result.return_code and python_missing(result):
        result = await exec_raw(
            env.exec,
            shlex.join(["sh", "-c", _PRISTINE_SHELL_PROBE, "probe", *paths]),
            user="root",
            timeout_sec=timeout_sec,
        )
        if result.return_code:
            diagnostic = diagnostic_result(result)
            raise RuntimeError(
                (diagnostic.stderr.render() or diagnostic.stdout.render() or "")[-500:]
            )
        lines = (result.stdout or "").splitlines()
        if len(lines) != len(paths):
            raise RuntimeError("the probe returned an incomplete listing")
        raw = {
            # A short line leaves its counts unknown; the resolved path is the
            # rest of the line, spaces and all.
            path: dict(
                zip(
                    ("exists", "link", "entries", "bytes", "dir", "resolved"),
                    line.split(None, 5),
                    strict=False,
                )
            )
            for path, line in zip(paths, lines, strict=True)
        }
    elif result.return_code:
        diagnostic = diagnostic_result(result)
        raise RuntimeError(
            (diagnostic.stderr.render() or diagnostic.stdout.render() or "")[-500:]
        )
    else:
        raw = json.loads(result.stdout or "")
    resolved = {}
    for path in paths:
        value = raw[path].get("resolved")
        resolved[path] = (
            value if isinstance(value, str) and value.startswith("/") else None
        )
    return {
        "workspace": workspace,
        "logs": logs,
        "paths": {path: _measured(raw[path]) for path in paths},
        "resolved": resolved,
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


def _overlap_error(path: str, owned: str) -> str:
    return f"{path} overlaps verifier-owned {owned}"


def declared_transfer_roots(
    workspace: str | None, artifacts: Sequence[Any]
) -> list[str]:
    """The sandbox paths the task and the run say the transfer comes from.

    The workspace (a declared ``[sandbox].workdir`` or the image's WORKDIR)
    and each declared artifact, a relative one joined to the workspace.
    """
    from benchflow.task.artifacts import as_artifact_config

    roots = [workspace] if workspace else []
    for item in artifacts:
        source = as_artifact_config(item).source
        if not source.startswith("/"):
            if not workspace:
                continue
            source = str(PurePosixPath(workspace) / source)
        roots.append(source)
    return roots


def check_declared_transfer(workspace: str | None, artifacts: Sequence[Any]) -> None:
    """Refuse a workspace or declared artifact in a verifier-owned folder.

    The launch gate refuses the ones task.toml declares; this catches a run
    overlay's artifacts, SDK callers that skip the gate and a workspace taken
    from the image's WORKDIR. These paths come from the task and the run,
    never from the agent, so the trial is unscored, not scored 0.
    """
    for root in declared_transfer_roots(workspace, artifacts):
        owned = verifier_owned_overlap(root)
        if owned is not None:
            raise SeparateVerifierError(_overlap_error(root, owned))


def _owned_member(manifest: Any) -> tuple[str, str] | None:
    """The first member a bundle packs into a verifier-owned path, if any.

    Every member lies under the bundle's capture root, so a root that is
    neither ``/`` nor at, under or above a verifier-owned path holds none;
    only the others are walked. A single file's root is its folder, which may
    be ``/`` or ``/logs`` for an ordinary file, so the members are what is
    judged, never the root alone.
    """
    root = PurePosixPath(manifest.workspace)
    if str(root) != "/" and verifier_owned_overlap(str(root)) is None:
        return None
    for entry in manifest.entries:
        path = str(root / _safe_relative(entry.path))
        owned = verifier_owned_overlap(path)
        if owned is not None:
            return path, owned
    return None


def _captured_where_discovered(source: str, manifest: Any) -> bool:
    """Whether capture read a declared artifact where discovery found it.

    Discovery resolves the artifact's path, then capture resolves it again,
    so a link swapped in between (``/out`` pointing at ``/tests``) moves what
    is read. A folder is captured at its own path; a file at its folder, as
    that one file (none when the task excludes it).
    """
    path, root = PurePosixPath(source), PurePosixPath(manifest.workspace)
    if root == path:
        return True
    return root == path.parent and all(
        entry.kind == "file" and entry.path == path.name for entry in manifest.entries
    )


def _system_overlap(path: str) -> str | None:
    """The system folder ``path`` equals or lies under, if any."""
    normal = PurePosixPath(posixpath.normpath(path))
    for system in _SYSTEM_DIRS:
        folder = PurePosixPath(system)
        if normal == folder or folder in normal.parents:
            return system
    return None


def _system_landing(root: str, manifest: Any) -> tuple[str, str] | None:
    """The first place a folder's bundle lands in a system folder, if any.

    The root itself, or (for a root above one, such as ``/usr``) the first
    member under one. Every member lies under the root, so other roots hold
    none.
    """
    system = _system_overlap(root)
    if system is not None:
        return root, system
    base = PurePosixPath(root)
    if not any(base in PurePosixPath(s).parents for s in _SYSTEM_DIRS):
        return None
    for entry in manifest.entries:
        path = str(base / _safe_relative(entry.path))
        system = _system_overlap(path)
        if system is not None:
            return path, system
    return None


def _check_resolved_roots(
    manifest: Any,
    subs: dict[str, Any],
    pristine: dict[str, Any] | None,
    artifacts: Sequence[Any] | None,
    workspace: str | None = None,
) -> None:
    """Refuse a capture that lands anywhere its declaration does not allow.

    Capture resolves the workspace and each declared artifact in the agent's
    sandbox, so a symlink the agent left on the way (``/app/out`` pointing
    at ``/tests``, or an ancestor ``/out`` pointing at ``/usr``) moves what
    crosses there, and a link swapped in after discovery moves it again
    before capture reads it. Each captured artifact's bundle (``subs``, by
    bundle path) is judged by where its members are packed, not by the path
    discovery recorded:

    - in a verifier-owned path: 0 when the declared path resolved elsewhere
      before the agent ran, as for a declared artifact left as a symlink;
      otherwise (the task's own image, or no clean control) unscored;
    - captured somewhere other than where discovery found it: unscored, a
      change during capture;
    - anywhere but the declared path (``workspace``, or the artifact joined
      to it) or where that path resolved before the agent ran: 0 when the
      clean control resolved it, else unscored;
    - a folder landing in, or packing members into, a system folder
      (``_SYSTEM_DIRS``), or a file landing in one other than the exact file
      the task declared: unscored, the image's or task's doing, since any
      redirect the agent made is refused above.

    ``artifacts`` are the declarations in capture order; None (a caller that
    has none) checks only the verifier-owned paths and the discovery split.
    ``workspace`` defaults to the clean control's.
    """
    from benchflow.task.artifacts import as_artifact_config

    declared_root = workspace or (pristine["workspace"] if pristine else None)
    resolved = (pristine or {}).get("resolved") or {}

    def refuse(key: str | None, landed: str, owned: str) -> NoReturn:
        before = resolved.get(key) if key else None
        if isinstance(before, str) and verifier_owned_overlap(before) is None:
            raise SolutionTransferRefused(
                f"{key} resolves to {landed}, which overlaps verifier-owned "
                f"{owned}; before the agent ran it resolved to {before}"
            )
        raise SeparateVerifierError(_overlap_error(landed, owned))

    def authorize(label: str, key: str, landed: str, folder: Any | None) -> None:
        # ``folder`` is the bundle's manifest when the landing is a folder.
        declared = posixpath.normpath(key)
        before = resolved.get(key)
        if landed not in (declared, before):
            if isinstance(before, str):
                raise SolutionTransferRefused(
                    f"{label} {key} resolves to {landed}; before the agent ran "
                    f"it resolved to {before}"
                )
            raise SeparateVerifierError(
                f"{label} {key} resolves to {landed}, not where it was declared"
            )
        if folder is not None:
            system = _system_landing(landed, folder)
            # Tasks may edit an installed Python package. Only an explicit
            # package subtree at its declared spelling crosses; never the
            # library/search root or a redirect into it. Root helpers use
            # the sealed stdlib-only runtime, and these transferred paths
            # remain excluded from verifier plugin/tool discovery.
            if (
                label == "declared artifact"
                and landed == declared
                and re.fullmatch(
                    r"/usr/(?:local/)?lib(?:64)?/python[0-9]+\.[0-9]+/"
                    r"(?:site|dist)-packages/[^/]+(?:/.*)?",
                    landed,
                )
            ):
                system = None
        elif landed != declared:
            system = (landed, s) if (s := _system_overlap(landed)) else None
        else:
            system = None
        if system is not None:
            info = (pristine or {}).get("paths", {}).get(key) or {}
            if folder is not None and info.get("dir") is False:
                # No folder there before the agent ran (absent, or a file):
                # the solution made the folder, so its result, not the task's.
                raise SolutionTransferRefused(
                    f"{label} {key} is a folder that lands {system[0]} in "
                    f"system folder {system[1]}; before the agent ran it was "
                    "not a folder"
                )
            raise SeparateVerifierError(
                f"{label} {key} lands {system[0]} in system folder {system[1]}"
            )

    # Verifier-owned landings and discovery splits first, for every root:
    # they name the most specific cause. Then where each root may land.
    owned = verifier_owned_overlap(manifest.workspace)
    if owned is not None:
        refuse(declared_root, manifest.workspace, owned)
    base = declared_root or manifest.workspace
    captured = []
    for index, artifact in enumerate(manifest.artifacts):
        if artifact.status != "captured":
            continue
        key = None
        if artifacts is not None and index < len(artifacts):
            key = str(PurePosixPath(base) / as_artifact_config(artifacts[index]).source)
        owned = verifier_owned_overlap(artifact.source)
        if owned is not None:
            refuse(key, artifact.source, owned)
        sub = subs[artifact.bundle_path]
        landed = _owned_member(sub)
        if landed is not None:
            refuse(key, *landed)
        if not _captured_where_discovered(artifact.source, sub):
            raise SeparateVerifierError(
                f"declared artifact {artifact.source} changed between discovery "
                f"and capture (captured under {sub.workspace})"
            )
        captured.append((key, artifact.source, sub))
    if declared_root is not None:
        authorize("workspace", declared_root, manifest.workspace, manifest)
    if artifacts is None:
        return
    for key, source, sub in captured:
        if key is None:
            raise SeparateVerifierError(
                f"captured artifact {source} has no declaration"
            )
        folder = sub if sub.workspace == source else None
        authorize("declared artifact", key, source, folder)


def _stage_logs_file(
    artifacts_dir: Path, relative: PurePosixPath, staged: Path, expected: Any
) -> int:
    """Copy one collected ``/logs/artifacts`` file once, privately; return its mode.

    On Docker the trial's ``artifacts/`` is the agent container's live bind
    mount, so a leftover process can change a file between two reads, or swap
    it for a symlink to a host file. The file is opened once, without
    following links, checked to be a regular file, and copied; that copy is
    what is hashed against the collection manifest and what is packed.
    """
    label = f"/logs/artifacts/{relative}"
    try:
        source = (ReachablePath.directory(artifacts_dir) / relative).open("rb")
    except OSError as exc:
        if isinstance(exc, ReachableWriteError) or exc.errno in (
            errno.ELOOP,
            errno.ENOTDIR,
            errno.EMLINK,
        ):
            raise SeparateVerifierError(f"{label} changed since collection") from exc
        raise SeparateVerifierError(f"{label} is gone: {exc}") from exc
    digest = hashlib.sha256()
    try:
        fd = source.fileno()
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SeparateVerifierError(f"{label} changed since collection")
        out = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        with os.fdopen(out, "wb") as stream:
            while chunk := os.read(fd, 1 << 20):
                digest.update(chunk)
                stream.write(chunk)
    except OSError as exc:
        raise SeparateVerifierError(f"{label} is gone: {exc}") from exc
    finally:
        source.close()
    if digest.hexdigest() != expected:
        raise SeparateVerifierError(f"{label} changed since collection")
    return 0o755 if info.st_mode & stat.S_IXUSR else 0o644


class _Payload:
    """One tar whose member names are sandbox paths without the leading ``/``."""

    def __init__(self, archive: tarfile.TarFile) -> None:
        self.archive = archive
        self.names: dict[str, bytes] = {}
        self.parents: set[str] = set()
        self.links: list[tarfile.TarInfo] = []
        # Each file member's snapshot digest and mode, and each link's target, to tell
        # an overlap (one freeze captured it twice, PY8-MAIN-3) from a clash.
        self.written: dict[str, tuple[str, int]] = {}
        self.link_targets: dict[str, str] = {}
        self.files = 0
        self.bytes = 0
        # Every member, by the transfer root it was added under.
        self.root = "/"
        self.members: dict[str, list[str]] = {}

    def _info(
        self, target: PurePosixPath, kind: bytes, mode: int
    ) -> tarfile.TarInfo | None:
        name = str(target).lstrip("/")
        if not name:
            raise SeparateVerifierError("empty transfer path")
        if name in self.names:
            if kind == self.names[name] == tarfile.DIRTYPE:
                return None
            raise SeparateVerifierError(f"two transfers write {target}")
        if any(
            self.names.get(str(parent)) not in (None, tarfile.DIRTYPE)
            for parent in PurePosixPath(name).parents
        ) or (kind != tarfile.DIRTYPE and name in self.parents):
            raise SeparateVerifierError(
                f"transfer path has a non-directory parent: {target}"
            )
        self.names[name] = kind
        self.parents.update(str(parent) for parent in PurePosixPath(name).parents)
        self.members.setdefault(self.root, []).append(name)
        info = tarfile.TarInfo(name)
        info.type, info.mode, info.uid, info.gid = kind, mode, 0, 0
        info.mtime = int(time.time())
        return info

    def directory(self, target: PurePosixPath, mode: int | None = None) -> None:
        info = self._info(target, tarfile.DIRTYPE, 0o755 if mode is None else mode)
        if info is not None:
            self.archive.addfile(info)

    def file(
        self,
        target: PurePosixPath,
        source: Path | ReachablePath,
        mode: int | None = None,
    ) -> None:
        with snapshot_file(source) as snapshot:
            if mode is None:
                mode = 0o755 if snapshot.stat().st_mode & stat.S_IXUSR else 0o644
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            earlier = self.written.get(str(target).lstrip("/"))
            if earlier is not None and earlier == (digest, mode):
                # Overlapping capture roots (an include path holding the
                # workspace or a declared artifact, PY8-MAIN-3) captured this
                # file twice in one freeze; the same bytes and mode cross once.
                # The bytes compared are the no-follow snapshot's, never the
                # raw source path's.
                return
            info = self._info(target, tarfile.REGTYPE, mode)
            if info is None:
                raise SeparateVerifierError(f"two transfers write {target}")
            self.written[info.name] = (digest, mode)
            info.size = snapshot.stat().st_size
            with snapshot.open("rb") as stream:
                self.archive.addfile(info, stream)
            self.files += 1
            self.bytes += info.size

    def symlink(self, target: PurePosixPath, link_target: str) -> None:
        if self.link_targets.get(str(target).lstrip("/")) == link_target:
            return  # the same link, captured by two overlapping roots
        info = self._info(target, tarfile.SYMTYPE, 0o777)
        if info is not None:
            info.linkname = link_target
            self.links.append(info)
            self.link_targets[info.name] = link_target

    def finish(self) -> None:
        # No later member may extract through a link from any earlier bundle.
        for info in self.links:
            self.archive.addfile(info)


def _logs_entries(collected: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Validate one closed artifact namespace; never resolve links on the host.

    Collection records refused links; only plain unchanged relative targets
    can be delivered. Recheck the complete link population, including refused
    links, so a legacy or incomplete inventory cannot admit a chain.
    """
    entries: dict[str, dict[str, Any]] = {}
    for item in collected["files"]:
        if item.get("collection") != 0:
            continue
        if item.get("kind") == "symlink" and item.get("escapes") is True:
            continue
        path = _safe_relative(item["path"])
        name = str(path)
        if name in entries or item.get("kind") not in {"file", "directory", "symlink"}:
            raise SeparateVerifierError(f"invalid artifact inventory entry: {name}")
        entries[name] = dict(item)

    # Older inventories omit directories. Their file/link parents are still
    # explicit directories in this namespace, never aliases to other entries.
    for name in list(entries):
        for parent in PurePosixPath(name).parents:
            if str(parent) == ".":
                continue
            if str(parent) in entries and entries[str(parent)]["kind"] != "directory":
                raise SeparateVerifierError(
                    f"artifact has a symlink or file parent: {name}"
                )
            entries.setdefault(str(parent), {"path": str(parent), "kind": "directory"})

    links: dict[str, str] = {}
    for name, item in entries.items():
        source = root / name
        try:
            mode = source.lstat().st_mode
            expected = {
                "file": stat.S_ISREG,
                "directory": stat.S_ISDIR,
                "symlink": stat.S_ISLNK,
            }
            if not expected[item["kind"]](mode):
                raise SeparateVerifierError(
                    f"artifact kind changed since collection: {name}"
                )
            if item["kind"] == "symlink":
                target = item.get("target")
                if (
                    not isinstance(target, str)
                    or not target
                    or os.readlink(source) != target
                ):
                    raise SeparateVerifierError(
                        f"artifact symlink changed since collection: {name}"
                    )
                links[name] = target
        except OSError as exc:
            raise SeparateVerifierError(
                f"artifact entry is gone: {name}: {exc}"
            ) from exc

    population = {
        item["path"]: item.get("target") or ""
        for item in collected["files"]
        if item.get("collection") == 0 and item.get("kind") == "symlink"
    }
    _, excluded = plain_link_targets(population)
    if excluded.intersection(links):
        raise SeparateVerifierError(
            f"artifact symlink is not a plain relative path: {min(excluded.intersection(links))}"
        )
    return sorted(
        entries.values(),
        key=lambda e: (e["kind"] == "symlink", len(e["path"]), e["path"]),
    )


def _add_bundle(
    payload: _Payload,
    bundle: Path,
    manifest: Any,
    only: Sequence[PurePosixPath] | None = None,
) -> None:
    """Pack *bundle*'s frozen tree; with *only*, just the entries at or under
    those paths (relative to its workspace)."""
    root = _absolute(manifest.workspace)
    payload.root = str(root)
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
        relative = _safe_relative(entry.path)
        if only is not None and not any(
            relative == kept or kept in relative.parents for kept in only
        ):
            continue
        target = root / relative
        mode = None if entry.mode is None else entry.mode & 0o777
        if entry.kind == "directory":
            payload.directory(target, mode)
        elif entry.kind == "file":
            payload.file(target, tree / entry.path, mode)
        elif entry.link_target is not None:
            payload.symlink(target, entry.link_target)


def _declared_in_workspace(
    manifest: Any,
) -> tuple[list[PurePosixPath], list[str]] | None:
    """The declared artifacts inside the frozen workspace, as paths relative
    to it, and the folders among them (absolute); None when one is the
    workspace itself, which then crosses whole."""
    root = _absolute(manifest.workspace)
    kinds = {_safe_relative(e.path): e.kind for e in manifest.entries}
    kept: list[PurePosixPath] = []
    folders: list[str] = []
    for artifact in manifest.artifacts:
        if artifact.status != "in_workspace":
            continue
        try:
            relative = PurePosixPath(posixpath.normpath(artifact.source)).relative_to(
                root
            )
        except ValueError:
            continue
        if relative == PurePosixPath("."):
            return None
        kept.append(relative)
        if kinds.get(relative) == "directory":
            folders.append(str(root / relative))
    return kept, folders


def build_transfer_payload(
    rollout_dir: Path,
    dest: Path,
    *,
    pristine: dict[str, Any] | None = None,
    capture_over_limit: bool = False,
    artifacts: Sequence[Any] | None = None,
    workspace: str | None = None,
    artifacts_only: bool = False,
) -> dict[str, Any]:
    """Pack the frozen workspace, declared artifacts and ``/logs/artifacts``.

    With ``artifacts_only`` (a task.toml task's verifier built from
    ``tests/Dockerfile``: :func:`_format_verifier`) the workspace contributes
    only its declared artifacts: that format's verifier keeps the files its
    own image put in the workspace and gets each declared artifact over them
    at its path. The inventory's ``artifacts_only`` says which happened, and
    ``emptied`` lists the declared folders the install empties first.

    Every byte is checked against the manifest written when it was captured.
    Returns the transfer inventory; raises :class:`SeparateVerifierError`
    when anything the verifier needs is missing or does not match, or
    :class:`SolutionTransferRefused` when the solution's own files are why,
    judged against ``pristine`` (:func:`record_pristine_outputs`).
    ``artifacts`` are the task's declared artifacts, in order, for that
    judgement, and ``workspace`` the declared workspace (the agent's cwd):
    every packed root must land where they say (:func:`_check_resolved_roots`);
    without ``artifacts`` only verifier-owned paths and the discovery split are
    checked. The inventory's ``custody`` lists every member per root.
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
        left_out: dict[str, list[dict[str, str]]] = {}
        by_bundle = {}
        for artifact in manifest.artifacts:
            if artifact.bundle_path is None:
                continue
            sub = bundle / artifact.bundle_path
            sub_manifest = EvidenceManifest.model_validate_json(
                (sub / "manifest.json").read_text()
            )
            validate_workspace(sub / "workspace", sub_manifest)
            subs.append((sub, sub_manifest))
            left_out[artifact.bundle_path] = [
                {"path": exclusion.original_path, "reason": exclusion.reason}
                for exclusion in sub_manifest.exclusions
            ]
            by_bundle[artifact.bundle_path] = sub_manifest
    except (EvidenceError, OSError, ValueError) as exc:
        raise SeparateVerifierError(
            f"frozen workspace does not match its manifest: {exc}"
        ) from exc

    try:
        collected = _read_json(Path(rollout_dir) / MANIFEST_NAME)
    except (OSError, ValueError) as exc:
        raise SeparateVerifierError(
            "artifact collection manifest is unreadable"
        ) from exc
    if (
        not isinstance(collected, dict)
        or collected.get("version") != 1
        or not isinstance(collected.get("collections"), list)
        or not collected["collections"]
        or any(not isinstance(record, dict) for record in collected["collections"])
        or not isinstance(collected.get("files"), list)
        or any(not isinstance(record, dict) for record in collected["files"])
        or collected["collections"][0].get("kind") != "logs"
        or collected["collections"][0].get("status")
        not in {"collected", "empty", "missing", *_FAILED_COLLECTIONS}
    ):
        raise SeparateVerifierError(
            "artifact collection manifest is missing or incomplete"
        )
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
    # After the collection check, whose refusal (a declared artifact left as
    # a symlink) names the cause more precisely than where the link led.
    _check_resolved_roots(manifest, by_bundle, pristine, artifacts, workspace)

    logs_entries = _logs_entries(collected, Path(rollout_dir) / "artifacts")
    logs_files = [f for f in logs_entries if f["kind"] == "file"]
    declared = _declared_in_workspace(manifest) if artifacts_only else None
    kept, emptied = declared if declared is not None else (None, [])

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    # Private to this process (mode 0700), outside every folder a sandbox
    # mounts: the one copy of each /logs/artifacts file that is hashed and packed.
    staging = Path(tempfile.mkdtemp(prefix=".transfer-staging-", dir=dest.parent))
    try:
        with tarfile.open(dest, "x", format=tarfile.PAX_FORMAT) as archive:
            payload = _Payload(archive)
            _add_bundle(payload, bundle, manifest, only=kept)
            for sub, sub_manifest in subs:
                _add_bundle(payload, sub, sub_manifest)
            logs_root = PurePosixPath(LOGS_ARTIFACTS)
            payload.root = LOGS_ARTIFACTS
            for index, item in enumerate(logs_entries):
                relative = _safe_relative(item["path"])
                if item["kind"] == "directory":
                    payload.directory(logs_root / relative)
                    continue
                if item["kind"] == "symlink":
                    payload.symlink(logs_root / relative, item["target"])
                    continue
                staged = staging / str(index)
                mode = _stage_logs_file(
                    Path(rollout_dir) / "artifacts",
                    relative,
                    staged,
                    item.get("sha256"),
                )
                for parent in reversed(list(relative.parents)[:-1]):
                    payload.directory(logs_root / parent)
                payload.file(logs_root / relative, staged, mode)
                staged.unlink()
            payload.finish()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    with dest.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "workspace": manifest.workspace,
        # Only the paths whose contents crossed came from the agent. Parent
        # directories of individual files, and the rest of an artifacts-only
        # workspace, still contain the verifier image's own tools.
        "agent_paths": [
            *(
                [manifest.workspace]
                if kept is None
                else [str(PurePosixPath(manifest.workspace) / p) for p in kept]
            ),
            *(a.source for a in manifest.artifacts if a.status == "captured"),
            LOGS_ARTIFACTS,
        ],
        "files": payload.files,
        "bytes": payload.bytes,
        "artifacts_only": kept is not None,
        "emptied": emptied,
        "declared_artifacts": len(subs),
        "missing_artifacts": [
            a.source for a in manifest.artifacts if a.status == "missing"
        ],
        # Paths captured beside the workspace (declared artifacts and
        # ``--separate-verifier-include``) and what became of each: captured,
        # missing or already in_workspace, and what capture left out of a
        # captured one (credentials, package caches, links out of it, its own
        # excludes), so nothing an include names is dropped silently
        # (PY8-MAIN-6).
        "artifacts": [
            {
                "source": artifact.source,
                "status": artifact.status,
                **(
                    {"left_out": left_out[artifact.bundle_path or ""]}
                    if left_out.get(artifact.bundle_path or "")
                    else {}
                ),
            }
            for artifact in manifest.artifacts
        ],
        "logs_artifacts": len(logs_files),
        "excluded": len(manifest.exclusions),
        "sha256": sha256,
        "custody": {
            "tar_sha256": sha256,
            "tar_bytes": dest.stat().st_size,
            "members": [
                _member_list(root, names) for root, names in payload.members.items()
            ],
        },
    }


def _member_list(root: str, names: list[str]) -> dict[str, Any]:
    """One transfer root's tar members, as paths relative to it."""
    prefix = root.strip("/")
    relative = sorted(
        name[len(prefix) + 1 :] if prefix and name.startswith(prefix + "/") else name
        for name in names
    )
    return {
        "root": root,
        "count": len(relative),
        "members": relative[:_LIST_LIMIT],
        # Over every member, so a list cut at the limit can still be checked.
        "members_sha256": hashlib.sha256(
            "\n".join(relative).encode("utf-8", "surrogateescape")
        ).hexdigest(),
    }


def _warn_left_out(summary: dict[str, Any], includes: Sequence[str]) -> None:
    """Name every include path, and what of it, that does not cross whole."""
    for item in summary.get("artifacts", ()):
        if item["source"] not in includes or not item.get("left_out"):
            continue
        left = item["left_out"]
        logger.warning(
            "--separate-verifier-include %s: %d path(s) under it do not reach "
            "the verifier (%s)",
            item["source"],
            len(left),
            ", ".join(f"{e['path']}: {e['reason']}" for e in left[:5]),
        )


def _capture_declarations(rollout: Any) -> list[Any]:
    """What capture read beside the workspace, in its order: the task's
    artifacts, then each ``--separate-verifier-include`` path none of them
    holds whole (``capture_terminal_workspace``)."""
    from benchflow.rollout._review import _uncovered_includes
    from benchflow.task.config import ArtifactConfig

    artifacts = list(rollout._task.config.artifacts)
    includes = getattr(rollout._config, "separate_verifier_include", ())
    return [
        *artifacts,
        *(
            ArtifactConfig(source=path)
            for path in _uncovered_includes(includes, artifacts)
        ),
    ]


# --- agent sandbox: what does not cross -------------------------------------

# Runs in the agent's sandbox as root after the agent stops, before hardening.
# Advisory only: it lists regular files created or changed during the agent
# phase outside everything that crosses (the workspace, include paths), so a
# verifier that reads them does not silently score 0. Software, scratch,
# provider and harness trees (/oracle, /tests, BenchFlow's own trees in
# _HARNESS_TREES) are skipped, as are package caches and the
# agent's own home.
# Folders the outside-writes listing never walks: the system's, and what the
# verifier gets anyway.
_OUTSIDE_WRITES_SYSTEM = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/tmp",
    "/var",
    "/usr",
    "/etc",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/boot",
    "/logs",
    "/nix",
    "/snap",
    "/lost+found",
    "/oracle",
    "/solution",
    "/tests",
    "/verifier",
)
# BenchFlow's own trees in the agent's sandbox, skipped by both listings (the
# script below and its find form): each name, what is under it, and its
# siblings that start with the name and "-" or "_" (the egress proxy's
# /opt/runtime-egress, the OpenTelemetry receiver's /opt/runtime-otel,
# /opt/runtime_agentcore_shim.py; root's stages /.runtime-kit-<hex>,
# /.runtime-credential-<hex>, /.runtime_build_snapshot and the like).
_HARNESS_TREES = (RUNTIME_ROOT, "/.runtime")
_HARNESS_SEPARATORS = ("/", "-", "_")
_OUTSIDE_WRITES_SCRIPT = r"""
import json, os, stat, sys
root, since, limit = sys.argv[1], float(sys.argv[2]), int(sys.argv[5])
keep, components = json.loads(sys.argv[3]), json.loads(sys.argv[4])
system = tuple(json.loads(sys.argv[6]))
skip = [os.path.join(root, p.lstrip("/")).rstrip("/") for p in (*system, *keep)]
harness, separators = json.loads(sys.argv[7]), json.loads(sys.argv[8])
harness = [os.path.join(root, p.lstrip("/")) for p in harness]
harness_prefixes = tuple(h + s for h in harness for s in separators)
def skipped(path):
    return path in harness or path.startswith(harness_prefixes) or any(
        path == s or path.startswith(s + "/") for s in skip) or any(
        path.endswith("/" + c) for c in components)
device = os.lstat(root).st_dev
count, sample, by_root = 0, [], {}
for directory, dirs, files in os.walk(root, followlinks=False):
    kept = []
    for name in dirs:
        path = os.path.join(directory, name)
        try:
            if not skipped(path) and os.lstat(path).st_dev == device:
                kept.append(name)
        except OSError:
            pass
    dirs[:] = kept
    for name in files:
        path = os.path.join(directory, name)
        if skipped(path):
            continue
        try:
            info = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode) or max(info.st_mtime, info.st_ctime) < since:
            continue
        count += 1
        shown = "/" + os.path.relpath(path, root)
        top = "/" + shown.split("/")[1]
        by_root[top] = by_root.get(top, 0) + 1
        if len(sample) < 20:
            sample.append(shown)
        if count >= limit:
            break
    if count >= limit:
        break
print(json.dumps({"count": count, "truncated": count >= limit,
                  "by_directory": by_root, "sample": sample,
                  "basis": "mtime+ctime"}))
"""
_OUTSIDE_WRITES_LIMIT = 100_000


async def scan_outside_workspace_writes(
    env: Any,
    *,
    since: float,
    keep: list[str],
    root: str = "/",
    timeout_sec: int = 120,
) -> dict[str, Any] | None:
    """Files written after ``since`` that a separate verifier will not see.

    ``keep`` lists what crosses or belongs to the harness (workspace, include
    paths, the agent's home). Returns ``None`` when the scan cannot run; it
    never fails the rollout.
    """
    from benchflow.review.evidence import PACKAGE_CACHE_PATHS
    from benchflow.sandbox.protocol import SANDBOX_RUNTIME_STATE_PATHS

    components = [*PACKAGE_CACHE_PATHS, *SANDBOX_RUNTIME_STATE_PATHS]
    try:
        # The protected Python, isolated (C8); a sandbox
        # without one (127) takes dx-f's find listing below.
        command = await root_python_command(
            env,
            _OUTSIDE_WRITES_SCRIPT,
            root,
            repr(since),
            json.dumps(keep),
            json.dumps(components),
            str(_OUTSIDE_WRITES_LIMIT),
            json.dumps(_OUTSIDE_WRITES_SYSTEM),
            json.dumps(_HARNESS_TREES),
            json.dumps(_HARNESS_SEPARATORS),
        )
        result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout_sec)
        if python_missing(result):
            # A shell-only image: the same listing with find (DF-p4-upgrader-9).
            # Its own exit status decides: a find that cannot list records
            # nothing (None and the warning below), never "0 files". NUL
            # frames the paths, so they cross as a receipt (a session
            # command's output on Daytona loses every NUL).
            result = await receipt_exec(
                env,
                _outside_writes_find(root, since, keep, components),
                user="root",
                timeout_sec=timeout_sec,
            )
            if getattr(result, "return_code", 0):
                raise RuntimeError(
                    f"find exited {result.return_code}: "
                    + (result.stderr or result.stdout or "")[-500:]
                )
            return _outside_writes_report(root, result.stdout or "")
        if getattr(result, "return_code", 0):
            raise RuntimeError((result.stderr or result.stdout or "")[-500:])
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except Exception as exc:
        # One line, no traceback, at the default level (DF-p4-upgrader-9).
        logger.warning("Could not list files outside the workspace (%s)", exc)
        logger.debug("Outside-workspace listing failed", exc_info=True)
        return None
    return report if isinstance(report, dict) else None


def _find_glob(path: str) -> str:
    """``path`` with the glob characters of find's -path escaped."""
    return "".join("\\" + c if c in "*?[]\\" else c for c in path)


# The python3-free listing's output cap, in bytes (NUL-separated paths).
_OUTSIDE_WRITES_FIND_BYTES = 64 * 1024**2


def _outside_writes_find(
    root: str, since: float, keep: list[str], components: list[str]
) -> str:
    """The python3-free form of :data:`_OUTSIDE_WRITES_SCRIPT`, as one sh script.

    The same folders are pruned; it prints each regular file changed after
    ``since``, NUL-terminated, so a name holding a newline is one entry. GNU
    find compares mtime and ctime (``-newermt``/``-newerct``); a find without
    them (BusyBox, Alpine) compares mtime with ``-newer`` against a stamp
    file ``touch -d`` dates one second before ``since``'s second (it takes
    whole seconds, and -newer is strict), so a file written in the agent
    phase's first second is listed too; BusyBox has no usable ctime test, so
    a file whose mtime was set back escapes it, and the record says
    ``basis: mtime`` (review DX-e R2-L2). The first entry printed is that
    basis. The script's exit status is find's own: no pipe can hide a find
    that refused its arguments (review DX-e M1).
    """

    def under(path: str) -> str:
        return posixpath.join(root, path.lstrip("/")).rstrip("/") or "/"

    pruned = [under(p) for p in (*_OUTSIDE_WRITES_SYSTEM, *keep)]
    tests: list[str] = []
    for path in pruned:
        tests += [
            "-path",
            _find_glob(path),
            "-o",
            "-path",
            _find_glob(path) + "/*",
            "-o",
        ]
    for tree in _HARNESS_TREES:
        tests += ["-path", _find_glob(under(tree)), "-o"]
        for separator in _HARNESS_SEPARATORS:
            tests += ["-path", _find_glob(under(tree)) + separator + "*", "-o"]
    for component in components:
        tests += ["-path", "*/" + _find_glob(component.strip("/")), "-o"]
    stamp = f"@{since:.6f}"
    whole = f"@{int(since) - 1}"
    walk = shlex.join(["find", root, "-xdev", "(", *tests[:-1], ")", "-prune", "-o"])
    gnu = shlex.join(["(", "-newermt", stamp, "-o", "-newerct", stamp, ")"])
    return "\n".join(
        [
            "set -u",
            "out=$(mktemp) || exit 70",
            'ref=$(mktemp) || { rm -f "$out"; exit 70; }',
            'trap \'rm -f "$out" "$ref"\' EXIT',
            "if find / -maxdepth 0 -newermt @0 >/dev/null 2>&1; then",
            "  basis=mtime+ctime",
            f'  {walk} -type f {gnu} -print0 > "$out"',
            "else",
            "  basis=mtime",
            f'  touch -d {shlex.quote(whole)} "$ref" || exit 71',
            f'  {walk} -type f -newer "$ref" -print0 > "$out"',
            "fi",
            "rc=$?",
            '[ "$rc" -eq 0 ] || exit "$rc"',
            "printf 'basis=%s\\0' \"$basis\"",
            f'head -c {_OUTSIDE_WRITES_FIND_BYTES} "$out"',
        ]
    )


def _outside_writes_report(root: str, stdout: str) -> dict[str, Any]:
    """The listing's report from the NUL-separated paths of
    :func:`_outside_writes_find`."""
    entries = stdout.split("\0")
    basis = None
    if entries and entries[0].startswith("basis="):
        basis = entries.pop(0)[len("basis=") :]
    # The last piece is empty after a final NUL, or a path cut at the cap.
    cut = entries.pop() != "" if entries else False
    truncated = cut or len(entries) >= _OUTSIDE_WRITES_LIMIT
    entries = entries[:_OUTSIDE_WRITES_LIMIT]
    by_root: dict[str, int] = {}
    shown_paths = []
    for path in entries:
        shown = "/" + posixpath.relpath(path, root)
        shown_paths.append(shown)
        top = "/" + shown.split("/")[1]
        by_root[top] = by_root.get(top, 0) + 1
    return {
        "count": len(entries),
        "truncated": truncated,
        "by_directory": by_root,
        "sample": shown_paths[:20],
        **({"basis": basis} if basis else {}),
    }


# --- sandbox side ------------------------------------------------------------


def _install_script(
    remote: str,
    workspace: str,
    sha256: str,
    root: str = "/",
    *,
    keep_workspace: bool = False,
    emptied: Sequence[str] = (),
) -> str:
    """Unpack the transfer at ``root`` (the sandbox's ``/``; a test's folder).

    With ``keep_workspace`` (an artifacts-only transfer) the workspace keeps
    what the verifier image put there and only the declared folders in
    ``emptied`` are emptied first, as the task.toml format's verifier
    receives its artifacts. Shell and tar only: a verifier image may have no
    Python.
    """
    quoted = shlex.quote(remote)
    # Directories this creates (a missing workspace, /logs/artifacts) get the
    # verifier's modes whatever the runtime's mask; tar restores its own. The
    # mask is lockdown.VERIFIER_UMASK, which the rollout kernel must not import.
    lines = ["set -e", "umask 022"]
    # The check needs coreutils' sha256sum; an image without it says so, and
    # the record keeps which happened.
    lines.append(
        "if command -v sha256sum >/dev/null 2>&1; then "
        f"echo {shlex.quote(sha256 + '  ' + remote)} | sha256sum -c - >/dev/null; "
        f"echo {_SHA256_CHECKED}; else echo {_SHA256_SKIPPED}; fi"
    )
    path = PurePosixPath(workspace)
    target = shlex.quote(posixpath.join(root, workspace.lstrip("/")))
    if keep_workspace:
        for folder in emptied:
            declared = shlex.quote(posixpath.join(root, folder.lstrip("/")))
            lines.append(f"if [ -L {declared} ]; then rm -f -- {declared}; fi")
            lines.append(
                f"if [ -d {declared} ]; then rm -rf -- {declared}/* "
                f"{declared}/.[!.]* {declared}/..?*; fi"
            )
    elif str(path) not in _SHARED_DIRS:
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


# Tar opens each member by its path, so a link the verifier image keeps on the
# way moves where the member lands: GNU tar follows one at every folder that
# has no member of its own (a root and its ancestors), busybox tar at every
# folder. The leaf itself is never followed (both replace a file or link
# there). For each folder on the way to a member the probe prints, by index,
# where a link there resolves, "?" when it leads to no folder. ``$1`` roots
# the paths (a test's folder).
_DESTINATIONS_DONE = "BENCHFLOW_DESTINATIONS_DONE"


def _destination_paths(
    names: Sequence[str],
    roots: Sequence[str],
    workspace: str,
    emptied: Sequence[str] | None = None,
) -> list[str]:
    """Every folder the unpack walks through or creates on the way to a member.

    ``names`` are the archive's members as absolute paths, folders included;
    ``roots`` the transfer roots (an empty one is still created or emptied).
    A dedicated workspace's own folder and everything under it are left out:
    the install script removes a link there and empties it first. With
    ``emptied`` (an artifacts-only transfer, which keeps the workspace) the
    folders the install empties are left out instead.
    """
    if emptied is None:
        cleared = [] if workspace in _SHARED_DIRS else [PurePosixPath(workspace)]
    else:
        cleared = [PurePosixPath(folder) for folder in emptied]
    paths: dict[str, None] = {}
    for name in (*roots, *names):
        path = PurePosixPath(name)
        for folder in (*reversed(path.parents[:-1]), path):
            if any(folder == c or c in folder.parents for c in cleared):
                break
            paths.setdefault(str(folder))
    return list(paths)


def _destination_script(paths: Sequence[str]) -> str:
    quoted = " ".join(shlex.quote(p) for p in paths)
    return "\n".join(
        [
            'R="${1:-}"',
            "i=0",
            f"for p in {quoted}; do",
            '  if [ -L "$R$p" ]; then',
            "    r=$(cd -P -- \"$R$p\" 2>/dev/null && pwd -P) || r='?'",
            '    printf \'%s %s\\n\' "$i" "${r#"$R"}"',
            "  fi",
            "  i=$((i+1))",
            "done",
            f"echo {_DESTINATIONS_DONE}",
        ]
    )


def _judge_destinations(
    names: Sequence[str], links: dict[str, str | None]
) -> list[dict[str, Any]]:
    """Refuse a member the verifier image's own links would land badly.

    ``links`` maps each folder on the way that is a link to where it
    resolves (None: nowhere a folder can be). A member under a link lands at
    the link's resolution plus the rest of its path; one landing in a
    verifier-owned or system folder, or under a link that leads nowhere, is
    the task's or image's doing, so the transfer fails unscored. Returns the
    links that moved members, for the custody record.
    """
    moved: dict[str, str] = {}
    for name in names:
        path = PurePosixPath(name)
        link = next((str(f) for f in (path, *path.parents) if str(f) in links), None)
        if link is None:
            continue
        target = links[link]
        if target is None:
            raise SeparateVerifierError(
                f"{link} in the verifier image is a link that leads to no "
                f"folder, so {name} cannot land where it was packed"
            )
        rest = path.relative_to(link)
        physical = str(PurePosixPath(target) / rest) if rest.parts else target
        place = verifier_owned_overlap(physical) or _system_overlap(physical)
        if place is not None:
            raise SeparateVerifierError(
                f"{link} in the verifier image is a link to {target}, so "
                f"{name} would land at {physical}, in {place}"
            )
        moved[link] = target
    return [{"path": p, "resolves_to": t} for p, t in sorted(moved.items())]


async def _check_destinations(
    env: Any, archive: Path, summary: dict[str, Any], timeout_sec: int
) -> list[dict[str, Any]]:
    """Where the members land in the verifier sandbox, before anything changes.

    Runs before the workspace is emptied or anything is unpacked; nothing in
    the verifier sandbox has run yet to change the answer afterwards.
    """
    from benchflow.rollout._artifacts import LOGS_ARTIFACTS

    names, walked = [], []
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            name = "/" + member.name.lstrip("/")
            names.append(name)
            # A folder is walked into; a file or link only replaced.
            walked.append(name if member.isdir() else posixpath.dirname(name))
    roots = [str(PurePosixPath(r)) for r in summary["agent_paths"]]
    paths = _destination_paths(
        walked,
        roots,
        summary["workspace"],
        summary.get("emptied", []) if summary.get("artifacts_only") else None,
    )
    script = archive.with_name(archive.name + ".destinations.sh")
    script.write_text(_destination_script(paths))
    remote = f"/tmp/runtime-verifier-destinations-{uuid.uuid4().hex}.sh"
    quoted = shlex.quote(remote)
    # The script names paths taken from the transfer's member names.
    admission = admit(env, "main", IngressKind.SEPARATE_VERIFIER_TRANSFER)
    try:
        await env.upload_file(script, remote, admission=admission)
        result = await exec_raw(
            env.exec,
            shlex.join(["sh", "-c", f"sh {quoted}; s=$?; rm -f -- {quoted}; exit $s"]),
            user="root",
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        raise SeparateVerifierError(
            f"destination check failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        script.unlink(missing_ok=True)
    lines = (result.stdout or "").splitlines()
    if result.return_code or not lines or lines[-1] != _DESTINATIONS_DONE:
        detail = (
            diagnostic_result(result).stderr.render()
            or diagnostic_result(result).stdout.render()
            or ""
        ).strip()[-500:]
        raise SeparateVerifierError(
            f"destination check failed (exit {result.return_code}): {detail}"
        )
    links: dict[str, str | None] = {}
    for line in lines[:-1]:
        index, _, target = line.partition(" ")
        if not index.isdigit() or int(index) >= len(paths):
            raise SeparateVerifierError(f"destination check printed {line!r}")
        links[paths[int(index)]] = target if target.startswith("/") else None
    # The workspace and /logs/artifacts are created (and the workspace
    # emptied) even when nothing lands in them; an artifact root with no
    # members is never touched.
    touched = [str(PurePosixPath(summary["workspace"])), LOGS_ARTIFACTS]
    return _judge_destinations(names + touched, links)


async def _install_payload(
    env: Any, archive: Path, summary: dict[str, Any], timeout_sec: int
) -> bool | None:
    """Upload and unpack the transfer; whether the in-sandbox sha256 check ran.

    None when the script's report is missing from its output.
    """
    remote = f"/tmp/runtime-verifier-transfer-{uuid.uuid4().hex}.tar"
    admission = admit(env, "main", IngressKind.SEPARATE_VERIFIER_TRANSFER)
    try:
        await env.upload_file(archive, remote, admission=admission)
        script = _install_script(
            remote,
            summary["workspace"],
            summary["sha256"],
            keep_workspace=bool(summary.get("artifacts_only")),
            emptied=summary.get("emptied", ()),
        )
        result = await exec_raw(
            env.exec,
            shlex.join(["sh", "-c", script]),
            user="root",
            timeout_sec=timeout_sec,
        )
    except SeparateVerifierError:
        raise
    except Exception as exc:
        raise SeparateVerifierError(
            f"upload failed: {type(exc).__name__}: {exc}"
        ) from exc
    if result.return_code:
        detail = (
            diagnostic_result(result).stderr.render()
            or diagnostic_result(result).stdout.render()
            or ""
        ).strip()[-500:]
        raise SeparateVerifierError(
            f"unpack failed (exit {result.return_code}): {detail}"
        )
    lines = (result.stdout or "").splitlines()
    if _SHA256_CHECKED in lines:
        return True
    if _SHA256_SKIPPED in lines:
        return False
    return None


def _format_verifier(task: Any, plan: VerifierImage) -> bool:
    """A task.toml task's verifier built from ``tests/Dockerfile``.

    That format bundles the tests in the verifier image and gives the
    verifier the declared artifacts, each at its path over what the image
    put there: Terminal-Bench 4.1's live-database-cutover bakes its pristine
    ``/app/api`` git baseline into the image and grades the agent's
    ``/tmp/agent.patch`` against it.
    """
    return (
        plan.source == "tests/Dockerfile"
        and getattr(task.paths, "uses_native_verifier_dir", False) is not True
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
    # Collect hooks ran in the agent's sandbox before the freeze
    # (_collect_hooks); the verifier sandbox never runs them again.
    config.verifier.collect = []
    view.config = config
    if getattr(view, "document", None) is not None:
        view.document = None
    # The task.toml format bundles a separate verifier's tests in the image
    # built from tests/: the verifier uses /tests as that Dockerfile left it
    # (Verifier._bundled_tests).
    view._benchflow_bundled_tests = _format_verifier(task, plan)
    return view


def _stage_context(
    root: Path,
    task_name: str,
    plan: VerifierImage,
    *,
    planes: Any = None,
    context_root: Path | None = None,
    base_image_override: str | None = None,
) -> Path:
    """A task path whose ``environment/`` is the verifier image's build context.

    Named ``<task>__verifier`` so a Docker image tag never collides with the
    agent image built from the task's own ``environment/``.

    With the ``environment/Dockerfile`` fallback the verifier image is meant
    to be the agent's image without the agent's changes, so the run's
    ``--base-image-override`` and ``--context-root`` staging are applied to
    this copy as Rollout.setup applies them to the agent's (base first, then
    the COPY sources). Without them the build would miss the staged COPY
    sources (every trial a verifier error) or start FROM another base than
    the agent's (tests scored in another toolchain). Both rewrites go through
    the rollout's ``planes``, as the agent's do. A ``tests/Dockerfile`` or a
    ``docker_image`` is the task's own verifier image and is left as
    declared. Skills are never injected here: they are the agent's.
    """
    context = root / "context" / f"{task_name}__verifier"
    if context.exists():
        shutil.rmtree(context)
    environment = context / "environment"
    if plan.context_dir is not None:
        shutil.copytree(plan.context_dir, environment, symlinks=False)
    else:
        environment.mkdir(parents=True)
    if plan.source == "environment/Dockerfile":
        if base_image_override:
            planes.override_dockerfile_base_image(context, base_image_override)
        if context_root:
            planes.stage_dockerfile_deps(context, Path(context_root))
    return context


def _trajectory_record(rollout: Any, paths: RolloutPaths) -> dict[str, Any] | None:
    """The trajectory published to the verifier's ``/logs/agent``, if any.

    It is read after the verifier ran, as root, in a sandbox it may lock
    down: Terminal-Bench 4.1's distributed-dedup makes every world-writable
    folder root's and 0700, the bind-mounted ``/logs/agent`` among them. A
    copy the host can no longer read is recorded as such; it never costs
    the score the verifier already gave.
    """
    # agent/ is a held capability: read through its no-follow handles.
    published = paths.agent_dir / "acp_trajectory.jsonl"
    record = {
        "source": getattr(rollout, "_trajectory_source", None) or "acp",
        "path": "/logs/agent/acp_trajectory.jsonl",
    }
    try:
        if not published.is_file():
            return None
        data = published.read_bytes()
    except OSError as exc:
        return {**record, "unreadable": f"{type(exc).__name__}: {exc.strerror or exc}"}
    return {**record, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


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


def _default_verify(
    rollout: Any, trajectory: list[dict] | None = None, *, agent: str | None = None
) -> Verify:
    """The normal verifier path in the verifier sandbox.

    ``trajectory`` is what the verifier reads at ``/logs/agent``: the snapshot
    taken before the agent's sandbox was released, or, in the serial order,
    the rollout's trajectory as it stands. ``agent`` is the agent that wrote
    it (a regrade's trial agent); by default the rollout's.
    """

    async def verify(
        env: Any,
        task: Any,
        paths: RolloutPaths,
        timing: dict,
        *,
        workspace: str,
        agent_paths: tuple[str, ...] | None = None,
    ) -> tuple[dict | None, str | None]:
        from benchflow.agents.registry import is_scripted_agent
        from benchflow.rollout._setup import (
            _publish_trajectory_for_verifier,
            _verify_rollout,
        )

        await _publish_trajectory_for_verifier(
            env,
            rollout._trajectory if trajectory is None else trajectory,
            paths.agent_dir,
            publish_reply=not is_scripted_agent(agent or rollout._config.primary_agent),
        )
        with recording_trial_warnings() as warnings:
            try:
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
                    agent_paths=(workspace,) if agent_paths is None else agent_paths,
                )
            finally:
                kept = vars(rollout).setdefault("_warnings", [])
                kept.extend(w for w in warnings if w not in kept)
        diagnostics = getattr(rollout, "_diagnostics", None)
        if timeout_diag is not None and diagnostics is not None:
            diagnostics.set(timeout_diag)
        return rewards, error

    return verify


def _display_name(name: str) -> str:
    # A name that is not UTF-8 is listed with its bytes escaped.
    return os.fsencode(name).decode("utf-8", "backslashreplace")


def _list_tree(path: Path) -> dict[str, Any]:
    """Every entry under ``path``, read without following a symlink.

    Each entry has its name and type; a regular file the host can read also
    gets its size and sha256 (no sha256 over 64 MiB), and one it cannot read,
    or a folder it cannot list, ``unreadable: true``. The first 1,000 entries
    are kept, with the total count (``truncated`` once the walk stops, at
    100,000 entries or 64 levels deep). Only kept files are read, at most
    256 MiB in all; a file past that gets its size only, and the listing
    ``hash_budget_exhausted: true``.
    """
    entries: list[dict[str, Any]] = []
    state = {"total": 0, "truncated": False, "hashed": 0, "exhausted": False}

    def visit(folder: ReachablePath, prefix: str, depth: int) -> None:
        for child in sorted(folder.iterdir()):
            name = child.name
            if state["total"] >= _WALK_LIMIT:
                state["truncated"] = True
                return
            state["total"] += 1
            entry: dict[str, Any] = {"name": prefix + _display_name(name)}
            kept = len(entries) < _LIST_LIMIT
            if kept:
                entries.append(entry)
            try:
                info = child.lstat()
            except OSError:
                entry.update(type="unknown", unreadable=True)
                continue
            if stat.S_ISLNK(info.st_mode):
                entry["type"] = "symlink"
                with contextlib.suppress(OSError):
                    entry["target"] = _display_name(child.readlink_text())
            elif stat.S_ISDIR(info.st_mode):
                entry["type"] = "directory"
                if depth >= _WALK_DEPTH:
                    state["truncated"] = True
                    continue
                try:
                    visit(child, entry["name"] + "/", depth + 1)
                except OSError:
                    entry["unreadable"] = True
            elif stat.S_ISREG(info.st_mode):
                entry["type"] = "file"
                if not kept:
                    # Counted, never read: the record does not show it.
                    continue
                budget = _HASH_BUDGET - state["hashed"]
                limit = max(0, min(_HASH_LIMIT, budget))
                digest, read = _file_digest(folder, name, limit)
                state["hashed"] += read
                entry.update(digest)
                if "sha256" not in digest and (
                    budget < digest.get("bytes", 0) <= _HASH_LIMIT
                ):
                    # Unhashed only because the budget ran out.
                    state["exhausted"] = True
            else:
                entry["type"] = "other"

    unreadable = False
    try:
        visit(ReachablePath.directory(path), "", 0)
    except OSError:
        unreadable = True
    listed: dict[str, Any] = {"total": state["total"], "entries": entries}
    if state["truncated"]:
        listed["truncated"] = True
    if state["exhausted"]:
        listed["hash_budget_exhausted"] = True
    if unreadable:
        listed["unreadable"] = True
    return listed


def _file_digest(
    folder: ReachablePath, name: str, limit: int
) -> tuple[dict[str, Any], int]:
    """Size and sha256 of a regular file, and how many bytes were read.

    A file over ``limit`` bytes gets its size only; at most ``limit`` + 1
    bytes are ever read.
    """
    try:
        stream = (folder / name).open("rb")
    except OSError:
        return {"unreadable": True}, 0
    size = 0
    try:
        fd = stream.fileno()
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return {"unreadable": True}, 0
        if info.st_size > limit:
            return {"bytes": info.st_size}, 0
        digest = hashlib.sha256()
        while chunk := os.read(fd, min(1 << 20, limit + 1 - size)):
            digest.update(chunk)
            size += len(chunk)
            if size > limit:
                # Grew while it was read: a live writer.
                return {"bytes": size}, size
        return {"bytes": size, "sha256": digest.hexdigest()}, size
    except OSError:
        return {"unreadable": True}, size
    finally:
        stream.close()


def _move_aside(canonical: Path, record_root: Path) -> Path | None:
    """Move the trial's ``verifier/`` out of the way; return where it went.

    On Docker that folder is the agent container's ``/logs/verifier`` bind
    mount, still live: the mount follows the moved folder, so a process left
    running there keeps writing into it, never into the published one. Moving
    a folder to another parent needs write permission on the folder itself,
    which root in the agent's container can take away; a rename beside it in
    the trial folder needs only the trial folder's, so that is the fallback.
    """
    if not canonical.exists() and not canonical.is_symlink():
        return None
    record_root.mkdir(parents=True, exist_ok=True)
    for parent in (record_root, canonical.parent):
        target = parent / AGENT_VERIFIER_DIR
        suffix = 0
        while target.exists() or target.is_symlink():
            suffix += 1
            target = parent / f"{AGENT_VERIFIER_DIR}-{suffix}"
        try:
            source = ReachablePath.directory(canonical.parent) / canonical.name
            destination = ReachablePath.directory(parent) / target.name
            source.rename(destination)
        except OSError:
            if parent == canonical.parent:
                raise
            continue
        # The agent sandbox's mount follows the folder: its teardown must
        # name it there, not read this move as a swap (DF-1).
        note_reachable_move(target)
        return target
    return None


def _publish_verifier_outputs(
    source: Path | ReachablePath | None, rollout_dir: Path, record: dict[str, Any]
) -> None:
    """Make the trial's ``verifier/`` exactly the verifier sandbox's outputs.

    Stage a full copy of ``source`` privately on the same filesystem, move the agent
    run's ``verifier/`` aside (:func:`_move_aside`), and rename the copy into
    place, as verifier recovery publishes. Nothing the agent's run left there
    survives, and a plant named like a verifier output cannot block the copy.
    The moved folder is listed in ``record["planted_in_verifier_dir"]`` and the
    published files in ``record["published_verifier_files"]``. ``source`` None
    (the verifier never ran) leaves ``verifier/`` empty. A copy that fails
    still moves the agent's folder aside, leaves ``verifier/`` empty, and then
    raises.
    """
    # Use the same lexical absolute spelling as the already-pinned output root.
    # Do not resolve mutable links or relax destination traversal checks.
    rollout_dir = Path(os.path.abspath(rollout_dir))
    canonical = rollout_dir / "verifier"
    failure: BaseException | None = None
    # The copy is staged in a folder proven unreachable from every sandbox, on
    # the trial folder's filesystem, and renamed into place through held
    # folders: never inside the trial folder when a sandbox mounts all of it
    # (Apple's /logs; reachable_write staging, R10-1).
    with private_folder(canonical) as private:
        staged: Path | None = None
        if source is not None and source.is_dir():
            staged = private / "verifier"
            try:
                if isinstance(source, ReachablePath):
                    source.copy_to(staged)
                else:
                    snapshot_tree(source, staged)
            except Exception as exc:
                failure = exc
                shutil.rmtree(staged, ignore_errors=True)
                staged = None
        moved = _move_aside(canonical, rollout_dir / RECORD_DIR)
        if moved is not None:
            planted = _list_tree(moved)
            planted["moved_to"] = moved.relative_to(rollout_dir).as_posix()
            record["planted_in_verifier_dir"] = planted
        if staged is not None:
            _rename_into(staged, rollout_dir, "verifier")
        else:
            destination(canonical).mkdir()
    record["published_verifier_files"] = _list_tree(canonical)
    if failure is not None:
        raise failure


def _forget_verifier_folder(rollout: Any) -> None:
    """The trial's ``verifier/`` was replaced: drop its held capability.

    Later controller writes (rewards, timed_out) then open the published
    folder, not the agent's run's folder moved aside under ``RECORD_DIR``.
    """
    paths = getattr(rollout, "_rollout_paths", None)
    forget = getattr(paths, "forget_held", None)
    if callable(forget):
        forget("verifier")


def _rename_into(staged: Path, folder: Path, name: str) -> None:
    """Publish with both source staging and destination ancestors retained."""
    destination(folder / name).publish(staged)


async def _publish_off_loop(
    source: Path | ReachablePath | None, rollout_dir: Path, record: dict[str, Any]
) -> None:
    """Run :func:`_publish_verifier_outputs` in a worker thread.

    It reads and hashes files the agent could write, which must not stall
    the job's event loop and every other trial on it. A cancel waits for the
    thread to finish before it is raised, so ``verifier/`` and ``record`` are
    never changed after the caller has moved on.
    """
    work = asyncio.ensure_future(
        asyncio.to_thread(_publish_verifier_outputs, source, rollout_dir, record)
    )
    try:
        await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait({work})
        if not work.cancelled():
            work.exception()  # retrieved: the cancel is what propagates
        raise


def _use_verifier_layer(
    rollout: Any, env: Any, task: Any, vtask: Any
) -> dict[str, Any] | None:
    """Start the verifier sandbox from the task's verifier layer (verifier_layers).

    The cache preserves the verifier image and adds labels only, keyed by
    the task and test files alone and built once per key: concurrent trials,
    of this process or another, share it, each from its own alias tag. It is
    planned from the task, never from anything the agent's sandbox held.
    Returns the record of why there is none, if there is none.
    """
    from benchflow.sandbox.env_layers import fresh_install_requested
    from benchflow.sandbox.verifier_layers import (
        plan_verifier_layer,
        verifier_layer_requested,
    )

    use = getattr(type(env), "use_verifier_layer", None)
    if not callable(use):
        if not verifier_layer_requested():
            # Off (the default): no record, as on a backend with layers.
            return None
        return {
            "status": "unsupported",
            "reason": f"the {rollout._config.environment} verifier sandbox builds "
            "no layers; its image is built or pulled per trial",
        }
    recipe, record = plan_verifier_layer(
        task,
        separate=True,
        fresh_install=fresh_install_requested(
            getattr(rollout._config, "fresh_install", None)
        ),
        allow_internet=bool(getattr(vtask.config.sandbox, "allow_internet", True)),
        sandbox_env=getattr(vtask.config.sandbox, "env", None),
    )
    use(env, recipe)
    return record


def _verifier_layer_record(
    env: Any, planned: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The verifier sandbox's layer record: hit or miss, key, seconds, seed."""
    from benchflow.sandbox.env_layers import VERIFIER

    records = getattr(env, "layer_records", None)
    if isinstance(records, dict) and isinstance(records.get(VERIFIER), dict):
        return dict(records[VERIFIER])
    return planned


async def _stop(env: Any) -> str | None:
    """Keep one teardown task alive until it settles, even on repeated cancel."""
    work = asyncio.create_task(_stop_with_timeout(env))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait({work})
        if not work.cancelled():
            work.exception()
        raise


async def _stop_with_timeout(env: Any) -> str | None:
    try:
        await asyncio.wait_for(env.stop(delete=True), timeout=_STOP_TIMEOUT_SEC)
    except Exception as exc:
        logger.warning("Verifier sandbox cleanup failed", exc_info=True)
        return f"{type(exc).__name__}: {exc}"
    return None


#: ``BENCHFLOW_VERIFIER_PREPARE=serial`` keeps the base order in separate mode:
#: the verifier sandbox starts after the transfer is packed and the agent's
#: sandbox stops in ``cleanup()``, after the verifier. For A/B arms and rollback.
VERIFIER_PREPARE_ENV = "BENCHFLOW_VERIFIER_PREPARE"

#: The order a separate verifier runs in (``order`` in its record).
SERIAL = "serial"
OVERLAP = "overlap"

#: Timing keys of the agent's release that the record copies beside the
#: verifier's own.
_RELEASE_TIMING = (
    timing_spans.AGENT_RELEASE,
    timing_spans.AGENT_SANDBOX_STOP_BEFORE_VERIFY,
)


def serial_order_requested() -> bool:
    """Whether ``BENCHFLOW_VERIFIER_PREPARE=serial`` asks for the base order."""
    return os.environ.get(VERIFIER_PREPARE_ENV, "").strip().lower() == SERIAL


#: ``BENCHFLOW_VERIFIER_PREPARE=entry`` starts the verifier sandbox at verify()
#: entry, as before the warm-up (the overlap order without it).
ENTRY = "entry"


def warmup_requested() -> bool:
    """Whether the verifier sandbox may start as the agent's turn begins.

    The default on Docker and remote Docker; ``BENCHFLOW_VERIFIER_PREPARE``
    set to ``serial`` or ``entry`` keeps the later starts.
    """
    value = os.environ.get(VERIFIER_PREPARE_ENV, "").strip().lower()
    return value not in (SERIAL, ENTRY)


# The check after the agent sandbox's stop: one ``docker ps`` or one Daytona
# lookup.
_AGENT_GONE_CHECK_TIMEOUT_SEC = 30
# Only a terminal state proves release: DESTROYING and desired_state are
# deletion requests, and can persist after the confirmed-delete wait times out.
_DAYTONA_GONE_STATES = frozenset({"destroyed"})


def _state_name(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _daytona_gone(handle: Any) -> bool:
    return _state_name(getattr(handle, "state", None)) in _DAYTONA_GONE_STATES


async def agent_sandbox_left_running(env: Any, handle: Any = None) -> str | None:
    """Why the agent's sandbox may still be up after its stop, or None.

    The providers do not raise when their delete fails: Daytona logs the
    sandbox as left running, Docker's forced kill and remote Docker's leftover
    removal only log. So the stop's return is checked here: on Docker and
    remote Docker, the compose project's containers (only running ones when
    ``keep_containers`` stopped them on purpose); on Daytona, the sandbox
    ``handle`` taken before the stop (the stop clears the sandbox's own,
    deleted or not), whose state the SDK refreshes after a delete; on Modal,
    that handle's ``poll()`` (an exit code once the sandbox ended). A check
    that cannot answer counts as up.
    """
    from benchflow.sandbox.docker import DockerSandbox

    # Looked up, not imported: a Modal sandbox means its module is loaded, and
    # the Modal extra's imports must not run for any other backend's check.
    modal_impl = sys.modules.get("benchflow.sandbox.modal_impl")
    try:
        if modal_impl is not None and isinstance(env, modal_impl.ModalSandbox):
            # The Modal Sandbox handle taken before the stop: poll() gives its
            # exit code once it has ended, None while it still runs.
            poll = getattr(getattr(handle, "poll", None), "aio", None)
            if not callable(poll):
                return "could not confirm the agent's Modal sandbox stopped (no handle)"
            code = await asyncio.wait_for(poll(), timeout=_AGENT_GONE_CHECK_TIMEOUT_SEC)
            if code is None:
                return (
                    f"Modal sandbox {getattr(handle, 'object_id', '?')} is still "
                    "running after its terminate"
                )
            return None
        if isinstance(env, DockerSandbox):
            project = env.compose_project_name
            kept = bool(getattr(env, "_keep_containers", False))
            result = await asyncio.wait_for(
                env._docker_cli(
                    [
                        "ps",
                        "-q" if kept else "-aq",
                        "--filter",
                        f"label=com.docker.compose.project={project}",
                    ],
                    check=False,
                    raw_output=True,
                ),
                timeout=_AGENT_GONE_CHECK_TIMEOUT_SEC,
            )
            if result.return_code != 0:
                return (
                    f"could not list compose project {project}'s containers "
                    f"after its stop (docker ps exited {result.return_code})"
                )
            left = str(result.stdout or "").split()
            if left:
                return (
                    f"{len(left)} container(s) of compose project {project} "
                    f"still {'running' if kept else 'there'} after its stop"
                )
            return None
        if handle is None or not callable(getattr(handle, "refresh_data", None)):
            if getattr(type(env), "requires_release_handle", False) is True:
                return "could not confirm the agent's sandbox stopped (no handle)"
            return None
        if _daytona_gone(handle):
            return None
        sandbox_id = getattr(handle, "id", None)
        try:
            await asyncio.wait_for(
                handle.refresh_data(), timeout=_AGENT_GONE_CHECK_TIMEOUT_SEC
            )
        except Exception as exc:
            if type(exc).__name__ == "DaytonaNotFoundError" or (
                getattr(exc, "status_code", None) == 404
            ):
                return None
            raise
        if _daytona_gone(handle):
            return None
        return (
            f"Daytona sandbox {sandbox_id} is still "
            f"{_state_name(getattr(handle, 'state', None)) or 'there'} after its "
            "delete"
        )
    except Exception as exc:
        return (
            f"could not confirm the agent's sandbox stopped: {describe_exception(exc)}"
        )


class VerifierSandboxRun:
    """One trial's separate verifier sandbox, from its plan to its deletion.

    In the overlap order (``Rollout.verify``) :meth:`begin_start` starts the
    sandbox from task files alone, while the agent's sandbox is still being
    captured (Docker, remote Docker) or right after it stopped (Daytona), and
    :meth:`finish` packs the transfer, waits for the sandbox, installs,
    verifies and publishes. In the serial order (:func:`run_separate_verifier`)
    :meth:`finish` does all of it in the base order. The sandbox is deleted
    exactly once: by :meth:`finish`, or by :meth:`discard` when the trial fails
    before it; ``verifier-sandbox.json`` is written then.

    ``source_dir`` is the trial folder whose saved outputs cross (a regrade's
    stored trial); by default the rollout's own. The record and the published
    ``verifier/`` are always the rollout's.
    """

    def __init__(
        self,
        rollout: Any,
        *,
        order: str = SERIAL,
        create_environment: CreateEnvironment | None = None,
        source_dir: Path | None = None,
    ) -> None:
        self.rollout = rollout
        self.order = order
        self._create = create_environment
        self.rollout_dir = Path(rollout._rollout_dir)
        self.source_dir = Path(source_dir) if source_dir else self.rollout_dir
        self.root = self.rollout_dir / RECORD_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.timing: dict[str, Any] = rollout._timing
        self.record: dict[str, Any] = {
            "mode": "separate",
            "backend": rollout._config.environment,
            "order": order,
            "status": "pending",
            "image_source": None,
            "sandbox_id": None,
            "transfer": None,
            # Files the agent phase wrote outside everything that crosses; see
            # scan_outside_workspace_writes. None when the scan did not run.
            "outside_workspace_writes": getattr(
                rollout, "_outside_workspace_writes", None
            ),
            "error": None,
            # The serial order stops the agent's sandbox in cleanup(), after
            # the verifier; the overlap order clears this once it stopped.
            "agent_sandbox_alive_during_verify": True,
        }
        self.plan: VerifierImage | None = None
        self._plan_error: BaseException | None = None
        self._planned = False
        self.paths: RolloutPaths | None = None
        self.vtask: Any = None
        self.env: Any = None
        self.start_task: asyncio.Task[None] | None = None
        # The trajectory the verifier reads, when taken apart from the
        # rollout's own (the overlap order).
        self.trajectory: list[dict] | None = None
        self._finish_started: float | None = None
        self._created_at: float | None = None
        self._agent_gone_at: float | None = None
        # The one deletion (see :meth:`_delete`); ``_deleted`` once it is done.
        self._delete_task: asyncio.Future[None] | None = None
        self._deleted = False
        self._charge = None
        self._layer_plan: dict[str, Any] | None = None
        # A warm-up start (the agent's turn begins): not charged until
        # verify() enters (begin_billing), so the warm-up never counts
        # against max_sandbox_seconds (PERF-C review M1).
        self._warm = False
        self._billed_from: float | None = None

    # --- before the transfer --------------------------------------------------

    def check_plan(self) -> None:
        """Plan the verifier image from the task's files; :meth:`finish` raises
        a failure, in the base order's place."""
        if self._planned:
            return
        self._planned = True
        task = self.rollout._task
        try:
            plan = plan_verifier_image(task.config, Path(task.paths.task_dir))
            self.record["image_source"] = plan.source
            services = verifier_context_services(plan)
            if services:
                # The launch gate refuses this; an SDK caller may not have run it.
                raise SeparateVerifierError(
                    f"image from {plan.source} would start compose services "
                    f"({', '.join(services)}); a separate verifier sandbox has none"
                )
            self.plan = plan
        except Exception as exc:
            self._plan_error = exc

    def begin_start(self, *, warm: bool = False) -> None:
        """Start the verifier sandbox as a task (nothing when its plan failed).

        ``warm``: the warm-up's start as the agent's turn begins; its charge
        and its overlap count only from :meth:`begin_billing`.
        """
        self.check_plan()
        if self.plan is not None and self.start_task is None:
            self._warm = warm
            self.start_task = asyncio.create_task(
                self._start(), name=f"{self.rollout._rollout_name}-verifier-start"
            )

    def begin_billing(self) -> None:
        """verify() entered: a warmed-up sandbox is charged from now on.

        The seconds it existed before (during the agent's turn) are recorded
        as ``verifier_sandbox_warmup`` and never charged; from here it is
        charged as a start at verify() entry was. A no-op for any other start.
        """
        if not self._warm or self._billed_from is not None:
            return
        from benchflow.budget import start_verifier_charge

        self._billed_from = timing_spans.clock()
        if self._deleted:
            return
        self._charge = start_verifier_charge()
        if self._agent_gone_at is not None:
            self._charge.finish()

    async def _start(self) -> None:
        """Stage the image's context, create the sandbox and start it.

        Task files only: nothing from the agent's run is read here.
        """
        from benchflow.budget import start_verifier_charge
        from benchflow.sandbox.metadata import persist_sandbox_info

        rollout = self.rollout
        task = rollout._task
        assert self.plan is not None
        paths = RolloutPaths(rollout_dir=self.root)
        paths.mkdir()
        self.paths = paths
        self.vtask = _verifier_task(task, self.plan)
        context = await asyncio.to_thread(
            _stage_context,
            self.root,
            task.name,
            self.plan,
            planes=getattr(rollout, "_planes", None),
            context_root=getattr(rollout._config, "context_root", None),
            base_image_override=getattr(rollout._config, "base_image_override", None),
        )
        create = self._create or _default_create(rollout)
        if not self._warm:
            self._charge = start_verifier_charge()
        self.env = create(self.vtask, context, paths)
        self._layer_plan = _use_verifier_layer(rollout, self.env, task, self.vtask)
        t0 = self._created_at = timing_spans.clock()
        if self._agent_gone_at is not None and self._charge is not None:
            self._charge.finish()
        try:
            await self.env.start(force_build=False)
            # V2's protected verifier runtime, from the fresh image before any
            # of the agent's bytes arrive. The workspace is the declared one
            # (the agent's cwd), which the transfer's manifest must match.
            await rollout._planes.prepare_separate_verifier_runtime(
                self.env, task=self.vtask, workspace=str(rollout._agent_cwd)
            )
        except Exception as exc:
            # finish() sets the status: a transfer failure found meanwhile
            # comes first, as in the serial order.
            raise SeparateVerifierError(
                f"sandbox failed to start: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            self.timing["verifier_sandbox_setup"] = timing_spans.clock() - t0
        sandbox_id = getattr(self.env, "sandbox_id", None)
        self.record["sandbox_id"] = sandbox_id if isinstance(sandbox_id, str) else None
        persist_sandbox_info(self.env, self.root)

    def snapshot_trajectory(self, trajectory: list[dict] | None) -> None:
        """Keep the trajectory the verifier will read.

        Taken after the scrape and before the agent's release, whose ACP
        reconcile may repair events in the rollout's own copy: the verifier
        reads what it read when it ran before that reconcile.
        """
        self.trajectory = copy.deepcopy(list(trajectory or []))

    def agent_stopped(self, error: str | None) -> None:
        """Record the agent sandbox's stop before the transfer.

        ``error`` is None when it stopped; otherwise it is still billed until
        ``cleanup()`` stops it, and counts as overlap the whole time.
        """
        self.record.setdefault("custody", {})["agent_stop_error"] = error
        if error is None:
            self._agent_gone_at = timing_spans.clock()
            if self._charge is not None:
                self._charge.finish()
            self.record["agent_sandbox_alive_during_verify"] = False

    # --- the transfer, the verifier, the deletion -----------------------------

    async def finish(
        self, verify: Verify | None = None
    ) -> tuple[dict | None, str | None]:
        """Transfer, verify, publish and delete; return ``(rewards, error)``."""
        rollout = self.rollout
        task = rollout._task
        record, timing = self.record, self.timing
        archive = self.root / "transfer.tar"
        rewards: dict | None = None
        error: str | None = None
        published = False
        self._finish_started = timing_spans.clock()
        try:
            self.check_plan()
            if self._plan_error is not None:
                raise self._plan_error
            record["pristine"] = getattr(rollout, "_pristine_outputs", None)
            # The overlap order makes this run before the workspace capture
            # lists the outside writes: read the listing now, after it.
            record["outside_workspace_writes"] = getattr(
                rollout, "_outside_workspace_writes", None
            )
            t0 = timing_spans.clock()
            try:
                check_declared_transfer(
                    getattr(rollout, "_agent_cwd", None), task.config.artifacts
                )
                summary = await asyncio.to_thread(
                    build_transfer_payload,
                    self.source_dir,
                    archive,
                    pristine=record["pristine"],
                    capture_over_limit=getattr(rollout, "_capture_over_limit", False),
                    artifacts=_capture_declarations(rollout),
                    workspace=str(rollout._agent_cwd)
                    if getattr(rollout, "_agent_cwd", None)
                    else None,
                    # A task that declares no artifacts gets its workspace,
                    # as BenchFlow always sent it: the format's verifier
                    # would get nothing from the agent.
                    artifacts_only=self.plan is not None
                    and _format_verifier(task, self.plan)
                    and bool(task.config.artifacts),
                )
            except SolutionTransferRefused:
                raise
            except SeparateVerifierError as exc:
                capture = getattr(rollout, "_export_error", None)
                detail = f" ({capture})" if capture else ""
                raise SeparateVerifierError(f"transfer failed: {exc}{detail}") from exc
            finally:
                timing["verifier_pack"] = timing_spans.clock() - t0
            record["transfer"] = {
                k: v for k, v in summary.items() if k not in ("sha256", "custody")
            }
            custody = record["custody"] = {
                **record.get("custody", {}),
                **summary["custody"],
            }
            _warn_left_out(
                summary,
                getattr(
                    getattr(rollout, "_config", None), "separate_verifier_include", ()
                ),
            )
            if self.start_task is None:
                await self._start()
            else:
                with timing_spans.span(
                    timing_spans.VERIFIER_SANDBOX_READY_WAIT, into=timing
                ):
                    await self.start_task
            env, paths = self.env, self.paths
            assert paths is not None
            # Seal utility copies in the pristine target, before transferring any
            # bytes the agent could have changed (including mounted tool aliases).
            await rollout._planes.snapshot_build_config(
                env, workspace=summary["workspace"]
            )
            t0 = timing_spans.clock()
            try:
                custody["destinations"] = await _check_destinations(
                    env,
                    archive,
                    summary,
                    int(rollout._config.sandbox_setup_timeout or 600),
                )
                custody["sha256_checked_in_sandbox"] = await _install_payload(
                    env,
                    archive,
                    summary,
                    int(rollout._config.sandbox_setup_timeout or 600),
                )
                # Sidecar artifacts land at their source paths, as the
                # task.toml format's verifier reads them (_collect_hooks).
                await install_sidecar_artifacts(env, self.rollout_dir, custody)
                # They are agent input too, even when no main-container
                # artifact happens to share their parent directory.
                # The transfer record shares this list with the inventory.
                summary["agent_paths"].extend(
                    path
                    for path in custody.get("sidecar_artifacts", {}).get("crossed", [])
                    if path not in summary["agent_paths"]
                )
            except SeparateVerifierError as exc:
                raise SeparateVerifierError(f"transfer failed: {exc}") from exc
            finally:
                timing["verifier_transfer"] = timing_spans.clock() - t0
            run = verify or _default_verify(rollout, self.trajectory)
            rewards, error = await run(
                env,
                self.vtask,
                paths,
                timing,
                workspace=summary["workspace"],
                agent_paths=tuple(summary["agent_paths"]),
            )
            custody["trajectory"] = _trajectory_record(rollout, paths)
            published = True
            try:
                await _publish_off_loop(paths.verifier_dir, self.rollout_dir, record)
            except Exception as exc:
                if verifier_infrastructure_failure(exc):
                    raise SeparateVerifierError(
                        "verifier output publication failed: "
                        + verifier_failure_reason(exc)
                    ) from exc
                # The verifier user can plant a FIFO, unreadable folder or
                # changing tree here. Publication cannot discard its score.
                warning = "verifier output publication failed: " + describe_exception(
                    exc
                )
                if error is None:
                    warning = "Verifier scored 0: " + warning
                logger.warning(warning)
                vars(rollout).setdefault("_warnings", []).append(warning)
                record["publication_warning"] = warning
                if error is None:
                    rewards = {"reward": 0.0}
            _forget_verifier_folder(rollout)
            record["status"] = "complete" if error is None else "verifier_failed"
        except SolutionTransferRefused as exc:
            # The solution's own files: its result, not a failure to assess it.
            # No verifier error, so the retry loop cannot resample it away.
            rewards, error = {"reward": 0.0}, None
            record["status"] = "refused"
            record["refusal"] = (
                f"the solution's outputs cannot reach the verifier: {exc}"
            )
            logger.warning("%s; scored 0", record["refusal"])
        except SeparateVerifierError as exc:
            rewards, error = None, f"{ERROR_PREFIX} {exc}"
            if record["status"] == "pending":
                record["status"] = (
                    "transfer_failed"
                    if "transfer failed" in str(exc)
                    else "sandbox_failed"
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
            # Publication and settling the prestart are part of custody too:
            # a cancellation at either await must still drain the one deletion.
            work = asyncio.create_task(self._finalize(published, archive, error))
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                record["status"] = "interrupted"
                while not work.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait({work})
                if not work.cancelled():
                    work.exception()
                raise
        return rewards, error

    async def _finalize(
        self, published: bool, archive: Path, error: str | None
    ) -> None:
        try:
            try:
                if not published:
                    try:
                        await _publish_off_loop(None, self.rollout_dir, self.record)
                        _forget_verifier_folder(self.rollout)
                    except Exception as exc:
                        logger.warning(
                            "Could not move the agent's verifier/ aside: %s", exc
                        )
                        self.record["publish_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    await self._settle_start(
                        cancel=self.record["status"] == "interrupted"
                    )
                finally:
                    await self._delete()
        finally:
            archive.unlink(missing_ok=True)
            self.record["error"] = error
            self._write_record()

    async def discard(self, *, cancelled: bool) -> None:
        """Delete the verifier sandbox of a trial that failed before :meth:`finish`.

        A start still running is cancelled (on a cancellation) or awaited
        first, so a sandbox it creates meanwhile is deleted too. The trial's
        own exception is what its caller raises; this keeps the error and
        status the trial gets without a separate verifier.
        """
        if self._deleted:
            return
        self.record["status"] = "interrupted" if cancelled else "abandoned"
        work = asyncio.create_task(self._discard(cancelled=cancelled))
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            self.record["status"] = "interrupted"
            while not work.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait({work})
            if not work.cancelled():
                work.exception()
            raise

    async def _discard(self, *, cancelled: bool) -> None:
        try:
            await self._settle_start(cancel=cancelled)
        finally:
            try:
                await self._delete()
            finally:
                self._write_record()

    async def _settle_start(self, *, cancel: bool = False) -> None:
        task = self.start_task
        if task is None:
            return
        if not task.done():
            if cancel:
                task.cancel()
            await asyncio.wait({task})
        if not task.cancelled():
            task.exception()  # retrieved: whoever awaited it reported it

    async def _delete(self) -> None:
        """Delete the verifier sandbox, once, and record its seconds.

        ``sandbox_seconds`` is its life, from the start of ``env.start`` to the
        deletion's return. In the overlap order ``verifier_sandbox_overlap`` is
        the part of that life the agent's sandbox still existed: until its stop
        returned, or the whole life when that stop failed (``cleanup()`` stops
        it later).

        The deletion runs as one task that a cancel of the caller does not
        reach: the cancel waits for it to finish and is raised after, so a
        sandbox whose stop was under way is not left running (a second call
        joins the same task, so ``env.stop`` runs once).
        """
        if self._deleted:
            return
        if self._delete_task is None:
            self._delete_task = asyncio.ensure_future(self._teardown())
        work = self._delete_task
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            while not work.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait({work})
            if not work.cancelled():
                work.exception()  # retrieved: the cancel is what propagates
            raise

    async def _teardown(self) -> None:
        record, timing = self.record, self.timing
        if self.env is None:
            if self._charge is not None:
                self._charge.finish()
            self._deleted = True
            record["sandbox_seconds"] = 0.0
            if self.order == OVERLAP:
                timing[timing_spans.VERIFIER_SANDBOX_OVERLAP] = 0.0
            return
        t0 = timing_spans.clock()
        try:
            cleanup_error = await _stop(self.env)
        finally:
            if self._charge is not None:
                self._charge.finish()
        deleted_at = timing_spans.clock()
        self._deleted = True
        timing["verifier_sandbox_teardown"] = deleted_at - t0
        if cleanup_error:
            record["cleanup_error"] = cleanup_error
        created_at = self._created_at if self._created_at is not None else t0
        record["sandbox_seconds"] = round(deleted_at - created_at, 3)
        if self.order == OVERLAP:
            timing["verifier_sandbox_total"] = deleted_at - created_at
            gone = deleted_at
            if self._agent_gone_at is not None:
                gone = min(self._agent_gone_at, deleted_at)
            # A warmed-up sandbox counts from verify() entry (begin_billing);
            # its seconds during the agent's turn are only recorded.
            counted_from = created_at
            if self._warm:
                billed = self._billed_from
                counted_from = max(created_at, deleted_at if billed is None else billed)
                timing["verifier_sandbox_warmup"] = max(
                    0.0, min(counted_from, deleted_at) - created_at
                )
            timing[timing_spans.VERIFIER_SANDBOX_OVERLAP] = max(
                0.0, gone - counted_from
            )
        else:
            timing["verifier_sandbox_total"] = deleted_at - created_at

    def _write_record(self) -> None:
        from benchflow.review.persistence import write_json_atomic

        if self.env is not None:
            self.record["verifier_layer"] = _verifier_layer_record(
                self.env, self._layer_plan
            )
        self.record["timing"] = {
            k: round(v, 3)
            for k, v in self.timing.items()
            if isinstance(v, int | float)
            and (k.startswith("verifier") or k in _RELEASE_TIMING)
        }
        write_json_atomic(self.root / RECORD_FILE, self.record)


async def run_separate_verifier(
    rollout: Any,
    *,
    create_environment: CreateEnvironment | None = None,
    verify: Verify | None = None,
) -> tuple[dict | None, str | None]:
    """Score ``rollout`` in a fresh verifier sandbox, in the serial order.

    The verifier sandbox starts after the transfer is packed, and the agent's
    sandbox is still up while it runs (an externally owned sandbox,
    ``BENCHFLOW_VERIFIER_PREPARE=serial``, a verify() outside the rollout's
    own lifecycle). Returns ``(rewards, error)``.
    """
    run = VerifierSandboxRun(
        rollout, order=SERIAL, create_environment=create_environment
    )
    return await run.finish(verify)


def read_record(rollout_dir: Path) -> dict[str, Any] | None:
    """The separate verifier record of a finished rollout, if it ran one."""
    path = Path(rollout_dir) / RECORD_DIR / RECORD_FILE
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
