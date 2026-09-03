"""Capture final agent-created workspace files for isolated rubric review.

Large benchmark workspaces often contain gigabytes of immutable task inputs.
Copying the entire directory would both duplicate those inputs and risk
including agent credentials. Instead, Benchflow records a metadata baseline
after task upload and exports every regular file created or changed afterward.
The reviewer receives that delta plus an auditable manifest.
"""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import PurePosixPath
from typing import Any

WORKSPACE_DIRNAME = "workspace"
WORKSPACE_BASELINE_FILENAME = "workspace-baseline.json"
WORKSPACE_MANIFEST_FILENAME = "workspace-manifest.json"

_CAPTURE_SCRIPT = r"""
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

mode, root_arg, baseline_arg, destination_arg, manifest_arg = sys.argv[1:]
root = Path(root_arg).resolve()
baseline_path = Path(baseline_arg)
destination = Path(destination_arg)
manifest_path = Path(manifest_arg)

EXCLUDED_ROOT_DIRS = {
    ".aws", ".azure", ".cache", ".claude", ".codex", ".config", ".daytona",
    ".gemini", ".gnupg", ".kube", ".local", ".npm", ".ssh",
}
EXCLUDED_DIR_NAMES = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "node_modules",
}
EXCLUDED_FILE_NAMES = {
    ".gitconfig", ".netrc", ".npmrc", ".pypirc", "auth.json", "credentials",
    "credentials.json", "id_ed25519", "id_rsa",
}


def exclusion(relative):
    parts = relative.parts
    if not parts:
        return None
    if any(part in EXCLUDED_ROOT_DIRS for part in parts):
        return "runtime credential/cache directory"
    if any(part in EXCLUDED_DIR_NAMES for part in parts[:-1]):
        return "generated dependency or VCS directory"
    name = parts[-1]
    lower = name.lower()
    if lower == ".env" or lower.startswith(".env."):
        return "environment secret file"
    if lower in EXCLUDED_FILE_NAMES or lower.endswith((".pem", ".key", ".p12", ".pfx")):
        return "credential-like file"
    return None


def inventory():
    files = {}
    excluded = {}
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        relative_dir = directory_path.relative_to(root)
        kept_dirs = []
        for name in sorted(dirnames):
            relative = relative_dir / name
            reason = exclusion(relative)
            candidate = directory_path / name
            if reason is not None:
                excluded[relative.as_posix()] = reason
            elif candidate.is_symlink():
                excluded[relative.as_posix()] = "symbolic link"
            else:
                kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            candidate = directory_path / name
            relative = candidate.relative_to(root)
            reason = exclusion(relative)
            if reason is not None:
                excluded[relative.as_posix()] = reason
                continue
            try:
                info = candidate.lstat()
            except OSError:
                excluded[relative.as_posix()] = "unreadable"
                continue
            if not stat.S_ISREG(info.st_mode):
                excluded[relative.as_posix()] = "non-regular file"
                continue
            files[relative.as_posix()] = {
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }
    return files, excluded


current, excluded = inventory()
if mode == "baseline":
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps({"version": 1, "root": str(root), "files": current}, sort_keys=True),
        encoding="utf-8",
    )
    raise SystemExit(0)

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
if baseline.get("version") != 1 or baseline.get("root") != str(root):
    raise RuntimeError("workspace baseline does not match the final workspace")
before = baseline.get("files")
if not isinstance(before, dict):
    raise RuntimeError("workspace baseline has no file inventory")

if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)
copied = []
tree_hash = hashlib.sha256()
for relative_text in sorted(current):
    if before.get(relative_text) == current[relative_text]:
        continue
    source = root / Path(relative_text)
    target = destination / Path(relative_text)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    item = {
        "path": relative_text,
        "size": current[relative_text]["size"],
        "sha256": digest.hexdigest(),
    }
    copied.append(item)
    tree_hash.update(
        (relative_text + "\0" + item["sha256"] + "\n").encode("utf-8")
    )

manifest = {
    "version": 1,
    "source_workspace": str(root),
    "capture": "files created or modified after task upload",
    "copied_files": copied,
    "copied_file_count": len(copied),
    "copied_bytes": sum(item["size"] for item in copied),
    "deleted_files": sorted(set(before) - set(current)),
    "excluded_paths": [
        {"path": path, "reason": excluded[path]} for path in sorted(excluded)
    ],
    "tree_sha256": tree_hash.hexdigest(),
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
baseline_path.unlink(missing_ok=True)
"""


def _capture_command(
    mode: str,
    *,
    workspace: str,
    baseline_path: str,
    destination_path: str,
    manifest_path: str,
) -> str:
    encoded = base64.b64encode(_CAPTURE_SCRIPT.encode("utf-8")).decode("ascii")
    args = " ".join(
        shlex.quote(value)
        for value in (
            mode,
            workspace,
            baseline_path,
            destination_path,
            manifest_path,
        )
    )
    return f"printf %s {shlex.quote(encoded)} | base64 -d | python3 - {args}"


def _artifact_paths() -> tuple[str, str, str]:
    root = PurePosixPath("/logs/artifacts")
    return (
        str(root / WORKSPACE_BASELINE_FILENAME),
        str(root / WORKSPACE_DIRNAME),
        str(root / WORKSPACE_MANIFEST_FILENAME),
    )


async def record_workspace_baseline(sandbox: Any, workspace: str) -> None:
    """Record the pre-agent workspace inventory inside the sandbox."""

    baseline, destination, manifest = _artifact_paths()
    result = await sandbox.exec(
        _capture_command(
            "baseline",
            workspace=workspace,
            baseline_path=baseline,
            destination_path=destination,
            manifest_path=manifest,
        ),
        user="root",
        timeout_sec=300,
    )
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or "")[-500:]
        raise RuntimeError(f"workspace baseline capture failed: {detail}")


async def export_workspace_delta(
    sandbox: Any,
    workspace: str,
    *,
    host_artifacts_dir: Any,
) -> None:
    """Export created/modified workspace files into rollout artifacts."""

    baseline, destination, manifest = _artifact_paths()
    result = await sandbox.exec(
        _capture_command(
            "final",
            workspace=workspace,
            baseline_path=baseline,
            destination_path=destination,
            manifest_path=manifest,
        ),
        user="root",
        timeout_sec=900,
    )
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or "")[-500:]
        raise RuntimeError(f"workspace output capture failed: {detail}")
    if not getattr(sandbox, "is_mounted", False):
        host_artifacts_dir.mkdir(parents=True, exist_ok=True)
        host_workspace = host_artifacts_dir / WORKSPACE_DIRNAME
        host_workspace.mkdir(parents=True, exist_ok=True)
        await sandbox.download_dir(destination, host_workspace)
        await sandbox.download_file(
            manifest,
            host_artifacts_dir / WORKSPACE_MANIFEST_FILENAME,
        )


def read_workspace_manifest(rollout_dir: Any) -> dict[str, Any] | None:
    """Read a captured workspace manifest from a host rollout directory."""

    path = rollout_dir / "artifacts" / WORKSPACE_MANIFEST_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
