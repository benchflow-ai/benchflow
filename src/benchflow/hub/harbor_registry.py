"""Harbor registry compatibility inventory and smoke checks."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from benchflow._utils.task_authoring import check_task

DEFAULT_HARBOR_REGISTRY_URL = (
    "https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json"
)
DEFAULT_HARBOR_HUB_URL = "https://hub.harborframework.com/"


@dataclass(frozen=True, slots=True)
class HarborTaskRef:
    """One task entry selected from Harbor's registry."""

    dataset: str
    version: str | None
    task: str
    git_url: str
    git_commit_id: str | None
    path: str
    index: int


def load_harbor_registry(source: str | Path) -> list[dict[str, Any]]:
    """Load Harbor's registry from a URL or local JSON file."""
    source_text = str(source)
    parsed = urlparse(source_text)
    if parsed.scheme in {"http", "https"}:
        with urlopen(source_text, timeout=30) as response:
            raw = response.read().decode("utf-8")
    else:
        path = Path(source).expanduser()
        raw = path.read_text()

    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Harbor registry must be a JSON list")
    return data


def select_harbor_tasks(
    registry: list[dict[str, Any]],
    *,
    tasks_per_dataset: int = 2,
) -> list[HarborTaskRef]:
    """Select up to ``tasks_per_dataset`` task refs per registry dataset."""
    if tasks_per_dataset < 1:
        raise ValueError("tasks_per_dataset must be >= 1")

    selected: list[HarborTaskRef] = []
    for dataset in registry:
        name = str(dataset.get("name") or "")
        if not name:
            continue
        version = dataset.get("version")
        tasks = dataset.get("tasks") or []
        if not isinstance(tasks, list):
            continue
        for index, task in enumerate(tasks[:tasks_per_dataset]):
            if not isinstance(task, dict):
                continue
            task_name = str(task.get("name") or "")
            git_url = str(task.get("git_url") or "")
            path = str(task.get("path") or "")
            if not task_name or not git_url or not path:
                continue
            selected.append(
                HarborTaskRef(
                    dataset=name,
                    version=str(version) if version is not None else None,
                    task=task_name,
                    git_url=git_url,
                    git_commit_id=(
                        str(task["git_commit_id"])
                        if task.get("git_commit_id") is not None
                        else None
                    ),
                    path=path,
                    index=index,
                )
            )
    return selected


def check_harbor_registry(
    registry_source: str | Path = DEFAULT_HARBOR_REGISTRY_URL,
    *,
    tasks_per_dataset: int = 2,
    level: str = "inventory",
    out: Path | None = None,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run a Harbor compatibility inventory or structural task check."""
    if level not in {"inventory", "check"}:
        raise ValueError("level must be one of: inventory, check")

    registry = load_harbor_registry(registry_source)
    refs = select_harbor_tasks(registry, tasks_per_dataset=tasks_per_dataset)
    if limit is not None:
        refs = refs[:limit]

    cache = cache_dir or Path(".cache") / "hub" / "harbor"
    records = [_record_for_ref(ref, level=level, cache_dir=cache) for ref in refs]

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    return records


def records_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load compatibility records from a JSONL report."""
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record on line {line_no} must be an object")
        records.append(record)
    return records


def _record_for_ref(
    ref: HarborTaskRef,
    *,
    level: str,
    cache_dir: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "framework": "harbor",
        "env_uid": harbor_env_uid(ref),
        "hub_url": DEFAULT_HARBOR_HUB_URL,
        "dataset": ref.dataset,
        "version": ref.version,
        "task": ref.task,
        "task_index": ref.index,
        "git_url": ref.git_url,
        "source_repo": _repo_label(ref.git_url),
        "source_ref": ref.git_commit_id,
        "source_path": ref.path,
        "level": level,
        "badges": ["parse"],
        "status": "pass",
        "blocked_reason": None,
        "notes": [],
    }

    if level == "inventory":
        return record

    try:
        task_dir = materialize_harbor_task(ref, cache_dir=cache_dir)
    except Exception as exc:
        record["status"] = "blocked"
        record["blocked_reason"] = str(exc)
        return record

    if task_dir.exists():
        record["badges"].append("package")
    issues = check_task(task_dir)
    if issues:
        record["status"] = "fail"
        record["notes"] = issues
    else:
        record["badges"].append("check")
    return record


def harbor_env_uid(ref: HarborTaskRef) -> str:
    """Return the hosted-environment UID for a Harbor registry task ref."""
    version = ref.git_commit_id or ref.version or "HEAD"
    return f"harbor:{ref.dataset}/{ref.task}@{version}"


def materialize_harbor_task(ref: HarborTaskRef, *, cache_dir: Path) -> Path:
    """Return a local path for a Harbor task ref.

    Local-path ``git_url`` values are supported for tests and local smoke runs.
    Remote git URLs are sparse-cloned and cached by URL/ref hash.
    """
    local = _local_repo_path(ref.git_url)
    if local is not None:
        return local / ref.path

    repo_dir = _cached_repo_dir(ref, cache_dir)
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            ["clone", "--filter=blob:none", "--sparse", ref.git_url, str(repo_dir)]
        )

    checkout = ref.git_commit_id
    if checkout and checkout != "HEAD":
        _run_git(["-C", str(repo_dir), "fetch", "--depth", "1", "origin", checkout])
        _run_git(["-C", str(repo_dir), "checkout", checkout])

    _run_git(["-C", str(repo_dir), "sparse-checkout", "set", ref.path])
    return repo_dir / ref.path


def _cached_repo_dir(ref: HarborTaskRef, cache_dir: Path) -> Path:
    key = hashlib.sha256(
        f"{ref.git_url}@{ref.git_commit_id or 'HEAD'}".encode()
    ).hexdigest()[:16]
    return cache_dir / _safe_name(_repo_label(ref.git_url)) / key


def _local_repo_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _repo_label(git_url: str) -> str:
    parsed = urlparse(git_url)
    path = parsed.path if parsed.scheme else git_url
    path = path.removesuffix(".git").strip("/")
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return path or git_url


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _run_git(args: list[str]) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required for remote Harbor registry checks")
    try:
        subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(detail or f"git {' '.join(args)} failed") from exc


def records_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return basic pass/fail/block counts for compatibility records."""
    summary = {"total": len(records), "pass": 0, "fail": 0, "blocked": 0}
    for record in records:
        status = record.get("status")
        if status in {"pass", "fail", "blocked"}:
            summary[status] += 1
    return summary


def records_to_markdown(records: list[dict[str, Any]]) -> str:
    """Render records as a compact Markdown table."""
    rows = [
        "| Dataset | Task | Status | Badges | Notes |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        notes = record.get("blocked_reason") or "; ".join(record.get("notes", []))
        rows.append(
            "| {dataset} | {task} | {status} | {badges} | {notes} |".format(
                dataset=record.get("dataset", ""),
                task=record.get("task", ""),
                status=record.get("status", ""),
                badges=", ".join(record.get("badges", [])),
                notes=notes or "",
            )
        )
    return "\n".join(rows) + "\n"


def dataclass_record(ref: HarborTaskRef) -> dict[str, Any]:
    """Expose a serializable task-ref representation for tests and callers."""
    return asdict(ref)
