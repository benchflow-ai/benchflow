#!/usr/bin/env python3
"""Upload BenchFlow e2e job trees to HuggingFace (rollout artifacts only)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ID = "benchflow/v05-integration-e2e-artifacts"
ROLLout_MARKERS = ("result.json", "config.json")
SKIP_DIR_NAMES = {".venv", "__pycache__", "node_modules", ".git"}
MAX_FILE_MB = 50


def _is_rollout_dir(path: Path) -> bool:
    return path.is_dir() and (path / "result.json").is_file()


def _find_rollout_dirs(jobs_root: Path) -> list[Path]:
    found: list[Path] = []
    if not jobs_root.is_dir():
        return found
    for result in jobs_root.rglob("result.json"):
        found.append(result.parent)
    return sorted(set(found))


def _copy_rollout(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in item.parts):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def main() -> int:
    jobs_roots = [Path(p) for p in sys.argv[1:]] or [Path("/tmp/bf-audit-jobs")]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Set HF_TOKEN or HUGGINGFACE_TOKEN", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    staging = Path(f"/tmp/hf-e2e-staging/{stamp}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    manifest_rollouts: list[dict] = []
    for jobs_root in jobs_roots:
        if not jobs_root.exists():
            continue
        for rollout in _find_rollout_dirs(jobs_root):
            rel = rollout.relative_to(jobs_root)
            dest = staging / "jobs" / jobs_root.name / rel
            _copy_rollout(rollout, dest)
            result_path = rollout / "result.json"
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                result = {}
            manifest_rollouts.append(
                {
                    "jobs_root": str(jobs_root),
                    "rollout_path": str(rollout),
                    "upload_path": str(dest.relative_to(staging)),
                    "task_name": result.get("task_name"),
                    "agent": result.get("agent"),
                    "model": result.get("model"),
                    "error": result.get("error"),
                    "rewards": result.get("rewards"),
                    "trajectory_source": result.get("trajectory_source"),
                }
            )

    manifest = {
        "uploaded_at": stamp,
        "repo_id": REPO_ID,
        "benchflow_branch": os.environ.get("BENCHFLOW_GIT_BRANCH", ""),
        "rollout_count": len(manifest_rollouts),
        "rollouts": manifest_rollouts,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = staging / "README.md"
    readme.write_text(
        f"# v0.5 integration e2e artifacts ({stamp})\n\n"
        f"Auto-uploaded rollout directories from cloud user-audit.\n\n"
        f"- Rollouts: {len(manifest_rollouts)}\n"
        f"- See `manifest.json` for per-rollout metadata.\n"
    )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub required: uv pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    path_in_repo = f"runs/{stamp}"
    print(f"Uploading {staging} -> {REPO_ID}/{path_in_repo} ({len(manifest_rollouts)} rollouts)")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=f"e2e artifacts {stamp} ({len(manifest_rollouts)} rollouts)",
    )
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}/tree/main/{path_in_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
