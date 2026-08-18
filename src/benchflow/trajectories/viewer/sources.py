"""Trajectory sources beyond the local filesystem (hf:// datasets)."""

import sys
from pathlib import Path

_HF_VIEWER_FILES = (
    "result.json",
    "timing.json",
    "prompts.json",
    "trajectory/acp_trajectory.jsonl",
    "verifier/*",
)


def _parse_hf_spec(spec: str) -> tuple[str, str | None, str]:
    """Split ``hf://org/name[@revision][/subpath]`` into its parts.

    Tolerates ``hf:/`` (a Path round-trip collapses the double slash) and
    returns ``(repo_id, revision, subpath)``.
    """
    raw = spec.split(":", 1)[1].lstrip("/")
    parts = [p for p in raw.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"invalid HF dataset spec {spec!r} — expected hf://<org>/<name>[/subpath]"
        )
    repo_id = "/".join(parts[:2])
    revision = None
    if "@" in parts[1]:
        name, _, revision = parts[1].partition("@")
        repo_id = f"{parts[0]}/{name}"
    return repo_id, revision, "/".join(parts[2:])


def _resolve_hf_source(spec: str) -> Path:
    """Materialize the viewer-relevant slice of an HF dataset locally.

    Downloads (into the shared huggingface_hub cache, so repeat views are
    incremental) only the files the viewer renders — trajectories plus
    result/timing/prompts/verifier sidecars — and returns the local
    directory to serve.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print("huggingface_hub is required for hf:// sources")
        sys.exit(1)

    repo_id, revision, subpath = _parse_hf_spec(spec)
    scope = f"{subpath.rstrip('/')}/" if subpath else ""
    patterns = []
    for name in _HF_VIEWER_FILES:
        patterns.append(f"{scope}{name}")
        patterns.append(f"{scope}**/{name}")
    print(f"Fetching {repo_id}" + (f" @ {revision}" if revision else "") + " …")
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
    )
    root = Path(local) / subpath if subpath else Path(local)
    if not root.is_dir():
        print(f"No such path in dataset {repo_id}: {subpath}")
        sys.exit(1)
    return root
