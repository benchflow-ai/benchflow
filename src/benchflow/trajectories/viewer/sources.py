"""Trajectory sources: local paths and hf:// dataset slices, parsed and typed.

The CLI hands ``bench eval view``'s argument here as a **string** — never
through :class:`pathlib.Path`, whose normalization collapses ``hf://`` into
``hf:/``. :func:`parse_source` turns it into an explicit
``LocalPathSource | HfDatasetSource`` and validates dataset subpaths.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .payload import VERIFIER_SIDECARS

# Exact files the viewer reads from a rollout dir — the hf:// download
# allowlist. Precise paths only, no wildcards: a live resolution of the
# example dataset with a ``verifier/*`` pattern pulled 418 files (~31 MB of
# OBJ/MP4/PDF/NPZ artifacts) the viewer never consumes. The verifier portion
# derives from payload.VERIFIER_SIDECARS so it cannot drift wider than what
# payload._load_verifier actually reads; a regression test rejects widening.
_HF_VIEWER_FILES = (
    "result.json",
    "timing.json",
    "prompts.json",
    "trajectory/acp_trajectory.jsonl",
    *(f"verifier/{name}" for name in VERIFIER_SIDECARS),
)

# bench review reports live beside the runs (``jobs/review-<stamp>/``), so a
# scoped source also fetches its parent's review directories. Small JSON only.
_HF_REVIEW_FILES = ("review_report.json",)

_REPO_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GLOB_METACHARACTERS = frozenset("*?[]")


class ViewerSourceError(RuntimeError):
    """A syntactically valid viewer source could not be materialized."""


@dataclass(frozen=True)
class LocalPathSource:
    """A plain filesystem source: rollout dir, job tree, or session file."""

    path: Path


@dataclass(frozen=True)
class HfDatasetSource:
    """An ``hf://<org>/<name>[@revision][/subpath]`` dataset slice."""

    repo_id: str
    revision: str | None
    subpath: str


def _invalid_subpath_segment(segment: str) -> bool:
    """Whether a dataset path segment can escape or widen an exact scope."""
    return (
        not segment
        or segment in (".", "..")
        or "\\" in segment
        or ":" in segment
        or "\x00" in segment
        or any(character in segment for character in _GLOB_METACHARACTERS)
    )


def parse_source(raw: str) -> LocalPathSource | HfDatasetSource:
    """Parse a viewer source argument into its typed form.

    Raises :class:`ValueError` on malformed HF specs — including the
    ``hf:/`` spelling produced by a ``Path`` round trip, which gets a
    pointed message instead of silent acceptance.
    """
    if raw.startswith("hf://"):
        return _parse_hf(raw)
    if raw.startswith("hf:"):
        raise ValueError(
            f"malformed HF spec {raw!r} — write hf://<org>/<name>[/subpath]. "
            "(A pathlib.Path round trip collapses the double slash; keep the "
            "spec a string.)"
        )
    return LocalPathSource(path=Path(raw))


def _parse_hf(raw: str) -> HfDatasetSource:
    spec = raw[len("hf://") :]
    # A single trailing slash is a harmless spelling of the dataset root.
    # Empty segments anywhere else are ambiguous and must not be normalized
    # away (``org/name//run`` is not the same input as ``org/name/run``).
    if spec.endswith("/"):
        spec = spec[:-1]
    parts = spec.split("/")
    if any(not part for part in parts):
        raise ValueError(f"empty path segment in HF dataset spec {raw!r}")
    if len(parts) < 2:
        raise ValueError(
            f"invalid HF dataset spec {raw!r} — expected hf://<org>/<name>[/subpath]"
        )
    org, name = parts[0], parts[1]
    revision: str | None = None
    if "@" in name:
        name, _, revision = name.partition("@")
        if not revision:
            raise ValueError(f"empty revision in HF dataset spec {raw!r}")
    for part in (org, name):
        if not _REPO_PART_RE.match(part):
            raise ValueError(f"invalid HF repo component {part!r} in {raw!r}")
    subpath_parts = parts[2:]
    for segment in subpath_parts:
        if _invalid_subpath_segment(segment):
            raise ValueError(f"invalid dataset subpath segment {segment!r} in {raw!r}")
    return HfDatasetSource(
        repo_id=f"{org}/{name}",
        revision=revision,
        subpath="/".join(subpath_parts),
    )


def resolve_hf_dataset(source: HfDatasetSource) -> Path:
    """Materialize the viewer-relevant slice of an HF dataset locally.

    Downloads (into the shared huggingface_hub cache, so repeat views are
    incremental) only the files the viewer renders — see _HF_VIEWER_FILES —
    and returns the local directory to serve.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ViewerSourceError(
            "huggingface_hub is required for hf:// sources"
        ) from exc

    # Revalidate constructed sources before their subpath enters glob-style
    # ``allow_patterns``.  Containment after download protects filesystem
    # access; this check also prevents ``*``/``[]`` from silently widening the
    # amount downloaded in the first place.
    subpath_parts = source.subpath.split("/") if source.subpath else []
    invalid_segment = next(
        (segment for segment in subpath_parts if _invalid_subpath_segment(segment)),
        None,
    )
    if invalid_segment is not None:
        raise ViewerSourceError(
            f"Invalid dataset subpath segment {invalid_segment!r}: {source.subpath}"
        )

    scope = f"{source.subpath}/" if source.subpath else ""
    patterns = []
    for name in _HF_VIEWER_FILES:
        patterns.append(f"{scope}{name}")
        patterns.append(f"{scope}**/{name}")
    parent = f"{'/'.join(subpath_parts[:-1])}/" if len(subpath_parts) > 1 else ""
    for name in _HF_REVIEW_FILES:
        patterns.append(f"{scope}**/{name}")
        patterns.append(f"{parent}review*/{name}")
        patterns.append(f"{parent}review*/**/{name}")
    print(
        f"Fetching {source.repo_id}"
        + (f" @ {source.revision}" if source.revision else "")
        + " …"
    )
    try:
        local = snapshot_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=patterns,
        )
    except Exception as exc:
        revision = f" at revision {source.revision}" if source.revision else ""
        raise ViewerSourceError(
            f"Could not fetch dataset {source.repo_id}{revision}: {exc}"
        ) from exc
    snapshot_root = Path(local).resolve()
    root = (snapshot_root / source.subpath).resolve()
    if not root.is_relative_to(snapshot_root):
        raise ViewerSourceError(
            f"Dataset subpath resolves outside downloaded snapshot: {source.subpath}"
        )
    if not root.is_dir():
        raise ViewerSourceError(
            f"No such path in dataset {source.repo_id}: {source.subpath}"
        )
    return root
