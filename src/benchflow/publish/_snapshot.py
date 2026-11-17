"""Publication uses the shared no-follow snapshot boundary."""

# Keep the publication API; traversal and copying live with all reachable I/O.
import os  # noqa: F401  (compatibility for callers probing platform support)
from pathlib import Path

from benchflow._utils.raw_evidence import is_raw_evidence, refuse_raw_evidence
from benchflow._utils.reachable_write import (
    _open_snapshot_directory as _open_directory,
)
from benchflow._utils.reachable_write import snapshot_folder
from benchflow._utils.reachable_write import (
    snapshot_regular_files as regular_files,
)

__all__ = ["_open_directory", "regular_files", "staged_folder"]


def staged_folder(folder: Path):
    """A private snapshot of *folder* to publish, without a hosted run's raw
    evidence (``is_raw_evidence``). A *folder* that is raw evidence or sits
    inside it is refused (``RawEvidenceError``)."""
    refuse_raw_evidence(folder)

    def raw(path: Path, directory: bool) -> bool:
        return directory and is_raw_evidence(folder / path.parent, path.name)

    return snapshot_folder(folder, exclude=raw)
