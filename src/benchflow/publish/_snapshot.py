"""Publication uses the shared no-follow snapshot boundary."""

# Keep the publication API; traversal and copying live with all reachable I/O.
import os  # noqa: F401  (compatibility for callers probing platform support)

from benchflow._utils.reachable_write import (
    _open_snapshot_directory as _open_directory,
)
from benchflow._utils.reachable_write import (
    snapshot_folder as staged_folder,
)
from benchflow._utils.reachable_write import (
    snapshot_regular_files as regular_files,
)

__all__ = ["_open_directory", "regular_files", "staged_folder"]
