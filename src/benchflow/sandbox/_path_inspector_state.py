"""Controller-owned provenance for the inspector copied into a checkpoint."""

import dataclasses
import re

from benchflow.sandbox.protocol import SandboxImage

_KEY = "benchflow.path-inspector.main"
_FOLDER = re.compile(r"/etc/\.benchflow-bash-env-[0-9a-f]{32}\Z")


def checkpoint(sandbox, image: SandboxImage) -> SandboxImage:
    saved = getattr(sandbox, "_benchflow_path_inspectors", None)
    folder = saved.get("main") if isinstance(saved, dict) else None
    if folder is None:
        return image
    if not isinstance(folder, str) or not _FOLDER.fullmatch(folder):
        raise ValueError("invalid pre-agent PATH inspector record")
    return dataclasses.replace(image, meta={**image.meta, _KEY: folder})


def read(image: SandboxImage) -> str | None:
    folder = image.meta.get(_KEY)
    if folder is not None and (
        not isinstance(folder, str) or not _FOLDER.fullmatch(folder)
    ):
        raise ValueError("invalid checkpoint PATH inspector record")
    return folder


def restored(sandbox, folder: str | None) -> None:
    # Only main is replaced by a container snapshot. Other live services keep
    # their own copies, and must never inherit main's directory as evidence.
    saved = getattr(sandbox, "_benchflow_path_inspectors", None)
    saved = dict(saved) if isinstance(saved, dict) else {}
    saved.pop("main", None)
    if folder is not None:
        saved["main"] = folder
    sandbox._benchflow_path_inspectors = saved
