"""Path rules for Harbor-style ``artifacts = [...]`` declarations.

A declaration is a sandbox ``source`` (absolute, or relative to the agent
workspace) and an optional ``destination`` inside the trial's ``artifacts/``
folder. Neither may climb with ``..``; a destination may not be absolute.
The collector (``benchflow.rollout._artifacts``) re-checks these rules, so a
config that bypassed the pre-launch gate still cannot write outside the trial.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from benchflow.task.config import ArtifactConfig

#: Backends that collect root artifacts (exec + tar download + manifest).
ARTIFACT_SANDBOXES: frozenset[str] = frozenset({"docker", "remote-docker", "daytona"})


def as_artifact_config(item: str | ArtifactConfig) -> ArtifactConfig:
    return ArtifactConfig(source=item) if isinstance(item, str) else item


def artifact_destination(config: ArtifactConfig) -> PurePosixPath:
    """Where a declared artifact lands, relative to the trial ``artifacts/``."""
    if config.destination:
        return PurePosixPath(config.destination)
    return PurePosixPath(PurePosixPath(config.source).name or "artifact")


def artifact_spec_issue(item: str | ArtifactConfig) -> str | None:
    """Why a declaration is unsafe, or ``None`` when it may be collected."""
    config = as_artifact_config(item)
    source = PurePosixPath(config.source) if config.source.strip() else None
    if source is None or ".." in source.parts:
        return f"unsafe source {config.source!r}: must be a path without '..'"
    if config.destination is not None:
        destination = PurePosixPath(config.destination)
        if (
            destination.is_absolute()
            or ".." in destination.parts
            or not destination.parts
            or str(destination) == "."
        ):
            return (
                f"unsafe destination {config.destination!r}: must be a relative "
                "path inside the trial artifacts folder, without '..'"
            )
    return None
