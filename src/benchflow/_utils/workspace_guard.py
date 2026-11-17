"""Folders never attached as a workspace to upload.

A workspace is zipped as it is, with nothing masked. A local trial or job
folder (what an eval writes) holds what the run left, as it was: the agent's
files under agent/, the trial's config and its result. So a run's folder, or
any folder inside one, is never attached as a workspace. A trial folder holds agent/ beside result.json or
config.json; a job folder holds its own record (summary.json, evaluation.json
or the job lock) beside trial folders.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRIAL_FILES = ("result.json", "config.json")
_JOB_FILES = ("summary.json", "evaluation.json", ".evaluation.lock")


def _is_trial_folder(folder: Path) -> bool:
    return (folder / "agent").is_dir() and any(
        (folder / name).is_file() for name in _TRIAL_FILES
    )


def _holds_trials(folder: Path) -> bool:
    try:
        return any(
            child.is_dir() and _is_trial_folder(child) for child in folder.iterdir()
        )
    except OSError:
        return False


def _is_job_folder(folder: Path) -> bool:
    """A folder holding trial folders, with or without its own record (a job
    a Rollout writes directly has none)."""
    return _holds_trials(folder)


def is_run_folder(folder: Path) -> bool:
    """Whether *folder* is a local trial or job folder: one that a workspace
    archive leaves out wherever it sits in the workspace."""
    return _is_trial_folder(folder) or _is_job_folder(folder)


def run_folder_refusal(path: Path) -> str | None:
    """Why *path*, given as a workspace to upload as it is, is refused: it is
    a local trial or job folder or sits inside one, read from the absolute
    path as given and from the one its links resolve to. None otherwise."""
    path = Path(path).expanduser()
    for spelling in dict.fromkeys((Path(os.path.abspath(path)), path.resolve())):
        for folder in (spelling, *spelling.parents):
            if _is_trial_folder(folder):
                kind = "trial"
            elif _is_job_folder(folder) and (
                # A parent counts as a job by its own record too, so a home
                # or project folder that merely holds a trial elsewhere does
                # not refuse every folder beside it.
                folder == spelling
                or any((folder / name).exists() for name in _JOB_FILES)
            ):
                kind = "job"
            else:
                continue
            return (
                f"{path} is a local {kind} folder or sits inside one ({folder}), "
                "and a workspace is uploaded as it is, with nothing masked; pass "
                "the project folder the agent worked in as the workspace, and the "
                "trial itself as what to upload (its trajectory is masked)"
            )
    return None


def workspace_refusal(path: Path) -> str | None:
    """Why *path* is refused as a workspace to upload as it is (see
    :func:`run_folder_refusal`). None when it is not."""
    return run_folder_refusal(path)


class WorkspaceRefusedError(ValueError):
    """A folder given as a workspace to upload is a local trial or job folder
    or sits inside one."""


def refuse_workspace(path: Path) -> None:
    """Raise :class:`WorkspaceRefusedError` with :func:`workspace_refusal`'s
    reason when *path* may not be uploaded as a workspace."""
    reason = workspace_refusal(path)
    if reason is not None:
        raise WorkspaceRefusedError(reason)
