"""Folders a run keeps on this machine only.

A hosted run's raw vf-eval output is evidence for whoever ran it, unmasked:
nothing uploads, publishes, packs or shows it to a reviewer. ``hosted_env/``
holds it (vf-eval's stdout and stderr, and ``vf-results/``: raw completions
and the resolved config with every env arg and the base URL). A run written
before ``vf-results/`` moved there keeps it, and its ``.venv/``, beside
``hosted_env/``. Every path that copies a run or job folder elsewhere asks
:func:`is_raw_evidence` of each entry, and refuses a folder it is given that
is raw evidence or sits inside it (:func:`raw_evidence_refusal`).
"""

from __future__ import annotations

import os
from pathlib import Path

HOSTED_RAW_DIR = "hosted_env"
_BESIDE_HOSTED_RAW = frozenset({"vf-results", ".venv"})


def is_raw_evidence(folder: Path, name: str) -> bool:
    """Whether the entry *name* of *folder* stays on this machine:
    ``hosted_env`` anywhere, and ``vf-results`` or ``.venv`` in a folder that
    holds ``hosted_env`` (a hosted run's folder from before the move)."""
    if name == HOSTED_RAW_DIR:
        return True
    return name in _BESIDE_HOSTED_RAW and (folder / HOSTED_RAW_DIR).is_dir()


def raw_evidence_folder(path: Path) -> Path | None:
    """The raw-evidence folder *path* is or sits inside: *path* or the
    nearest of its parents that :func:`is_raw_evidence` names, read from the
    absolute path as given and from the one its links resolve to. None when
    neither has one."""
    path = Path(path).expanduser()
    for spelling in dict.fromkeys((Path(os.path.abspath(path)), path.resolve())):
        for folder in (spelling, *spelling.parents):
            if folder.name and is_raw_evidence(folder.parent, folder.name):
                return folder
    return None


def raw_evidence_refusal(path: Path) -> str | None:
    """Why *path*, given to upload, publish, pack or show to a reviewer, is
    refused: it is a hosted run's raw evidence or sits inside it. None when it
    is neither."""
    raw = raw_evidence_folder(path)
    if raw is None:
        return None
    return (
        f"{path} holds a hosted run's raw vf-eval output ({raw}), which is "
        f"never uploaded or published; pass the run folder ({raw.parent}) or "
        "a folder above it"
    )


# A local trial or job folder (what an eval writes) holds what the run left, as
# it was: the agent's files under agent/, the trial's config and its result. A
# workspace is zipped as it is, with nothing masked, so a run's folder, or any
# folder inside one, is never attached as a workspace. A trial folder holds agent/ beside result.json or
# config.json; a job folder holds its own record (summary.json, evaluation.json
# or the job lock) beside trial folders.
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
    """Why *path* is refused as a workspace to upload as it is: a hosted
    run's raw evidence (:func:`raw_evidence_refusal`) or a local trial or job
    folder (:func:`run_folder_refusal`). None when it is neither."""
    return raw_evidence_refusal(path) or run_folder_refusal(path)


def refuse_workspace(path: Path) -> None:
    """Raise :class:`RawEvidenceError` with :func:`workspace_refusal`'s
    reason when *path* may not be uploaded as a workspace."""
    reason = workspace_refusal(path)
    if reason is not None:
        raise RawEvidenceError(reason)


class RawEvidenceError(ValueError):
    """A folder given to upload, publish, pack or show to a reviewer is a
    hosted run's raw evidence or sits inside it."""


def refuse_raw_evidence(path: Path) -> None:
    """Raise :class:`RawEvidenceError` with :func:`raw_evidence_refusal`'s
    reason when *path* is raw evidence or sits inside it."""
    reason = raw_evidence_refusal(path)
    if reason is not None:
        raise RawEvidenceError(reason)
