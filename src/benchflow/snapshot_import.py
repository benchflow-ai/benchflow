"""Load a completed run's exported stage snapshots back into Docker.

The import half of ``--keep-snapshots`` (rollout-branching RFC §3.6): a run
that captured stage boundaries leaves ``stage_snapshots.json`` behind, and
with the flag each captured ``bf-snap-*`` image was ``docker save``d to
``<run_dir>/snapshots/<ref>.tar`` before cleanup destroyed it — with the
tar's path, content sha256 and image id recorded per stage. This module makes
"branch later from a completed run" real: it verifies the recorded sha256,
``docker load``s the tar, and confirms the recorded ref resolves to the
recorded image id — a snapshot is only reported restored when
``docker image inspect`` would agree.

Fail-closed by design: an entry recorded ``ephemeral: true`` (the image died
with the run), a tar whose digest does not match its record, or a loaded
image whose id differs from the recorded one all raise
:class:`SnapshotImportError` naming exactly what disagreed. CLI surface:
``bench eval import-snapshots <run-dir>``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.branch_policy import _file_sha256

logger = logging.getLogger(__name__)

#: Seam for the two docker CLI calls the import makes (``load`` and
#: ``image inspect``); tests substitute a recorder, the default shells out.
DockerRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class SnapshotImportError(RuntimeError):
    """A stage-snapshot import that cannot restore what the run recorded."""


@dataclass(frozen=True)
class ImportedSnapshot:
    """One stage snapshot restored into the local Docker image store."""

    stage: str
    #: The recorded (and now once again resolvable) image ref, ``bf-snap-…``.
    sandbox_ref: str
    #: The id ``docker image inspect`` reports for the loaded ref — equal to
    #: the recorded id whenever the run recorded one.
    image_id: str
    tar_path: Path


def _run_docker(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False, timeout=600
    )


def read_stage_snapshots(run_dir: Path | str) -> dict[str, dict[str, Any]]:
    """The ``stages`` mapping of ``<run_dir>/stage_snapshots.json``.

    Raises :class:`SnapshotImportError` when the file is absent or does not
    carry a ``stages`` mapping — there is nothing recorded to import.
    """
    path = Path(run_dir) / "stage_snapshots.json"
    if not path.exists():
        raise SnapshotImportError(
            f"no stage_snapshots.json under {Path(run_dir)} — this is not a "
            "run directory of a rollout that captured stage snapshots"
        )
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SnapshotImportError(f"unreadable {path}: {exc}") from exc
    stages = payload.get("stages") if isinstance(payload, dict) else None
    if not isinstance(stages, dict):
        raise SnapshotImportError(f"{path} carries no 'stages' mapping")
    return stages


def _resolve_tar_path(run_dir: Path, exported: dict[str, Any]) -> Path:
    """The exported tar's location, tolerating a relocated run directory.

    The export records an absolute path on the machine that ran the eval; a
    run folder copied elsewhere (scp, artifact download) keeps the tar at
    ``<run_dir>/snapshots/<basename>``, so that is the fallback when the
    recorded path does not exist here.
    """
    recorded = Path(str(exported.get("path")))
    if recorded.exists():
        return recorded
    relocated = Path(run_dir) / "snapshots" / recorded.name
    if relocated.exists():
        return relocated
    raise SnapshotImportError(
        f"exported snapshot tar not found: neither the recorded path "
        f"{recorded} nor {relocated} exists"
    )


def _load_one(
    run_dir: Path,
    stage: str,
    entry: dict[str, Any],
    *,
    run_docker: DockerRunner,
) -> ImportedSnapshot:
    """Verify, ``docker load``, and identity-check one exported stage entry."""
    exported = entry.get("exported")
    if not isinstance(exported, dict):
        raise SnapshotImportError(
            f"stage {stage!r} has no exported snapshot — its recorded ref "
            f"({entry.get('sandbox_ref')!r}) is ephemeral: the image died "
            "with the run's cleanup. Re-run the evaluation with "
            "--keep-snapshots to retain an importable tar."
        )
    sandbox_ref = entry.get("sandbox_ref")
    if not sandbox_ref:
        raise SnapshotImportError(
            f"stage {stage!r} records no sandbox-layer image ref to restore"
        )
    tar_path = _resolve_tar_path(run_dir, exported)
    recorded_sha = exported.get("sha256")
    if not recorded_sha:
        raise SnapshotImportError(
            f"stage {stage!r}'s export record carries no sha256 — the tar at "
            f"{tar_path} cannot be verified against what the run exported"
        )
    actual_sha = _file_sha256(tar_path)
    if actual_sha != recorded_sha:
        raise SnapshotImportError(
            f"stage {stage!r}'s tar {tar_path} does not match its record: "
            f"recorded {recorded_sha}, found {actual_sha} — refusing to load "
            "a tar that is not the one the run exported"
        )
    load = run_docker(["load", "-i", str(tar_path)])
    if load.returncode != 0:
        raise SnapshotImportError(
            f"docker load of {tar_path} failed: "
            f"{(load.stderr or load.stdout or '').strip()}"
        )
    inspect = run_docker(["image", "inspect", "--format", "{{.Id}}", str(sandbox_ref)])
    if inspect.returncode != 0:
        raise SnapshotImportError(
            f"docker load of {tar_path} succeeded but the recorded ref "
            f"{sandbox_ref!r} does not resolve: "
            f"{(inspect.stderr or inspect.stdout or '').strip()}"
        )
    loaded_id = (inspect.stdout or "").strip()
    recorded_id = exported.get("image_id")
    if recorded_id and loaded_id != recorded_id:
        raise SnapshotImportError(
            f"stage {stage!r} loaded, but {sandbox_ref!r} now names image "
            f"{loaded_id}, not the recorded {recorded_id} — the ref resolves "
            "to a different world than the run snapshotted"
        )
    logger.info("stage %r snapshot restored: %s (%s)", stage, sandbox_ref, loaded_id)
    return ImportedSnapshot(
        stage=stage,
        sandbox_ref=str(sandbox_ref),
        image_id=loaded_id,
        tar_path=tar_path,
    )


def import_stage_snapshots(
    run_dir: Path | str,
    *,
    stages: Iterable[str] | None = None,
    run_docker: DockerRunner | None = None,
) -> list[ImportedSnapshot]:
    """Restore a completed run's exported stage snapshots into Docker.

    ``stages=None`` (default) imports every stage whose entry carries an
    ``exported`` record; naming stages explicitly fails closed on one that
    was not recorded or not exported. After a successful import each returned
    :class:`ImportedSnapshot`'s ``sandbox_ref`` resolves locally again —
    verified sha256 on the tar, verified image id on the loaded ref — so the
    recorded boundary can be branched (``DockerSandbox.restore`` /
    ``Rollout.branch_at_stage`` machinery, or a plain ``docker run``).

    Raises :class:`SnapshotImportError`; never partially lies — a stage is
    only present in the result when its image verifiably resolves.
    """
    # Resolved at call time (not bound as a parameter default) so tests can
    # substitute the module-level runner.
    docker = run_docker if run_docker is not None else _run_docker
    run_path = Path(run_dir)
    recorded = read_stage_snapshots(run_path)
    if stages is None:
        selected = sorted(
            stage
            for stage, entry in recorded.items()
            if isinstance(entry, dict) and isinstance(entry.get("exported"), dict)
        )
        if not selected:
            raise SnapshotImportError(
                f"no exported stage snapshots under {run_path} — every "
                f"recorded stage ({sorted(recorded) or 'none'}) is ephemeral: "
                "the images died with the run's cleanup. Re-run the "
                "evaluation with --keep-snapshots to retain importable tars."
            )
    else:
        selected = list(stages)
        missing = [stage for stage in selected if stage not in recorded]
        if missing:
            raise SnapshotImportError(
                f"stage(s) {missing!r} were not captured by this run — it "
                f"recorded {sorted(recorded)!r}"
            )
    return [
        _load_one(run_path, stage, recorded[stage], run_docker=docker)
        for stage in selected
    ]
