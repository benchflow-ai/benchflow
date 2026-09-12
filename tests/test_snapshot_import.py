"""The import half of ``--keep-snapshots``, and the eval-run retention flag.

Guards "feat(rollout): stage snapshots record their lifetime;
--keep-snapshots on bench eval run with a tested import path" (PR #1046
second review, P1-B): a completed run's exported stage snapshot must be
loadable later — sha256-verified tar, ``docker load``, and the loaded image
id checked against the recorded one — and the ``bench eval run`` flag must
reach ``RolloutConfig.keep_snapshots`` through the whole planning stack.

Unit tests against a fake docker runner — no Docker daemon, no API keys. The
live end-to-end proof (real ``docker save`` → ``rmi`` → import) lives in
``tests/test_branch_composed_docker.py``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.snapshot_import import (
    ImportedSnapshot,
    SnapshotImportError,
    import_stage_snapshots,
)

runner = CliRunner()

_IMAGE_ID = "sha256:" + "ab" * 32
_REF = "bf-snap-demo-1234"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _run_dir(
    tmp_path: Path,
    *,
    tar_bytes: bytes = b"exported-image-bytes",
    image_id: str | None = _IMAGE_ID,
    recorded_sha: str | None = None,
    exported: bool = True,
    stage: str = "pre-verify",
) -> Path:
    """A completed run directory: stage_snapshots.json (+ exported tar)."""
    run_dir = tmp_path / "run"
    entry: dict[str, Any] = {
        "environment_ref": None,
        "sandbox_ref": _REF,
        "layers": ["sandbox"],
        "exchanges_completed": None,
        "ephemeral": not exported,
        "exported": None,
    }
    if exported:
        tar_path = run_dir / "snapshots" / f"{_REF}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        tar_path.write_bytes(tar_bytes)
        entry["exported"] = {
            "path": str(tar_path),
            "sha256": recorded_sha or _sha256(tar_bytes),
            "image_id": image_id,
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_snapshots.json").write_text(
        json.dumps({"schema_version": 1, "stages": {stage: entry}}, indent=2) + "\n"
    )
    return run_dir


class _FakeDocker:
    """Records docker CLI calls; serves configurable load/inspect results."""

    def __init__(
        self, *, loaded_id: str = _IMAGE_ID, load_rc: int = 0, inspect_rc: int = 0
    ) -> None:
        self.calls: list[list[str]] = []
        self.loaded_id = loaded_id
        self.load_rc = load_rc
        self.inspect_rc = inspect_rc

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args[0] == "load":
            return subprocess.CompletedProcess(
                args, self.load_rc, stdout="", stderr="boom" if self.load_rc else ""
            )
        assert args[:3] == ["image", "inspect", "--format"]
        return subprocess.CompletedProcess(
            args,
            self.inspect_rc,
            stdout=f"{self.loaded_id}\n" if self.inspect_rc == 0 else "",
            stderr="No such image" if self.inspect_rc else "",
        )


# 1. The load path: verify → load → identity-check


def test_import_loads_verifies_and_returns_the_recorded_image(tmp_path: Path):
    run_dir = _run_dir(tmp_path)
    docker = _FakeDocker()

    imported = import_stage_snapshots(run_dir, run_docker=docker)

    tar_path = run_dir / "snapshots" / f"{_REF}.tar"
    assert imported == [
        ImportedSnapshot(
            stage="pre-verify",
            sandbox_ref=_REF,
            image_id=_IMAGE_ID,
            tar_path=tar_path,
        )
    ]
    assert docker.calls == [
        ["load", "-i", str(tar_path)],
        ["image", "inspect", "--format", "{{.Id}}", _REF],
    ]


def test_a_loaded_id_that_differs_from_the_recorded_one_fails_closed(
    tmp_path: Path,
):
    """The identity check is the point: a ref resolving to a *different*
    image than the run snapshotted must not be reported restored."""
    other_id = "sha256:" + "ff" * 32
    run_dir = _run_dir(tmp_path)

    with pytest.raises(SnapshotImportError, match="different world"):
        import_stage_snapshots(run_dir, run_docker=_FakeDocker(loaded_id=other_id))


def test_a_run_recorded_without_an_image_id_still_verifies_resolution(
    tmp_path: Path,
):
    """image_id is best-effort at export time (an unreadable tar records
    null); the import then verifies the ref resolves and reports the id
    docker actually loaded — never a guess, never a skipped load check."""
    run_dir = _run_dir(tmp_path, image_id=None)

    [imported] = import_stage_snapshots(run_dir, run_docker=_FakeDocker())

    assert imported.image_id == _IMAGE_ID


def test_a_tampered_tar_is_refused_before_docker_is_ever_invoked(tmp_path: Path):
    run_dir = _run_dir(tmp_path, recorded_sha=_sha256(b"what-the-run-exported"))
    docker = _FakeDocker()

    with pytest.raises(SnapshotImportError, match="does not match its record"):
        import_stage_snapshots(run_dir, run_docker=docker)

    assert docker.calls == []


def test_a_failed_docker_load_surfaces_its_stderr(tmp_path: Path):
    run_dir = _run_dir(tmp_path)

    with pytest.raises(SnapshotImportError, match=r"docker load .* failed: boom"):
        import_stage_snapshots(run_dir, run_docker=_FakeDocker(load_rc=1))


def test_a_ref_that_does_not_resolve_after_load_fails_closed(tmp_path: Path):
    run_dir = _run_dir(tmp_path)

    with pytest.raises(SnapshotImportError, match="does not resolve"):
        import_stage_snapshots(run_dir, run_docker=_FakeDocker(inspect_rc=1))


def test_a_relocated_run_dir_finds_the_tar_beside_the_file(tmp_path: Path):
    """The export records an absolute path on the eval machine; a copied run
    folder keeps the tar at <run_dir>/snapshots/<basename>, and the import
    must find it there instead of dying on the stale absolute path."""
    run_dir = _run_dir(tmp_path)
    moved = tmp_path / "copied-elsewhere" / "run"
    moved.parent.mkdir()
    run_dir.rename(moved)

    [imported] = import_stage_snapshots(moved, run_docker=_FakeDocker())

    assert imported.tar_path == moved / "snapshots" / f"{_REF}.tar"


# 2. Honest refusals: ephemeral refs and unknown stages


def test_an_ephemeral_entry_fails_closed_naming_the_flag(tmp_path: Path):
    """A plain run's refs are marked ephemeral at cleanup; asking to import
    one must say why there is nothing to import and how to get it."""
    run_dir = _run_dir(tmp_path, exported=False)
    docker = _FakeDocker()

    with pytest.raises(SnapshotImportError, match="--keep-snapshots"):
        import_stage_snapshots(run_dir, stages=["pre-verify"], run_docker=docker)
    with pytest.raises(SnapshotImportError, match="--keep-snapshots"):
        import_stage_snapshots(run_dir, run_docker=docker)

    assert docker.calls == []


def test_an_unrecorded_stage_lists_what_the_run_captured(tmp_path: Path):
    run_dir = _run_dir(tmp_path)

    with pytest.raises(SnapshotImportError, match=r"\['pre-verify'\]"):
        import_stage_snapshots(run_dir, stages=["env-ready"], run_docker=_FakeDocker())


def test_a_directory_without_the_artifact_is_not_a_snapshot_run(tmp_path: Path):
    with pytest.raises(SnapshotImportError, match=r"no stage_snapshots\.json"):
        import_stage_snapshots(tmp_path, run_docker=_FakeDocker())


# 3. CLI surface: bench eval import-snapshots


def test_cli_import_snapshots_prints_each_restored_ref(tmp_path: Path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr("benchflow.snapshot_import._run_docker", _FakeDocker())

    result = runner.invoke(app, ["eval", "import-snapshots", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert _REF in result.output
    assert _IMAGE_ID in result.output


def test_cli_import_snapshots_fails_cleanly_on_ephemeral_runs(
    tmp_path: Path, monkeypatch
):
    run_dir = _run_dir(tmp_path, exported=False)
    monkeypatch.setattr("benchflow.snapshot_import._run_docker", _FakeDocker())

    result = runner.invoke(app, ["eval", "import-snapshots", str(run_dir)])

    assert result.exit_code == 1
    assert "--keep-snapshots" in result.output


# 4. The eval-run flag reaches RolloutConfig through the planning stack


def _task_dir(tmp_path: Path) -> Path:
    task = tmp_path / "tasks" / "demo-task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
    (task / "instruction.md").write_text("solve it\n", encoding="utf-8")
    return task


def test_keep_snapshots_threads_from_eval_request_to_rollout_config(
    tmp_path: Path,
):
    """--keep-snapshots → EvalCreateRequest → EvaluationConfig →
    task_rollout_config → RolloutConfig.keep_snapshots, and its absence means
    False — the same semantics as bench eval ablate's flag."""
    from benchflow.eval_plan import EvalCreateRequest, build_eval_plan
    from benchflow.evaluation import task_rollout_config

    task = _task_dir(tmp_path)
    for flag in (True, False):
        plan = build_eval_plan(
            EvalCreateRequest(
                tasks_dir=task, jobs_dir=str(tmp_path / "jobs"), keep_snapshots=flag
            )
        )
        eval_config = plan.make_eval_config()
        assert eval_config.keep_snapshots is flag
        rollout_config = task_rollout_config(
            eval_config, task, job_name="j", jobs_dir=tmp_path / "jobs"
        )
        assert rollout_config.keep_snapshots is flag


def test_cli_keep_snapshots_reaches_the_evaluation_config(tmp_path: Path, monkeypatch):
    """The flag half at the CLI surface: `bench eval run --keep-snapshots`
    lands on the EvaluationConfig the batch runner receives."""
    seen: list[Any] = []
    monkeypatch.setattr(
        "benchflow.cli.main.run_batch_eval",
        lambda plan, tasks_dir, config: seen.append(config),
    )
    task = _task_dir(tmp_path)

    result = runner.invoke(
        app,
        ["eval", "run", "--tasks-dir", str(task), "--keep-snapshots"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["eval", "run", "--tasks-dir", str(task)])
    assert result.exit_code == 0, result.output

    assert [config.keep_snapshots for config in seen] == [True, False]


def test_worker_payload_round_trips_keep_snapshots():
    """Sharded worker runs must not silently drop retention: the payload the
    parent writes and the config the worker rebuilds agree on the flag."""
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig

    config = EvaluationConfig(keep_snapshots=True)
    shard = EvalShard(index=0, task_names=["t"], concurrency=1)
    payload = _config_payload(config, shard=shard)
    assert payload["keep_snapshots"] is True
    assert _evaluation_config(payload).keep_snapshots is True
    payload["keep_snapshots"] = False
    assert _evaluation_config(payload).keep_snapshots is False
